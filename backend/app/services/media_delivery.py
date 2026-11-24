"""
Media delivery and HTTP Range streaming service for TrafficLens-AI.

Provides RFC 9110 compliant chunked byte-range streaming for video media,
enforces server-side path containment, and prevents filesystem traversal.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union
from typing_extensions import Literal

from fastapi import HTTPException, status
from fastapi.responses import Response, StreamingResponse

from backend.app.core.logging import get_logger

logger = get_logger("trafficlens.media_delivery")

# Supported media extensions mapped to standard MIME types
EXTENSION_MIME_MAP = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}

# Default streaming buffer chunk size (256 KB)
DEFAULT_CHUNK_SIZE = 256 * 1024


def get_media_content_type(file_path: Path, fallback_type: str = "video/mp4") -> str:
    """Resolves standard media Content-Type from file extension."""
    ext = file_path.suffix.lower()
    return EXTENSION_MIME_MAP.get(ext, fallback_type)


def validate_path_containment(target_path: Path, permitted_directory: Path) -> bool:
    """
    Verifies that target_path resolves strictly within permitted_directory.
    Protects against path traversal attacks (e.g. '../').
    """
    try:
        resolved_target = target_path.resolve()
        resolved_permitted = permitted_directory.resolve()
        return resolved_target.is_relative_to(resolved_permitted)
    except Exception as exc:
        logger.warning(f"Error checking path containment for '{target_path}': {exc}")
        return False


def parse_range_header(
    range_header: Optional[str],
    total_size: int,
) -> Union[Tuple[int, int], Literal["UNSATISFIABLE"], None]:
    """
    Parses an RFC 9110 HTTP Range header for a single byte range.

    Returns:
        - (start, end): Satisfiable byte range (both inclusive, 0-indexed).
        - "UNSATISFIABLE": Range cannot be satisfied (HTTP 416).
        - None: Header is absent, syntactically invalid, or requests multiple ranges
          (in which case RFC 9110 mandates falling back to a full 200 response).
    """
    if not range_header or not range_header.startswith("bytes="):
        return None

    range_spec = range_header[6:].strip()
    if not range_spec:
        return None

    # Multiple ranges (e.g. bytes=0-50, 100-200) -> fallback to full content per RFC 9110
    if "," in range_spec:
        return None

    # Case 1: Standard range `bytes={start}-{end}`
    m = re.match(r"^(\d+)-(\d+)$", range_spec)
    if m:
        start = int(m.group(1))
        end = int(m.group(2))
        if start > end:
            return "UNSATISFIABLE"
        if total_size == 0 or start >= total_size:
            return "UNSATISFIABLE"
        end = min(end, total_size - 1)
        return (start, end)

    # Case 2: Open-ended range `bytes={start}-`
    m = re.match(r"^(\d+)-$", range_spec)
    if m:
        start = int(m.group(1))
        if total_size == 0 or start >= total_size:
            return "UNSATISFIABLE"
        end = total_size - 1
        return (start, end)

    # Case 3: Suffix range `bytes=-{suffix_len}`
    m = re.match(r"^-(\d+)$", range_spec)
    if m:
        suffix_len = int(m.group(1))
        if suffix_len <= 0 or total_size == 0:
            return "UNSATISFIABLE"
        if suffix_len >= total_size:
            start = 0
        else:
            start = total_size - suffix_len
        end = total_size - 1
        return (start, end)

    # Syntactically invalid format (e.g. non-numeric, multiple hyphens) -> ignore Range header
    return None


def file_chunk_generator(
    file_path: Path,
    start: int,
    length: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[bytes]:
    """
    Yields chunks of bytes from file_path starting at byte offset start up to length bytes.
    Memory footprint is bounded to chunk_size regardless of file size.
    """
    with open(file_path, "rb") as f:
        f.seek(start)
        remaining = length
        while remaining > 0:
            read_len = min(remaining, chunk_size)
            chunk = f.read(read_len)
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def create_media_stream_response(
    file_path: Path,
    range_header: Optional[str] = None,
    content_type: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    permitted_directory: Optional[Path] = None,
) -> Response:
    """
    Constructs an RFC 9110 compliant streaming response for a media file:
    - 200 OK: Full file content when no Range header is provided or Range is invalid.
    - 206 Partial Content: Sliced chunked stream when a valid Range is requested.
    - 416 Range Not Satisfiable: When the requested byte range cannot be satisfied.

    Enforces:
    - Standardized private no-store Cache-Control header.
    - Optional defense-in-depth path containment verification against permitted_directory.
    - Canonical path resolution so the validated path matches the opened file.
    - Guaranteed file handle release via context-managed generator.
    """
    resolved_path = file_path.resolve()

    # Defense-in-depth directory containment check
    if permitted_directory is not None:
        if not validate_path_containment(resolved_path, permitted_directory):
            logger.warning(
                f"Security containment failure: '{resolved_path}' escapes permitted '{permitted_directory}'."
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: target media path escapes permitted storage directory.",
            )

    if not resolved_path.exists() or not resolved_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Media file '{resolved_path.name}' not found on storage.",
        )

    total_size = resolved_path.stat().st_size
    media_type = content_type or get_media_content_type(resolved_path)

    parsed = parse_range_header(range_header, total_size)

    # Standard security caching header: strictly private, no intermediary or browser disk caching
    cache_header = "private, no-store"

    # 1. Unsatisfiable range -> HTTP 416
    if parsed == "UNSATISFIABLE":
        return Response(
            status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
            headers={
                "Content-Range": f"bytes */{total_size}",
                "Accept-Ranges": "bytes",
                "Cache-Control": cache_header,
            },
            media_type="text/plain",
            content="Requested range not satisfiable.",
        )

    # 2. No Range or Ignored Range -> HTTP 200 Full Content
    if parsed is None:
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(total_size),
            "Content-Type": media_type,
            "Cache-Control": cache_header,
        }
        if total_size == 0:
            return Response(
                content=b"",
                status_code=status.HTTP_200_OK,
                headers=headers,
                media_type=media_type,
            )
        return StreamingResponse(
            file_chunk_generator(resolved_path, start=0, length=total_size, chunk_size=chunk_size),
            status_code=status.HTTP_200_OK,
            headers=headers,
            media_type=media_type,
        )

    # 3. Satisfiable Range -> HTTP 206 Partial Content
    start, end = parsed
    content_length = end - start + 1
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Range": f"bytes {start}-{end}/{total_size}",
        "Content-Length": str(content_length),
        "Content-Type": media_type,
        "Cache-Control": cache_header,
    }
    return StreamingResponse(
        file_chunk_generator(resolved_path, start=start, length=content_length, chunk_size=chunk_size),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        headers=headers,
        media_type=media_type,
    )
