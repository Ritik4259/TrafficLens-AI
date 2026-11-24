"""
Video asset ingestion and job dispatch endpoints for TrafficLens-AI.
"""

from __future__ import annotations
import threading
import uuid
from pathlib import Path
from typing import Optional, List
import cv2
from fastapi import APIRouter, Depends, File, Header, HTTPException, Response, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.core.config import Settings
from backend.app.core.logging import get_logger
from backend.app.dependencies import get_app_settings, get_db_session
from backend.app.repositories.job_repository import SQLAlchemyJobRepository
from backend.app.schemas import (
    JobStatus,
    VideoMetadataSchema,
    VideoUploadResponseSchema,
    ProcessingJobCreateSchema,
    ProcessingJobResponseSchema,
)
from backend.app.services.job_service import JobService
from backend.app.services.media_delivery import (
    create_media_stream_response,
    validate_path_containment,
)
from backend.app.services.worker import get_job_worker, QueueFullError

logger = get_logger("trafficlens.api.videos")
router = APIRouter(prefix="/videos", tags=["Videos"])
_video_job_lock = threading.Lock()


@router.get(
    "",
    response_model=List[VideoMetadataSchema],
    summary="List uploaded video assets",
)
def list_videos(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db_session),
) -> List[VideoMetadataSchema]:
    """Retrieves list of uploaded video assets with metadata."""
    repo = SQLAlchemyJobRepository(db)
    assets = repo.list_video_assets(limit=limit, offset=offset)
    return [
        VideoMetadataSchema(
            video_id=asset.id,
            filename=asset.filename,
            content_type=asset.content_type,
            file_size_bytes=asset.file_size_bytes,
            duration_seconds=asset.duration_seconds,
            fps=asset.fps,
            width=asset.width,
            height=asset.height,
            frame_count=int(asset.duration_seconds * asset.fps) if asset.duration_seconds and asset.fps else None,
            uploaded_at=asset.created_at,
        )
        for asset in assets
    ]


@router.post(
    "",
    response_model=VideoUploadResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and validate a video file",
)
async def upload_video(
    file: UploadFile = File(..., description="Multipart video file stream"),
    db: Session = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> VideoUploadResponseSchema:
    """
    Streams a video file to disk, verifies readability via OpenCV, extracts metadata,
    and registers a VideoAsset record in the database.
    """
    original_filename = file.filename or "video.mp4"
    ext = Path(original_filename).suffix.lower()

    # 1. Validate file extension
    if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported video extension '{ext}'. Allowed extensions: {settings.ALLOWED_VIDEO_EXTENSIONS}"
            ),
        )

    # 2. Setup safe storage path
    video_id = f"vid_{uuid.uuid4().hex[:12]}"
    safe_filename = f"{video_id}{ext}"
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    storage_path = (settings.UPLOAD_DIR / safe_filename).resolve()

    # Verify containment within configured upload directory
    if not storage_path.is_relative_to(settings.UPLOAD_DIR.resolve()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: target path escapes configured upload directory.",
        )

    bytes_written = 0
    chunk_size = 1024 * 1024  # 1 MB streaming chunks

    try:
        with open(storage_path, "wb") as f_out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > settings.MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=(
                            f"File size exceeds maximum upload limit of {settings.MAX_UPLOAD_SIZE_BYTES} bytes "
                            f"({settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB)."
                        ),
                    )
                f_out.write(chunk)

    except HTTPException:
        # Cleanup partially written file on limit error
        if storage_path.exists():
            storage_path.unlink()
        raise
    except Exception as e:
        logger.error(f"Error streaming video file upload: {e}", exc_info=True)
        if storage_path.exists():
            storage_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stream and store video file: {e}",
        )

    # 3. Reject empty files
    if bytes_written == 0:
        if storage_path.exists():
            storage_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    # 4. Validate video stream readability and extract metadata using OpenCV
    cap = cv2.VideoCapture(str(storage_path))
    if not cap.isOpened():
        if storage_path.exists():
            storage_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file cannot be opened as a valid video stream.",
        )

    ret, first_frame = cap.read()
    if not ret or first_frame is None or first_frame.size == 0:
        cap.release()
        if storage_path.exists():
            storage_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file contains no readable video frames or is corrupted.",
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or first_frame.shape[1]
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or first_frame.shape[0]
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0 or fps > 240.0:
        fps = 25.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frame_count = 1
    duration = frame_count / fps if fps > 0 else 0.0

    cap.release()

    # 5. Persist VideoAsset record
    repo = SQLAlchemyJobRepository(db)
    asset = repo.create_video_asset(
        video_id=video_id,
        filename=original_filename,
        storage_path=str(storage_path),
        file_size_bytes=bytes_written,
        content_type=file.content_type or "video/mp4",
        duration_seconds=round(duration, 2),
        fps=round(fps, 2),
        width=width,
        height=height,
    )
    db.commit()

    logger.info(
        f"Registered VideoAsset {video_id} ({original_filename}, {bytes_written} bytes, "
        f"{width}x{height} @ {fps:.1f}fps, {duration:.1f}s)"
    )

    metadata = VideoMetadataSchema(
        video_id=video_id,
        filename=original_filename,
        content_type=asset.content_type,
        file_size_bytes=bytes_written,
        duration_seconds=asset.duration_seconds,
        fps=asset.fps,
        width=width,
        height=height,
        frame_count=frame_count,
        uploaded_at=asset.created_at,
    )

    return VideoUploadResponseSchema(
        video_id=video_id,
        metadata=metadata,
        message="Video uploaded successfully and validated.",
    )


@router.get(
    "/{video_id}",
    response_model=VideoMetadataSchema,
    summary="Get video asset metadata",
)
def get_video(
    video_id: str,
    db: Session = Depends(get_db_session),
) -> VideoMetadataSchema:
    """Retrieves metadata for an existing VideoAsset."""
    repo = SQLAlchemyJobRepository(db)
    asset = repo.get_video_asset(video_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video asset '{video_id}' not found.",
        )

    return VideoMetadataSchema(
        video_id=asset.id,
        filename=asset.filename,
        content_type=asset.content_type,
        file_size_bytes=asset.file_size_bytes,
        duration_seconds=asset.duration_seconds,
        fps=asset.fps,
        width=asset.width,
        height=asset.height,
        uploaded_at=asset.created_at,
    )


@router.get(
    "/{video_id}/content",
    summary="Stream raw source video media with HTTP Range support",
    responses={
        200: {"description": "Full video content"},
        206: {"description": "Partial video content (Range response)"},
        400: {"description": "Invalid storage path or security containment failure"},
        404: {"description": "Video asset or media file not found"},
        416: {"description": "Range Not Satisfiable"},
    },
)
def stream_video_content(
    video_id: str,
    range: Optional[str] = Header(None, alias="Range"),
    db: Session = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> Response:
    """
    Streams the source video media file for an uploaded video asset.
    Validates ownership via the database record, enforces strict server-side directory
    containment to prevent path traversal, and supports RFC 9110 HTTP Range requests
    for browser video playback and seeking.
    """
    repo = SQLAlchemyJobRepository(db)
    asset = repo.get_video_asset(video_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video asset '{video_id}' not found.",
        )

    # Resolve server-side path
    target_path = Path(asset.storage_path).resolve()
    permitted_dir = settings.UPLOAD_DIR.resolve()

    if not validate_path_containment(target_path, permitted_dir):
        logger.warning(
            f"Security alert: video asset {video_id} storage_path '{asset.storage_path}' "
            f"escapes upload directory '{permitted_dir}'."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid video storage path: asset escapes permitted upload directory.",
        )

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source video file for '{video_id}' not found on server storage.",
        )

    return create_media_stream_response(
        file_path=target_path,
        range_header=range,
        content_type=asset.content_type,
        permitted_directory=permitted_dir,
    )


@router.post(
    "/{video_id}/jobs",
    response_model=ProcessingJobResponseSchema,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create and queue a video processing job for an uploaded video",
)
def create_video_job(
    video_id: str,
    payload: Optional[ProcessingJobCreateSchema] = None,
    db: Session = Depends(get_db_session),
) -> ProcessingJobResponseSchema:
    """
    Submits a video asset for asynchronous processing.
    Rejects duplicate active scheduling (409 Conflict) if a job is already QUEUED or PROCESSING.
    """
    repo = SQLAlchemyJobRepository(db)
    asset = repo.get_video_asset(video_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video asset '{video_id}' not found.",
        )

    with _video_job_lock:
        # Check duplicate scheduling policy
        active_jobs = [j for j in asset.jobs if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING)]
        if active_jobs:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Video '{video_id}' already has an active processing job ({active_jobs[0].id}) "
                    f"in state {active_jobs[0].status.value}. Duplicate concurrent processing is prohibited."
                ),
            )

        # Prepare job create schema
        job_payload = payload or ProcessingJobCreateSchema(video_id=video_id)
        job_payload.video_id = video_id
        if not job_payload.sequence_name:
            job_payload.sequence_name = f"seq_{video_id}"

        # Extract metadata dictionary for lines and rois
        config_meta = {}
        if job_payload.lines:
            config_meta["lines"] = [l.model_dump() for l in job_payload.lines]
        if job_payload.rois:
            config_meta["rois"] = [r.model_dump() for r in job_payload.rois]

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        try:
            created_job_model = repo.create_job(
                job_id=job_id,
                video_id=video_id,
                sequence_name=job_payload.sequence_name,
                status=JobStatus.QUEUED,
                config_metadata=config_meta if config_meta else None,
            )
            db.flush()
            # Enqueue to background worker before committing to prevent stranded QUEUED jobs
            get_job_worker().enqueue_job(job_id)
            db.commit()
        except QueueFullError as e:
            db.rollback()
            logger.warning(
                f"Worker queue full while scheduling job {job_id} for video {video_id}. Rolled back transaction."
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            )

        return repo._model_to_schema(created_job_model)
