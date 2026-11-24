"""
Comprehensive test suite for backend media delivery and HTTP Range streaming.

Validates RFC 9110 compliance:
- Full video streaming (200 OK)
- Byte range requests: single range, open-ended, suffix, edge cases (206 Partial Content)
- Invalid and unsatisfiable ranges (416 Range Not Satisfiable)
- Malformed range headers (RFC 9110 fallback to 200 OK)
- Security boundaries: path traversal rejections and directory containment
- Error handling: missing assets, missing files on disk, incomplete jobs
- Zero-byte files
- Artifact metadata listing endpoint (clean filenames, no filesystem paths, serving endpoints)
"""

from __future__ import annotations
import uuid
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.db.session import SessionLocal
from backend.app.main import app
from backend.app.models.job import ProcessingJob
from backend.app.models.video import VideoAsset
from backend.app.repositories.job_repository import SQLAlchemyJobRepository
from backend.app.schemas.job import JobStatus
from backend.app.services.media_delivery import (
    parse_range_header,
    validate_path_containment,
)

client = TestClient(app)
settings = get_settings()


# ==============================================================================
# Unit Tests for Range Header Parser & Path Containment
# ==============================================================================

def test_parse_range_header_unit():
    """Unit test for RFC 9110 Range header parser logic."""
    total = 1000

    # 1. Valid ranges
    assert parse_range_header("bytes=0-499", total) == (0, 499)
    assert parse_range_header("bytes=500-999", total) == (500, 999)
    assert parse_range_header("bytes=0-0", total) == (0, 0)
    assert parse_range_header("bytes=999-999", total) == (999, 999)

    # 2. Open-ended ranges
    assert parse_range_header("bytes=100-", total) == (100, 999)
    assert parse_range_header("bytes=0-", total) == (0, 999)

    # 3. Suffix ranges
    assert parse_range_header("bytes=-500", total) == (500, 999)
    assert parse_range_header("bytes=-1", total) == (999, 999)
    assert parse_range_header("bytes=-1500", total) == (0, 999)  # Suffix larger than file

    # 4. End pos exceeding total size (clamped per RFC 9110)
    assert parse_range_header("bytes=100-5000", total) == (100, 999)

    # 5. Unsatisfiable ranges
    assert parse_range_header("bytes=1000-", total) == "UNSATISFIABLE"
    assert parse_range_header("bytes=1500-2000", total) == "UNSATISFIABLE"
    assert parse_range_header("bytes=500-400", total) == "UNSATISFIABLE"
    assert parse_range_header("bytes=-0", total) == "UNSATISFIABLE"
    assert parse_range_header("bytes=0-", 0) == "UNSATISFIABLE"

    # 6. Syntactically invalid headers (RFC 9110 mandates ignoring and serving 200)
    assert parse_range_header("bytes=invalid", total) is None
    assert parse_range_header("bytes=--10", total) is None
    assert parse_range_header("items=0-10", total) is None
    assert parse_range_header(None, total) is None
    assert parse_range_header("", total) is None
    assert parse_range_header("bytes=0-10, 20-30", total) is None  # Multiple ranges


def test_path_containment_unit(tmp_path: Path):
    """Unit test for server-side path traversal prevention."""
    base_dir = tmp_path / "storage"
    base_dir.mkdir(parents=True, exist_ok=True)

    safe_file = base_dir / "safe.mp4"
    safe_file.write_bytes(b"data")
    assert validate_path_containment(safe_file, base_dir) is True

    # Traversal escaping base_dir
    traversal_file = base_dir / ".." / "secret.txt"
    assert validate_path_containment(traversal_file, base_dir) is False

    deep_safe = base_dir / "subdir" / "video.mp4"
    assert validate_path_containment(deep_safe, base_dir) is True


# ==============================================================================
# Integration Tests: Source Video Streaming (GET /videos/{video_id}/content)
# ==============================================================================

@pytest.fixture
def sample_source_video():
    """Creates a registered VideoAsset with a deterministic synthetic media file."""
    video_id = f"vid_test_{uuid.uuid4().hex[:10]}"
    filename = f"{video_id}.mp4"
    upload_dir = settings.UPLOAD_DIR.resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename

    # 1,200 bytes of distinct test pattern
    content = bytes((i % 256 for i in range(1200)))
    file_path.write_bytes(content)

    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        asset = repo.create_video_asset(
            video_id=video_id,
            filename=f"user_original_{filename}",
            storage_path=str(file_path),
            file_size_bytes=len(content),
            content_type="video/mp4",
            duration_seconds=10.0,
            fps=25.0,
            width=640,
            height=480,
        )
        db.commit()

    yield {"video_id": video_id, "file_path": file_path, "content": content}

    # Cleanup
    if file_path.exists():
        try:
            file_path.unlink()
        except OSError:
            pass


def test_source_video_full_stream(sample_source_video):
    """Validates full video content delivery without Range header (200 OK)."""
    vid = sample_source_video["video_id"]
    expected = sample_source_video["content"]

    resp = client.get(f"/api/v1/videos/{vid}/content")
    assert resp.status_code == 200
    assert resp.headers.get("Accept-Ranges") == "bytes"
    assert resp.headers.get("Content-Length") == str(len(expected))
    assert resp.headers.get("Content-Type") == "video/mp4"
    assert resp.headers.get("Cache-Control") == "private, no-store"
    assert resp.content == expected


def test_source_video_single_byte_range(sample_source_video):
    """Validates standard single byte range request (206 Partial Content)."""
    vid = sample_source_video["video_id"]
    expected = sample_source_video["content"]

    # Request bytes 100 to 199 (100 bytes total)
    resp = client.get(
        f"/api/v1/videos/{vid}/content",
        headers={"Range": "bytes=100-199"},
    )
    assert resp.status_code == 206
    assert resp.headers.get("Accept-Ranges") == "bytes"
    assert resp.headers.get("Content-Range") == f"bytes 100-199/{len(expected)}"
    assert resp.headers.get("Content-Length") == "100"
    assert resp.headers.get("Content-Type") == "video/mp4"
    assert resp.headers.get("Cache-Control") == "private, no-store"
    assert resp.content == expected[100:200]


def test_source_video_open_ended_range(sample_source_video):
    """Validates open-ended range request (bytes=start-)."""
    vid = sample_source_video["video_id"]
    expected = sample_source_video["content"]
    total = len(expected)

    # Request from byte 1000 to end
    resp = client.get(
        f"/api/v1/videos/{vid}/content",
        headers={"Range": "bytes=1000-"},
    )
    assert resp.status_code == 206
    assert resp.headers.get("Content-Range") == f"bytes 1000-{total - 1}/{total}"
    assert resp.headers.get("Content-Length") == str(total - 1000)
    assert resp.content == expected[1000:]


def test_source_video_suffix_range(sample_source_video):
    """Validates suffix range request (bytes=-suffix)."""
    vid = sample_source_video["video_id"]
    expected = sample_source_video["content"]
    total = len(expected)

    # Request the final 150 bytes
    resp = client.get(
        f"/api/v1/videos/{vid}/content",
        headers={"Range": "bytes=-150"},
    )
    assert resp.status_code == 206
    assert resp.headers.get("Content-Range") == f"bytes {total - 150}-{total - 1}/{total}"
    assert resp.headers.get("Content-Length") == "150"
    assert resp.content == expected[-150:]


def test_source_video_edge_ranges(sample_source_video):
    """Validates boundary conditions: first byte, last byte, clamped upper bound."""
    vid = sample_source_video["video_id"]
    expected = sample_source_video["content"]
    total = len(expected)

    # 1. Exactly first byte (0-0)
    resp_first = client.get(
        f"/api/v1/videos/{vid}/content",
        headers={"Range": "bytes=0-0"},
    )
    assert resp_first.status_code == 206
    assert resp_first.headers.get("Content-Range") == f"bytes 0-0/{total}"
    assert resp_first.headers.get("Content-Length") == "1"
    assert resp_first.content == expected[0:1]

    # 2. Exactly last byte via suffix (-1)
    resp_last = client.get(
        f"/api/v1/videos/{vid}/content",
        headers={"Range": "bytes=-1"},
    )
    assert resp_last.status_code == 206
    assert resp_last.headers.get("Content-Range") == f"bytes {total - 1}-{total - 1}/{total}"
    assert resp_last.headers.get("Content-Length") == "1"
    assert resp_last.content == expected[-1:]

    # 3. End position greater than file size (clamped to total - 1)
    resp_clamp = client.get(
        f"/api/v1/videos/{vid}/content",
        headers={"Range": "bytes=1100-5000"},
    )
    assert resp_clamp.status_code == 206
    assert resp_clamp.headers.get("Content-Range") == f"bytes 1100-{total - 1}/{total}"
    assert resp_clamp.headers.get("Content-Length") == str(total - 1100)
    assert resp_clamp.content == expected[1100:]


def test_source_video_unsatisfiable_ranges(sample_source_video):
    """Validates 416 Range Not Satisfiable when range is outside file bounds."""
    vid = sample_source_video["video_id"]
    total = len(sample_source_video["content"])

    # 1. Start >= total_size
    resp1 = client.get(
        f"/api/v1/videos/{vid}/content",
        headers={"Range": f"bytes={total}-"},
    )
    assert resp1.status_code == 416
    assert resp1.headers.get("Content-Range") == f"bytes */{total}"
    assert resp1.headers.get("Accept-Ranges") == "bytes"
    assert resp1.headers.get("Cache-Control") == "private, no-store"

    # 2. Start > End
    resp2 = client.get(
        f"/api/v1/videos/{vid}/content",
        headers={"Range": "bytes=500-400"},
    )
    assert resp2.status_code == 416
    assert resp2.headers.get("Content-Range") == f"bytes */{total}"

    # 3. Suffix length 0
    resp3 = client.get(
        f"/api/v1/videos/{vid}/content",
        headers={"Range": "bytes=-0"},
    )
    assert resp3.status_code == 416
    assert resp3.headers.get("Content-Range") == f"bytes */{total}"


def test_source_video_malformed_range_header_fallback(sample_source_video):
    """Validates RFC 9110 behavior: syntactically invalid Range headers are ignored (200 OK)."""
    vid = sample_source_video["video_id"]
    expected = sample_source_video["content"]

    resp = client.get(
        f"/api/v1/videos/{vid}/content",
        headers={"Range": "bytes=invalid_syntax"},
    )
    assert resp.status_code == 200
    assert resp.content == expected


def test_source_video_missing_record():
    """Validates 404 when video_id does not exist in database."""
    resp = client.get("/api/v1/videos/vid_non_existent/content")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_source_video_missing_file_on_disk():
    """Validates 404 when database record exists but on-disk file was removed."""
    uid = uuid.uuid4().hex[:8]
    vid = f"vid_ghost_{uid}"
    ghost_path = settings.UPLOAD_DIR.resolve() / f"{vid}.mp4"

    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        asset = repo.create_video_asset(
            video_id=vid,
            filename="ghost.mp4",
            storage_path=str(ghost_path),
            file_size_bytes=500,
        )
        db.commit()

    try:
        resp = client.get(f"/api/v1/videos/{vid}/content")
        assert resp.status_code == 404
        assert "not found on server storage" in resp.json()["detail"].lower()
    finally:
        with SessionLocal() as db:
            a = db.query(VideoAsset).filter(VideoAsset.id == vid).first()
            if a:
                db.delete(a)
                db.commit()


def test_source_video_path_traversal_rejection():
    """Validates 400 Bad Request when database record storage_path escapes upload directory."""
    uid = uuid.uuid4().hex[:8]
    vid = f"vid_hack_{uid}"
    malicious_path = str(settings.UPLOAD_DIR.resolve() / ".." / f"secret_{uid}.mp4")

    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        asset = repo.create_video_asset(
            video_id=vid,
            filename="hack.mp4",
            storage_path=malicious_path,
            file_size_bytes=500,
        )
        db.commit()

    try:
        resp = client.get(f"/api/v1/videos/{vid}/content")
        assert resp.status_code == 400
        assert "escapes permitted upload directory" in resp.json()["detail"]
    finally:
        with SessionLocal() as db:
            a = db.query(VideoAsset).filter(VideoAsset.id == vid).first()
            if a:
                db.delete(a)
                db.commit()


def test_source_video_zero_byte_file():
    """Validates empty file edge cases (200 for full stream, 416 for Range request)."""
    uid = uuid.uuid4().hex[:8]
    vid = f"vid_zero_{uid}"
    file_path = settings.UPLOAD_DIR.resolve() / f"{vid}.mp4"
    file_path.write_bytes(b"")

    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        repo.create_video_asset(
            video_id=vid,
            filename="empty.mp4",
            storage_path=str(file_path),
            file_size_bytes=0,
        )
        db.commit()

    try:
        # Full content on 0-byte file -> 200 with Content-Length 0
        resp_full = client.get(f"/api/v1/videos/{vid}/content")
        assert resp_full.status_code == 200
        assert resp_full.headers.get("Content-Length") == "0"
        assert resp_full.headers.get("Cache-Control") == "private, no-store"
        assert resp_full.content == b""

        # Any range request on 0-byte file -> 416 Unsatisfiable
        resp_range = client.get(
            f"/api/v1/videos/{vid}/content",
            headers={"Range": "bytes=0-"},
        )
        assert resp_range.status_code == 416
        assert resp_range.headers.get("Content-Range") == "bytes */0"
        assert resp_range.headers.get("Cache-Control") == "private, no-store"
    finally:
        if file_path.exists():
            file_path.unlink()
        with SessionLocal() as db:
            a = db.query(VideoAsset).filter(VideoAsset.id == vid).first()
            if a:
                db.delete(a)
                db.commit()


# ==============================================================================
# Integration Tests: Annotated Video Streaming & Artifact Listing
# ==============================================================================

@pytest.fixture
def completed_job_with_artifacts():
    """Sets up a completed job with synthetic on-disk artifacts in output directory."""
    uid = uuid.uuid4().hex[:8]
    vid = f"vid_done_{uid}"
    jid = f"job_done_{uid}"

    output_dir = settings.OUTPUT_DIR.resolve() / jid
    output_dir.mkdir(parents=True, exist_ok=True)

    json_file = output_dir / f"seq_{uid}_analytics.json"
    csv_file = output_dir / f"seq_{uid}_telemetry.csv"
    mp4_file = output_dir / f"seq_{uid}_annotated.mp4"

    json_content = b'{"status": "ok"}'
    csv_content = b"frame,track_id\n1,10\n"
    mp4_content = bytes((i % 256 for i in range(800)))

    json_file.write_bytes(json_content)
    csv_file.write_bytes(csv_content)
    mp4_file.write_bytes(mp4_content)

    artifacts = {
        "analytics_json": str(json_file),
        "telemetry_csv": str(csv_file),
        "annotated_video": str(mp4_file),
    }

    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        repo.create_video_asset(
            video_id=vid,
            filename=f"input_{uid}.mp4",
            storage_path=str(settings.UPLOAD_DIR.resolve() / f"{vid}.mp4"),
            file_size_bytes=1000,
        )
        job = repo.create_job(
            job_id=jid,
            video_id=vid,
            sequence_name=f"seq_{uid}",
            status=JobStatus.COMPLETED,
            config_metadata={"results": {"artifacts": artifacts}},
        )
        db.commit()

    yield {
        "job_id": jid,
        "video_id": vid,
        "mp4_file": mp4_file,
        "mp4_content": mp4_content,
        "json_file": json_file,
        "csv_file": csv_file,
    }

    # Cleanup artifacts
    for f in (json_file, csv_file, mp4_file):
        if f.exists():
            try:
                f.unlink()
            except OSError:
                pass
    if output_dir.exists():
        try:
            output_dir.rmdir()
        except OSError:
            pass


def test_annotated_video_streaming_success(completed_job_with_artifacts):
    """Validates 200 and 206 streaming of rendered annotated video."""
    jid = completed_job_with_artifacts["job_id"]
    content = completed_job_with_artifacts["mp4_content"]
    total = len(content)

    # 1. Full stream
    resp_full = client.get(f"/api/v1/jobs/{jid}/annotated-video")
    assert resp_full.status_code == 200
    assert resp_full.headers.get("Accept-Ranges") == "bytes"
    assert resp_full.headers.get("Content-Length") == str(total)
    assert resp_full.headers.get("Content-Type") == "video/mp4"
    assert resp_full.headers.get("Cache-Control") == "private, no-store"
    assert resp_full.content == content

    # 2. Range stream (bytes 50-149)
    resp_range = client.get(
        f"/api/v1/jobs/{jid}/annotated-video",
        headers={"Range": "bytes=50-149"},
    )
    assert resp_range.status_code == 206
    assert resp_range.headers.get("Content-Range") == f"bytes 50-149/{total}"
    assert resp_range.headers.get("Content-Length") == "100"
    assert resp_range.headers.get("Cache-Control") == "private, no-store"
    assert resp_range.content == content[50:150]


def test_annotated_video_non_completed_job_rejection():
    """Validates 409 Conflict when requesting annotated video for QUEUED, PROCESSING, or FAILED jobs."""
    uid = uuid.uuid4().hex[:8]
    vid = f"vid_prog_{uid}"

    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        repo.create_video_asset(
            video_id=vid,
            filename=f"prog_{uid}.mp4",
            storage_path=str(settings.UPLOAD_DIR.resolve() / f"{vid}.mp4"),
            file_size_bytes=100,
        )
        repo.create_job(job_id=f"job_q_{uid}", video_id=vid, sequence_name="SQ", status=JobStatus.QUEUED)
        repo.create_job(job_id=f"job_p_{uid}", video_id=vid, sequence_name="SP", status=JobStatus.PROCESSING)
        repo.create_job(job_id=f"job_f_{uid}", video_id=vid, sequence_name="SF", status=JobStatus.FAILED)
        db.commit()

    try:
        for jid, st in [(f"job_q_{uid}", 409), (f"job_p_{uid}", 409), (f"job_f_{uid}", 409)]:
            resp = client.get(f"/api/v1/jobs/{jid}/annotated-video")
            assert resp.status_code == st
    finally:
        with SessionLocal() as db:
            for j in [f"job_q_{uid}", f"job_p_{uid}", f"job_f_{uid}"]:
                db.query(ProcessingJob).filter(ProcessingJob.id == j).delete()
            db.query(VideoAsset).filter(VideoAsset.id == vid).delete()
            db.commit()


def test_annotated_video_missing_artifact_metadata():
    """Validates 404 when completed job did not generate an annotated video."""
    uid = uuid.uuid4().hex[:8]
    jid = f"job_novid_{uid}"
    vid = f"vid_{uid}"

    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        repo.create_video_asset(
            video_id=vid,
            filename="sample.mp4",
            storage_path=str(settings.UPLOAD_DIR.resolve() / f"{uid}.mp4"),
            file_size_bytes=100,
        )
        repo.create_job(
            job_id=jid,
            video_id=vid,
            sequence_name="S_NOVID",
            status=JobStatus.COMPLETED,
            config_metadata={"results": {"artifacts": {}}},  # No annotated_video key
        )
        db.commit()

    try:
        resp = client.get(f"/api/v1/jobs/{jid}/annotated-video")
        assert resp.status_code == 404
        assert "was not generated" in resp.json()["detail"].lower()
    finally:
        with SessionLocal() as db:
            db.query(ProcessingJob).filter(ProcessingJob.id == jid).delete()
            db.query(VideoAsset).filter(VideoAsset.id == vid).delete()
            db.commit()


def test_annotated_video_missing_file_on_disk():
    """Validates 404 when metadata references an artifact that does not exist on disk."""
    uid = uuid.uuid4().hex[:8]
    jid = f"job_missingfile_{uid}"
    vid = f"vid_{uid}"
    ghost_output = settings.OUTPUT_DIR.resolve() / jid / "missing_annotated.mp4"

    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        repo.create_video_asset(
            video_id=vid,
            filename="sample.mp4",
            storage_path=str(settings.UPLOAD_DIR.resolve() / f"{uid}.mp4"),
            file_size_bytes=100,
        )
        repo.create_job(
            job_id=jid,
            video_id=vid,
            sequence_name="S_GHOST",
            status=JobStatus.COMPLETED,
            config_metadata={"results": {"artifacts": {"annotated_video": str(ghost_output)}}},
        )
        db.commit()

    try:
        resp = client.get(f"/api/v1/jobs/{jid}/annotated-video")
        assert resp.status_code == 404
        assert "not found on server storage" in resp.json()["detail"].lower()
    finally:
        with SessionLocal() as db:
            db.query(ProcessingJob).filter(ProcessingJob.id == jid).delete()
            db.query(VideoAsset).filter(VideoAsset.id == vid).delete()
            db.commit()


def test_annotated_video_path_traversal_rejection():
    """Validates 403 Forbidden when annotated video path escapes output directory."""
    uid = uuid.uuid4().hex[:8]
    jid = f"job_hack_{uid}"
    vid = f"vid_{uid}"
    malicious_path = str(settings.OUTPUT_DIR.resolve() / ".." / f"system_secret_{uid}.mp4")

    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        repo.create_video_asset(
            video_id=vid,
            filename="sample.mp4",
            storage_path=str(settings.UPLOAD_DIR.resolve() / f"{uid}.mp4"),
            file_size_bytes=100,
        )
        repo.create_job(
            job_id=jid,
            video_id=vid,
            sequence_name="S_HACK",
            status=JobStatus.COMPLETED,
            config_metadata={"results": {"artifacts": {"annotated_video": malicious_path}}},
        )
        db.commit()

    try:
        resp = client.get(f"/api/v1/jobs/{jid}/annotated-video")
        assert resp.status_code == 403
        assert "escapes permitted output directory" in resp.json()["detail"]
    finally:
        with SessionLocal() as db:
            db.query(ProcessingJob).filter(ProcessingJob.id == jid).delete()
            db.query(VideoAsset).filter(VideoAsset.id == vid).delete()
            db.commit()


def test_list_job_artifacts_metadata_endpoint(completed_job_with_artifacts):
    """
    Validates GET /api/v1/jobs/{job_id}/artifacts:
    - Never returns absolute filesystem paths
    - Returns clean filenames, format types, availability status, and serving URLs
    """
    jid = completed_job_with_artifacts["job_id"]
    resp = client.get(f"/api/v1/jobs/{jid}/artifacts")
    assert resp.status_code == 200
    data = resp.json()

    assert data["job_id"] == jid
    assert data["status"] == "COMPLETED"
    artifacts = data["artifacts"]
    assert len(artifacts) == 3

    artifacts_by_name = {a["name"]: a for a in artifacts}

    # 1. Analytics JSON
    json_art = artifacts_by_name["analytics_json"]
    assert json_art["artifact_type"] == "json"
    assert json_art["is_available"] is True
    assert json_art["serving_url"] == f"/api/v1/jobs/{jid}/results"
    assert "storage" not in json_art["filename"]
    assert "\\" not in json_art["filename"]
    assert "/" not in json_art["filename"]
    assert json_art["file_size_bytes"] > 0

    # 2. Telemetry CSV
    csv_art = artifacts_by_name["telemetry_csv"]
    assert csv_art["artifact_type"] == "csv"
    assert csv_art["is_available"] is True
    assert csv_art["serving_url"] is None  # No arbitrary download endpoint
    assert "storage" not in csv_art["filename"]
    assert csv_art["file_size_bytes"] > 0

    # 3. Annotated MP4
    mp4_art = artifacts_by_name["annotated_video"]
    assert mp4_art["artifact_type"] == "video"
    assert mp4_art["is_available"] is True
    assert mp4_art["serving_url"] == f"/api/v1/jobs/{jid}/annotated-video"
    assert "storage" not in mp4_art["filename"]
    assert mp4_art["file_size_bytes"] == 800


def test_list_job_artifacts_non_completed_job():
    """Validates 409 Conflict when listing artifacts for an incomplete job."""
    uid = uuid.uuid4().hex[:8]
    jid = f"job_q_art_{uid}"
    vid = f"vid_{uid}"

    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        repo.create_video_asset(
            video_id=vid,
            filename="sample.mp4",
            storage_path=str(settings.UPLOAD_DIR.resolve() / f"{uid}.mp4"),
            file_size_bytes=100,
        )
        repo.create_job(job_id=jid, video_id=vid, sequence_name="SQ", status=JobStatus.QUEUED)
        db.commit()

    try:
        resp = client.get(f"/api/v1/jobs/{jid}/artifacts")
        assert resp.status_code == 409
    finally:
        with SessionLocal() as db:
            db.query(ProcessingJob).filter(ProcessingJob.id == jid).delete()
            db.query(VideoAsset).filter(VideoAsset.id == vid).delete()
            db.commit()


# ==============================================================================
# Security & Cache Policy Audits
# ==============================================================================

def test_cache_control_consistency_audit(sample_source_video, completed_job_with_artifacts):
    """
    Audits HTTP Cache-Control consistency across source and annotated video endpoints.
    Verifies that 200 OK, 206 Partial Content, and 416 Range Not Satisfiable
    strictly and uniformly return 'private, no-store'.
    """
    vid = sample_source_video["video_id"]
    jid = completed_job_with_artifacts["job_id"]
    expected_cache_policy = "private, no-store"

    # Source video: 200, 206, 416
    r_src_200 = client.get(f"/api/v1/videos/{vid}/content")
    assert r_src_200.status_code == 200
    assert r_src_200.headers.get("Cache-Control") == expected_cache_policy

    r_src_206 = client.get(f"/api/v1/videos/{vid}/content", headers={"Range": "bytes=0-99"})
    assert r_src_206.status_code == 206
    assert r_src_206.headers.get("Cache-Control") == expected_cache_policy

    r_src_416 = client.get(f"/api/v1/videos/{vid}/content", headers={"Range": "bytes=999999-"})
    assert r_src_416.status_code == 416
    assert r_src_416.headers.get("Cache-Control") == expected_cache_policy

    # Annotated video: 200, 206, 416
    r_ann_200 = client.get(f"/api/v1/jobs/{jid}/annotated-video")
    assert r_ann_200.status_code == 200
    assert r_ann_200.headers.get("Cache-Control") == expected_cache_policy

    r_ann_206 = client.get(f"/api/v1/jobs/{jid}/annotated-video", headers={"Range": "bytes=0-99"})
    assert r_ann_206.status_code == 206
    assert r_ann_206.headers.get("Cache-Control") == expected_cache_policy

    r_ann_416 = client.get(f"/api/v1/jobs/{jid}/annotated-video", headers={"Range": "bytes=999999-"})
    assert r_ann_416.status_code == 416
    assert r_ann_416.headers.get("Cache-Control") == expected_cache_policy


def test_file_handle_closed_on_generator_interruption(tmp_path: Path):
    """
    Verifies that file_chunk_generator deterministically releases underlying file handles
    when client closes connection prematurely or generator is aborted.
    """
    from backend.app.services.media_delivery import file_chunk_generator

    test_file = tmp_path / "test_handle.bin"
    test_file.write_bytes(b"A" * 1024 * 1024)  # 1 MB

    gen = file_chunk_generator(test_file, start=0, length=1024 * 1024, chunk_size=1024)
    chunk = next(gen)
    assert len(chunk) == 1024

    # Close generator abruptly (simulates client disconnect / connection drop)
    gen.close()

    # Verify that file can be immediately deleted or written without Windows file-lock (WinError 32)
    test_file.unlink()
    assert not test_file.exists()


def test_path_containment_rejects_external_symlinks(tmp_path: Path):
    """
    Verifies that validate_path_containment resolves canonical paths and rejects
    any symlink pointing outside the permitted directory boundary.
    """
    base_dir = tmp_path / "allowed_uploads"
    base_dir.mkdir()
    secret_dir = tmp_path / "private_system"
    secret_dir.mkdir()
    secret_file = secret_dir / "passwords.txt"
    secret_file.write_text("classified")

    symlink_file = base_dir / "leak_symlink.txt"
    try:
        symlink_file.symlink_to(secret_file)
        # If OS supports symlinks without elevated privilege
        assert validate_path_containment(symlink_file, base_dir) is False
    except OSError:
        # Standard unprivileged Windows environment without Developer Mode
        # Test path resolution containment logic directly
        resolved_secret = secret_file.resolve()
        assert validate_path_containment(resolved_secret, base_dir) is False

