"""
Tests for video ingestion, upload validation, background worker orchestration, and recovery.
"""

from __future__ import annotations
import io
import tempfile
import time
import uuid
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings, get_settings
from backend.app.db.session import SessionLocal
from backend.app.main import app
from backend.app.models.job import ProcessingJob
from backend.app.models.video import VideoAsset
from backend.app.repositories.job_repository import SQLAlchemyJobRepository
from backend.app.schemas.job import JobStatus
from backend.app.services.cv_pipeline_adapter import CVPipelineAdapter
from backend.app.services.worker import JobWorker, get_job_worker, QueueFullError

client = TestClient(app)


@pytest.fixture(autouse=True)
def ensure_worker_running():
    """Ensures background worker is started during test execution."""
    worker = get_job_worker()
    worker.start()
    yield


def generate_synthetic_video_bytes(num_frames: int = 10, width: int = 160, height: int = 120, fps: float = 25.0) -> bytes:
    """Creates a temporary in-memory MP4 video stream."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
    for f in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw moving colored block
        x = int(10 + f * 5)
        y = int(20 + f * 3)
        cv2.rectangle(frame, (x, y), (x + 30, y + 20), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()

    video_bytes = tmp_path.read_bytes()
    try:
        tmp_path.unlink()
    except PermissionError:
        pass
    return video_bytes


# ==============================================================================
# 1. Video Upload & Metadata Extraction Tests
# ==============================================================================

def test_successful_video_upload_and_metadata():
    """Validates streaming upload, OpenCV metadata extraction, and persistence."""
    video_bytes = generate_synthetic_video_bytes(num_frames=12, width=160, height=120, fps=25.0)

    response = client.post(
        "/api/v1/videos",
        files={"file": ("test_traffic.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    assert response.status_code == 201
    data = response.json()
    assert "video_id" in data
    vid = data["video_id"]
    meta = data["metadata"]

    assert meta["filename"] == "test_traffic.mp4"
    assert meta["width"] == 160
    assert meta["height"] == 120
    assert meta["fps"] == 25.0
    assert meta["frame_count"] == 12
    assert meta["duration_seconds"] == 0.48

    # Query video endpoint
    query_resp = client.get(f"/api/v1/videos/{vid}")
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    assert q_data["video_id"] == vid
    assert q_data["width"] == 160


def test_upload_invalid_and_corrupt_files():
    """Validates rejection of empty files, text files, and unsupported extensions."""
    # 1. Unsupported extension
    resp_ext = client.post(
        "/api/v1/videos",
        files={"file": ("malicious.exe", io.BytesIO(b"content"), "application/octet-stream")},
    )
    assert resp_ext.status_code == 415
    assert "Unsupported video extension" in resp_ext.json()["detail"]

    # 2. Empty file (0 bytes)
    resp_empty = client.post(
        "/api/v1/videos",
        files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")},
    )
    assert resp_empty.status_code == 400
    assert "empty" in resp_empty.json()["detail"]

    # 3. Corrupt non-video content with .mp4 extension
    resp_corrupt = client.post(
        "/api/v1/videos",
        files={"file": ("fake.mp4", io.BytesIO(b"Not a real video stream"), "video/mp4")},
    )
    assert resp_corrupt.status_code == 400
    assert "cannot be opened as a valid video stream" in resp_corrupt.json()["detail"]


def test_file_size_limit_and_partial_cleanup():
    """Validates that oversized uploads are aborted and partial files cleaned up."""
    from backend.app.dependencies import get_app_settings
    custom_settings = Settings(MAX_UPLOAD_SIZE_BYTES=500)
    app.dependency_overrides[get_app_settings] = lambda: custom_settings
    try:
        payload = b"X" * 1024  # 1 KB > 500 B limit
        resp = client.post(
            "/api/v1/videos",
            files={"file": ("large.mp4", io.BytesIO(payload), "video/mp4")},
        )
        assert resp.status_code == 413
        assert "exceeds maximum upload limit" in resp.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_app_settings, None)


def test_get_nonexistent_video():
    """Validates 404 response for unknown video assets."""
    resp = client.get("/api/v1/videos/vid_non_existent_123")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ==============================================================================
# 2. Job Creation and Duplicate Scheduling Tests
# ==============================================================================

def test_video_job_creation_and_duplicate_prevention():
    """Validates job creation and rejection of duplicate active jobs for same video."""
    video_bytes = generate_synthetic_video_bytes(num_frames=5)
    up_resp = client.post(
        "/api/v1/videos",
        files={"file": ("traffic_dup.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    vid = up_resp.json()["video_id"]

    # 1. Create first job -> 202 Accepted
    job_resp1 = client.post(f"/api/v1/videos/{vid}/jobs")
    assert job_resp1.status_code == 202
    job1_data = job_resp1.json()
    assert job1_data["status"] == "QUEUED"
    assert job1_data["video_id"] == vid

    # 2. Immediate duplicate job creation attempt -> 409 Conflict
    dup_resp = client.post(f"/api/v1/videos/{vid}/jobs")
    assert dup_resp.status_code == 409
    assert "already has an active processing job" in dup_resp.json()["detail"]


# ==============================================================================
# 3. Worker Execution & Status Transitions Tests
# ==============================================================================

def test_worker_processing_and_results_flow():
    """Validates asynchronous worker execution, completion, and results endpoint."""
    video_bytes = generate_synthetic_video_bytes(num_frames=8, width=160, height=120)
    up_resp = client.post(
        "/api/v1/videos",
        files={"file": ("worker_flow.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    vid = up_resp.json()["video_id"]

    # Create job with custom counting line
    job_payload = {
        "sequence_name": "SYNTH_TEST",
        "lines": [
            {
                "line_id": "line_test",
                "name": "Test Crossline",
                "p1": [0.0, 60.0],
                "p2": [160.0, 60.0],
                "positive_direction_label": "downstream",
                "negative_direction_label": "upstream",
                "cooldown_frames": 2,
                "min_track_age": 1,
            }
        ],
    }
    job_resp = client.post(f"/api/v1/videos/{vid}/jobs", json=job_payload)
    assert job_resp.status_code == 202
    job_id = job_resp.json()["job_id"]

    # Wait for background worker execution (allowing model weight loading time)
    worker = get_job_worker()
    max_wait_s = 35.0
    t0 = time.time()
    final_status = "QUEUED"

    while time.time() - t0 < max_wait_s:
        status_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert status_resp.status_code == 200
        cur_status = status_resp.json()["status"]
        if cur_status in ("COMPLETED", "FAILED"):
            final_status = cur_status
            break
        time.sleep(0.3)

    assert final_status == "COMPLETED", f"Job failed to complete: {client.get(f'/api/v1/jobs/{job_id}').json()}"

    # Query results endpoint
    results_resp = client.get(f"/api/v1/jobs/{job_id}/results")
    assert results_resp.status_code == 200
    res_data = results_resp.json()
    assert res_data["status"] == "COMPLETED"
    assert res_data["detection_summary"] is not None
    assert res_data["tracking_summary"] is not None
    assert res_data["analytics_summary"] is not None
    assert "analytics_json" in res_data["artifacts"]
    assert Path(res_data["artifacts"]["analytics_json"]).exists()


def test_job_results_in_progress_rejection():
    """Validates that GET /results rejects jobs that are not yet COMPLETED (409 Conflict)."""
    uid = uuid.uuid4().hex[:8]
    vid = f"vid_pend_{uid}"
    jid = f"job_pend_{uid}"
    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        asset = repo.create_video_asset(
            video_id=vid,
            filename=f"pend_{uid}.mp4",
            storage_path=f"/pend_{uid}.mp4",
            file_size_bytes=100,
        )
        job = repo.create_job(job_id=jid, video_id=vid, sequence_name="S_PEND")
        db.commit()

    resp = client.get(f"/api/v1/jobs/{jid}/results")
    assert resp.status_code == 409
    assert "Results are not available yet" in resp.json()["detail"]


# ==============================================================================
# 4. Startup Recovery & Worker Exception Tests
# ==============================================================================

def test_startup_recovery_interrupted_jobs():
    """Validates that worker marks interrupted PROCESSING jobs as FAILED on startup."""
    uid = uuid.uuid4().hex[:8]
    vid = f"vid_rec_{uid}"
    jid = f"job_interrupted_{uid}"
    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        asset = repo.create_video_asset(
            video_id=vid,
            filename=f"rec_{uid}.mp4",
            storage_path=f"/rec_{uid}.mp4",
            file_size_bytes=100,
        )
        # Insert a job stuck in PROCESSING (simulating server crash)
        job = repo.create_job(
            job_id=jid,
            video_id=vid,
            sequence_name="S_REC",
            status=JobStatus.PROCESSING,
        )
        db.commit()

    # Trigger recovery routine
    worker = JobWorker()
    worker.recover_jobs()

    with SessionLocal() as db:
        recovered = db.get(ProcessingJob, jid)
        assert recovered.status == JobStatus.FAILED
        assert recovered.error_code == "INTERRUPTED_BY_SHUTDOWN"
        assert "interrupted by application restart" in recovered.error_message


def test_worker_missing_video_file_handling():
    """Validates that worker transitions job to FAILED when source video file is missing."""
    uid = uuid.uuid4().hex[:8]
    vid = f"vid_missing_{uid}"
    jid = f"job_missing_{uid}"
    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        asset = repo.create_video_asset(
            video_id=vid,
            filename=f"missing_{uid}.mp4",
            storage_path=f"/non_existent/path/missing_{uid}.mp4",
            file_size_bytes=100,
        )
        job = repo.create_job(
            job_id=jid,
            video_id=vid,
            sequence_name="S_MISSING",
            status=JobStatus.QUEUED,
        )
        db.commit()

    worker = JobWorker()
    worker._process_single_job(jid)

    with SessionLocal() as db:
        failed_job = db.get(ProcessingJob, jid)
        assert failed_job.status == JobStatus.FAILED
        assert failed_job.error_code == "VIDEO_NOT_FOUND"


def test_worker_queue_capacity_limit_and_503(monkeypatch):
    """Validates that worker queue capacity is enforced and rejects when full (503)."""
    worker = get_job_worker()

    def mock_enqueue(job_id: str):
        raise QueueFullError(
            f"Worker queue is at maximum capacity ({worker.settings.WORKER_MAX_QUEUE_SIZE}). Please retry later."
        )

    monkeypatch.setattr(worker, "enqueue_job", mock_enqueue)

    # Attempt to queue a job
    video_bytes = generate_synthetic_video_bytes(num_frames=3)
    up_resp = client.post(
        "/api/v1/videos",
        files={"file": ("q_test.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    vid = up_resp.json()["video_id"]

    resp = client.post(f"/api/v1/videos/{vid}/jobs")
    assert resp.status_code == 503
    assert "Worker queue is at maximum capacity" in resp.json()["detail"]


def test_concurrent_duplicate_scheduling_prevention():
    """Validates that concurrent job creation requests cannot schedule duplicate active jobs."""
    import concurrent.futures

    video_bytes = generate_synthetic_video_bytes(num_frames=3)
    up_resp = client.post(
        "/api/v1/videos",
        files={"file": ("race_test.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    vid = up_resp.json()["video_id"]

    def submit_job():
        return client.post(f"/api/v1/videos/{vid}/jobs")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(submit_job) for _ in range(5)]
        results = [f.result() for f in futures]

    status_codes = [r.status_code for r in results]
    # Exactly one request must succeed with 202 Accepted, and all others rejected with 409 Conflict
    assert status_codes.count(202) == 1
    assert status_codes.count(409) == 4


def test_queue_overflow_rollback_allows_clean_retry(monkeypatch):
    """Validates that a queue-full response (503) does not leave a stranded QUEUED job in DB."""
    worker = get_job_worker()
    call_count = 0
    orig_enqueue = worker.enqueue_job

    def failing_enqueue(job_id):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise QueueFullError("Worker queue is at maximum capacity (1). Please retry later.")
        return orig_enqueue(job_id)

    monkeypatch.setattr(worker, "enqueue_job", failing_enqueue)

    video_bytes = generate_synthetic_video_bytes(num_frames=3)
    up_resp = client.post(
        "/api/v1/videos",
        files={"file": ("rollback_test.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    assert up_resp.status_code == 201
    vid = up_resp.json()["video_id"]

    # 1. Post job while queue is full -> 503
    resp1 = client.post(f"/api/v1/videos/{vid}/jobs")
    assert resp1.status_code == 503
    assert "Worker queue is at maximum capacity" in resp1.json()["detail"]

    # 2. Verify NO stranded active job exists in DB for this video
    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        asset = repo.get_video_asset(vid)
        active_jobs = [j for j in asset.jobs if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING)]
        assert len(active_jobs) == 0

    # 3. Retry scheduling job for the same video -> Must succeed with 202 Accepted (not 409 Conflict)
    resp2 = client.post(f"/api/v1/videos/{vid}/jobs")
    assert resp2.status_code == 202
    job_id = resp2.json()["job_id"]
    assert resp2.json()["status"] == "QUEUED"

    # Cleanup test job
    with SessionLocal() as db:
        j = db.get(ProcessingJob, job_id)
        if j:
            db.delete(j)
        a = db.get(VideoAsset, vid)
        if a:
            db.delete(a)
        db.commit()


def test_durable_scheduling_redispatch_pending_jobs():
    """Validates that persisted QUEUED jobs are redispatched dynamically without application restart."""
    uid = uuid.uuid4().hex[:8]
    vid = f"vid_disp_{uid}"
    jid = f"job_disp_{uid}"

    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        repo.create_video_asset(
            video_id=vid,
            filename=f"disp_{uid}.mp4",
            storage_path=f"/fake_path/{uid}.mp4",
            file_size_bytes=100,
        )
        repo.create_job(job_id=jid, video_id=vid, sequence_name="S_DISP", status=JobStatus.QUEUED)
        db.commit()

    worker = JobWorker()
    assert worker._queue.qsize() == 0

    # Redispatch scans DB and enqueues the pending QUEUED job
    count = worker.redispatch_pending_jobs()
    assert count >= 1
    assert jid in worker._queued_ids

    # Calling redispatch again must not duplicate the job in the queue
    count2 = worker.redispatch_pending_jobs()
    assert count2 == 0

    # Cleanup test job
    with SessionLocal() as db:
        j = db.get(ProcessingJob, jid)
        if j:
            db.delete(j)
        a = db.get(VideoAsset, vid)
        if a:
            db.delete(a)
        db.commit()


def test_shutdown_during_active_processing_interrupts_cleanly(monkeypatch):
    """Validates that worker.stop() interrupts in-flight processing cleanly to FAILED status."""
    import threading
    from sqlalchemy import delete

    uid = uuid.uuid4().hex[:8]
    vid = f"vid_shut_{uid}"
    jid = f"job_shut_{uid}"
    fake_path = Path(f"storage/uploads/fake_shut_{uid}.mp4")
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    fake_path.touch()

    with SessionLocal() as db:
        db.execute(delete(ProcessingJob).where(ProcessingJob.status == JobStatus.QUEUED))
        repo = SQLAlchemyJobRepository(db)
        repo.create_video_asset(
            video_id=vid,
            filename=f"shut_{uid}.mp4",
            storage_path=str(fake_path),
            file_size_bytes=100,
        )
        repo.create_job(job_id=jid, video_id=vid, sequence_name="S_SHUT", status=JobStatus.QUEUED)
        db.commit()

    in_inference_event = threading.Event()

    def mock_hang_until_stop(*args, **kwargs):
        in_inference_event.set()
        stop_event = kwargs.get("stop_event")
        if stop_event:
            # Wait for stop signal
            if stop_event.wait(timeout=5.0):
                raise InterruptedError("Video processing interrupted by shutdown.")
        raise RuntimeError("Inference did not receive stop signal")

    monkeypatch.setattr(CVPipelineAdapter, "process_video_file", mock_hang_until_stop)

    test_worker = JobWorker()
    test_worker.start()

    # Wait until inference starts
    assert in_inference_event.wait(timeout=5.0)

    # Stop worker while inference is active
    test_worker.stop(timeout=2.0)

    # Verify job in DB is marked FAILED with INTERRUPTED_BY_SHUTDOWN
    with SessionLocal() as db:
        job = db.get(ProcessingJob, jid)
        assert job.status == JobStatus.FAILED
        assert job.error_code == "INTERRUPTED_BY_SHUTDOWN"
        if job:
            db.delete(job)
        a = db.get(VideoAsset, vid)
        if a:
            db.delete(a)
        db.commit()

    if fake_path.exists():
        fake_path.unlink()


def test_terminal_state_cannot_be_overwritten_by_delayed_completion(monkeypatch):
    """Validates that a terminal state (FAILED) cannot be overwritten if inference finishes later."""
    uid = uuid.uuid4().hex[:8]
    vid = f"vid_term_{uid}"
    jid = f"job_term_{uid}"
    fake_path = Path(f"storage/uploads/fake_{uid}.mp4")
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    fake_path.touch()

    with SessionLocal() as db:
        repo = SQLAlchemyJobRepository(db)
        repo.create_video_asset(
            video_id=vid,
            filename=f"term_{uid}.mp4",
            storage_path=str(fake_path),
            file_size_bytes=100,
        )
        repo.create_job(job_id=jid, video_id=vid, sequence_name="S_TERM", status=JobStatus.QUEUED)
        db.commit()

    def mock_process_and_abort(*args, **kwargs):
        # Simulate worker.stop() timeout marking job as FAILED while inference was in flight
        with SessionLocal() as db:
            from sqlalchemy import update
            db.execute(
                update(ProcessingJob)
                .where(ProcessingJob.id == jid)
                .values(
                    status=JobStatus.FAILED,
                    error_code="INTERRUPTED_BY_SHUTDOWN",
                    error_message="Job interrupted by shutdown timeout",
                )
            )
            db.commit()

        from backend.app.schemas import DetectionSummarySchema, TrackingSummarySchema
        from experiments.src.analytics import SequenceAnalyticsReport
        report = SequenceAnalyticsReport(
            sequence_name="mock", total_frames=1, duration_s=1.0, fps=25.0,
            total_unique_vehicles=0, line_counts={}, flow_windows=[], stopped_vehicle_events=[],
            peak_occupancy_per_roi={}, average_occupancy_per_roi={}, congestion_distribution_per_roi={},
        )
        return report, DetectionSummarySchema(total_detections=0, mean_confidence=0.0, class_distribution={}), TrackingSummarySchema(total_unique_tracks=0), {}

    monkeypatch.setattr(CVPipelineAdapter, "process_video_file", mock_process_and_abort)

    worker = JobWorker()
    worker._process_single_job(jid)

    # Verify that status remains FAILED and was NOT overwritten by COMPLETED
    with SessionLocal() as db:
        job = db.get(ProcessingJob, jid)
        assert job.status == JobStatus.FAILED
        assert job.error_code == "INTERRUPTED_BY_SHUTDOWN"
        if job:
            db.delete(job)
        a = db.get(VideoAsset, vid)
        if a:
            db.delete(a)
        db.commit()

    if fake_path.exists():
        fake_path.unlink()
