"""
Comprehensive tests for Scene Interpretation FastAPI endpoints in TrafficLens-AI.

Covers:
1. Successful interpretation request (POST /api/v1/jobs/{job_id}/interpretations)
2. Unknown job ID rejection (404 Not Found)
3. Non-completed CV job rejection (409 Conflict for active, 400 Bad Request for failed)
4. Duplicate active interpretation rejection (409 Conflict)
5. Queue capacity exhaustion handling (503 Service Unavailable with clean record rollback)
6. Successful status retrieval (GET /api/v1/interpretations/{id} and GET /api/v1/jobs/{job_id}/interpretation)
7. Unknown interpretation ID handling (404 Not Found)
8. Completed interpretation response serialization (analytical fields populated, error is None)
9. Failed interpretation response serialization (analytical fields None, sanitized error exposed)
10. Interpretation attempt history ordering (GET /api/v1/jobs/{job_id}/interpretations)
11. Successful retry of a failed interpretation (creates new attempt, preserves previous failed record)
12. Retry rejection for non-failed records (400 Bad Request for completed/active/unrequested)
13. Maximum attempt limit enforcement (409 Conflict when 3 total attempts reached)
14. Concurrent creation and retry serialization (_interpretation_create_lock)
15. Non-exposure of sensitive internal fields (no base64 images, server paths, or tracebacks)
"""

from __future__ import annotations
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.db.session import SessionLocal
from backend.app.main import app
from backend.app.models.job import ProcessingJob
from backend.app.models.scene_interpretation import (
    SceneInterpretation,
    SceneInterpretationStatus,
)
from backend.app.models.video import VideoAsset
from backend.app.repositories.scene_interpretation_repository import (
    SQLAlchemySceneInterpretationRepository,
)
from backend.app.schemas.job import JobStatus
from backend.app.services.vlm_worker import (
    QueueFullError,
    get_interpretation_worker,
)

client = TestClient(app)


# ------------------------------------------------------------------------------
# Test Setup & Teardown Helpers
# ------------------------------------------------------------------------------

def _create_api_test_job(
    cv_status: JobStatus = JobStatus.COMPLETED,
    interpretation_status: Optional[SceneInterpretationStatus] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    summary: Optional[str] = None,
) -> tuple[str, str, Optional[str]]:
    """Seeds VideoAsset, ProcessingJob, and optional SceneInterpretation in DB."""
    uid = uuid.uuid4().hex[:8]
    vid = f"vid_api_{uid}"
    jid = f"job_api_{uid}"
    iid = f"interp_api_{uid}" if interpretation_status else None

    with SessionLocal() as db:
        video = VideoAsset(
            id=vid,
            filename=f"test_{uid}.mp4",
            storage_path=f"storage/uploads/test_{uid}.mp4",
            file_size_bytes=1024,
            fps=25.0,
            duration_seconds=1.0,
        )
        db.add(video)
        db.flush()

        job = ProcessingJob(
            id=jid,
            video_id=vid,
            sequence_name="SEQ_API_TEST",
            status=cv_status,
            config_metadata={"results": {"analytics_summary": {}, "detection_summary": {}, "tracking_summary": {}}},
        )
        db.add(job)
        db.flush()

        if interpretation_status:
            now = datetime.now(timezone.utc)
            interp = SceneInterpretation(
                id=iid,
                job_id=jid,
                status=interpretation_status,
                provider_type="mock",
                model_name="mock-qwen3-vl-2b",
                summary=summary,
                error_code=error_code,
                error_message=error_message,
                started_at=now,
                completed_at=now if interpretation_status in (SceneInterpretationStatus.COMPLETED, SceneInterpretationStatus.FAILED) else None,
                duration_seconds=1.5 if interpretation_status in (SceneInterpretationStatus.COMPLETED, SceneInterpretationStatus.FAILED) else None,
            )
            if interpretation_status == SceneInterpretationStatus.COMPLETED:
                interp.environmental_context = {"weather": "clear", "lighting": "daylight", "pavement": "dry"}
                interp.scene_description = {"roadway_topology": "divided_highway", "features": ["shoulder"]}
                interp.event_interpretations = [
                    {"event_id": "evt_1", "track_id": 1, "interpretation": "Stopped vehicle waiting."}
                ]
                interp.epistemic_claims = [
                    {"claim_id": "c1", "category": "inference", "statement": "Traffic flowing normally.", "evidence_rationale": "Clear roadway."}
                ]
                interp.evidence_package_id = f"evpkg_{jid}_3k"
                interp.prompt_tokens = 150
                interp.completion_tokens = 50
                interp.total_tokens = 200

            db.add(interp)

        db.commit()

    return vid, jid, iid


def _delete_api_test_job(vid: str, jid: str, iid: Optional[str] = None) -> None:
    """Removes test records from database."""
    with SessionLocal() as db:
        if iid:
            interp = db.get(SceneInterpretation, iid)
            if interp:
                db.delete(interp)
        # Delete any other interpretations for job
        repo = SQLAlchemySceneInterpretationRepository(db)
        for a in repo.list_by_job_id(jid):
            db.delete(a)

        job = db.get(ProcessingJob, jid)
        if job:
            db.delete(job)
        video = db.get(VideoAsset, vid)
        if video:
            db.delete(video)
        db.commit()


# ------------------------------------------------------------------------------
# 1. Successful Interpretation Request
# ------------------------------------------------------------------------------

def test_request_scene_interpretation_success():
    """Validates successful POST /jobs/{job_id}/interpretations returning 202 Accepted and QUEUED."""
    vid, jid, _ = _create_api_test_job(cv_status=JobStatus.COMPLETED)
    try:
        response = client.post(f"/api/v1/jobs/{jid}/interpretations")
        assert response.status_code == 202
        data = response.json()

        assert data["job_id"] == jid
        assert data["status"] == "QUEUED"
        assert data["id"] is not None
        assert data["id"].startswith("interp_")
        assert data["created_at"] is not None

        # Verify DB status
        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)
            saved = repo.get_by_id(data["id"])
            assert saved is not None
            assert saved.status == SceneInterpretationStatus.QUEUED
    finally:
        _delete_api_test_job(vid, jid)


# ------------------------------------------------------------------------------
# 2. Unknown Job ID Handling
# ------------------------------------------------------------------------------

def test_request_scene_interpretation_unknown_job():
    """Validates 404 Not Found when requesting interpretation for non-existent job."""
    response = client.post("/api/v1/jobs/job_unknown_nonexistent_12345/interpretations")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ------------------------------------------------------------------------------
# 3. Non-Completed CV Job Rejection
# ------------------------------------------------------------------------------

def test_request_scene_interpretation_rejects_active_cv_job():
    """Validates 409 Conflict when CV job is still QUEUED or PROCESSING."""
    vid, jid, _ = _create_api_test_job(cv_status=JobStatus.PROCESSING)
    try:
        response = client.post(f"/api/v1/jobs/{jid}/interpretations")
        assert response.status_code == 409
        assert "still in progress" in response.json()["detail"].lower()
    finally:
        _delete_api_test_job(vid, jid)


def test_request_scene_interpretation_rejects_failed_cv_job():
    """Validates 400 Bad Request when CV job is FAILED."""
    vid, jid, _ = _create_api_test_job(cv_status=JobStatus.FAILED)
    try:
        response = client.post(f"/api/v1/jobs/{jid}/interpretations")
        assert response.status_code == 400
        assert "failed cv job" in response.json()["detail"].lower()
    finally:
        _delete_api_test_job(vid, jid)


# ------------------------------------------------------------------------------
# 4. Duplicate Active Interpretation Rejection
# ------------------------------------------------------------------------------

def test_request_scene_interpretation_rejects_duplicate_active():
    """Validates 409 Conflict when job already has an active (QUEUED/PROCESSING) interpretation."""
    vid, jid, iid = _create_api_test_job(
        cv_status=JobStatus.COMPLETED,
        interpretation_status=SceneInterpretationStatus.QUEUED,
    )
    try:
        response = client.post(f"/api/v1/jobs/{jid}/interpretations")
        assert response.status_code == 409
        assert "already in status" in response.json()["detail"].lower()
    finally:
        _delete_api_test_job(vid, jid, iid)


# ------------------------------------------------------------------------------
# 5. Queue Capacity Exhaustion Handling
# ------------------------------------------------------------------------------

def test_request_scene_interpretation_queue_capacity_exhaustion(monkeypatch):
    """Validates 503 Service Unavailable and database rollback when worker queue is full."""
    vid, jid, _ = _create_api_test_job(cv_status=JobStatus.COMPLETED)
    worker = get_interpretation_worker()

    def mock_enqueue_full(*args, **kwargs):
        raise QueueFullError("Worker queue is at maximum capacity")

    monkeypatch.setattr(worker, "enqueue_interpretation", mock_enqueue_full)

    try:
        response = client.post(f"/api/v1/jobs/{jid}/interpretations")
        assert response.status_code == 503
        assert response.headers.get("Retry-After") == "5"
        assert "queue is full" in response.json()["detail"].lower()

        # Verify that created record was cleaned up and does not remain stranded in DB
        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)
            active = repo.get_active_for_job(jid)
            assert active is None
            assert repo.count_by_job_id(jid) == 0
    finally:
        _delete_api_test_job(vid, jid)


# ------------------------------------------------------------------------------
# 6. Status Retrieval & Virtual NOT_REQUESTED Status
# ------------------------------------------------------------------------------

def test_get_interpretation_by_id_success():
    """Validates GET /api/v1/interpretations/{id} returns interpretation details."""
    vid, jid, iid = _create_api_test_job(
        cv_status=JobStatus.COMPLETED,
        interpretation_status=SceneInterpretationStatus.PROCESSING,
    )
    try:
        response = client.get(f"/api/v1/interpretations/{iid}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == iid
        assert data["job_id"] == jid
        assert data["status"] == "PROCESSING"
        # Incomplete result protection: analytical fields must be None
        assert data["summary"] is None
        assert data["environmental_context"] is None
        assert data["error"] is None
    finally:
        _delete_api_test_job(vid, jid, iid)


def test_get_latest_interpretation_not_requested():
    """Validates GET /jobs/{job_id}/interpretation returns virtual NOT_REQUESTED when none exists."""
    vid, jid, _ = _create_api_test_job(cv_status=JobStatus.COMPLETED)
    try:
        response = client.get(f"/api/v1/jobs/{jid}/interpretation")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "NOT_REQUESTED"
        assert data["job_id"] == jid
        assert data["id"] is None
    finally:
        _delete_api_test_job(vid, jid)


# ------------------------------------------------------------------------------
# 7. Unknown Interpretation ID Handling
# ------------------------------------------------------------------------------

def test_get_interpretation_by_id_unknown():
    """Validates 404 Not Found for non-existent interpretation ID."""
    response = client.get("/api/v1/interpretations/interp_nonexistent_12345")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ------------------------------------------------------------------------------
# 8. Completed Interpretation Response Serialization
# ------------------------------------------------------------------------------

def test_get_completed_interpretation_serialization():
    """Validates serialization of COMPLETED interpretation exposing all analytical fields."""
    vid, jid, iid = _create_api_test_job(
        cv_status=JobStatus.COMPLETED,
        interpretation_status=SceneInterpretationStatus.COMPLETED,
        summary="Clear weather with moderate traffic flow across all lanes.",
    )
    try:
        response = client.get(f"/api/v1/interpretations/{iid}")
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "COMPLETED"
        assert data["summary"] == "Clear weather with moderate traffic flow across all lanes."
        assert data["environmental_context"] is not None
        assert data["environmental_context"]["weather"] == "clear"
        assert data["scene_description"]["roadway_topology"] == "divided_highway"
        assert len(data["event_interpretations"]) == 1
        assert len(data["epistemic_claims"]) == 1
        assert data["evidence_package_id"].startswith(f"evpkg_{jid}")
        assert data["total_tokens"] == 200
        assert data["error"] is None
    finally:
        _delete_api_test_job(vid, jid, iid)


# ------------------------------------------------------------------------------
# 9. Failed Interpretation Response Serialization
# ------------------------------------------------------------------------------

def test_get_failed_interpretation_serialization():
    """Validates serialization of FAILED interpretation: analytical fields stripped, sanitized error exposed."""
    vid, jid, iid = _create_api_test_job(
        cv_status=JobStatus.COMPLETED,
        interpretation_status=SceneInterpretationStatus.FAILED,
        error_code="VLM_PROVIDER_TIMEOUT",
        error_message="Remote inference timed out after 30.0s.",
    )
    try:
        response = client.get(f"/api/v1/interpretations/{iid}")
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "FAILED"
        assert data["error"] is not None
        assert data["error"]["error_code"] == "VLM_PROVIDER_TIMEOUT"
        assert "timed out" in data["error"]["error_message"].lower()

        # Strict incomplete result protection: all analytical fields must be None
        assert data["summary"] is None
        assert data["environmental_context"] is None
        assert data["scene_description"] is None
        assert data["event_interpretations"] is None
        assert data["epistemic_claims"] is None
        assert data["evidence_package_id"] is None
    finally:
        _delete_api_test_job(vid, jid, iid)


# ------------------------------------------------------------------------------
# 10. Interpretation Attempt History Ordering
# ------------------------------------------------------------------------------

def test_get_interpretation_history_ordering():
    """Validates GET /jobs/{job_id}/interpretations returns historical attempts in reverse-chronological order."""
    vid, jid, _ = _create_api_test_job(cv_status=JobStatus.COMPLETED)
    iids = []

    try:
        # Create 2 sequential attempts with distinct timestamps
        for idx in range(2):
            iid_k = f"interp_hist_{idx}_{uuid.uuid4().hex[:6]}"
            with SessionLocal() as db:
                interp = SceneInterpretation(
                    id=iid_k,
                    job_id=jid,
                    status=SceneInterpretationStatus.FAILED,
                    provider_type="mock",
                    model_name="mock-qwen3-vl-2b",
                    error_code="TEST_ERROR",
                    error_message=f"Attempt {idx} failure.",
                    created_at=datetime(2026, 11, 23, 10, idx, 0, tzinfo=timezone.utc),
                )
                db.add(interp)
                db.commit()
            iids.append(iid_k)

        response = client.get(f"/api/v1/jobs/{jid}/interpretations")
        assert response.status_code == 200
        data = response.json()

        assert data["job_id"] == jid
        assert data["total_attempts"] == 2
        assert len(data["attempts"]) == 2
        # Verify chronological ordering by repository convention (oldest first)
        assert data["attempts"][0]["id"] == iids[0]
        assert data["attempts"][1]["id"] == iids[1]

        # Also verify ascending=false (newest first)
        resp_desc = client.get(f"/api/v1/jobs/{jid}/interpretations?ascending=false")
        assert resp_desc.status_code == 200
        data_desc = resp_desc.json()
        assert data_desc["attempts"][0]["id"] == iids[1]
        assert data_desc["attempts"][1]["id"] == iids[0]
    finally:
        _delete_api_test_job(vid, jid)


# ------------------------------------------------------------------------------
# 11. Successful Retry of a Failed Interpretation
# ------------------------------------------------------------------------------

def test_retry_failed_interpretation_success():
    """Validates retry of a FAILED interpretation creates a new QUEUED attempt preserving prior failure."""
    vid, jid, iid_orig = _create_api_test_job(
        cv_status=JobStatus.COMPLETED,
        interpretation_status=SceneInterpretationStatus.FAILED,
        error_code="VLM_PROVIDER_TIMEOUT",
        error_message="Original attempt timed out.",
    )
    try:
        response = client.post(f"/api/v1/jobs/{jid}/interpretations/retry")
        assert response.status_code == 202
        data = response.json()

        assert data["status"] == "QUEUED"
        assert data["job_id"] == jid
        assert data["id"] != iid_orig  # Must be a NEW record ID

        # Verify both records exist in DB
        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)
            assert repo.count_by_job_id(jid) == 2

            orig = repo.get_by_id(iid_orig)
            assert orig.status == SceneInterpretationStatus.FAILED
            assert orig.error_code == "VLM_PROVIDER_TIMEOUT"

            new_attempt = repo.get_by_id(data["id"])
            assert new_attempt.status == SceneInterpretationStatus.QUEUED
    finally:
        _delete_api_test_job(vid, jid)


# ------------------------------------------------------------------------------
# 12. Retry Rejection for Non-Failed Records
# ------------------------------------------------------------------------------

def test_retry_rejected_for_completed_interpretation():
    """Validates 400 Bad Request when attempting to retry a COMPLETED interpretation."""
    vid, jid, iid = _create_api_test_job(
        cv_status=JobStatus.COMPLETED,
        interpretation_status=SceneInterpretationStatus.COMPLETED,
        summary="Successful interpretation.",
    )
    try:
        response = client.post(f"/api/v1/jobs/{jid}/interpretations/retry")
        assert response.status_code == 400
        assert "only failed interpretations can be retried" in response.json()["detail"].lower()
    finally:
        _delete_api_test_job(vid, jid, iid)


def test_retry_rejected_when_no_prior_interpretation():
    """Validates 400 Bad Request when attempting to retry a job with no prior interpretations."""
    vid, jid, _ = _create_api_test_job(cv_status=JobStatus.COMPLETED)
    try:
        response = client.post(f"/api/v1/jobs/{jid}/interpretations/retry")
        assert response.status_code == 400
        assert "no prior interpretation" in response.json()["detail"].lower()
    finally:
        _delete_api_test_job(vid, jid)


# ------------------------------------------------------------------------------
# 13. Maximum Attempt Limit Enforcement
# ------------------------------------------------------------------------------

def test_maximum_attempt_limit_enforced():
    """Validates 409 Conflict when attempting to create/retry beyond MAX_INTERPRETATION_ATTEMPTS (3)."""
    vid, jid, _ = _create_api_test_job(cv_status=JobStatus.COMPLETED)
    settings = get_settings()
    max_attempts = settings.MAX_INTERPRETATION_ATTEMPTS

    try:
        # Seed max_attempts failed attempts in DB
        for i in range(max_attempts):
            iid_k = f"interp_max_{i}_{uuid.uuid4().hex[:6]}"
            with SessionLocal() as db:
                interp = SceneInterpretation(
                    id=iid_k,
                    job_id=jid,
                    status=SceneInterpretationStatus.FAILED,
                    provider_type="mock",
                    model_name="mock-qwen3-vl-2b",
                    error_code="TEST_MAX_ATTEMPT",
                    error_message=f"Attempt {i + 1} failure.",
                )
                db.add(interp)
                db.commit()

        # 1. Attempting new request must be rejected with 409
        resp_post = client.post(f"/api/v1/jobs/{jid}/interpretations")
        assert resp_post.status_code == 409
        assert "maximum interpretation attempt limit" in resp_post.json()["detail"].lower()

        # 2. Attempting retry must also be rejected with 409
        resp_retry = client.post(f"/api/v1/jobs/{jid}/interpretations/retry")
        assert resp_retry.status_code == 409
        assert "maximum interpretation attempt limit" in resp_retry.json()["detail"].lower()
    finally:
        _delete_api_test_job(vid, jid)


# ------------------------------------------------------------------------------
# 14. Concurrent Request Protection
# ------------------------------------------------------------------------------

def test_concurrent_request_protection():
    """Validates that concurrent POST requests serialize safely and reject race duplicates with 409."""
    vid, jid, _ = _create_api_test_job(cv_status=JobStatus.COMPLETED)
    num_threads = 5
    barrier = threading.Barrier(num_threads)
    responses = []

    def _worker():
        barrier.wait(timeout=5.0)
        resp = client.post(f"/api/v1/jobs/{jid}/interpretations")
        responses.append(resp.status_code)

    try:
        threads = [threading.Thread(target=_worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # Exactly one request must succeed with 202 Accepted; the others must be rejected with 409 Conflict
        assert responses.count(202) == 1
        assert responses.count(409) == num_threads - 1
    finally:
        _delete_api_test_job(vid, jid)


# ------------------------------------------------------------------------------
# 15. Non-Exposure of Sensitive Internal Fields
# ------------------------------------------------------------------------------

def test_no_exposure_of_sensitive_internal_fields():
    """Validates that public schemas never expose base64 payloads, filesystem paths, or tracebacks."""
    vid, jid, iid = _create_api_test_job(
        cv_status=JobStatus.COMPLETED,
        interpretation_status=SceneInterpretationStatus.COMPLETED,
        summary="Public summary without filesystem paths.",
    )
    try:
        response = client.get(f"/api/v1/interpretations/{iid}")
        assert response.status_code == 200
        raw_text = response.text

        # Ensure no raw base64 or internal filesystem paths leaked
        assert "image_base64" not in raw_text
        assert "data:image" not in raw_text
        assert "storage/uploads" not in raw_text
        assert "traceback" not in raw_text.lower()
        assert "gemini_api_key" not in raw_text.lower()
    finally:
        _delete_api_test_job(vid, jid, iid)
