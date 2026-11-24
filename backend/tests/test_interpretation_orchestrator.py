"""
Comprehensive tests for InterpretationOrchestratorService in TrafficLens-AI.

Verifies:
- Successful end-to-end orchestration using MockVLMProvider
- Evidence package construction and provider invocation
- Pydantic validation failure handling
- Grounding verification failure handling
- Provider timeout and connection exception handling
- Missing CV telemetry / results handling
- Incomplete / invalid CV job state handling
- Missing video source file handling
- Terminal state protection (no mutation of COMPLETED/FAILED records)
- CV job isolation (completed CV jobs are NEVER failed by downstream VLM errors)
- InterpretationWorker integration with default orchestrator handler
"""

from __future__ import annotations
import asyncio
import io
import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pytest
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.db.session import SessionLocal
from backend.app.models.job import ProcessingJob
from backend.app.models.scene_interpretation import (
    SceneInterpretation,
    SceneInterpretationStatus,
)
from backend.app.models.video import VideoAsset
from backend.app.repositories.job_repository import SQLAlchemyJobRepository
from backend.app.repositories.scene_interpretation_repository import (
    SQLAlchemySceneInterpretationRepository,
)
from backend.app.schemas.job import JobStatus
from backend.app.schemas.vlm import (
    EvidencePackageSchema,
    SceneInterpretationResponseSchema,
)
from backend.app.services.vlm.mock_provider import MockVLMProvider
from backend.app.services.vlm.orchestrator import InterpretationOrchestratorService
from backend.app.services.vlm.provider import (
    VLMConnectionError,
    VLMTimeoutError,
    VLMError,
)
from backend.app.services.vlm.scene_interpretation_service import (
    SceneInterpretationService,
)
from backend.app.services.vlm_worker import InterpretationWorker


# ------------------------------------------------------------------------------
# Synthetic Video & Telemetry Fixtures
# ------------------------------------------------------------------------------

def _create_synthetic_video_file(
    num_frames: int = 15,
    width: int = 160,
    height: int = 120,
    fps: float = 25.0,
) -> Path:
    """Generates a small valid MP4 video file on disk for OpenCV decoding."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
    for f in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw moving rectangle
        x = int(10 + f * 5)
        y = int(20 + f * 3)
        cv2.rectangle(frame, (x, y), (x + 30, y + 20), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return tmp_path


def _get_sample_cv_results() -> Dict[str, Any]:
    """Provides complete realistic CV pipeline output payload."""
    return {
        "analytics_summary": {
            "sequence_name": "TEST_ORCH_SEQ",
            "total_frames": 15,
            "duration_s": 0.6,
            "fps": 25.0,
            "total_unique_vehicles": 2,
            "line_counts": {"line_1": 2},
            "flow_windows": [{"window_idx": 0, "flow_vpm": 120.0}],
            "stopped_vehicle_events": [
                {
                    "event_id": "stop_1_0.20",
                    "track_id": 1,
                    "start_timestamp_s": 0.20,
                    "timestamp_s": 0.20,
                    "duration_s": 5.5,
                    "roi_id": "roi_1",
                    "details": {
                        "status": "stopped",
                        "duration_s": 5.5,
                        "stationary_frames": 10,
                    },
                }
            ],
            "congestion_distribution_per_roi": {"roi_1": {"free_flow": 1.0}},
            "peak_occupancy_per_roi": {"roi_1": 1.0},
        },
        "detection_summary": {
            "total_detections": 30,
            "mean_confidence": 0.88,
            "class_distribution": {"car": 30},
        },
        "tracking_summary": {
            "total_unique_tracks": 2,
        },
    }


def _setup_test_orchestration_data(
    cv_status: JobStatus = JobStatus.COMPLETED,
    interp_status: SceneInterpretationStatus = SceneInterpretationStatus.PROCESSING,
    include_results: bool = True,
    video_exists: bool = True,
) -> Tuple[str, str, str, Optional[Path]]:
    """
    Seeds database with VideoAsset, ProcessingJob, and SceneInterpretation.
    Returns (video_id, job_id, interpretation_id, video_path).
    """
    uid = uuid.uuid4().hex[:8]
    vid = f"vid_orch_{uid}"
    jid = f"job_orch_{uid}"
    iid = f"interp_orch_{uid}"

    video_path: Optional[Path] = None
    if video_exists:
        video_path = _create_synthetic_video_file(num_frames=15, fps=25.0)
        storage_path = str(video_path)
    else:
        storage_path = f"storage/uploads/missing_{uid}.mp4"

    results = _get_sample_cv_results() if include_results else None

    with SessionLocal() as db:
        video = VideoAsset(
            id=vid,
            filename=f"test_{uid}.mp4",
            storage_path=storage_path,
            file_size_bytes=1024,
            fps=25.0,
            duration_seconds=0.6,
            width=160,
            height=120,
        )
        db.add(video)
        db.flush()

        job = ProcessingJob(
            id=jid,
            video_id=vid,
            sequence_name="SEQ_ORCH_TEST",
            status=cv_status,
            config_metadata={"results": results} if results else {},
        )
        db.add(job)
        db.flush()

        interp = SceneInterpretation(
            id=iid,
            job_id=jid,
            status=interp_status,
            provider_type="mock",
            model_name="mock-qwen3-vl-2b",
            started_at=datetime.now(timezone.utc) if interp_status == SceneInterpretationStatus.PROCESSING else None,
        )
        db.add(interp)
        db.commit()

    return vid, jid, iid, video_path


def _cleanup_test_orchestration_data(
    vid: str,
    jid: str,
    iid: str,
    video_path: Optional[Path] = None,
) -> None:
    """Removes seeded records and synthetic video file."""
    if video_path and video_path.exists():
        try:
            video_path.unlink()
        except Exception:
            pass

    with SessionLocal() as db:
        interp = db.get(SceneInterpretation, iid)
        if interp:
            db.delete(interp)
        job = db.get(ProcessingJob, jid)
        if job:
            db.delete(job)
        video = db.get(VideoAsset, vid)
        if video:
            db.delete(video)
        db.commit()


# ------------------------------------------------------------------------------
# 1. Successful End-to-End Orchestration
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_successful_end_to_end():
    """Validates complete execution flow: keyframes -> evidence -> mock provider -> grounding -> DB."""
    vid, jid, iid, video_path = _setup_test_orchestration_data()
    try:
        mock_provider = MockVLMProvider()
        with SessionLocal() as db:
            orchestrator = InterpretationOrchestratorService(
                db=db,
                provider=mock_provider,
            )
            completed_interp = await orchestrator.execute_interpretation(iid)

            assert completed_interp is not None
            assert completed_interp.status == SceneInterpretationStatus.COMPLETED
            assert completed_interp.completed_at is not None
            assert completed_interp.duration_seconds is not None
            assert completed_interp.summary is not None
            assert len(completed_interp.summary) > 0

            # Verify structured component fields
            assert completed_interp.environmental_context is not None
            assert "weather" in completed_interp.environmental_context
            assert "lighting" in completed_interp.environmental_context
            assert "pavement" in completed_interp.environmental_context

            assert completed_interp.scene_description is not None
            assert "roadway_topology" in completed_interp.scene_description

            assert completed_interp.event_interpretations is not None
            assert len(completed_interp.event_interpretations) == 1
            assert completed_interp.event_interpretations[0]["event_id"] == "stop_1_0.20"

            assert completed_interp.epistemic_claims is not None
            assert len(completed_interp.epistemic_claims) > 0

            # Evidence package id recorded
            assert completed_interp.evidence_package_id is not None
            assert completed_interp.evidence_package_id.startswith(f"evpkg_{jid}")

            # Token accounting
            assert completed_interp.prompt_tokens is not None
            assert completed_interp.completion_tokens is not None
            assert completed_interp.total_tokens == completed_interp.prompt_tokens + completed_interp.completion_tokens

        # Verify DB persistence directly in fresh session
        with SessionLocal() as verify_db:
            repo = SQLAlchemySceneInterpretationRepository(verify_db)
            saved = repo.get_by_id(iid)
            assert saved is not None
            assert saved.status == SceneInterpretationStatus.COMPLETED

            # Primary CV job is untouched and remains COMPLETED
            job_repo = SQLAlchemyJobRepository(verify_db)
            job = job_repo.get_job_model(jid)
            assert job is not None
            assert job.status == JobStatus.COMPLETED
    finally:
        _cleanup_test_orchestration_data(vid, jid, iid, video_path)


# ------------------------------------------------------------------------------
# 2. Evidence Package Construction & Provider Invocation
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_evidence_package_construction():
    """Validates that extracted keyframes, stopped events, and telemetry are assembled correctly."""
    vid, jid, iid, video_path = _setup_test_orchestration_data()
    received_packages: List[EvidencePackageSchema] = []

    class CapturingMockProvider(MockVLMProvider):
        async def generate_scene_understanding(self, evidence: EvidencePackageSchema) -> SceneInterpretationResponseSchema:
            received_packages.append(evidence)
            return await super().generate_scene_understanding(evidence)

    try:
        capturing_provider = CapturingMockProvider()
        with SessionLocal() as db:
            orchestrator = InterpretationOrchestratorService(
                db=db,
                provider=capturing_provider,
            )
            result = await orchestrator.execute_interpretation(iid)
            assert result is not None
            assert result.status == SceneInterpretationStatus.COMPLETED

        assert len(received_packages) == 1
        pkg = received_packages[0]
        assert pkg.job_id == jid
        assert pkg.video_id == vid
        assert len(pkg.keyframes) > 0
        assert len(pkg.keyframes) <= 5  # Budget enforcement (K <= 5)
        assert len(pkg.events) == 1
        assert pkg.events[0].event_id == "stop_1_0.20"
        assert pkg.events[0].track_id == 1
        assert pkg.telemetry_summary["total_unique_tracks"] == 2
    finally:
        _cleanup_test_orchestration_data(vid, jid, iid, video_path)


# ------------------------------------------------------------------------------
# 3. Provider Timeout & Network Exception Handling
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_provider_timeout():
    """Validates that VLM timeouts cleanly transition interpretation to FAILED without throwing."""
    vid, jid, iid, video_path = _setup_test_orchestration_data()
    try:
        timeout_provider = MockVLMProvider(
            forced_exception=VLMTimeoutError("Remote inference timed out after 30.0s")
        )
        with SessionLocal() as db:
            orchestrator = InterpretationOrchestratorService(
                db=db,
                provider=timeout_provider,
            )
            failed_interp = await orchestrator.execute_interpretation(iid)

            assert failed_interp is not None
            assert failed_interp.status == SceneInterpretationStatus.FAILED
            assert failed_interp.error_code == "VLM_PROVIDER_TIMEOUT"
            assert "Remote inference timed out after 30.0s" in failed_interp.error_message
            assert failed_interp.completed_at is not None

        # Verify parent CV job remains COMPLETED
        with SessionLocal() as db:
            job = db.get(ProcessingJob, jid)
            assert job.status == JobStatus.COMPLETED
    finally:
        _cleanup_test_orchestration_data(vid, jid, iid, video_path)


@pytest.mark.asyncio
async def test_orchestrator_provider_connection_error():
    """Validates that network connection errors cleanly transition interpretation to FAILED."""
    vid, jid, iid, video_path = _setup_test_orchestration_data()
    try:
        conn_provider = MockVLMProvider(
            forced_exception=VLMConnectionError("Connection refused by endpoint")
        )
        with SessionLocal() as db:
            orchestrator = InterpretationOrchestratorService(
                db=db,
                provider=conn_provider,
            )
            failed_interp = await orchestrator.execute_interpretation(iid)

            assert failed_interp is not None
            assert failed_interp.status == SceneInterpretationStatus.FAILED
            assert failed_interp.error_code == "VLM_PROVIDER_CONNECTION_ERROR"
            assert "Connection refused" in failed_interp.error_message
    finally:
        _cleanup_test_orchestration_data(vid, jid, iid, video_path)


# ------------------------------------------------------------------------------
# 4. Grounding Verification Failure Handling
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_grounding_verification_internal_error(monkeypatch):
    """Validates that unexpected verifier crashes are caught and marked as GROUNDING_VERIFICATION_FAILED."""
    vid, jid, iid, video_path = _setup_test_orchestration_data()

    def crashing_verifier(*args, **kwargs):
        raise RuntimeError("Grounding verifier internal assertion failure")

    monkeypatch.setattr(
        "backend.app.services.vlm.orchestrator.verify_scene_interpretation_grounding",
        crashing_verifier,
    )

    try:
        with SessionLocal() as db:
            orchestrator = InterpretationOrchestratorService(
                db=db,
                provider=MockVLMProvider(),
            )
            failed_interp = await orchestrator.execute_interpretation(iid)

            assert failed_interp is not None
            assert failed_interp.status == SceneInterpretationStatus.FAILED
            assert failed_interp.error_code == "GROUNDING_VERIFICATION_FAILED"
            assert "Grounding verifier internal assertion failure" in failed_interp.error_message
    finally:
        _cleanup_test_orchestration_data(vid, jid, iid, video_path)


# ------------------------------------------------------------------------------
# 5. Missing Evidence & Telemetry Results Handling
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_missing_cv_results():
    """Validates that completed jobs missing CV results are marked FAILED (CV_RESULTS_MISSING)."""
    vid, jid, iid, video_path = _setup_test_orchestration_data(include_results=False)
    try:
        with SessionLocal() as db:
            orchestrator = InterpretationOrchestratorService(
                db=db,
                provider=MockVLMProvider(),
            )
            failed_interp = await orchestrator.execute_interpretation(iid)

            assert failed_interp is not None
            assert failed_interp.status == SceneInterpretationStatus.FAILED
            assert failed_interp.error_code == "CV_RESULTS_MISSING"
            assert "No CV telemetry results available" in failed_interp.error_message
    finally:
        _cleanup_test_orchestration_data(vid, jid, iid, video_path)


# ------------------------------------------------------------------------------
# 6. Incomplete / Invalid CV Job State Handling
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_rejects_non_completed_cv_job():
    """Validates that interpretations are rejected if the parent CV job is not COMPLETED."""
    vid, jid, iid, video_path = _setup_test_orchestration_data(cv_status=JobStatus.PROCESSING)
    try:
        with SessionLocal() as db:
            orchestrator = InterpretationOrchestratorService(
                db=db,
                provider=MockVLMProvider(),
            )
            failed_interp = await orchestrator.execute_interpretation(iid)

            assert failed_interp is not None
            assert failed_interp.status == SceneInterpretationStatus.FAILED
            assert failed_interp.error_code == "CV_JOB_NOT_COMPLETED"
            assert "can only run on COMPLETED CV jobs" in failed_interp.error_message
    finally:
        _cleanup_test_orchestration_data(vid, jid, iid, video_path)


# ------------------------------------------------------------------------------
# 7. Missing Video Source File Handling
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_missing_video_source_file():
    """Validates that missing video files on disk fail with VIDEO_NOT_FOUND."""
    vid, jid, iid, video_path = _setup_test_orchestration_data(video_exists=False)
    try:
        with SessionLocal() as db:
            orchestrator = InterpretationOrchestratorService(
                db=db,
                provider=MockVLMProvider(),
            )
            failed_interp = await orchestrator.execute_interpretation(iid)

            assert failed_interp is not None
            assert failed_interp.status == SceneInterpretationStatus.FAILED
            assert failed_interp.error_code == "VIDEO_NOT_FOUND"
            assert "not found on disk" in failed_interp.error_message
    finally:
        _cleanup_test_orchestration_data(vid, jid, iid, video_path)


# ------------------------------------------------------------------------------
# 8. Terminal State Protection (Preservation of COMPLETED / FAILED)
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_preserves_terminal_state():
    """Validates that calling execute_interpretation on an already terminal record does nothing."""
    # 1. Already completed record
    vid1, jid1, iid1, video_path1 = _setup_test_orchestration_data(
        interp_status=SceneInterpretationStatus.COMPLETED
    )
    # 2. Already failed record
    vid2, jid2, iid2, video_path2 = _setup_test_orchestration_data(
        interp_status=SceneInterpretationStatus.FAILED
    )
    try:
        with SessionLocal() as db:
            orchestrator = InterpretationOrchestratorService(
                db=db,
                provider=MockVLMProvider(),
            )
            res1 = await orchestrator.execute_interpretation(iid1)
            res2 = await orchestrator.execute_interpretation(iid2)

            assert res1.status == SceneInterpretationStatus.COMPLETED
            assert res2.status == SceneInterpretationStatus.FAILED

        with SessionLocal() as verify_db:
            repo = SQLAlchemySceneInterpretationRepository(verify_db)
            rec1 = repo.get_by_id(iid1)
            rec2 = repo.get_by_id(iid2)
            assert rec1.status == SceneInterpretationStatus.COMPLETED
            assert rec2.status == SceneInterpretationStatus.FAILED
    finally:
        _cleanup_test_orchestration_data(vid1, jid1, iid1, video_path1)
        _cleanup_test_orchestration_data(vid2, jid2, iid2, video_path2)


# ------------------------------------------------------------------------------
# 9. Worker-to-Orchestrator Handler Integration
# ------------------------------------------------------------------------------

def test_worker_to_orchestrator_default_integration():
    """Validates that InterpretationWorker executes InterpretationOrchestratorService by default."""
    vid, jid, iid, video_path = _setup_test_orchestration_data(
        interp_status=SceneInterpretationStatus.QUEUED
    )
    try:
        # Create worker without custom handler (defaults to InterpretationOrchestratorService)
        # Configure MockVLMProvider via settings or environment
        custom_settings = Settings(VLM_ENABLED=True, VLM_PROVIDER_TYPE="mock")
        worker = InterpretationWorker(settings=custom_settings)

        # Process the single queued task
        worker._process_single_interpretation(iid)

        # Verify that interpretation completed successfully via default orchestrator pipeline
        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)
            interp = repo.get_by_id(iid)
            assert interp is not None
            assert interp.status == SceneInterpretationStatus.COMPLETED
            assert interp.summary is not None
            assert interp.evidence_package_id is not None
    finally:
        _cleanup_test_orchestration_data(vid, jid, iid, video_path)


# ------------------------------------------------------------------------------
# 10. Cooperative Cancellation via Stop Event
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_cooperative_stop_event():
    """Validates that setting stop_event raises InterruptedError and transitions task to FAILED."""
    vid, jid, iid, video_path = _setup_test_orchestration_data()
    stop_event = asyncio.Event() if False else None
    import threading
    t_stop_event = threading.Event()
    t_stop_event.set()  # Stop event pre-set

    try:
        with SessionLocal() as db:
            orchestrator = InterpretationOrchestratorService(
                db=db,
                provider=MockVLMProvider(),
            )
            with pytest.raises(InterruptedError) as exc_info:
                await orchestrator.execute_interpretation(iid, stop_event=t_stop_event)

            assert "received cooperative shutdown stop signal" in str(exc_info.value)
    finally:
        _cleanup_test_orchestration_data(vid, jid, iid, video_path)


# ------------------------------------------------------------------------------
# 11. Schema / Output Validation Failure Handling
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_pydantic_output_parsing_failure():
    """Validates that provider output parsing/validation failures are recorded cleanly."""
    from backend.app.services.vlm.provider import VLMOutputParsingError

    vid, jid, iid, video_path = _setup_test_orchestration_data()
    try:
        parsing_error_provider = MockVLMProvider(
            forced_exception=VLMOutputParsingError("Model output failed Pydantic validation: missing required field 'weather'")
        )
        with SessionLocal() as db:
            orchestrator = InterpretationOrchestratorService(
                db=db,
                provider=parsing_error_provider,
            )
            failed_interp = await orchestrator.execute_interpretation(iid)

            assert failed_interp is not None
            assert failed_interp.status == SceneInterpretationStatus.FAILED
            assert failed_interp.error_code == "VLM_OUTPUT_PARSING_ERROR"
            assert "failed Pydantic validation" in failed_interp.error_message
    finally:
        _cleanup_test_orchestration_data(vid, jid, iid, video_path)


# ------------------------------------------------------------------------------
# 12. Database Persistence Failure Handling
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_persistence_failure_rollback(monkeypatch):
    """Validates that database persistence failures cleanly roll back and attempt failure recording."""
    vid, jid, iid, video_path = _setup_test_orchestration_data()

    try:
        with SessionLocal() as db:
            orchestrator = InterpretationOrchestratorService(
                db=db,
                provider=MockVLMProvider(),
            )

            def failing_complete(*args, **kwargs):
                raise RuntimeError("Simulated low-level disk I/O database write failure")

            monkeypatch.setattr(orchestrator.repo, "complete", failing_complete)

            failed_interp = await orchestrator.execute_interpretation(iid)
            assert failed_interp is not None
            assert failed_interp.status == SceneInterpretationStatus.FAILED
            assert failed_interp.error_code == "DATABASE_PERSISTENCE_FAILED"
            assert "Failed to persist completed interpretation results" in failed_interp.error_message
    finally:
        _cleanup_test_orchestration_data(vid, jid, iid, video_path)

