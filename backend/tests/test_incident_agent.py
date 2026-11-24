"""
Deterministic unit and integration tests for Evidence-Grounded Traffic Incident Agent.

Covers:
1. Successful report generation with CV and VLM evidence
2. Report generation using CV evidence alone
3. Missing processing job rejection
4. Missing CV telemetry handling
5. Missing or failed VLM interpretation graceful degradation
6. Unsupported factual claims detection and neutralization
7. Missing or invalid evidence references validation
8. LLM provider failures handling
9. Invalid structured output handling
10. Report validation and serialization
11. Persistence and historical audit retention (no overwrite)
12. FastAPI HTTP endpoints for incident reports
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.app.db.session import SessionLocal, init_db
from backend.app.main import app
from backend.app.models.job import ProcessingJob
from backend.app.models.incident_report import IncidentReport
from backend.app.models.scene_interpretation import (
    SceneInterpretation,
    SceneInterpretationStatus,
)
from backend.app.models.video import VideoAsset
from backend.app.repositories.incident_report_repository import (
    SQLAlchemyIncidentReportRepository,
)
from backend.app.schemas.incident_report import (
    IncidentReportResponseSchema,
    IncidentReportSchema,
    ReportConfidenceEnum,
    TrafficConditionsSchema,
)
from backend.app.schemas.job import JobStatus
from backend.app.services.agent.evidence_normalizer import (
    EvidenceNormalizer,
    NormalizedEvidenceContext,
)
from backend.app.services.agent.grounding_validator import GroundingValidator
from backend.app.services.agent.graph import create_incident_agent_graph
from backend.app.services.agent.provider import (
    MockIncidentReportProvider,
)
from backend.app.services.agent.service import IncidentReportService

client = TestClient(app)

# Ensure all database tables exist
init_db()


# ------------------------------------------------------------------------------
# Test Fixtures & Setup Helpers
# ------------------------------------------------------------------------------

def _create_sample_cv_results() -> Dict[str, Any]:
    """Generates standard CV telemetry results dictionary for testing."""
    return {
        "analytics_summary": {
            "sequence_name": "MVI_TEST_SEQ",
            "duration_s": 24.5,
            "total_frames": 612,
            "fps": 25.0,
            "total_unique_vehicles": 8,
            "line_counts": {
                "line_main": {"downstream": 6, "upstream": 2}
            },
            "peak_occupancy_per_roi": {"roi_main": 0.42},
            "average_occupancy_per_roi": {"roi_main": 0.28},
            "congestion_distribution_per_roi": {
                "roi_main": {"low": 300, "moderate": 250, "high": 62}
            },
            "stopped_vehicle_events": [
                {
                    "event_id": "stop_4_3.20",
                    "track_id": 4,
                    "start_timestamp_s": 3.20,
                    "duration_s": 7.5,
                    "location": [320.0, 480.0],
                    "roi_id": "roi_main",
                    "status": "resolved",
                }
            ],
        },
        "detection_summary": {
            "total_detections": 128,
            "mean_confidence": 0.88,
            "class_distribution": {"car": 110, "bus": 18},
        },
        "tracking_summary": {
            "total_unique_tracks": 8,
            "mostly_tracked_count": 6,
            "mostly_lost_count": 2,
            "id_switches": 1,
        },
    }


def _seed_db_job(
    cv_status: JobStatus = JobStatus.COMPLETED,
    include_cv_results: bool = True,
    vlm_status: Optional[SceneInterpretationStatus] = None,
) -> Tuple[str, str, Optional[str]]:
    """Seeds test records in DB for VideoAsset, ProcessingJob, and SceneInterpretation."""
    uid = uuid.uuid4().hex[:8]
    vid = f"vid_agent_{uid}"
    jid = f"job_agent_{uid}"
    iid = f"interp_agent_{uid}" if vlm_status else None

    with SessionLocal() as db:
        video = VideoAsset(
            id=vid,
            filename=f"agent_test_{uid}.mp4",
            storage_path=f"storage/uploads/agent_test_{uid}.mp4",
            file_size_bytes=2048,
            fps=25.0,
            duration_seconds=24.5,
        )
        db.add(video)
        db.flush()

        results_payload = _create_sample_cv_results() if include_cv_results else {}
        job = ProcessingJob(
            id=jid,
            video_id=vid,
            sequence_name="SEQ_AGENT_TEST",
            status=cv_status,
            config_metadata={"results": results_payload} if include_cv_results else {},
        )
        db.add(job)
        db.flush()

        if vlm_status:
            now = datetime.now(timezone.utc)
            interp = SceneInterpretation(
                id=iid,
                job_id=jid,
                status=vlm_status,
                provider_type="mock",
                model_name="mock-vlm-2b",
                summary="Clear daytime traffic on urban divided corridor.",
                started_at=now,
                completed_at=now if vlm_status == SceneInterpretationStatus.COMPLETED else None,
                duration_seconds=1.8,
            )
            if vlm_status == SceneInterpretationStatus.COMPLETED:
                interp.environmental_context = {
                    "weather": "clear",
                    "lighting": "daylight",
                    "pavement": "dry",
                    "visibility_hazards": [],
                    "confidence": "direct_observation",
                }
                interp.scene_description = {
                    "roadway_topology": "divided_highway",
                    "features": ["dry asphalt pavement", "paved shoulder"],
                }
                interp.event_interpretations = [
                    {
                        "event_id": "stop_4_3.20",
                        "track_id": 4,
                        "probable_cause": "Queue congestion build-up",
                        "interpretation": "Vehicle remained stationary due to temporary traffic queue.",
                    }
                ]
            elif vlm_status == SceneInterpretationStatus.FAILED:
                interp.error_code = "VLM_PROVIDER_TIMEOUT"
                interp.error_message = "VLM provider request timed out."

            db.add(interp)

        db.commit()

    return vid, jid, iid


def _cleanup_db_job(vid: str, jid: str, iid: Optional[str] = None) -> None:
    """Cleans up seeded DB records."""
    with SessionLocal() as db:
        try:
            # Delete incident reports associated with this job
            db.execute(text(f"DELETE FROM incident_reports WHERE job_id = '{jid}'"))
            if iid:
                db.execute(text(f"DELETE FROM scene_interpretations WHERE id = '{iid}'"))
            db.execute(text(f"DELETE FROM processing_jobs WHERE id = '{jid}'"))
            db.execute(text(f"DELETE FROM video_assets WHERE id = '{vid}'"))
            db.commit()
        except Exception:
            db.rollback()


# ------------------------------------------------------------------------------
# Test 1: Successful Report Generation with CV and VLM Evidence
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_successful_report_generation_with_cv_and_vlm():
    """Validates incident report generation with both CV telemetry and completed VLM scene interpretation."""
    vid, jid, iid = _seed_db_job(
        cv_status=JobStatus.COMPLETED,
        include_cv_results=True,
        vlm_status=SceneInterpretationStatus.COMPLETED,
    )
    try:
        with SessionLocal() as db:
            service = IncidentReportService(
                db=db,
                provider=MockIncidentReportProvider(),
            )
            resp = await service.generate_incident_report(jid)

            assert resp.status == "COMPLETED"
            assert resp.validation_passed is True
            assert resp.report is not None
            assert resp.report.job_id == jid
            assert resp.report.traffic_conditions.visual_context_available is True
            assert resp.report.traffic_conditions.total_tracked_instances == 8
            assert resp.report.traffic_conditions.total_detections == 128
            assert len(resp.report.stopped_vehicles) == 1
            assert resp.report.stopped_vehicles[0].track_id == 4
            assert resp.report.stopped_vehicles[0].duration_s == 7.5
            assert resp.report.confidence_classification == ReportConfidenceEnum.HIGH
            assert len(resp.report.visual_observations) > 0
            assert resp.report.validation_diagnostics is not None
            assert resp.report.validation_diagnostics.validation_passed is True
    finally:
        _cleanup_db_job(vid, jid, iid)


# ------------------------------------------------------------------------------
# Test 2: Report Generation Using CV Evidence Alone
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_report_generation_cv_evidence_alone():
    """Validates report generation when no VLM interpretation was requested."""
    vid, jid, _ = _seed_db_job(
        cv_status=JobStatus.COMPLETED,
        include_cv_results=True,
        vlm_status=None,
    )
    try:
        with SessionLocal() as db:
            service = IncidentReportService(
                db=db,
                provider=MockIncidentReportProvider(),
            )
            resp = await service.generate_incident_report(jid)

            assert resp.status == "COMPLETED"
            assert resp.validation_passed is True
            assert resp.report is not None
            assert resp.report.traffic_conditions.visual_context_available is False
            assert len(resp.report.visual_observations) == 0
            assert any(
                "Visual scene interpretation was unavailable" in lim
                for lim in resp.report.limitations
            )
            assert resp.report.confidence_classification in (
                ReportConfidenceEnum.MEDIUM,
                ReportConfidenceEnum.LOW,
            )
    finally:
        _cleanup_db_job(vid, jid)


# ------------------------------------------------------------------------------
# Test 3: Missing Processing Job Rejection
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_processing_job_rejection():
    """Validates KeyError rejection when job ID does not exist."""
    with SessionLocal() as db:
        service = IncidentReportService(db=db, provider=MockIncidentReportProvider())
        with pytest.raises(KeyError, match="ProcessingJob 'job_nonexistent' not found"):
            await service.generate_incident_report("job_nonexistent")


# ------------------------------------------------------------------------------
# Test 4: Missing CV Telemetry Handling
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_cv_telemetry_handling():
    """Validates clean failure when completed job has no CV results."""
    vid, jid, _ = _seed_db_job(
        cv_status=JobStatus.COMPLETED,
        include_cv_results=False,
    )
    try:
        with SessionLocal() as db:
            service = IncidentReportService(db=db, provider=MockIncidentReportProvider())
            resp = await service.generate_incident_report(jid)

            assert resp.status == "FAILED"
            assert resp.validation_passed is False
            assert "No CV telemetry results available" in (resp.error_message or "")
    finally:
        _cleanup_db_job(vid, jid)


# ------------------------------------------------------------------------------
# Test 5: Missing or Failed VLM Interpretation Graceful Degradation
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failed_vlm_interpretation_graceful_degradation():
    """Validates that a FAILED VLM record does not block report generation from CV telemetry."""
    vid, jid, iid = _seed_db_job(
        cv_status=JobStatus.COMPLETED,
        include_cv_results=True,
        vlm_status=SceneInterpretationStatus.FAILED,
    )
    try:
        with SessionLocal() as db:
            service = IncidentReportService(db=db, provider=MockIncidentReportProvider())
            resp = await service.generate_incident_report(jid)

            assert resp.status == "COMPLETED"
            assert resp.validation_passed is True
            assert resp.report is not None
            assert resp.report.traffic_conditions.visual_context_available is False
            assert len(resp.report.visual_observations) == 0
    finally:
        _cleanup_db_job(vid, jid, iid)


# ------------------------------------------------------------------------------
# Test 6: Unsupported Factual Claims Detection & Neutralization
# ------------------------------------------------------------------------------

def test_unsupported_factual_claims_neutralization():
    """Validates that GroundingValidator strips hallucinated collisions and uncalibrated speeds."""
    norm = EvidenceNormalizer.normalize(
        job_id="job_test",
        video_id="vid_test",
        sequence_name="SEQ_TEST",
        cv_results=_create_sample_cv_results(),
    )

    provider = MockIncidentReportProvider(inject_unsupported_claim=True)
    import asyncio
    raw_report = asyncio.run(provider.generate_report(norm))

    assert "collision" in raw_report.executive_summary
    assert "75 km/h" in raw_report.executive_summary

    validated_report, diagnostics = GroundingValidator.validate_and_correct(raw_report, norm)

    # Collision must be sanitized/demoted
    assert "collision" not in validated_report.executive_summary
    assert "[unverified claim]" in validated_report.executive_summary
    # 75 km/h must be suppressed
    assert "75 km/h" not in validated_report.executive_summary
    assert "[uncalibrated displacement]" in validated_report.executive_summary

    assert len(diagnostics.warnings) >= 2
    assert any("Detected ungrounded incident assertions" in w for w in diagnostics.warnings)
    assert any("Prohibited uncalibrated speed assertions" in w for w in diagnostics.warnings)


# ------------------------------------------------------------------------------
# Test 7: Missing Evidence References Validation
# ------------------------------------------------------------------------------

def test_missing_evidence_references_handling():
    """Validates that references to unavailable VLM context are marked unverified."""
    norm = EvidenceNormalizer.normalize(
        job_id="job_test",
        video_id="vid_test",
        sequence_name="SEQ_TEST",
        cv_results=_create_sample_cv_results(),
        scene_interpretation=None,  # No VLM
    )

    provider = MockIncidentReportProvider(inject_visual_claim_without_vlm=True)
    import asyncio
    raw_report = asyncio.run(provider.generate_report(norm))

    # Add invalid VLM reference
    from backend.app.schemas.incident_report import EvidenceReferenceSchema
    raw_report.evidence_references.append(
        EvidenceReferenceSchema(
            source="vlm_scene_interpretation",
            metric_or_field="environmental_context.weather",
            asserted_value="rain",
            verified=True,
        )
    )

    validated_report, diagnostics = GroundingValidator.validate_and_correct(raw_report, norm)

    # The invalid reference must be marked verified=False
    vlm_ref = next(r for r in validated_report.evidence_references if r.source == "vlm_scene_interpretation")
    assert vlm_ref.verified is False
    assert len(validated_report.visual_observations) == 0
    assert any("Removed 1 visual observations" in w for w in diagnostics.warnings)


# ------------------------------------------------------------------------------
# Test 8: LLM Provider Failures Handling
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_provider_failures_handling():
    """Validates that service handles provider exceptions gracefully and records FAILED status."""
    vid, jid, _ = _seed_db_job(cv_status=JobStatus.COMPLETED, include_cv_results=True)
    try:
        with SessionLocal() as db:
            service = IncidentReportService(
                db=db,
                provider=MockIncidentReportProvider(should_fail=True),
            )
            resp = await service.generate_incident_report(jid)

            assert resp.status == "FAILED"
            assert resp.validation_passed is False
            assert "Simulated mock LLM provider failure" in (resp.error_message or "")
    finally:
        _cleanup_db_job(vid, jid)


# ------------------------------------------------------------------------------
# Test 9: Invalid Structured Output Handling
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_structured_output_handling():
    """Validates handling when provider returns invalid schema structure."""
    vid, jid, _ = _seed_db_job(cv_status=JobStatus.COMPLETED, include_cv_results=True)
    try:
        with SessionLocal() as db:
            service = IncidentReportService(
                db=db,
                provider=MockIncidentReportProvider(invalid_json=True),
            )
            resp = await service.generate_incident_report(jid)

            assert resp.status == "FAILED"
            assert resp.validation_passed is False
            assert "Invalid mock output schema" in (resp.error_message or "")
    finally:
        _cleanup_db_job(vid, jid)


# ------------------------------------------------------------------------------
# Test 10: Report Validation and Serialization
# ------------------------------------------------------------------------------

def test_report_validation_and_serialization():
    """Validates that IncidentReportSchema serializes to and deserializes from JSON cleanly."""
    norm = EvidenceNormalizer.normalize(
        job_id="job_ser_test",
        video_id="vid_ser_test",
        sequence_name="SEQ_SER",
        cv_results=_create_sample_cv_results(),
    )
    provider = MockIncidentReportProvider()
    import asyncio
    rep = asyncio.run(provider.generate_report(norm))

    # Serialize to JSON dict
    rep_dict = rep.model_dump(mode="json")
    assert isinstance(rep_dict, dict)
    assert rep_dict["job_id"] == "job_ser_test"

    # Re-deserialize
    re_parsed = IncidentReportSchema(**rep_dict)
    assert re_parsed.report_id == rep.report_id
    assert re_parsed.traffic_conditions.total_tracked_instances == 8


# ------------------------------------------------------------------------------
# Test 11: Persistence and Historical Audit Retention (No Overwrite)
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_persistence_and_audit_history_retention():
    """Validates that multiple reports for the same job generate distinct records and are not overwritten."""
    vid, jid, _ = _seed_db_job(cv_status=JobStatus.COMPLETED, include_cv_results=True)
    try:
        with SessionLocal() as db:
            service = IncidentReportService(db=db, provider=MockIncidentReportProvider())
            # Run report generation 1
            resp1 = await service.generate_incident_report(jid)
            # Run report generation 2
            resp2 = await service.generate_incident_report(jid)

            assert resp1.report_id != resp2.report_id

            # Verify both exist in history
            history = service.list_reports_for_job(jid)
            assert history.total_reports >= 2
            report_ids = [r.report_id for r in history.reports]
            assert resp1.report_id in report_ids
            assert resp2.report_id in report_ids

            # Verify latest
            latest = service.get_latest_report_for_job(jid)
            assert latest is not None
            assert latest.report_id == resp2.report_id
    finally:
        _cleanup_db_job(vid, jid)


# ------------------------------------------------------------------------------
# Test 12: FastAPI HTTP Endpoints for Incident Reports
# ------------------------------------------------------------------------------

def test_api_incident_report_endpoints():
    """Validates REST endpoints for generating, fetching, and listing incident reports."""
    from backend.app.dependencies import get_app_settings
    from backend.app.core.config import Settings
    app.dependency_overrides[get_app_settings] = lambda: Settings(_env_file=None, GEMINI_API_KEY=None)
    vid, jid, _ = _seed_db_job(cv_status=JobStatus.COMPLETED, include_cv_results=True)
    try:
        # 1. Generate report via POST
        res_post = client.post(f"/api/v1/jobs/{jid}/incident-reports")
        assert res_post.status_code == 201
        data_post = res_post.json()
        assert data_post["job_id"] == jid
        assert data_post["status"] == "COMPLETED"
        report_id = data_post["report_id"]

        # 2. Get latest report via GET
        res_latest = client.get(f"/api/v1/jobs/{jid}/incident-report")
        assert res_latest.status_code == 200
        assert res_latest.json()["report_id"] == report_id

        # 3. Get specific report by ID
        res_by_id = client.get(f"/api/v1/incident-reports/{report_id}")
        assert res_by_id.status_code == 200
        assert res_by_id.json()["report_id"] == report_id

        # 4. List report history
        res_hist = client.get(f"/api/v1/jobs/{jid}/incident-reports")
        assert res_hist.status_code == 200
        assert res_hist.json()["total_reports"] >= 1

        # 5. Non-existent job rejection
        res_missing = client.post("/api/v1/jobs/job_missing_123/incident-reports")
        assert res_missing.status_code == 404
    finally:
        app.dependency_overrides.pop(get_app_settings, None)
        _cleanup_db_job(vid, jid)


def test_api_rejection_of_active_and_failed_jobs():
    """Validates that API returns 409 for active jobs and 400 for failed jobs."""
    vid_p, jid_p, _ = _seed_db_job(cv_status=JobStatus.PROCESSING)
    vid_f, jid_f, _ = _seed_db_job(cv_status=JobStatus.FAILED)
    try:
        # Reject active job with 409
        res_p = client.post(f"/api/v1/jobs/{jid_p}/incident-reports")
        assert res_p.status_code == 409

        # Reject failed job with 400
        res_f = client.post(f"/api/v1/jobs/{jid_f}/incident-reports")
        assert res_f.status_code == 400
    finally:
        _cleanup_db_job(vid_p, jid_p)
        _cleanup_db_job(vid_f, jid_f)


# ------------------------------------------------------------------------------
# Test 13: Fabricated Numerical Values Correction
# ------------------------------------------------------------------------------

def test_fabricated_numerical_values_correction():
    """Validates that GroundingValidator detects numerical drift and replaces with source truth."""
    norm = EvidenceNormalizer.normalize(
        job_id="job_drift",
        video_id="vid_drift",
        sequence_name="SEQ_DRIFT",
        cv_results=_create_sample_cv_results(),
    )
    provider = MockIncidentReportProvider()
    import asyncio
    rep = asyncio.run(provider.generate_report(norm))

    # Fabricate telemetry values
    rep.traffic_conditions.total_detections = 9999
    rep.traffic_conditions.total_tracked_instances = 8888
    rep.traffic_conditions.duration_s = 999.0

    corrected, diag = GroundingValidator.validate_and_correct(rep, norm)

    # Values must be restored to source truth
    assert corrected.traffic_conditions.total_detections == norm.total_detections
    assert corrected.traffic_conditions.total_tracked_instances == norm.total_unique_tracks
    assert corrected.traffic_conditions.duration_s == norm.duration_s

    assert diag.demoted_claims >= 3
    assert any("Corrected total_detections" in w for w in diag.warnings)
    assert any("Corrected total_tracked_instances" in w for w in diag.warnings)
    assert any("Corrected duration_s" in w for w in diag.warnings)


# ------------------------------------------------------------------------------
# Test 14: Invalid and Nonexistent Evidence References
# ------------------------------------------------------------------------------

def test_invalid_and_nonexistent_evidence_references():
    """Validates that references to unknown sources, invalid fields, and missing events are rejected."""
    norm = EvidenceNormalizer.normalize(
        job_id="job_refs",
        video_id="vid_refs",
        sequence_name="SEQ_REFS",
        cv_results=_create_sample_cv_results(),
    )
    provider = MockIncidentReportProvider()
    import asyncio
    rep = asyncio.run(provider.generate_report(norm))

    from backend.app.schemas.incident_report import EvidenceReferenceSchema
    rep.evidence_references = [
        EvidenceReferenceSchema(source="satellite_radar", metric_or_field="speed", asserted_value=50, verified=True),
        EvidenceReferenceSchema(source="cv_telemetry", metric_or_field="unknown_metric_foo", asserted_value=1, verified=True),
        EvidenceReferenceSchema(source="cv_event", metric_or_field="stop_999_99.0", asserted_value=1, verified=True),
    ]

    corrected, diag = GroundingValidator.validate_and_correct(rep, norm)

    # All three invalid references must be marked verified=False
    assert all(r.verified is False for r in corrected.evidence_references)
    assert diag.demoted_claims >= 3
    assert any("Invalid evidence reference source" in w for w in diag.warnings)
    assert any("Unknown CV telemetry metric" in w for w in diag.warnings)
    assert any("Referenced CV event 'stop_999_99.0' not found" in w for w in diag.warnings)


# ------------------------------------------------------------------------------
# Test 15: Unsupported Accident Claims in Observed Events
# ------------------------------------------------------------------------------

def test_unsupported_accident_claims_in_observed_events():
    """Validates that fabricated accidents in observed_events are completely stripped."""
    norm = EvidenceNormalizer.normalize(
        job_id="job_obs_evt",
        video_id="vid_obs_evt",
        sequence_name="SEQ_OBS",
        cv_results=_create_sample_cv_results(),
    )
    provider = MockIncidentReportProvider()
    import asyncio
    rep = asyncio.run(provider.generate_report(norm))

    from backend.app.schemas.incident_report import ObservedEventSchema
    rep.observed_events.append(
        ObservedEventSchema(
            event_type="head_on_collision",
            timestamp_s=5.0,
            description="Two vehicles crashed into highway barrier.",
            supporting_evidence_ref="none",
        )
    )

    corrected, diag = GroundingValidator.validate_and_correct(rep, norm)

    # Event must be stripped
    assert not any("collision" in e.event_type.lower() for e in corrected.observed_events)
    assert any("Stripped unsupported event 'head_on_collision'" in w for w in diag.warnings)
    assert any("head_on_collision" in u for u in corrected.unverified_aspects)


# ------------------------------------------------------------------------------
# Test 16: Census Phrasing Correction
# ------------------------------------------------------------------------------

def test_census_phrasing_clarification():
    """Validates that claims of physical vehicle counts are clarified to tracked instances."""
    norm = EvidenceNormalizer.normalize(
        job_id="job_census",
        video_id="vid_census",
        sequence_name="SEQ_CENSUS",
        cv_results=_create_sample_cv_results(),
    )
    provider = MockIncidentReportProvider()
    import asyncio
    rep = asyncio.run(provider.generate_report(norm))

    rep.executive_summary = "A total of 8 unique physical vehicles traversed the intersection."
    corrected, diag = GroundingValidator.validate_and_correct(rep, norm)

    assert "8 unique physical vehicles" not in corrected.executive_summary
    assert "8 tracked vehicle instances" in corrected.executive_summary
    assert any("tracker identity semantics" in w.lower() for w in diag.warnings)


# ------------------------------------------------------------------------------
# Test 17: Provider Timeout Handling and API HTTP 504
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_provider_timeout_handling():
    """Validates that timeout in service records FAILED status and API returns HTTP 504."""
    vid, jid, _ = _seed_db_job(cv_status=JobStatus.COMPLETED, include_cv_results=True)
    try:
        class TimingOutProvider:
            async def generate_report(self, evidence):
                import asyncio
                await asyncio.sleep(0.5)
                raise TimeoutError("LLM call timed out after 30s")

        with SessionLocal() as db:
            from backend.app.core.config import Settings
            cfg = Settings(INCIDENT_AGENT_TIMEOUT=0.2)
            service = IncidentReportService(db=db, settings=cfg, provider=TimingOutProvider())
            resp = await service.generate_incident_report(jid)

            assert resp.status == "FAILED"
            assert resp.validation_passed is False
            assert "timed out" in resp.error_message.lower()

        # Check API HTTP 504 mapping
        from unittest.mock import patch, AsyncMock
        with patch.object(IncidentReportService, "generate_incident_report", new=AsyncMock(return_value=resp)):
            res_api = client.post(f"/api/v1/jobs/{jid}/incident-reports")
            assert res_api.status_code == 504
            assert "timed out" in res_api.json()["detail"].lower()
    finally:
        _cleanup_db_job(vid, jid)


# ------------------------------------------------------------------------------
# Test 18: Persistence Failure Without False Success
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_persistence_failure_without_false_success():
    """Validates that database persistence failure triggers rollback and raises an exception."""
    vid, jid, _ = _seed_db_job(cv_status=JobStatus.COMPLETED, include_cv_results=True)
    try:
        from unittest.mock import MagicMock
        mock_db = MagicMock()
        mock_db.commit.side_effect = RuntimeError("Disk full / DB connection lost")

        service = IncidentReportService(db=mock_db, provider=MockIncidentReportProvider())
        with pytest.raises(RuntimeError, match="Database persistence failure"):
            await service.generate_incident_report(jid)

        mock_db.rollback.assert_called()
    finally:
        _cleanup_db_job(vid, jid)


# ------------------------------------------------------------------------------
# Test 19: Concurrent Duplicate Generation Requests
# ------------------------------------------------------------------------------

def test_concurrent_duplicate_generation_requests():
    """Validates that concurrent generation requests are serialized via mutex."""
    from backend.app.dependencies import get_app_settings
    from backend.app.core.config import Settings
    app.dependency_overrides[get_app_settings] = lambda: Settings(_env_file=None, GEMINI_API_KEY=None)
    vid, jid, _ = _seed_db_job(cv_status=JobStatus.COMPLETED, include_cv_results=True)
    try:
        import concurrent.futures

        def call_endpoint():
            return client.post(f"/api/v1/jobs/{jid}/incident-reports")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut1 = executor.submit(call_endpoint)
            fut2 = executor.submit(call_endpoint)
            res1 = fut1.result()
            res2 = fut2.result()

        assert res1.status_code == 201
        assert res2.status_code == 201
        assert res1.json()["report_id"] != res2.json()["report_id"]
    finally:
        app.dependency_overrides.pop(get_app_settings, None)
        _cleanup_db_job(vid, jid)


# ------------------------------------------------------------------------------
# Test 20: Historical Report Retention After Failed Generation
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_historical_report_retention_after_failed_generation():
    """Validates that an earlier successful report is preserved even after a subsequent failed generation."""
    vid, jid, _ = _seed_db_job(cv_status=JobStatus.COMPLETED, include_cv_results=True)
    try:
        with SessionLocal() as db:
            service = IncidentReportService(db=db, provider=MockIncidentReportProvider())
            # 1. Successful generation
            success_resp = await service.generate_incident_report(jid)
            assert success_resp.status == "COMPLETED"

            # 2. Failed generation
            failed_service = IncidentReportService(
                db=db, provider=MockIncidentReportProvider(should_fail=True)
            )
            failed_resp = await failed_service.generate_incident_report(jid)
            assert failed_resp.status == "FAILED"

            # 3. Retrieve history
            history = service.list_reports_for_job(jid)
            assert history.total_reports >= 2

            # Confirm success report is still in history intact
            saved_success = next(r for r in history.reports if r.report_id == success_resp.report_id)
            assert saved_success.status == "COMPLETED"
            assert saved_success.report is not None
            assert saved_success.report.executive_summary is not None
    finally:
        _cleanup_db_job(vid, jid)


# ------------------------------------------------------------------------------
# Test 21: Timeout Cancellation Prevents COMPLETED Persistence
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_cancellation_prevents_completed_persistence():
    """Validates that a provider timed out by asyncio.wait_for cannot persist a COMPLETED report."""
    vid, jid, _ = _seed_db_job(cv_status=JobStatus.COMPLETED, include_cv_results=True)
    try:
        class SlowProviderThatWouldSucceed:
            async def generate_report(self, evidence):
                import asyncio
                await asyncio.sleep(0.5)
                # If cancellation failed to abort, this would return a valid report
                return await MockIncidentReportProvider().generate_report(evidence)

        with SessionLocal() as db:
            from backend.app.core.config import Settings
            cfg = Settings(INCIDENT_AGENT_TIMEOUT=0.1)
            service = IncidentReportService(
                db=db,
                settings=cfg,
                provider=SlowProviderThatWouldSucceed(),
            )
            resp = await service.generate_incident_report(jid)

            assert resp.status == "FAILED"
            assert resp.report is None
            assert resp.validation_passed is False
            assert "timed out" in (resp.error_message or "").lower()

            # Confirm database record is strictly FAILED with empty report_data
            repo = SQLAlchemyIncidentReportRepository(db)
            saved = repo.get_by_id(resp.report_id)
            assert saved is not None
            assert saved.status == "FAILED"
            assert saved.summary is None
            assert saved.report_data == {}
            assert saved.validation_passed is False

            # Verify no COMPLETED report was ever saved
            history = repo.list_by_job_id(jid)
            assert all(r.status == "FAILED" for r in history)
    finally:
        _cleanup_db_job(vid, jid)


# ------------------------------------------------------------------------------
# Test 22: SQLAlchemy Session Usable After Rollback
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sqlalchemy_session_usable_after_rollback():
    """Validates that session remains fully operational after service rollback."""
    vid, jid, _ = _seed_db_job(cv_status=JobStatus.COMPLETED, include_cv_results=True)
    try:
        with SessionLocal() as db:
            service = IncidentReportService(db=db, provider=MockIncidentReportProvider())
            # Intentionally inject an invalid model that causes commit failure
            from unittest.mock import patch
            with patch.object(db, "commit", side_effect=[RuntimeError("Simulated write failure"), None]):
                with pytest.raises(RuntimeError, match="Database persistence failure"):
                    await service.generate_incident_report(jid)

            # Verify that session is still active and queries execute without PendingRollbackError
            job = db.query(ProcessingJob).filter(ProcessingJob.id == jid).first()
            assert job is not None
            assert job.id == jid
    finally:
        _cleanup_db_job(vid, jid)


# ------------------------------------------------------------------------------
# Test 23: GroundingValidator Strips Fabricated Stoppage Event Citations
# ------------------------------------------------------------------------------

def test_grounding_validator_strips_fabricated_stoppage_event_citations():
    """Validates that observed_events claiming stoppage with fabricated citations are stripped."""
    norm = EvidenceNormalizer.normalize(
        job_id="job_fab_stop",
        video_id="vid_fab_stop",
        sequence_name="SEQ_FAB_STOP",
        cv_results=_create_sample_cv_results(),
    )
    provider = MockIncidentReportProvider()
    import asyncio
    rep = asyncio.run(provider.generate_report(norm))

    from backend.app.schemas.incident_report import ObservedEventSchema
    # Add a fabricated stopped_vehicle event that references nonexistent track 999
    rep.observed_events.append(
        ObservedEventSchema(
            event_type="stopped_vehicle",
            timestamp_s=4.0,
            description="Fabricated vehicle stoppage on highway.",
            supporting_evidence_ref="stop_999_fabricated",
        )
    )

    corrected, diag = GroundingValidator.validate_and_correct(rep, norm)

    # The fabricated stoppage event must be stripped from observed_events
    assert not any(e.supporting_evidence_ref == "stop_999_fabricated" for e in corrected.observed_events)
    assert any("Stripped fabricated stopped_vehicle event" in w for w in diag.warnings)
    assert any("Unsupported stopped vehicle event citation" in u for u in corrected.unverified_aspects)


# ------------------------------------------------------------------------------
# Test 24: GroundingValidator Sanitizes Stoppage Probable Cause
# ------------------------------------------------------------------------------

def test_grounding_validator_sanitizes_stoppage_probable_cause():
    """Validates that narrative probable_cause in stopped_vehicles is sanitized for collisions and speeds."""
    norm = EvidenceNormalizer.normalize(
        job_id="job_pc_sanitize",
        video_id="vid_pc_sanitize",
        sequence_name="SEQ_PC_SANITIZE",
        cv_results=_create_sample_cv_results(),
    )
    norm.vlm_available = True
    provider = MockIncidentReportProvider()
    import asyncio
    rep = asyncio.run(provider.generate_report(norm))

    assert len(rep.stopped_vehicles) > 0
    rep.stopped_vehicles[0].probable_cause = "Vehicle stopped following a severe collision at 60 km/h."

    corrected, diag = GroundingValidator.validate_and_correct(rep, norm)

    pc = corrected.stopped_vehicles[0].probable_cause
    assert "collision" not in pc
    assert "60 km/h" not in pc
    assert "[unverified claim]" in pc
    assert "[uncalibrated displacement]" in pc
