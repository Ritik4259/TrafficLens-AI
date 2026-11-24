"""
Unit and integration tests for SceneInterpretation Pydantic schemas,
state machine transitions, and SQLAlchemy repository operations.
"""

from __future__ import annotations
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.session import create_db_engine, init_db
from backend.app.models.job import ProcessingJob
from backend.app.models.video import VideoAsset
from backend.app.models.scene_interpretation import (
    SceneInterpretation,
    SceneInterpretationStatus,
)
from backend.app.schemas.job import JobStatus
from backend.app.schemas.vlm import (
    EpistemicClaimSchema,
    ClaimCategoryEnum,
    TelemetryProvenanceSchema,
)
from backend.app.schemas.scene_interpretation import (
    SceneInterpretationStatus as SchemaPersistedStatus,
    SceneInterpretationLifecycleStatus,
    ALLOWED_INTERPRETATION_TRANSITIONS,
    ANALYTICAL_RESULT_FIELDS,
    InvalidInterpretationStatusTransitionError,
    validate_interpretation_status_transition,
    sanitize_error_text,
    sanitize_error_details,
    SceneInterpretationCreateSchema,
    SceneInterpretationErrorSchema,
    EnvironmentalContextSchema,
    SceneDescriptionSchema,
    TrafficEventInterpretationSchema,
    SceneInterpretationResponseSchema,
    SceneInterpretationHistoryResponseSchema,
)
from backend.app.repositories.scene_interpretation_repository import (
    SceneInterpretationRepository,
    SQLAlchemySceneInterpretationRepository,
)


@pytest.fixture
def db_session():
    """Provides an isolated in-memory SQLite database session with foreign key enforcement."""
    test_engine = create_db_engine("sqlite:///:memory:")
    init_db(test_engine)
    TestingSessionLocal = sessionmaker(
        bind=test_engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        test_engine.dispose()


def _create_fixture_job(
    session: Session,
    video_id: str = "vid_repo_101",
    job_id: str = "job_repo_101",
) -> tuple[VideoAsset, ProcessingJob]:
    """Helper creating parent VideoAsset and ProcessingJob."""
    asset = VideoAsset(
        id=video_id,
        filename="test_video.mp4",
        storage_path=f"storage/uploads/{video_id}.mp4",
        file_size_bytes=1024,
    )
    session.add(asset)
    session.flush()

    job = ProcessingJob(
        id=job_id,
        video_id=video_id,
        sequence_name="SEQ_TEST",
        status=JobStatus.COMPLETED,
    )
    session.add(job)
    session.commit()
    return asset, job


def _get_sample_analytical_payloads() -> dict:
    """Returns sample valid analytical field values for testing incomplete result protection."""
    return {
        "summary": "Sample summary text.",
        "environmental_context": EnvironmentalContextSchema(weather="clear", lighting="daylight"),
        "scene_description": SceneDescriptionSchema(roadway_topology="highway", lane_count=3),
        "event_interpretations": [
            TrafficEventInterpretationSchema(
                event_id="evt_sample",
                interpretation="Vehicle pulled over.",
            )
        ],
        "epistemic_claims": [
            EpistemicClaimSchema(
                claim_id="clm_sample",
                category=ClaimCategoryEnum.INFERENCE,
                statement="Plausible inference.",
                evidence_rationale="Visual context.",
            )
        ],
        "unverified_aspects": ["Aspect cannot be confirmed."],
        "evidence_package_id": "evpkg_sample_123",
    }


# ==============================================================================
# 1. Pydantic Schemas Unit Tests
# ==============================================================================

class TestSceneInterpretationSchemas:
    """Test suite covering Pydantic validation, serialization, and protection rules."""

    def test_persisted_statuses_match_model(self):
        """Validates that SchemaPersistedStatus exactly matches model SceneInterpretationStatus."""
        assert {s.value for s in SchemaPersistedStatus} == {"QUEUED", "PROCESSING", "COMPLETED", "FAILED"}
        assert "NOT_REQUESTED" not in {s.value for s in SchemaPersistedStatus}

    def test_lifecycle_statuses_include_not_requested(self):
        """Confirms client-facing lifecycle statuses include virtual NOT_REQUESTED."""
        assert {s.value for s in SceneInterpretationLifecycleStatus} == {
            "NOT_REQUESTED", "QUEUED", "PROCESSING", "COMPLETED", "FAILED"
        }

    def test_create_schema_defaults_and_validation(self):
        """Verifies default values and constraints in SceneInterpretationCreateSchema."""
        schema = SceneInterpretationCreateSchema()
        assert schema.frame_budget == 5
        assert schema.temperature == 0.2
        assert schema.provider_type is None
        assert schema.model_name is None

        # Valid custom payload
        custom = SceneInterpretationCreateSchema(
            provider_type="gemini",
            model_name="gemini-3.8-flash",
            frame_budget=8,
            temperature=0.4,
        )
        assert custom.frame_budget == 8
        assert custom.model_name == "gemini-3.8-flash"

        # Invalid frame budget (< 1 or > 10)
        with pytest.raises(ValidationError):
            SceneInterpretationCreateSchema(frame_budget=0)
        with pytest.raises(ValidationError):
            SceneInterpretationCreateSchema(frame_budget=15)

        # Extra forbidden fields
        with pytest.raises(ValidationError):
            SceneInterpretationCreateSchema(unknown_field="injected")

    @pytest.mark.parametrize("status", [
        SceneInterpretationLifecycleStatus.NOT_REQUESTED,
        SceneInterpretationLifecycleStatus.QUEUED,
        SceneInterpretationLifecycleStatus.PROCESSING,
        SceneInterpretationLifecycleStatus.FAILED,
    ])
    @pytest.mark.parametrize("field_name", ANALYTICAL_RESULT_FIELDS)
    def test_individual_analytical_fields_rejected_for_non_completed_status(self, status, field_name):
        """Ensures every analytical field is individually rejected when status is not COMPLETED."""
        samples = _get_sample_analytical_payloads()
        kwargs = {
            "job_id": "job_guard_test",
            "status": status,
            field_name: samples[field_name],
        }
        with pytest.raises(ValidationError) as exc_info:
            SceneInterpretationResponseSchema(**kwargs)
        assert f"cannot expose analytical result field '{field_name}'" in str(exc_info.value)

    @pytest.mark.parametrize("status", [
        SceneInterpretationLifecycleStatus.NOT_REQUESTED,
        SceneInterpretationLifecycleStatus.QUEUED,
        SceneInterpretationLifecycleStatus.PROCESSING,
        SceneInterpretationLifecycleStatus.FAILED,
    ])
    def test_all_analytical_fields_in_combination_rejected_for_non_completed_status(self, status):
        """Ensures all analytical fields combined are rejected for non-COMPLETED status."""
        samples = _get_sample_analytical_payloads()
        kwargs = {
            "job_id": "job_guard_combo",
            "status": status,
            **samples,
        }
        with pytest.raises(ValidationError) as exc_info:
            SceneInterpretationResponseSchema(**kwargs)
        assert "cannot expose analytical result field" in str(exc_info.value)

    def test_operational_metadata_allowed_on_incomplete_statuses(self):
        """Verifies operational metadata (tokens, duration, timestamps, provider, error) is allowed on incomplete statuses."""
        now = datetime.now(timezone.utc)
        # Operational fields on FAILED status
        resp_failed = SceneInterpretationResponseSchema(
            id="interp_failed_op",
            job_id="job_op_01",
            video_id="vid_op_01",
            status=SceneInterpretationLifecycleStatus.FAILED,
            created_at=now,
            started_at=now,
            completed_at=now,
            duration_seconds=5.2,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
            prompt_tokens=500,
            completion_tokens=50,
            total_tokens=550,
            error=SceneInterpretationErrorSchema(error_code="RATE_LIMIT", error_message="Quota exceeded"),
        )
        assert resp_failed.status == SceneInterpretationLifecycleStatus.FAILED
        assert resp_failed.total_tokens == 550
        assert resp_failed.duration_seconds == 5.2
        assert resp_failed.summary is None
        assert resp_failed.epistemic_claims is None

    def test_completed_status_requires_summary(self):
        """Verifies that COMPLETED status requires a valid non-empty summary."""
        # None summary
        with pytest.raises(ValidationError) as exc1:
            SceneInterpretationResponseSchema(
                job_id="job_comp_nosum",
                status=SceneInterpretationLifecycleStatus.COMPLETED,
                summary=None,
            )
        assert "Completed interpretation must contain a non-empty summary" in str(exc1.value)

        # Empty / whitespace summary
        with pytest.raises(ValidationError) as exc2:
            SceneInterpretationResponseSchema(
                job_id="job_comp_whitesum",
                status=SceneInterpretationLifecycleStatus.COMPLETED,
                summary="   ",
            )
        assert "Completed interpretation must contain a non-empty summary" in str(exc2.value)

    def test_completed_result_serialization(self):
        """Validates serialization of full completed interpretation result."""
        samples = _get_sample_analytical_payloads()
        now = datetime.now(timezone.utc)
        response = SceneInterpretationResponseSchema(
            id="interp_comp_99",
            job_id="job_test_99",
            video_id="vid_test_99",
            status=SceneInterpretationLifecycleStatus.COMPLETED,
            created_at=now,
            started_at=now,
            completed_at=now,
            duration_seconds=3.5,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
            prompt_tokens=1200,
            completion_tokens=250,
            total_tokens=1450,
            error=None,
            **samples,
        )

        assert response.status == SceneInterpretationLifecycleStatus.COMPLETED
        assert response.summary == "Sample summary text."
        assert response.environmental_context.weather == "clear"
        assert response.scene_description.lane_count == 3
        assert len(response.event_interpretations) == 1
        assert len(response.epistemic_claims) == 1
        assert response.evidence_package_id == "evpkg_sample_123"
        assert response.error is None

    def test_null_optional_metadata(self):
        """Verifies that optional fields (timing, tokens, error) default cleanly to None."""
        queued_response = SceneInterpretationResponseSchema(
            id="interp_q_01",
            job_id="job_q_01",
            status=SceneInterpretationLifecycleStatus.QUEUED,
        )
        assert queued_response.id == "interp_q_01"
        assert queued_response.video_id is None
        assert queued_response.started_at is None
        assert queued_response.completed_at is None
        assert queued_response.duration_seconds is None
        assert queued_response.prompt_tokens is None
        assert queued_response.completion_tokens is None
        assert queued_response.total_tokens is None
        assert queued_response.summary is None
        assert queued_response.error is None

    def test_not_requested_factory(self):
        """Verifies the virtual NOT_REQUESTED response construction."""
        resp = SceneInterpretationResponseSchema.not_requested(job_id="job_no_vlm", video_id="vid_101")
        assert resp.id is None
        assert resp.job_id == "job_no_vlm"
        assert resp.video_id == "vid_101"
        assert resp.status == SceneInterpretationLifecycleStatus.NOT_REQUESTED
        assert resp.summary is None
        assert resp.error is None

    def test_invalid_status_rejection(self):
        """Verifies that unapproved status strings are rejected."""
        with pytest.raises(ValidationError):
            SceneInterpretationResponseSchema(
                job_id="job_bad",
                status="RANDOM_INVALID_STATUS",  # type: ignore
            )

    def test_orm_model_serialization(self, db_session: Session):
        """Verifies from_attributes / model_validate directly adapts SQLAlchemy model instances."""
        _, job = _create_fixture_job(db_session)
        now = datetime.now(timezone.utc)

        # 1. ORM model in QUEUED status
        interp_orm_queued = SceneInterpretation(
            id="interp_orm_q",
            job_id=job.id,
            status=SceneInterpretationStatus.QUEUED,
            created_at=now,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
        )
        db_session.add(interp_orm_queued)
        db_session.commit()

        schema_queued = SceneInterpretationResponseSchema.model_validate(interp_orm_queued)
        assert schema_queued.id == "interp_orm_q"
        assert schema_queued.job_id == job.id
        assert schema_queued.video_id == job.video_id
        assert schema_queued.status == SceneInterpretationLifecycleStatus.QUEUED
        assert schema_queued.summary is None
        assert schema_queued.error is None

        # 2. ORM model in FAILED status with error
        interp_orm_failed = SceneInterpretation(
            id="interp_orm_f",
            job_id=job.id,
            status=SceneInterpretationStatus.FAILED,
            created_at=now,
            completed_at=now,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
            error_code="VLM_TIMEOUT",
            error_message="Gateway timeout",
            error_details={"timeout_s": 30.0},
        )
        db_session.add(interp_orm_failed)
        db_session.commit()

        schema_failed = SceneInterpretationResponseSchema.model_validate(interp_orm_failed)
        assert schema_failed.status == SceneInterpretationLifecycleStatus.FAILED
        assert schema_failed.error is not None
        assert schema_failed.error.error_code == "VLM_TIMEOUT"
        assert schema_failed.error.error_message == "Gateway timeout"
        assert schema_failed.error.error_details["timeout_s"] == 30.0


# ==============================================================================
# 2. Error Sanitization Regression Tests
# ==============================================================================

class TestErrorSanitization:
    """Verifies scrubbing of credentials, absolute filesystem paths, tracebacks, and raw responses."""

    def test_credentials_scrubbed(self):
        """Verifies Gemini, OpenAI, Bearer tokens, and password query parameters are redacted."""
        raw_msg = (
            "Failed calling Gemini endpoint with API key  and "
            "sk-1234567890abcdefghijklmnopqrstuv and Bearer  and "
            "api_key=mySecretKey123456"
        )
        sanitized = sanitize_error_text(raw_msg)
        assert "AIzaSy" not in sanitized
        assert "sk-1234" not in sanitized
        assert "eyJhbGci" not in sanitized
        assert "mySecretKey" not in sanitized
        assert "[REDACTED_CREDENTIAL]" in sanitized

    def test_local_filesystem_paths_scrubbed(self):
        """Verifies local Windows and Unix absolute paths are redacted."""
        raw_msg = (
            "File read failure at C:\\Users\\bhard\\Desktop\\TrafficLens AI\\backend\\data\\video.mp4 and "
            "D:/projects/TrafficLens/secret.txt and /home/ubuntu/app/storage/uploads/123.mp4 and "
            "file:///C:/Users/bhard/sensitive.json"
        )
        sanitized = sanitize_error_text(raw_msg)
        assert "C:\\Users\\bhard" not in sanitized
        assert "D:/projects/TrafficLens" not in sanitized
        assert "/home/ubuntu/app" not in sanitized
        assert "sensitive.json" not in sanitized
        assert "[REDACTED_PATH]" in sanitized

    def test_tracebacks_scrubbed(self):
        """Verifies Python exception tracebacks and line references are redacted."""
        raw_msg = (
            "Inference crash occurred:\n"
            "Traceback (most recent call last):\n"
            '  File "C:\\Users\\bhard\\Desktop\\TrafficLens AI\\backend\\app\\services\\vlm\\gemini_provider.py", line 145, in generate\n'
            "    raise ZeroDivisionError('integer division by zero')\n"
            "ZeroDivisionError: integer division by zero"
        )
        sanitized = sanitize_error_text(raw_msg)
        assert "Traceback" not in sanitized
        assert "gemini_provider.py" not in sanitized
        assert "[TRACEBACK REDACTED]" in sanitized or "[LINE REFERENCE REDACTED]" in sanitized

    def test_raw_provider_json_dump_scrubbed(self):
        """Verifies raw candidate JSON payloads are redacted."""
        raw_msg = (
            'Gemini response parsing error on raw payload: '
            '{"candidates": [{"content": {"parts": [{"text": "secret model dump"}]}}]}'
        )
        sanitized = sanitize_error_text(raw_msg)
        assert "secret model dump" not in sanitized
        assert "[RAW RESPONSE REDACTED]" in sanitized

    def test_useful_error_codes_and_actionable_messages_preserved(self):
        """Verifies standard error codes and clean actionable text are preserved."""
        raw_msg = "VLM provider request timed out after 30.0 seconds. Please retry with a smaller frame budget."
        sanitized = sanitize_error_text(raw_msg)
        assert sanitized == raw_msg

        err_schema = SceneInterpretationErrorSchema(
            error_code="VLM_PROVIDER_TIMEOUT",
            error_message=raw_msg,
            error_details={"timeout_s": 30.0, "retryable": True},
        )
        assert err_schema.error_code == "VLM_PROVIDER_TIMEOUT"
        assert err_schema.error_message == raw_msg
        assert err_schema.error_details["timeout_s"] == 30.0
        assert err_schema.error_details["retryable"] is True

    def test_error_details_recursive_sanitization(self):
        """Verifies error_details dictionary drops sensitive debug keys and cleans strings."""
        details = {
            "http_status": 500,
            "retryable": False,
            "raw_response": {"candidates": [{"content": "secret"}]},
            "stack_trace": "Traceback: ...",
            "api_key": "AIzaSySecret",
            "path_info": "C:\\Users\\bhard\\AppData\\Local\\Temp\\sample.mp4",
            "sub_dict": {
                "secret_token": "sk-mySecretOpenAIKey12345678",
                "nested_path": "/var/log/trafficlens.log",
            },
            "list_items": ["Normal notice", "Found key AIzaSyTestKey9999999999999999999999"],
        }
        cleaned = sanitize_error_details(details)
        assert cleaned is not None
        assert "raw_response" not in cleaned
        assert "stack_trace" not in cleaned
        assert "api_key" not in cleaned
        assert cleaned["http_status"] == 500
        assert "[REDACTED_PATH]" in cleaned["path_info"]
        assert "C:\\Users" not in cleaned["path_info"]
        assert "[REDACTED_CREDENTIAL]" in cleaned["sub_dict"]["secret_token"]
        assert "[REDACTED_PATH]" in cleaned["sub_dict"]["nested_path"]
        assert "[REDACTED_CREDENTIAL]" in cleaned["list_items"][1]


# ==============================================================================
# 3. Lifecycle State Transition Validator Tests
# ==============================================================================

class TestLifecycleTransitions:
    """Tests the state transition validator and rejection of illegal/terminal mutations."""

    def test_allowed_transitions(self):
        """Verifies strictly permitted transitions: QUEUED->PROCESSING, PROCESSING->COMPLETED|FAILED."""
        # QUEUED -> PROCESSING
        validate_interpretation_status_transition(
            SceneInterpretationStatus.QUEUED,
            SceneInterpretationStatus.PROCESSING,
        )

        # PROCESSING -> COMPLETED
        validate_interpretation_status_transition(
            SceneInterpretationStatus.PROCESSING,
            SceneInterpretationStatus.COMPLETED,
        )

        # PROCESSING -> FAILED
        validate_interpretation_status_transition(
            SceneInterpretationStatus.PROCESSING,
            SceneInterpretationStatus.FAILED,
        )

    def test_terminal_states_reject_all_outgoing_transitions(self):
        """Verifies that COMPLETED and FAILED reject all transitions (immutable records)."""
        terminal_states = [SceneInterpretationStatus.COMPLETED, SceneInterpretationStatus.FAILED]
        all_targets = [
            SceneInterpretationStatus.QUEUED,
            SceneInterpretationStatus.PROCESSING,
            SceneInterpretationStatus.COMPLETED,
            SceneInterpretationStatus.FAILED,
        ]

        for term in terminal_states:
            for target in all_targets:
                with pytest.raises(InvalidInterpretationStatusTransitionError):
                    validate_interpretation_status_transition(term, target)

    def test_disallowed_transitions(self):
        """Verifies illegal skips and backwards transitions are rejected."""
        # QUEUED cannot jump directly to COMPLETED or FAILED
        with pytest.raises(InvalidInterpretationStatusTransitionError):
            validate_interpretation_status_transition(
                SceneInterpretationStatus.QUEUED,
                SceneInterpretationStatus.COMPLETED,
            )
        with pytest.raises(InvalidInterpretationStatusTransitionError):
            validate_interpretation_status_transition(
                SceneInterpretationStatus.QUEUED,
                SceneInterpretationStatus.FAILED,
            )

        # PROCESSING cannot revert to QUEUED
        with pytest.raises(InvalidInterpretationStatusTransitionError):
            validate_interpretation_status_transition(
                SceneInterpretationStatus.PROCESSING,
                SceneInterpretationStatus.QUEUED,
            )

        # Same state transitions are rejected
        with pytest.raises(InvalidInterpretationStatusTransitionError):
            validate_interpretation_status_transition(
                SceneInterpretationStatus.QUEUED,
                SceneInterpretationStatus.QUEUED,
            )


# ==============================================================================
# 4. Repository Layer Integration & Transaction Tests
# ==============================================================================

class TestSceneInterpretationRepository:
    """Comprehensive tests for SQLAlchemySceneInterpretationRepository."""

    def test_create_and_get_by_id(self, db_session: Session):
        """Tests basic creation and ID lookup."""
        _, job = _create_fixture_job(db_session)
        repo = SQLAlchemySceneInterpretationRepository(db_session)

        interp = SceneInterpretation(
            id="interp_repo_01",
            job_id=job.id,
            status=SceneInterpretationStatus.QUEUED,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
        )
        created = repo.create(interp)
        assert created.id == "interp_repo_01"

        # Lookup
        found = repo.get_by_id("interp_repo_01")
        assert found is not None
        assert found.job_id == job.id
        assert found.status == SceneInterpretationStatus.QUEUED
        assert found.job.video_id == job.video_id

        # Missing lookup returns None
        assert repo.get_by_id("non_existent_id") is None

    def test_active_interpretation_lookup(self, db_session: Session):
        """Tests get_active_for_job only returning QUEUED or PROCESSING tasks."""
        _, job = _create_fixture_job(db_session)
        repo = SQLAlchemySceneInterpretationRepository(db_session)

        # 1. No interpretations -> None
        assert repo.get_active_for_job(job.id) is None

        # 2. Add QUEUED task -> returns active
        interp1 = SceneInterpretation(
            id="act_q_01",
            job_id=job.id,
            status=SceneInterpretationStatus.QUEUED,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
        )
        repo.create(interp1)
        active = repo.get_active_for_job(job.id)
        assert active is not None
        assert active.id == "act_q_01"

        # 3. Transition to PROCESSING -> still active
        repo.claim_for_processing("act_q_01")
        active = repo.get_active_for_job(job.id)
        assert active is not None
        assert active.status == SceneInterpretationStatus.PROCESSING

        # 4. Transition to COMPLETED -> no active task remains
        repo.complete("act_q_01", summary="Completed summary.")
        assert repo.get_active_for_job(job.id) is None

    def test_historical_attempt_listing_and_counting(self, db_session: Session):
        """Tests list_by_job_id (chronological order) and count_by_job_id."""
        _, job = _create_fixture_job(db_session)
        repo = SQLAlchemySceneInterpretationRepository(db_session)

        t1 = datetime(2026, 11, 23, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 11, 23, 10, 5, 0, tzinfo=timezone.utc)
        t3 = datetime(2026, 11, 23, 10, 10, 0, tzinfo=timezone.utc)

        # Attempt 1: Failed
        repo.create(SceneInterpretation(
            id="attempt_01",
            job_id=job.id,
            status=SceneInterpretationStatus.FAILED,
            created_at=t1,
            completed_at=t1,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
            error_code="TIMEOUT",
        ))

        # Attempt 2: Failed
        repo.create(SceneInterpretation(
            id="attempt_02",
            job_id=job.id,
            status=SceneInterpretationStatus.FAILED,
            created_at=t2,
            completed_at=t2,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
            error_code="RATE_LIMIT",
        ))

        # Attempt 3: Completed
        repo.create(SceneInterpretation(
            id="attempt_03",
            job_id=job.id,
            status=SceneInterpretationStatus.COMPLETED,
            created_at=t3,
            completed_at=t3,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
            summary="All clear",
        ))

        # Count total attempts
        assert repo.count_by_job_id(job.id) == 3

        # Chronological listing (ascending=True, default)
        asc_list = repo.list_by_job_id(job.id, ascending=True)
        assert [r.id for r in asc_list] == ["attempt_01", "attempt_02", "attempt_03"]

        # Reverse-chronological listing (ascending=False)
        desc_list = repo.list_by_job_id(job.id, ascending=False)
        assert [r.id for r in desc_list] == ["attempt_03", "attempt_02", "attempt_01"]

        # Latest attempt
        latest = repo.get_latest_by_job_id(job.id)
        assert latest is not None
        assert latest.id == "attempt_03"

    def test_list_queued_and_list_processing(self, db_session: Session):
        """Verifies FIFO listing of QUEUED records and discovery of PROCESSING records."""
        _, job1 = _create_fixture_job(db_session, video_id="v_q1", job_id="j_q1")
        _, job2 = _create_fixture_job(db_session, video_id="v_q2", job_id="j_q2")
        repo = SQLAlchemySceneInterpretationRepository(db_session)

        t1 = datetime(2026, 11, 23, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 11, 23, 10, 1, 0, tzinfo=timezone.utc)

        repo.create(SceneInterpretation(
            id="q_early",
            job_id=job1.id,
            status=SceneInterpretationStatus.QUEUED,
            created_at=t1,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
        ))
        repo.create(SceneInterpretation(
            id="q_later",
            job_id=job2.id,
            status=SceneInterpretationStatus.QUEUED,
            created_at=t2,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
        ))

        queued = repo.list_queued()
        assert len(queued) == 2
        assert queued[0].id == "q_early"  # FIFO order
        assert queued[1].id == "q_later"

        # Claim the first one
        repo.claim_for_processing("q_early")

        # Now only 1 is QUEUED, and 1 is PROCESSING
        assert len(repo.list_queued()) == 1
        processing = repo.list_processing()
        assert len(processing) == 1
        assert processing[0].id == "q_early"

    def test_conditional_claim_atomicity_and_cache_freshness(self, db_session: Session):
        """Verifies atomic conditional update and that in-memory objects are refreshed (not stale)."""
        _, job = _create_fixture_job(db_session)
        repo = SQLAlchemySceneInterpretationRepository(db_session)

        repo.create(SceneInterpretation(
            id="claim_fresh_01",
            job_id=job.id,
            status=SceneInterpretationStatus.QUEUED,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
        ))

        # Eagerly load instance into session identity map
        loaded_before_claim = repo.get_by_id("claim_fresh_01")
        assert loaded_before_claim is not None
        assert loaded_before_claim.status == SceneInterpretationStatus.QUEUED

        # Claim for processing
        claimed = repo.claim_for_processing("claim_fresh_01")
        assert claimed is not None
        assert claimed.status == SceneInterpretationStatus.PROCESSING
        assert claimed.started_at is not None

        # Verify the previously-loaded instance in session identity map is refreshed and NOT stale
        assert loaded_before_claim.status == SceneInterpretationStatus.PROCESSING
        assert loaded_before_claim.started_at is not None

        # Second claim attempt on same record returns None (already PROCESSING)
        second_claim = repo.claim_for_processing("claim_fresh_01")
        assert second_claim is None

        # Non-existent ID returns None
        assert repo.claim_for_processing("non_existent_id") is None

    def test_atomic_completion_same_transaction(self, db_session: Session):
        """Verifies atomic transition from PROCESSING to COMPLETED with results in a single transaction."""
        _, job = _create_fixture_job(db_session)
        repo = SQLAlchemySceneInterpretationRepository(db_session)

        repo.create(SceneInterpretation(
            id="comp_test_01",
            job_id=job.id,
            status=SceneInterpretationStatus.QUEUED,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
        ))
        repo.claim_for_processing("comp_test_01")

        completed = repo.complete(
            interpretation_id="comp_test_01",
            summary="Traffic flowing normally with no stoppages.",
            environmental_context={"weather": "clear", "lighting": "daylight"},
            scene_description={"roadway_topology": "divided_highway", "lane_count": 4},
            prompt_tokens=1000,
            completion_tokens=200,
            total_tokens=1200,
        )
        assert completed is not None
        assert completed.status == SceneInterpretationStatus.COMPLETED
        assert completed.summary == "Traffic flowing normally with no stoppages."
        assert completed.environmental_context["weather"] == "clear"
        assert completed.scene_description["lane_count"] == 4
        assert completed.completed_at is not None
        assert completed.duration_seconds is not None
        assert completed.prompt_tokens == 1000
        assert completed.total_tokens == 1200

        # Attempting to mutate completed record raises InvalidInterpretationStatusTransitionError
        with pytest.raises(InvalidInterpretationStatusTransitionError):
            repo.complete("comp_test_01", summary="Mutate after terminal")

    def test_atomic_failure_same_transaction_with_sanitization(self, db_session: Session):
        """Verifies atomic transition from PROCESSING to FAILED with sanitized error in same transaction."""
        _, job = _create_fixture_job(db_session)
        repo = SQLAlchemySceneInterpretationRepository(db_session)

        repo.create(SceneInterpretation(
            id="fail_test_01",
            job_id=job.id,
            status=SceneInterpretationStatus.QUEUED,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
        ))
        repo.claim_for_processing("fail_test_01")

        failed = repo.fail(
            interpretation_id="fail_test_01",
            error_code="VLM_PROVIDER_TIMEOUT",
            error_message="Gemini API key AIzaSyTestKey999 timed out reading C:\\Users\\bhard\\video.mp4 after 30s.",
            error_details={"timeout_s": 30.0, "api_key": "AIzaSySecret", "path": "D:/secret.txt"},
        )
        assert failed is not None
        assert failed.status == SceneInterpretationStatus.FAILED
        assert failed.error_code == "VLM_PROVIDER_TIMEOUT"
        # Verify sanitization applied to persisted database fields
        assert "AIzaSy" not in failed.error_message
        assert "C:\\Users" not in failed.error_message
        assert "api_key" not in failed.error_details
        assert "D:/secret" not in failed.error_details["path"]
        assert failed.completed_at is not None
        assert failed.duration_seconds is not None

        # Attempting to transition failed record raises InvalidInterpretationStatusTransitionError
        with pytest.raises(InvalidInterpretationStatusTransitionError):
            repo.complete("fail_test_01", summary="Cannot complete after failure")

    def test_repository_methods_do_not_commit_unexpectedly(self, db_session: Session):
        """Verifies repo operations use flush, not commit, leaving transaction boundary to caller."""
        _, job = _create_fixture_job(db_session)
        repo = SQLAlchemySceneInterpretationRepository(db_session)

        # Create record (flushed but uncommitted)
        repo.create(SceneInterpretation(
            id="uncommitted_01",
            job_id=job.id,
            status=SceneInterpretationStatus.QUEUED,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
        ))

        # Verify it is visible within current transaction
        assert repo.get_by_id("uncommitted_01") is not None

        # Roll back transaction -> uncommitted record vanishes
        db_session.rollback()
        assert repo.get_by_id("uncommitted_01") is None

    def test_integrity_error_rollback_behavior(self, db_session: Session):
        """Verifies session rollback on partial unique index violation."""
        _, job = _create_fixture_job(db_session)
        repo = SQLAlchemySceneInterpretationRepository(db_session)

        # First active interpretation (persisted and committed)
        repo.create(SceneInterpretation(
            id="int_uniq_01",
            job_id=job.id,
            status=SceneInterpretationStatus.QUEUED,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
        ))
        db_session.commit()

        # Second active interpretation for same job raises IntegrityError
        int_uniq_02 = SceneInterpretation(
            id="int_uniq_02",
            job_id=job.id,
            status=SceneInterpretationStatus.QUEUED,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
        )
        with pytest.raises(IntegrityError):
            repo.create(int_uniq_02)

        # Verify session was cleanly rolled back and remains healthy
        retrieved = repo.get_by_id("int_uniq_01")
        assert retrieved is not None
        assert repo.get_by_id("int_uniq_02") is None

    def test_schema_conversion_helpers(self, db_session: Session):
        """Tests repo.to_response_schema() and repo.to_history_schema()."""
        _, job = _create_fixture_job(db_session)
        repo = SQLAlchemySceneInterpretationRepository(db_session)

        interp = repo.create(SceneInterpretation(
            id="conv_01",
            job_id=job.id,
            status=SceneInterpretationStatus.QUEUED,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
        ))

        # to_response_schema
        resp = repo.to_response_schema(interp)
        assert isinstance(resp, SceneInterpretationResponseSchema)
        assert resp.id == "conv_01"
        assert resp.job_id == job.id
        assert resp.video_id == job.video_id
        assert resp.status == SceneInterpretationLifecycleStatus.QUEUED

        # to_history_schema
        hist = repo.to_history_schema(job.id)
        assert isinstance(hist, SceneInterpretationHistoryResponseSchema)
        assert hist.job_id == job.id
        assert hist.total_attempts == 1
        assert len(hist.attempts) == 1
        assert hist.attempts[0].id == "conv_01"
