"""
Unit and integration tests for SceneInterpretation model, relationships,
partial unique index enforcement, UTC timestamps, JSON fields, and Alembic migrations.
"""

from __future__ import annotations
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import pytest
from sqlalchemy import select, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from alembic import command
from alembic.config import Config

from backend.app.db.session import create_db_engine, init_db
from backend.app.models.job import ProcessingJob
from backend.app.models.video import VideoAsset
from backend.app.models.scene_interpretation import (
    SceneInterpretation,
    SceneInterpretationStatus,
)
from backend.app.schemas.job import JobStatus


@pytest.fixture
def in_memory_db():
    """Provides an isolated in-memory SQLite database session with PRAGMA foreign_keys=ON."""
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


def _create_fixture_asset_and_job(
    session: Session,
    video_id: str = "vid_test_101",
    job_id: str = "job_test_101",
) -> tuple[VideoAsset, ProcessingJob]:
    """Helper creating parent VideoAsset and ProcessingJob in COMPLETED status."""
    asset = VideoAsset(
        id=video_id,
        filename="highway_test.mp4",
        storage_path=f"storage/uploads/{video_id}.mp4",
        file_size_bytes=1024 * 1024,
        duration_seconds=30.0,
        fps=25.0,
        width=1280,
        height=720,
    )
    session.add(asset)
    session.flush()

    job = ProcessingJob(
        id=job_id,
        video_id=video_id,
        sequence_name="SEQ_TEST_101",
        status=JobStatus.COMPLETED,
        duration_seconds=12.5,
    )
    session.add(job)
    session.commit()
    return asset, job


# ==============================================================================
# 1. Creation and Retrieval Tests
# ==============================================================================

def test_create_and_retrieve_scene_interpretation(in_memory_db: Session):
    """Validates persisting and retrieving a basic SceneInterpretation record."""
    _, job = _create_fixture_asset_and_job(in_memory_db)

    now = datetime.now(timezone.utc)
    interp = SceneInterpretation(
        id="interp_abc123456789",
        job_id=job.id,
        status=SceneInterpretationStatus.QUEUED,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
        created_at=now,
    )
    in_memory_db.add(interp)
    in_memory_db.commit()

    retrieved = in_memory_db.get(SceneInterpretation, "interp_abc123456789")
    assert retrieved is not None
    assert retrieved.id == "interp_abc123456789"
    assert retrieved.job_id == job.id
    assert retrieved.status == SceneInterpretationStatus.QUEUED
    assert retrieved.provider_type == "gemini"
    assert retrieved.model_name == "gemini-3.8-flash"
    assert retrieved.started_at is None
    assert retrieved.completed_at is None
    assert retrieved.duration_seconds is None


# ==============================================================================
# 2. Relationship and Back-Populates Tests (Without Denormalized video_id)
# ==============================================================================

def test_processing_job_relationship_and_back_populates(in_memory_db: Session):
    """Verifies bidirectional relationship between ProcessingJob and SceneInterpretation,
    and navigating to VideoAsset through job without a redundant video_id on SceneInterpretation."""
    _, job = _create_fixture_asset_and_job(in_memory_db)

    interp = SceneInterpretation(
        id="interp_rel_01",
        job_id=job.id,
        status=SceneInterpretationStatus.QUEUED,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
    )
    in_memory_db.add(interp)
    in_memory_db.commit()

    # Refresh job to test eager / lazy relationship loading
    in_memory_db.refresh(job)
    assert len(job.scene_interpretations) == 1
    assert job.scene_interpretations[0].id == "interp_rel_01"
    assert job.scene_interpretations[0].job.id == job.id
    # Video is resolved cleanly through the job relationship
    assert interp.job.video_id == job.video_id
    assert interp.job.video_asset.id == job.video_id
    # Assert video_id is not a direct column on SceneInterpretation
    mapper = inspect(SceneInterpretation)
    column_names = [col.key for col in mapper.columns]
    assert "video_id" not in column_names


# ==============================================================================
# 3. Enum Integrity: NOT_REQUESTED Excluded from Persisted Enum
# ==============================================================================

def test_status_enum_contains_only_persisted_lifecycle_states():
    """Confirms SceneInterpretationStatus contains exactly QUEUED, PROCESSING, COMPLETED, FAILED.
    NOT_REQUESTED is virtual and must NOT be an enum member of the persisted model."""
    valid_members = {s.value for s in SceneInterpretationStatus}
    assert valid_members == {"QUEUED", "PROCESSING", "COMPLETED", "FAILED"}
    assert "NOT_REQUESTED" not in valid_members


# ==============================================================================
# 4. Multiple Historical Attempts and Ordering Tests
# ==============================================================================

def test_multiple_historical_attempts_ordering(in_memory_db: Session):
    """Verifies that multiple historical attempts are stored and ordered by created_at DESC."""
    _, job = _create_fixture_asset_and_job(in_memory_db)

    t1 = datetime(2026, 11, 23, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 11, 23, 10, 5, 0, tzinfo=timezone.utc)

    # Attempt 1 failed
    interp1 = SceneInterpretation(
        id="interp_hist_01",
        job_id=job.id,
        status=SceneInterpretationStatus.FAILED,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
        created_at=t1,
        completed_at=t1,
        error_code="VLM_PROVIDER_TIMEOUT",
        error_message="Gateway timeout",
    )
    in_memory_db.add(interp1)
    in_memory_db.commit()

    # Attempt 2 completed
    interp2 = SceneInterpretation(
        id="interp_hist_02",
        job_id=job.id,
        status=SceneInterpretationStatus.COMPLETED,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
        created_at=t2,
        completed_at=t2,
        summary="Clear day with moderate traffic.",
    )
    in_memory_db.add(interp2)
    in_memory_db.commit()

    in_memory_db.refresh(job)
    assert len(job.scene_interpretations) == 2
    # Verify order_by="desc(SceneInterpretation.created_at)"
    assert job.scene_interpretations[0].id == "interp_hist_02"
    assert job.scene_interpretations[1].id == "interp_hist_01"


# ==============================================================================
# 5. Partial Unique Index Enforcement Tests
# ==============================================================================

def test_partial_unique_index_rejects_second_active_interpretation(in_memory_db: Session):
    """Verifies that uix_active_interpretation_per_job rejects a second active interpretation for the same job."""
    _, job = _create_fixture_asset_and_job(in_memory_db)

    # First active task: QUEUED
    interp1 = SceneInterpretation(
        id="interp_act_01",
        job_id=job.id,
        status=SceneInterpretationStatus.QUEUED,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
    )
    in_memory_db.add(interp1)
    in_memory_db.commit()

    # Second active task for same job: QUEUED -> must raise IntegrityError
    interp2 = SceneInterpretation(
        id="interp_act_02",
        job_id=job.id,
        status=SceneInterpretationStatus.QUEUED,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
    )
    in_memory_db.add(interp2)
    with pytest.raises(IntegrityError):
        in_memory_db.commit()
    in_memory_db.rollback()

    # Second active task for same job: PROCESSING -> must also raise IntegrityError
    interp3 = SceneInterpretation(
        id="interp_act_03",
        job_id=job.id,
        status=SceneInterpretationStatus.PROCESSING,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
    )
    in_memory_db.add(interp3)
    with pytest.raises(IntegrityError):
        in_memory_db.commit()
    in_memory_db.rollback()


def test_partial_unique_index_allows_different_jobs_active(in_memory_db: Session):
    """Verifies that multiple jobs can each have one active interpretation simultaneously."""
    _, job1 = _create_fixture_asset_and_job(in_memory_db, video_id="vid_m1", job_id="job_m1")
    _, job2 = _create_fixture_asset_and_job(in_memory_db, video_id="vid_m2", job_id="job_m2")

    interp1 = SceneInterpretation(
        id="interp_m1",
        job_id=job1.id,
        status=SceneInterpretationStatus.QUEUED,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
    )
    interp2 = SceneInterpretation(
        id="interp_m2",
        job_id=job2.id,
        status=SceneInterpretationStatus.PROCESSING,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
    )
    in_memory_db.add(interp1)
    in_memory_db.add(interp2)
    in_memory_db.commit()

    assert in_memory_db.get(SceneInterpretation, "interp_m1") is not None
    assert in_memory_db.get(SceneInterpretation, "interp_m2") is not None


def test_create_interpretation_after_previous_failed(in_memory_db: Session):
    """Verifies that once an interpretation transitions to FAILED, a new active interpretation can be inserted."""
    _, job = _create_fixture_asset_and_job(in_memory_db)

    # Initial attempt fails
    interp1 = SceneInterpretation(
        id="interp_fail_01",
        job_id=job.id,
        status=SceneInterpretationStatus.FAILED,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
        error_code="VLM_PROVIDER_TIMEOUT",
    )
    in_memory_db.add(interp1)
    in_memory_db.commit()

    # Retry creates a new record in QUEUED status: allowed because interp1 is not QUEUED/PROCESSING
    interp2 = SceneInterpretation(
        id="interp_fail_02",
        job_id=job.id,
        status=SceneInterpretationStatus.QUEUED,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
    )
    in_memory_db.add(interp2)
    in_memory_db.commit()

    assert in_memory_db.get(SceneInterpretation, "interp_fail_01").status == SceneInterpretationStatus.FAILED
    assert in_memory_db.get(SceneInterpretation, "interp_fail_02").status == SceneInterpretationStatus.QUEUED


def test_create_interpretation_after_previous_completed(in_memory_db: Session):
    """Verifies that once an interpretation transitions to COMPLETED, another interpretation can be created."""
    _, job = _create_fixture_asset_and_job(in_memory_db)

    # Initial attempt completed
    interp1 = SceneInterpretation(
        id="interp_comp_01",
        job_id=job.id,
        status=SceneInterpretationStatus.COMPLETED,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
        summary="Initial run summary.",
    )
    in_memory_db.add(interp1)
    in_memory_db.commit()

    # Subsequent re-evaluation created in QUEUED status: allowed because interp1 is COMPLETED
    interp2 = SceneInterpretation(
        id="interp_comp_02",
        job_id=job.id,
        status=SceneInterpretationStatus.QUEUED,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
    )
    in_memory_db.add(interp2)
    in_memory_db.commit()

    assert in_memory_db.get(SceneInterpretation, "interp_comp_01").status == SceneInterpretationStatus.COMPLETED
    assert in_memory_db.get(SceneInterpretation, "interp_comp_02").status == SceneInterpretationStatus.QUEUED


# ==============================================================================
# 6. Foreign Key & Cascade Deletion Tests
# ==============================================================================

def test_foreign_key_enforcement_and_cascade_delete(in_memory_db: Session):
    """Verifies foreign key constraints and cascade deletion of interpretations when job is deleted."""
    asset, job = _create_fixture_asset_and_job(in_memory_db, video_id="vid_casc", job_id="job_casc")

    # 1. Reject orphan interpretation referencing non-existent job
    with pytest.raises(IntegrityError):
        bad_interp = SceneInterpretation(
            id="interp_orphan",
            job_id="job_does_not_exist",
            status=SceneInterpretationStatus.QUEUED,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
        )
        in_memory_db.add(bad_interp)
        in_memory_db.commit()
    in_memory_db.rollback()

    # 2. Add valid interpretation
    interp = SceneInterpretation(
        id="interp_casc_01",
        job_id=job.id,
        status=SceneInterpretationStatus.COMPLETED,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
    )
    in_memory_db.add(interp)
    in_memory_db.commit()

    assert in_memory_db.get(SceneInterpretation, "interp_casc_01") is not None

    # 3. Delete parent ProcessingJob -> SceneInterpretation must cascade delete
    in_memory_db.delete(job)
    in_memory_db.commit()

    assert in_memory_db.get(SceneInterpretation, "interp_casc_01") is None


# ==============================================================================
# 7. JSON Persistence & Sanitized Error Storage (No raw_response)
# ==============================================================================

def test_json_fields_and_sanitized_error_storage(in_memory_db: Session):
    """Verifies structured JSON columns, token accounting, error diagnostics,
    and confirms that raw_response is eliminated from the model."""
    _, job = _create_fixture_asset_and_job(in_memory_db)

    t_created = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)
    t_started = datetime(2026, 11, 23, 12, 0, 1, tzinfo=timezone.utc)
    t_completed = datetime(2026, 11, 23, 12, 0, 5, tzinfo=timezone.utc)

    env_ctx = {
        "weather": "clear",
        "lighting": "daylight",
        "pavement": "dry",
        "camera_angle": "elevated_diagonal",
    }
    scene_desc = {
        "roadway_topology": "divided_highway",
        "lane_count": 3,
        "traffic_density_qualitative": "moderate",
    }
    events = [
        {
            "event_id": "evt_stopped_01",
            "event_type": "stopped_vehicle",
            "track_id": 12,
            "interpretation": "Vehicle stopped on shoulder with hazard lights.",
        }
    ]
    claims = [
        {
            "category": "cv_fact",
            "text": "Track 12 remained stationary for 14.5 seconds.",
            "citations": ["kf_job_12_f100"],
            "subject": "Track 12",
            "predicate": "stationary_duration",
            "value": 14.5,
            "unit": "seconds",
        },
        {
            "category": "visual_observation",
            "text": "Vehicle is situated on right road shoulder.",
            "citations": ["kf_job_12_f100"],
        },
    ]
    unverified = ["Signal state out of frame."]
    err_details = {"http_status": 200, "finish_reason": "STOP"}

    interp = SceneInterpretation(
        id="interp_full_01",
        job_id=job.id,
        status=SceneInterpretationStatus.COMPLETED,
        created_at=t_created,
        started_at=t_started,
        completed_at=t_completed,
        duration_seconds=4.0,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
        prompt_tokens=1500,
        completion_tokens=350,
        total_tokens=1850,
        summary="Clear day with moderate divided highway traffic.",
        environmental_context=env_ctx,
        scene_description=scene_desc,
        event_interpretations=events,
        epistemic_claims=claims,
        unverified_aspects=unverified,
        evidence_package_id="evpkg_job_101_3k",
        error_code=None,
        error_message=None,
        error_details=err_details,
    )
    in_memory_db.add(interp)
    in_memory_db.commit()

    # Query back in a clean read
    row = in_memory_db.get(SceneInterpretation, "interp_full_01")
    assert row is not None
    assert row.created_at.tzinfo == timezone.utc
    assert row.started_at.tzinfo == timezone.utc
    assert row.completed_at.tzinfo == timezone.utc
    assert row.prompt_tokens == 1500
    assert row.completion_tokens == 350
    assert row.total_tokens == 1850
    assert row.environmental_context["weather"] == "clear"
    assert row.scene_description["lane_count"] == 3
    assert len(row.event_interpretations) == 1
    assert row.event_interpretations[0]["track_id"] == 12
    assert len(row.epistemic_claims) == 2
    assert row.epistemic_claims[0]["predicate"] == "stationary_duration"
    assert row.unverified_aspects == ["Signal state out of frame."]
    assert row.error_details["finish_reason"] == "STOP"

    # Confirm raw_response is eliminated from the model mapping
    mapper = inspect(SceneInterpretation)
    column_names = [col.key for col in mapper.columns]
    assert "raw_response" not in column_names


# ==============================================================================
# 8. Uniqueness Violation Session Recovery Test
# ==============================================================================

def test_uniqueness_violation_transaction_recovery(in_memory_db: Session):
    """Verifies that an IntegrityError from partial index does not corrupt the session after rollback."""
    _, job = _create_fixture_asset_and_job(in_memory_db)

    # 1. Insert first active interpretation
    interp1 = SceneInterpretation(
        id="interp_tx_01",
        job_id=job.id,
        status=SceneInterpretationStatus.QUEUED,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
    )
    in_memory_db.add(interp1)
    in_memory_db.commit()

    # 2. Attempt duplicate active interpretation
    interp2 = SceneInterpretation(
        id="interp_tx_02",
        job_id=job.id,
        status=SceneInterpretationStatus.QUEUED,
        provider_type="gemini",
        model_name="gemini-3.8-flash",
    )
    in_memory_db.add(interp2)
    with pytest.raises(IntegrityError):
        in_memory_db.commit()

    # 3. Rollback recovers the session cleanly
    in_memory_db.rollback()

    # 4. Verify initial record is intact and uncorrupted
    assert in_memory_db.get(SceneInterpretation, "interp_tx_01") is not None
    assert in_memory_db.get(SceneInterpretation, "interp_tx_02") is None

    # 5. Subsequent valid operation on the same session succeeds
    job.sequence_name = "UPDATED_SEQ_NAME"
    in_memory_db.commit()

    in_memory_db.refresh(job)
    assert job.sequence_name == "UPDATED_SEQ_NAME"


# ==============================================================================
# 9. Alembic Migration Upgrade and Downgrade Integration Test
# ==============================================================================

def test_alembic_scene_interpretation_migration_lifecycle():
    """Runs programmatic Alembic upgrade to head and downgrade to base on temp SQLite DB,
    and audits that index pruning and column requirements are strictly observed."""
    project_root = Path(__file__).resolve().parents[2]
    alembic_ini_path = project_root / "alembic.ini"
    assert alembic_ini_path.exists(), "alembic.ini missing"

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        tmp_db_path = Path(tmp_file.name)

    try:
        temp_db_url = f"sqlite:///{tmp_db_path.as_posix()}"
        alembic_cfg = Config(str(alembic_ini_path))
        alembic_cfg.set_main_option("sqlalchemy.url", temp_db_url)

        # 1. Upgrade to head
        command.upgrade(alembic_cfg, "head")

        # Verify tables and schema on migrated database
        eng = create_db_engine(temp_db_url)
        with eng.connect() as conn:
            inspector = inspect(conn)
            tables = inspector.get_table_names()
            assert "video_assets" in tables
            assert "processing_jobs" in tables
            assert "scene_interpretations" in tables

            # Audit columns on scene_interpretations table
            cols = {col["name"]: col for col in inspector.get_columns("scene_interpretations")}
            assert "id" in cols
            assert "job_id" in cols
            assert "status" in cols
            assert "summary" in cols
            assert "environmental_context" in cols
            assert "scene_description" in cols
            assert "event_interpretations" in cols
            assert "epistemic_claims" in cols
            assert "unverified_aspects" in cols
            assert "evidence_package_id" in cols
            assert "error_code" in cols
            assert "error_message" in cols
            assert "error_details" in cols
            # Confirm removed columns are not in DB schema
            assert "video_id" not in cols
            assert "raw_response" not in cols

            # Audit indexes on scene_interpretations table
            indexes = inspector.get_indexes("scene_interpretations")
            idx_names = {idx["name"]: idx for idx in indexes}
            # The 3 approved indexes must exist
            assert "uix_active_interpretation_per_job" in idx_names
            assert idx_names["uix_active_interpretation_per_job"]["unique"] in (1, True)
            assert "ix_scene_interpretations_job_created" in idx_names
            assert "ix_scene_interpretations_status_created" in idx_names
            # Redundant indexes must NOT exist
            assert "ix_scene_interpretations_id" not in idx_names
            assert "ix_scene_interpretations_job_id" not in idx_names
            assert "ix_scene_interpretations_video_id" not in idx_names
            assert "ix_scene_interpretations_status" not in idx_names
            assert "ix_scene_interpretations_created_at" not in idx_names

        # 2. Test constraint enforcement on the Alembic-migrated schema
        TestingSession = sessionmaker(bind=eng, class_=Session, expire_on_commit=False)
        with TestingSession() as session:
            v = VideoAsset(
                id="vid_mig_01",
                filename="mig.mp4",
                storage_path="/data/mig.mp4",
                file_size_bytes=1000,
            )
            j = ProcessingJob(
                id="job_mig_01",
                video_id="vid_mig_01",
                sequence_name="SEQ_MIG",
                status=JobStatus.COMPLETED,
            )
            session.add(v)
            session.add(j)
            session.commit()

            # Insert first QUEUED
            s1 = SceneInterpretation(
                id="interp_m_01",
                job_id="job_mig_01",
                status=SceneInterpretationStatus.QUEUED,
                provider_type="gemini",
                model_name="gemini-3.8-flash",
            )
            session.add(s1)
            session.commit()

            # Insert second QUEUED -> IntegrityError
            s2 = SceneInterpretation(
                id="interp_m_02",
                job_id="job_mig_01",
                status=SceneInterpretationStatus.QUEUED,
                provider_type="gemini",
                model_name="gemini-3.8-flash",
            )
            session.add(s2)
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()

        # 3. Downgrade to previous revision (ccecc71bbc1f)
        command.downgrade(alembic_cfg, "ccecc71bbc1f")

        with eng.connect() as conn:
            inspector = inspect(conn)
            tables_mid = inspector.get_table_names()
            assert "scene_interpretations" not in tables_mid
            assert "processing_jobs" in tables_mid
            assert "video_assets" in tables_mid

        # 4. Downgrade to base
        command.downgrade(alembic_cfg, "base")

        with eng.connect() as conn:
            inspector = inspect(conn)
            tables_base = inspector.get_table_names()
            assert "scene_interpretations" not in tables_base
            assert "processing_jobs" not in tables_base
            assert "video_assets" not in tables_base

        eng.dispose()
    finally:
        if tmp_db_path.exists():
            try:
                tmp_db_path.unlink()
            except PermissionError:
                pass
