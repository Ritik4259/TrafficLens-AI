"""
Tests for SQLAlchemy 2.0 database models, sessions, persistence, and Alembic migrations.
"""

from __future__ import annotations
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from alembic import command
from alembic.config import Config

from backend.app.core.config import Settings
from backend.app.db.session import create_db_engine, init_db, get_db
from backend.app.models.base import Base
from backend.app.models.job import ProcessingJob
from backend.app.models.video import VideoAsset
from backend.app.repositories.job_repository import SQLAlchemyJobRepository
from backend.app.schemas.job import (
    JobStatus,
    ProcessingJobCreateSchema,
    ProcessingErrorSchema,
    InvalidJobStatusTransitionError,
)
from backend.app.services.job_service import JobService


@pytest.fixture
def in_memory_db():
    """Provides an isolated in-memory SQLite database session for each test."""
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


# ==============================================================================
# 1. Database Initialization and Session Lifecycle Tests
# ==============================================================================

def test_db_initialization_and_session_lifecycle():
    """Validates session commit, rollback, and lifecycle in isolated database."""
    test_engine = create_db_engine("sqlite:///:memory:")
    init_db(test_engine)
    TestingSession = sessionmaker(bind=test_engine, class_=Session, expire_on_commit=False)

    # Test commit path
    with TestingSession() as session:
        asset = VideoAsset(
            id="vid_test_01",
            filename="traffic.mp4",
            storage_path="/storage/traffic.mp4",
            file_size_bytes=1000,
        )
        session.add(asset)
        session.commit()

    # Query in fresh session
    with TestingSession() as session:
        queried = session.get(VideoAsset, "vid_test_01")
        assert queried is not None
        assert queried.filename == "traffic.mp4"

    # Test rollback on exception
    with pytest.raises(RuntimeError):
        with TestingSession() as session:
            asset2 = VideoAsset(
                id="vid_test_02",
                filename="traffic2.mp4",
                storage_path="/storage/traffic2.mp4",
                file_size_bytes=2000,
            )
            session.add(asset2)
            session.flush()
            raise RuntimeError("Simulated crash")

    with TestingSession() as session:
        assert session.get(VideoAsset, "vid_test_02") is None


# ==============================================================================
# 2. Model Creation, Relationships, and Eager Loading Tests
# ==============================================================================

def test_video_asset_and_job_creation(in_memory_db: Session):
    """Validates VideoAsset and ProcessingJob creation and default values."""
    repo = SQLAlchemyJobRepository(in_memory_db)

    asset = repo.create_video_asset(
        video_id="vid_100",
        filename="highway.mp4",
        storage_path="/data/highway.mp4",
        file_size_bytes=5242880,
        content_type="video/mp4",
        duration_seconds=30.5,
        fps=25.0,
        width=1920,
        height=1080,
    )
    in_memory_db.commit()

    assert asset.id == "vid_100"
    assert asset.width == 1920
    assert asset.created_at.tzinfo is not None

    job = repo.create_job(
        job_id="job_100",
        video_id="vid_100",
        sequence_name="MVI_20011",
        status=JobStatus.QUEUED,
        config_metadata={"cooldown": 15},
    )
    in_memory_db.commit()

    assert job.id == "job_100"
    assert job.status == JobStatus.QUEUED
    assert job.config_metadata == {"cooldown": 15}
    assert job.created_at.tzinfo is not None


def test_job_retrieval_and_relationship_loading(in_memory_db: Session):
    """Validates bidirectional relationship between VideoAsset and ProcessingJob."""
    repo = SQLAlchemyJobRepository(in_memory_db)

    asset = repo.create_video_asset(
        video_id="vid_rel",
        filename="intersection.mp4",
        storage_path="/data/intersection.mp4",
        file_size_bytes=1000000,
    )
    job1 = repo.create_job(job_id="job_rel_1", video_id="vid_rel", sequence_name="SEQ_1")
    job2 = repo.create_job(job_id="job_rel_2", video_id="vid_rel", sequence_name="SEQ_2")
    in_memory_db.commit()

    # Eager load relationship from job side
    loaded_job = repo.get_job_model("job_rel_1", load_relationship=True)
    assert loaded_job is not None
    assert loaded_job.video_asset is not None
    assert loaded_job.video_asset.id == "vid_rel"
    assert loaded_job.video_asset.filename == "intersection.mp4"

    # Load from asset side
    loaded_asset = repo.get_video_asset("vid_rel")
    assert loaded_asset is not None
    assert len(loaded_asset.jobs) == 2
    assert {j.id for j in loaded_asset.jobs} == {"job_rel_1", "job_rel_2"}


# ==============================================================================
# 3. Persisted Status Transitions and Rejection Tests
# ==============================================================================

def test_persisted_status_transitions(in_memory_db: Session):
    """Validates that valid status transitions persist cleanly to database rows."""
    repo = SQLAlchemyJobRepository(in_memory_db)
    service = JobService(repository=repo)

    # 1. Register asset & create job
    repo.create_video_asset(
        video_id="vid_lifecycle",
        filename="test.mp4",
        storage_path="/data/test.mp4",
        file_size_bytes=1024,
    )
    in_memory_db.commit()

    job_schema = service.create_job(
        ProcessingJobCreateSchema(video_id="vid_lifecycle", sequence_name="SEQ_LIFECYCLE")
    )
    in_memory_db.commit()
    job_id = job_schema.job_id
    assert job_schema.status == JobStatus.QUEUED

    # 2. Transition to PROCESSING
    job_proc = service.transition_job_status(job_id, JobStatus.PROCESSING)
    in_memory_db.commit()
    assert job_proc.status == JobStatus.PROCESSING
    assert job_proc.started_at is not None

    # Check database row directly
    db_job1 = in_memory_db.get(ProcessingJob, job_id)
    assert db_job1.status == JobStatus.PROCESSING
    assert db_job1.started_at is not None
    assert db_job1.completed_at is None

    # 3. Transition to COMPLETED
    job_comp = service.transition_job_status(job_id, JobStatus.COMPLETED)
    in_memory_db.commit()
    assert job_comp.status == JobStatus.COMPLETED
    assert job_comp.completed_at is not None
    assert job_comp.duration_seconds is not None

    # Check database row directly
    db_job2 = in_memory_db.get(ProcessingJob, job_id)
    assert db_job2.status == JobStatus.COMPLETED
    assert db_job2.completed_at is not None
    assert db_job2.duration_seconds >= 0.0


def test_persisted_failure_transition(in_memory_db: Session):
    """Validates that failure details and error codes persist to database rows."""
    repo = SQLAlchemyJobRepository(in_memory_db)
    service = JobService(repository=repo)

    repo.create_video_asset(
        video_id="vid_err",
        filename="corrupt.mp4",
        storage_path="/data/corrupt.mp4",
        file_size_bytes=100,
    )
    in_memory_db.commit()

    job = service.create_job(ProcessingJobCreateSchema(video_id="vid_err"))
    in_memory_db.commit()

    err = ProcessingErrorSchema(
        error_code="VIDEO_DECODE_ERROR",
        message="Corrupt h264 bitstream",
        details={"frame": 42},
    )
    failed_job = service.transition_job_status(job.job_id, JobStatus.FAILED, error=err)
    in_memory_db.commit()

    assert failed_job.status == JobStatus.FAILED
    assert failed_job.error is not None
    assert failed_job.error.error_code == "VIDEO_DECODE_ERROR"

    # Query DB row directly
    row = in_memory_db.get(ProcessingJob, job.job_id)
    assert row.status == JobStatus.FAILED
    assert row.error_code == "VIDEO_DECODE_ERROR"
    assert row.error_message == "Corrupt h264 bitstream"
    assert row.error_details == {"frame": 42}


def test_invalid_transitions_and_missing_records(in_memory_db: Session):
    """Validates rejection of illegal transitions and non-existent job queries."""
    repo = SQLAlchemyJobRepository(in_memory_db)
    service = JobService(repository=repo)

    repo.create_video_asset(video_id="vid_inv", filename="v.mp4", storage_path="/v.mp4", file_size_bytes=10)
    in_memory_db.commit()
    job = service.create_job(ProcessingJobCreateSchema(video_id="vid_inv"))
    in_memory_db.commit()

    # Attempt illegal jump QUEUED -> COMPLETED
    with pytest.raises(InvalidJobStatusTransitionError):
        service.transition_job_status(job.job_id, JobStatus.COMPLETED)

    # Ensure DB row remained QUEUED
    row = in_memory_db.get(ProcessingJob, job.job_id)
    assert row.status == JobStatus.QUEUED

    # Non-existent job
    with pytest.raises(KeyError):
        service.transition_job_status("non_existent_job_id", JobStatus.PROCESSING)


# ==============================================================================
# 4. Constraints: Foreign Keys, Uniqueness, and Cascade Deletion
# ==============================================================================

def test_foreign_key_and_uniqueness_constraints(in_memory_db: Session):
    """Validates unique storage_path, foreign-key integrity, and cascade deletion."""
    repo = SQLAlchemyJobRepository(in_memory_db)

    # 1. Unique constraint on storage_path
    repo.create_video_asset(
        video_id="vid_u1",
        filename="u1.mp4",
        storage_path="/data/unique.mp4",
        file_size_bytes=100,
    )
    in_memory_db.commit()

    with pytest.raises(IntegrityError):
        repo.create_video_asset(
            video_id="vid_u2",
            filename="u2.mp4",
            storage_path="/data/unique.mp4",  # Duplicate storage_path
            file_size_bytes=200,
        )
        in_memory_db.commit()
    in_memory_db.rollback()

    # 2. Foreign-key constraint enforcement (PRAGMA foreign_keys=ON)
    with pytest.raises(IntegrityError):
        repo.create_job(
            job_id="job_orphan",
            video_id="non_existent_video_fk",  # Foreign key does not exist
            sequence_name="SEQ_ORPHAN",
        )
        in_memory_db.commit()
    in_memory_db.rollback()

    # 3. Cascade deletion
    asset = repo.create_video_asset(
        video_id="vid_cascade",
        filename="cascade.mp4",
        storage_path="/data/cascade.mp4",
        file_size_bytes=100,
    )
    job1 = repo.create_job(job_id="job_c1", video_id="vid_cascade", sequence_name="S1")
    job2 = repo.create_job(job_id="job_c2", video_id="vid_cascade", sequence_name="S2")
    in_memory_db.commit()

    assert in_memory_db.get(ProcessingJob, "job_c1") is not None
    assert in_memory_db.get(ProcessingJob, "job_c2") is not None

    # Delete parent VideoAsset
    in_memory_db.delete(asset)
    in_memory_db.commit()

    # Children must be cascade-deleted
    assert in_memory_db.get(ProcessingJob, "job_c1") is None
    assert in_memory_db.get(ProcessingJob, "job_c2") is None


# ==============================================================================
# 5. Alembic Upgrade and Downgrade Lifecycle Test
# ==============================================================================

def test_alembic_upgrade_and_downgrade_isolated():
    """Runs programmatic Alembic upgrade to head and downgrade to base on temp SQLite DB."""
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

        # Verify tables created
        eng = create_db_engine(temp_db_url)
        with eng.connect() as conn:
            from sqlalchemy import inspect
            inspector = inspect(conn)
            tables = inspector.get_table_names()
            assert "video_assets" in tables
            assert "processing_jobs" in tables
            assert "scene_interpretations" in tables
            assert "alembic_version" in tables

        # 2. Downgrade to base
        command.downgrade(alembic_cfg, "base")

        with eng.connect() as conn:
            inspector = inspect(conn)
            tables_after = inspector.get_table_names()
            assert "video_assets" not in tables_after
            assert "processing_jobs" not in tables_after
            assert "scene_interpretations" not in tables_after

        eng.dispose()
    finally:
        if tmp_db_path.exists():
            try:
                tmp_db_path.unlink()
            except PermissionError:
                pass
