"""
Concrete SQLAlchemy-backed Job and Video Asset repository implementation.

Preserves the JobRepository protocol boundary while providing full database persistence,
relationship loading, and error handling for SQLite and PostgreSQL.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Optional, Protocol, Tuple, Any
from sqlalchemy import select, update
from sqlalchemy.orm import Session, joinedload

from backend.app.models.job import ProcessingJob
from backend.app.models.video import VideoAsset
from backend.app.schemas.job import (
    JobStatus,
    ProcessingJobResponseSchema,
    ProcessingErrorSchema,
)


class JobRepository(Protocol):
    """Protocol defining the repository contract for job persistence."""

    def save(self, job: ProcessingJobResponseSchema) -> ProcessingJobResponseSchema:
        """Persists or updates a job entity."""
        ...

    def get_by_id(self, job_id: str) -> Optional[ProcessingJobResponseSchema]:
        """Retrieves a job by its unique identifier."""
        ...

    def list_jobs(self, limit: int = 50, offset: int = 0) -> List[ProcessingJobResponseSchema]:
        """Lists jobs with pagination."""
        ...


class SQLAlchemyJobRepository:
    """
    Concrete SQLAlchemy 2.0 repository managing VideoAsset and ProcessingJob entities.
    """

    def __init__(self, session: Session):
        self.session = session

    # --------------------------------------------------------------------------
    # VideoAsset Operations
    # --------------------------------------------------------------------------

    def create_video_asset(
        self,
        video_id: str,
        filename: str,
        storage_path: str,
        file_size_bytes: int,
        content_type: str = "video/mp4",
        duration_seconds: Optional[float] = None,
        fps: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> VideoAsset:
        """Creates and persists a new VideoAsset."""
        asset = VideoAsset(
            id=video_id,
            filename=filename,
            storage_path=storage_path,
            content_type=content_type,
            file_size_bytes=file_size_bytes,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
        )
        self.session.add(asset)
        self.session.flush()
        return asset

    def get_video_asset(self, video_id: str) -> Optional[VideoAsset]:
        """Retrieves a VideoAsset by ID."""
        stmt = select(VideoAsset).where(VideoAsset.id == video_id)
        return self.session.scalars(stmt).first()

    def list_video_assets(self, limit: int = 50, offset: int = 0) -> List[VideoAsset]:
        """Retrieves a list of VideoAssets ordered by creation timestamp descending."""
        stmt = (
            select(VideoAsset)
            .order_by(VideoAsset.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.scalars(stmt).all())

    # --------------------------------------------------------------------------
    # ProcessingJob Model Operations
    # --------------------------------------------------------------------------

    def create_job(
        self,
        job_id: str,
        video_id: str,
        sequence_name: str,
        status: JobStatus = JobStatus.QUEUED,
        config_metadata: Optional[dict] = None,
    ) -> ProcessingJob:
        """Creates and persists a new ProcessingJob ORM model."""
        job = ProcessingJob(
            id=job_id,
            video_id=video_id,
            sequence_name=sequence_name,
            status=status,
            config_metadata=config_metadata,
        )
        self.session.add(job)
        self.session.flush()
        return job

    def get_job_model(self, job_id: str, load_relationship: bool = False) -> Optional[ProcessingJob]:
        """Retrieves raw ProcessingJob model, optionally eagerly loading video_asset."""
        stmt = select(ProcessingJob).where(ProcessingJob.id == job_id)
        if load_relationship:
            stmt = stmt.options(joinedload(ProcessingJob.video_asset))
        return self.session.scalars(stmt).first()

    def update_job_status(
        self,
        job_id: str,
        new_status: JobStatus,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        duration_seconds: Optional[float] = None,
        error: Optional[ProcessingErrorSchema] = None,
    ) -> Optional[ProcessingJob]:
        """Updates job status and associated timestamps/errors in the database."""
        job = self.get_job_model(job_id)
        if not job:
            return None

        job.status = new_status
        if started_at is not None:
            job.started_at = started_at
        if completed_at is not None:
            job.completed_at = completed_at
        if duration_seconds is not None:
            job.duration_seconds = duration_seconds

        if error is not None:
            job.error_code = error.error_code
            job.error_message = error.message
            job.error_details = error.details

        self.session.flush()
        return job

    # --------------------------------------------------------------------------
    # JobRepository Protocol Implementation
    # --------------------------------------------------------------------------

    def save(self, job: ProcessingJobResponseSchema) -> ProcessingJobResponseSchema:
        """
        Saves or updates a job from its Pydantic representation.
        """
        existing = self.get_job_model(job.job_id)
        if existing:
            existing.status = job.status
            existing.started_at = job.started_at
            existing.completed_at = job.completed_at
            existing.duration_seconds = job.duration_seconds
            if job.error:
                existing.error_code = job.error.error_code
                existing.error_message = job.error.message
                existing.error_details = job.error.details
            self.session.flush()
            return self._model_to_schema(existing)
        else:
            # Create new
            new_job = ProcessingJob(
                id=job.job_id,
                video_id=job.video_id,
                sequence_name=job.sequence_name,
                status=job.status,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                duration_seconds=job.duration_seconds,
            )
            if job.error:
                new_job.error_code = job.error.error_code
                new_job.error_message = job.error.message
                new_job.error_details = job.error.details
            self.session.add(new_job)
            self.session.flush()
            return self._model_to_schema(new_job)

    def get_by_id(self, job_id: str) -> Optional[ProcessingJobResponseSchema]:
        """Retrieves a job by ID formatted as ProcessingJobResponseSchema."""
        model = self.get_job_model(job_id)
        if not model:
            return None
        return self._model_to_schema(model)

    def list_jobs(self, limit: int = 50, offset: int = 0) -> List[ProcessingJobResponseSchema]:
        """Lists jobs ordered by created_at descending."""
        stmt = (
            select(ProcessingJob)
            .order_by(ProcessingJob.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        models = self.session.scalars(stmt).all()
        return [self._model_to_schema(m) for m in models]

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def _model_to_schema(model: ProcessingJob) -> ProcessingJobResponseSchema:
        """Maps ORM model to Pydantic contract."""
        error_schema = None
        if model.error_code or model.error_message:
            error_schema = ProcessingErrorSchema(
                error_code=model.error_code or "UNKNOWN_ERROR",
                message=model.error_message or "",
                details=model.error_details,
                occurred_at=model.completed_at or model.created_at,
            )

        return ProcessingJobResponseSchema(
            job_id=model.id,
            video_id=model.video_id,
            sequence_name=model.sequence_name,
            status=model.status,
            created_at=model.created_at,
            started_at=model.started_at,
            completed_at=model.completed_at,
            duration_seconds=model.duration_seconds,
            error=error_schema,
        )
