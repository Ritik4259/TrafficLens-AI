"""
Job domain service managing lifecycle state transitions and execution flow.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Optional
import uuid

from backend.app.repositories.job_repository import JobRepository
from backend.app.schemas import (
    JobStatus,
    ProcessingJobCreateSchema,
    ProcessingJobResponseSchema,
    ProcessingErrorSchema,
    validate_job_status_transition,
)


class JobService:
    """
    Coordinates processing job lifecycles and enforces state machine transitions.
    Persists state via a JobRepository (e.g. SQLAlchemyJobRepository).
    """

    def __init__(self, repository: Optional[JobRepository] = None):
        self.repository = repository
        self._in_memory_jobs: Dict[str, ProcessingJobResponseSchema] = {}

    def create_job(self, payload: ProcessingJobCreateSchema) -> ProcessingJobResponseSchema:
        """Creates and persists a new job in QUEUED status."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        seq_name = payload.sequence_name or f"seq_{payload.video_id}"

        job = ProcessingJobResponseSchema(
            job_id=job_id,
            video_id=payload.video_id,
            sequence_name=seq_name,
            status=JobStatus.QUEUED,
            created_at=now,
            started_at=None,
            completed_at=None,
            duration_seconds=None,
            error=None,
        )

        if self.repository:
            return self.repository.save(job)

        self._in_memory_jobs[job_id] = job
        return job

    def transition_job_status(
        self,
        job_id: str,
        target_status: JobStatus,
        error: Optional[ProcessingErrorSchema] = None,
    ) -> ProcessingJobResponseSchema:
        """
        Transitions a job to a new status following valid state transitions.
        Persists the change via the configured repository.
        Raises InvalidJobStatusTransitionError if the transition is disallowed.
        """
        if self.repository:
            job = self.repository.get_by_id(job_id)
        else:
            job = self._in_memory_jobs.get(job_id)

        if not job:
            raise KeyError(f"Job {job_id} not found")

        # Enforce state transition rules
        validate_job_status_transition(job.status, target_status)

        now = datetime.now(timezone.utc)
        started_at = job.started_at
        completed_at = job.completed_at
        duration = job.duration_seconds

        if target_status == JobStatus.PROCESSING and started_at is None:
            started_at = now
        elif target_status in {JobStatus.COMPLETED, JobStatus.FAILED}:
            completed_at = now
            if started_at:
                duration = max(0.0, (completed_at - started_at).total_seconds())

        updated_job = job.model_copy(
            update={
                "status": target_status,
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": duration,
                "error": error if target_status == JobStatus.FAILED else None,
            }
        )

        if self.repository:
            return self.repository.save(updated_job)

        self._in_memory_jobs[job_id] = updated_job
        return updated_job

    def get_job(self, job_id: str) -> Optional[ProcessingJobResponseSchema]:
        """Retrieves a job by ID."""
        if self.repository:
            return self.repository.get_by_id(job_id)
        return self._in_memory_jobs.get(job_id)
