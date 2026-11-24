"""
Tests for asynchronous job processing lifecycle and state machine transitions.
"""

from __future__ import annotations
import pytest

from backend.app.schemas import (
    JobStatus,
    ProcessingJobCreateSchema,
    ProcessingErrorSchema,
    InvalidJobStatusTransitionError,
    validate_job_status_transition,
)
from backend.app.services.job_service import JobService


def test_allowed_job_lifecycle_progression():
    """Validates the standard happy path: QUEUED -> PROCESSING -> COMPLETED."""
    service = JobService()
    job = service.create_job(ProcessingJobCreateSchema(video_id="vid_001", sequence_name="MVI_20011"))
    assert job.status == JobStatus.QUEUED
    assert job.started_at is None
    assert job.completed_at is None

    # Step 1: Transition to PROCESSING
    job_proc = service.transition_job_status(job.job_id, JobStatus.PROCESSING)
    assert job_proc.status == JobStatus.PROCESSING
    assert job_proc.started_at is not None
    assert job_proc.completed_at is None

    # Step 2: Transition to COMPLETED
    job_comp = service.transition_job_status(job.job_id, JobStatus.COMPLETED)
    assert job_comp.status == JobStatus.COMPLETED
    assert job_comp.completed_at is not None
    assert job_comp.duration_seconds is not None
    assert job_comp.duration_seconds >= 0.0


def test_allowed_job_failure_paths():
    """Validates failure transitions: QUEUED -> FAILED and PROCESSING -> FAILED."""
    service = JobService()

    # Case 1: Failure during QUEUED (e.g. invalid video format)
    job1 = service.create_job(ProcessingJobCreateSchema(video_id="vid_corrupt"))
    err1 = ProcessingErrorSchema(error_code="INVALID_VIDEO", message="Corrupt header")
    job1_failed = service.transition_job_status(job1.job_id, JobStatus.FAILED, error=err1)
    assert job1_failed.status == JobStatus.FAILED
    assert job1_failed.error is not None
    assert job1_failed.error.error_code == "INVALID_VIDEO"

    # Case 2: Failure during PROCESSING (e.g. out of memory / pipeline error)
    job2 = service.create_job(ProcessingJobCreateSchema(video_id="vid_002"))
    service.transition_job_status(job2.job_id, JobStatus.PROCESSING)
    err2 = ProcessingErrorSchema(error_code="INFERENCE_ERROR", message="Inference timed out")
    job2_failed = service.transition_job_status(job2.job_id, JobStatus.FAILED, error=err2)
    assert job2_failed.status == JobStatus.FAILED
    assert job2_failed.error is not None
    assert job2_failed.error.error_code == "INFERENCE_ERROR"


@pytest.mark.parametrize(
    "source,target",
    [
        (JobStatus.QUEUED, JobStatus.COMPLETED),      # Cannot skip processing
        (JobStatus.COMPLETED, JobStatus.PROCESSING),   # Cannot restart finished job
        (JobStatus.COMPLETED, JobStatus.FAILED),       # Terminal state
        (JobStatus.COMPLETED, JobStatus.QUEUED),       # Terminal state
        (JobStatus.FAILED, JobStatus.PROCESSING),      # Terminal state
        (JobStatus.FAILED, JobStatus.COMPLETED),       # Terminal state
        (JobStatus.FAILED, JobStatus.QUEUED),          # Terminal state
    ],
)
def test_prohibited_job_status_transitions(source: JobStatus, target: JobStatus):
    """Ensures invalid state transitions raise InvalidJobStatusTransitionError."""
    with pytest.raises(InvalidJobStatusTransitionError) as exc_info:
        validate_job_status_transition(source, target)
    assert f"Cannot transition job status from {source.value} to {target.value}" in str(exc_info.value)
