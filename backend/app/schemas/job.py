"""
Video processing job schemas and lifecycle state machine for TrafficLens-AI.

Defines the asynchronous job lifecycle (QUEUED -> PROCESSING -> COMPLETED/FAILED),
allowed status transitions, video upload contracts, and error structures.
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, model_validator

from .common import CountingLineConfigSchema, ROIConfigSchema


class JobStatus(str, Enum):
    """Lifecycle statuses for asynchronous video processing jobs."""
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# State machine allowed transitions
ALLOWED_JOB_TRANSITIONS: Dict[JobStatus, Set[JobStatus]] = {
    JobStatus.QUEUED: {JobStatus.PROCESSING, JobStatus.FAILED},
    JobStatus.PROCESSING: {JobStatus.COMPLETED, JobStatus.FAILED},
    JobStatus.COMPLETED: set(),  # Terminal state
    JobStatus.FAILED: set(),     # Terminal state
}


class InvalidJobStatusTransitionError(ValueError):
    """Raised when an illegal job status transition is attempted."""

    def __init__(self, current_status: JobStatus, target_status: JobStatus):
        super().__init__(
            f"Cannot transition job status from {current_status.value} to {target_status.value}. "
            f"Allowed target states: {[s.value for s in ALLOWED_JOB_TRANSITIONS.get(current_status, set())]}."
        )
        self.current_status = current_status
        self.target_status = target_status


def validate_job_status_transition(current_status: JobStatus, target_status: JobStatus) -> None:
    """Validates that a status transition is permitted by the job lifecycle."""
    allowed = ALLOWED_JOB_TRANSITIONS.get(current_status, set())
    if target_status not in allowed:
        raise InvalidJobStatusTransitionError(current_status, target_status)


class VideoMetadataSchema(BaseModel):
    """Metadata describing an uploaded video file."""
    video_id: str = Field(..., description="Unique video asset identifier")
    filename: str = Field(..., description="Original uploaded filename")
    content_type: str = Field(..., description="MIME content type (e.g., video/mp4)")
    file_size_bytes: int = Field(..., ge=0, description="File size on disk in bytes")
    duration_seconds: Optional[float] = Field(default=None, gt=0.0, description="Video duration in seconds")
    fps: Optional[float] = Field(default=None, gt=0.0, description="Video frame rate in frames per second")
    width: Optional[int] = Field(default=None, gt=0, description="Frame width in pixels")
    height: Optional[int] = Field(default=None, gt=0, description="Frame height in pixels")
    frame_count: Optional[int] = Field(default=None, ge=0, description="Total video frames in asset")
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC upload timestamp",
    )


class VideoUploadResponseSchema(BaseModel):
    """Response returned upon successful video upload."""
    video_id: str = Field(..., description="Generated video identifier")
    metadata: VideoMetadataSchema = Field(..., description="Extracted video metadata")
    message: str = Field(default="Video uploaded successfully and ready for processing.")


class ProcessingErrorSchema(BaseModel):
    """Structured processing failure details."""
    error_code: str = Field(..., description="Canonical error code string")
    message: str = Field(..., description="Human-readable error description")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional contextual metadata or stack trace")
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC failure timestamp",
    )


class ProcessingJobCreateSchema(BaseModel):
    """Payload to create and queue an asynchronous video processing job."""
    video_id: Optional[str] = Field(default=None, description="Identifier of the uploaded video to process")
    sequence_name: Optional[str] = Field(default=None, description="Optional logical sequence name")
    lines: Optional[List[CountingLineConfigSchema]] = Field(
        default=None,
        description="Custom virtual counting lines. If omitted, default crosslines are applied.",
    )
    rois: Optional[List[ROIConfigSchema]] = Field(
        default=None,
        description="Custom polygonal regions of interest. If omitted, default zones are applied.",
    )


class ProcessingJobResponseSchema(BaseModel):
    """Response schema describing job state and timestamps."""
    job_id: str = Field(..., description="Unique job identifier")
    video_id: str = Field(..., description="Target video identifier")
    sequence_name: str = Field(..., description="Logical sequence name")
    status: JobStatus = Field(..., description="Current lifecycle state (QUEUED, PROCESSING, COMPLETED, FAILED)")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Timestamp when processing began")
    completed_at: Optional[datetime] = Field(default=None, description="Timestamp when processing finished")
    duration_seconds: Optional[float] = Field(default=None, description="Total processing wall-clock time in seconds")
    error: Optional[ProcessingErrorSchema] = Field(default=None, description="Error details if status is FAILED")


class JobArtifactItemSchema(BaseModel):
    """Metadata describing a single job-generated artifact file without exposing server filesystem paths."""
    name: str = Field(..., description="Logical artifact key (e.g. analytics_json, telemetry_csv, annotated_video)")
    filename: str = Field(..., description="Clean artifact filename without directory path")
    artifact_type: str = Field(..., description="Data format category: 'json', 'csv', 'video'")
    is_available: bool = Field(..., description="True if the artifact file physically exists on server storage")
    file_size_bytes: Optional[int] = Field(default=None, description="Artifact file size in bytes if available")
    serving_url: Optional[str] = Field(default=None, description="Relative API serving URL if supported")


class JobArtifactsResponseSchema(BaseModel):
    """List of all available artifacts associated with a processing job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Processing job status")
    artifacts: List[JobArtifactItemSchema] = Field(default_factory=list, description="List of generated artifacts")

