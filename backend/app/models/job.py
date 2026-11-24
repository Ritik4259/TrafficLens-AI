"""
SQLAlchemy model for video processing jobs.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from sqlalchemy import Enum, Float, ForeignKey, Index, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.schemas.job import JobStatus
from .base import Base, UTCDateTime

if TYPE_CHECKING:
    from .video import VideoAsset
    from .scene_interpretation import SceneInterpretation
    from .incident_report import IncidentReport


class ProcessingJob(Base):
    """
    Represents an asynchronous video processing job associated with a VideoAsset.
    """
    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, index=True)
    video_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("video_assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status_enum", native_enum=False),
        nullable=False,
        default=JobStatus.QUEUED,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Error details if status == FAILED
    error_code: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Job configuration metadata (virtual lines, ROIs, execution parameters)
    config_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Relationships
    video_asset: Mapped["VideoAsset"] = relationship("VideoAsset", back_populates="jobs")
    scene_interpretations: Mapped[List["SceneInterpretation"]] = relationship(
        "SceneInterpretation",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="desc(SceneInterpretation.created_at)",
    )
    incident_reports: Mapped[List["IncidentReport"]] = relationship(
        "IncidentReport",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="desc(IncidentReport.created_at)",
    )

    __table_args__ = (
        Index("ix_processing_jobs_status_created_at", "status", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<ProcessingJob id={self.id} video_id={self.video_id} status={self.status.value}>"
