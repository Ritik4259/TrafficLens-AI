"""
SQLAlchemy model for multimodal scene interpretations.
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from sqlalchemy import Enum as SQLEnum, Float, ForeignKey, Index, Integer, JSON, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UTCDateTime

if TYPE_CHECKING:
    from .job import ProcessingJob


class SceneInterpretationStatus(str, Enum):
    """
    Lifecycle states for a persisted scene interpretation task.
    
    Note: NOT_REQUESTED is a virtual client-facing status representing the absence
    of an interpretation record and is never stored in the database.
    QUEUED and PROCESSING represent active non-terminal states.
    COMPLETED and FAILED are terminal, immutable states.
    """
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SceneInterpretation(Base):
    """
    Represents an asynchronous multimodal scene understanding execution attempt for a ProcessingJob.
    """
    __tablename__ = "scene_interpretations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("processing_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[SceneInterpretationStatus] = mapped_column(
        SQLEnum(
            SceneInterpretationStatus,
            name="scene_interpretation_status_enum",
            native_enum=False,
        ),
        nullable=False,
        default=SceneInterpretationStatus.QUEUED,
    )

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Provider and model accounting
    provider_type: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Structured interpretation output fields (no base64 images stored in database)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    environmental_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    scene_description: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    event_interpretations: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    epistemic_claims: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    unverified_aspects: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    evidence_package_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Error details if status == FAILED
    error_code: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_message: Mapped[Optional[Text]] = mapped_column(Text, nullable=True)
    error_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Relationships
    job: Mapped["ProcessingJob"] = relationship("ProcessingJob", back_populates="scene_interpretations")

    @property
    def video_id(self) -> Optional[str]:
        """Resolves parent video_id from associated ProcessingJob."""
        return self.job.video_id if self.job else None

    __table_args__ = (
        Index("ix_scene_interpretations_job_created", "job_id", "created_at"),
        Index("ix_scene_interpretations_status_created", "status", "created_at"),
        Index(
            "uix_active_interpretation_per_job",
            "job_id",
            unique=True,
            sqlite_where=text("status IN ('QUEUED', 'PROCESSING')"),
            postgresql_where=text("status IN ('QUEUED', 'PROCESSING')"),
        ),
    )

    def __repr__(self) -> str:
        return f"<SceneInterpretation id={self.id} job_id={self.job_id} status={self.status.value}>"
