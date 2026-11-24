"""
SQLAlchemy model for uploaded video assets.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import BigInteger, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UTCDateTime

if TYPE_CHECKING:
    from .job import ProcessingJob


class VideoAsset(Base):
    """
    Represents an uploaded video file registered in the system.
    """
    __tablename__ = "video_assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False, default="video/mp4")
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    jobs: Mapped[List["ProcessingJob"]] = relationship(
        "ProcessingJob",
        back_populates="video_asset",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_video_assets_created_at", "created_at"),
        Index("ix_video_assets_filename", "filename"),
    )

    def __repr__(self) -> str:
        return f"<VideoAsset id={self.id} filename={self.filename} size={self.file_size_bytes}>"
