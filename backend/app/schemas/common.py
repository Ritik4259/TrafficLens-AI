"""
Common and geometric schemas for TrafficLens-AI API contracts.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class Point2DSchema(BaseModel):
    """2D Cartesian coordinates (x, y) in pixel space."""
    x: float = Field(..., description="Horizontal pixel coordinate")
    y: float = Field(..., description="Vertical pixel coordinate")


class BBoxSchema(BaseModel):
    """Bounding box in absolute frame coordinates."""
    left: float = Field(..., ge=0.0, description="Bounding box x-coordinate")
    top: float = Field(..., ge=0.0, description="Bounding box y-coordinate")
    width: float = Field(..., gt=0.0, description="Bounding box width in pixels")
    height: float = Field(..., gt=0.0, description="Bounding box height in pixels")


class CountingLineConfigSchema(BaseModel):
    """Configuration for a virtual counting line."""
    line_id: str = Field(..., min_length=1, description="Unique virtual line identifier")
    name: str = Field(..., min_length=1, description="Human-readable line label")
    p1: Tuple[float, float] = Field(..., description="Start coordinate (x1, y1)")
    p2: Tuple[float, float] = Field(..., description="End coordinate (x2, y2)")
    positive_direction_label: str = Field(default="downstream", description="Label for positive normal crossings")
    negative_direction_label: str = Field(default="upstream", description="Label for negative normal crossings")
    cooldown_frames: int = Field(default=15, ge=1, description="Debounce cooldown window in frames")
    min_track_age: int = Field(default=2, ge=1, description="Minimum track frames before crossing is counted")


class ROIConfigSchema(BaseModel):
    """Configuration for a polygonal road region of interest."""
    roi_id: str = Field(..., min_length=1, description="Unique ROI identifier")
    name: str = Field(..., min_length=1, description="Human-readable region name")
    polygon: List[Tuple[float, float]] = Field(..., min_length=3, description="List of vertices [(x, y), ...]")
    description: str = Field(default="", description="Optional description of the region")
