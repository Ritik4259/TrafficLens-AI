"""
Configuration dataclasses for traffic analytics components.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class CountingLineConfig:
    """Configuration for a virtual counting line."""
    line_id: str
    name: str
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    positive_direction_label: str = "downstream"
    negative_direction_label: str = "upstream"
    cooldown_frames: int = 15
    min_track_age: int = 2


@dataclass
class ROIConfig:
    """Configuration for a polygonal road region of interest."""
    roi_id: str
    name: str
    polygon: List[Tuple[float, float]]
    description: str = ""


@dataclass
class FlowConfig:
    """Configuration for traffic flow calculation."""
    window_size_seconds: float = 60.0
    step_size_seconds: float = 10.0


@dataclass
class DensityConfig:
    """Configuration for ROI traffic density estimation."""
    method: str = "box_intersection"  # "box_intersection" or "centroid"


@dataclass
class CongestionConfig:
    """Thresholds for rule-based congestion classification."""
    low_occupancy_thresh: float = 0.15
    high_occupancy_thresh: float = 0.40
    low_speed_thresh_px_s: float = 25.0
    stopped_ratio_thresh: float = 0.35
    min_vehicles_for_congestion: int = 2


@dataclass
class StoppedVehicleConfig:
    """Configuration for stopped vehicle event detection."""
    min_duration_seconds: float = 3.0
    max_displacement_px: float = 15.0
    sliding_window_frames: int = 75  # 3.0s * 25.0 fps


@dataclass
class SpeedConfig:
    """Configuration for vehicle speed estimation."""
    fps: float = 25.0
    smoothing_window_frames: int = 5
    is_calibrated: bool = False
    homography_matrix: Optional[List[List[float]]] = None


@dataclass
class AnalyticsConfig:
    """Root configuration for the complete traffic analytics pipeline."""
    fps: float = 25.0
    frame_width: float = 960.0
    frame_height: float = 540.0
    lines: List[CountingLineConfig] = field(default_factory=list)
    rois: List[ROIConfig] = field(default_factory=list)
    flow: FlowConfig = field(default_factory=FlowConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    congestion: CongestionConfig = field(default_factory=CongestionConfig)
    stopped_vehicle: StoppedVehicleConfig = field(default_factory=StoppedVehicleConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
