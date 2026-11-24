"""
Structured analytics output models for TrafficLens-AI.

Defines strongly-typed dataclasses for line crossings, flow aggregations,
ROI densities, congestion states, stopped vehicle events, and sequence reports.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any


class CongestionLevel(str, Enum):
    """Discrete traffic congestion levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class CrossingEvent:
    """Represents a single vehicle track crossing a virtual counting line."""
    line_id: str
    track_id: int
    frame_idx: int
    timestamp_s: float
    direction: str
    crossing_point: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_id": self.line_id,
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
            "timestamp_s": round(self.timestamp_s, 3),
            "direction": self.direction,
            "crossing_point": [round(self.crossing_point[0], 2), round(self.crossing_point[1], 2)],
        }


@dataclass
class LineFlow:
    """Traffic flow measurement for a counting line over a time window."""
    line_id: str
    window_start_s: float
    window_end_s: float
    total_count: int
    flow_rate_vpm: float  # Vehicles per minute
    direction_counts: Dict[str, int] = field(default_factory=dict)
    direction_flows_vpm: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_id": self.line_id,
            "window_start_s": round(self.window_start_s, 2),
            "window_end_s": round(self.window_end_s, 2),
            "total_count": self.total_count,
            "flow_rate_vpm": round(self.flow_rate_vpm, 2),
            "direction_counts": dict(self.direction_counts),
            "direction_flows_vpm": {k: round(v, 2) for k, v in self.direction_flows_vpm.items()},
        }


@dataclass
class ROIDensity:
    """
    Traffic density measurements for a spatial region of interest (ROI).
    
    Attributes:
        roi_id: Identifier of the region.
        frame_idx: Current frame index.
        timestamp_s: Current video timestamp in seconds.
        vehicle_count: Number of unique active vehicle tracks inside the ROI.
        occupancy_ratio: Proportion of ROI area covered by vehicle bounding boxes [0.0, 1.0].
        vehicle_ids: List of track IDs currently within the ROI.
    """
    roi_id: str
    frame_idx: int
    timestamp_s: float
    vehicle_count: int
    occupancy_ratio: float
    vehicle_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roi_id": self.roi_id,
            "frame_idx": self.frame_idx,
            "timestamp_s": round(self.timestamp_s, 3),
            "vehicle_count": self.vehicle_count,
            "occupancy_ratio": round(self.occupancy_ratio, 4),
            "vehicle_ids": list(self.vehicle_ids),
        }


@dataclass
class CongestionState:
    """
    Explainable congestion classification for a region of interest.
    
    Attributes:
        roi_id: Identifier of the region.
        frame_idx: Current frame index.
        timestamp_s: Current video timestamp in seconds.
        level: Low, Moderate, or High.
        occupancy_ratio: Supporting occupancy metric.
        vehicle_count: Supporting active vehicle count.
        avg_displacement_px: Average displacement of vehicles in ROI over recent window.
        stopped_vehicle_count: Number of stopped vehicles in ROI.
        reason: Human-readable explanation of why this level was assigned.
    """
    roi_id: str
    frame_idx: int
    timestamp_s: float
    level: CongestionLevel
    occupancy_ratio: float
    vehicle_count: int
    avg_displacement_px: float
    stopped_vehicle_count: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roi_id": self.roi_id,
            "frame_idx": self.frame_idx,
            "timestamp_s": round(self.timestamp_s, 3),
            "level": self.level.value,
            "occupancy_ratio": round(self.occupancy_ratio, 4),
            "vehicle_count": self.vehicle_count,
            "avg_displacement_px": round(self.avg_displacement_px, 2),
            "stopped_vehicle_count": self.stopped_vehicle_count,
            "reason": self.reason,
        }


@dataclass
class StoppedVehicleEvent:
    """
    Event generated when a vehicle remains stationary beyond a threshold duration.
    """
    track_id: int
    start_frame: int
    start_timestamp_s: float
    end_frame: Optional[int] = None
    end_timestamp_s: Optional[float] = None
    duration_s: float = 0.0
    location: Tuple[float, float] = (0.0, 0.0)
    roi_id: Optional[str] = None
    status: str = "active"  # "active" or "resolved"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "start_timestamp_s": round(self.start_timestamp_s, 2),
            "end_frame": self.end_frame,
            "end_timestamp_s": round(self.end_timestamp_s, 2) if self.end_timestamp_s is not None else None,
            "duration_s": round(self.duration_s, 2),
            "location": [round(self.location[0], 2), round(self.location[1], 2)],
            "roi_id": self.roi_id,
            "status": self.status,
        }


@dataclass
class SpeedMeasurement:
    """
    Speed measurement abstraction.
    
    Explicitly tracks calibration status. In UA-DETRAC, speed_kmh is None
    because camera calibration and road geometry are unavailable.
    """
    track_id: int
    frame_idx: int
    speed_px_per_s: float
    speed_kmh: Optional[float] = None
    is_calibrated: bool = False
    calibration_note: str = (
        "Uncalibrated pixel displacement. Camera homography / road geometry unavailable in UA-DETRAC."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
            "speed_px_per_s": round(self.speed_px_per_s, 2),
            "speed_kmh": round(self.speed_kmh, 2) if self.speed_kmh is not None else None,
            "is_calibrated": self.is_calibrated,
            "calibration_note": self.calibration_note,
        }


@dataclass
class FrameTelemetry:
    """Combined traffic analytics snapshot for a single video frame."""
    frame_idx: int
    timestamp_s: float
    crossings: List[CrossingEvent] = field(default_factory=list)
    roi_densities: List[ROIDensity] = field(default_factory=list)
    congestion_states: List[CongestionState] = field(default_factory=list)
    active_stopped_events: List[StoppedVehicleEvent] = field(default_factory=list)
    resolved_stopped_events: List[StoppedVehicleEvent] = field(default_factory=list)
    speed_measurements: List[SpeedMeasurement] = field(default_factory=list)
    active_track_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_s": round(self.timestamp_s, 3),
            "active_track_count": self.active_track_count,
            "crossings": [c.to_dict() for c in self.crossings],
            "roi_densities": [d.to_dict() for d in self.roi_densities],
            "congestion_states": [c.to_dict() for c in self.congestion_states],
            "active_stopped_events": [e.to_dict() for e in self.active_stopped_events],
            "resolved_stopped_events": [e.to_dict() for e in self.resolved_stopped_events],
            "speed_measurements": [s.to_dict() for s in self.speed_measurements],
        }


@dataclass
class SequenceAnalyticsReport:
    """Aggregate analytics report across a complete video sequence."""
    sequence_name: str
    total_frames: int
    duration_s: float
    fps: float
    total_unique_vehicles: int
    line_counts: Dict[str, Dict[str, int]]
    flow_windows: List[LineFlow]
    stopped_vehicle_events: List[StoppedVehicleEvent]
    peak_occupancy_per_roi: Dict[str, float]
    average_occupancy_per_roi: Dict[str, float]
    congestion_distribution_per_roi: Dict[str, Dict[str, int]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_name": self.sequence_name,
            "total_frames": self.total_frames,
            "duration_s": round(self.duration_s, 2),
            "fps": round(self.fps, 2),
            "total_unique_vehicles": self.total_unique_vehicles,
            "line_counts": self.line_counts,
            "flow_windows": [f.to_dict() for f in self.flow_windows],
            "stopped_vehicle_events": [e.to_dict() for e in self.stopped_vehicle_events],
            "peak_occupancy_per_roi": {k: round(v, 4) for k, v in self.peak_occupancy_per_roi.items()},
            "average_occupancy_per_roi": {k: round(v, 4) for k, v in self.average_occupancy_per_roi.items()},
            "congestion_distribution_per_roi": self.congestion_distribution_per_roi,
        }
