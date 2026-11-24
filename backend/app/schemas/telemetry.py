"""
Telemetry, detection, tracking, and traffic analytics API contracts for TrafficLens-AI.

These schemas map directly to the structured outputs produced by the CV modules:
- YOLO11n vehicle detection
- ByteTrack multi-object tracking
- Spatiotemporal traffic analytics (counting, flow, ROI density, congestion, stopped-vehicle events)
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field


class DetectionSummarySchema(BaseModel):
    """Aggregate vehicle detection metrics."""
    total_detections: int = Field(..., ge=0, description="Total bounding boxes detected across all frames")
    mean_confidence: float = Field(..., ge=0.0, le=1.0, description="Mean confidence score of detections")
    class_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Detection counts per vehicle category before class-agnostic unification",
    )


class TrackingSummarySchema(BaseModel):
    """Multi-object vehicle tracking evaluation and summary metrics."""
    total_unique_tracks: int = Field(..., ge=0, description="Total unique vehicle track IDs assigned")
    mostly_tracked_count: Optional[int] = Field(default=None, ge=0, description="Vehicles tracked for >=80% of lifespan (MT)")
    mostly_lost_count: Optional[int] = Field(default=None, ge=0, description="Vehicles tracked for <=20% of lifespan (ML)")
    id_switches: Optional[int] = Field(default=None, ge=0, description="Total identity switches across the sequence (IDSW)")
    mota: Optional[float] = Field(default=None, description="Multiple Object Tracking Accuracy CLEAR metric")
    hota: Optional[float] = Field(default=None, description="Higher Order Tracking Accuracy metric")
    idf1: Optional[float] = Field(default=None, description="Identification F1-score")


class CrossingEventSchema(BaseModel):
    """Single vehicle track crossing a virtual counting line."""
    line_id: str = Field(..., description="Virtual counting line identifier")
    track_id: int = Field(..., description="Unique vehicle track identifier")
    frame_idx: int = Field(..., ge=0, description="Video frame number")
    timestamp_s: float = Field(..., ge=0.0, description="Frame timestamp in seconds")
    direction: str = Field(..., description="Crossing direction ('downstream' or 'upstream')")
    crossing_point: Tuple[float, float] = Field(..., description="Intersection coordinate (x, y) in pixel space")


class LineFlowSchema(BaseModel):
    """Traffic flow measurement over a defined time window."""
    line_id: str = Field(..., description="Virtual counting line identifier")
    window_start_s: float = Field(..., ge=0.0, description="Window start time in seconds")
    window_end_s: float = Field(..., ge=0.0, description="Window end time in seconds")
    total_count: int = Field(..., ge=0, description="Total vehicle crossings in window")
    flow_rate_vpm: float = Field(..., ge=0.0, description="Traffic throughput in Vehicles Per Minute (VPM)")
    direction_counts: Dict[str, int] = Field(default_factory=dict, description="Crossing counts per direction")
    direction_flows_vpm: Dict[str, float] = Field(default_factory=dict, description="VPM throughput per direction")


class ROIDensitySchema(BaseModel):
    """Traffic presence and spatial occupancy within a region of interest."""
    roi_id: str = Field(..., description="Region of interest identifier")
    frame_idx: int = Field(..., ge=0, description="Video frame number")
    timestamp_s: float = Field(..., ge=0.0, description="Frame timestamp in seconds")
    vehicle_count: int = Field(..., ge=0, description="Number of unique active vehicle tracks inside the ROI")
    occupancy_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Spatial ratio of ROI area covered by vehicle 2D footprints [0.0, 1.0]",
    )
    vehicle_ids: List[int] = Field(default_factory=list, description="IDs of vehicle tracks currently within the ROI")


class CongestionLevelEnum(str, Enum):
    """Discrete traffic congestion levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class CongestionStateSchema(BaseModel):
    """Explainable rule-based congestion classification for an ROI."""
    roi_id: str = Field(..., description="Region of interest identifier")
    frame_idx: int = Field(..., ge=0, description="Video frame number")
    timestamp_s: float = Field(..., ge=0.0, description="Frame timestamp in seconds")
    level: CongestionLevelEnum = Field(..., description="Classified level (low, moderate, high)")
    occupancy_ratio: float = Field(..., ge=0.0, le=1.0, description="ROI spatial occupancy ratio")
    vehicle_count: int = Field(..., ge=0, description="Active vehicle count in ROI")
    avg_displacement_px: float = Field(..., ge=0.0, description="Average pixel displacement of vehicles in ROI")
    stopped_vehicle_count: int = Field(..., ge=0, description="Number of stationary vehicles in ROI")
    reason: str = Field(..., description="Human-readable justification for this classification")


class StoppedVehicleEventSchema(BaseModel):
    """Event representing a vehicle remaining stationary beyond the configured threshold."""
    track_id: int = Field(..., description="Vehicle track identifier")
    start_frame: int = Field(..., ge=0, description="Frame when stationary state began")
    start_timestamp_s: float = Field(..., ge=0.0, description="Timestamp when stationary state began (seconds)")
    end_frame: Optional[int] = Field(default=None, ge=0, description="Frame when vehicle moved or departed")
    end_timestamp_s: Optional[float] = Field(default=None, ge=0.0, description="Timestamp when event resolved (seconds)")
    duration_s: float = Field(..., ge=0.0, description="Total stationary duration in seconds")
    location: Tuple[float, float] = Field(..., description="Last known center coordinate (x, y)")
    roi_id: Optional[str] = Field(default=None, description="Associated ROI if vehicle is inside a defined zone")
    status: str = Field(..., description="'active' if ongoing or 'resolved' once vehicle resumes motion/departs")


class SpeedMeasurementSchema(BaseModel):
    """
    Vehicle speed telemetry.
    
    Reflects the uncalibrated nature of UA-DETRAC: speed_kmh is None unless
    camera homography calibration is supplied.
    """
    track_id: int = Field(..., description="Vehicle track identifier")
    frame_idx: int = Field(..., ge=0, description="Video frame number")
    speed_px_per_s: float = Field(..., ge=0.0, description="Pixel displacement rate in pixels/second")
    speed_kmh: Optional[float] = Field(default=None, ge=0.0, description="Calibrated metric speed in km/h (None if uncalibrated)")
    is_calibrated: bool = Field(default=False, description="True if metric homography was applied")
    calibration_note: str = Field(
        default="Uncalibrated pixel displacement. Camera homography unavailable.",
        description="Explanation of calibration status",
    )


class TrafficAnalyticsSummarySchema(BaseModel):
    """Sequence-level summary matching the output of SequenceAnalyticsReport."""
    sequence_name: str = Field(..., description="Sequence or video identifier")
    total_frames: int = Field(..., gt=0, description="Total video frames processed")
    duration_s: float = Field(..., gt=0.0, description="Total sequence duration in seconds")
    fps: float = Field(..., gt=0.0, description="Frame rate in frames per second")
    total_unique_vehicles: int = Field(..., ge=0, description="Total unique vehicles tracked across sequence")
    line_counts: Dict[str, Dict[str, int]] = Field(
        ...,
        description="Cumulative vehicle crossings per line and direction",
    )
    flow_windows: List[LineFlowSchema] = Field(
        default_factory=list,
        description="Vehicles per minute flow time-series windows",
    )
    stopped_vehicle_events: List[StoppedVehicleEventSchema] = Field(
        default_factory=list,
        description="Stationary vehicle events recorded in sequence",
    )
    peak_occupancy_per_roi: Dict[str, float] = Field(
        default_factory=dict,
        description="Peak spatial occupancy ratio observed per ROI",
    )
    average_occupancy_per_roi: Dict[str, float] = Field(
        default_factory=dict,
        description="Mean spatial occupancy ratio per ROI across sequence",
    )
    congestion_distribution_per_roi: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Frame count breakdown across congestion levels per ROI",
    )


class JobTelemetryResponseSchema(BaseModel):
    """Full telemetry payload for a completed video processing job."""
    job_id: str = Field(..., description="Job identifier")
    video_id: str = Field(..., description="Video asset identifier")
    sequence_name: str = Field(..., description="Sequence name")
    detection_summary: Optional[DetectionSummarySchema] = Field(default=None, description="Detection overview")
    tracking_summary: Optional[TrackingSummarySchema] = Field(default=None, description="Tracking performance")
    analytics_summary: Optional[TrafficAnalyticsSummarySchema] = Field(
        default=None,
        description="Comprehensive traffic analytics telemetry",
    )


class JobResultsResponseSchema(BaseModel):
    """Structured results payload for a completed video processing job."""
    job_id: str = Field(..., description="Job identifier")
    video_id: str = Field(..., description="Video asset identifier")
    sequence_name: str = Field(..., description="Logical sequence name")
    status: str = Field(default="COMPLETED", description="Job status")
    duration_seconds: Optional[float] = Field(default=None, description="Processing wall-clock time in seconds")
    detection_summary: Optional[DetectionSummarySchema] = Field(default=None, description="Vehicle detection summary")
    tracking_summary: Optional[TrackingSummarySchema] = Field(default=None, description="Multi-object tracking summary")
    analytics_summary: Optional[TrafficAnalyticsSummarySchema] = Field(
        default=None,
        description="Core traffic analytics summary (counts, flow, density, congestion, stopped vehicles)",
    )
    artifacts: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary mapping artifact names to on-disk file paths (e.g. JSON, CSV, MP4)",
    )
    annotation_diagnostics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed diagnostic telemetry for video annotation rendering",
    )
