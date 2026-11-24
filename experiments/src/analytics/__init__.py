"""
TrafficLens-AI Core Traffic Analytics Package.
"""

from .config import (
    AnalyticsConfig,
    CountingLineConfig,
    ROIConfig,
    FlowConfig,
    DensityConfig,
    CongestionConfig,
    StoppedVehicleConfig,
    SpeedConfig,
)
from .geometry import (
    segments_intersect,
    determine_crossing_direction,
    point_in_polygon,
    polygon_area,
    box_polygon_intersection_area,
)
from .models import (
    CrossingEvent,
    LineFlow,
    ROIDensity,
    CongestionLevel,
    CongestionState,
    StoppedVehicleEvent,
    SpeedMeasurement,
    FrameTelemetry,
    SequenceAnalyticsReport,
)
from .counting import LineCrossingCounter
from .flow import FlowCalculator
from .density import ROIDensityEstimator
from .congestion import CongestionClassifier
from .events import StoppedVehicleDetector
from .speed import BaseSpeedEstimator, PixelSpeedEstimator, CalibratedSpeedEstimator
from .pipeline import TrafficAnalyticsPipeline

__all__ = [
    "AnalyticsConfig",
    "CountingLineConfig",
    "ROIConfig",
    "FlowConfig",
    "DensityConfig",
    "CongestionConfig",
    "StoppedVehicleConfig",
    "SpeedConfig",
    "segments_intersect",
    "determine_crossing_direction",
    "point_in_polygon",
    "polygon_area",
    "box_polygon_intersection_area",
    "CrossingEvent",
    "LineFlow",
    "ROIDensity",
    "CongestionLevel",
    "CongestionState",
    "StoppedVehicleEvent",
    "SpeedMeasurement",
    "FrameTelemetry",
    "SequenceAnalyticsReport",
    "LineCrossingCounter",
    "FlowCalculator",
    "ROIDensityEstimator",
    "CongestionClassifier",
    "StoppedVehicleDetector",
    "BaseSpeedEstimator",
    "PixelSpeedEstimator",
    "CalibratedSpeedEstimator",
    "TrafficAnalyticsPipeline",
]
