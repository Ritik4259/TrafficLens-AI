"""
Foundational tests for backend Pydantic schemas and data contracts.
"""

from __future__ import annotations
import pytest
from pydantic import ValidationError

from backend.app.schemas import (
    Point2DSchema,
    BBoxSchema,
    CountingLineConfigSchema,
    ROIConfigSchema,
    VideoMetadataSchema,
    ProcessingErrorSchema,
    CrossingEventSchema,
    LineFlowSchema,
    ROIDensitySchema,
    CongestionLevelEnum,
    CongestionStateSchema,
    StoppedVehicleEventSchema,
    SpeedMeasurementSchema,
    TrafficAnalyticsSummarySchema,
)


def test_bbox_schema_valid_and_invalid():
    """Validates BBoxSchema positive dimensions and rejection of non-positive widths."""
    valid_box = BBoxSchema(left=10.0, top=20.0, width=50.0, height=40.0)
    assert valid_box.left == 10.0
    assert valid_box.width == 50.0

    # Negative width must be rejected
    with pytest.raises(ValidationError):
        BBoxSchema(left=10.0, top=20.0, width=-5.0, height=40.0)

    # Zero height must be rejected
    with pytest.raises(ValidationError):
        BBoxSchema(left=10.0, top=20.0, width=50.0, height=0.0)


def test_counting_line_and_roi_config_schemas():
    """Validates line configurations and polygon vertex constraints."""
    line = CountingLineConfigSchema(
        line_id="line_01",
        name="Main Road",
        p1=(100.0, 200.0),
        p2=(500.0, 200.0),
        positive_direction_label="downstream",
        negative_direction_label="upstream",
        cooldown_frames=15,
        min_track_age=2,
    )
    assert line.line_id == "line_01"
    assert line.cooldown_frames == 15

    # Polygon requires at least 3 vertices
    roi = ROIConfigSchema(
        roi_id="roi_01",
        name="Active Zone",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
        description="Test zone",
    )
    assert len(roi.polygon) == 4

    with pytest.raises(ValidationError):
        ROIConfigSchema(roi_id="roi_bad", name="Bad Zone", polygon=[(0, 0), (10, 10)])


def test_congestion_state_schema_validation():
    """Validates congestion state level enum and field ranges."""
    state = CongestionStateSchema(
        roi_id="roi_01",
        frame_idx=100,
        timestamp_s=4.0,
        level=CongestionLevelEnum.MODERATE,
        occupancy_ratio=0.25,
        vehicle_count=3,
        avg_displacement_px=15.2,
        stopped_vehicle_count=1,
        reason="Moderate occupancy observed.",
    )
    assert state.level == CongestionLevelEnum.MODERATE
    assert state.occupancy_ratio == 0.25

    # Occupancy ratio cannot exceed 1.0
    with pytest.raises(ValidationError):
        CongestionStateSchema(
            roi_id="roi_01",
            frame_idx=100,
            timestamp_s=4.0,
            level=CongestionLevelEnum.HIGH,
            occupancy_ratio=1.5,
            vehicle_count=5,
            avg_displacement_px=0.0,
            stopped_vehicle_count=5,
            reason="Overflow.",
        )


def test_speed_measurement_schema_uncalibrated():
    """Validates that speed measurement defaults to uncalibrated and allows speed_kmh=None."""
    speed = SpeedMeasurementSchema(
        track_id=1,
        frame_idx=50,
        speed_px_per_s=120.5,
        speed_kmh=None,
        is_calibrated=False,
    )
    assert speed.is_calibrated is False
    assert speed.speed_kmh is None
    assert "Uncalibrated" in speed.calibration_note


def test_stopped_vehicle_event_schema():
    """Validates stopped vehicle event attributes and status."""
    event = StoppedVehicleEventSchema(
        track_id=6,
        start_frame=10,
        start_timestamp_s=0.4,
        end_frame=60,
        end_timestamp_s=2.4,
        duration_s=2.0,
        location=(120.0, 180.0),
        roi_id="roi_side",
        status="resolved",
    )
    assert event.track_id == 6
    assert event.status == "resolved"
    assert event.duration_s == 2.0
