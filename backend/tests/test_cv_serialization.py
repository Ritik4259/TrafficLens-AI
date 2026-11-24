"""
Tests validating serialization of real CV analytics outputs against Pydantic contracts.
"""

from __future__ import annotations
import json
from pathlib import Path
import pytest

from backend.app.schemas import TrafficAnalyticsSummarySchema
from backend.app.services.cv_pipeline_adapter import CVPipelineAdapter
from experiments.src.analytics import SequenceAnalyticsReport, LineFlow, StoppedVehicleEvent


def test_serialize_real_mvi_20011_analytics_output():
    """Validates real generated MVI_20011 analytics JSON against TrafficAnalyticsSummarySchema."""
    json_path = Path("experiments/outputs/analytics/MVI_20011/MVI_20011_analytics.json")
    assert json_path.exists(), f"Missing real artifact: {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary_data = data["summary"]
    schema_obj = TrafficAnalyticsSummarySchema.model_validate(summary_data)

    assert schema_obj.sequence_name == "MVI_20011"
    assert schema_obj.total_frames == 664
    assert schema_obj.fps == 25.0
    assert schema_obj.total_unique_vehicles == 65
    assert "line_downstream" in schema_obj.line_counts
    assert schema_obj.line_counts["line_downstream"]["total"] == 25
    assert len(schema_obj.stopped_vehicle_events) > 0


def test_serialize_real_mvi_40131_analytics_output():
    """Validates real generated MVI_40131 analytics JSON against TrafficAnalyticsSummarySchema."""
    json_path = Path("experiments/outputs/analytics/MVI_40131/MVI_40131_analytics.json")
    assert json_path.exists(), f"Missing real artifact: {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary_data = data["summary"]
    schema_obj = TrafficAnalyticsSummarySchema.model_validate(summary_data)

    assert schema_obj.sequence_name == "MVI_40131"
    assert schema_obj.total_frames == 1645
    assert schema_obj.total_unique_vehicles == 103
    assert "line_corridor" in schema_obj.line_counts
    assert schema_obj.line_counts["line_corridor"]["total"] == 55


def test_cv_pipeline_adapter_conversion():
    """Validates that CVPipelineAdapter correctly maps native domain model to Pydantic contract."""
    native_report = SequenceAnalyticsReport(
        sequence_name="MVI_TEST",
        total_frames=100,
        duration_s=4.0,
        fps=25.0,
        total_unique_vehicles=5,
        line_counts={"line_1": {"total": 2, "downstream": 2, "upstream": 0}},
        flow_windows=[
            LineFlow(
                line_id="line_1",
                window_start_s=0.0,
                window_end_s=4.0,
                total_count=2,
                flow_rate_vpm=30.0,
                direction_counts={"downstream": 2},
                direction_flows_vpm={"downstream": 30.0},
            )
        ],
        stopped_vehicle_events=[
            StoppedVehicleEvent(
                track_id=3,
                start_frame=10,
                start_timestamp_s=0.4,
                duration_s=3.0,
                location=(50.0, 50.0),
                roi_id="roi_test",
                status="active",
            )
        ],
        peak_occupancy_per_roi={"roi_test": 0.25},
        average_occupancy_per_roi={"roi_test": 0.15},
        congestion_distribution_per_roi={"roi_test": {"low": 80, "moderate": 20, "high": 0}},
    )

    schema_obj = CVPipelineAdapter.convert_analytics_report_to_schema(native_report)
    assert isinstance(schema_obj, TrafficAnalyticsSummarySchema)
    assert schema_obj.sequence_name == "MVI_TEST"
    assert schema_obj.flow_windows[0].flow_rate_vpm == 30.0
    assert schema_obj.stopped_vehicle_events[0].track_id == 3
