"""
Unit and integration tests for core traffic analytics module.

Validates geometry primitives, line crossing counting, flow aggregation,
ROI density estimation, rule-based congestion classification, stopped-vehicle
event lifecycle, speed estimation abstraction, and sequence isolation.
"""

from __future__ import annotations
import math
import numpy as np
import pytest

from experiments.src.dataset.parser import BBox
from experiments.src.tracking.tracker import Track
from experiments.src.analytics import (
    AnalyticsConfig,
    CountingLineConfig,
    ROIConfig,
    FlowConfig,
    DensityConfig,
    CongestionConfig,
    StoppedVehicleConfig,
    SpeedConfig,
    ROIDensity,
    segments_intersect,
    determine_crossing_direction,
    point_in_polygon,
    polygon_area,
    box_polygon_intersection_area,
    LineCrossingCounter,
    FlowCalculator,
    ROIDensityEstimator,
    CongestionClassifier,
    CongestionLevel,
    StoppedVehicleDetector,
    PixelSpeedEstimator,
    CalibratedSpeedEstimator,
    TrafficAnalyticsPipeline,
)


# ==============================================================================
# 1. Geometry Primitive Tests
# ==============================================================================

def test_geometry_segment_intersection():
    """Tests orthogonal, parallel, skew, and collinear segment intersections."""
    # Orthogonal crossing
    hit, pt = segments_intersect((0.0, 5.0), (10.0, 5.0), (5.0, 0.0), (5.0, 10.0))
    assert hit is True
    assert pt is not None
    assert math.isclose(pt[0], 5.0) and math.isclose(pt[1], 5.0)

    # Parallel disjoint segments
    hit, pt = segments_intersect((0.0, 0.0), (10.0, 0.0), (0.0, 5.0), (10.0, 5.0))
    assert hit is False
    assert pt is None

    # Skew non-intersecting
    hit, pt = segments_intersect((0.0, 0.0), (2.0, 2.0), (5.0, 5.0), (10.0, 10.0))
    assert hit is False

    # T-junction intersection
    hit, pt = segments_intersect((0.0, 5.0), (10.0, 5.0), (5.0, 5.0), (5.0, 10.0))
    assert hit is True
    assert pt is not None
    assert math.isclose(pt[0], 5.0) and math.isclose(pt[1], 5.0)


def test_geometry_direction_determination():
    """Tests directional resolution using normal dot product."""
    # Horizontal line from (0, 100) to (200, 100)
    line_p1 = (0.0, 100.0)
    line_p2 = (200.0, 100.0)

    # Downward movement (top to bottom)
    dir_down = determine_crossing_direction(
        prev_pos=(100.0, 50.0),
        curr_pos=(100.0, 150.0),
        line_p1=line_p1,
        line_p2=line_p2,
        positive_label="downstream",
        negative_label="upstream",
    )
    assert dir_down == "downstream"

    # Upward movement (bottom to top)
    dir_up = determine_crossing_direction(
        prev_pos=(100.0, 150.0),
        curr_pos=(100.0, 50.0),
        line_p1=line_p1,
        line_p2=line_p2,
        positive_label="downstream",
        negative_label="upstream",
    )
    assert dir_up == "upstream"


def test_geometry_point_in_polygon_and_area():
    """Tests point-in-polygon ray-casting and Shoelace polygon area."""
    poly = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    assert polygon_area(poly) == 10000.0

    # Inside
    assert point_in_polygon((50.0, 50.0), poly) is True
    # Outside
    assert point_in_polygon((150.0, 50.0), poly) is False
    assert point_in_polygon((-10.0, 50.0), poly) is False


def test_geometry_box_polygon_intersection():
    """Tests Sutherland-Hodgman box-polygon intersection area."""
    roi = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]

    # Box fully inside
    b_inside = BBox(left=10.0, top=10.0, width=20.0, height=20.0)
    area = box_polygon_intersection_area(b_inside, roi)
    assert math.isclose(area, 400.0, rel_tol=1e-4)

    # Box fully outside
    b_outside = BBox(left=200.0, top=200.0, width=50.0, height=50.0)
    assert box_polygon_intersection_area(b_outside, roi) == 0.0

    # Box half overlapping (left half inside: x from 80 to 100, y from 10 to 30 -> 20x20 = 400)
    b_overlap = BBox(left=80.0, top=10.0, width=40.0, height=20.0)
    overlap_area = box_polygon_intersection_area(b_overlap, roi)
    assert math.isclose(overlap_area, 400.0, rel_tol=1e-4)


# ==============================================================================
# 2. Line Crossing & Vehicle Counting Tests
# ==============================================================================

def test_counting_positive_and_negative_crossing():
    """Tests bidirectional crossing detection across a single line."""
    line = CountingLineConfig(
        line_id="line_main",
        name="Main Road",
        p1=(0.0, 100.0),
        p2=(200.0, 100.0),
        positive_direction_label="downstream",
        negative_direction_label="upstream",
        cooldown_frames=5,
        min_track_age=2,
    )
    counter = LineCrossingCounter([line], fps=25.0)

    # Track 1: moving downstream (from y=80 to y=120)
    t1 = Track(track_id=1, bbox=BBox(90, 110, 20, 20), confidence=0.9, age=2, history=[(100.0, 80.0), (100.0, 120.0)])
    # Track 2: moving upstream (from y=120 to y=80)
    t2 = Track(track_id=2, bbox=BBox(140, 70, 20, 20), confidence=0.9, age=2, history=[(150.0, 120.0), (150.0, 80.0)])

    events = counter.update([t1, t2], frame_idx=10)
    assert len(events) == 2

    e1 = next(e for e in events if e.track_id == 1)
    assert e1.direction == "downstream"
    assert math.isclose(e1.crossing_point[1], 100.0)

    e2 = next(e for e in events if e.track_id == 2)
    assert e2.direction == "upstream"
    assert math.isclose(e2.crossing_point[1], 100.0)

    counts = counter.get_counts("line_main")
    assert counts["downstream"] == 1
    assert counts["upstream"] == 1
    assert counts["total"] == 2


def test_counting_duplicate_prevention_and_jitter():
    """Tests that trajectory jitter near the line does not cause duplicate counts."""
    line = CountingLineConfig(
        line_id="line_main",
        name="Main Road",
        p1=(0.0, 100.0),
        p2=(200.0, 100.0),
        cooldown_frames=10,
        min_track_age=1,
    )
    counter = LineCrossingCounter([line], fps=25.0)

    # Frame 1: crosses downstream (90 -> 110)
    t_f1 = Track(track_id=1, bbox=BBox(90, 100, 20, 20), confidence=0.9, age=2, history=[(100.0, 90.0), (100.0, 110.0)])
    events_f1 = counter.update([t_f1], frame_idx=1)
    assert len(events_f1) == 1

    # Frame 2: jitter moves backwards slightly crossing upstream (110 -> 98) during cooldown
    t_f2 = Track(track_id=1, bbox=BBox(90, 90, 20, 20), confidence=0.9, age=3, history=[(100.0, 110.0), (100.0, 98.0)])
    events_f2 = counter.update([t_f2], frame_idx=2)
    assert len(events_f2) == 0  # Blocked by cooldown

    # Frame 3: jitter moves forward again crossing downstream (98 -> 105)
    t_f3 = Track(track_id=1, bbox=BBox(90, 95, 20, 20), confidence=0.9, age=4, history=[(100.0, 98.0), (100.0, 105.0)])
    events_f3 = counter.update([t_f3], frame_idx=3)
    assert len(events_f3) == 0  # Blocked by cooldown and downstream already counted

    # Total counts remain strictly 1
    assert counter.get_counts("line_main")["total"] == 1


def test_counting_multiple_lines():
    """Tests tracking vehicle crossing multiple independent counting lines."""
    l1 = CountingLineConfig(line_id="line_north", name="North", p1=(0.0, 100.0), p2=(200.0, 100.0))
    l2 = CountingLineConfig(line_id="line_south", name="South", p1=(0.0, 200.0), p2=(200.0, 200.0))
    counter = LineCrossingCounter([l1, l2], fps=25.0)

    # Frame 10: crosses line 1 (80 -> 120)
    t_f10 = Track(track_id=5, bbox=BBox(90, 110, 20, 20), confidence=0.9, age=2, history=[(100.0, 80.0), (100.0, 120.0)])
    ev10 = counter.update([t_f10], frame_idx=10)
    assert len(ev10) == 1 and ev10[0].line_id == "line_north"

    # Frame 20: crosses line 2 (180 -> 220)
    t_f20 = Track(track_id=5, bbox=BBox(90, 210, 20, 20), confidence=0.9, age=3, history=[(100.0, 180.0), (100.0, 220.0)])
    ev20 = counter.update([t_f20], frame_idx=20)
    assert len(ev20) == 1 and ev20[0].line_id == "line_south"

    assert counter.get_counts("line_north")["total"] == 1
    assert counter.get_counts("line_south")["total"] == 1


def test_counting_incomplete_trajectories_and_termination():
    """Tests edge cases: single-point history, track appearing past line, track terminating before line."""
    line = CountingLineConfig(line_id="line_main", name="Main", p1=(0.0, 100.0), p2=(200.0, 100.0))
    counter = LineCrossingCounter([line], fps=25.0)

    # Track appears already beyond line (y=150) with 1 history point
    t_new = Track(track_id=1, bbox=BBox(90, 140, 20, 20), confidence=0.9, age=1, history=[(100.0, 150.0)])
    assert len(counter.update([t_new], frame_idx=1)) == 0

    # Track moves from y=150 to y=180 (moving away)
    t_move = Track(track_id=1, bbox=BBox(90, 170, 20, 20), confidence=0.9, age=2, history=[(100.0, 150.0), (100.0, 180.0)])
    assert len(counter.update([t_move], frame_idx=2)) == 0

    # Track terminates before line (moves 80 -> 95)
    t_term = Track(track_id=2, bbox=BBox(90, 85, 20, 20), confidence=0.9, age=2, history=[(100.0, 80.0), (100.0, 95.0)])
    assert len(counter.update([t_term], frame_idx=2)) == 0
    assert counter.get_counts("line_main")["total"] == 0


# ==============================================================================
# 3. Traffic Flow Tests
# ==============================================================================

def test_flow_window_aggregation():
    """Tests vehicles-per-minute flow aggregation over time windows."""
    line = CountingLineConfig(line_id="line_main", name="Main", p1=(0.0, 100.0), p2=(200.0, 100.0))
    flow_calc = FlowCalculator(FlowConfig(window_size_seconds=60.0), lines=[line])

    # 3 vehicles cross line_main between 0s and 60s
    counter = LineCrossingCounter([line], fps=25.0)
    for i in range(1, 4):
        t = Track(track_id=i, bbox=BBox(90, 110, 20, 20), confidence=0.9, age=2, history=[(100.0, 80.0), (100.0, 120.0)])
        ev = counter.update([t], frame_idx=i * 25, timestamp_s=float(i))
        flow_calc.add_events(ev)

    flow = flow_calc.calculate_window_flow("line_main", window_start_s=0.0, window_end_s=60.0)
    assert flow.total_count == 3
    # 3 vehicles in 60s (1.0 minute) = 3.0 VPM
    assert math.isclose(flow.flow_rate_vpm, 3.0)
    assert flow.direction_counts["downstream"] == 3
    assert math.isclose(flow.direction_flows_vpm["downstream"], 3.0)


# ==============================================================================
# 4. ROI Density & Semantic Distinction Tests
# ==============================================================================

def test_roi_density_and_occupancy():
    """Tests vehicle counting and bounding-box spatial occupancy in ROI."""
    # 100x100 ROI -> area = 10,000 px^2
    roi = ROIConfig(roi_id="roi_test", name="Test Zone", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
    density_estimator = ROIDensityEstimator([roi], fps=25.0)

    # Vehicle 1: 50x50 box inside -> area = 2500 px^2
    t1 = Track(track_id=1, bbox=BBox(0, 0, 50, 50), confidence=0.9)
    # Vehicle 2: 50x50 box inside -> area = 2500 px^2
    t2 = Track(track_id=2, bbox=BBox(50, 50, 50, 50), confidence=0.9)
    # Vehicle 3: outside ROI
    t3 = Track(track_id=3, bbox=BBox(200, 200, 40, 40), confidence=0.9)

    densities = density_estimator.update([t1, t2, t3], frame_idx=1)
    assert len(densities) == 1
    d = densities[0]
    assert d.vehicle_count == 2
    assert set(d.vehicle_ids) == {1, 2}
    # Occupancy ratio = (2500 + 2500) / 10000 = 0.50 (50%)
    assert math.isclose(d.occupancy_ratio, 0.50, rel_tol=1e-4)


# ==============================================================================
# 5. Congestion Classification Tests
# ==============================================================================

def test_congestion_classification_thresholds():
    """Tests rule-based congestion classification with explainable reasons."""
    classifier = CongestionClassifier(
        CongestionConfig(
            low_occupancy_thresh=0.15,
            high_occupancy_thresh=0.40,
            stopped_ratio_thresh=0.35,
            min_vehicles_for_congestion=2,
        ),
        fps=25.0,
    )

    # 1. Below vehicle threshold -> LOW
    d_empty = ROIDensity(roi_id="roi_1", frame_idx=1, timestamp_s=1.0, vehicle_count=1, occupancy_ratio=0.10, vehicle_ids=[1])
    s_low = classifier.classify_roi(d_empty, tracks_in_roi=[], stopped_track_ids=set())
    assert s_low.level == CongestionLevel.LOW
    assert "below threshold" in s_low.reason

    # 2. High spatial occupancy (50% > 40%) -> HIGH
    d_high_occ = ROIDensity(roi_id="roi_1", frame_idx=2, timestamp_s=2.0, vehicle_count=4, occupancy_ratio=0.50, vehicle_ids=[1, 2, 3, 4])
    s_high = classifier.classify_roi(d_high_occ, tracks_in_roi=[], stopped_track_ids=set())
    assert s_high.level == CongestionLevel.HIGH
    assert "exceeds high threshold" in s_high.reason

    # 3. High stopped proportion (2 of 3 vehicles stopped = 66% > 35%) -> HIGH
    d_stopped = ROIDensity(roi_id="roi_1", frame_idx=3, timestamp_s=3.0, vehicle_count=3, occupancy_ratio=0.20, vehicle_ids=[1, 2, 3])
    s_stopped = classifier.classify_roi(d_stopped, tracks_in_roi=[], stopped_track_ids={1, 2})
    assert s_stopped.level == CongestionLevel.HIGH
    assert "stopped" in s_stopped.reason

    # 4. Moderate occupancy (25%) -> MODERATE
    d_mod = ROIDensity(roi_id="roi_1", frame_idx=4, timestamp_s=4.0, vehicle_count=3, occupancy_ratio=0.25, vehicle_ids=[1, 2, 3])
    s_mod = classifier.classify_roi(d_mod, tracks_in_roi=[], stopped_track_ids=set())
    assert s_mod.level == CongestionLevel.MODERATE
    assert "Moderate" in s_mod.reason


# ==============================================================================
# 6. Stopped Vehicle Event Lifecycle Tests
# ==============================================================================

def test_stopped_vehicle_lifecycle():
    """Tests stopped vehicle event detection, duration accumulation, and movement resolution."""
    cfg = StoppedVehicleConfig(
        min_duration_seconds=2.0,
        max_displacement_px=10.0,
        sliding_window_frames=50,  # 2.0s at 25 fps
    )
    detector = StoppedVehicleDetector(cfg, fps=25.0)

    # Simulate 52 frames of stationary vehicle at (100, 100) with small +-1px jitter (51 intervals = 2.04s)
    pos = (100.0, 100.0)
    for f in range(1, 53):
        jitter_x = 100.0 + (0.5 if f % 2 == 0 else -0.5)
        t = Track(track_id=1, bbox=BBox(jitter_x - 10, 90, 20, 20), confidence=0.9, age=f, history=[pos, (jitter_x, 100.0)])
        active, resolved = detector.update([t], frame_idx=f, timestamp_s=f / 25.0)

    # At frame 52 (>2.0s elapsed), event should trigger active
    assert len(active) == 1
    ev = active[0]
    assert ev.track_id == 1
    assert ev.status == "active"
    assert ev.duration_s >= 2.0

    # Frame 53 to 60: vehicle starts driving fast (moves to 150, 100)
    t_moving = Track(track_id=1, bbox=BBox(140, 90, 20, 20), confidence=0.9, age=60, history=[pos, (150.0, 100.0)])
    active_post, resolved_post = detector.update([t_moving], frame_idx=60, timestamp_s=60 / 25.0)

    assert len(active_post) == 0
    assert len(resolved_post) == 1
    res_ev = resolved_post[0]
    assert res_ev.track_id == 1
    assert res_ev.status == "resolved"
    assert res_ev.end_frame == 60


# ==============================================================================
# 7. Speed Estimation Abstraction Tests
# ==============================================================================

def test_speed_estimator_abstraction():
    """Validates that uncalibrated estimator does not fabricate km/h and calibrated estimator works."""
    # 1. Uncalibrated PixelSpeedEstimator
    pixel_est = PixelSpeedEstimator(SpeedConfig(fps=25.0, smoothing_window_frames=5))
    track = Track(track_id=1, bbox=BBox(0, 0, 10, 10), confidence=0.9, history=[(0.0, 0.0), (10.0, 0.0)])
    meas = pixel_est.estimate_speed(track, frame_idx=2)
    assert meas.is_calibrated is False
    assert meas.speed_kmh is None
    # 10 pixels over 1 frame (1/25 s) = 250 px/s
    assert math.isclose(meas.speed_px_per_s, 250.0)
    assert "Uncalibrated" in meas.calibration_note

    # 2. CalibratedSpeedEstimator with known 0.1 m/pixel scaling homography
    # [ [0.1, 0, 0], [0, 0.1, 0], [0, 0, 1] ]
    H = np.diag([0.1, 0.1, 1.0])
    cal_est = CalibratedSpeedEstimator(homography_matrix=H, fps=25.0)
    meas_cal = cal_est.estimate_speed(track, frame_idx=2)
    assert meas_cal.is_calibrated is True
    assert meas_cal.speed_kmh is not None
    # 10 px * 0.1 m/px = 1.0 meter in 1/25 s = 25 m/s = 90 km/h
    assert math.isclose(meas_cal.speed_kmh, 90.0)


# ==============================================================================
# 8. Full Pipeline & Sequence Isolation Tests
# ==============================================================================

def test_pipeline_sequence_isolation():
    """Validates that pipeline reset guarantees complete sequence isolation."""
    line = CountingLineConfig(line_id="line_test", name="Test", p1=(0, 50), p2=(100, 50))
    roi = ROIConfig(roi_id="roi_test", name="Zone", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
    config = AnalyticsConfig(lines=[line], rois=[roi], fps=25.0)
    pipeline = TrafficAnalyticsPipeline(config)

    # Run synthetic sequence 1
    tracks_seq1 = {
        1: [Track(track_id=1, bbox=BBox(40, 20, 20, 20), confidence=0.9, age=1, history=[(50, 20)])],
        2: [Track(track_id=1, bbox=BBox(40, 80, 20, 20), confidence=0.9, age=2, history=[(50, 20), (50, 80)])],
    }
    telems1, report1 = pipeline.process_sequence(tracks_seq1, sequence_name="seq1")
    assert report1.line_counts["line_test"]["total"] == 1
    assert report1.total_unique_vehicles == 1

    # Reset
    pipeline.reset()
    assert len(pipeline.frame_telemetries) == 0
    assert pipeline.counter.get_counts("line_test")["total"] == 0

    # Run synthetic sequence 2 with empty tracks
    tracks_seq2 = {
        1: [],
        2: [],
    }
    telems2, report2 = pipeline.process_sequence(tracks_seq2, sequence_name="seq2")
    assert report2.line_counts["line_test"]["total"] == 0
    assert report2.total_unique_vehicles == 0
