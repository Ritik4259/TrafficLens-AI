"""
Script to execute core traffic analytics on curated UA-DETRAC test sequences.

Processes tracked vehicle outputs, detects virtual line crossings, computes traffic flow,
estimates ROI densities, classifies congestion, detects stopped-vehicle events,
measures runtime latency overhead, and exports telemetry (JSON, CSV, keyframes, video).
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

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
    TrafficAnalyticsPipeline,
)
from experiments.src.analytics.visualizer import AnalyticsVisualizer


def load_tracks_from_json(json_path: Path) -> Dict[int, List[Track]]:
    """Loads pre-tracked vehicles and histories from tracking JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracks_by_frame: Dict[int, List[Track]] = {}
    for f_str, t_list in data["frames"].items():
        f_idx = int(f_str)
        tracks = []
        for item in t_list:
            b = item["bbox"]
            bbox = BBox(left=b["left"], top=b["top"], width=b["width"], height=b["height"])
            track = Track(
                track_id=item["track_id"],
                bbox=bbox,
                confidence=item["confidence"],
                class_name=item.get("class_name", "vehicle"),
                age=item.get("age", 1),
                history=[(pt[0], pt[1]) for pt in item.get("history", [])],
            )
            tracks.append(track)
        tracks_by_frame[f_idx] = tracks

    return tracks_by_frame


def get_sequence_configs() -> Dict[str, AnalyticsConfig]:
    """Builds realistic counting lines and ROIs for test sequences."""
    configs = {}

    # Sequence 1: MVI_20011 (Night, Unstable camera)
    line_20011 = CountingLineConfig(
        line_id="line_downstream",
        name="Main Road Crossline",
        p1=(420.0, 270.0),
        p2=(880.0, 270.0),
        positive_direction_label="downstream",
        negative_direction_label="upstream",
        cooldown_frames=15,
        min_track_age=2,
    )
    roi_main_20011 = ROIConfig(
        roi_id="roi_main_road",
        name="Active Roadway",
        polygon=[(480.0, 100.0), (820.0, 100.0), (940.0, 520.0), (520.0, 520.0)],
        description="Main multilane vehicular roadway",
    )
    roi_side_20011 = ROIConfig(
        roi_id="roi_side_zone",
        name="Side Stopping Zone",
        polygon=[(60.0, 130.0), (280.0, 130.0), (280.0, 300.0), (60.0, 300.0)],
        description="Side stopping and parking zone",
    )
    configs["MVI_20011"] = AnalyticsConfig(
        fps=25.0,
        frame_width=960.0,
        frame_height=540.0,
        lines=[line_20011],
        rois=[roi_main_20011, roi_side_20011],
        flow=FlowConfig(window_size_seconds=30.0, step_size_seconds=10.0),
        density=DensityConfig(method="box_intersection"),
        congestion=CongestionConfig(
            low_occupancy_thresh=0.15,
            high_occupancy_thresh=0.40,
            stopped_ratio_thresh=0.35,
            min_vehicles_for_congestion=2,
        ),
        stopped_vehicle=StoppedVehicleConfig(
            min_duration_seconds=3.0,
            max_displacement_px=20.0,  # camera vibration resilience
            sliding_window_frames=75,
        ),
        speed=SpeedConfig(fps=25.0, smoothing_window_frames=5),
    )

    # Sequence 2: MVI_40131 (Rainy, Unstable camera)
    line_40131 = CountingLineConfig(
        line_id="line_corridor",
        name="Corridor Crossline",
        p1=(120.0, 260.0),
        p2=(860.0, 260.0),
        positive_direction_label="downstream",
        negative_direction_label="upstream",
        cooldown_frames=15,
        min_track_age=2,
    )
    roi_main_40131 = ROIConfig(
        roi_id="roi_corridor",
        name="Expressway Corridor",
        polygon=[(120.0, 120.0), (860.0, 120.0), (920.0, 500.0), (80.0, 500.0)],
        description="Highway vehicular corridor",
    )
    configs["MVI_40131"] = AnalyticsConfig(
        fps=25.0,
        frame_width=960.0,
        frame_height=540.0,
        lines=[line_40131],
        rois=[roi_main_40131],
        flow=FlowConfig(window_size_seconds=30.0, step_size_seconds=10.0),
        density=DensityConfig(method="box_intersection"),
        congestion=CongestionConfig(
            low_occupancy_thresh=0.15,
            high_occupancy_thresh=0.40,
            stopped_ratio_thresh=0.35,
            min_vehicles_for_congestion=2,
        ),
        stopped_vehicle=StoppedVehicleConfig(
            min_duration_seconds=3.0,
            max_displacement_px=20.0,
            sliding_window_frames=75,
        ),
        speed=SpeedConfig(fps=25.0, smoothing_window_frames=5),
    )

    return configs


def run_sequence_analytics(
    sequence_id: str,
    tracks_json_path: Path,
    image_dir: Path,
    output_dir: Path,
    config: AnalyticsConfig,
) -> Dict[str, Any]:
    """Runs the analytics pipeline on a single sequence and produces all outputs."""
    print(f"\n{'='*70}\nRunning Analytics on {sequence_id}\n{'='*70}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks_by_frame = load_tracks_from_json(tracks_json_path)
    total_frames = len(tracks_by_frame)
    print(f"Loaded {total_frames} frames of tracking data.")

    pipeline = TrafficAnalyticsPipeline(config)
    visualizer = AnalyticsVisualizer(config)

    # Measure latency overhead per frame
    frame_latencies_ms: List[float] = []
    telemetries = []
    pipeline.reset()

    sorted_frames = sorted(tracks_by_frame.keys())
    t0_all = time.perf_counter()

    for f_idx in sorted_frames:
        t_start = time.perf_counter()
        telem = pipeline.process_frame(tracks_by_frame[f_idx], f_idx)
        lat_ms = (time.perf_counter() - t_start) * 1000.0
        frame_latencies_ms.append(lat_ms)
        telemetries.append(telem)

    total_time_s = time.perf_counter() - t0_all
    avg_latency_ms = float(np.mean(frame_latencies_ms)) if frame_latencies_ms else 0.0
    p95_latency_ms = float(np.percentile(frame_latencies_ms, 95)) if frame_latencies_ms else 0.0

    # Build sequence report
    duration_s = total_frames / config.fps
    flow_windows = pipeline.flow_calc.compute_sequence_tumbling_windows(duration_s)
    all_stopped = pipeline.stopped_detector.get_all_events()

    from experiments.src.analytics.models import SequenceAnalyticsReport
    report = SequenceAnalyticsReport(
        sequence_name=sequence_id,
        total_frames=total_frames,
        duration_s=duration_s,
        fps=config.fps,
        total_unique_vehicles=len(pipeline._unique_track_ids),
        line_counts=pipeline.counter.get_counts(),
        flow_windows=flow_windows,
        stopped_vehicle_events=all_stopped,
        peak_occupancy_per_roi=pipeline.density_estimator.get_peak_occupancy(),
        average_occupancy_per_roi=pipeline.density_estimator.get_average_occupancy(),
        congestion_distribution_per_roi=pipeline.congestion_classifier.get_distribution(),
    )

    # 1. Export JSON
    json_path = output_dir / f"{sequence_id}_analytics.json"
    pipeline.export_to_json(json_path, report, telemetries, include_frame_telemetries=False)
    print(f"Exported analytics JSON: {json_path}")

    # 2. Export CSV
    csv_path = output_dir / f"{sequence_id}_telemetry.csv"
    pipeline.export_to_csv(csv_path, telemetries)
    print(f"Exported telemetry CSV: {csv_path}")

    # 3. Export Keyframes
    keyframes_dir = output_dir / "annotated_keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    if sequence_id == "MVI_20011":
        sample_indices = [50, 150, 300, 450, 600]
    else:
        sample_indices = [100, 300, 600, 900, 1200, 1500]

    telemetry_map = {t.frame_idx: t for t in telemetries}
    for k_idx in sample_indices:
        if k_idx in telemetry_map:
            img_file = visualizer._find_frame_image(image_dir, k_idx)
            if img_file:
                out_img = keyframes_dir / f"{sequence_id}_frame_{k_idx:05d}.jpg"
                visualizer.save_keyframe(
                    img_file,
                    tracks_by_frame[k_idx],
                    telemetry_map[k_idx],
                    out_img,
                    pipeline.counter.get_counts(),
                )
    print(f"Exported {len(sample_indices)} annotated keyframes to: {keyframes_dir}")

    # 4. Render Video
    video_path = output_dir / f"{sequence_id}_analytics.mp4"
    max_video_frames = 664 if sequence_id == "MVI_20011" else 500  # render 500 frames for 40131 for swift generation
    print(f"Rendering annotated MP4 video ({max_video_frames} frames)...")
    visualizer.render_video(
        image_dir,
        tracks_by_frame,
        telemetries,
        video_path,
        pipeline.counter.get_counts(),
        max_frames=max_video_frames,
    )
    print(f"Exported annotated MP4 video: {video_path}")

    stats = {
        "sequence_id": sequence_id,
        "total_frames": total_frames,
        "duration_s": duration_s,
        "total_time_s": total_time_s,
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "unique_vehicles": len(pipeline._unique_track_ids),
        "line_counts": pipeline.counter.get_counts(),
        "stopped_events_count": len(all_stopped),
        "peak_occupancy": pipeline.density_estimator.get_peak_occupancy(),
        "average_occupancy": pipeline.density_estimator.get_average_occupancy(),
        "congestion_distribution": pipeline.congestion_classifier.get_distribution(),
    }
    return stats


def main():
    root = Path(__file__).resolve().parents[2]
    configs = get_sequence_configs()

    results = {}
    for sid in ["MVI_20011", "MVI_40131"]:
        tracks_json = root / f"experiments/outputs/tracking/{sid}/{sid}_tracking.json"
        img_dir = root / f"data/raw/images/{sid}"
        out_dir = root / f"experiments/outputs/analytics/{sid}"

        stats = run_sequence_analytics(
            sequence_id=sid,
            tracks_json_path=tracks_json,
            image_dir=img_dir,
            output_dir=out_dir,
            config=configs[sid],
        )
        results[sid] = stats

    # Save summary stats
    summary_path = root / "experiments/outputs/analytics/analytics_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll analytics runs completed! Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
