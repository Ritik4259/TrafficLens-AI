"""
Traffic analytics pipeline orchestration module.

Orchestrates line crossing counting, flow calculation, ROI density estimation,
explainable congestion classification, stopped-vehicle event detection, and speed estimation.
Maintains sequence-level isolation and exports structured telemetry to JSON and CSV.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from experiments.src.tracking.tracker import Track
from .config import AnalyticsConfig
from .counting import LineCrossingCounter
from .flow import FlowCalculator
from .density import ROIDensityEstimator
from .congestion import CongestionClassifier
from .events import StoppedVehicleDetector
from .speed import BaseSpeedEstimator, PixelSpeedEstimator, CalibratedSpeedEstimator
from .models import FrameTelemetry, SequenceAnalyticsReport, LineFlow, StoppedVehicleEvent


class TrafficAnalyticsPipeline:
    """
    Unified analytics pipeline processing vehicle tracks per frame and per sequence.
    """

    def __init__(self, config: Optional[AnalyticsConfig] = None):
        self.config = config or AnalyticsConfig()

        self.counter = LineCrossingCounter(self.config.lines, fps=self.config.fps)
        self.flow_calc = FlowCalculator(self.config.flow, self.config.lines)
        self.density_estimator = ROIDensityEstimator(self.config.rois, self.config.density, fps=self.config.fps)
        self.congestion_classifier = CongestionClassifier(self.config.congestion, fps=self.config.fps)
        self.stopped_detector = StoppedVehicleDetector(self.config.stopped_vehicle, self.config.rois, fps=self.config.fps)

        if self.config.speed.homography_matrix is not None:
            self.speed_estimator: BaseSpeedEstimator = CalibratedSpeedEstimator(
                self.config.speed.homography_matrix, fps=self.config.fps
            )
        else:
            self.speed_estimator = PixelSpeedEstimator(self.config.speed)

        self.frame_telemetries: List[FrameTelemetry] = []
        self._unique_track_ids: Set[int] = set()

    def reset(self) -> None:
        """Resets all analytics components and buffers for sequence boundary isolation."""
        self.counter.reset()
        self.flow_calc.reset()
        self.density_estimator.reset()
        self.congestion_classifier.reset()
        self.stopped_detector.reset()
        self.speed_estimator.reset()
        self.frame_telemetries.clear()
        self._unique_track_ids.clear()

    def process_frame(
        self,
        tracks: List[Track],
        frame_idx: int,
        timestamp_s: Optional[float] = None,
    ) -> FrameTelemetry:
        """
        Executes the full analytics suite for a single frame.
        
        Args:
            tracks: List of active Track instances in the frame.
            frame_idx: Frame index (1-based or 0-based).
            timestamp_s: Optional timestamp in seconds.
            
        Returns:
            FrameTelemetry object encapsulating all measurements for the frame.
        """
        if timestamp_s is None:
            timestamp_s = max(0.0, (frame_idx - 1) / self.config.fps)

        # Track unique vehicle IDs seen
        for t in tracks:
            self._unique_track_ids.add(t.track_id)

        # 1. Line crossings
        crossings = self.counter.update(tracks, frame_idx, timestamp_s)
        if crossings:
            self.flow_calc.add_events(crossings)

        # 2. ROI density
        roi_densities = self.density_estimator.update(tracks, frame_idx, timestamp_s)

        # 3. Stopped vehicle events
        active_stopped, resolved_stopped = self.stopped_detector.update(tracks, frame_idx, timestamp_s)
        stopped_track_ids = {e.track_id for e in active_stopped}

        # 4. Congestion classification
        congestion_states = []
        for density in roi_densities:
            tracks_in_roi = [t for t in tracks if t.track_id in density.vehicle_ids]
            state = self.congestion_classifier.classify_roi(density, tracks_in_roi, stopped_track_ids)
            congestion_states.append(state)

        # 5. Speed measurements
        speed_measurements = [
            self.speed_estimator.estimate_speed(t, frame_idx, timestamp_s)
            for t in tracks
        ]

        telemetry = FrameTelemetry(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            crossings=crossings,
            roi_densities=roi_densities,
            congestion_states=congestion_states,
            active_stopped_events=active_stopped,
            resolved_stopped_events=resolved_stopped,
            speed_measurements=speed_measurements,
            active_track_count=len(tracks),
        )

        self.frame_telemetries.append(telemetry)
        return telemetry

    def process_sequence(
        self,
        tracks_by_frame: Dict[int, List[Track]],
        sequence_name: str = "sequence",
    ) -> Tuple[List[FrameTelemetry], SequenceAnalyticsReport]:
        """
        Processes an entire sequence of tracked frames. Enforces clean sequence isolation.
        
        Args:
            tracks_by_frame: Mapping of frame index to active tracks.
            sequence_name: Sequence identifier.
            
        Returns:
            (List of FrameTelemetry, SequenceAnalyticsReport)
        """
        self.reset()
        sorted_frames = sorted(tracks_by_frame.keys())

        for f_idx in sorted_frames:
            self.process_frame(tracks_by_frame[f_idx], f_idx)

        total_frames = len(sorted_frames)
        duration_s = total_frames / self.config.fps if total_frames > 0 else 0.0

        # Sequence-level flow aggregation
        flow_windows = self.flow_calc.compute_sequence_tumbling_windows(duration_s)
        all_stopped = self.stopped_detector.get_all_events()

        report = SequenceAnalyticsReport(
            sequence_name=sequence_name,
            total_frames=total_frames,
            duration_s=duration_s,
            fps=self.config.fps,
            total_unique_vehicles=len(self._unique_track_ids),
            line_counts=self.counter.get_counts(),
            flow_windows=flow_windows,
            stopped_vehicle_events=all_stopped,
            peak_occupancy_per_roi=self.density_estimator.get_peak_occupancy(),
            average_occupancy_per_roi=self.density_estimator.get_average_occupancy(),
            congestion_distribution_per_roi=self.congestion_classifier.get_distribution(),
        )

        return list(self.frame_telemetries), report

    def export_to_json(
        self,
        output_path: Path,
        report: SequenceAnalyticsReport,
        telemetries: Optional[List[FrameTelemetry]] = None,
        include_frame_telemetries: bool = False,
    ) -> None:
        """Exports sequence report and optional frame-by-frame telemetry to structured JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {
            "summary": report.to_dict(),
        }
        if include_frame_telemetries and telemetries:
            data["frame_telemetries"] = [t.to_dict() for t in telemetries]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def export_to_csv(
        self,
        output_path: Path,
        telemetries: List[FrameTelemetry],
    ) -> None:
        """Exports frame-by-frame telemetry time series to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_idx",
                "timestamp_s",
                "active_track_count",
                "crossing_events_count",
                "roi_id",
                "roi_vehicle_count",
                "roi_occupancy_ratio",
                "congestion_level",
                "active_stopped_count",
            ])

            for t in telemetries:
                if t.roi_densities:
                    for d, c in zip(t.roi_densities, t.congestion_states):
                        writer.writerow([
                            t.frame_idx,
                            round(t.timestamp_s, 3),
                            t.active_track_count,
                            len(t.crossings),
                            d.roi_id,
                            d.vehicle_count,
                            round(d.occupancy_ratio, 4),
                            c.level.value,
                            len(t.active_stopped_events),
                        ])
                else:
                    writer.writerow([
                        t.frame_idx,
                        round(t.timestamp_s, 3),
                        t.active_track_count,
                        len(t.crossings),
                        "",
                        0,
                        0.0,
                        "",
                        len(t.active_stopped_events),
                    ])
