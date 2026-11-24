"""
ByteTrack multi-object vehicle tracker module.

Wraps the battle-tested BYTETracker engine, preserving raw low-confidence
detections down to the 0.05 floor, managing track lifecycles, and maintaining
persistent trajectories.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ultralytics.trackers.byte_tracker import BYTETracker

from experiments.src.dataset.parser import BBox
from experiments.src.detection.metrics import Prediction
from experiments.src.detection.detector import apply_class_agnostic_nms
from .config import TrackingConfig


@dataclass
class Track:
    """Represents an active vehicle track in a video frame."""
    track_id: int
    bbox: BBox
    confidence: float
    class_name: str = "vehicle"
    age: int = 1
    history: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        """Center coordinate (cx, cy) of the vehicle bounding box."""
        return self.bbox.center


class _TrackerInputWrapper:
    """NumPy-backed detections exposing conf, cls, and xywh for BYTETracker."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __getitem__(self, mask: np.ndarray) -> _TrackerInputWrapper:
        return _TrackerInputWrapper(self.xywh[mask], self.conf[mask], self.cls[mask])

    def __len__(self) -> int:
        return len(self.conf)


class VehicleTracker:
    """
    Multi-object vehicle tracker utilizing ByteTrack with Kalman filtering
    and two-stage high/low confidence association.
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig()
        self._init_byte_tracker()
        self._track_histories: Dict[int, List[Tuple[float, float]]] = {}
        self._track_ages: Dict[int, int] = {}
        self.frame_count: int = 0

    def _init_byte_tracker(self) -> None:
        args = SimpleNamespace(
            track_high_thresh=self.config.track_high_thresh,
            track_low_thresh=self.config.track_low_thresh,
            new_track_thresh=self.config.new_track_thresh,
            track_buffer=self.config.track_buffer,
            match_thresh=self.config.match_thresh,
            fuse_score=self.config.fuse_score,
        )
        self.byte_tracker = BYTETracker(args)

    def reset(self) -> None:
        """Resets tracker state, internal frame counters, and track IDs."""
        self._init_byte_tracker()
        self._track_histories.clear()
        self._track_ages.clear()
        self.frame_count = 0

    def update(
        self,
        predictions: List[Prediction],
        frame_idx: Optional[int] = None,
    ) -> List[Track]:
        """
        Updates the tracker with detections for a single frame.

        Args:
            predictions: Candidate vehicle detections (retaining scores down to floor).
            frame_idx: Optional frame index.

        Returns:
            List of active Track instances in the current frame.
        """
        self.frame_count += 1

        # Apply class-agnostic NMS to collapse duplicate vehicle predictions if configured
        filtered_preds = predictions
        if self.config.nms_iou_threshold is not None and len(filtered_preds) > 1:
            filtered_preds = apply_class_agnostic_nms(filtered_preds, iou_threshold=self.config.nms_iou_threshold)

        # Filter out invalid / degenerate bounding boxes
        valid_preds = [
            p for p in filtered_preds
            if p.bbox.width > 0 and p.bbox.height > 0 and p.bbox.area >= self.config.min_box_area
        ]

        if not valid_preds:
            # Empty frame: update Kalman filters with empty array
            empty_input = _TrackerInputWrapper(
                xywh=np.empty((0, 4), dtype=np.float32),
                conf=np.empty(0, dtype=np.float32),
                cls=np.empty(0, dtype=np.int32),
            )
            raw_tracks = self.byte_tracker.update(empty_input)
        else:
            xywh_list = []
            conf_list = []
            cls_list = []

            for p in valid_preds:
                # ByteTracker expects [cx, cy, width, height]
                cx, cy = p.bbox.center
                xywh_list.append([cx, cy, p.bbox.width, p.bbox.height])
                conf_list.append(p.confidence)
                cls_list.append(0)  # Unified vehicle class

            tracker_input = _TrackerInputWrapper(
                xywh=np.array(xywh_list, dtype=np.float32),
                conf=np.array(conf_list, dtype=np.float32),
                cls=np.array(cls_list, dtype=np.int32),
            )
            raw_tracks = self.byte_tracker.update(tracker_input)

        active_tracks: List[Track] = []

        if len(raw_tracks) > 0:
            for row in raw_tracks:
                # row format: [x1, y1, x2, y2, track_id, score, cls, idx]
                x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                track_id = int(row[4])
                score = float(row[5])

                bbox = BBox(left=x1, top=y1, width=max(1.0, x2 - x1), height=max(1.0, y2 - y1))
                center = bbox.center

                # Update trajectory history
                if track_id not in self._track_histories:
                    self._track_histories[track_id] = []
                self._track_histories[track_id].append(center)
                if len(self._track_histories[track_id]) > self.config.max_history_len:
                    self._track_histories[track_id] = self._track_histories[track_id][-self.config.max_history_len:]

                # Update track age
                self._track_ages[track_id] = self._track_ages.get(track_id, 0) + 1

                track = Track(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=score,
                    class_name="vehicle",
                    age=self._track_ages[track_id],
                    history=list(self._track_histories[track_id]),
                )
                active_tracks.append(track)

        return active_tracks

    def track_sequence(
        self,
        predictions_by_frame: Dict[int, List[Prediction]],
    ) -> Tuple[Dict[int, List[Track]], Dict[str, Any]]:
        """
        Runs tracking across a complete sequence. Resets tracker at sequence start.

        Returns:
            (tracks_by_frame, timing_stats)
        """
        self.reset()
        tracks_by_frame: Dict[int, List[Track]] = {}
        sorted_frames = sorted(predictions_by_frame.keys())

        start_time = time.perf_counter()
        frame_latencies_ms: List[float] = []

        for f_idx in sorted_frames:
            f_start = time.perf_counter()
            frame_preds = predictions_by_frame[f_idx]
            tracks = self.update(frame_preds, frame_idx=f_idx)
            f_elapsed = (time.perf_counter() - f_start) * 1000.0
            frame_latencies_ms.append(f_elapsed)
            tracks_by_frame[f_idx] = tracks

        total_time_s = time.perf_counter() - start_time
        total_frames = len(sorted_frames)
        fps = total_frames / max(1e-6, total_time_s)
        avg_latency_ms = float(np.mean(frame_latencies_ms)) if frame_latencies_ms else 0.0

        stats = {
            "total_frames": total_frames,
            "total_time_s": total_time_s,
            "avg_latency_ms": avg_latency_ms,
            "fps": fps,
            "unique_track_ids": len(self._track_histories),
        }

        return tracks_by_frame, stats
