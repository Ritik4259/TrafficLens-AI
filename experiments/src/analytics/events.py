"""
Stopped-vehicle and traffic event detection module.

Tracks trajectory displacements across sliding temporal windows to detect
stationary vehicles, maintains event lifecycles (active -> resolved), and
associates events with spatial regions of interest (ROIs).
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Set, Tuple, Any

from experiments.src.tracking.tracker import Track
from .config import StoppedVehicleConfig, ROIConfig
from .geometry import point_in_polygon
from .models import StoppedVehicleEvent


class StoppedVehicleDetector:
    """
    Detects stopped vehicles based on spatial trajectory displacement over time.
    
    Guarantees:
    - Purely displacement-based (independent of detector confidence fluctuations).
    - Thresholded against tracking jitter (e.g. bounding box noise / camera shake).
    - Lifecycle management: creation, duration updates, and clean resolution upon resumption of motion or departure.
    - Strict avoidance of duplicate active events for the same vehicle track.
    """

    def __init__(
        self,
        config: Optional[StoppedVehicleConfig] = None,
        rois: Optional[List[ROIConfig]] = None,
        fps: float = 25.0,
    ):
        self.config = config or StoppedVehicleConfig()
        self.rois = rois or []
        self.fps = fps

        # Track trajectory buffer: track_id -> List[Tuple[int, float, Tuple[float, float]]]
        # Elements are: (frame_idx, timestamp_s, (cx, cy))
        self._track_buffers: Dict[int, List[Tuple[int, float, Tuple[float, float]]]] = {}

        # Active stopped events: track_id -> StoppedVehicleEvent
        self._active_events: Dict[int, StoppedVehicleEvent] = {}

        # Historical completed events
        self._completed_events: List[StoppedVehicleEvent] = []

    def reset(self) -> None:
        """Clears all tracking buffers, active events, and event histories."""
        self._track_buffers.clear()
        self._active_events.clear()
        self._completed_events.clear()

    def _find_roi_for_point(self, point: Tuple[float, float]) -> Optional[str]:
        """Finds the first ROI polygon containing the point."""
        for roi in self.rois:
            if point_in_polygon(point, roi.polygon):
                return roi.roi_id
        return None

    def update(
        self,
        tracks: List[Track],
        frame_idx: int,
        timestamp_s: Optional[float] = None,
    ) -> Tuple[List[StoppedVehicleEvent], List[StoppedVehicleEvent]]:
        """
        Processes active tracks for a frame, updating stationary states and event lifecycles.
        
        Args:
            tracks: List of active Track instances in the current frame.
            frame_idx: Current frame index.
            timestamp_s: Optional timestamp in seconds.
            
        Returns:
            (active_events_snapshot, newly_resolved_events)
        """
        if timestamp_s is None:
            timestamp_s = max(0.0, (frame_idx - 1) / self.fps)

        current_track_ids = {t.track_id for t in tracks}
        newly_resolved_events: List[StoppedVehicleEvent] = []

        # 1. Update buffers for active tracks
        for track in tracks:
            tid = track.track_id
            pos = track.center

            if tid not in self._track_buffers:
                self._track_buffers[tid] = []
            self._track_buffers[tid].append((frame_idx, timestamp_s, pos))

            # Maintain sliding window (need at least ceil(min_duration * fps) + 5 frames to observe full duration)
            max_buf_len = max(
                int(math.ceil(self.config.min_duration_seconds * self.fps)) + 5,
                self.config.sliding_window_frames + 5,
            )
            if len(self._track_buffers[tid]) > max_buf_len:
                self._track_buffers[tid] = self._track_buffers[tid][-max_buf_len:]

            buf = self._track_buffers[tid]
            time_span_s = buf[-1][1] - buf[0][1]

            # Check if track has been observed for at least min_duration_seconds
            if time_span_s >= self.config.min_duration_seconds - 1e-4:
                # Calculate maximum displacement relative to window start position
                anchor_pos = buf[0][2]
                max_disp = max(math.hypot(p[2][0] - anchor_pos[0], p[2][1] - anchor_pos[1]) for p in buf)

                is_stationary = max_disp <= self.config.max_displacement_px

                if is_stationary:
                    if tid not in self._active_events:
                        # Initiate new stopped vehicle event
                        roi_id = self._find_roi_for_point(pos)
                        event = StoppedVehicleEvent(
                            track_id=tid,
                            start_frame=buf[0][0],
                            start_timestamp_s=buf[0][1],
                            duration_s=time_span_s,
                            location=pos,
                            roi_id=roi_id,
                            status="active",
                        )
                        self._active_events[tid] = event
                    else:
                        # Update ongoing active event
                        ev = self._active_events[tid]
                        ev.duration_s = timestamp_s - ev.start_timestamp_s
                        ev.location = pos
                else:
                    # Vehicle was stopped but has resumed motion
                    if tid in self._active_events:
                        ev = self._active_events.pop(tid)
                        ev.end_frame = frame_idx
                        ev.end_timestamp_s = timestamp_s
                        ev.duration_s = timestamp_s - ev.start_timestamp_s
                        ev.status = "resolved"
                        self._completed_events.append(ev)
                        newly_resolved_events.append(ev)

        # 2. Handle terminated tracks (left the field of view while stopped)
        terminated_ids = [tid for tid in self._active_events.keys() if tid not in current_track_ids]
        for tid in terminated_ids:
            ev = self._active_events.pop(tid)
            last_entry = self._track_buffers[tid][-1] if tid in self._track_buffers and self._track_buffers[tid] else None
            ev.end_frame = last_entry[0] if last_entry else frame_idx
            ev.end_timestamp_s = last_entry[1] if last_entry else timestamp_s
            ev.duration_s = ev.end_timestamp_s - ev.start_timestamp_s
            ev.status = "resolved"
            self._completed_events.append(ev)
            newly_resolved_events.append(ev)

        # Clean up stale track buffers for tracks absent for > 150 frames
        absent_ids = [
            tid for tid, buf in self._track_buffers.items()
            if tid not in current_track_ids and (frame_idx - buf[-1][0]) > 150
        ]
        for tid in absent_ids:
            del self._track_buffers[tid]

        active_snapshot = list(self._active_events.values())
        return active_snapshot, newly_resolved_events

    def get_all_events(self) -> List[StoppedVehicleEvent]:
        """Returns all completed and currently active stopped vehicle events."""
        return list(self._completed_events) + list(self._active_events.values())
