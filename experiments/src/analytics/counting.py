"""
Virtual counting line and directional vehicle crossing detection module.

Provides robust trajectory-based line crossing detection, directional resolution,
debouncing to prevent boundary jitter duplicates, and independent multi-line counting.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any

from experiments.src.tracking.tracker import Track
from .config import CountingLineConfig
from .geometry import segments_intersect, determine_crossing_direction
from .models import CrossingEvent


class LineCrossingCounter:
    """
    Detects and records directional vehicle crossings over virtual lines.
    
    Guarantees:
    - Trajectory segment intersection verification.
    - Deterministic direction assignment via vector geometry.
    - At most one count per track per configured line and direction.
    - Debounce cooldown to prevent jitter-induced multi-counts.
    - Clean sequence isolation via reset().
    """

    def __init__(self, lines: List[CountingLineConfig], fps: float = 25.0):
        self.lines = {line.line_id: line for line in lines}
        self.fps = fps
        # Map: (line_id, track_id) -> {"last_frame": int, "directions_counted": Set[str]}
        self._crossing_history: Dict[Tuple[str, int], Dict[str, Any]] = {}
        # Cumulative counts: line_id -> direction -> count
        self._counts: Dict[str, Dict[str, int]] = {
            line.line_id: {
                line.positive_direction_label: 0,
                line.negative_direction_label: 0,
                "total": 0,
            }
            for line in lines
        }
        # Chronological log of crossing events
        self.events: List[CrossingEvent] = []

    def reset(self) -> None:
        """Resets all crossing states, histories, and cumulative counts."""
        self._crossing_history.clear()
        self.events.clear()
        for line_id, line in self.lines.items():
            self._counts[line_id] = {
                line.positive_direction_label: 0,
                line.negative_direction_label: 0,
                "total": 0,
            }

    def update(
        self,
        tracks: List[Track],
        frame_idx: int,
        timestamp_s: Optional[float] = None,
    ) -> List[CrossingEvent]:
        """
        Processes active tracks for a single frame and detects line crossings.
        
        Args:
            tracks: List of active Track instances in the current frame.
            frame_idx: Current frame number.
            timestamp_s: Optional timestamp in seconds (defaults to frame_idx / fps).
            
        Returns:
            List of new CrossingEvents detected in this frame.
        """
        if timestamp_s is None:
            timestamp_s = max(0.0, (frame_idx - 1) / self.fps)

        frame_events: List[CrossingEvent] = []

        for track in tracks:
            # Require minimum trajectory history to form a segment
            if len(track.history) < 2:
                continue

            prev_pos = track.history[-2]
            curr_pos = track.history[-1]

            # If track hasn't moved between frames, skip
            if prev_pos == curr_pos:
                continue

            for line_id, line_cfg in self.lines.items():
                # Enforce track maturity if configured
                if track.age < line_cfg.min_track_age:
                    continue

                # Check intersection between trajectory segment and virtual line
                hit, intersect_pt = segments_intersect(
                    prev_pos,
                    curr_pos,
                    line_cfg.p1,
                    line_cfg.p2,
                )
                if not hit or intersect_pt is None:
                    continue

                # Determine crossing direction
                direction = determine_crossing_direction(
                    prev_pos,
                    curr_pos,
                    line_cfg.p1,
                    line_cfg.p2,
                    positive_label=line_cfg.positive_direction_label,
                    negative_label=line_cfg.negative_direction_label,
                )

                # Check debouncing and single-count per direction rules
                hist_key = (line_id, track.track_id)
                track_hist = self._crossing_history.get(hist_key)

                if track_hist is not None:
                    # Enforce cooldown against boundary jitter
                    if frame_idx - track_hist["last_frame"] < line_cfg.cooldown_frames:
                        continue
                    # Enforce at most once per line and direction
                    if direction in track_hist["directions_counted"]:
                        continue
                else:
                    track_hist = {"last_frame": 0, "directions_counted": set()}
                    self._crossing_history[hist_key] = track_hist

                # Record crossing
                track_hist["last_frame"] = frame_idx
                track_hist["directions_counted"].add(direction)

                event = CrossingEvent(
                    line_id=line_id,
                    track_id=track.track_id,
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    direction=direction,
                    crossing_point=intersect_pt,
                )
                frame_events.append(event)
                self.events.append(event)

                # Update cumulative statistics
                self._counts[line_id][direction] = self._counts[line_id].get(direction, 0) + 1
                self._counts[line_id]["total"] = self._counts[line_id].get("total", 0) + 1

        return frame_events

    def get_counts(self, line_id: Optional[str] = None) -> Dict[str, Any]:
        """Returns cumulative counts for a specific line or all lines."""
        if line_id is not None:
            return dict(self._counts.get(line_id, {}))
        return {lid: dict(counts) for lid, counts in self._counts.items()}
