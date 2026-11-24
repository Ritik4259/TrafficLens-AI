"""
Traffic flow calculation module.

Calculates vehicle throughput (vehicles per minute - VPM) over configurable
tumbling and rolling time windows, resolved by counting line and direction.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any

from .config import FlowConfig, CountingLineConfig
from .models import CrossingEvent, LineFlow


class FlowCalculator:
    """
    Computes traffic flow rates (Vehicles Per Minute) over configurable time windows.
    """

    def __init__(
        self,
        config: Optional[FlowConfig] = None,
        lines: Optional[List[CountingLineConfig]] = None,
    ):
        self.config = config or FlowConfig()
        self.lines = {line.line_id: line for line in (lines or [])}
        self._events: List[CrossingEvent] = []

    def reset(self) -> None:
        """Clears accumulated crossing events."""
        self._events.clear()

    def add_events(self, events: List[CrossingEvent]) -> None:
        """Appends new crossing events to the internal buffer."""
        self._events.extend(events)

    def calculate_window_flow(
        self,
        line_id: str,
        window_start_s: float,
        window_end_s: float,
        events: Optional[List[CrossingEvent]] = None,
    ) -> LineFlow:
        """
        Calculates flow rate for a specific counting line across a time window.
        
        Args:
            line_id: Virtual line identifier.
            window_start_s: Window start timestamp in seconds.
            window_end_s: Window end timestamp in seconds.
            events: Optional custom event list (defaults to internal event buffer).
            
        Returns:
            LineFlow structured model.
        """
        target_events = events if events is not None else self._events
        duration_s = max(1e-3, window_end_s - window_start_s)
        duration_min = duration_s / 60.0

        # Filter events for this line falling strictly within [window_start_s, window_end_s)
        window_events = [
            e for e in target_events
            if e.line_id == line_id and window_start_s <= e.timestamp_s < window_end_s
        ]

        total_count = len(window_events)
        flow_rate_vpm = total_count / duration_min

        direction_counts: Dict[str, int] = {}
        direction_flows_vpm: Dict[str, float] = {}

        for e in window_events:
            direction_counts[e.direction] = direction_counts.get(e.direction, 0) + 1

        for direction, count in direction_counts.items():
            direction_flows_vpm[direction] = count / duration_min

        return LineFlow(
            line_id=line_id,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            total_count=total_count,
            flow_rate_vpm=flow_rate_vpm,
            direction_counts=direction_counts,
            direction_flows_vpm=direction_flows_vpm,
        )

    def compute_sequence_tumbling_windows(
        self,
        total_duration_s: float,
        events: Optional[List[CrossingEvent]] = None,
    ) -> List[LineFlow]:
        """
        Partitions the entire sequence into discrete consecutive time windows
        and calculates flow for each configured line.
        """
        target_events = events if events is not None else self._events
        window_size = self.config.window_size_seconds
        results: List[LineFlow] = []

        if total_duration_s <= 0 or not self.lines:
            return results

        # Create tumbling windows: [0, W), [W, 2W), ...
        start = 0.0
        while start < total_duration_s:
            end = min(total_duration_s, start + window_size)
            for line_id in self.lines.keys():
                flow = self.calculate_window_flow(
                    line_id=line_id,
                    window_start_s=start,
                    window_end_s=end,
                    events=target_events,
                )
                results.append(flow)
            start += window_size

        return results

    def compute_current_rolling_flow(
        self,
        current_timestamp_s: float,
        window_size_s: Optional[float] = None,
    ) -> List[LineFlow]:
        """
        Computes rolling flow over the immediate preceding time window.
        """
        win_size = window_size_s or self.config.window_size_seconds
        start_s = max(0.0, current_timestamp_s - win_size)
        results: List[LineFlow] = []

        for line_id in self.lines.keys():
            flow = self.calculate_window_flow(
                line_id=line_id,
                window_start_s=start_s,
                window_end_s=current_timestamp_s,
            )
            results.append(flow)

        return results
