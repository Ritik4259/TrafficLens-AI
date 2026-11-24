"""
Rule-based traffic congestion estimation module.

Provides an explainable, configurable multi-metric heuristic baseline classifying
traffic into LOW, MODERATE, or HIGH congestion with human-readable rationale.

Note:
This rule-based classifier serves as an explainable heuristic baseline and
requires empirical calibration against real-world ground-truth traffic data.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Set, Tuple, Any

from experiments.src.tracking.tracker import Track
from .config import CongestionConfig
from .models import CongestionLevel, CongestionState, ROIDensity


class CongestionClassifier:
    """
    Explainable rule-based classifier evaluating ROI congestion state.
    """

    def __init__(self, config: Optional[CongestionConfig] = None, fps: float = 25.0):
        self.config = config or CongestionConfig()
        self.fps = fps
        # Accumulator: roi_id -> {level -> count}
        self._level_counts: Dict[str, Dict[str, int]] = {}

    def reset(self) -> None:
        """Clears accumulated congestion classifications."""
        self._level_counts.clear()

    def classify_roi(
        self,
        roi_density: ROIDensity,
        tracks_in_roi: List[Track],
        stopped_track_ids: Set[int],
    ) -> CongestionState:
        """
        Classifies congestion level for a single ROI in the current frame.
        
        Args:
            roi_density: Current density metrics for the ROI.
            tracks_in_roi: Active Track objects currently in the ROI.
            stopped_track_ids: Set of track IDs currently classified as stopped.
            
        Returns:
            CongestionState with classification level, supporting measurements, and reason.
        """
        roi_id = roi_density.roi_id
        occupancy = roi_density.occupancy_ratio
        vehicle_count = roi_density.vehicle_count

        # Count stopped vehicles inside this ROI
        stopped_in_roi = [tid for tid in roi_density.vehicle_ids if tid in stopped_track_ids]
        stopped_count = len(stopped_in_roi)
        stopped_ratio = stopped_count / max(1, vehicle_count) if vehicle_count > 0 else 0.0

        # Calculate average displacement across vehicles in ROI
        displacements_px: List[float] = []
        for t in tracks_in_roi:
            if len(t.history) >= 2:
                p_prev = t.history[-2]
                p_curr = t.history[-1]
                disp = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
                displacements_px.append(disp)

        avg_disp_frame = sum(displacements_px) / len(displacements_px) if displacements_px else 0.0
        avg_speed_px_s = avg_disp_frame * self.fps

        # Classification decision logic
        has_speed_data = len(displacements_px) > 0

        if vehicle_count < self.config.min_vehicles_for_congestion:
            level = CongestionLevel.LOW
            reason = (
                f"Low congestion: Active vehicles ({vehicle_count}) below threshold "
                f"({self.config.min_vehicles_for_congestion}); free-flow condition."
            )
        elif (
            occupancy >= self.config.high_occupancy_thresh
            or (stopped_ratio >= self.config.stopped_ratio_thresh and occupancy >= self.config.low_occupancy_thresh)
            or (has_speed_data and avg_speed_px_s < self.config.low_speed_thresh_px_s and occupancy >= 0.30)
        ):
            level = CongestionLevel.HIGH
            if stopped_ratio >= self.config.stopped_ratio_thresh:
                reason = (
                    f"High congestion: Elevated stopped vehicle ratio ({stopped_count}/{vehicle_count} = "
                    f"{stopped_ratio:.0%}) with spatial occupancy {occupancy:.1%}."
                )
            elif occupancy >= self.config.high_occupancy_thresh:
                reason = (
                    f"High congestion: Spatial occupancy ({occupancy:.1%}) exceeds high threshold "
                    f"({self.config.high_occupancy_thresh:.1%}) with low average displacement ({avg_speed_px_s:.1f} px/s)."
                )
            else:
                reason = (
                    f"High congestion: Queued slow traffic (avg speed {avg_speed_px_s:.1f} px/s) "
                    f"with occupancy {occupancy:.1%}."
                )
        elif (
            occupancy >= self.config.low_occupancy_thresh
            or (stopped_count >= 1 and vehicle_count >= self.config.min_vehicles_for_congestion)
        ):
            level = CongestionLevel.MODERATE
            reason = (
                f"Moderate congestion: Occupancy ({occupancy:.1%}) exceeds baseline threshold "
                f"with {vehicle_count} vehicles present (stopped: {stopped_count})."
            )
        else:
            level = CongestionLevel.LOW
            reason = (
                f"Low congestion: Free-flowing traffic (occupancy {occupancy:.1%}, "
                f"avg speed {avg_speed_px_s:.1f} px/s)."
            )

        # Update distribution counts
        if roi_id not in self._level_counts:
            self._level_counts[roi_id] = {l.value: 0 for l in CongestionLevel}
        self._level_counts[roi_id][level.value] += 1

        return CongestionState(
            roi_id=roi_id,
            frame_idx=roi_density.frame_idx,
            timestamp_s=roi_density.timestamp_s,
            level=level,
            occupancy_ratio=occupancy,
            vehicle_count=vehicle_count,
            avg_displacement_px=avg_disp_frame,
            stopped_vehicle_count=stopped_count,
            reason=reason,
        )

    def get_distribution(self) -> Dict[str, Dict[str, int]]:
        """Returns the distribution of congestion levels per ROI."""
        return {roi_id: dict(counts) for roi_id, counts in self._level_counts.items()}
