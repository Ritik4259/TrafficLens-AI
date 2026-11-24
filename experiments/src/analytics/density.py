"""
ROI-based traffic density estimation module.

Computes active vehicle counts, bounding-box spatial occupancy ratios,
and density time series across configurable polygonal road regions.

Semantics & Terminology:
- Vehicle Count (veh): Discrete count of unique active vehicle tracks in the ROI.
- Occupancy Ratio ([0.0, 1.0]): Spatial ratio of the ROI area covered by vehicle 2D footprints.
- Physical Traffic Density (veh/km/lane): Calibrated traffic engineering metric.
  NOT reported without metric camera calibration and lane geometry.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any

from experiments.src.tracking.tracker import Track
from .config import DensityConfig, ROIConfig
from .geometry import point_in_polygon, polygon_area, box_polygon_intersection_area
from .models import ROIDensity


class ROIDensityEstimator:
    """
    Estimates vehicle count and spatial occupancy ratios for polygonal ROIs.
    """

    def __init__(
        self,
        rois: List[ROIConfig],
        config: Optional[DensityConfig] = None,
        fps: float = 25.0,
    ):
        self.rois = {roi.roi_id: roi for roi in rois}
        self.config = config or DensityConfig()
        self.fps = fps
        # Precompute ROI areas
        self.roi_areas: Dict[str, float] = {
            roi.roi_id: polygon_area(roi.polygon) for roi in rois
        }
        # Time-series history: roi_id -> List[ROIDensity]
        self._history: Dict[str, List[ROIDensity]] = {roi.roi_id: [] for roi in rois}

    def reset(self) -> None:
        """Clears accumulated density time series."""
        for roi_id in self.rois.keys():
            self._history[roi_id].clear()

    def update(
        self,
        tracks: List[Track],
        frame_idx: int,
        timestamp_s: Optional[float] = None,
    ) -> List[ROIDensity]:
        """
        Calculates density metrics for each configured ROI in the current frame.
        
        Args:
            tracks: List of active Track instances in the current frame.
            frame_idx: Current frame index.
            timestamp_s: Optional timestamp in seconds.
            
        Returns:
            List of ROIDensity instances for the frame.
        """
        if timestamp_s is None:
            timestamp_s = max(0.0, (frame_idx - 1) / self.fps)

        frame_densities: List[ROIDensity] = []

        for roi_id, roi_cfg in self.rois.items():
            roi_area = self.roi_areas.get(roi_id, 0.0)
            if roi_area <= 0:
                continue

            active_vehicle_ids: List[int] = []
            total_overlap_area = 0.0

            for track in tracks:
                # Calculate bounding box intersection with ROI
                overlap_area = box_polygon_intersection_area(track.bbox, roi_cfg.polygon)
                
                # Check presence: centroid inside ROI or non-zero box intersection
                is_inside = point_in_polygon(track.center, roi_cfg.polygon) or overlap_area > 0.0

                if is_inside:
                    active_vehicle_ids.append(track.track_id)
                if overlap_area > 0.0:
                    total_overlap_area += overlap_area

            # Calculate occupancy ratio capped at 1.0 (to handle overlapping bounding boxes)
            occupancy_ratio = min(1.0, total_overlap_area / max(1e-6, roi_area))

            density_record = ROIDensity(
                roi_id=roi_id,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                vehicle_count=len(active_vehicle_ids),
                occupancy_ratio=occupancy_ratio,
                vehicle_ids=active_vehicle_ids,
            )

            frame_densities.append(density_record)
            self._history[roi_id].append(density_record)

        return frame_densities

    def get_peak_occupancy(self) -> Dict[str, float]:
        """Returns peak occupancy ratio observed per ROI."""
        peak = {}
        for roi_id, history in self._history.items():
            peak[roi_id] = max((r.occupancy_ratio for r in history), default=0.0)
        return peak

    def get_average_occupancy(self) -> Dict[str, float]:
        """Returns average occupancy ratio observed per ROI."""
        avg = {}
        for roi_id, history in self._history.items():
            if history:
                avg[roi_id] = sum(r.occupancy_ratio for r in history) / len(history)
            else:
                avg[roi_id] = 0.0
        return avg
