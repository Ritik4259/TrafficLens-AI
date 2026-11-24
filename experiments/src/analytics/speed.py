"""
Vehicle speed estimation abstraction module.

Provides a modular architecture for vehicle speed estimation.
Because UA-DETRAC does not provide camera calibration matrices, lens intrinsics,
or metric road geometry, this module strictly avoids reporting fabricated km/h metrics.
Instead, it provides pixel displacement rates (px/s) and defines the exact interface
required for future planar homography-calibrated metric estimation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from experiments.src.tracking.tracker import Track
from .config import SpeedConfig
from .models import SpeedMeasurement


class BaseSpeedEstimator(ABC):
    """Abstract base class for vehicle speed estimation."""

    @property
    @abstractmethod
    def is_calibrated(self) -> bool:
        """Indicates whether the estimator produces calibrated metric units (km/h)."""
        pass

    @abstractmethod
    def estimate_speed(
        self,
        track: Track,
        frame_idx: int,
        timestamp_s: Optional[float] = None,
    ) -> SpeedMeasurement:
        """Computes speed measurement for a given track in the current frame."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets internal velocity smoothing buffers."""
        pass


class PixelSpeedEstimator(BaseSpeedEstimator):
    """
    Uncalibrated speed estimator computing smoothed 2D pixel displacement rate.
    
    Attributes:
        fps: Video frame rate.
        window_frames: Moving average window for trajectory displacement smoothing.
    """

    def __init__(self, config: Optional[SpeedConfig] = None):
        self.config = config or SpeedConfig()
        self.fps = self.config.fps
        self.window_frames = max(2, self.config.smoothing_window_frames)

    @property
    def is_calibrated(self) -> bool:
        return False

    def reset(self) -> None:
        pass

    def estimate_speed(
        self,
        track: Track,
        frame_idx: int,
        timestamp_s: Optional[float] = None,
    ) -> SpeedMeasurement:
        """
        Calculates pixel displacement speed (px/s) over recent trajectory history.
        """
        if len(track.history) < 2:
            return SpeedMeasurement(
                track_id=track.track_id,
                frame_idx=frame_idx,
                speed_px_per_s=0.0,
                speed_kmh=None,
                is_calibrated=False,
                calibration_note="Trajectory history too short (<2 points); speed 0.",
            )

        # Use recent history up to window_frames
        pts = track.history[-self.window_frames:]
        p_start = pts[0]
        p_end = pts[-1]
        elapsed_frames = len(pts) - 1
        elapsed_time_s = elapsed_frames / self.fps

        disp_px = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
        speed_px_per_s = disp_px / max(1e-4, elapsed_time_s)

        return SpeedMeasurement(
            track_id=track.track_id,
            frame_idx=frame_idx,
            speed_px_per_s=speed_px_per_s,
            speed_kmh=None,
            is_calibrated=False,
            calibration_note=(
                "Uncalibrated pixel displacement. Metric km/h unavailable without "
                "camera calibration matrix or road homography."
            ),
        )


class CalibratedSpeedEstimator(BaseSpeedEstimator):
    """
    Calibrated speed estimator utilizing a 3x3 planar homography matrix (H)
    to transform image coordinates (u, v) into world ground-plane coordinates (X, Y) in meters.
    """

    def __init__(self, homography_matrix: np.ndarray, fps: float = 25.0):
        self.H = np.asarray(homography_matrix, dtype=np.float64)
        if self.H.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got shape {self.H.shape}")
        self.fps = fps

    @property
    def is_calibrated(self) -> bool:
        return True

    def reset(self) -> None:
        pass

    def _image_to_world(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """Maps an image point (u, v) to ground-plane world point (X, Y) in meters."""
        vec = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
        world_homog = self.H @ vec
        if abs(world_homog[2]) < 1e-9:
            return 0.0, 0.0
        return float(world_homog[0] / world_homog[2]), float(world_homog[1] / world_homog[2])

    def estimate_speed(
        self,
        track: Track,
        frame_idx: int,
        timestamp_s: Optional[float] = None,
    ) -> SpeedMeasurement:
        """
        Estimates metric speed in km/h using calibrated planar homography projection.
        """
        if len(track.history) < 2:
            return SpeedMeasurement(
                track_id=track.track_id,
                frame_idx=frame_idx,
                speed_px_per_s=0.0,
                speed_kmh=0.0,
                is_calibrated=True,
                calibration_note="Calibrated planar homography projection.",
            )

        p_start_img = track.history[-2]
        p_end_img = track.history[-1]

        # Calculate pixel displacement
        disp_px = math.hypot(p_end_img[0] - p_start_img[0], p_end_img[1] - p_start_img[1])
        speed_px_per_s = disp_px * self.fps

        # Project image points to ground plane (meters)
        x1_m, y1_m = self._image_to_world(p_start_img)
        x2_m, y2_m = self._image_to_world(p_end_img)

        dist_m = math.hypot(x2_m - x1_m, y2_m - y1_m)
        speed_mps = dist_m * self.fps
        speed_kmh = speed_mps * 3.6

        return SpeedMeasurement(
            track_id=track.track_id,
            frame_idx=frame_idx,
            speed_px_per_s=speed_px_per_s,
            speed_kmh=speed_kmh,
            is_calibrated=True,
            calibration_note="Calibrated planar homography projection.",
        )
