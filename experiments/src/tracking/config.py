"""
Configuration dataclasses for ByteTrack multi-object tracking and evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrackingConfig:
    """Configuration parameters for ByteTrack vehicle tracking."""
    track_high_thresh: float = 0.50     # Threshold for stage 1 high-confidence association
    track_low_thresh: float = 0.10      # Threshold for stage 2 low-confidence association
    new_track_thresh: float = 0.60      # Threshold for activating a new track
    track_buffer: int = 30              # Frames to retain lost tracks before removal
    match_thresh: float = 0.80          # IoU matching threshold for Kalman association
    frame_rate: int = 25                # Video frame rate (UA-DETRAC standard)
    fuse_score: bool = False            # Whether to fuse detection score in Kalman association
    min_box_area: float = 10.0          # Minimum box area to track (filters degenerates)
    max_history_len: int = 60           # Number of historical trajectory points to store
    nms_iou_threshold: Optional[float] = 0.50  # NMS IoU threshold to collapse duplicate detections


@dataclass
class TrackingEvaluationConfig:
    """Configuration parameters for tracking benchmark evaluation."""
    iou_threshold: float = 0.50                 # Minimum IoU to consider track matching ground truth
    ignore_overlap_threshold: float = 0.50      # Overlap threshold with ignore regions to exclude FP
    hota_alpha_steps: int = 19                  # Standard 19 localization thresholds for HOTA
