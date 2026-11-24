"""
Detection and Evaluation Configurations for YOLO11n Baseline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# COCO 80-class mapping relevant to road vehicles:
# 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_VEHICLE_CLASS_IDS: Tuple[int, ...] = (2, 3, 5, 7)
COCO_EXCLUDED_VEHICLE_IDS: Tuple[int, ...] = (1,)  # 1: bicycle explicitly excluded

COCO_CLASS_NAMES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class DetectionConfig:
    """Inference parameters for YOLO vehicle detection."""
    model_name: str = "yolo11n.pt"
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.45
    img_size: int = 640
    device: str = "cpu"
    target_class_ids: Tuple[int, ...] = COCO_VEHICLE_CLASS_IDS
    unified_class_name: str = "vehicle"
    class_agnostic_nms: bool = True


@dataclass
class EvaluationConfig:
    """Ground-truth evaluation parameters."""
    primary_iou_threshold: float = 0.50
    iou_thresholds: List[float] = field(
        default_factory=lambda: [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 to 0.95
    )
    ignore_overlap_threshold: float = 0.50
    confidence_threshold: float = 0.25
    pr_curve_num_points: int = 101  # Standard 101-point COCO interpolation
