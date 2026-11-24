"""
Pretrained YOLO Vehicle Detector with Class-Agnostic Filtering and Batch Processing.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ultralytics import YOLO

from experiments.src.dataset.parser import BBox
from .config import (
    DetectionConfig,
    COCO_VEHICLE_CLASS_IDS,
    COCO_EXCLUDED_VEHICLE_IDS,
    COCO_CLASS_NAMES,
)
from .metrics import Prediction


def apply_class_agnostic_nms(
    predictions: List[Prediction],
    iou_threshold: float = 0.45,
) -> List[Prediction]:
    """
    Performs class-agnostic Non-Maximum Suppression (NMS) over vehicle predictions.
    
    If YOLO outputs multiple detections on the same physical vehicle under different
    COCO classes (e.g. both 'car' and 'truck' with IoU > iou_threshold),
    this collapses them to the single highest-confidence detection to prevent
    artificially inflating false positives in a unified-class evaluation.
    """
    if len(predictions) <= 1:
        return predictions

    # Sort descending by confidence
    sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)
    kept: List[Prediction] = []

    for pred in sorted_preds:
        suppress = False
        for kept_pred in kept:
            if pred.bbox.iou(kept_pred.bbox) > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(pred)

    return kept


class VehicleDetector:
    """Wrapper around Ultralytics YOLO with COCO-to-vehicle mapping."""

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        
        # Determine device
        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = self.config.device

        t0 = time.perf_counter()
        self.model = YOLO(self.config.model_name)
        # Warmup model once with a dummy run
        dummy = np.zeros((self.config.img_size, self.config.img_size, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=self.config.img_size, device=self.device, verbose=False)
        self.model_load_time_ms = (time.perf_counter() - t0) * 1000.0

    def _process_yolo_result(
        self,
        result,
        conf_thresh: float,
        iou_thresh: float,
    ) -> List[Prediction]:
        """Parses a single Ultralytics Results object into class-agnostic vehicle Predictions."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        raw_preds: List[Prediction] = []
        for box in boxes:
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())

            if confidence < conf_thresh:
                continue

            # Filter strictly to target COCO vehicle classes (2: car, 3: motorcycle, 5: bus, 7: truck)
            # Exclude bicycles (1) and all non-road-vehicle classes
            if cls_id not in self.config.target_class_ids or cls_id in COCO_EXCLUDED_VEHICLE_IDS:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            bbox = BBox(left=x1, top=y1, width=max(0.0, x2 - x1), height=max(0.0, y2 - y1))

            raw_coco_name = COCO_CLASS_NAMES.get(cls_id, "unknown")
            raw_preds.append(
                Prediction(
                    bbox=bbox,
                    confidence=confidence,
                    class_name=self.config.unified_class_name,
                    raw_coco_class=raw_coco_name,
                )
            )

        if self.config.class_agnostic_nms:
            return apply_class_agnostic_nms(raw_preds, iou_threshold=iou_thresh)
        return raw_preds

    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> Tuple[List[Prediction], float]:
        """
        Runs vehicle detection on a single image.

        Returns:
            (List[Prediction], inference_latency_ms)
        """
        conf_thresh = conf if conf is not None else self.config.confidence_threshold
        iou_thresh = iou if iou is not None else self.config.nms_iou_threshold

        t0 = time.perf_counter()
        results = self.model.predict(
            source=image_input,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=self.config.img_size,
            device=self.device,
            verbose=False,
        )
        inference_latency_ms = (time.perf_counter() - t0) * 1000.0

        if not results:
            return [], inference_latency_ms

        final_preds = self._process_yolo_result(results[0], conf_thresh, iou_thresh)
        return final_preds, inference_latency_ms

    def predict_sequence(
        self,
        image_paths: List[Path],
        conf: Optional[float] = None,
        batch_size: int = 16,
        max_frames: Optional[int] = None,
    ) -> Tuple[Dict[int, List[Prediction]], Dict[str, float]]:
        """
        Runs batched inference over an ordered sequence of frame images.

        Returns:
            (predictions_by_frame_idx, performance_stats)
        """
        conf_thresh = conf if conf is not None else self.config.confidence_threshold
        iou_thresh = self.config.nms_iou_threshold

        selected_paths = image_paths[:max_frames] if max_frames else image_paths
        total_frames = len(selected_paths)

        preds_by_frame: Dict[int, List[Prediction]] = {}
        batch_latencies_ms: List[float] = []

        t_start = time.perf_counter()

        # Process in batches
        for batch_start in range(0, total_frames, batch_size):
            batch_slice = selected_paths[batch_start : batch_start + batch_size]
            batch_inputs = [str(p) for p in batch_slice]

            t0 = time.perf_counter()
            results = self.model.predict(
                source=batch_inputs,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=self.config.img_size,
                device=self.device,
                verbose=False,
            )
            batch_lat_ms = (time.perf_counter() - t0) * 1000.0
            batch_latencies_ms.append(batch_lat_ms)

            for offset, r in enumerate(results):
                frame_idx = batch_start + offset + 1
                preds_by_frame[frame_idx] = self._process_yolo_result(r, conf_thresh, iou_thresh)

        total_elapsed_s = time.perf_counter() - t_start
        total_inference_s = sum(batch_latencies_ms) / 1000.0

        stats = {
            "total_frames": total_frames,
            "total_inference_s": total_inference_s,
            "total_elapsed_s": total_elapsed_s,
            "avg_latency_ms": float(total_inference_s / max(1, total_frames) * 1000.0),
            "fps": float(total_frames / total_inference_s) if total_inference_s > 0 else 0.0,
            "device": self.device,
            "img_size": self.config.img_size,
            "batch_size": batch_size,
            "model_load_time_ms": self.model_load_time_ms,
        }
        return preds_by_frame, stats
