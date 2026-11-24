"""
Metrics computation module for vehicle detection evaluation.

Implements greedy 1-to-1 IoU matching, proper ignore-region exclusion rules,
and unconstrained COCO-style AP@0.50 and AP@0.50:0.95 calculation across
the full Precision-Recall curve independent of the operating confidence threshold.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from experiments.src.dataset.parser import BBox, IgnoredRegion


@dataclass
class Prediction:
    """Predicted vehicle detection."""
    bbox: BBox
    confidence: float
    class_name: str = "vehicle"
    raw_coco_class: Optional[str] = None


def compute_iou_matrix(preds: List[BBox], gts: List[BBox]) -> np.ndarray:
    """Computes M x N IoU matrix between M predicted boxes and N ground truth boxes."""
    if not preds or not gts:
        return np.zeros((len(preds), len(gts)), dtype=np.float32)

    ious = np.zeros((len(preds), len(gts)), dtype=np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            ious[i, j] = p.iou(g)
    return ious


def match_frame_detections(
    predictions: List[Prediction],
    ground_truth: List[BBox],
    ignored_region: Optional[IgnoredRegion] = None,
    iou_threshold: float = 0.50,
    ignore_overlap_threshold: float = 0.50,
) -> Tuple[List[bool], List[bool], List[bool]]:
    """
    Performs greedy 1-to-1 IoU matching for a single frame.

    Evaluation Protocol:
    1. Predictions are processed in descending order of confidence.
    2. Each prediction attempts to greedily match the available ground-truth box
       with the highest IoU >= iou_threshold.
    3. If matched, the prediction is marked as a True Positive (TP), and the
       ground-truth box is marked as claimed (strictly 1-to-1 matching).
    4. If the prediction DOES NOT match any ground-truth box:
       - If its area overlap ratio with the ignored region >= ignore_overlap_threshold,
         it is marked as 'ignored' (NOT a False Positive).
       - Otherwise, it is marked as a False Positive (FP).

    Returns:
        (is_tp_list, is_fp_list, is_ignored_list) aligned with predictions.
    """
    num_preds = len(predictions)
    is_tp = [False] * num_preds
    is_fp = [False] * num_preds
    is_ignored = [False] * num_preds

    if num_preds == 0:
        return is_tp, is_fp, is_ignored

    # Sort indices by confidence descending
    sorted_pred_indices = sorted(range(num_preds), key=lambda i: predictions[i].confidence, reverse=True)
    gt_matched = [False] * len(ground_truth)

    for pred_idx in sorted_pred_indices:
        pred_box = predictions[pred_idx].bbox
        best_iou = -1.0
        best_gt_idx = -1

        # Attempt to match an unclaimed ground-truth box FIRST
        for gt_idx, gt_box in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue
            iou = pred_box.iou(gt_box)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            is_tp[pred_idx] = True
            gt_matched[best_gt_idx] = True
        else:
            # Unmatched candidate: check if it falls inside an ignore region
            if ignored_region is not None and ignored_region.is_box_ignored(
                pred_box, threshold=ignore_overlap_threshold
            ):
                is_ignored[pred_idx] = True
            else:
                is_fp[pred_idx] = True

    return is_tp, is_fp, is_ignored


def compute_ap_101_point(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Computes Average Precision (AP) using standard 101-point COCO interpolation:
    AP = (1/101) * sum(max(p(r)) for r in [0, 0.01, ..., 1.0]).
    """
    if len(recall) == 0 or len(precision) == 0:
        return 0.0

    recall_thresholds = np.linspace(0.0, 1.0, 101)
    precision_interpolated = []

    for r_thresh in recall_thresholds:
        precisions_at_or_above_r = precision[recall >= r_thresh]
        if len(precisions_at_or_above_r) > 0:
            precision_interpolated.append(np.max(precisions_at_or_above_r))
        else:
            precision_interpolated.append(0.0)

    return float(np.mean(precision_interpolated))


def compute_pr_curve_and_ap(
    predictions_by_frame: Dict[int, List[Prediction]],
    ground_truth_by_frame: Dict[int, List[BBox]],
    ignored_region: Optional[IgnoredRegion] = None,
    iou_threshold: float = 0.50,
    ignore_overlap_threshold: float = 0.50,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes unconstrained Average Precision (AP) across all candidate predictions
    independent of any operational confidence cutoff.
    """
    total_gts = sum(len(gts) for gts in ground_truth_by_frame.values())
    if total_gts == 0:
        return 0.0, np.array([]), np.array([])

    all_scores = []
    all_is_tp = []
    all_is_fp = []

    all_frames = sorted(set(predictions_by_frame.keys()) | set(ground_truth_by_frame.keys()))

    for frame_num in all_frames:
        frame_preds = predictions_by_frame.get(frame_num, [])
        frame_gts = ground_truth_by_frame.get(frame_num, [])

        is_tp, is_fp, is_ign = match_frame_detections(
            predictions=frame_preds,
            ground_truth=frame_gts,
            ignored_region=ignored_region,
            iou_threshold=iou_threshold,
            ignore_overlap_threshold=ignore_overlap_threshold,
        )

        for p, tp, fp, ign in zip(frame_preds, is_tp, is_fp, is_ign):
            if ign:
                continue
            all_scores.append(p.confidence)
            all_is_tp.append(1 if tp else 0)
            all_is_fp.append(1 if fp else 0)

    if not all_scores:
        return 0.0, np.array([]), np.array([])

    sort_indices = np.argsort(all_scores)[::-1]
    sorted_tp = np.array(all_is_tp)[sort_indices]
    sorted_fp = np.array(all_is_fp)[sort_indices]

    cum_tp = np.cumsum(sorted_tp)
    cum_fp = np.cumsum(sorted_fp)

    recalls = cum_tp / total_gts
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

    ap = compute_ap_101_point(recalls, precisions)
    return ap, recalls, precisions


def compute_operating_point_metrics(
    predictions_by_frame: Dict[int, List[Prediction]],
    ground_truth_by_frame: Dict[int, List[BBox]],
    ignored_region: Optional[IgnoredRegion] = None,
    iou_threshold: float = 0.50,
    confidence_threshold: float = 0.25,
    ignore_overlap_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Computes Precision, Recall, F1, TP, FP, FN at a specific operating confidence threshold.
    """
    total_gts = sum(len(gts) for gts in ground_truth_by_frame.values())
    total_tp = 0
    total_fp = 0
    total_ignored = 0

    all_frames = sorted(set(predictions_by_frame.keys()) | set(ground_truth_by_frame.keys()))

    for frame_num in all_frames:
        # Filter predictions at operating confidence threshold
        frame_preds = [
            p for p in predictions_by_frame.get(frame_num, [])
            if p.confidence >= confidence_threshold
        ]
        frame_gts = ground_truth_by_frame.get(frame_num, [])

        is_tp, is_fp, is_ign = match_frame_detections(
            predictions=frame_preds,
            ground_truth=frame_gts,
            ignored_region=ignored_region,
            iou_threshold=iou_threshold,
            ignore_overlap_threshold=ignore_overlap_threshold,
        )

        total_tp += sum(is_tp)
        total_fp += sum(is_fp)
        total_ignored += sum(is_ign)

    total_fn = max(0, total_gts - total_tp)
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_gts)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "total_gts": int(total_gts),
        "ignored_detections": int(total_ignored),
    }


def compute_sequence_metrics_at_iou(
    predictions_by_frame: Dict[int, List[Prediction]],
    ground_truth_by_frame: Dict[int, List[BBox]],
    ignored_region: Optional[IgnoredRegion] = None,
    iou_threshold: float = 0.50,
    confidence_threshold: float = 0.25,
    ignore_overlap_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Computes detection metrics at a single IoU threshold:
    - AP is computed across all raw predictions without confidence truncation.
    - Precision, Recall, F1, TP, FP, FN are computed at confidence_threshold.
    """
    ap, _, _ = compute_pr_curve_and_ap(
        predictions_by_frame=predictions_by_frame,
        ground_truth_by_frame=ground_truth_by_frame,
        ignored_region=ignored_region,
        iou_threshold=iou_threshold,
        ignore_overlap_threshold=ignore_overlap_threshold,
    )
    op = compute_operating_point_metrics(
        predictions_by_frame=predictions_by_frame,
        ground_truth_by_frame=ground_truth_by_frame,
        ignored_region=ignored_region,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
        ignore_overlap_threshold=ignore_overlap_threshold,
    )
    return {
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence_threshold,
        "ap": ap,
        "precision": op["precision"],
        "recall": op["recall"],
        "f1": op["f1"],
        "tp": op["tp"],
        "fp": op["fp"],
        "fn": op["fn"],
        "total_gts": op["total_gts"],
        "ignored_detections": op["ignored_detections"],
    }


def compute_sequence_evaluation_full(
    predictions_by_frame: Dict[int, List[Prediction]],
    ground_truth_by_frame: Dict[int, List[BBox]],
    ignored_region: Optional[IgnoredRegion] = None,
    iou_thresholds: Optional[List[float]] = None,
    confidence_threshold: float = 0.25,
    ignore_overlap_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Performs complete sequence evaluation:
    - AP@0.50, AP@0.75, AP@0.50:0.95 computed unconstrained across all raw predictions.
    - Operating metrics (P, R, F1, TP, FP, FN, Ignored) computed at confidence_threshold.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]

    aps = []
    ap50 = 0.0
    ap75 = 0.0

    for iou in iou_thresholds:
        ap_val, _, _ = compute_pr_curve_and_ap(
            predictions_by_frame=predictions_by_frame,
            ground_truth_by_frame=ground_truth_by_frame,
            ignored_region=ignored_region,
            iou_threshold=iou,
            ignore_overlap_threshold=ignore_overlap_threshold,
        )
        aps.append(ap_val)
        if abs(iou - 0.50) < 1e-4:
            ap50 = ap_val
        if abs(iou - 0.75) < 1e-4:
            ap75 = ap_val

    ap50_95 = float(np.mean(aps)) if aps else 0.0

    # Operating metrics at primary threshold (0.50) and selected confidence
    op_metrics = compute_operating_point_metrics(
        predictions_by_frame=predictions_by_frame,
        ground_truth_by_frame=ground_truth_by_frame,
        ignored_region=ignored_region,
        iou_threshold=0.50,
        confidence_threshold=confidence_threshold,
        ignore_overlap_threshold=ignore_overlap_threshold,
    )

    return {
        "confidence_threshold": confidence_threshold,
        "ap50": float(ap50),
        "ap75": float(ap75),
        "ap50_95": float(ap50_95),
        "precision_at_50": op_metrics["precision"],
        "recall_at_50": op_metrics["recall"],
        "f1_at_50": op_metrics["f1"],
        "tp_at_50": op_metrics["tp"],
        "fp_at_50": op_metrics["fp"],
        "fn_at_50": op_metrics["fn"],
        "total_gts": op_metrics["total_gts"],
        "ignored_preds": op_metrics["ignored_detections"],
    }


# Backwards compatibility alias
compute_ap_range_summary = compute_sequence_evaluation_full
