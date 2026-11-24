"""
Evaluation engine for vehicle detection baseline against UA-DETRAC ground truth.
"""

from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from experiments.src.dataset.parser import BBox, SequenceAnnotation
from .config import EvaluationConfig
from .metrics import (
    Prediction,
    compute_sequence_evaluation_full,
    compute_sequence_metrics_at_iou,
)


class DetectionEvaluator:
    """Evaluates predicted vehicle detections against ground truth annotations."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()

    @staticmethod
    def extract_gt_boxes_by_frame(seq_ann: SequenceAnnotation) -> Dict[int, List[BBox]]:
        """Extracts ground truth vehicle bounding boxes keyed by frame number."""
        gt_by_frame: Dict[int, List[BBox]] = {}
        for frame_num, frame_ann in seq_ann.frames.items():
            gt_by_frame[frame_num] = [t.bbox for t in frame_ann.targets]
        return gt_by_frame

    def evaluate_sequence(
        self,
        predictions_by_frame: Dict[int, List[Prediction]],
        seq_ann: SequenceAnnotation,
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates predictions on a single sequence:
        - Computes AP unconstrained across all raw predictions.
        - Computes operating point metrics (P, R, F1, TP, FP, FN, Ignored) at confidence_threshold.
        """
        conf_thresh = (
            confidence_threshold
            if confidence_threshold is not None
            else self.config.confidence_threshold
        )
        gt_by_frame = self.extract_gt_boxes_by_frame(seq_ann)

        summary = compute_sequence_evaluation_full(
            predictions_by_frame=predictions_by_frame,
            ground_truth_by_frame=gt_by_frame,
            ignored_region=seq_ann.ignored_region,
            iou_thresholds=self.config.iou_thresholds,
            confidence_threshold=conf_thresh,
            ignore_overlap_threshold=self.config.ignore_overlap_threshold,
        )
        summary["sequence_name"] = seq_ann.name
        summary["condition"] = seq_ann.scene_condition
        summary["camera_state"] = seq_ann.camera_state
        summary["total_frames"] = seq_ann.total_frames
        return summary

    def sweep_confidence_thresholds(
        self,
        predictions_by_seq: Dict[str, Dict[int, List[Prediction]]],
        sequences: Dict[str, SequenceAnnotation],
        candidate_thresholds: Optional[List[float]] = None,
    ) -> Tuple[float, List[Dict[str, float]]]:
        """
        Sweeps confidence thresholds across validation sequences to identify
        the threshold that maximizes validation F1-score.
        Note: AP values remain constant across confidence thresholds as AP is
        evaluated unconstrained across the full Precision-Recall curve.
        """
        if candidate_thresholds is None:
            candidate_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]

        sweep_results = []
        best_conf = 0.25
        best_f1 = -1.0

        for conf in candidate_thresholds:
            total_tp = 0
            total_fp = 0
            total_fn = 0
            aps_50 = []
            aps_50_95 = []

            for seq_name, preds_by_frame in predictions_by_seq.items():
                seq_ann = sequences[seq_name]
                res = self.evaluate_sequence(preds_by_frame, seq_ann, confidence_threshold=conf)
                total_tp += res["tp_at_50"]
                total_fp += res["fp_at_50"]
                total_fn += res["fn_at_50"]
                aps_50.append(res["ap50"])
                aps_50_95.append(res["ap50_95"])

            micro_precision = total_tp / max(1, total_tp + total_fp)
            micro_recall = total_tp / max(1, total_tp + total_fn)
            micro_f1 = (
                2 * micro_precision * micro_recall / max(1e-8, micro_precision + micro_recall)
            )
            mean_ap50 = float(np.mean(aps_50)) if aps_50 else 0.0
            mean_ap50_95 = float(np.mean(aps_50_95)) if aps_50_95 else 0.0

            row = {
                "confidence_threshold": conf,
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
                "mean_ap50": mean_ap50,
                "mean_ap50_95": mean_ap50_95,
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
            }
            sweep_results.append(row)

            if micro_f1 > best_f1:
                best_f1 = micro_f1
                best_conf = conf

        return best_conf, sweep_results

    def evaluate_dataset(
        self,
        predictions_by_seq: Dict[str, Dict[int, List[Prediction]]],
        sequences: Dict[str, SequenceAnnotation],
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates all sequences in a split and aggregates metrics.
        """
        conf_thresh = (
            confidence_threshold
            if confidence_threshold is not None
            else self.config.confidence_threshold
        )

        per_seq_results = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gts = 0
        total_ignored_preds = 0
        all_ap50 = []
        all_ap75 = []
        all_ap50_95 = []

        for seq_name, preds_by_frame in predictions_by_seq.items():
            seq_ann = sequences[seq_name]
            res = self.evaluate_sequence(preds_by_frame, seq_ann, confidence_threshold=conf_thresh)
            per_seq_results.append(res)

            total_tp += res["tp_at_50"]
            total_fp += res["fp_at_50"]
            total_fn += res["fn_at_50"]
            total_gts += res["total_gts"]
            total_ignored_preds += res["ignored_preds"]

            all_ap50.append(res["ap50"])
            all_ap75.append(res["ap75"])
            all_ap50_95.append(res["ap50_95"])

        micro_precision = total_tp / max(1, total_tp + total_fp)
        micro_recall = total_tp / max(1, total_tp + total_fn)
        micro_f1 = 2 * micro_precision * micro_recall / max(1e-8, micro_precision + micro_recall)

        aggregate = {
            "confidence_threshold": conf_thresh,
            "primary_iou_threshold": self.config.primary_iou_threshold,
            "mean_ap50": float(np.mean(all_ap50)) if all_ap50 else 0.0,
            "mean_ap75": float(np.mean(all_ap75)) if all_ap75 else 0.0,
            "mean_ap50_95": float(np.mean(all_ap50_95)) if all_ap50_95 else 0.0,
            "micro_precision_at_50": float(micro_precision),
            "micro_recall_at_50": float(micro_recall),
            "micro_f1_at_50": float(micro_f1),
            "total_ground_truth": total_gts,
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "total_ignored_detections": total_ignored_preds,
            "per_sequence": per_seq_results,
        }
        return aggregate
