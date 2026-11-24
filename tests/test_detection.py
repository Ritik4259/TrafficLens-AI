"""
Unit tests for YOLO detection pipeline, class mapping, NMS, and evaluation metrics.
"""

import pytest
import numpy as np
from pathlib import Path

from experiments.src.dataset.parser import (
    BBox,
    IgnoredRegion,
    SequenceAnnotation,
    FrameAnnotation,
    Target,
    VehicleAttribute,
)
from experiments.src.detection.config import (
    DetectionConfig,
    EvaluationConfig,
    COCO_VEHICLE_CLASS_IDS,
    COCO_EXCLUDED_VEHICLE_IDS,
)
from experiments.src.detection.metrics import (
    Prediction,
    compute_iou_matrix,
    match_frame_detections,
    compute_ap_101_point,
    compute_pr_curve_and_ap,
    compute_operating_point_metrics,
    compute_sequence_metrics_at_iou,
    compute_sequence_evaluation_full,
    compute_ap_range_summary,
)
from experiments.src.detection.detector import apply_class_agnostic_nms
from experiments.src.detection.evaluator import DetectionEvaluator


class TestClassMappingAndFiltering:
    def test_coco_vehicle_mapping_includes_target_classes(self):
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        assert 2 in COCO_VEHICLE_CLASS_IDS
        assert 3 in COCO_VEHICLE_CLASS_IDS
        assert 5 in COCO_VEHICLE_CLASS_IDS
        assert 7 in COCO_VEHICLE_CLASS_IDS

    def test_bicycle_is_explicitly_excluded(self):
        # 1: bicycle must be in excluded list and not in target classes
        assert 1 in COCO_EXCLUDED_VEHICLE_IDS
        assert 1 not in COCO_VEHICLE_CLASS_IDS


class TestClassAgnosticNMS:
    def test_overlapping_duplicate_predictions_collapsed(self):
        """
        If YOLO predicts both 'car' and 'truck' on the same physical vehicle
        with high IoU overlap (>0.45), class-agnostic NMS collapses them to the
        single highest-confidence prediction.
        """
        box1 = BBox(left=100.0, top=100.0, width=80.0, height=60.0)
        box2 = BBox(left=102.0, top=101.0, width=78.0, height=59.0)  # >0.90 IoU with box1
        box3 = BBox(left=400.0, top=200.0, width=70.0, height=50.0)  # separate vehicle

        p1 = Prediction(bbox=box1, confidence=0.85, class_name="vehicle", raw_coco_class="car")
        p2 = Prediction(bbox=box2, confidence=0.72, class_name="vehicle", raw_coco_class="truck")
        p3 = Prediction(bbox=box3, confidence=0.90, class_name="vehicle", raw_coco_class="bus")

        nms_results = apply_class_agnostic_nms([p1, p2, p3], iou_threshold=0.45)
        # p2 should be suppressed by p1
        assert len(nms_results) == 2
        confidences = {p.confidence for p in nms_results}
        assert confidences == {0.90, 0.85}

    def test_non_overlapping_predictions_preserved(self):
        p1 = Prediction(bbox=BBox(0, 0, 50, 50), confidence=0.8)
        p2 = Prediction(bbox=BBox(100, 100, 50, 50), confidence=0.7)
        nms_results = apply_class_agnostic_nms([p1, p2], iou_threshold=0.45)
        assert len(nms_results) == 2


class TestDetectionMatchingAndEvaluation:
    def test_one_to_one_matching_prevents_double_counting(self):
        """
        A single ground truth box can match at most ONE prediction.
        The second overlapping prediction must be scored as a False Positive (FP).
        """
        gt_box = BBox(left=100.0, top=100.0, width=100.0, height=100.0)
        
        # High confidence prediction matching GT
        p1 = Prediction(bbox=BBox(100.0, 100.0, 100.0, 100.0), confidence=0.90)
        # Lower confidence prediction on the exact same GT
        p2 = Prediction(bbox=BBox(105.0, 105.0, 95.0, 95.0), confidence=0.60)

        is_tp, is_fp, is_ign = match_frame_detections(
            predictions=[p1, p2],
            ground_truth=[gt_box],
            iou_threshold=0.50,
        )

        assert is_tp == [True, False]
        assert is_fp == [False, True]
        assert is_ign == [False, False]

    def test_ignored_region_excludes_detections_from_false_positives(self):
        """
        A detection inside an ignore region must NOT be penalized as a False Positive.
        """
        ign_box = BBox(left=0.0, top=0.0, width=200.0, height=200.0)
        ignored_region = IgnoredRegion(boxes=[ign_box])

        # Prediction falling completely inside ignore region
        p_ign = Prediction(bbox=BBox(20.0, 20.0, 50.0, 50.0), confidence=0.80)
        # Normal prediction on roadway with no GT
        p_road = Prediction(bbox=BBox(400.0, 300.0, 60.0, 50.0), confidence=0.75)

        is_tp, is_fp, is_ign = match_frame_detections(
            predictions=[p_ign, p_road],
            ground_truth=[],  # No ground truth
            ignored_region=ignored_region,
            iou_threshold=0.50,
            ignore_overlap_threshold=0.50,
        )

        # p_ign is ignored (not FP)
        assert is_ign[0] is True
        assert is_fp[0] is False
        assert is_tp[0] is False

        # p_road is an unignored FP
        assert is_ign[1] is False
        assert is_fp[1] is True
        assert is_tp[1] is False

    def test_gt_matching_takes_precedence_over_ignore_region(self):
        """
        If a valid ground-truth vehicle overlaps an ignore region, a prediction
        matching that ground-truth box MUST be marked as True Positive, NOT ignored.
        """
        ign_box = BBox(left=0.0, top=0.0, width=200.0, height=200.0)
        ignored_region = IgnoredRegion(boxes=[ign_box])

        # Ground truth vehicle situated inside/partially overlapping the ignore region
        gt_box = BBox(left=50.0, top=50.0, width=60.0, height=60.0)
        # Accurate prediction matching this GT box
        pred = Prediction(bbox=BBox(52.0, 51.0, 58.0, 59.0), confidence=0.88)

        is_tp, is_fp, is_ign = match_frame_detections(
            predictions=[pred],
            ground_truth=[gt_box],
            ignored_region=ignored_region,
            iou_threshold=0.50,
            ignore_overlap_threshold=0.50,
        )

        assert is_tp == [True]
        assert is_fp == [False]
        assert is_ign == [False]

    def test_partial_overlap_below_threshold_is_false_positive(self):
        """
        A candidate detection overlapping an ignore region by less than the threshold (0.50)
        with no ground truth must be classified as a False Positive (FP).
        """
        ign_box = BBox(left=0.0, top=0.0, width=100.0, height=100.0)
        ignored_region = IgnoredRegion(boxes=[ign_box])

        # Box of size 100x100 starting at (80, 0): intersection is 20x100 = 2000 px (20% of box area)
        pred = Prediction(bbox=BBox(left=80.0, top=0.0, width=100.0, height=100.0), confidence=0.75)

        is_tp, is_fp, is_ign = match_frame_detections(
            predictions=[pred],
            ground_truth=[],
            ignored_region=ignored_region,
            iou_threshold=0.50,
            ignore_overlap_threshold=0.50,
        )

        assert is_tp == [False]
        assert is_fp == [True]
        assert is_ign == [False]

    def test_empty_predictions_produces_zero_precision_recall(self):
        gt_box = BBox(100.0, 100.0, 50.0, 50.0)
        summary = compute_sequence_metrics_at_iou(
            predictions_by_frame={1: []},
            ground_truth_by_frame={1: [gt_box]},
            iou_threshold=0.50,
        )
        assert summary["tp"] == 0
        assert summary["fp"] == 0
        assert summary["fn"] == 1
        assert summary["precision"] == 0.0
        assert summary["recall"] == 0.0
        assert summary["f1"] == 0.0
        assert summary["ap"] == 0.0

    def test_empty_ground_truth_produces_zero_tps(self):
        pred = Prediction(bbox=BBox(100, 100, 50, 50), confidence=0.90)
        summary = compute_sequence_metrics_at_iou(
            predictions_by_frame={1: [pred]},
            ground_truth_by_frame={1: []},
            iou_threshold=0.50,
        )
        assert summary["tp"] == 0
        assert summary["fp"] == 1
        assert summary["fn"] == 0
        assert summary["recall"] == 0.0
        assert summary["precision"] == 0.0

    def test_perfect_predictions_yields_ap_one(self):
        box1 = BBox(50, 50, 40, 40)
        box2 = BBox(200, 200, 60, 60)
        preds = [
            Prediction(bbox=box1, confidence=0.95),
            Prediction(bbox=box2, confidence=0.88),
        ]
        gts = [box1, box2]
        summary = compute_sequence_metrics_at_iou(
            predictions_by_frame={1: preds},
            ground_truth_by_frame={1: gts},
            iou_threshold=0.50,
        )
        assert summary["tp"] == 2
        assert summary["fp"] == 0
        assert summary["fn"] == 0
        assert summary["precision"] == 1.0
        assert summary["recall"] == 1.0
        assert summary["f1"] == 1.0
        assert summary["ap"] == 1.0

    def test_ap_101_point_monotonic_interpolation(self):
        recalls = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        precisions = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        ap = compute_ap_101_point(recalls, precisions)
        assert 0.6 <= ap <= 1.0

    def test_ap_unconstrained_by_operating_confidence(self):
        """
        Verify that AP is evaluated across the full Precision-Recall curve
        without being truncated by the operating confidence threshold.
        """
        gt1 = BBox(50.0, 50.0, 40.0, 40.0)
        gt2 = BBox(200.0, 200.0, 60.0, 60.0)
        
        # High confidence prediction matching gt1
        p1 = Prediction(bbox=gt1, confidence=0.90)
        # Low confidence prediction matching gt2 (below typical operating cutoff 0.25)
        p2 = Prediction(bbox=gt2, confidence=0.15)

        summary = compute_sequence_evaluation_full(
            predictions_by_frame={1: [p1, p2]},
            ground_truth_by_frame={1: [gt1, gt2]},
            iou_thresholds=[0.50],
            confidence_threshold=0.25,
        )

        # Operating point metrics at conf=0.25 only see p1
        assert summary["tp_at_50"] == 1
        assert summary["fn_at_50"] == 1
        assert summary["recall_at_50"] == 0.50

        # BUT AP@0.50 spans the full PR curve including p2 (recall=1.0, precision=1.0)
        assert summary["ap50"] == 1.0

    def test_confidence_sweep_constant_ap(self):
        """
        Verify that sweeping confidence thresholds keeps AP constant while
        altering operating precision, recall, and F1.
        """
        gt1 = BBox(50.0, 50.0, 40.0, 40.0)
        p1 = Prediction(bbox=gt1, confidence=0.80)
        p_fp = Prediction(bbox=BBox(300.0, 300.0, 40.0, 40.0), confidence=0.30)

        evaluator = DetectionEvaluator()
        seq_ann = SequenceAnnotation(
            name="TEST_SEQ",
            camera_state="stable",
            scene_condition="daylight",
            fog_status="no_fog",
            ignored_region=IgnoredRegion(boxes=[]),
            frames={
                1: FrameAnnotation(
                    frame_num=1,
                    density=1,
                    targets=[Target(target_id=1, bbox=gt1, attribute=VehicleAttribute(vehicle_type="car"))],
                )
            },
        )

        best_conf, sweep_results = evaluator.sweep_confidence_thresholds(
            predictions_by_seq={"TEST_SEQ": {1: [p1, p_fp]}},
            sequences={"TEST_SEQ": seq_ann},
            candidate_thresholds=[0.20, 0.40],
        )

        # AP remains identical across both thresholds
        assert sweep_results[0]["mean_ap50"] == sweep_results[1]["mean_ap50"]
        # Precision differs (at conf 0.20, p_fp is included; at conf 0.40, p_fp is excluded)
        assert sweep_results[0]["precision"] < sweep_results[1]["precision"]
