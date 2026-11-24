"""
Full Evaluation Pipeline for YOLO11n Vehicle Detection Baseline.

Performs:
1. Validation sequence inference and threshold tuning (F1 maximization).
2. Test sequence inference and fixed-threshold evaluation.
3. Metric calculation: Precision, Recall, F1, AP@0.50, AP@0.75, AP@0.50:0.95.
   - AP is computed unconstrained across all ranked candidate detections.
   - Operating metrics (P, R, F1, TP, FP, FN, Ignored) are computed at operating threshold.
4. Generates performance and accuracy reports (JSON + Markdown).
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.src.dataset.parser import parse_detrac_xml, SequenceAnnotation, BBox
from experiments.src.dataset.curator import SequenceCurator
from experiments.src.detection.config import DetectionConfig, EvaluationConfig
from experiments.src.detection.detector import VehicleDetector
from experiments.src.detection.evaluator import DetectionEvaluator
from experiments.src.detection.metrics import Prediction


def get_or_predict_sequence(
    detector: VehicleDetector,
    seq_name: str,
    img_paths: List[Path],
    cache_dir: Path,
) -> Tuple[Dict[int, List[Prediction]], Dict[str, Any]]:
    """
    Runs raw model inference on a sequence (or loads cached raw predictions).
    Caches raw predictions down to detector threshold (0.05) to ensure
    rapid, deterministic, and reproducible re-evaluations.
    """
    cache_file = cache_dir / f"preds_raw_{seq_name}.json"
    stats_file = cache_dir / f"stats_raw_{seq_name}.json"

    if cache_file.exists() and stats_file.exists():
        print(f"[*] Loading cached raw predictions for {seq_name} from {cache_file}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)

        preds = {
            int(k): [
                Prediction(
                    bbox=BBox(
                        left=p["bbox"]["left"],
                        top=p["bbox"]["top"],
                        width=p["bbox"]["width"],
                        height=p["bbox"]["height"],
                    ),
                    confidence=p["confidence"],
                    class_name=p.get("class_name", "vehicle"),
                    raw_coco_class=p.get("raw_coco_class"),
                )
                for p in v
            ]
            for k, v in data.items()
        }
        return preds, stats

    print(f"[*] Running inference on {seq_name} ({len(img_paths)} frames)...")
    preds, stats = detector.predict_sequence(img_paths)
    stats["sequence"] = seq_name

    # Cache raw predictions
    cache_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        str(k): [
            {
                "bbox": {
                    "left": p.bbox.left,
                    "top": p.bbox.top,
                    "width": p.bbox.width,
                    "height": p.bbox.height,
                },
                "confidence": round(float(p.confidence), 4),
                "class_name": p.class_name,
                "raw_coco_class": p.raw_coco_class,
            }
            for p in v
        ]
        for k, v in preds.items()
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f)
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return preds, stats


def evaluate_baseline(
    max_frames_per_seq: Optional[int] = None,
    candidate_thresholds: Optional[List[float]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    print("=" * 80)
    print("🚦 YOLO11n Pretrained Vehicle Detection Baseline Evaluation")
    print("=" * 80)

    eval_out = output_dir or (PROJECT_ROOT / "experiments" / "outputs" / "evaluation")
    eval_out.mkdir(parents=True, exist_ok=True)
    cache_dir = PROJECT_ROOT / "data" / "cache" / "predictions"

    manifest_path = PROJECT_ROOT / "data" / "processed" / "curated_subset" / "curated_manifest.json"
    manifest = SequenceCurator.load_manifest(manifest_path)

    val_seq_names = manifest.splits["val"]
    test_seq_names = manifest.splits["test"]
    print(f"[*] Validation Sequences (for threshold selection): {val_seq_names}")
    print(f"[*] Test Sequences (strictly held out for final reporting): {test_seq_names}")

    # Initialize Detector and Evaluator
    det_config = DetectionConfig(confidence_threshold=0.05)  # raw candidates for PR curve
    eval_config = EvaluationConfig()
    detector = VehicleDetector(det_config)
    evaluator = DetectionEvaluator(eval_config)

    images_root = PROJECT_ROOT / "data" / "raw" / "images"
    annotations_root = PROJECT_ROOT / "data" / "raw" / "annotations"

    # Step 1: Run inference on Validation Sequences
    print("\n" + "=" * 80)
    print("Phase 1: Validation Set Inference & Threshold Selection")
    print("=" * 80)

    val_sequences: Dict[str, SequenceAnnotation] = {}
    val_predictions: Dict[str, Dict[int, List[Prediction]]] = {}
    val_stats = []

    for seq_name in val_seq_names:
        seq_ann = parse_detrac_xml(annotations_root / f"{seq_name}.xml")
        val_sequences[seq_name] = seq_ann
        
        img_paths = sorted(list((images_root / seq_name).glob("*.jpg")))
        if max_frames_per_seq:
            img_paths = img_paths[:max_frames_per_seq]

        preds, stats = get_or_predict_sequence(
            detector=detector,
            seq_name=seq_name,
            img_paths=img_paths,
            cache_dir=cache_dir,
        )
        val_predictions[seq_name] = preds
        val_stats.append(stats)
        print(f"    Latency: {stats['avg_latency_ms']:.1f}ms | FPS: {stats['fps']:.1f}")

    # Threshold sweep on validation set
    print("\n[*] Sweeping confidence thresholds on validation set...")
    best_conf, sweep_results = evaluator.sweep_confidence_thresholds(
        predictions_by_seq=val_predictions,
        sequences=val_sequences,
        candidate_thresholds=candidate_thresholds,
    )

    print("\nValidation Threshold Sweep Table:")
    sweep_df = pd.DataFrame(sweep_results)
    print(sweep_df[["confidence_threshold", "precision", "recall", "f1", "mean_ap50", "mean_ap50_95"]].to_string(index=False))
    print(f"\n[+] Selected Optimal Validation Confidence Threshold: {best_conf:.2f}")

    # Step 2: Run inference on Test Sequences
    print("\n" + "=" * 80)
    print("Phase 2: Test Set Evaluation (Fixed Threshold = {:.2f})".format(best_conf))
    print("=" * 80)

    test_sequences: Dict[str, SequenceAnnotation] = {}
    test_predictions: Dict[str, Dict[int, List[Prediction]]] = {}
    test_stats = []

    for seq_name in test_seq_names:
        seq_ann = parse_detrac_xml(annotations_root / f"{seq_name}.xml")
        test_sequences[seq_name] = seq_ann
        
        img_paths = sorted(list((images_root / seq_name).glob("*.jpg")))
        if max_frames_per_seq:
            img_paths = img_paths[:max_frames_per_seq]

        preds, stats = get_or_predict_sequence(
            detector=detector,
            seq_name=seq_name,
            img_paths=img_paths,
            cache_dir=cache_dir,
        )
        test_predictions[seq_name] = preds
        test_stats.append(stats)
        print(f"    Latency: {stats['avg_latency_ms']:.1f}ms | FPS: {stats['fps']:.1f}")

    # Evaluate test set with selected best_conf
    test_results = evaluator.evaluate_dataset(
        predictions_by_seq=test_predictions,
        sequences=test_sequences,
        confidence_threshold=best_conf,
    )

    print("\n" + "=" * 80)
    print("📊 Test Set Evaluation Results (IoU=0.50):")
    print("=" * 80)
    print(f"Fixed Confidence Threshold: {test_results['confidence_threshold']}")
    print(f"AP@0.50:                   {test_results['mean_ap50']:.4f}")
    print(f"AP@0.75:                   {test_results['mean_ap75']:.4f}")
    print(f"AP@0.50:0.95:              {test_results['mean_ap50_95']:.4f}")
    print(f"Precision@0.50:            {test_results['micro_precision_at_50']:.4f}")
    print(f"Recall@0.50:               {test_results['micro_recall_at_50']:.4f}")
    print(f"F1-Score@0.50:             {test_results['micro_f1_at_50']:.4f}")
    print(f"Ground Truth Boxes:        {test_results['total_ground_truth']}")
    print(f"True Positives:            {test_results['total_true_positives']}")
    print(f"False Positives:           {test_results['total_false_positives']}")
    print(f"False Negatives:           {test_results['total_false_negatives']}")
    print(f"Ignored Detections:        {test_results['total_ignored_detections']}")

    print("\nPer-Sequence Test Breakdown:")
    per_seq_df = pd.DataFrame(test_results["per_sequence"])
    print(per_seq_df[["sequence_name", "condition", "camera_state", "precision_at_50", "recall_at_50", "f1_at_50", "ap50", "ap50_95"]].to_string(index=False))

    # Overall Performance Benchmark
    all_perf = val_stats + test_stats
    avg_latency = float(np.mean([s["avg_latency_ms"] for s in all_perf]))
    avg_fps = float(np.mean([s["fps"] for s in all_perf]))
    print(f"\n[+] Overall Inference Throughput: {avg_fps:.1f} FPS (Avg Latency: {avg_latency:.1f}ms on {detector.device})")

    # Save Reports
    full_report = {
        "model": "YOLO11n (COCO pretrained, no fine-tuning)",
        "device": detector.device,
        "selected_confidence_threshold": best_conf,
        "validation_sweep": sweep_results,
        "test_evaluation": test_results,
        "performance": {
            "avg_latency_ms": avg_latency,
            "avg_fps": avg_fps,
            "device": detector.device,
            "sequence_benchmarks": all_perf,
        },
    }

    report_json_path = eval_out / "yolo11n_baseline_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[+] Saved complete evaluation report to: {report_json_path}")

    # Generate Markdown Report
    report_md_path = eval_out / "yolo11n_baseline_report.md"
    md_content = f"""# Pretrained YOLO11n Vehicle Detection Baseline Report

## Configuration
- **Model**: YOLO11n (Ultralytics pretrained COCO weights, zero fine-tuning)
- **Class Mapping**: Unified `vehicle` (COCO `car`, `motorcycle`, `bus`, `truck`) with class-agnostic NMS
- **Excluded Classes**: `bicycle` (1) and non-vehicle categories
- **Selected Confidence Threshold**: `{best_conf:.2f}` (selected via validation set F1 maximization)
- **Matching Protocol**: 1-to-1 greedy IoU matching with 0.50 ignore-region area overlap exclusion
- **Inference Device**: `{detector.device.upper()}` ($960 \\times 540$ px)

---

## 1. Test Set Detection Performance (Unseen Test Sequences)

| Metric | Score | Note |
| :--- | :--- | :--- |
| **AP@0.50** | **{test_results['mean_ap50'] * 100:.2f}%** | Mean Average Precision at IoU=0.50 (101-point COCO interpolation) |
| **AP@0.75** | **{test_results['mean_ap75'] * 100:.2f}%** | Strict localization threshold |
| **AP@0.50:0.95** | **{test_results['mean_ap50_95'] * 100:.2f}%** | Standard benchmark mean over 10 IoU steps |
| **Precision@0.50** | **{test_results['micro_precision_at_50'] * 100:.2f}%** | Fraction of predictions matching real vehicles |
| **Recall@0.50** | **{test_results['micro_recall_at_50'] * 100:.2f}%** | Fraction of ground truth vehicles detected |
| **F1-Score@0.50** | **{test_results['micro_f1_at_50'] * 100:.2f}%** | Harmonic balance between Precision & Recall |
| **True Positives** | **{test_results['total_true_positives']:,}** | Correctly localized vehicle detections |
| **False Positives** | **{test_results['total_false_positives']:,}** | Erroneous vehicle detections |
| **False Negatives** | **{test_results['total_false_negatives']:,}** | Ground truth vehicles missed |
| **Ignored Detections** | **{test_results['total_ignored_detections']:,}** | Detections falling in ignore regions (not penalized) |

---

## 2. Per-Sequence Test Breakdown

| Sequence | Condition | Camera | Frames | Precision@0.50 | Recall@0.50 | F1@0.50 | AP@0.50 | AP@0.50:0.95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for s in test_results["per_sequence"]:
        md_content += f"| **{s['sequence_name']}** | `{s['condition']}` | `{s['camera_state']}` | {s['total_frames']} | {s['precision_at_50']*100:.1f}% | {s['recall_at_50']*100:.1f}% | {s['f1_at_50']*100:.1f}% | {s['ap50']*100:.1f}% | {s['ap50_95']*100:.1f}% |\n"

    md_content += f"""
---

## 3. Inference Throughput & Latency

- **Average Inference Latency**: **{avg_latency:.2f} ms** per frame
- **Inference Throughput**: **{avg_fps:.1f} FPS** (pure model inference on {detector.device.upper()})
- **Model Load Time**: **{detector.model_load_time_ms:.1f} ms**
- **Image Resolution**: Native $960 \\times 540$ px
"""
    report_md_path.write_text(md_content, encoding="utf-8")
    print(f"[+] Saved markdown evaluation report to: {report_md_path}")

    return full_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO11n detection baseline")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per sequence for evaluation")
    args = parser.parse_args()

    evaluate_baseline(max_frames_per_seq=args.max_frames)


if __name__ == "__main__":
    main()
