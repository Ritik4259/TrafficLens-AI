"""
Full Evaluation & Parameter Tuning Pipeline for ByteTrack Vehicle Tracking.

Performs:
1. Validation sequence parameter sweep (tuning track_high_thresh, match_thresh, track_buffer).
2. Configuration freezing.
3. Held-out test sequence evaluation using official TrackEval HOTA and motmetrics.
4. Export of MOTChallenge txt and JSON tracking results.
5. Generation of comprehensive tracking benchmark report (JSON + Markdown).
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
from experiments.src.detection.config import DetectionConfig
from experiments.src.detection.detector import VehicleDetector
from experiments.src.detection.metrics import Prediction
from experiments.src.tracking.config import TrackingConfig, TrackingEvaluationConfig
from experiments.src.tracking.tracker import Track, VehicleTracker
from experiments.src.tracking.exporter import MOTExporter
from experiments.src.tracking.evaluator import TrackingEvaluator


def load_or_predict_raw(
    seq_name: str,
    img_paths: List[Path],
    cache_dir: Path,
    detector: Optional[VehicleDetector] = None,
) -> Tuple[Dict[int, List[Prediction]], Dict[str, Any]]:
    """Loads raw detections (down to 0.05 floor) from cache or runs detector."""
    cache_file = cache_dir / f"preds_raw_{seq_name}.json"
    stats_file = cache_dir / f"stats_raw_{seq_name}.json"

    if cache_file.exists() and stats_file.exists():
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

    if detector is None:
        detector = VehicleDetector(DetectionConfig(confidence_threshold=0.05))

    preds, stats = detector.predict_sequence(img_paths)
    stats["sequence"] = seq_name
    return preds, stats


def evaluate_tracking_pipeline(
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    print("=" * 80)
    print("🚦 ByteTrack Multi-Object Vehicle Tracking Evaluation Pipeline")
    print("=" * 80)

    out_root = output_dir or (PROJECT_ROOT / "experiments" / "outputs" / "tracking")
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = PROJECT_ROOT / "data" / "cache" / "predictions"

    manifest_path = PROJECT_ROOT / "data" / "processed" / "curated_subset" / "curated_manifest.json"
    manifest = SequenceCurator.load_manifest(manifest_path)

    val_seq_names = manifest.splits["val"]
    test_seq_names = manifest.splits["test"]
    print(f"[*] Validation Sequences (for parameter tuning): {val_seq_names}")
    print(f"[*] Test Sequences (strictly held out for final reporting): {test_seq_names}")

    images_root = PROJECT_ROOT / "data" / "raw" / "images"
    annotations_root = PROJECT_ROOT / "data" / "raw" / "annotations"

    # Step 1: Load raw detections for validation sequences
    print("\n" + "=" * 80)
    print("Phase 1: Validation Set Parameter Sweep")
    print("=" * 80)

    val_sequences: Dict[str, SequenceAnnotation] = {}
    val_predictions: Dict[str, Dict[int, List[Prediction]]] = {}
    val_det_stats: List[Dict[str, Any]] = []

    for seq_name in val_seq_names:
        seq_ann = parse_detrac_xml(annotations_root / f"{seq_name}.xml")
        val_sequences[seq_name] = seq_ann
        img_paths = sorted(list((images_root / seq_name).glob("*.jpg")))
        preds, stats = load_or_predict_raw(seq_name, img_paths, cache_dir)
        val_predictions[seq_name] = preds
        val_det_stats.append(stats)
        print(f"[*] Loaded raw detections for {seq_name}: {len(preds)} frames, {stats['fps']:.1f} det FPS")

    # Parameter grid sweep on validation set
    evaluator = TrackingEvaluator()
    print("\n[*] Sweeping tracking parameters across validation sequences...")
    high_thresholds = [0.40, 0.50, 0.60]
    match_thresholds = [0.70, 0.80]
    track_buffers = [20, 30, 45]

    best_config, sweep_results = evaluator.sweep_parameters(
        predictions_by_seq=val_predictions,
        sequences=val_sequences,
        high_thresholds=high_thresholds,
        match_thresholds=match_thresholds,
        track_buffers=track_buffers,
    )

    sweep_df = pd.DataFrame(sweep_results)
    print("\nValidation Parameter Sweep Table:")
    print(sweep_df[["track_high_thresh", "match_thresh", "track_buffer", "mean_hota", "mean_mota", "mean_idf1", "total_idsw"]].to_string(index=False))

    print(f"\n[+] Selected Optimal Tracking Configuration:")
    print(f"    track_high_thresh = {best_config.track_high_thresh}")
    print(f"    track_low_thresh  = {best_config.track_low_thresh}")
    print(f"    new_track_thresh  = {best_config.new_track_thresh}")
    print(f"    match_thresh      = {best_config.match_thresh}")
    print(f"    track_buffer      = {best_config.track_buffer}")

    # Step 2: Evaluate Held-Out Test Sequences with Frozen Configuration
    print("\n" + "=" * 80)
    print("Phase 2: Held-Out Test Set Tracking Evaluation (Frozen Configuration)")
    print("=" * 80)

    test_sequences: Dict[str, SequenceAnnotation] = {}
    test_predictions: Dict[str, Dict[int, List[Prediction]]] = {}
    test_det_stats: List[Dict[str, Any]] = []

    for seq_name in test_seq_names:
        seq_ann = parse_detrac_xml(annotations_root / f"{seq_name}.xml")
        test_sequences[seq_name] = seq_ann
        img_paths = sorted(list((images_root / seq_name).glob("*.jpg")))
        preds, stats = load_or_predict_raw(seq_name, img_paths, cache_dir)
        test_predictions[seq_name] = preds
        test_det_stats.append(stats)
        print(f"[*] Loaded raw detections for {seq_name}: {len(preds)} frames, {stats['fps']:.1f} det FPS")

    # Run frozen tracker on test sequences
    frozen_tracker = VehicleTracker(best_config)
    test_tracks_by_seq: Dict[str, Dict[int, List[Track]]] = {}
    test_tracking_stats: List[Dict[str, Any]] = []

    for seq_name, preds_by_frame in test_predictions.items():
        print(f"[*] Tracking sequence {seq_name}...")
        tracks, stats = frozen_tracker.track_sequence(preds_by_frame)
        test_tracks_by_seq[seq_name] = tracks
        stats["sequence"] = seq_name
        test_tracking_stats.append(stats)
        print(f"    Tracking Time: {stats['total_time_s']:.2f}s | Latency: {stats['avg_latency_ms']:.2f}ms | Track FPS: {stats['fps']:.1f} | Unique IDs: {stats['unique_track_ids']}")

        # Export MOT format text file and JSON
        seq_out_dir = out_root / seq_name
        mot_txt_path = seq_out_dir / f"{seq_name}_tracking.txt"
        json_path = seq_out_dir / f"{seq_name}_tracking.json"
        gt_txt_path = seq_out_dir / f"{seq_name}_gt.txt"

        MOTExporter.export_tracks_to_mot_txt(tracks, mot_txt_path)
        MOTExporter.export_tracks_to_json(tracks, json_path, metadata={"sequence": seq_name, "config": best_config.__dict__})
        MOTExporter.export_gt_to_mot_txt(test_sequences[seq_name], gt_txt_path)
        print(f"    Exported MOT tracks to: {mot_txt_path}")

    # Evaluate test tracking results
    test_eval_results = evaluator.evaluate_dataset(test_tracks_by_seq, test_sequences)

    print("\n" + "=" * 80)
    print("📊 Held-Out Test Set Tracking Results:")
    print("=" * 80)
    print(f"HOTA:                   {test_eval_results['mean_hota'] * 100:.2f}%")
    print(f"DetA:                   {test_eval_results['mean_deta'] * 100:.2f}%")
    print(f"AssA:                   {test_eval_results['mean_assa'] * 100:.2f}%")
    print(f"MOTA:                   {test_eval_results['mean_mota'] * 100:.2f}%")
    print(f"IDF1:                   {test_eval_results['mean_idf1'] * 100:.2f}%")
    print(f"Total ID Switches:      {test_eval_results['total_idsw']}")
    print(f"Total Fragmentations:   {test_eval_results['total_frag']}")
    print(f"Mostly Tracked (MT):    {test_eval_results['total_mostly_tracked']}")
    print(f"Mostly Lost (ML):       {test_eval_results['total_mostly_lost']}")
    print(f"False Positives:        {test_eval_results['total_fp']}")
    print(f"False Negatives:        {test_eval_results['total_fn']}")
    print(f"Ignored Detections:     {test_eval_results['total_ignored_detections']}")

    print("\nPer-Sequence Test Breakdown:")
    per_seq_df = pd.DataFrame(test_eval_results["per_sequence"])
    print(per_seq_df[["sequence_name", "condition", "camera_state", "hota", "deta", "assa", "mota", "idf1", "idsw", "mostly_tracked", "mostly_lost"]].to_string(index=False))

    # Pipeline Performance Benchmark
    avg_det_latency = float(np.mean([s["avg_latency_ms"] for s in test_det_stats]))
    avg_trk_latency = float(np.mean([s["avg_latency_ms"] for s in test_tracking_stats]))
    total_pipeline_latency = avg_det_latency + avg_trk_latency
    end_to_end_fps = 1000.0 / total_pipeline_latency if total_pipeline_latency > 0 else 0.0

    print("\n" + "=" * 80)
    print("⚡ Pipeline Latency & Throughput Benchmark:")
    print("=" * 80)
    print(f"Detection Inference Latency: {avg_det_latency:.2f} ms")
    print(f"ByteTrack Update Latency:    {avg_trk_latency:.2f} ms")
    print(f"Total Pipeline Latency:      {total_pipeline_latency:.2f} ms")
    print(f"End-to-End Throughput:       {end_to_end_fps:.1f} FPS (CPU)")

    # Save structured reports
    full_report = {
        "tracker": "ByteTrack (two-stage association with Kalman filtering)",
        "detector": "YOLO11n (COCO pretrained, zero fine-tuning)",
        "selected_tracking_config": best_config.__dict__,
        "validation_parameter_sweep": sweep_results,
        "test_tracking_evaluation": test_eval_results,
        "performance": {
            "avg_detection_latency_ms": avg_det_latency,
            "avg_tracking_latency_ms": avg_trk_latency,
            "total_pipeline_latency_ms": total_pipeline_latency,
            "end_to_end_fps": end_to_end_fps,
            "sequence_benchmarks": test_tracking_stats,
        },
    }

    report_json_path = out_root / "tracking_baseline_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[+] Saved complete tracking report to: {report_json_path}")

    # Generate Markdown Report
    report_md_path = out_root / "tracking_baseline_report.md"
    md_content = f"""# ByteTrack Multi-Object Vehicle Tracking Baseline Report

## Configuration
- **Tracker**: ByteTrack (Ultralytics maintained engine, 2-stage association)
- **Detector**: YOLO11n (COCO pretrained weights, zero fine-tuning)
- **Track High Threshold**: `{best_config.track_high_thresh:.2f}` (selected via validation HOTA/MOTA sweep)
- **Track Low Threshold**: `{best_config.track_low_thresh:.2f}` (recovers occluded/low-confidence candidate vehicles)
- **New Track Threshold**: `{best_config.new_track_thresh:.2f}`
- **Match Threshold**: `{best_config.match_thresh:.2f}`
- **Track Buffer**: `{best_config.track_buffer}` frames
- **Frame Rate**: `{best_config.frame_rate}` FPS
- **Evaluation Protocols**: Official **TrackEval** (HOTA, DetA, AssA) and **motmetrics** (MOTA, IDF1, CLEAR)

---

## 1. Held-Out Test Set Tracking Performance

| Metric | Score | Description |
| :--- | :--- | :--- |
| **HOTA** | **{test_eval_results['mean_hota'] * 100:.2f}%** | Higher Order Tracking Accuracy (unified detection & association) |
| **DetA** | **{test_eval_results['mean_deta'] * 100:.2f}%** | Detection Accuracy (mean over 19 localization thresholds) |
| **AssA** | **{test_eval_results['mean_assa'] * 100:.2f}%** | Association Accuracy (trajectory consistency over time) |
| **MOTA** | **{test_eval_results['mean_mota'] * 100:.2f}%** | Multiple Object Tracking Accuracy (CLEAR metric) |
| **IDF1** | **{test_eval_results['mean_idf1'] * 100:.2f}%** | ID F1-Score (global identity preservation) |
| **ID Switches (IDSW)** | **{test_eval_results['total_idsw']}** | Total vehicle identity swaps across all test sequences |
| **Fragmentations (Frag)** | **{test_eval_results['total_frag']}** | Track trajectory interruptions |
| **Mostly Tracked (MT)** | **{test_eval_results['total_mostly_tracked']}** | Vehicles tracked for $\ge 80\%$ of their trajectory |
| **Mostly Lost (ML)** | **{test_eval_results['total_mostly_lost']}** | Vehicles tracked for $\le 20\%$ of their trajectory |
| **False Positives (FP)** | **{test_eval_results['total_fp']:,}** | Erroneous tracker bounding boxes |
| **False Negatives (FN)** | **{test_eval_results['total_fn']:,}** | Missed ground truth vehicle instances |
| **Ignored Detections** | **{test_eval_results['total_ignored_detections']:,}** | Tracks in unannotated background parking lots (unpenalized) |

---

## 2. Per-Sequence Test Breakdown

| Sequence | Condition | Camera | Frames | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | MT | ML |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for s in test_eval_results["per_sequence"]:
        md_content += f"| **{s['sequence_name']}** | `{s['condition']}` | `{s['camera_state']}` | {s['total_frames']} | {s['hota']*100:.1f}% | {s['deta']*100:.1f}% | {s['assa']*100:.1f}% | {s['mota']*100:.1f}% | {s['idf1']*100:.1f}% | {s['idsw']} | {s['mostly_tracked']} | {s['mostly_lost']} |\n"

    md_content += f"""
---

## 3. End-to-End Latency & Throughput Benchmark

- **Detection Inference Latency**: **{avg_det_latency:.2f} ms** per frame
- **ByteTrack Update Latency**: **{avg_trk_latency:.2f} ms** per frame
- **Total Pipeline Latency**: **{total_pipeline_latency:.2f} ms** per frame
- **End-to-End Throughput**: **{end_to_end_fps:.1f} FPS** (CPU batched execution)
- **Tracking Overhead**: Only **{avg_trk_latency / total_pipeline_latency * 100:.1f}%** of total frame processing time
"""
    report_md_path.write_text(md_content, encoding="utf-8")
    print(f"[+] Saved markdown tracking report to: {report_md_path}")

    return full_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate ByteTrack tracking pipeline")
    args = parser.parse_args()
    evaluate_tracking_pipeline()


if __name__ == "__main__":
    main()
