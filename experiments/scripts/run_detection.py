"""
Inference script for YOLO11n vehicle detection baseline.
Supports single-image and sequence-level inference, exporting annotated frames,
video clips, and structured prediction JSONs.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.src.dataset.parser import parse_detrac_xml, SequenceAnnotation
from experiments.src.detection.config import DetectionConfig
from experiments.src.detection.detector import VehicleDetector
from experiments.src.detection.visualizer import DetectionVisualizer
from experiments.src.detection.metrics import Prediction


def run_sequence_detection(
    sequence_id: str,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_frames: Optional[int] = None,
    save_images: bool = True,
    save_video: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    images_dir = PROJECT_ROOT / "data" / "raw" / "images" / sequence_id
    annotations_path = PROJECT_ROOT / "data" / "raw" / "annotations" / f"{sequence_id}.xml"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Sequence images directory not found: {images_dir}")

    # Load annotations for GT and ignore regions if available
    seq_ann = parse_detrac_xml(annotations_path) if annotations_path.exists() else None

    # Gather frame paths in natural sorted order
    image_paths = sorted(list(images_dir.glob("*.jpg")))
    if max_frames:
        image_paths = image_paths[:max_frames]

    print(f"[*] Initializing YOLO11n detector for sequence {sequence_id} ({len(image_paths)} frames)...")
    config = DetectionConfig(
        confidence_threshold=conf_thresh,
        nms_iou_threshold=iou_thresh,
    )
    detector = VehicleDetector(config)

    # Output directory
    seq_out = output_dir or (PROJECT_ROOT / "experiments" / "outputs" / "detection" / sequence_id)
    seq_out.mkdir(parents=True, exist_ok=True)

    print(f"[*] Running inference on {len(image_paths)} frames...")
    preds_by_frame, stats = detector.predict_sequence(image_paths, conf=conf_thresh)

    print(f"[+] Inference Complete:")
    print(f"    - Total Inference Time: {stats['total_inference_s']:.2f}s")
    print(f"    - Avg Latency per Frame: {stats['avg_latency_ms']:.2f}ms")
    print(f"    - Pure Model FPS:        {stats['fps']:.1f} FPS")
    print(f"    - Device:                {stats['device']}")

    # Export Predictions to JSON
    json_export = {}
    for f_idx, preds in preds_by_frame.items():
        json_export[f_idx] = [
            {
                "bbox": {"left": p.bbox.left, "top": p.bbox.top, "width": p.bbox.width, "height": p.bbox.height},
                "confidence": round(p.confidence, 4),
                "class_name": p.class_name,
                "raw_coco_class": p.raw_coco_class,
            }
            for p in preds
        ]

    pred_json_path = seq_out / "predictions.json"
    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)
    print(f"[+] Saved structured predictions to: {pred_json_path}")

    # Generate Annotated Sample Images
    if save_images:
        img_out_dir = seq_out / "annotated_frames"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        # Save sample keyframes (e.g. frame 1, 50, 100, 200)
        sample_indices = [1, min(50, len(image_paths)), min(100, len(image_paths))]
        for idx in sample_indices:
            img_path = image_paths[idx - 1]
            img = cv2.imread(str(img_path))
            if img is not None:
                gt_boxes = [t.bbox for t in seq_ann.frames[idx].targets] if (seq_ann and idx in seq_ann.frames) else None
                ign_region = seq_ann.ignored_region if seq_ann else None
                annotated = DetectionVisualizer.draw_detections(
                    image=img,
                    predictions=preds_by_frame.get(idx, []),
                    ground_truth=gt_boxes,
                    ignored_region=ign_region,
                    frame_num=idx,
                )
                sample_file = img_out_dir / f"frame_{idx:05d}_annotated.jpg"
                cv2.imwrite(str(sample_file), annotated)
        print(f"[+] Saved annotated sample frames to: {img_out_dir}")

    # Generate Video Clip
    if save_video and len(image_paths) > 0:
        video_path = seq_out / f"{sequence_id}_annotated.mp4"
        print(f"[*] Rendering annotated video to {video_path}...")
        gt_by_frame = {f_idx: [t.bbox for t in f_ann.targets] for f_idx, f_ann in seq_ann.frames.items()} if seq_ann else None
        DetectionVisualizer.create_video_from_frames(
            image_paths=image_paths,
            predictions_by_frame=preds_by_frame,
            ground_truth_by_frame=gt_by_frame,
            ignored_region=seq_ann.ignored_region if seq_ann else None,
            output_video_path=video_path,
            fps=25,
        )
        print(f"[+] Saved annotated video to: {video_path}")

    return {
        "sequence_id": sequence_id,
        "stats": stats,
        "predictions_by_frame": preds_by_frame,
        "predictions_file": str(pred_json_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run YOLO11n vehicle detection baseline")
    parser.add_argument("--sequence", type=str, default="MVI_20011", help="Sequence ID (e.g. MVI_20011)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process (for quick smoke tests)")
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    args = parser.parse_args()

    run_sequence_detection(
        sequence_id=args.sequence,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        max_frames=args.max_frames,
        save_video=not args.no_video,
    )


if __name__ == "__main__":
    from typing import Any
    main()
