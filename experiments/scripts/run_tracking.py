"""
CLI inference script for running ByteTrack vehicle tracking and rendering annotated video.

Outputs:
1. Annotated MP4 video with vehicle trajectories, persistent IDs, and HUD.
2. Keyframe image captures illustrating:
   - Vehicles entering the scene
   - Vehicles leaving the scene
   - Temporarily occluded vehicles
   - Dense traffic
   - Nighttime traffic
3. Standard MOTChallenge format TXT and structured JSON exports.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.src.dataset.parser import parse_detrac_xml, BBox
from experiments.src.detection.config import DetectionConfig
from experiments.src.detection.detector import VehicleDetector
from experiments.src.detection.metrics import Prediction
from experiments.src.tracking.config import TrackingConfig
from experiments.src.tracking.tracker import Track, VehicleTracker
from experiments.src.tracking.visualizer import TrackingVisualizer
from experiments.src.tracking.exporter import MOTExporter


def run_sequence_tracking(
    sequence_id: str = "MVI_20011",
    track_high_thresh: float = 0.40,
    track_low_thresh: float = 0.10,
    new_track_thresh: float = 0.50,
    track_buffer: int = 45,
    match_thresh: float = 0.70,
    max_frames: Optional[int] = None,
    save_video: bool = True,
    save_keyframes: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"🚦 Running ByteTrack Vehicle Tracking on Sequence: {sequence_id}")
    print("=" * 80)

    seq_out = output_dir or (PROJECT_ROOT / "experiments" / "outputs" / "tracking" / sequence_id)
    seq_out.mkdir(parents=True, exist_ok=True)
    cache_dir = PROJECT_ROOT / "data" / "cache" / "predictions"

    images_dir = PROJECT_ROOT / "data" / "raw" / "images" / sequence_id
    ann_path = PROJECT_ROOT / "data" / "raw" / "annotations" / f"{sequence_id}.xml"

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    seq_ann = parse_detrac_xml(ann_path) if ann_path.exists() else None
    image_paths = sorted(list(images_dir.glob("*.jpg")))
    if max_frames:
        image_paths = image_paths[:max_frames]

    print(f"[*] Sequence frames: {len(image_paths)}")
    if seq_ann:
        print(f"[*] Condition: {seq_ann.scene_condition} | Camera: {seq_ann.camera_state} | Static Ignore Boxes: {seq_ann.static_ignored_box_count}")

    # Load raw detections from cache or run detector
    cache_file = cache_dir / f"preds_raw_{sequence_id}.json"
    if cache_file.exists():
        print(f"[*] Loading raw detections from cache: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        preds_by_frame = {
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
                )
                for p in v
            ]
            for k, v in data.items()
            if not max_frames or int(k) <= max_frames
        }
    else:
        print(f"[*] Cache not found. Running YOLO11n detection on {len(image_paths)} frames...")
        detector = VehicleDetector(DetectionConfig(confidence_threshold=0.05))
        preds_by_frame, det_stats = detector.predict_sequence(image_paths)

    # Initialize Tracker
    config = TrackingConfig(
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=new_track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
    )
    tracker = VehicleTracker(config)

    print(f"[*] Tracking with ByteTrack (high={track_high_thresh}, low={track_low_thresh}, buffer={track_buffer})...")
    tracks_by_frame, track_stats = tracker.track_sequence(preds_by_frame)
    print(f"[+] Tracking complete: {track_stats['fps']:.1f} FPS | Avg Latency: {track_stats['avg_latency_ms']:.2f}ms | Unique IDs: {track_stats['unique_track_ids']}")

    # Export MOTChallenge TXT and JSON
    mot_txt_path = seq_out / f"{sequence_id}_tracking.txt"
    json_path = seq_out / f"{sequence_id}_tracking.json"
    MOTExporter.export_tracks_to_mot_txt(tracks_by_frame, mot_txt_path)
    MOTExporter.export_tracks_to_json(tracks_by_frame, json_path, metadata={"sequence": sequence_id, "config": config.__dict__})
    print(f"[+] Saved MOT format tracks to: {mot_txt_path}")
    print(f"[+] Saved structured JSON tracks to: {json_path}")

    # Generate Annotated Keyframes
    if save_keyframes and len(image_paths) > 0:
        keyframe_dir = seq_out / "annotated_keyframes"
        keyframe_dir.mkdir(parents=True, exist_ok=True)

        # Select representative keyframes across the sequence
        total_f = len(image_paths)
        sample_indices = [
            1,
            min(50, total_f),
            min(150, total_f),
            min(300, total_f),
            min(500, total_f),
            total_f,
        ]
        sample_indices = sorted(list(set(sample_indices)))

        for idx in sample_indices:
            img = cv2.imread(str(image_paths[idx - 1]))
            if img is not None:
                gt_boxes = [t.bbox for t in seq_ann.frames[idx].targets] if (seq_ann and idx in seq_ann.frames) else None
                annotated = TrackingVisualizer.draw_tracks(
                    image=img,
                    tracks=tracks_by_frame.get(idx, []),
                    ground_truth=gt_boxes,
                    ignored_region=seq_ann.ignored_region if seq_ann else None,
                    frame_num=idx,
                    sequence_name=sequence_id,
                )
                kf_path = keyframe_dir / f"frame_{idx:05d}_tracked.jpg"
                cv2.imwrite(str(kf_path), annotated)
        print(f"[+] Saved representative annotated keyframes to: {keyframe_dir}")

    # Generate Video Clip
    if save_video and len(image_paths) > 0:
        video_path = seq_out / f"{sequence_id}_tracked.mp4"
        print(f"[*] Rendering annotated tracking video to {video_path}...")
        gt_by_frame = {f: [t.bbox for t in seq_ann.frames[f].targets] for f in seq_ann.frames} if seq_ann else None
        TrackingVisualizer.create_tracking_video(
            image_paths=image_paths,
            tracks_by_frame=tracks_by_frame,
            ground_truth_by_frame=gt_by_frame,
            ignored_region=seq_ann.ignored_region if seq_ann else None,
            output_video_path=video_path,
            fps=25,
            sequence_name=sequence_id,
        )
        print(f"[+] Saved annotated tracking video to: {video_path}")

    return {
        "sequence_id": sequence_id,
        "track_stats": track_stats,
        "tracks_by_frame": tracks_by_frame,
        "mot_txt_file": str(mot_txt_path),
        "json_file": str(json_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run ByteTrack vehicle tracking on a sequence")
    parser.add_argument("--sequence", type=str, default="MVI_20011", help="Sequence ID (e.g. MVI_20011)")
    parser.add_argument("--high-thresh", type=float, default=0.40, help="High confidence threshold")
    parser.add_argument("--low-thresh", type=float, default=0.10, help="Low confidence threshold")
    parser.add_argument("--new-thresh", type=float, default=0.50, help="New track confidence threshold")
    parser.add_argument("--match-thresh", type=float, default=0.70, help="IoU match threshold")
    parser.add_argument("--track-buffer", type=int, default=45, help="Track buffer in frames")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to track")
    parser.add_argument("--no-video", action="store_true", help="Skip video rendering")
    args = parser.parse_args()

    run_sequence_tracking(
        sequence_id=args.sequence,
        track_high_thresh=args.high_thresh,
        track_low_thresh=args.low_thresh,
        new_track_thresh=args.new_thresh,
        match_thresh=args.match_thresh,
        track_buffer=args.track_buffer,
        max_frames=args.max_frames,
        save_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
