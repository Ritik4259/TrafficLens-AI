"""
Verification and Demonstration Script for UA-DETRAC Dataset Parser & Inspector.

Runs end-to-end XML parsing against REAL extracted UA-DETRAC sequences,
produces comprehensive statistical inspection across all 8 curated sequences,
and renders visual ground truth annotations onto ACTUAL traffic camera frames.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.src.dataset.parser import (
    parse_detrac_xml,
    parse_all_sequences,
    CANONICAL_VEHICLE_TYPES,
)
from experiments.src.dataset.inspector import DatasetInspector
from experiments.src.dataset.curator import SequenceCurator


def main():
    print("=" * 80)
    print("🚦 UA-DETRAC Dataset Verification on REAL Curated Benchmark Sequences")
    print("=" * 80)

    annotations_dir = PROJECT_ROOT / "data" / "raw" / "annotations"
    images_dir = PROJECT_ROOT / "data" / "raw" / "images"

    # Step 1: Parse all real extracted sequences
    print(f"\n[*] Scanning XML annotations in {annotations_dir}...")
    sequences = parse_all_sequences(annotations_dir)
    print(f"[+] Successfully loaded {len(sequences)} sequence annotations: {sorted(list(sequences.keys()))}")

    # Step 2: Verification of Sequence MVI_20011 (Real Ground Truth)
    seq = sequences["MVI_20011"]
    print(f"\n[+] Detailed Audit for Sequence {seq.name}:")
    print(f"    - Camera State:     {seq.camera_state}")
    print(f"    - Scene Condition:  {seq.scene_condition}")
    print(f"    - Fog Status:       {seq.fog_status}")
    print(f"    - Total Frames:     {seq.total_frames:,}")
    print(f"    - Vehicle Targets:  {seq.total_boxes:,}")
    print(f"    - Unique Vehicles:  {len(seq.get_all_target_ids())}")
    print(f"    - Ignored Boxes:    {len(seq.ignored_region.boxes)}")

    # Verify frame alignment with actual images
    seq_img_dir = images_dir / seq.name
    actual_imgs = list(seq_img_dir.glob("*.jpg"))
    print(f"    - Actual JPG Frames on Disk: {len(actual_imgs):,}")
    assert len(actual_imgs) == seq.total_frames, f"Frame count mismatch: {len(actual_imgs)} != {seq.total_frames}"
    print("    - Frame Alignment Check: PASSED (Exact 1:1 match)")

    # Step 3: Global Dataset Inspection across all 8 sequences
    print("\n" + "=" * 80)
    print("📊 Aggregate Statistical Inspection (Curated 8-Sequence Benchmark)")
    print("=" * 80)
    dataset_stats = DatasetInspector.inspect_dataset(sequences)
    print(f"Total Sequences:        {dataset_stats['total_sequences']}")
    print(f"Total Video Frames:     {dataset_stats['total_frames']:,}")
    print(f"Total Bounding Boxes:   {dataset_stats['total_bounding_boxes']:,}")
    print(f"Conditions Represented: {dataset_stats['conditions']}")
    print(f"Camera Stability:       {dataset_stats['camera_states']}")
    print(f"Class Distribution:     {dataset_stats['class_distribution']}")

    print("\nPer-Sequence Breakdown:")
    for s in dataset_stats["sequences"]:
        print(f"  - {s['name']:<10} | Cond: {s['scene_condition']:<7} ({s['camera_state']:<8}) | "
              f"Frames: {s['total_frames']:<5} | Vehicles: {s['total_bounding_boxes']:<6} | "
              f"IDs: {s['unique_vehicles']:<3} | Ignored: {s['ignored_box_count']}")

    # Step 4: Render Annotation Visualization onto ACTUAL Video Frame
    out_dir = PROJECT_ROOT / "experiments" / "outputs" / "inspections"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_num = 1
    real_img_path = seq.get_frame_image_path(seq_img_dir, frame_num)
    print(f"\n[*] Reading real video frame from: {real_img_path}")
    assert real_img_path.exists(), f"Missing frame image: {real_img_path}"
    real_img = cv2.imread(str(real_img_path))
    assert real_img is not None, "Failed to decode image!"
    print(f"    - Image resolution: {real_img.shape[1]}x{real_img.shape[0]} px, {real_img.shape[2]} channels")

    frame_ann = seq.frames[frame_num]
    vis_img = DatasetInspector.render_annotations(
        real_img, frame_ann, seq.ignored_region, draw_ignore=True, draw_labels=True
    )
    vis_path = out_dir / "sample_annotation_MVI_20011_frame1.jpg"
    cv2.imwrite(str(vis_path), vis_img)
    print(f"[+] Saved REAL traffic camera annotated visualization to: {vis_path}")

    # Also render a mid-sequence frame with active vehicle traffic (e.g. frame 100)
    if 100 in seq.frames:
        f100_path = seq.get_frame_image_path(seq_img_dir, 100)
        f100_img = cv2.imread(str(f100_path))
        if f100_img is not None:
            vis_f100 = DatasetInspector.render_annotations(
                f100_img, seq.frames[100], seq.ignored_region, draw_ignore=True, draw_labels=True
            )
            vis_f100_path = out_dir / "sample_annotation_MVI_20011_frame100.jpg"
            cv2.imwrite(str(vis_f100_path), vis_f100)
            print(f"[+] Saved active traffic frame visualization to: {vis_f100_path}")

    # Step 5: Verify Curated Manifest
    curator = SequenceCurator(seed=42)
    manifest = curator.get_default_curated_subset()
    manifest_path = PROJECT_ROOT / "data" / "processed" / "curated_subset" / "curated_manifest.json"
    curator.save_manifest(manifest, manifest_path)
    print(f"\n[+] Curated manifest updated at: {manifest_path}")

    print("\n" + "=" * 80)
    print("✅ All real UA-DETRAC dataset verifications and alignments succeeded!")
    print("=" * 80)


if __name__ == "__main__":
    main()
