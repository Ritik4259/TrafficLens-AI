"""
Selective Extraction and Annotation Conversion for UA-DETRAC Curated Benchmark.

Extracts ONLY the 8 curated sequences from ua_detrac_training_set.zip into:
  data/raw/images/<sequence_id>/image000001.jpg ...
  data/raw/annotations/<sequence_id>.xml
Preserves original source annotations.csv and annotations.zip without modifying archives.
Includes frame contiguity and duplicate validation checks.
"""

from __future__ import annotations
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DESKTOP = Path(r"c:\Users\bhard\Desktop")
TRAIN_ZIP = DESKTOP / "ua_detrac_training_set.zip"

CURATED_SEQUENCES: Dict[str, Dict[str, str]] = {
    "MVI_39801": {"prefix": "detrac_17_MVI_39801", "condition": "cloudy", "camera_state": "stable"},
    "MVI_40141": {"prefix": "detrac_24_MVI_40141", "condition": "rainy", "camera_state": "stable"},
    "MVI_63521": {"prefix": "detrac_52_MVI_63521", "condition": "sunny", "camera_state": "stable"},
    "MVI_63561": {"prefix": "detrac_58_MVI_63561", "condition": "sunny", "camera_state": "unstable"},
    "MVI_20012": {"prefix": "detrac_2_MVI_20012", "condition": "night", "camera_state": "unstable"},
    "MVI_39811": {"prefix": "detrac_18_MVI_39811", "condition": "cloudy", "camera_state": "stable"},
    "MVI_20011": {"prefix": "detrac_1_MVI_20011", "condition": "night", "camera_state": "unstable"},
    "MVI_40131": {"prefix": "detrac_23_MVI_40131", "condition": "rainy", "camera_state": "unstable"},
}


def validate_frame_contiguity(frame_numbers: List[int], sequence_name: str) -> None:
    """Verifies that frame numbers are strictly contiguous starting from 1 with no gaps or duplicates."""
    if not frame_numbers:
        raise ValueError(f"Sequence {sequence_name} has no frames!")
    
    sorted_frames = sorted(frame_numbers)
    if len(sorted_frames) != len(set(sorted_frames)):
        duplicates = [x for x in sorted_frames if sorted_frames.count(x) > 1]
        raise ValueError(f"Sequence {sequence_name} has duplicate frame numbers: {set(duplicates)}")
    
    expected = list(range(1, len(sorted_frames) + 1))
    if sorted_frames != expected:
        missing = set(expected) - set(sorted_frames)
        raise ValueError(f"Sequence {sequence_name} has missing frames! Gaps detected: {sorted(list(missing))[:10]}")


def convert_csv_to_sequence_xml(
    df: pd.DataFrame,
    sequence_name: str,
    condition: str,
    camera_state: str,
) -> str:
    """Converts sequence DataFrame into canonical UA-DETRAC XML string."""
    seq_elem = ET.Element("sequence", name=sequence_name)
    ET.SubElement(
        seq_elem,
        "sequence_attribute",
        camera_state=camera_state,
        sence_condition=condition,
        fog_status="clear",
    )

    # 1. Extract Ignore Regions (unique static ignore boxes across frames)
    ign_df = df[df["class"] == "ignored"]
    ignored_elem = ET.SubElement(seq_elem, "ignored_region")
    if not ign_df.empty:
        unique_ign_boxes = ign_df.drop_duplicates(subset=["xmin", "ymin", "xmax", "ymax"])
        for _, row in unique_ign_boxes.iterrows():
            left = float(row["xmin"])
            top = float(row["ymin"])
            width = float(row["xmax"] - row["xmin"])
            height = float(row["ymax"] - row["ymin"])
            ET.SubElement(
                ignored_elem,
                "box",
                left=f"{left:.2f}",
                top=f"{top:.2f}",
                width=f"{width:.2f}",
                height=f"{height:.2f}",
            )

    # 2. Extract Frames and Vehicles
    df["frame_num"] = df["filename"].apply(lambda fn: int(re.search(r"(\d+)", fn).group(1)))
    frame_numbers = df["frame_num"].unique().tolist()
    validate_frame_contiguity(frame_numbers, sequence_name)

    grouped_frames = df.groupby("frame_num")

    for frame_num in sorted(grouped_frames.groups.keys()):
        frame_group = grouped_frames.get_group(frame_num)
        vehicle_rows = frame_group[frame_group["class"] == "vehicle"]

        density = len(vehicle_rows)
        frame_elem = ET.SubElement(seq_elem, "frame", num=str(frame_num), density=str(density))
        target_list_elem = ET.SubElement(frame_elem, "target_list")

        for _, row in vehicle_rows.iterrows():
            target_id = int(row["target_id"]) if pd.notna(row["target_id"]) else 1
            left = float(row["xmin"])
            top = float(row["ymin"])
            width = float(row["xmax"] - row["xmin"])
            height = float(row["ymax"] - row["ymin"])

            target_elem = ET.SubElement(target_list_elem, "target", id=str(target_id))
            ET.SubElement(
                target_elem,
                "box",
                left=f"{left:.2f}",
                top=f"{top:.2f}",
                width=f"{width:.2f}",
                height=f"{height:.2f}",
            )
            ET.SubElement(
                target_elem,
                "attribute",
                vehicle_type="vehicle",
                color="unknown",
                speed="0.0",
                truncation_ratio="0.0",
                occluded_ratio="0.0",
            )

    rough_str = ET.tostring(seq_elem, encoding="utf-8")
    reparsed = minidom.parseString(rough_str)
    return reparsed.toprettyxml(indent="    ")


def extract_curated_sequences():
    print("=" * 80)
    print("🚀 Extracting Curated UA-DETRAC Benchmark Sequences (8 of 60)")
    print(f"Source Archive: {TRAIN_ZIP}")
    print("=" * 80)

    images_root = PROJECT_ROOT / "data" / "raw" / "images"
    annotations_root = PROJECT_ROOT / "data" / "raw" / "annotations"
    images_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(TRAIN_ZIP, "r") as zf:
        archive_names = set(zf.namelist())

        summary_records = []
        for seq_name, meta in CURATED_SEQUENCES.items():
            prefix = meta["prefix"]
            seq_img_dir = images_root / seq_name
            seq_img_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[*] Processing {seq_name} ({prefix})...")

            csv_path_in_zip = f"{prefix}/annotations.csv"
            if csv_path_in_zip not in archive_names:
                raise FileNotFoundError(f"Missing {csv_path_in_zip} in archive!")

            csv_bytes = zf.read(csv_path_in_zip)
            df = pd.read_csv(io.BytesIO(csv_bytes))

            raw_csv_path = annotations_root / f"{seq_name}_annotations.csv"
            raw_csv_path.write_bytes(csv_bytes)

            inner_zip_in_zip = f"{prefix}/annotations.zip"
            if inner_zip_in_zip in archive_names:
                raw_inner_zip_path = annotations_root / f"{seq_name}_voc_annotations.zip"
                raw_inner_zip_path.write_bytes(zf.read(inner_zip_in_zip))

            xml_str = convert_csv_to_sequence_xml(
                df=df,
                sequence_name=seq_name,
                condition=meta["condition"],
                camera_state=meta["camera_state"],
            )
            xml_output_path = annotations_root / f"{seq_name}.xml"
            xml_output_path.write_text(xml_str, encoding="utf-8")

            seq_jpgs = [n for n in archive_names if n.startswith(prefix + "/") and n.lower().endswith(".jpg")]
            print(f"    Extracting {len(seq_jpgs)} frames to {seq_img_dir}...")

            for jpg_name in seq_jpgs:
                filename = jpg_name.split("/")[-1]
                target_file = seq_img_dir / filename
                if not target_file.exists():
                    target_file.write_bytes(zf.read(jpg_name))

            num_extracted_frames = len(list(seq_img_dir.glob("*.jpg")))
            vehicle_boxes = len(df[df["class"] == "vehicle"])
            ignored_boxes = len(df[df["class"] == "ignored"])
            unique_targets = len(df[df["class"] == "vehicle"]["target_id"].unique())

            summary_records.append({
                "sequence": seq_name,
                "condition": meta["condition"],
                "camera_state": meta["camera_state"],
                "frames_extracted": num_extracted_frames,
                "vehicle_annotations": vehicle_boxes,
                "ignored_annotations": ignored_boxes,
                "unique_vehicles": unique_targets,
                "xml_path": str(xml_output_path),
            })
            print(f"    [+] Successfully verified {seq_name}: {num_extracted_frames} frames, {vehicle_boxes} vehicle boxes.")

    print("\n" + "=" * 80)
    print("📊 Extraction & Canonical Conversion Summary:")
    print("=" * 80)
    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))

    total_frames = summary_df["frames_extracted"].sum()
    total_vehicles = summary_df["vehicle_annotations"].sum()
    print(f"\nTotal Frames Extracted: {total_frames:,}")
    print(f"Total Ground Truth Vehicle Boxes: {total_vehicles:,}")
    print("All original archives preserved intact.")


if __name__ == "__main__":
    extract_curated_sequences()
