"""
MOTChallenge format exporter and serialization for tracking results.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from experiments.src.dataset.parser import BBox, SequenceAnnotation
from .tracker import Track


class MOTExporter:
    """Handles serialization of tracks to standard MOTChallenge TXT and JSON formats."""

    @staticmethod
    def export_tracks_to_mot_txt(
        tracks_by_frame: Dict[int, List[Track]],
        output_path: Path,
    ) -> None:
        """
        Exports tracking results in standard MOTChallenge format:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []

        for frame_idx in sorted(tracks_by_frame.keys()):
            for t in tracks_by_frame[frame_idx]:
                line = (
                    f"{frame_idx},{t.track_id},"
                    f"{t.bbox.left:.2f},{t.bbox.top:.2f},"
                    f"{t.bbox.width:.2f},{t.bbox.height:.2f},"
                    f"{t.confidence:.4f},-1,-1,-1\n"
                )
                lines.append(line)

        output_path.write_text("".join(lines), encoding="utf-8")

    @staticmethod
    def export_tracks_to_json(
        tracks_by_frame: Dict[int, List[Track]],
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Exports tracks with trajectory histories to structured JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serialized: Dict[str, Any] = {
            "metadata": metadata or {},
            "frames": {},
        }

        for frame_idx, tracks in tracks_by_frame.items():
            serialized["frames"][str(frame_idx)] = [
                {
                    "track_id": t.track_id,
                    "bbox": {
                        "left": round(t.bbox.left, 2),
                        "top": round(t.bbox.top, 2),
                        "width": round(t.bbox.width, 2),
                        "height": round(t.bbox.height, 2),
                    },
                    "confidence": round(t.confidence, 4),
                    "class_name": t.class_name,
                    "age": t.age,
                    "history": [[round(x, 1), round(y, 1)] for x, y in t.history],
                }
                for t in tracks
            ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    @staticmethod
    def export_gt_to_mot_txt(
        seq_ann: SequenceAnnotation,
        output_path: Path,
        include_ignored_boxes: bool = True,
    ) -> None:
        """
        Exports sequence ground truth to standard MOTChallenge format:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <consider>, <class>, <visibility>
        - Valid vehicles: consider=1, class=1, visibility=1.0
        - Ignored regions (if enabled): id=-1, consider=0, class=0
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []

        for frame_idx in sorted(seq_ann.frames.keys()):
            f_ann = seq_ann.frames[frame_idx]
            for target in f_ann.targets:
                b = target.bbox
                lines.append(
                    f"{frame_idx},{target.target_id},"
                    f"{b.left:.2f},{b.top:.2f},{b.width:.2f},{b.height:.2f},"
                    f"1,1,1.0\n"
                )

            # Frame-level ignore region boxes
            if include_ignored_boxes and seq_ann.ignored_region:
                for ign_box in seq_ann.ignored_region.boxes:
                    lines.append(
                        f"{frame_idx},-1,"
                        f"{ign_box.left:.2f},{ign_box.top:.2f},{ign_box.width:.2f},{ign_box.height:.2f},"
                        f"0,0,0.0\n"
                    )

        output_path.write_text("".join(lines), encoding="utf-8")

    @staticmethod
    def load_mot_txt(txt_path: Path) -> Dict[int, List[Track]]:
        """Parses a MOTChallenge format tracking text file back into Track objects."""
        tracks_by_frame: Dict[int, List[Track]] = {}
        if not txt_path.exists():
            return tracks_by_frame

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 7:
                    continue
                frame_idx = int(float(parts[0]))
                track_id = int(float(parts[1]))
                if track_id < 0:
                    continue  # ignore region marker

                left = float(parts[2])
                top = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                conf = float(parts[6])

                track = Track(
                    track_id=track_id,
                    bbox=BBox(left=left, top=top, width=width, height=height),
                    confidence=conf,
                )
                if frame_idx not in tracks_by_frame:
                    tracks_by_frame[frame_idx] = []
                tracks_by_frame[frame_idx].append(track)

        return tracks_by_frame
