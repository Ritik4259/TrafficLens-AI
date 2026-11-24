"""
Dataset Inspection and Visualization Module for UA-DETRAC.

Computes sequence-level and dataset-level statistics and renders visual annotations
with color-coded bounding boxes, target IDs, vehicle classes, and ignore regions.
"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from .parser import BBox, FrameAnnotation, IgnoredRegion, SequenceAnnotation


# Canonical class visualization colors (BGR format for OpenCV)
CLASS_COLORS = {
    "car": (0, 200, 0),        # Green
    "bus": (220, 100, 0),      # Blue
    "van": (0, 165, 255),      # Orange
    "others": (180, 0, 180),   # Purple
    "unknown": (128, 128, 128) # Gray
}


@dataclass
class SequenceStatistics:
    """Summary statistics for a single video sequence."""
    name: str
    camera_state: str
    scene_condition: str
    fog_status: str
    total_frames: int
    unique_vehicles: int
    total_bounding_boxes: int
    class_distribution: Dict[str, int]
    density_min: int
    density_max: int
    density_avg: float
    ignored_box_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "camera_state": self.camera_state,
            "scene_condition": self.scene_condition,
            "fog_status": self.fog_status,
            "total_frames": self.total_frames,
            "unique_vehicles": self.unique_vehicles,
            "total_bounding_boxes": self.total_bounding_boxes,
            "class_distribution": self.class_distribution,
            "density": {
                "min": self.density_min,
                "max": self.density_max,
                "avg": round(self.density_avg, 2),
            },
            "ignored_box_count": self.ignored_box_count,
        }


class DatasetInspector:
    """Inspector providing statistical analysis and annotation rendering."""

    @staticmethod
    def inspect_sequence(seq: SequenceAnnotation) -> SequenceStatistics:
        """Computes comprehensive statistics for a single sequence."""
        class_counter: Counter = Counter()
        densities: List[int] = []

        for frame in seq.frames.values():
            densities.append(frame.density)
            for target in frame.targets:
                class_counter[target.attribute.vehicle_type] += 1

        density_min = min(densities) if densities else 0
        density_max = max(densities) if densities else 0
        density_avg = sum(densities) / len(densities) if densities else 0.0

        return SequenceStatistics(
            name=seq.name,
            camera_state=seq.camera_state,
            scene_condition=seq.scene_condition,
            fog_status=seq.fog_status,
            total_frames=seq.total_frames,
            unique_vehicles=len(seq.get_all_target_ids()),
            total_bounding_boxes=seq.total_boxes,
            class_distribution=dict(class_counter),
            density_min=density_min,
            density_max=density_max,
            density_avg=density_avg,
            ignored_box_count=len(seq.ignored_region.boxes),
        )

    @staticmethod
    def inspect_dataset(sequences: Dict[str, SequenceAnnotation]) -> Dict[str, Any]:
        """Aggregates statistics across all sequences in a dataset."""
        total_frames = 0
        total_boxes = 0
        overall_classes: Counter = Counter()
        conditions: Counter = Counter()
        camera_states: Counter = Counter()
        sequence_summaries: List[Dict[str, Any]] = []

        for seq in sequences.values():
            stats = DatasetInspector.inspect_sequence(seq)
            sequence_summaries.append(stats.to_dict())
            total_frames += stats.total_frames
            total_boxes += stats.total_bounding_boxes
            overall_classes.update(stats.class_distribution)
            conditions[stats.scene_condition] += 1
            camera_states[stats.camera_state] += 1

        return {
            "total_sequences": len(sequences),
            "total_frames": total_frames,
            "total_bounding_boxes": total_boxes,
            "conditions": dict(conditions),
            "camera_states": dict(camera_states),
            "class_distribution": dict(overall_classes),
            "sequences": sequence_summaries,
        }

    @staticmethod
    def render_annotations(
        image: np.ndarray,
        frame_ann: FrameAnnotation,
        ignored_region: Optional[IgnoredRegion] = None,
        draw_ignore: bool = True,
        draw_labels: bool = True,
    ) -> np.ndarray:
        """
        Overlays vehicle bounding boxes, target IDs, vehicle classes, and ignore regions on an image.

        Args:
            image: Input image (BGR format).
            frame_ann: Target annotations for this frame.
            ignored_region: Ignored regions for this sequence.
            draw_ignore: If True, renders translucent yellow/gray overlays over ignore regions.
            draw_labels: If True, draws text tags for ID, class, and speed.

        Returns:
            Annotated BGR image as a new numpy array.
        """
        vis = image.copy()
        h, w = vis.shape[:2]

        # 1. Render Ignore Regions (Semi-transparent overlay)
        if draw_ignore and ignored_region and ignored_region.boxes:
            overlay = vis.copy()
            for box in ignored_region.boxes:
                x1, y1 = int(np.clip(box.x1, 0, w - 1)), int(np.clip(box.y1, 0, h - 1))
                x2, y2 = int(np.clip(box.x2, 0, w - 1)), int(np.clip(box.y2, 0, h - 1))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (70, 70, 70), -1)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    vis,
                    "IGNORED",
                    (x1 + 6, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            # Blend overlay
            cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

        # 2. Render Target Detections
        for target in frame_ann.targets:
            bbox = target.bbox
            attr = target.attribute
            color = CLASS_COLORS.get(attr.vehicle_type, CLASS_COLORS["unknown"])

            x1, y1 = int(np.clip(bbox.x1, 0, w - 1)), int(np.clip(bbox.y1, 0, h - 1))
            x2, y2 = int(np.clip(bbox.x2, 0, w - 1)), int(np.clip(bbox.y2, 0, h - 1))

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            if draw_labels:
                label = f"#{target.target_id} {attr.vehicle_type}"
                if attr.speed is not None and attr.speed > 0:
                    label += f" {attr.speed:.0f}km/h"

                # Text background banner
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                )
                banner_y1 = max(0, y1 - text_h - 6)
                banner_y2 = y1
                cv2.rectangle(
                    vis,
                    (x1, banner_y1),
                    (x1 + text_w + 6, banner_y2),
                    color,
                    -1,
                )
                cv2.putText(
                    vis,
                    label,
                    (x1 + 3, banner_y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 3. Render Sequence & Frame Header
        header = f"Frame: {frame_ann.frame_num} | Density: {frame_ann.density} | Targets: {frame_ann.target_count}"
        cv2.putText(
            vis,
            header,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return vis
