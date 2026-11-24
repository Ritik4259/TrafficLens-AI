"""
Visualization module for vehicle detections, ground truth, and ignore regions.
Supports single-image rendering and sequence-to-video export.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import cv2

from experiments.src.dataset.parser import BBox, IgnoredRegion
from .metrics import Prediction


class DetectionVisualizer:
    """Renders predictions, ground truth, and ignored regions on images and videos."""

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        predictions: List[Prediction],
        ground_truth: Optional[List[BBox]] = None,
        ignored_region: Optional[IgnoredRegion] = None,
        frame_num: Optional[int] = None,
        latency_ms: Optional[float] = None,
    ) -> np.ndarray:
        """
        Overlays bounding boxes and metadata onto an image.

        Colors (BGR):
        - Predictions: Bright Green (0, 220, 0)
        - Ground Truth: Cyan (240, 200, 0)
        - Ignored Regions: Dark translucent fill + Red border (0, 0, 255)
        """
        vis = image.copy()
        h, w = vis.shape[:2]

        # 1. Overlay Ignore Regions
        if ignored_region is not None and ignored_region.boxes:
            overlay = vis.copy()
            for ign in ignored_region.boxes:
                x1, y1 = int(np.clip(ign.x1, 0, w - 1)), int(np.clip(ign.y1, 0, h - 1))
                x2, y2 = int(np.clip(ign.x2, 0, w - 1)), int(np.clip(ign.y2, 0, h - 1))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 60, 60), -1)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    vis,
                    "IGNORED",
                    (x1 + 6, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

        # 2. Overlay Ground Truth (if provided)
        if ground_truth:
            for gt in ground_truth:
                gx1, gy1 = int(np.clip(gt.x1, 0, w - 1)), int(np.clip(gt.y1, 0, h - 1))
                gx2, gy2 = int(np.clip(gt.x2, 0, w - 1)), int(np.clip(gt.y2, 0, h - 1))
                cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (240, 200, 0), 1, cv2.LINE_AA)

        # 3. Overlay Predictions
        for pred in predictions:
            b = pred.bbox
            x1, y1 = int(np.clip(b.x1, 0, w - 1)), int(np.clip(b.y1, 0, h - 1))
            x2, y2 = int(np.clip(b.x2, 0, w - 1)), int(np.clip(b.y2, 0, h - 1))

            color = (0, 230, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            label = f"{pred.class_name} {pred.confidence * 100:.0f}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            y_banner_1 = max(0, y1 - th - 5)
            y_banner_2 = y1
            cv2.rectangle(vis, (x1, y_banner_1), (x1 + tw + 6, y_banner_2), color, -1)
            cv2.putText(
                vis,
                label,
                (x1 + 3, y_banner_2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 4. Top Telemetry Banner
        banner_text = f"Detections: {len(predictions)}"
        if ground_truth is not None:
            banner_text += f" | GT: {len(ground_truth)}"
        if frame_num is not None:
            banner_text = f"Frame {frame_num} | " + banner_text
        if latency_ms is not None:
            banner_text += f" | Latency: {latency_ms:.1f}ms"

        cv2.putText(
            vis,
            banner_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    @classmethod
    def create_video_from_frames(
        cls,
        image_paths: List[Path],
        predictions_by_frame: Dict[int, List[Prediction]],
        ground_truth_by_frame: Optional[Dict[int, List[BBox]]] = None,
        ignored_region: Optional[IgnoredRegion] = None,
        output_video_path: Union[str, Path] = "output.mp4",
        fps: int = 25,
        max_frames: Optional[int] = None,
    ) -> Path:
        """
        Compiles an annotated sequence video from frame images and detection results.
        """
        out_path = Path(output_video_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        selected_paths = image_paths[:max_frames] if max_frames else image_paths
        if not selected_paths:
            raise ValueError("No images provided for video creation!")

        first_img = cv2.imread(str(selected_paths[0]))
        if first_img is None:
            raise ValueError(f"Failed to load initial image: {selected_paths[0]}")
        h, w = first_img.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        for idx, img_path in enumerate(selected_paths, start=1):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            frame_preds = predictions_by_frame.get(idx, [])
            frame_gts = ground_truth_by_frame.get(idx, []) if ground_truth_by_frame else None

            annotated = cls.draw_detections(
                image=img,
                predictions=frame_preds,
                ground_truth=frame_gts,
                ignored_region=ignored_region,
                frame_num=idx,
            )
            writer.write(annotated)

        writer.release()
        return out_path
