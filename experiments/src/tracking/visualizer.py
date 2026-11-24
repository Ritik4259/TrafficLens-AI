"""
Visualization module for multi-object tracking.

Renders bounding boxes with persistent IDs, trajectory motion trails,
translucent ignore regions, and compiles full annotated video clips.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from experiments.src.dataset.parser import BBox, IgnoredRegion
from .tracker import Track


class TrackingVisualizer:
    """Renders persistent tracking annotations, historical trajectories, and video clips."""

    @staticmethod
    def get_track_color(track_id: int) -> Tuple[int, int, int]:
        """Generates a consistent, visually distinct BGR color for a given track ID."""
        # Golden ratio hue distribution for high distinctness between successive IDs
        golden_ratio_conjugate = 0.618033988749895
        h = int(((track_id * golden_ratio_conjugate) % 1.0) * 180)
        s = 200
        v = 240
        hsv_pixel = np.uint8([[[h, s, v]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))

    @classmethod
    def draw_tracks(
        cls,
        image: np.ndarray,
        tracks: List[Track],
        ground_truth: Optional[List[BBox]] = None,
        ignored_region: Optional[IgnoredRegion] = None,
        frame_num: Optional[int] = None,
        sequence_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draws active tracks, historical motion trails, and optional ground truth / ignore zones.
        """
        vis_img = image.copy()
        h, w = vis_img.shape[:2]

        # Draw ignored regions (semi-transparent orange overlay)
        if ignored_region and ignored_region.boxes:
            overlay = vis_img.copy()
            for ign_box in ignored_region.boxes:
                x1, y1, x2, y2 = map(int, ign_box.to_xyxy())
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 140, 255), -1)
                cv2.putText(
                    overlay,
                    "IGNORED",
                    (x1 + 6, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.addWeighted(overlay, 0.22, vis_img, 0.78, 0, vis_img)

        # Draw ground truth boxes (subtle dotted/thin green rectangles)
        if ground_truth:
            for gt in ground_truth:
                gx1, gy1, gx2, gy2 = map(int, gt.to_xyxy())
                cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), (0, 220, 0), 1, cv2.LINE_AA)

        # Draw tracking trajectories and bounding boxes
        for track in tracks:
            color = cls.get_track_color(track.track_id)
            x1, y1, x2, y2 = map(int, track.bbox.to_xyxy())

            # Draw trajectory path trail
            if len(track.history) > 1:
                pts = np.array([[int(px), int(py)] for px, py in track.history], dtype=np.int32)
                # Draw polyline with fading thickness
                cv2.polylines(vis_img, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
                # Small circle at vehicle center
                cx, cy = map(int, track.center)
                cv2.circle(vis_img, (cx, cy), 3, color, -1, cv2.LINE_AA)

            # Draw vehicle bounding box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label badge
            label = f"#{track.track_id} {track.confidence:.2f}"
            font_scale = 0.50
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            badge_y1 = max(0, y1 - th - 6)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = x1 + tw + 8

            # Badge background
            cv2.rectangle(vis_img, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            # Text (white or black depending on brightness)
            text_color = (255, 255, 255)
            cv2.putText(
                vis_img,
                label,
                (badge_x1 + 4, badge_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        # Header HUD
        header_text = []
        if sequence_name:
            header_text.append(f"Seq: {sequence_name}")
        if frame_num is not None:
            header_text.append(f"Frame: {frame_num:05d}")
        header_text.append(f"Active Tracks: {len(tracks)}")
        hud_str = " | ".join(header_text)

        cv2.rectangle(vis_img, (10, 10), (10 + len(hud_str) * 11, 38), (20, 20, 20), -1)
        cv2.putText(
            vis_img,
            hud_str,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return vis_img

    @classmethod
    def create_tracking_video(
        cls,
        image_paths: List[Path],
        tracks_by_frame: Dict[int, List[Track]],
        ground_truth_by_frame: Optional[Dict[int, List[BBox]]] = None,
        ignored_region: Optional[IgnoredRegion] = None,
        output_video_path: Path = Path("tracking_output.mp4"),
        fps: int = 25,
        sequence_name: Optional[str] = None,
    ) -> Path:
        """Renders an annotated tracking MP4 video."""
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        if not image_paths:
            raise ValueError("No images provided for video creation")

        first_img = cv2.imread(str(image_paths[0]))
        if first_img is None:
            raise ValueError(f"Could not read first image: {image_paths[0]}")
        h, w = first_img.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        try:
            for idx, img_path in enumerate(image_paths, start=1):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                frame_tracks = tracks_by_frame.get(idx, [])
                frame_gts = ground_truth_by_frame.get(idx, None) if ground_truth_by_frame else None

                annotated = cls.draw_tracks(
                    image=img,
                    tracks=frame_tracks,
                    ground_truth=frame_gts,
                    ignored_region=ignored_region,
                    frame_num=idx,
                    sequence_name=sequence_name,
                )
                writer.write(annotated)
        finally:
            writer.release()

        return output_video_path
