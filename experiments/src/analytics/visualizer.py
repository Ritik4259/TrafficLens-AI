"""
Analytics visualizer module for TrafficLens-AI.

Renders virtual counting lines, ROI polygons with translucent occupancy shading,
congestion badges, vehicle trajectory trails, and stopped-vehicle event highlights.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from experiments.src.tracking.tracker import Track
from .config import AnalyticsConfig, CountingLineConfig, ROIConfig
from .models import FrameTelemetry, CongestionLevel, StoppedVehicleEvent


class AnalyticsVisualizer:
    """
    Renders analytics annotations onto video frames and compiles annotated MP4 videos.
    """

    CONGESTION_COLORS = {
        CongestionLevel.LOW: (0, 200, 0),       # Green
        CongestionLevel.MODERATE: (0, 165, 255), # Amber / Orange
        CongestionLevel.HIGH: (0, 0, 255),      # Red
    }

    def __init__(self, config: AnalyticsConfig):
        self.config = config

    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Golden ratio color hash for distinct track identification."""
        hue = int((track_id * 137.5) % 180)
        hsv_pixel = np.uint8([[[hue, 220, 240]]])
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        telemetry: FrameTelemetry,
        line_counts: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> np.ndarray:
        """
        Draws complete analytics overlays onto a video frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Draw ROIs with translucent fills
        overlay = annotated.copy()
        for density, cong_state in zip(telemetry.roi_densities, telemetry.congestion_states):
            roi_cfg = next((r for r in self.config.rois if r.roi_id == density.roi_id), None)
            if not roi_cfg or len(roi_cfg.polygon) < 3:
                continue

            poly_pts = np.array(roi_cfg.polygon, dtype=np.int32).reshape((-1, 1, 2))
            color = self.CONGESTION_COLORS.get(cong_state.level, (0, 200, 0))

            # Translucent fill
            cv2.fillPoly(overlay, [poly_pts], color)
            # Boundary
            cv2.polylines(annotated, [poly_pts], isClosed=True, color=color, thickness=2)

            # ROI label badge at polygon centroid
            cx = int(np.mean([p[0] for p in roi_cfg.polygon]))
            cy = int(np.mean([p[1] for p in roi_cfg.polygon]))
            badge_text = f"{roi_cfg.name}: {density.vehicle_count} veh | Occ: {density.occupancy_ratio:.1%} | {cong_state.level.value.upper()}"
            cv2.putText(
                annotated,
                badge_text,
                (max(10, cx - 120), max(20, cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Blend translucent ROI fills
        cv2.addWeighted(overlay, 0.20, annotated, 0.80, 0, annotated)

        # 2. Draw virtual counting lines
        for line in self.config.lines:
            p1 = (int(line.p1[0]), int(line.p1[1]))
            p2 = (int(line.p2[0]), int(line.p2[1]))

            # Thick cyan line
            cv2.line(annotated, p1, p2, (255, 255, 0), 2, cv2.LINE_AA)

            # Draw directional normal tick mark at midpoint
            mid_x = (line.p1[0] + line.p2[0]) / 2.0
            mid_y = (line.p1[1] + line.p2[1]) / 2.0
            line_dx = line.p2[0] - line.p1[0]
            line_dy = line.p2[1] - line.p1[1]
            length = math.hypot(line_dx, line_dy)
            if length > 0:
                nx = -line_dy / length * 20.0
                ny = line_dx / length * 20.0
                norm_pt = (int(mid_x + nx), int(mid_y + ny))
                cv2.arrowedLine(annotated, (int(mid_x), int(mid_y)), norm_pt, (0, 255, 255), 2, tipLength=0.3)

            # Line counts text
            l_counts = (line_counts or {}).get(line.line_id, {})
            total_l = l_counts.get("total", 0)
            down_l = l_counts.get(line.positive_direction_label, 0)
            up_l = l_counts.get(line.negative_direction_label, 0)
            line_str = f"{line.name}: {total_l} (In: {down_l}, Out: {up_l})"
            cv2.putText(
                annotated,
                line_str,
                (p1[0] + 5, p1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Draw active stopped vehicle highlights
        stopped_track_ids = {e.track_id: e for e in telemetry.active_stopped_events}

        # 4. Draw active tracks and trajectories
        speed_map = {s.track_id: s for s in telemetry.speed_measurements}

        for track in tracks:
            tid = track.track_id
            is_stopped = tid in stopped_track_ids
            color = (0, 0, 255) if is_stopped else self._get_track_color(tid)

            # Draw trajectory trail
            if len(track.history) > 1:
                pts = np.array(track.history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Draw bounding box
            x1, y1, x2, y2 = int(track.bbox.x1), int(track.bbox.y1), int(track.bbox.x2), int(track.bbox.y2)
            box_thick = 3 if is_stopped else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thick)

            # Label text
            speed_val = speed_map.get(tid)
            speed_str = f"{speed_val.speed_px_per_s:.0f}px/s" if speed_val else ""
            if is_stopped:
                dur = stopped_track_ids[tid].duration_s
                label = f"ID:{tid} STOPPED ({dur:.1f}s)"
            else:
                label = f"ID:{tid} {speed_str}"

            # Label banner
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, max(th + 4, y1)), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, max(th, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 5. Top HUD
        hud_bg = annotated[0:40, 0:w].copy()
        cv2.rectangle(hud_bg, (0, 0), (w, 40), (20, 20, 20), -1)
        cv2.addWeighted(hud_bg, 0.75, annotated[0:40, 0:w], 0.25, 0, annotated[0:40, 0:w])

        hud_text = (
            f"Frame: {telemetry.frame_idx} | Time: {telemetry.timestamp_s:.1f}s | "
            f"Active Veh: {telemetry.active_track_count} | Crossings: {len(telemetry.crossings)} | "
            f"Stopped Veh: {len(telemetry.active_stopped_events)}"
        )
        cv2.putText(
            annotated,
            hud_text,
            (15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return annotated

    def _find_frame_image(self, image_dir: Path, f_idx: int) -> Optional[Path]:
        """Resolves frame image path supporting common naming formats."""
        for pattern in [f"image{f_idx:06d}.jpg", f"img{f_idx:05d}.jpg", f"img{f_idx:06d}.jpg", f"image{f_idx:05d}.jpg"]:
            candidate = image_dir / pattern
            if candidate.exists():
                return candidate
        return None

    def save_keyframe(
        self,
        image_path: Path,
        tracks: List[Track],
        telemetry: FrameTelemetry,
        output_path: Path,
        line_counts: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> None:
        """Annotates a single frame and saves as an image."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame = cv2.imread(str(image_path))
        if frame is not None:
            annotated = self.annotate_frame(frame, tracks, telemetry, line_counts)
            cv2.imwrite(str(output_path), annotated)

    def render_video(
        self,
        image_dir: Path,
        tracks_by_frame: Dict[int, List[Track]],
        telemetries: List[FrameTelemetry],
        output_path: Path,
        line_counts: Optional[Dict[str, Dict[str, int]]] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        """Renders and saves full annotated video to MP4."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_map = {t.frame_idx: t for t in telemetries}

        sorted_frames = sorted(tracks_by_frame.keys())
        if max_frames:
            sorted_frames = sorted_frames[:max_frames]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer: Optional[cv2.VideoWriter] = None

        for f_idx in sorted_frames:
            img_path = self._find_frame_image(image_dir, f_idx)
            if not img_path:
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_path), fourcc, self.config.fps, (w, h))

            t_data = telemetry_map.get(f_idx)
            if t_data is not None:
                frame = self.annotate_frame(frame, tracks_by_frame[f_idx], t_data, line_counts)

            writer.write(frame)

        if writer is not None:
            writer.release()
