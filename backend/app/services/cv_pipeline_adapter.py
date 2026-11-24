"""
Computer Vision pipeline adapter service.

Serves as the architectural seam decoupling the backend services and HTTP endpoints
from the underlying computer vision execution engine (experiments/src/...).
The backend interacts solely through structured schemas and this adapter boundary.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
import re
import threading
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from backend.app.core.logging import get_logger

logger = get_logger("trafficlens.adapter")

from experiments.src.detection.detector import VehicleDetector
from experiments.src.detection.config import DetectionConfig
from experiments.src.tracking.tracker import VehicleTracker, Track
from experiments.src.tracking.config import TrackingConfig
from experiments.src.analytics import (
    AnalyticsConfig,
    CountingLineConfig,
    ROIConfig,
    TrafficAnalyticsPipeline,
    SequenceAnalyticsReport,
)
from experiments.src.analytics.visualizer import AnalyticsVisualizer

from backend.app.schemas import (
    CountingLineConfigSchema,
    ROIConfigSchema,
    TrafficAnalyticsSummarySchema,
    LineFlowSchema,
    StoppedVehicleEventSchema,
    DetectionSummarySchema,
    TrackingSummarySchema,
)


class CVPipelineAdapter:
    """
    Adapter orchestrating execution of the offline or batch CV pipeline
    and converting native domain models into backend Pydantic contracts.
    """

    @staticmethod
    def convert_line_schema_to_config(schema: CountingLineConfigSchema) -> CountingLineConfig:
        """Converts API line schema to native CV config."""
        return CountingLineConfig(
            line_id=schema.line_id,
            name=schema.name,
            p1=schema.p1,
            p2=schema.p2,
            positive_direction_label=schema.positive_direction_label,
            negative_direction_label=schema.negative_direction_label,
            cooldown_frames=schema.cooldown_frames,
            min_track_age=schema.min_track_age,
        )

    @staticmethod
    def convert_roi_schema_to_config(schema: ROIConfigSchema) -> ROIConfig:
        """Converts API ROI schema to native CV config."""
        return ROIConfig(
            roi_id=schema.roi_id,
            name=schema.name,
            polygon=schema.polygon,
            description=schema.description,
        )

    @staticmethod
    def convert_analytics_report_to_schema(
        report: SequenceAnalyticsReport,
    ) -> TrafficAnalyticsSummarySchema:
        """Converts native SequenceAnalyticsReport to backend Pydantic contract."""
        flow_windows = [
            LineFlowSchema(
                line_id=f.line_id,
                window_start_s=f.window_start_s,
                window_end_s=f.window_end_s,
                total_count=f.total_count,
                flow_rate_vpm=f.flow_rate_vpm,
                direction_counts=dict(f.direction_counts),
                direction_flows_vpm=dict(f.direction_flows_vpm),
            )
            for f in report.flow_windows
        ]

        stopped_events = [
            StoppedVehicleEventSchema(
                track_id=e.track_id,
                start_frame=e.start_frame,
                start_timestamp_s=e.start_timestamp_s,
                end_frame=e.end_frame,
                end_timestamp_s=e.end_timestamp_s,
                duration_s=e.duration_s,
                location=e.location,
                roi_id=e.roi_id,
                status=e.status,
            )
            for e in report.stopped_vehicle_events
        ]

        return TrafficAnalyticsSummarySchema(
            sequence_name=report.sequence_name,
            total_frames=report.total_frames,
            duration_s=report.duration_s,
            fps=report.fps,
            total_unique_vehicles=report.total_unique_vehicles,
            line_counts=report.line_counts,
            flow_windows=flow_windows,
            stopped_vehicle_events=stopped_events,
            peak_occupancy_per_roi=report.peak_occupancy_per_roi,
            average_occupancy_per_roi=report.average_occupancy_per_roi,
            congestion_distribution_per_roi=report.congestion_distribution_per_roi,
        )

    @staticmethod
    def sanitize_artifact_slug(name: str) -> str:
        """
        Produces a pure ASCII filename slug safe for C++ OpenCV filesystem operations on all OS platforms.
        Replaces non-ASCII symbols (such as '…' or dashes) with ASCII equivalents and strips illegal characters.
        """
        s = name.replace("…", "...").replace("–", "-").replace("—", "-")
        s = re.sub(r"[^\w\-.]", "_", s, flags=re.ASCII)
        s = re.sub(r"_+", "_", s).strip("._")
        return s or "sequence"

    @classmethod
    def process_video_file(
        cls,
        video_path: Path,
        output_dir: Path,
        sequence_name: str,
        lines: Optional[List[CountingLineConfigSchema]] = None,
        rois: Optional[List[ROIConfigSchema]] = None,
        max_frames: Optional[int] = None,
        generate_annotated_video: bool = True,
        stop_event: Optional[threading.Event] = None,
    ) -> Tuple[SequenceAnalyticsReport, DetectionSummarySchema, TrackingSummarySchema, Dict[str, str], Dict[str, Any]]:
        """
        Executes end-to-end computer vision analysis on an arbitrary video file:
        1. Decodes frames via OpenCV VideoCapture.
        2. Detects vehicles using pretrained YOLO11n.
        3. Tracks vehicles with ByteTrack Kalman association.
        4. Computes spatiotemporal traffic telemetry (counting, flow, density, congestion, stopped events).
        5. Exports structured JSON, CSV, and optional annotated MP4 video with robust verification.
        
        Returns:
            (SequenceAnalyticsReport, DetectionSummarySchema, TrackingSummarySchema, artifacts_dict, annotation_diagnostics)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found at {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or np.isnan(fps):
            fps = 25.0

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540

        # Build native analytics configuration
        native_lines = [cls.convert_line_schema_to_config(l) for l in (lines or [])]
        native_rois = [cls.convert_roi_schema_to_config(r) for r in (rois or [])]

        analytics_config = AnalyticsConfig(
            fps=fps,
            frame_width=float(frame_width),
            frame_height=float(frame_height),
            lines=native_lines,
            rois=native_rois,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate a safe ASCII slug to avoid OpenCV C++ filename mangling on non-ASCII paths (e.g. Windows UTF-8 / ANSI)
        safe_slug = cls.sanitize_artifact_slug(sequence_name)
        annotated_video_path = output_dir / f"{safe_slug}_annotated.mp4"

        # Ensure dimensions are even (H.264 encoders strictly require even width & height)
        target_width = frame_width - (frame_width % 2)
        target_height = frame_height - (frame_height % 2)
        if (target_width, target_height) != (frame_width, frame_height):
            logger.info(
                f"Adjusted odd video dimensions for encoder compatibility: ({frame_width}, {frame_height}) -> ({target_width}, {target_height})"
            )

        video_writer: Optional[cv2.VideoWriter] = None
        selected_codec: Optional[str] = None
        video_writer_initialized = False
        annotation_init_error: Optional[str] = None

        if generate_annotated_video:
            # Codec candidate priority: browser-native H.264 ('avc1', 'H264', 'X264') then fallback 'mp4v'
            candidate_codecs = ("avc1", "H264", "X264", "mp4v")
            for codec in candidate_codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    vw = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (target_width, target_height))
                    if vw.isOpened():
                        video_writer = vw
                        selected_codec = codec
                        video_writer_initialized = True
                        logger.info(
                            f"Initialized VideoWriter with codec '{codec}' ({target_width}x{target_height} @ {fps:.2f}fps) at {annotated_video_path}"
                        )
                        break
                    else:
                        vw.release()
                except Exception as vw_exc:
                    logger.warning(f"VideoWriter attempt failed for codec '{codec}': {vw_exc}")

            if video_writer is None or not video_writer.isOpened():
                annotation_init_error = (
                    f"Failed to initialize VideoWriter with any candidate codec: {candidate_codecs}. "
                    "Skipping video annotation rendering to preserve core CV analytics."
                )
                logger.warning(annotation_init_error)
                video_writer = None

        # Instantiate pipeline components
        detector = VehicleDetector()
        tracker = VehicleTracker()
        analytics_pipeline = TrafficAnalyticsPipeline(analytics_config)
        visualizer = AnalyticsVisualizer(analytics_config)

        total_detections = 0
        conf_scores: List[float] = []
        class_counts: Dict[str, int] = {}
        frame_idx = 0
        frames_written = 0
        frame_write_errors: List[str] = []

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    logger.warning(
                        f"Inference loop received stop signal at frame {frame_idx}. Terminating early."
                    )
                    raise InterruptedError("Video processing interrupted by shutdown.")

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break

                # 1. Detection (conf=0.05 floor for ByteTrack two-stage matching)
                predictions, _ = detector.predict_image(frame, conf=0.05)
                total_detections += len(predictions)
                for p in predictions:
                    conf_scores.append(p.confidence)
                    coco_name = getattr(p, "raw_coco_class", "vehicle")
                    class_counts[coco_name] = class_counts.get(coco_name, 0) + 1

                # 2. Tracking
                tracks = tracker.update(predictions, frame_idx=frame_idx)

                # 3. Spatiotemporal Analytics
                timestamp_s = max(0.0, (frame_idx - 1) / fps)
                telemetry = analytics_pipeline.process_frame(tracks, frame_idx=frame_idx, timestamp_s=timestamp_s)

                # 4. Optional video annotation
                if video_writer is not None:
                    try:
                        annotated_frame = visualizer.annotate_frame(
                            frame,
                            tracks,
                            telemetry,
                            analytics_pipeline.counter.get_counts(),
                        )
                        if (annotated_frame.shape[1], annotated_frame.shape[0]) != (target_width, target_height):
                            annotated_frame = cv2.resize(annotated_frame, (target_width, target_height))
                        video_writer.write(annotated_frame)
                        frames_written += 1
                    except Exception as w_exc:
                        if len(frame_write_errors) < 5:
                            logger.warning(f"Error writing annotated frame {frame_idx}: {w_exc}")
                            frame_write_errors.append(f"Frame {frame_idx}: {str(w_exc)}")

        finally:
            cap.release()
            writer_finalized = False
            if video_writer is not None:
                try:
                    video_writer.release()
                    writer_finalized = True
                except Exception as rel_exc:
                    logger.error(f"Error releasing VideoWriter: {rel_exc}")

        # Verification and Artifact Registration
        verification_passed = False
        annotation_status = "UNAVAILABLE"
        annotation_error = None

        if not generate_annotated_video:
            annotation_status = "UNAVAILABLE"
            annotation_error = "Annotation generation disabled by configuration."
        elif video_writer is None:
            annotation_status = "FAILED"
            annotation_error = annotation_init_error or "VideoWriter initialization failed."
        else:
            # Check file existence; try fallback search if exact path missing
            if not annotated_video_path.exists():
                candidates = [f for f in output_dir.glob("*_annotated.mp4") if f.is_file()]
                if candidates:
                    try:
                        candidates[0].rename(annotated_video_path)
                        logger.info(f"Normalized mangled video filename {candidates[0].name} -> {annotated_video_path.name}")
                    except Exception as ren_e:
                        logger.warning(f"Could not rename candidate {candidates[0]}: {ren_e}")
                        annotated_video_path = candidates[0]

            if not annotated_video_path.exists():
                annotation_status = "FAILED"
                annotation_error = "Annotated video output file was not created on filesystem."
            else:
                fsize = annotated_video_path.stat().st_size
                if fsize == 0 or frames_written == 0:
                    annotation_status = "FAILED"
                    annotation_error = f"Annotated video output file is 0 bytes (frames written: {frames_written})."
                    try:
                        annotated_video_path.unlink()
                    except Exception:
                        pass
                else:
                    # Verify the generated video can be reopened and decoded
                    verify_cap = cv2.VideoCapture(str(annotated_video_path))
                    is_opened = verify_cap.isOpened()
                    can_read, _ = verify_cap.read() if is_opened else (False, None)
                    v_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_opened else 0
                    verify_cap.release()

                    if is_opened and can_read:
                        verification_passed = True
                        annotation_status = "COMPLETED"
                        logger.info(
                            f"Verified annotated video artifact: {annotated_video_path} "
                            f"({fsize} bytes, {v_frames} readable frames)."
                        )
                    else:
                        annotation_status = "FAILED"
                        annotation_error = "Generated video failed verification: cannot be decoded by VideoCapture."
                        logger.warning(f"Corrupt annotated video deleted: {annotated_video_path}")
                        try:
                            annotated_video_path.unlink()
                        except Exception:
                            pass

        # Build sequence analytics report
        duration_s = frame_idx / fps if frame_idx > 0 else 0.0
        flow_windows = analytics_pipeline.flow_calc.compute_sequence_tumbling_windows(duration_s)
        all_stopped = analytics_pipeline.stopped_detector.get_all_events()

        report = SequenceAnalyticsReport(
            sequence_name=sequence_name,
            total_frames=frame_idx,
            duration_s=duration_s,
            fps=fps,
            total_unique_vehicles=len(analytics_pipeline._unique_track_ids),
            line_counts=analytics_pipeline.counter.get_counts(),
            flow_windows=flow_windows,
            stopped_vehicle_events=all_stopped,
            peak_occupancy_per_roi=analytics_pipeline.density_estimator.get_peak_occupancy(),
            average_occupancy_per_roi=analytics_pipeline.density_estimator.get_average_occupancy(),
            congestion_distribution_per_roi=analytics_pipeline.congestion_classifier.get_distribution(),
        )

        # Export telemetry artifacts
        analytics_json_path = output_dir / f"{sequence_name}_analytics.json"
        analytics_pipeline.export_to_json(
            analytics_json_path,
            report,
            analytics_pipeline.frame_telemetries,
            include_frame_telemetries=True,
        )

        telemetry_csv_path = output_dir / f"{sequence_name}_telemetry.csv"
        analytics_pipeline.export_to_csv(telemetry_csv_path, analytics_pipeline.frame_telemetries)

        artifacts = {
            "analytics_json": str(analytics_json_path),
            "telemetry_csv": str(telemetry_csv_path),
        }
        if annotation_status == "COMPLETED" and verification_passed:
            artifacts["annotated_video"] = str(annotated_video_path)

        annotation_diagnostics = {
            "status": annotation_status,
            "selected_codec": selected_codec,
            "input_metadata": {
                "width": frame_width,
                "height": frame_height,
                "fps": fps,
                "total_frames_decoded": frame_idx,
                "container": video_path.suffix.lower(),
                "pixel_format": "BGR_UINT8",
            },
            "output_metadata": {
                "width": target_width,
                "height": target_height,
                "fps": fps,
                "path": str(annotated_video_path) if annotation_status == "COMPLETED" else None,
                "file_size_bytes": annotated_video_path.stat().st_size if (annotation_status == "COMPLETED" and annotated_video_path.exists()) else None,
            },
            "frames_written": frames_written,
            "writer_initialized": video_writer_initialized,
            "writer_finalized": writer_finalized,
            "verification_passed": verification_passed,
            "error_message": annotation_error,
        }

        # Save annotation diagnostics artifact
        diag_json_path = output_dir / f"{safe_slug}_annotation_diagnostics.json"
        try:
            diag_json_path.write_text(json.dumps(annotation_diagnostics, indent=2), encoding="utf-8")
            artifacts["annotation_diagnostics"] = str(diag_json_path)
        except Exception as d_err:
            logger.debug(f"Failed to write annotation diagnostics file: {d_err}")

        # Compute summary models
        mean_conf = float(np.mean(conf_scores)) if conf_scores else 0.0
        detection_summary = DetectionSummarySchema(
            total_detections=total_detections,
            mean_confidence=mean_conf,
            class_distribution=class_counts,
        )

        tracking_summary = TrackingSummarySchema(
            total_unique_tracks=len(analytics_pipeline._unique_track_ids),
        )

        return report, detection_summary, tracking_summary, artifacts, annotation_diagnostics
