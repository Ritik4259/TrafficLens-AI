"""
Multimodal Evidence Packaging Service for TrafficLens-AI.

Integrates extracted keyframes with deterministic computer vision analytics,
producing compact, strongly-typed evidence packages for VLM consumption while
strictly isolating binary payloads and internal filesystem paths from public contracts.
"""

from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union

from backend.app.schemas.vlm import (
    EvidencePackageSchema,
    EvidencePackagePublicSchema,
    KeyframeDetailSchema,
    EventReferenceSchema,
    ExtractionConfigSchema,
    ExtractionDiagnosticsSchema,
)
from backend.app.schemas.telemetry import (
    TrafficAnalyticsSummarySchema,
    DetectionSummarySchema,
    TrackingSummarySchema,
    StoppedVehicleEventSchema,
)
from backend.app.services.vlm.keyframe_extractor import ExtractedFrame


class EvidencePackager:
    """
    Packages deterministic CV outputs and extracted keyframe imagery into a structured EvidencePackage.
    """

    @staticmethod
    def package_evidence(
        video_id: str,
        job_id: str,
        sequence_name: str,
        extracted_keyframes: List[ExtractedFrame],
        analytics_summary: Optional[Union[TrafficAnalyticsSummarySchema, Dict[str, Any]]] = None,
        detection_summary: Optional[Union[DetectionSummarySchema, Dict[str, Any]]] = None,
        tracking_summary: Optional[Union[TrackingSummarySchema, Dict[str, Any]]] = None,
        stopped_events: Optional[List[Any]] = None,
        config: Optional[ExtractionConfigSchema] = None,
        diagnostics: Optional[ExtractionDiagnosticsSchema] = None,
    ) -> EvidencePackageSchema:
        """
        Builds a complete EvidencePackageSchema combining keyframes, CV telemetry, and events.

        Args:
            video_id: Video asset identifier.
            job_id: Processing job identifier.
            sequence_name: Sequence or camera identifier.
            extracted_keyframes: List of ExtractedFrame objects produced by KeyframeExtractor.
            analytics_summary: Sequence analytics summary model or dict.
            detection_summary: Detection overview model or dict.
            tracking_summary: Tracking metrics model or dict.
            stopped_events: Optional list of stopped vehicle events if not extracted from analytics_summary.
            config: Extraction configuration applied.
            diagnostics: Extraction diagnostics.

        Returns:
            EvidencePackageSchema ready for model consumption or conversion to public contract.
        """
        package_id = f"evpkg_{job_id}_{len(extracted_keyframes)}k"

        # 1. Transform Extracted Frames into KeyframeDetailSchema
        keyframe_details: List[KeyframeDetailSchema] = []
        for kf in extracted_keyframes:
            kf_id = f"kf_{job_id}_{kf.frame_idx}"
            keyframe_details.append(
                KeyframeDetailSchema(
                    keyframe_id=kf_id,
                    source_video_id=video_id,
                    source_job_id=job_id,
                    frame_idx=kf.frame_idx,
                    requested_frame_idx=kf.requested_frame_idx,
                    is_verified=kf.is_verified,
                    timestamp_s=kf.timestamp_s,
                    selection_reason=kf.selection_reason,
                    associated_event_ids=kf.associated_event_ids,
                    width=kf.width,
                    height=kf.height,
                    image_base64=kf.image_base64,
                    content_type="image/jpeg",
                )
            )

        # 2. Extract Events into EventReferenceSchema
        event_references: List[EventReferenceSchema] = []
        raw_events = stopped_events or []
        if not raw_events and analytics_summary is not None:
            if isinstance(analytics_summary, dict):
                raw_events = analytics_summary.get("stopped_vehicle_events", [])
            else:
                raw_events = getattr(analytics_summary, "stopped_vehicle_events", [])

        for idx, evt in enumerate(raw_events):
            track_id = getattr(evt, "track_id", None)
            if track_id is None and isinstance(evt, dict):
                track_id = evt.get("track_id")

            start_ts = getattr(evt, "start_timestamp_s", None)
            if start_ts is None and isinstance(evt, dict):
                start_ts = evt.get("start_timestamp_s")

            end_ts = getattr(evt, "end_timestamp_s", None)
            if end_ts is None and isinstance(evt, dict):
                end_ts = evt.get("end_timestamp_s")

            duration = getattr(evt, "duration_s", None)
            if duration is None and isinstance(evt, dict):
                duration = evt.get("duration_s", 0.0)

            roi_id = getattr(evt, "roi_id", None)
            if roi_id is None and isinstance(evt, dict):
                roi_id = evt.get("roi_id")

            status = getattr(evt, "status", "active")
            if isinstance(evt, dict):
                status = evt.get("status", "active")

            location = getattr(evt, "location", (0.0, 0.0))
            if isinstance(evt, dict):
                location = evt.get("location", (0.0, 0.0))

            start_frame = getattr(evt, "start_frame", None)
            if start_frame is None and isinstance(evt, dict):
                start_frame = evt.get("start_frame")

            ts_str = f"{start_ts:.2f}" if (start_ts is not None and math.isfinite(start_ts)) else f"f{start_frame or idx}"
            event_id = f"stop_{track_id or idx}_{ts_str}"
            event_references.append(
                EventReferenceSchema(
                    event_id=event_id,
                    event_type="stopped_vehicle",
                    track_id=track_id,
                    timestamp_s=float(start_ts) if (start_ts is not None and math.isfinite(start_ts)) else None,
                    duration_s=float(duration) if (duration is not None and math.isfinite(duration)) else None,
                    roi_id=roi_id,
                    details={
                        "start_frame": int(start_frame) if start_frame is not None else None,
                        "end_timestamp_s": float(end_ts) if (end_ts is not None and math.isfinite(end_ts)) else None,
                        "status": status,
                        "location": list(location) if location else None,
                    },
                )
            )

        # 3. Assemble Compact Telemetry Summary (No filesystem paths!)
        telemetry_summary: Dict[str, Any] = {}

        # Detection Summary
        if detection_summary is not None:
            if isinstance(detection_summary, dict):
                telemetry_summary["total_detections"] = detection_summary.get("total_detections", 0)
                telemetry_summary["mean_confidence"] = detection_summary.get("mean_confidence", 0.0)
            else:
                telemetry_summary["total_detections"] = getattr(detection_summary, "total_detections", 0)
                telemetry_summary["mean_confidence"] = getattr(detection_summary, "mean_confidence", 0.0)

        # Tracking Summary
        if tracking_summary is not None:
            if isinstance(tracking_summary, dict):
                telemetry_summary["total_unique_tracks"] = tracking_summary.get("total_unique_tracks", 0)
            else:
                telemetry_summary["total_unique_tracks"] = getattr(tracking_summary, "total_unique_tracks", 0)

        # Analytics Summary
        if analytics_summary is not None:
            if isinstance(analytics_summary, dict):
                telemetry_summary["total_frames"] = analytics_summary.get("total_frames", 0)
                telemetry_summary["duration_s"] = analytics_summary.get("duration_s", 0.0)
                telemetry_summary["fps"] = analytics_summary.get("fps", 0.0)
                telemetry_summary["line_counts"] = analytics_summary.get("line_counts", {})
                telemetry_summary["peak_occupancy_per_roi"] = analytics_summary.get("peak_occupancy_per_roi", {})
                telemetry_summary["average_occupancy_per_roi"] = analytics_summary.get("average_occupancy_per_roi", {})
                telemetry_summary["congestion_distribution_per_roi"] = analytics_summary.get(
                    "congestion_distribution_per_roi", {}
                )
            else:
                telemetry_summary["total_frames"] = getattr(analytics_summary, "total_frames", 0)
                telemetry_summary["duration_s"] = getattr(analytics_summary, "duration_s", 0.0)
                telemetry_summary["fps"] = getattr(analytics_summary, "fps", 0.0)
                telemetry_summary["line_counts"] = getattr(analytics_summary, "line_counts", {})
                telemetry_summary["peak_occupancy_per_roi"] = getattr(analytics_summary, "peak_occupancy_per_roi", {})
                telemetry_summary["average_occupancy_per_roi"] = getattr(analytics_summary, "average_occupancy_per_roi", {})
                telemetry_summary["congestion_distribution_per_roi"] = getattr(
                    analytics_summary, "congestion_distribution_per_roi", {}
                )

        return EvidencePackageSchema(
            evidence_package_id=package_id,
            job_id=job_id,
            video_id=video_id,
            sequence_name=sequence_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            keyframes=keyframe_details,
            events=event_references,
            telemetry_summary=telemetry_summary,
            config=config or ExtractionConfigSchema(),
            diagnostics=diagnostics or ExtractionDiagnosticsSchema(),
        )
