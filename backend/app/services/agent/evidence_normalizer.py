"""
Evidence Normalization Module for TrafficLens-AI Incident Agent.

Harmonizes raw computer vision telemetry, tracking statistics, spatiotemporal
analytics, and optional multimodal VLM scene interpretations into a unified,
immutable NormalizedEvidenceContext.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class NormalizedStoppedEvent(BaseModel):
    """Normalized stopped vehicle event representation."""
    event_id: str
    track_id: int
    start_timestamp_s: float
    duration_s: float
    location: Optional[Tuple[float, float]] = None
    roi_id: Optional[str] = None
    status: str = "resolved"
    vlm_probable_cause: Optional[str] = None
    vlm_context: Optional[str] = None


class NormalizedEvidenceContext(BaseModel):
    """
    Immutable, consolidated evidence context passed to prompt formatting,
    reasoning agents, and grounding validators.
    """
    job_id: str
    video_id: str
    sequence_name: str
    duration_s: float
    total_frames: int
    fps: float

    # CV Telemetry metrics
    total_detections: int = 0
    total_unique_tracks: int = 0
    mean_detection_confidence: float = 0.0
    class_distribution: Dict[str, int] = Field(default_factory=dict)
    line_counts: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    peak_occupancy_per_roi: Dict[str, float] = Field(default_factory=dict)
    average_occupancy_per_roi: Dict[str, float] = Field(default_factory=dict)
    congestion_distribution_per_roi: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    stopped_events: List[NormalizedStoppedEvent] = Field(default_factory=list)

    # Multimodal VLM Context
    vlm_available: bool = False
    vlm_missing_reason: Optional[str] = None
    environmental_context: Optional[Dict[str, Any]] = None
    roadway_topology: Optional[str] = None
    visual_observations: List[Dict[str, Any]] = Field(default_factory=list)
    epistemic_claims: List[Dict[str, Any]] = Field(default_factory=list)
    unverified_aspects: List[str] = Field(default_factory=list)


class EvidenceNormalizer:
    """Constructs a normalized evidence context from CV job results and VLM interpretation."""

    @staticmethod
    def normalize(
        job_id: str,
        video_id: str,
        sequence_name: str,
        cv_results: Optional[Dict[str, Any]],
        scene_interpretation: Optional[Any] = None,
    ) -> NormalizedEvidenceContext:
        """
        Builds a normalized evidence bundle.
        
        Args:
            job_id: Processing job identifier.
            video_id: Video asset identifier.
            sequence_name: Video sequence name.
            cv_results: Raw CV results dict from job.config_metadata["results"] or artifacts.
            scene_interpretation: Optional SceneInterpretation model or dict.
            
        Returns:
            NormalizedEvidenceContext guaranteed to be non-null.
        """
        cv_results = cv_results or {}
        analytics = cv_results.get("analytics_summary") or {}
        detections = cv_results.get("detection_summary") or {}
        tracking = cv_results.get("tracking_summary") or {}

        # 1. Base sequence metrics
        duration_s = float(analytics.get("duration_s", 0.0))
        total_frames = int(analytics.get("total_frames", 0))
        fps = float(analytics.get("fps", 0.0))

        # 2. CV metrics
        total_detections = int(detections.get("total_detections", 0))
        mean_conf = float(detections.get("mean_confidence", 0.0))
        class_dist = dict(detections.get("class_distribution", {}))

        total_tracks = int(tracking.get("total_unique_tracks", 0))
        if total_tracks == 0 and "total_unique_vehicles" in analytics:
            total_tracks = int(analytics.get("total_unique_vehicles", 0))

        line_counts = dict(analytics.get("line_counts", {}))
        peak_occ = dict(analytics.get("peak_occupancy_per_roi", {}))
        avg_occ = dict(analytics.get("average_occupancy_per_roi", {}))
        congestion_dist = dict(analytics.get("congestion_distribution_per_roi", {}))

        # 3. Stopped vehicle events
        raw_stopped = analytics.get("stopped_vehicle_events") or []
        normalized_stopped: List[NormalizedStoppedEvent] = []
        for idx, evt in enumerate(raw_stopped):
            if isinstance(evt, dict):
                tid = evt.get("track_id", 0)
                start_ts = float(evt.get("start_timestamp_s", 0.0))
                dur = float(evt.get("duration_s", 0.0))
                loc = evt.get("location")
                loc_tuple = tuple(loc) if isinstance(loc, (list, tuple)) and len(loc) == 2 else None
                roi = evt.get("roi_id")
                st = evt.get("status", "resolved")
                eid = evt.get("event_id") or f"stop_{tid}_{start_ts:.2f}"
            else:
                # Handle StoppedVehicleEventSchema instance
                tid = getattr(evt, "track_id", 0)
                start_ts = float(getattr(evt, "start_timestamp_s", 0.0))
                dur = float(getattr(evt, "duration_s", 0.0))
                loc_val = getattr(evt, "location", None)
                loc_tuple = tuple(loc_val) if isinstance(loc_val, (list, tuple)) and len(loc_val) == 2 else None
                roi = getattr(evt, "roi_id", None)
                st = getattr(evt, "status", "resolved")
                eid = f"stop_{tid}_{start_ts:.2f}"

            normalized_stopped.append(
                NormalizedStoppedEvent(
                    event_id=eid,
                    track_id=tid,
                    start_timestamp_s=start_ts,
                    duration_s=dur,
                    location=loc_tuple,
                    roi_id=roi,
                    status=st,
                )
            )

        # 4. Integrate VLM scene interpretation if available and COMPLETED
        vlm_available = False
        vlm_missing_reason: Optional[str] = None
        env_context: Optional[Dict[str, Any]] = None
        roadway_desc: Optional[str] = None
        visual_obs: List[Dict[str, Any]] = []
        epistemic_claims: List[Dict[str, Any]] = []
        unverified_aspects: List[str] = []

        if scene_interpretation is not None:
            # Check status
            status_val = getattr(scene_interpretation, "status", None)
            if hasattr(status_val, "value"):
                status_str = status_val.value
            elif isinstance(status_val, str):
                status_str = status_val
            elif isinstance(scene_interpretation, dict):
                status_str = scene_interpretation.get("status", "")
            else:
                status_str = ""

            if status_str == "COMPLETED":
                vlm_available = True
                if isinstance(scene_interpretation, dict):
                    env_context = scene_interpretation.get("environmental_context")
                    scene_desc = scene_interpretation.get("scene_description") or {}
                    roadway_desc = scene_desc.get("roadway_topology")
                    visual_obs = scene_desc.get("features", [])
                    if isinstance(visual_obs, list) and visual_obs and isinstance(visual_obs[0], str):
                        visual_obs = [{"description": feat} for feat in visual_obs]
                    epistemic_claims = scene_interpretation.get("epistemic_claims") or []
                    unverified_aspects = scene_interpretation.get("unverified_aspects") or []
                    vlm_events = scene_interpretation.get("event_interpretations") or []
                else:
                    env_context = getattr(scene_interpretation, "environmental_context", None)
                    scene_desc = getattr(scene_interpretation, "scene_description", None) or {}
                    roadway_desc = scene_desc.get("roadway_topology") if isinstance(scene_desc, dict) else None
                    raw_features = scene_desc.get("features", []) if isinstance(scene_desc, dict) else []
                    visual_obs = [{"description": feat} for feat in raw_features] if isinstance(raw_features, list) else []
                    epistemic_claims = getattr(scene_interpretation, "epistemic_claims", None) or []
                    unverified_aspects = getattr(scene_interpretation, "unverified_aspects", None) or []
                    vlm_events = getattr(scene_interpretation, "event_interpretations", None) or []

                # Cross-reference VLM interpretations with stopped events
                if vlm_events and isinstance(vlm_events, list):
                    vlm_event_map = {
                        e.get("track_id") if isinstance(e, dict) else getattr(e, "track_id", None): e
                        for e in vlm_events
                    }
                    for n_evt in normalized_stopped:
                        match = vlm_event_map.get(n_evt.track_id)
                        if match:
                            if isinstance(match, dict):
                                n_evt.vlm_probable_cause = match.get("probable_cause")
                                n_evt.vlm_context = match.get("interpretation")
                            else:
                                n_evt.vlm_probable_cause = getattr(match, "probable_cause", None)
                                n_evt.vlm_context = getattr(match, "interpretation", None)
            else:
                vlm_missing_reason = f"Scene interpretation status was '{status_str}' rather than 'COMPLETED'."
        else:
            vlm_missing_reason = "No scene interpretation requested or available for this job."

        return NormalizedEvidenceContext(
            job_id=job_id,
            video_id=video_id,
            sequence_name=sequence_name,
            duration_s=duration_s,
            total_frames=total_frames,
            fps=fps,
            total_detections=total_detections,
            total_unique_tracks=total_tracks,
            mean_detection_confidence=mean_conf,
            class_distribution=class_dist,
            line_counts=line_counts,
            peak_occupancy_per_roi=peak_occ,
            average_occupancy_per_roi=avg_occ,
            congestion_distribution_per_roi=congestion_dist,
            stopped_events=normalized_stopped,
            vlm_available=vlm_available,
            vlm_missing_reason=vlm_missing_reason,
            environmental_context=env_context,
            roadway_topology=roadway_desc,
            visual_observations=visual_obs,
            epistemic_claims=epistemic_claims,
            unverified_aspects=unverified_aspects,
        )
