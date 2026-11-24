"""
Deterministic Mock VLM Provider for TrafficLens-AI.

Enables fast, offline, and reliable unit and integration testing without
requiring GPU access, external networks, or third-party inference credentials.
"""

from __future__ import annotations
from typing import List, Optional
from backend.app.schemas.vlm import (
    EvidencePackageSchema,
    SceneInterpretationResponseSchema,
    EnvironmentalConditionsSchema,
    WeatherConditionEnum,
    IlluminationConditionEnum,
    RoadSurfaceConditionEnum,
    ConfidenceCategoryEnum,
    VisualObservationSchema,
    EventInterpretationSchema,
    TelemetryProvenanceSchema,
    EpistemicClaimSchema,
    ClaimCategoryEnum,
    ModelMetadataSchema,
    ProviderHealthSchema,
)
from backend.app.services.vlm.provider import VLMProvider


class MockVLMProvider:
    """
    In-memory deterministic mock satisfying the VLMProvider protocol.
    
    Dynamically generates fully grounded scene interpretations referencing the
    genuine keyframe IDs, event IDs, and telemetry summaries provided in the evidence package.
    """

    def __init__(
        self,
        model_name: str = "mock-qwen3-vl-2b",
        is_healthy: bool = True,
        forced_exception: Optional[Exception] = None,
    ):
        self._model_name = model_name
        self._is_healthy = is_healthy
        self._forced_exception = forced_exception

    @property
    def provider_type(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate_scene_understanding(
        self,
        evidence: EvidencePackageSchema,
    ) -> SceneInterpretationResponseSchema:
        """Generates deterministic, schema-compliant, and grounded scene understanding."""
        if self._forced_exception:
            raise self._forced_exception

        first_kf = evidence.keyframes[0] if evidence.keyframes else None
        first_kf_id = first_kf.keyframe_id if first_kf else None
        first_kf_ts = first_kf.timestamp_s if first_kf else 0.0

        visual_observations: List[VisualObservationSchema] = []
        if first_kf_id:
            visual_observations.append(
                VisualObservationSchema(
                    observation_id="obs_mock_01",
                    keyframe_id=first_kf_id,
                    frame_timestamp_s=first_kf_ts,
                    description="Elevated perspective showing multi-lane roadway with clear pavement markings.",
                    visual_cues=["pavement markings intact", "lane dividers visible"],
                )
            )

        event_interpretations: List[EventInterpretationSchema] = []
        for evt in evidence.events:
            event_interpretations.append(
                EventInterpretationSchema(
                    event_id=evt.event_id,
                    track_id=evt.track_id or 1,
                    start_timestamp_s=evt.timestamp_s,
                    end_timestamp_s=evt.details.get("end_timestamp_s") if evt.details else None,
                    cv_measured_duration_s=evt.duration_s or 5.0,
                    probable_cause="Stationary position consistent with traffic queue dissipation or signal waiting.",
                    contextual_explanation=(
                        f"Track {evt.track_id} remained stationary for {evt.duration_s or 0.0:.1f}s. "
                        "No hazard flashers, collision damage, or emergency responders are visible."
                    ),
                    confidence=ConfidenceCategoryEnum.SUPPORTED_INFERENCE,
                    supporting_observations=["obs_mock_01"] if first_kf_id else [],
                )
            )

        claims: List[EpistemicClaimSchema] = [
            EpistemicClaimSchema(
                claim_id="claim_mock_cv_01",
                category=ClaimCategoryEnum.CV_FACT,
                statement=(
                    f"Sequence exhibits {evidence.telemetry_summary.get('total_unique_tracks', 0)} tracked vehicle identities "
                    f"and {len(evidence.events)} stopped-vehicle events."
                ),
                keyframe_ids=[],
                frame_timestamps=[],
                associated_track_ids=[evt.track_id for evt in evidence.events if evt.track_id is not None],
                evidence_rationale="Extracted directly from CV tracking summary and spatial event monitors.",
                telemetry_provenance=TelemetryProvenanceSchema(
                    source_type="telemetry_summary",
                    field_path="total_unique_tracks",
                    numeric_value=float(evidence.telemetry_summary.get("total_unique_tracks", 0)),
                    subject="scene",
                    predicate="total_unique_tracks",
                    unit="tracks",
                ) if "total_unique_tracks" in evidence.telemetry_summary else None,
            ),
            EpistemicClaimSchema(
                claim_id="claim_mock_vis_01",
                category=ClaimCategoryEnum.VISUAL_OBSERVATION,
                statement="Dry asphalt road surface with daylight ambient illumination.",
                keyframe_ids=[first_kf_id] if first_kf_id else [],
                frame_timestamps=[first_kf_ts] if first_kf_ts is not None else [],
                associated_track_ids=[],
                evidence_rationale="Directly observed in baseline keyframe imagery.",
            ),
            EpistemicClaimSchema(
                claim_id="claim_mock_inf_01",
                category=ClaimCategoryEnum.INFERENCE,
                statement="Traffic flow conditions appear nominal without sustained multi-lane blockages.",
                keyframe_ids=[first_kf_id] if first_kf_id else [],
                frame_timestamps=[],
                associated_track_ids=[],
                evidence_rationale="Contextually supported by low stopped vehicle frequency and unoccluded lane paths.",
            ),
            EpistemicClaimSchema(
                claim_id="claim_mock_unver_01",
                category=ClaimCategoryEnum.UNVERIFIED,
                statement="Driver intent and internal mechanical operation cannot be determined from external camera view.",
                keyframe_ids=[],
                frame_timestamps=[],
                associated_track_ids=[],
                evidence_rationale="Outside camera field of view and resolution capability.",
            ),
        ]

        return SceneInterpretationResponseSchema(
            job_id=evidence.job_id,
            video_id=evidence.video_id,
            scene_summary=(
                f"Multi-lane roadway sequence '{evidence.sequence_name}' captured under clear daylight conditions. "
                f"Traffic flow is monitored across {len(evidence.keyframes)} keyframes with {len(evidence.events)} discrete events."
            ),
            environmental_conditions=EnvironmentalConditionsSchema(
                weather=WeatherConditionEnum.CLEAR,
                illumination=IlluminationConditionEnum.DAYLIGHT,
                road_surface=RoadSurfaceConditionEnum.DRY,
                visibility_hazards=[],
                confidence=ConfidenceCategoryEnum.DIRECT_OBSERVATION,
            ),
            roadway_description="Asphalt multi-lane highway/arterial road with defined longitudinal striping.",
            visual_observations=visual_observations,
            event_interpretations=event_interpretations,
            claims=claims,
            unverified_aspects=[
                "Signal phasing and control states outside visual perimeter.",
                "Driver cabin interior state.",
            ],
            model_metadata=ModelMetadataSchema(
                model_name=self._model_name,
                provider_type=self.provider_type,
                execution_time_ms=15,
                prompt_tokens=150,
                completion_tokens=300,
                quantization_precision=None,
            ),
        )

    async def health_check(self) -> ProviderHealthSchema:
        """Returns instantaneous healthy diagnostic state."""
        return ProviderHealthSchema(
            provider_type=self.provider_type,
            model_name=self._model_name,
            is_healthy=self._is_healthy,
            status_message="Mock provider operational." if self._is_healthy else "Mock provider unhealthy.",
            latency_ms=1,
        )
