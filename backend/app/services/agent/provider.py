"""
LLM Provider adapters for the TrafficLens-AI Incident Report Agent.

Provides abstract interface, native Google Gemini REST adapter, and a deterministic
mock provider for offline tests and local development.
"""

from __future__ import annotations
import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Protocol
import httpx
from pydantic import ValidationError

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.schemas.incident_report import (
    CongestionObservationSchema,
    EvidenceReferenceSchema,
    FollowUpRecommendationSchema,
    IncidentReportSchema,
    ObservedEventSchema,
    ReportConfidenceEnum,
    StoppedVehicleReportItemSchema,
    TrafficConditionsSchema,
    VisualObservationReportItemSchema,
)
from backend.app.services.agent.evidence_normalizer import NormalizedEvidenceContext
from backend.app.services.agent.prompts import (
    INCIDENT_AGENT_SYSTEM_PROMPT,
    format_evidence_prompt,
)

logger = get_logger("trafficlens.agent.provider")


class IncidentReportLLMProvider(Protocol):
    """Protocol contract for incident report LLM generation backends."""

    async def generate_report(
        self,
        evidence: NormalizedEvidenceContext,
    ) -> IncidentReportSchema:
        """Generates a structured IncidentReportSchema from evidence context."""
        ...


class GeminiIncidentReportProvider:
    """Google Gemini adapter for incident report generation using async httpx REST."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        api_base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_s: Optional[float] = None,
        settings: Optional[Settings] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        cfg = settings or get_settings()
        self._api_key = api_key or cfg.GEMINI_API_KEY
        raw_model = model_name or cfg.GEMINI_MODEL or "gemini-2.5-flash"
        self._model_name = raw_model[7:] if raw_model.startswith("models/") else raw_model
        self._api_base_url = (api_base_url or cfg.GEMINI_API_BASE_URL).rstrip("/")
        timeout_val = timeout_s if timeout_s is not None else getattr(cfg, "INCIDENT_AGENT_TIMEOUT", 30.0)
        self._timeout_s = timeout_val
        self._client = http_client

    async def generate_report(
        self,
        evidence: NormalizedEvidenceContext,
    ) -> IncidentReportSchema:
        """Invokes Gemini generateContent REST endpoint to produce an incident report."""
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured for incident report generation.")

        endpoint = f"{self._api_base_url}/models/{self._model_name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }

        user_content = format_evidence_prompt(evidence)
        payload = {
            "system_instruction": {
                "parts": [{"text": INCIDENT_AGENT_SYSTEM_PROMPT}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_content}],
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2,
                "maxOutputTokens": 8192,
            },
        }

        timeout_config = httpx.Timeout(
            timeout=self._timeout_s,
            connect=min(10.0, self._timeout_s),
            read=self._timeout_s,
            write=self._timeout_s,
            pool=self._timeout_s,
        )
        client = self._client or httpx.AsyncClient(timeout=timeout_config)
        try:
            response = await client.post(endpoint, headers=headers, json=payload)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Gemini API returned HTTP {response.status_code}: {response.text[:200]}"
                )

            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                raise RuntimeError("Gemini API returned 0 response candidates.")

            text_resp = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if not text_resp:
                raise RuntimeError("Gemini API returned empty text in candidate content.")

            clean_text = text_resp.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            parsed_json = json.loads(clean_text)
            if not parsed_json.get("report_id") or "<" in parsed_json.get("report_id", ""):
                parsed_json["report_id"] = f"rep_{uuid.uuid4().hex[:12]}"
            if not parsed_json.get("job_id") or "<" in parsed_json.get("job_id", ""):
                parsed_json["job_id"] = evidence.job_id
            if not parsed_json.get("video_id") or "<" in parsed_json.get("video_id", ""):
                parsed_json["video_id"] = evidence.video_id
            if not parsed_json.get("sequence_name") or "<" in parsed_json.get("sequence_name", ""):
                parsed_json["sequence_name"] = evidence.sequence_name

            return IncidentReportSchema(**parsed_json)
        finally:
            if self._client is None:
                await client.aclose()


class MockIncidentReportProvider:
    """
    Deterministic mock provider generating compliant, evidence-grounded
    incident reports for automated tests.
    """

    def __init__(
        self,
        should_fail: bool = False,
        invalid_json: bool = False,
        inject_unsupported_claim: bool = False,
        inject_visual_claim_without_vlm: bool = False,
        custom_report: Optional[IncidentReportSchema] = None,
    ):
        self.should_fail = should_fail
        self.invalid_json = invalid_json
        self.inject_unsupported_claim = inject_unsupported_claim
        self.inject_visual_claim_without_vlm = inject_visual_claim_without_vlm
        self.custom_report = custom_report

    async def generate_report(
        self,
        evidence: NormalizedEvidenceContext,
    ) -> IncidentReportSchema:
        """Returns deterministic IncidentReportSchema matching supplied evidence."""
        if self.should_fail:
            raise RuntimeError("Simulated mock LLM provider failure.")

        if self.invalid_json:
            raise ValidationError.from_exception_data(
                "Invalid mock output schema",
                [{"type": "missing", "loc": ("executive_summary",), "msg": "Field required"}],
            )

        if self.custom_report:
            return self.custom_report

        report_id = f"rep_{uuid.uuid4().hex[:12]}"
        stops = [
            StoppedVehicleReportItemSchema(
                event_id=s.event_id,
                track_id=s.track_id,
                start_timestamp_s=s.start_timestamp_s,
                duration_s=s.duration_s,
                location=s.location,
                roi_id=s.roi_id,
                probable_cause=s.vlm_probable_cause or ("Observed stationary duration" if evidence.vlm_available else None),
                contextual_explanation=s.vlm_context or f"Vehicle stationary for {s.duration_s:.1f}s.",
            )
            for s in evidence.stopped_events
        ]

        cong_obs = [
            CongestionObservationSchema(
                roi_id=roi_id,
                observed_level="moderate" if occ > 0.3 else "low",
                peak_occupancy=occ,
                average_occupancy=evidence.average_occupancy_per_roi.get(roi_id),
                is_heuristic_rule=True,
                description=f"ROI '{roi_id}' exhibited peak spatial occupancy of {occ:.1%}.",
            )
            for roi_id, occ in evidence.peak_occupancy_per_roi.items()
        ]

        vis_obs = []
        if evidence.vlm_available and evidence.visual_observations:
            for idx, vo in enumerate(evidence.visual_observations):
                vis_obs.append(
                    VisualObservationReportItemSchema(
                        observation_id=f"vis_{idx+1}",
                        description=vo.get("description", "Observed roadway feature"),
                        visual_cues=vo.get("visual_cues", []),
                    )
                )

        if self.inject_visual_claim_without_vlm and not evidence.vlm_available:
            vis_obs.append(
                VisualObservationReportItemSchema(
                    observation_id="vis_hallucinated",
                    description="Hallucinated wet reflective pavement visible on main road",
                )
            )

        summary_notes = [
            f"Traffic monitoring across sequence '{evidence.sequence_name}' recorded {evidence.total_detections} detections "
            f"and {evidence.total_unique_tracks} tracked vehicle instances over {evidence.duration_s:.1f} seconds."
        ]

        if stops:
            summary_notes.append(
                f"{len(stops)} stationary vehicle event(s) were recorded with durations up to "
                f"{max(s.duration_s for s in stops):.1f}s."
            )
        else:
            summary_notes.append("Continuous vehicle flow was observed with no persistent stationary events.")

        if self.inject_unsupported_claim:
            summary_notes.append("A severe vehicle collision occurred with vehicle moving at 75 km/h.")

        executive_summary = " ".join(summary_notes)

        confidence = (
            ReportConfidenceEnum.HIGH if evidence.vlm_available
            else ReportConfidenceEnum.MEDIUM
        )

        recommendations = [
            FollowUpRecommendationSchema(
                action="Review signal timing and detector thresholds",
                priority="low",
                rationale="Optimize traffic throughput based on observed queue patterns.",
                is_advisory=True,
            )
        ]

        if stops:
            recommendations.append(
                FollowUpRecommendationSchema(
                    action=f"Inspect zone for track ID {stops[0].track_id}",
                    priority="medium",
                    rationale=f"Vehicle remained stationary for {stops[0].duration_s:.1f}s.",
                    is_advisory=True,
                )
            )

        evidence_refs = [
            EvidenceReferenceSchema(
                source="cv_telemetry",
                metric_or_field="total_detections",
                asserted_value=evidence.total_detections,
                verified=True,
            ),
            EvidenceReferenceSchema(
                source="cv_telemetry",
                metric_or_field="total_unique_tracks",
                asserted_value=evidence.total_unique_tracks,
                verified=True,
            ),
        ]

        limitations = [
            "Camera view is uncalibrated. Vehicle speeds cannot be determined in km/h or mph."
        ]
        if not evidence.vlm_available:
            limitations.append(
                "Visual scene interpretation was unavailable; report is based solely on automated computer vision telemetry."
            )

        return IncidentReportSchema(
            report_id=report_id,
            job_id=evidence.job_id,
            video_id=evidence.video_id,
            sequence_name=evidence.sequence_name,
            executive_summary=executive_summary,
            traffic_conditions=TrafficConditionsSchema(
                sequence_name=evidence.sequence_name,
                duration_s=evidence.duration_s,
                total_detections=evidence.total_detections,
                total_tracked_instances=evidence.total_unique_tracks,
                overall_flow_rate_vpm=None,
                peak_occupancy_ratio=max(evidence.peak_occupancy_per_roi.values()) if evidence.peak_occupancy_per_roi else None,
                environmental_context=evidence.environmental_context,
                visual_context_available=evidence.vlm_available,
                visual_context_note=(
                    "VLM multimodal context available." if evidence.vlm_available
                    else "Generated from CV telemetry alone."
                ),
            ),
            observed_events=[
                ObservedEventSchema(
                    event_type="stopped_vehicle",
                    timestamp_s=s.start_timestamp_s,
                    description=f"Tracklet {s.track_id} stationary for {s.duration_s:.1f}s",
                    supporting_evidence_ref=s.event_id,
                )
                for s in stops
            ],
            congestion_observations=cong_obs,
            stopped_vehicles=stops,
            visual_observations=vis_obs,
            evidence_references=evidence_refs,
            confidence_classification=confidence,
            unverified_aspects=list(evidence.unverified_aspects),
            limitations=limitations,
            recommendations=recommendations,
        )
