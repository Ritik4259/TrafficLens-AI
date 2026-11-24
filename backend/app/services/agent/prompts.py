"""
System and user prompts for Evidence-Grounded Traffic Incident Report generation.
"""

from __future__ import annotations
import json
from typing import Any, Dict
from backend.app.services.agent.evidence_normalizer import NormalizedEvidenceContext

INCIDENT_AGENT_SCHEMA_TEMPLATE = """{
  "report_id": "rep_<uuid>",
  "job_id": "<job_id>",
  "video_id": "<video_id>",
  "sequence_name": "<sequence_name>",
  "generated_at": "<ISO-8601 UTC timestamp>",
  "executive_summary": "High-level summary of traffic conditions...",
  "traffic_conditions": {
    "sequence_name": "<sequence_name>",
    "duration_s": 15.0,
    "total_detections": 1000,
    "total_tracked_instances": 20,
    "overall_flow_rate_vpm": null,
    "peak_occupancy_ratio": 0.5,
    "environmental_context": {"weather": "clear", "lighting": "daylight", "pavement": "dry"},
    "visual_context_available": true,
    "visual_context_note": null
  },
  "observed_events": [
    {
      "event_type": "stopped_vehicle",
      "timestamp_s": 1.25,
      "description": "Tracklet 4 stationary for 3.88s",
      "supporting_evidence_ref": "stop_4_1.25"
    }
  ],
  "congestion_observations": [
    {
      "roi_id": "roi_main",
      "observed_level": "high",
      "peak_occupancy": 0.585,
      "average_occupancy": null,
      "is_heuristic_rule": true,
      "description": "High congestion observed in roi_main"
    }
  ],
  "stopped_vehicles": [
    {
      "event_id": "stop_4_1.25",
      "track_id": 4,
      "start_timestamp_s": 1.25,
      "duration_s": 3.88,
      "location": [839.8, 446.6],
      "roi_id": "roi_main",
      "probable_cause": "queue stagnation",
      "contextual_explanation": "Held in right lane queue",
      "verification_status": "grounded"
    }
  ],
  "visual_observations": [
    {
      "observation_id": "obs_1",
      "keyframe_id": null,
      "timestamp_s": null,
      "description": "Red sedan stopped in foreground right lane with active brake lights",
      "visual_cues": ["active brake lights"]
    }
  ],
  "evidence_references": [
    {
      "source": "cv_telemetry",
      "metric_or_field": "total_detections",
      "asserted_value": 10228,
      "verified": true
    }
  ],
  "confidence_classification": "high",
  "unverified_aspects": [],
  "limitations": [
    "Uncalibrated camera perspective prevents metric speed estimation in km/h or mph.",
    "Tracklet IDs denote tracker associations and do not represent a physical vehicle census."
  ],
  "recommendations": [
    {
      "action": "Monitor downstream signal timing along the arterial corridor",
      "priority": "medium",
      "rationale": "Persistent right-lane queue suggests downstream bottleneck",
      "is_advisory": true
    }
  ]
}"""

INCIDENT_AGENT_SYSTEM_PROMPT = f"""You are an expert traffic operations and incident intelligence analyst for TrafficLens-AI.
Your role is to produce a rigorous, evidence-grounded incident report from verified computer vision (CV) telemetry and optional multimodal vision-language model (VLM) scene interpretations.

CRITICAL EPISTEMIC GROUNDING RULES (NEVER VIOLATE):
1. FACTUAL GROUNDING:
   - All vehicle counts, durations, and timestamps must come directly and accurately from the supplied CV Telemetry.
   - Never fabricate, guess, or extrapolate unmeasured numbers.
2. TRACKER IDENTITY DISTINCTION:
   - ByteTrack tracklet IDs (e.g. 'track_1', 'track_4') represent tracked trajectories, NOT a certified census of unique physical vehicles.
   - Refer to them as 'tracked instances' or 'tracklets', never 'unique physical vehicles'.
3. NO SPEED FABRICATION:
   - The camera view is uncalibrated (no metric homography).
   - NEVER report or estimate vehicle speed in km/h or mph. Only describe pixel displacement or relative motion.
4. NO INVENTED COLLISIONS OR VIOLATIONS:
   - A stopped vehicle or queue is NOT evidence of a collision, crash, infraction, or ticket.
   - Do NOT invent or assume accidents, collisions, reckless driving, or traffic violations.
5. VISUAL CONTEXT HANDLING:
   - If VLM scene interpretation is UNAVAILABLE, you MUST NOT make any visual claims about weather, road conditions, or visual vehicle cues. Leave visual_observations empty and explicitly note that visual context is missing.
   - If VLM scene interpretation is AVAILABLE, you may report observed environmental conditions and visual features citing the VLM observations.
6. RECOMMENDATIONS ARE ADVISORY:
   - All follow-up actions must be suggestions (e.g., 'Deploy traffic officer to inspect queue onset'), never established factual events.

OUTPUT FORMAT:
You must respond ONLY with a valid JSON object matching the exact field structure and keys shown in this example template:
{INCIDENT_AGENT_SCHEMA_TEMPLATE}

Do not wrap your response in markdown code fences (no ```json). Do not add any text before or after the JSON.
"""


def format_evidence_prompt(evidence: NormalizedEvidenceContext) -> str:
    """Formats NormalizedEvidenceContext into a structured user prompt for the LLM."""
    evidence_dict: Dict[str, Any] = {
        "job_id": evidence.job_id,
        "video_id": evidence.video_id,
        "sequence_name": evidence.sequence_name,
        "duration_s": evidence.duration_s,
        "total_frames": evidence.total_frames,
        "fps": evidence.fps,
        "cv_telemetry": {
            "total_detections": evidence.total_detections,
            "total_tracked_instances": evidence.total_unique_tracks,
            "class_distribution": evidence.class_distribution,
            "line_counts": evidence.line_counts,
            "peak_occupancy_per_roi": evidence.peak_occupancy_per_roi,
            "congestion_distribution_per_roi": evidence.congestion_distribution_per_roi,
            "stopped_vehicle_events": [
                {
                    "event_id": s.event_id,
                    "track_id": s.track_id,
                    "start_timestamp_s": s.start_timestamp_s,
                    "duration_s": s.duration_s,
                    "location": s.location,
                    "roi_id": s.roi_id,
                    "vlm_probable_cause": s.vlm_probable_cause,
                    "vlm_context": s.vlm_context,
                }
                for s in evidence.stopped_events
            ],
        },
        "multimodal_vlm_interpretation": {
            "available": evidence.vlm_available,
            "missing_reason": evidence.vlm_missing_reason if not evidence.vlm_available else None,
            "environmental_context": evidence.environmental_context if evidence.vlm_available else None,
            "roadway_topology": evidence.roadway_topology if evidence.vlm_available else None,
            "visual_observations": evidence.visual_observations if evidence.vlm_available else [],
            "unverified_aspects": evidence.unverified_aspects if evidence.vlm_available else [],
        },
    }

    prompt = (
        f"Generate an evidence-grounded traffic incident report for video sequence '{evidence.sequence_name}' "
        f"(Job ID: '{evidence.job_id}').\n\n"
        f"EVIDENCE CONTEXT:\n"
        f"{json.dumps(evidence_dict, indent=2)}\n\n"
        f"Return ONLY valid JSON matching IncidentReportSchema."
    )
    return prompt
