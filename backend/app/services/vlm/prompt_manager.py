"""
Prompt Engineering and Epistemic Grounding Manager for TrafficLens-AI.

Constructs structured multimodal prompt payloads enforcing strict epistemic
boundaries, preventing numerical hallucination, and anchoring visual claims
to specific keyframe identifiers.
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from backend.app.schemas.vlm import EvidencePackageSchema, SceneInterpretationResponseSchema


class PromptManager:
    """
    Constructs prompts and message payloads for Vision-Language Models.
    
    Enforces strict epistemic constraints:
    - Never invent or estimate vehicle counts, speeds, or physical distances.
    - Classify every statement as cv_fact, visual_observation, inference, or unverified.
    - Anchor visual claims to explicit keyframe IDs.
    - Strictly format output as valid JSON.
    """

    SYSTEM_PROMPT = (
        "You are an expert multimodal traffic intelligence engine. You will be provided with "
        "deterministic computer vision (CV) measurements and a set of selected video keyframes.\n\n"
        "STRICT EPISTEMIC BOUNDARIES & RULES:\n"
        "1. DO NOT ESTIMATE OR INVENT vehicle counts, speeds, or physical distances. Numerical metrics "
        "are strictly owned by the CV pipeline.\n"
        "2. DO NOT ASSERT that a stopped vehicle is an accident or mechanical breakdown without visible "
        "supporting evidence (e.g. collision damage, hazard lights, pylons, or emergency responders). "
        "If the cause is unverified, explicitly classify it as unverified or queue waiting.\n"
        "3. ATTRIBUTION: Every visual observation MUST reference the specific 'keyframe_id' where "
        "it is directly visible.\n"
        "4. EPISTEMIC CATEGORIES: Classify every atomic claim into exactly one category:\n"
        "   - 'cv_fact': Statements directly citing CV telemetry, track IDs, durations, or line counts.\n"
        "   - 'visual_observation': Directly visible features in keyframes (e.g. wet road, brake lights).\n"
        "   - 'inference': Contextual hypotheses supported by both telemetry and visuals.\n"
        "   - 'unverified': Aspects that cannot be determined from the available camera view.\n"
        "5. OUTPUT FORMAT: Respond ONLY with a valid JSON object strictly matching the specified schema. "
        "Do not include markdown prose outside the JSON block."
    )

    @classmethod
    def build_user_prompt(cls, evidence: EvidencePackageSchema) -> str:
        """
        Builds the text prompt detailing CV facts, events, keyframe references,
        and the expected JSON schema.
        """
        keyframes_summary = []
        for kf in evidence.keyframes:
            ts_str = f"{kf.timestamp_s:.2f}s" if kf.timestamp_s is not None else "unknown timing"
            keyframes_summary.append({
                "keyframe_id": kf.keyframe_id,
                "frame_idx": kf.frame_idx,
                "timestamp_s": ts_str,
                "selection_reason": kf.selection_reason,
                "associated_events": kf.associated_event_ids,
            })

        events_summary = []
        for evt in evidence.events:
            ts_str = f"{evt.timestamp_s:.2f}s" if evt.timestamp_s is not None else "unknown"
            events_summary.append({
                "event_id": evt.event_id,
                "event_type": evt.event_type,
                "track_id": evt.track_id,
                "timestamp_s": ts_str,
                "duration_s": evt.duration_s,
                "roi_id": evt.roi_id,
                "details": evt.details,
            })

        user_content = {
            "task": "Perform qualitative multimodal scene understanding for this traffic sequence.",
            "sequence_info": {
                "sequence_name": evidence.sequence_name,
                "job_id": evidence.job_id,
                "video_id": evidence.video_id,
            },
            "cv_telemetry_summary": evidence.telemetry_summary,
            "cv_events": events_summary,
            "supplied_keyframes": keyframes_summary,
            "required_json_schema": {
                "scene_summary": "Comprehensive natural-language summary of roadway, weather, and traffic activity",
                "environmental_conditions": {
                    "weather": "clear | overcast | rain | fog | unknown",
                    "illumination": "daylight | twilight | night | unknown",
                    "road_surface": "dry | wet_reflective | snow_ice | unknown",
                    "visibility_hazards": ["list of hazards like glare, spray, or smoke"],
                    "confidence": "direct_observation | supported_inference | speculative",
                },
                "roadway_description": "Description of roadway geometry, visible lanes, dividers, or camera vantage",
                "visual_observations": [
                    {
                        "observation_id": "obs_1",
                        "keyframe_id": "keyframe_id providing evidence",
                        "frame_timestamp_s": 0.0,
                        "description": "Visual feature observed in this keyframe",
                        "visual_cues": ["cues like 'brake lights on'"],
                    }
                ],
                "event_interpretations": [
                    {
                        "event_id": "matching cv_event event_id",
                        "track_id": 1,
                        "start_timestamp_s": 0.0,
                        "cv_measured_duration_s": 0.0,
                        "probable_cause": "e.g. 'Waiting in queue for intersection clearance'",
                        "contextual_explanation": "Detailed explanation citing visual cues",
                        "confidence": "direct_observation | supported_inference | speculative",
                        "supporting_observations": ["obs_1"],
                    }
                ],
                "claims": [
                    {
                        "claim_id": "claim_1",
                        "category": "cv_fact | visual_observation | inference | unverified",
                        "statement": "Clear factual or inferential statement",
                        "keyframe_ids": ["keyframe_id if visual"],
                        "associated_track_ids": [1],
                        "evidence_rationale": "Why this statement is true based on CV or imagery",
                        "telemetry_provenance": {
                            "source_type": "telemetry_summary | event (required if category is cv_fact)",
                            "record_id": "event_id if source_type is event, else null",
                            "field_path": "e.g. 'duration_s', 'total_unique_tracks', 'line_counts.line_1'",
                            "numeric_value": 0.0,
                            "text_value": None,
                            "subject": "e.g. 'track_1', 'scene'",
                            "predicate": "e.g. 'stationary_duration', 'total_unique_tracks'",
                            "unit": "e.g. 'seconds', 'tracks', 'count'",
                        },
                    }
                ],
                "unverified_aspects": [
                    "Aspects that cannot be determined (e.g. traffic signal phase is out of camera frame)"
                ],
            },
        }

        return (
            "Analyze the following traffic sequence evidence and provided keyframe images.\n"
            f"{json.dumps(user_content, indent=2)}\n\n"
            "Produce the analysis strictly as a JSON object adhering to the schema above."
        )

    @classmethod
    def assemble_openai_messages(cls, evidence: EvidencePackageSchema) -> List[Dict[str, Any]]:
        """
        Assembles standard OpenAI-compatible multimodal chat completion messages:
        - System message enforcing epistemic rules.
        - User message containing text instructions and interleaved base64 image data URIs.
        """
        user_prompt_text = cls.build_user_prompt(evidence)

        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": user_prompt_text}
        ]

        # Interleave keyframe images with explicit metadata headers
        for kf in evidence.keyframes:
            user_content.append({
                "type": "text",
                "text": (
                    f"--- EVIDENCE KEYFRAME [keyframe_id: {kf.keyframe_id}] "
                    f"[frame_idx: {kf.frame_idx}] [reason: {kf.selection_reason}] ---"
                ),
            })
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{kf.content_type};base64,{kf.image_base64}",
                },
            })

        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @classmethod
    def assemble_gemini_parts(cls, evidence: EvidencePackageSchema) -> List[Dict[str, Any]]:
        """
        Assembles multimodal request parts for Google Gemini's native generateContent REST API.
        
        Formats:
        - Primary instructions part containing epistemic boundaries, CV facts, and output schema.
        - Interleaved text metadata labels and inlineData objects with base64 image data.
        """
        user_prompt_text = cls.build_user_prompt(evidence)
        combined_text = f"{cls.SYSTEM_PROMPT}\n\n{user_prompt_text}"

        parts: List[Dict[str, Any]] = [
            {"text": combined_text}
        ]

        # Interleave keyframe metadata labels and base64 inlineData parts
        for kf in evidence.keyframes:
            parts.append({
                "text": (
                    f"--- EVIDENCE KEYFRAME [keyframe_id: {kf.keyframe_id}] "
                    f"[frame_idx: {kf.frame_idx}] [reason: {kf.selection_reason}] ---"
                )
            })
            parts.append({
                "inlineData": {
                    "mimeType": kf.content_type or "image/jpeg",
                    "data": kf.image_base64,
                }
            })

        return parts

    @classmethod
    def get_gemini_response_schema(cls) -> Dict[str, Any]:
        """
        Generates a dereferenced OpenAPI-compliant JSON schema suitable for
        Gemini's generationConfig.responseSchema parameter.
        
        Inlines all Pydantic $defs, removes non-standard keywords, and uppercases types.
        """
        raw_schema = SceneInterpretationResponseSchema.model_json_schema()
        defs = raw_schema.get("$defs", {})

        def resolve(node: Any) -> Any:
            if isinstance(node, dict):
                if "$ref" in node:
                    ref_key = node["$ref"].split("/")[-1]
                    return resolve(defs.get(ref_key, {}))
                cleaned = {}
                for k, v in node.items():
                    if k in ("$defs", "title", "default", "prefixItems"):
                        continue
                    cleaned[k] = resolve(v)
                if "prefixItems" in node:
                    p_items = node["prefixItems"]
                    if p_items and isinstance(p_items, list):
                        cleaned["items"] = resolve(p_items[0])
                    cleaned["type"] = "ARRAY"
                if "type" in cleaned and isinstance(cleaned["type"], str):
                    cleaned["type"] = cleaned["type"].upper()
                return cleaned
            elif isinstance(node, list):
                return [resolve(x) for x in node]
            return node

        dereferenced = resolve(raw_schema)
        dereferenced.pop("$defs", None)
        return dereferenced
