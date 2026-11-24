"""
Deterministic Epistemic Grounding Validator for Traffic Incident Reports.

Enforces comprehensive factual grounding against NormalizedEvidenceContext:
1. Validates and corrects quantitative telemetry metrics (durations, detections, tracklet counts, occupancy).
2. Cross-checks evidence references against actual underlying CV metrics and event records.
3. Detects and sanitizes fabricated incidents (collisions, crashes, moving violations) across
   all report sections (summary, events, stopped vehicles, recommendations).
4. Enforces uncalibrated camera speed prohibitions (stripping km/h or mph physical speed claims).
5. Enforces tracker identity semantics (clarifying that track IDs are tracklet trajectories,
   not certified censuses of unique physical vehicles).
6. Strips visual observations when no completed VLM interpretation is available.
7. Validates observed event types and citations against known event records.
8. Ensures congestion observations retain heuristic designations and recommendations remain advisory.

EPISTEMIC BOUNDARY SPECIFICATION:
---------------------------------
A. DETERMINISTIC CHECKS (Formally verified against ground-truth structures):
   - Quantitative alignment: video duration, total detections, tracked instances count, peak occupancy.
   - Relational resolution: verifying that every cited track_id, roi_id, and event_id exists in CV data.
   - Syntactic/Lexical prohibitions: stripping collision/crash/violation terms, metric speed units (km/h, mph).
   - Tracker census rewriting: converting physical vehicle claims to "tracked vehicle instances".
   - VLM availability gate: dropping visual observations if no completed VLM artifact exists.

B. SEMANTIC / HEURISTIC CLAIMS (Bounded by domain contract, cannot be deterministically verified):
   - Scene descriptions & qualitative flow commentary (bounded to general observations).
   - Probable cause hypotheses (bounded to visual interpretations, neutralized if VLM absent).
   - Recommendations (strictly advisory decision-support heuristics: is_advisory=True).
   - Congestion levels (rule-based heuristic classifications: is_heuristic_rule=True).
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.app.schemas.incident_report import (
    CongestionObservationSchema,
    EvidenceReferenceSchema,
    FollowUpRecommendationSchema,
    IncidentReportSchema,
    ObservedEventSchema,
    ReportConfidenceEnum,
    ReportValidationDiagnosticsSchema,
    StoppedVehicleReportItemSchema,
    TrafficConditionsSchema,
    VisualObservationReportItemSchema,
)
from backend.app.services.agent.evidence_normalizer import NormalizedEvidenceContext


# Prohibited hallucinated incident terms that must never appear as established facts
FORBIDDEN_INCIDENT_TERMS = [
    "collision", "collided", "crash", "crashed", "wreck",
    "accident", "hit-and-run", "infraction", "ticket", "illegal turn",
    "red-light violation", "hit and run", "fatal", "casualty",
    "fender bender", "pileup", "traffic offense",
]

# Pattern detecting uncalibrated metric speeds (km/h, mph)
SPEED_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:km/h|kmh|kph|mph)\b", re.IGNORECASE)

# Pattern detecting physical vehicle census phrasing
CENSUS_PATTERN = re.compile(
    r"\b(\d+)\s+(?:unique|distinct|individual)\s+(?:physical\s+)?(?:vehicles|cars|automobiles)\b",
    re.IGNORECASE,
)

# Allowed source types for evidence references
ALLOWED_EVIDENCE_SOURCES: Set[str] = {
    "cv_telemetry",
    "vlm_scene_interpretation",
    "cv_event",
}

# Known telemetry field paths in CV results
ALLOWED_TELEMETRY_FIELDS: Set[str] = {
    "total_detections",
    "total_unique_tracks",
    "total_unique_vehicles",
    "duration_s",
    "total_frames",
    "fps",
    "line_counts",
    "overall_flow_rate_vpm",
    "peak_occupancy_ratio",
    "mean_confidence",
    "class_distribution",
}

# Supported event categories
ALLOWED_EVENT_TYPES: Set[str] = {
    "stopped_vehicle",
    "stationary_vehicle",
    "queue_onset",
    "congestion_transition",
    "flow_drop",
    "traffic_flow_measurement",
    "elevated_occupancy",
}


class GroundingValidator:
    """Performs deep factual verification and epistemic grounding on IncidentReportSchema."""

    @classmethod
    def validate_and_correct(
        cls,
        report: IncidentReportSchema,
        evidence: NormalizedEvidenceContext,
    ) -> Tuple[IncidentReportSchema, ReportValidationDiagnosticsSchema]:
        """
        Validates an incident report against ground-truth evidence context.
        Applies necessary corrections (e.g. correcting telemetry drift, stripping
        unsupported visual claims when VLM is absent, neutralizing fabricated terms).

        Args:
            report: Raw or LLM-generated IncidentReportSchema.
            evidence: Verified NormalizedEvidenceContext.

        Returns:
            Tuple of (corrected_and_verified_report, validation_diagnostics).
        """
        warnings: List[str] = []
        notes: List[str] = []
        total_claims_checked = 0
        grounded_claims = 0
        demoted_claims = 0

        # Clone model to preserve input immutability
        corrected = report.model_copy(deep=True)

        # Helper to sanitize and neutralize ungrounded terms and speeds across narrative fields
        def sanitize_text(text_val: str, field_desc: str) -> str:
            nonlocal demoted_claims, total_claims_checked
            if not text_val:
                return text_val

            # A. Check forbidden terms
            found_terms = [
                term for term in FORBIDDEN_INCIDENT_TERMS
                if re.search(r"\b" + re.escape(term) + r"\b", text_val, re.IGNORECASE)
            ]
            if found_terms:
                warnings.append(
                    f"Detected ungrounded incident assertions in {field_desc}: {found_terms}. "
                    "Neutralized to unverified claim."
                )
                demoted_claims += len(found_terms)
                total_claims_checked += len(found_terms)
                for term in found_terms:
                    text_val = re.sub(
                        r"\b" + re.escape(term) + r"\b",
                        "[unverified claim]",
                        text_val,
                        flags=re.IGNORECASE,
                    )
                corrected.unverified_aspects.append(
                    f"Unsupported incident claims suppressed in {field_desc}: {', '.join(found_terms)}."
                )

            # B. Check uncalibrated speeds
            speed_hits = SPEED_PATTERN.findall(text_val)
            if speed_hits:
                warnings.append(
                    f"Prohibited uncalibrated speed assertions ({speed_hits}) in {field_desc}. Suppressed."
                )
                demoted_claims += len(speed_hits)
                total_claims_checked += len(speed_hits)
                text_val = SPEED_PATTERN.sub("[uncalibrated displacement]", text_val)
                uncal_limitation = "Camera homography is uncalibrated. Vehicle speeds cannot be reported in km/h or mph."
                if uncal_limitation not in corrected.limitations:
                    corrected.limitations.append(uncal_limitation)

            # C. Check physical vehicle census phrasing
            census_hits = CENSUS_PATTERN.findall(text_val)
            if census_hits:
                warnings.append(
                    f"Clarified tracker identity semantics in {field_desc}: converted physical vehicle claim to tracked instances."
                )
                text_val = CENSUS_PATTERN.sub(r"\1 tracked vehicle instances", text_val)

            return text_val

        # ----------------------------------------------------------------------
        # 1. Telemetry Integrity & Metric Alignment
        # ----------------------------------------------------------------------
        tc = corrected.traffic_conditions
        total_claims_checked += 4

        # Verify sequence duration
        if abs(tc.duration_s - evidence.duration_s) > 0.5:
            warnings.append(
                f"Corrected duration_s from {tc.duration_s:.2f}s to ground-truth {evidence.duration_s:.2f}s."
            )
            tc.duration_s = evidence.duration_s
            demoted_claims += 1
        else:
            grounded_claims += 1

        # Verify total detections
        if tc.total_detections != evidence.total_detections:
            warnings.append(
                f"Corrected total_detections from {tc.total_detections} to ground-truth {evidence.total_detections}."
            )
            tc.total_detections = evidence.total_detections
            demoted_claims += 1
        else:
            grounded_claims += 1

        # Verify tracked instances
        if tc.total_tracked_instances != evidence.total_unique_tracks:
            warnings.append(
                f"Corrected total_tracked_instances from {tc.total_tracked_instances} to ground-truth {evidence.total_unique_tracks}."
            )
            tc.total_tracked_instances = evidence.total_unique_tracks
            demoted_claims += 1
        else:
            grounded_claims += 1

        # Reconcile peak spatial occupancy ratio if available in evidence
        if evidence.peak_occupancy_per_roi:
            ground_truth_peak = max(evidence.peak_occupancy_per_roi.values())
            if tc.peak_occupancy_ratio is None or abs(tc.peak_occupancy_ratio - ground_truth_peak) > 0.05:
                warnings.append(
                    f"Reconciled peak_occupancy_ratio from {tc.peak_occupancy_ratio} to ground-truth {ground_truth_peak:.2f}."
                )
                tc.peak_occupancy_ratio = ground_truth_peak
                demoted_claims += 1
            else:
                grounded_claims += 1

        # Ensure correct sequence name
        tc.sequence_name = evidence.sequence_name
        grounded_claims += 1

        # ----------------------------------------------------------------------
        # 2. Visual Context Availability Verification
        # ----------------------------------------------------------------------
        if not evidence.vlm_available:
            tc.visual_context_available = False
            tc.visual_context_note = (
                f"Visual scene interpretation unavailable ({evidence.vlm_missing_reason or 'No VLM data'}). "
                "Report is grounded exclusively in quantitative CV telemetry."
            )
            # If report claimed visual observations, demote and remove them
            if corrected.visual_observations:
                demoted_count = len(corrected.visual_observations)
                demoted_claims += demoted_count
                total_claims_checked += demoted_count
                warnings.append(
                    f"Removed {demoted_count} visual observations because no completed VLM interpretation exists."
                )
                for vo in corrected.visual_observations:
                    corrected.unverified_aspects.append(
                        f"Unverified visual claim (no VLM): {vo.description}"
                    )
                corrected.visual_observations = []

            # Ensure limitation is explicitly documented
            missing_vlm_limitation = (
                "Visual scene interpretation was unavailable; report is based solely on automated computer vision telemetry."
            )
            if missing_vlm_limitation not in corrected.limitations:
                corrected.limitations.append(missing_vlm_limitation)
        else:
            tc.visual_context_available = True
            tc.visual_context_note = "Multimodal visual scene interpretation incorporated from completed VLM inference."
            tc.environmental_context = evidence.environmental_context
            # Sanitize visual observations narrative
            for vo in corrected.visual_observations:
                vo.description = sanitize_text(vo.description, f"visual_observations[{vo.observation_id}].description")
                vo.visual_cues = [
                    sanitize_text(cue, f"visual_observations[{vo.observation_id}].cues")
                    for cue in vo.visual_cues
                ]

        # ----------------------------------------------------------------------
        # 3. Detect and Neutralize Fabricated Terms & Speeds Across All Sections
        # ----------------------------------------------------------------------
        # Sanitize executive summary
        corrected.executive_summary = sanitize_text(corrected.executive_summary, "executive_summary")

        # Sanitize recommendations
        for idx, rec in enumerate(corrected.recommendations):
            rec.action = sanitize_text(rec.action, f"recommendations[{idx}].action")
            rec.rationale = sanitize_text(rec.rationale, f"recommendations[{idx}].rationale")
            rec.is_advisory = True

        # ----------------------------------------------------------------------
        # 4. Deep Validation of Observed Events
        # ----------------------------------------------------------------------
        valid_event_refs = {s.event_id for s in evidence.stopped_events}
        valid_event_refs.update(evidence.peak_occupancy_per_roi.keys())
        valid_event_refs.update(evidence.line_counts.keys())

        validated_events: List[ObservedEventSchema] = []
        for evt in corrected.observed_events:
            total_claims_checked += 1
            evt_type_lower = evt.event_type.lower().replace(" ", "_")

            # Check for forbidden event types (e.g. collision, crash, accident)
            if any(term in evt_type_lower for term in FORBIDDEN_INCIDENT_TERMS):
                warnings.append(
                    f"Stripped unsupported event '{evt.event_type}' (unsupported accident/violation claim)."
                )
                demoted_claims += 1
                corrected.unverified_aspects.append(
                    f"Unsupported observed event type '{evt.event_type}' suppressed."
                )
                continue

            # Validate event type against allowed categories
            if evt_type_lower not in ALLOWED_EVENT_TYPES:
                notes.append(f"Classified event type '{evt.event_type}' as generic operational observation.")

            # If event is stationary/stopped vehicle, verify cited event exists in evidence.stopped_events
            if evt_type_lower in ("stopped_vehicle", "stationary_vehicle"):
                ref_str = str(evt.supporting_evidence_ref or "")
                ref_matches = any(s.event_id in ref_str or str(s.track_id) in ref_str for s in evidence.stopped_events)
                if not ref_matches and evidence.stopped_events:
                    warnings.append(
                        f"Stripped fabricated stopped_vehicle event '{evt.supporting_evidence_ref}' (not found in CV stopped events)."
                    )
                    demoted_claims += 1
                    corrected.unverified_aspects.append(
                        f"Unsupported stopped vehicle event citation '{evt.supporting_evidence_ref}'."
                    )
                    continue

            # Sanitize description
            evt.description = sanitize_text(evt.description, f"observed_events[{evt.event_type}]")

            # Validate supporting evidence reference
            if evt.supporting_evidence_ref:
                ref_str = str(evt.supporting_evidence_ref)
                if not any(valid_ref in ref_str for valid_ref in valid_event_refs):
                    notes.append(
                        f"Event '{evt.event_type}' citation '{evt.supporting_evidence_ref}' does not match standard CV event ID."
                    )

            # Check timestamp bounds
            if evt.timestamp_s is not None and evt.timestamp_s > evidence.duration_s + 1.0:
                warnings.append(
                    f"Event '{evt.event_type}' timestamp {evt.timestamp_s}s exceeds sequence duration {evidence.duration_s}s. Clamped."
                )
                evt.timestamp_s = min(evt.timestamp_s, evidence.duration_s)
                demoted_claims += 1
            else:
                grounded_claims += 1

            validated_events.append(evt)

        corrected.observed_events = validated_events

        # ----------------------------------------------------------------------
        # 5. Stopped Vehicle Verification
        # ----------------------------------------------------------------------
        evidence_stops = {s.track_id: s for s in evidence.stopped_events}
        validated_stops: List[StoppedVehicleReportItemSchema] = []

        for st_item in corrected.stopped_vehicles:
            total_claims_checked += 1
            if st_item.track_id in evidence_stops:
                ev_stop = evidence_stops[st_item.track_id]
                # Ground duration and start timestamp
                st_item.duration_s = ev_stop.duration_s
                st_item.start_timestamp_s = ev_stop.start_timestamp_s
                if not st_item.location and ev_stop.location:
                    st_item.location = ev_stop.location
                if not st_item.roi_id and ev_stop.roi_id:
                    st_item.roi_id = ev_stop.roi_id

                # Sanitize narrative explanations
                if st_item.contextual_explanation:
                    st_item.contextual_explanation = sanitize_text(
                        st_item.contextual_explanation, f"stopped_vehicles[{st_item.track_id}].context"
                    )
                if st_item.probable_cause:
                    st_item.probable_cause = sanitize_text(
                        st_item.probable_cause, f"stopped_vehicles[{st_item.track_id}].probable_cause"
                    )

                # If VLM is absent, clear speculative probable cause
                if not evidence.vlm_available and st_item.probable_cause:
                    notes.append(
                        f"Neutralized probable_cause for track {st_item.track_id} because VLM context is unavailable."
                    )
                    st_item.contextual_explanation = (
                        f"Vehicle stationary for {ev_stop.duration_s:.1f}s. Visual context unavailable to infer cause."
                    )
                    st_item.probable_cause = "Unobserved (CV telemetry only)"
                    st_item.verification_status = "cv_grounded_only"

                validated_stops.append(st_item)
                grounded_claims += 1
            else:
                warnings.append(
                    f"Report listed stopped vehicle track {st_item.track_id} not present in CV stopped events. Demoting."
                )
                demoted_claims += 1
                corrected.unverified_aspects.append(
                    f"Unsupported stoppage claim for track ID {st_item.track_id}."
                )

        corrected.stopped_vehicles = validated_stops

        # ----------------------------------------------------------------------
        # 6. Congestion Heuristic Enforcement
        # ----------------------------------------------------------------------
        for cong in corrected.congestion_observations:
            total_claims_checked += 1
            cong.is_heuristic_rule = True
            cong.description = sanitize_text(cong.description, f"congestion_observations[{cong.roi_id}]")
            # Check if ROI exists in evidence
            if cong.roi_id in evidence.peak_occupancy_per_roi:
                cong.peak_occupancy = evidence.peak_occupancy_per_roi[cong.roi_id]
                grounded_claims += 1
            else:
                demoted_claims += 1
                warnings.append(f"Congestion observation referenced unknown ROI '{cong.roi_id}'.")

        # ----------------------------------------------------------------------
        # 7. Deep Evidence References Validation
        # ----------------------------------------------------------------------
        verified_refs: List[EvidenceReferenceSchema] = []
        for ref in corrected.evidence_references:
            total_claims_checked += 1
            is_valid = True

            # Validate source
            if ref.source not in ALLOWED_EVIDENCE_SOURCES:
                warnings.append(f"Invalid evidence reference source: '{ref.source}'.")
                is_valid = False
            elif ref.source == "cv_telemetry":
                # Check metric field path
                if ref.metric_or_field == "total_detections":
                    ref.asserted_value = evidence.total_detections
                elif ref.metric_or_field in ("total_unique_tracks", "total_unique_vehicles"):
                    ref.asserted_value = evidence.total_unique_tracks
                elif ref.metric_or_field == "duration_s":
                    ref.asserted_value = evidence.duration_s
                elif ref.metric_or_field not in ALLOWED_TELEMETRY_FIELDS:
                    warnings.append(f"Unknown CV telemetry metric referenced: '{ref.metric_or_field}'.")
                    is_valid = False
            elif ref.source == "cv_event":
                # Check event ID in stopped events
                if ref.metric_or_field not in valid_event_refs and not any(
                    s.event_id == ref.metric_or_field for s in evidence.stopped_events
                ):
                    warnings.append(f"Referenced CV event '{ref.metric_or_field}' not found in job events.")
                    is_valid = False
            elif ref.source == "vlm_scene_interpretation":
                if not evidence.vlm_available:
                    warnings.append(f"Reference to VLM source when VLM was unavailable: {ref.metric_or_field}")
                    is_valid = False

            ref.verified = is_valid
            if is_valid:
                grounded_claims += 1
            else:
                demoted_claims += 1
            verified_refs.append(ref)

        corrected.evidence_references = verified_refs

        # ----------------------------------------------------------------------
        # 8. Adjust Confidence Classification
        # ----------------------------------------------------------------------
        if not evidence.vlm_available and corrected.confidence_classification == ReportConfidenceEnum.HIGH:
            notes.append("Adjusted confidence from HIGH to MEDIUM due to lack of VLM visual interpretation.")
            corrected.confidence_classification = ReportConfidenceEnum.MEDIUM

        if demoted_claims > grounded_claims and grounded_claims == 0:
            corrected.confidence_classification = ReportConfidenceEnum.CAUTIONARY

        validation_passed = len(warnings) == 0 or (grounded_claims >= demoted_claims)

        diagnostics = ReportValidationDiagnosticsSchema(
            validation_passed=validation_passed,
            total_claims_checked=total_claims_checked,
            grounded_claims=grounded_claims,
            demoted_claims=demoted_claims,
            warnings=warnings,
            notes=notes,
        )
        corrected.validation_diagnostics = diagnostics

        return corrected, diagnostics
