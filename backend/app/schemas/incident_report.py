"""
Pydantic schemas for Evidence-Grounded Traffic Incident Reports in TrafficLens-AI.

Defines domain contracts for structured incident reports, evidence references,
epistemic classifications, validation diagnostics, and API response payloads.
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class ReportConfidenceEnum(str, Enum):
    """Overall confidence classification of an incident report."""
    HIGH = "high"             # Comprehensive CV telemetry + verified VLM visual context
    MEDIUM = "medium"         # Robust CV telemetry alone, or partial VLM visual observations
    LOW = "low"               # Sparse telemetry, brief sequence, or significant unverified aspects
    CAUTIONARY = "cautionary" # Conflicting observations, decoding anomalies, or major gaps


class VerificationClassificationEnum(str, Enum):
    """Epistemic classification of individual analytical assertions."""
    GROUNDED_FACT = "grounded_fact"                 # Directly measured in CV telemetry
    DIRECT_VISUAL_OBSERVATION = "visual_observation" # Directly observable in keyframes via VLM
    SUPPORTED_INFERENCE = "supported_inference"     # Contextually deduced from facts and visuals
    UNVERIFIED_SPECULATION = "unverified"           # Plausible hypothesis lacking direct evidence


class TrafficConditionsSchema(BaseModel):
    """Quantitative baseline traffic flow and conditions summary."""
    sequence_name: str = Field(..., description="Sequence or video identifier")
    duration_s: float = Field(..., ge=0.0, description="Video duration in seconds")
    total_detections: int = Field(..., ge=0, description="Total bounding boxes detected")
    total_tracked_instances: int = Field(
        ...,
        ge=0,
        description="Total unique track IDs assigned (Note: represents tracker identities, not unique physical vehicle census)",
    )
    overall_flow_rate_vpm: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Peak or average traffic flow in vehicles per minute (VPM)",
    )
    peak_occupancy_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Highest spatial occupancy ratio observed across defined ROIs",
    )
    environmental_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Observed weather, illumination, and road surface from VLM scene interpretation if available",
    )
    visual_context_available: bool = Field(
        default=False,
        description="True if verified VLM scene interpretation was available; False if generated from CV telemetry alone",
    )
    visual_context_note: Optional[str] = Field(
        default=None,
        description="Explicit notice regarding presence or absence of multimodal visual interpretation",
    )


class CongestionObservationSchema(BaseModel):
    """Observed traffic congestion in a specific Region of Interest."""
    roi_id: str = Field(..., description="Region of interest identifier")
    observed_level: str = Field(..., description="Classified congestion level (low, moderate, high)")
    peak_occupancy: float = Field(..., ge=0.0, le=1.0, description="Peak spatial occupancy ratio [0.0, 1.0]")
    average_occupancy: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Mean spatial occupancy ratio across sequence duration",
    )
    is_heuristic_rule: bool = Field(
        default=True,
        description="Flag confirming this is an explainable rule-based classification rather than a certified traffic engineering measurement",
    )
    description: str = Field(..., description="Summary explanation of spatial occupancy and flow state in this zone")


class StoppedVehicleReportItemSchema(BaseModel):
    """Documented stationary or stopped vehicle incident."""
    event_id: str = Field(..., description="Unique stopped vehicle event identifier")
    track_id: int = Field(..., description="Vehicle tracklet identifier from ByteTrack")
    start_timestamp_s: float = Field(..., ge=0.0, description="Video timestamp when vehicle became stationary (seconds)")
    duration_s: float = Field(..., ge=0.0, description="Measured stationary duration in seconds")
    location: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Spatial coordinates (x, y) of the vehicle center in pixel space",
    )
    roi_id: Optional[str] = Field(default=None, description="Associated ROI if vehicle is within a defined zone")
    probable_cause: Optional[str] = Field(
        default=None,
        description="Interpreted cause from VLM (e.g. queue stagnation, traffic signal, obstruction) or None if unobserved",
    )
    contextual_explanation: Optional[str] = Field(
        default=None,
        description="Detailed context explaining why the vehicle remained stationary",
    )
    verification_status: str = Field(
        default="grounded",
        description="Verification state of this stoppage record (e.g. 'grounded', 'unverified')",
    )


class VisualObservationReportItemSchema(BaseModel):
    """Direct visual observation extracted from keyframes via multimodal VLM."""
    observation_id: str = Field(..., description="Unique observation identifier")
    keyframe_id: Optional[str] = Field(default=None, description="Identifier of the keyframe providing visual evidence")
    timestamp_s: Optional[float] = Field(default=None, ge=0.0, description="Frame timestamp in seconds")
    description: str = Field(..., description="Factual description of the observed physical feature")
    visual_cues: List[str] = Field(
        default_factory=list,
        description="Visual indicators noted (e.g. 'active brake lights', 'wet road reflection')",
    )


class ObservedEventSchema(BaseModel):
    """Significant classified traffic event or operational irregularity."""
    event_type: str = Field(..., description="Type of event (e.g. 'stopped_vehicle', 'queue_onset', 'flow_drop')")
    timestamp_s: Optional[float] = Field(default=None, ge=0.0, description="Timestamp in seconds when event occurred")
    description: str = Field(..., description="Summary of the event and its operational significance")
    supporting_evidence_ref: str = Field(..., description="Citation to underlying CV metric, event ID, or keyframe")


class EvidenceReferenceSchema(BaseModel):
    """Citation linking an asserted report claim directly to verified source telemetry."""
    source: str = Field(..., description="Source subsystem ('cv_telemetry', 'vlm_scene_interpretation', 'cv_event')")
    metric_or_field: str = Field(..., description="Telemetry metric or field cited (e.g. 'duration_s', 'line_counts')")
    asserted_value: Any = Field(..., description="Asserted quantitative or qualitative value")
    verified: bool = Field(default=True, description="Whether the assertion was verified against ground-truth data")


class FollowUpRecommendationSchema(BaseModel):
    """
    Suggested operational follow-up action.
    
    Clearly labelled as an advisory suggestion rather than an established factual event.
    """
    action: str = Field(..., description="Recommended operational action or inspection")
    priority: str = Field(default="medium", description="Priority level ('low', 'medium', 'high')")
    rationale: str = Field(..., description="Reasoning connecting the suggestion to observed traffic evidence")
    is_advisory: bool = Field(
        default=True,
        description="Explicit flag declaring that this is an advisory suggestion, not an established factual incident claim",
    )


class ReportValidationDiagnosticsSchema(BaseModel):
    """Diagnostics and integrity audit produced by the GroundingValidator."""
    validation_passed: bool = Field(..., description="Whether the report satisfied all grounding constraints")
    total_claims_checked: int = Field(default=0, ge=0, description="Total factual claims evaluated")
    grounded_claims: int = Field(default=0, ge=0, description="Number of claims verified against ground truth")
    demoted_claims: int = Field(default=0, ge=0, description="Number of ungrounded or speculative claims demoted")
    warnings: List[str] = Field(default_factory=list, description="Integrity warnings or suppressed unsupported terms")
    notes: List[str] = Field(default_factory=list, description="General validation notes")


class IncidentReportSchema(BaseModel):
    """
    Core structured schema for an Evidence-Grounded Traffic Incident Report.
    
    Adheres strictly to the project domain model, prevents hallucination of
    collisions or uncalibrated speeds, and enforces transparent epistemic boundaries.
    """
    report_id: str = Field(..., description="Unique incident report identifier ('rep_<uuid>')")
    job_id: str = Field(..., description="Associated completed ProcessingJob identifier")
    video_id: str = Field(..., description="Associated source video asset identifier")
    sequence_name: str = Field(..., description="Logical sequence name")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC generation timestamp",
    )
    executive_summary: str = Field(
        ...,
        description="High-level, evidence-grounded summary of traffic conditions and notable events",
    )
    traffic_conditions: TrafficConditionsSchema = Field(
        ...,
        description="Grounded quantitative traffic flow and condition metrics",
    )
    observed_events: List[ObservedEventSchema] = Field(
        default_factory=list,
        description="Chronological or prioritized traffic events (stopped vehicles, queue formations)",
    )
    congestion_observations: List[CongestionObservationSchema] = Field(
        default_factory=list,
        description="Explainable ROI congestion observations",
    )
    stopped_vehicles: List[StoppedVehicleReportItemSchema] = Field(
        default_factory=list,
        description="Detailed stopped vehicle incident items",
    )
    visual_observations: List[VisualObservationReportItemSchema] = Field(
        default_factory=list,
        description="Visual keyframe observations from VLM (empty if VLM was unavailable)",
    )
    evidence_references: List[EvidenceReferenceSchema] = Field(
        default_factory=list,
        description="Direct citations linking report assertions to underlying CV telemetry and VLM claims",
    )
    confidence_classification: ReportConfidenceEnum = Field(
        ...,
        description="Overall confidence rating based on data availability and verification completeness",
    )
    unverified_aspects: List[str] = Field(
        default_factory=list,
        description="Aspects of the scene or events that could not be verified from available evidence",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Explicit operational limitations (e.g. uncalibrated camera, lack of visual interpretation, occlusions)",
    )
    recommendations: List[FollowUpRecommendationSchema] = Field(
        default_factory=list,
        description="Operational follow-up actions, strictly designated as suggestions rather than factual assertions",
    )
    validation_diagnostics: Optional[ReportValidationDiagnosticsSchema] = Field(
        default=None,
        description="Grounding validation audit report",
    )


class IncidentReportResponseSchema(BaseModel):
    """API response schema wrapping an IncidentReport."""
    report_id: str = Field(..., description="Report identifier")
    job_id: str = Field(..., description="ProcessingJob identifier")
    video_id: str = Field(..., description="Video asset identifier")
    sequence_name: str = Field(..., description="Sequence name")
    status: str = Field(default="COMPLETED", description="Report status: 'COMPLETED' or 'FAILED'")
    created_at: datetime = Field(..., description="Creation UTC timestamp")
    report: Optional[IncidentReportSchema] = Field(default=None, description="Structured incident report payload")
    validation_passed: bool = Field(default=True, description="Whether the report passed epistemic grounding checks")
    error_message: Optional[str] = Field(default=None, description="Sanitized error description if generation failed")


class IncidentReportHistoryResponseSchema(BaseModel):
    """API response schema returning historical incident reports for a processing job."""
    job_id: str = Field(..., description="ProcessingJob identifier")
    total_reports: int = Field(..., ge=0, description="Total incident reports generated for this job")
    reports: List[IncidentReportResponseSchema] = Field(
        default_factory=list,
        description="List of incident reports in chronological order",
    )
