"""
Vision-Language Model (VLM) and Multimodal Evidence Schemas for TrafficLens-AI.

Defines Pydantic v2 data contracts for:
- Keyframe extraction configuration and diagnostics
- Multimodal evidence packages (internal and public representations)
- Telemetry and event references
- Scene understanding, environmental observations, and grounded epistemic claims
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Literal
from pydantic import BaseModel, Field


# ============================================================================
# 1. Extraction Configuration & Diagnostics
# ============================================================================

class ExtractionConfigSchema(BaseModel):
    """Configuration parameters governing deterministic keyframe extraction."""
    frame_budget: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of keyframes to extract (K budget)",
    )
    max_dimension: int = Field(
        default=960,
        ge=160,
        le=3840,
        description="Maximum bounding width or height in pixels for aspect-ratio preserved downscaling",
    )
    min_event_interval_s: float = Field(
        default=1.0,
        ge=0.0,
        description="Minimum temporal separation in seconds between candidate frames to avoid redundant captures",
    )
    jpeg_quality: int = Field(
        default=85,
        ge=10,
        le=100,
        description="JPEG compression quality parameter (1-100)",
    )
    include_pre_event_context: bool = Field(
        default=True,
        description="Whether to capture a context frame immediately prior to stationary vehicle event onset",
    )
    pre_event_offset_s: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Lead-in offset in seconds before event start for pre-event context frame",
    )


class ExtractionDiagnosticsSchema(BaseModel):
    """Runtime telemetry and diagnostic warnings from the keyframe extraction process."""
    total_candidates_identified: int = Field(
        default=0,
        ge=0,
        description="Total candidate frame timestamps identified before pruning",
    )
    candidates_pruned_duplicate: int = Field(
        default=0,
        ge=0,
        description="Number of candidate frames suppressed due to temporal proximity or redundancy",
    )
    candidates_pruned_budget: int = Field(
        default=0,
        ge=0,
        description="Number of candidate frames dropped due to exceeding the maximum frame budget (K)",
    )
    extracted_frame_count: int = Field(
        default=0,
        ge=0,
        description="Final count of successfully decoded and encoded keyframes",
    )
    verified_frame_count: int = Field(
        default=0,
        ge=0,
        description="Count of keyframes with verified decoder position",
    )
    unverified_frame_count: int = Field(
        default=0,
        ge=0,
        description="Count of keyframes whose decoder position could not be verified",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Diagnostic notices, e.g. fallback FPS, out-of-range clamps, decode errors",
    )
    source_fps: Optional[float] = Field(
        default=None,
        description="FPS value resolved from video stream or configuration",
    )
    source_total_frames: Optional[int] = Field(
        default=None,
        description="Total frame count reported by video container",
    )
    source_duration_s: Optional[float] = Field(
        default=None,
        description="Estimated or measured duration in seconds",
    )


# ============================================================================
# 2. Keyframe Metadata & Detail Schemas
# ============================================================================

class KeyframeMetadataSchema(BaseModel):
    """
    Public metadata describing an extracted video keyframe.
    
    Contains NO raw image binary or base64 strings to ensure fast, lightweight API responses.
    """
    keyframe_id: str = Field(..., description="Unique keyframe identifier (e.g. kf_{job_id}_{frame_idx})")
    source_video_id: str = Field(..., description="Identifier of the source video asset")
    source_job_id: str = Field(..., description="Identifier of the processing job")
    frame_idx: int = Field(..., ge=0, description="Exact 0-indexed frame index in the source video")
    requested_frame_idx: Optional[int] = Field(
        default=None,
        ge=0,
        description="Originally requested candidate frame index before decode verification",
    )
    is_verified: bool = Field(
        default=True,
        description="Whether the decoded frame index was verified via backend position queries",
    )
    timestamp_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Calculated frame timestamp in seconds (None if timing information unavailable)",
    )
    selection_reason: str = Field(
        ...,
        description="Documented justification for extraction (e.g. 'stopped_vehicle_start', 'scene_anchor')",
    )
    associated_event_ids: List[str] = Field(
        default_factory=list,
        description="List of CV event identifiers linked to this keyframe moment",
    )
    width: int = Field(..., gt=0, description="Image pixel width after resizing")
    height: int = Field(..., gt=0, description="Image pixel height after resizing")


class KeyframeDetailSchema(KeyframeMetadataSchema):
    """
    Complete keyframe representation including base64 encoded image data.
    
    Used strictly for internal VLM evidence packaging and model provider dispatch.
    """
    image_base64: str = Field(..., description="Base64 encoded JPEG image buffer")
    content_type: str = Field(default="image/jpeg", description="MIME image content type")


# ============================================================================
# 3. Telemetry and Event References
# ============================================================================

class TelemetryReferenceSchema(BaseModel):
    """Structured reference linking a specific time interval or metric to CV outputs."""
    telemetry_type: str = Field(
        ...,
        description="Category of telemetry (e.g. 'flow_window', 'roi_density', 'congestion_state', 'sequence_summary')",
    )
    time_interval_s: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Temporal interval [start_s, end_s] governing this telemetry record",
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Quantitative metrics dictionary sourced directly from CV analysis",
    )


class EventReferenceSchema(BaseModel):
    """Contextual descriptor for a discrete CV event incorporated into the evidence package."""
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of event (e.g. 'stopped_vehicle', 'congestion_transition')")
    track_id: Optional[int] = Field(default=None, description="ByteTrack vehicle track identifier if applicable")
    timestamp_s: Optional[float] = Field(default=None, ge=0.0, description="Start timestamp of event in seconds")
    duration_s: Optional[float] = Field(default=None, ge=0.0, description="Duration of event in seconds")
    roi_id: Optional[str] = Field(default=None, description="Associated ROI identifier if event occurred in a zone")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional structured attributes")


# ============================================================================
# 4. Multimodal Evidence Package Schemas
# ============================================================================

class EvidencePackagePublicSchema(BaseModel):
    """
    Public-facing schema for a packaged multimodal evidence artifact.
    
    Excludes massive base64 payloads to keep payloads compact and prevent filesystem leakages.
    """
    evidence_package_id: str = Field(..., description="Unique evidence package identifier")
    job_id: str = Field(..., description="Source processing job identifier")
    video_id: str = Field(..., description="Source video asset identifier")
    sequence_name: str = Field(..., description="Logical sequence name")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC creation timestamp",
    )
    keyframes: List[KeyframeMetadataSchema] = Field(
        default_factory=list,
        description="List of selected keyframe metadata entries (without image payloads)",
    )
    events: List[EventReferenceSchema] = Field(
        default_factory=list,
        description="List of discrete CV events represented in this evidence package",
    )
    telemetry_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="High-level sequence telemetry metrics (unique tracks, total flow, peak occupancy)",
    )
    config: ExtractionConfigSchema = Field(
        default_factory=ExtractionConfigSchema,
        description="Configuration parameters utilized during extraction",
    )
    diagnostics: ExtractionDiagnosticsSchema = Field(
        default_factory=ExtractionDiagnosticsSchema,
        description="Runtime diagnostics, candidate counts, and warnings",
    )


class EvidencePackageSchema(BaseModel):
    """
    Full multimodal evidence package with embedded keyframe image payloads.
    
    Used internally to pass images and structured telemetry to VLM providers.
    """
    evidence_package_id: str = Field(..., description="Unique evidence package identifier")
    job_id: str = Field(..., description="Source processing job identifier")
    video_id: str = Field(..., description="Source video asset identifier")
    sequence_name: str = Field(..., description="Logical sequence name")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC creation timestamp",
    )
    keyframes: List[KeyframeDetailSchema] = Field(
        default_factory=list,
        description="Selected keyframes including base64 image data",
    )
    events: List[EventReferenceSchema] = Field(
        default_factory=list,
        description="List of discrete CV events represented in this evidence package",
    )
    telemetry_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="High-level sequence telemetry metrics",
    )
    config: ExtractionConfigSchema = Field(
        default_factory=ExtractionConfigSchema,
        description="Configuration parameters utilized during extraction",
    )
    diagnostics: ExtractionDiagnosticsSchema = Field(
        default_factory=ExtractionDiagnosticsSchema,
        description="Runtime diagnostics, candidate counts, and warnings",
    )

    def to_public_schema(self) -> EvidencePackagePublicSchema:
        """Converts internal package with image data into clean public metadata schema."""
        public_keyframes = [
            KeyframeMetadataSchema(
                keyframe_id=kf.keyframe_id,
                source_video_id=kf.source_video_id,
                source_job_id=kf.source_job_id,
                frame_idx=kf.frame_idx,
                requested_frame_idx=kf.requested_frame_idx,
                is_verified=kf.is_verified,
                timestamp_s=kf.timestamp_s,
                selection_reason=kf.selection_reason,
                associated_event_ids=kf.associated_event_ids,
                width=kf.width,
                height=kf.height,
            )
            for kf in self.keyframes
        ]
        return EvidencePackagePublicSchema(
            evidence_package_id=self.evidence_package_id,
            job_id=self.job_id,
            video_id=self.video_id,
            sequence_name=self.sequence_name,
            created_at=self.created_at,
            keyframes=public_keyframes,
            events=self.events,
            telemetry_summary=self.telemetry_summary,
            config=self.config,
            diagnostics=self.diagnostics,
        )


# ============================================================================
# 5. Scene Understanding, Epistemic Claims & Interpretation Schemas
# ============================================================================

class ClaimCategoryEnum(str, Enum):
    """Epistemic classification of an analytical statement."""
    CV_FACT = "cv_fact"
    VISUAL_OBSERVATION = "visual_observation"
    INFERENCE = "inference"
    UNVERIFIED = "unverified"


class ConfidenceCategoryEnum(str, Enum):
    """Categorical certainty indicator avoiding uncalibrated floating point probabilities."""
    DIRECT_OBSERVATION = "direct_observation"   # Directly visible in keyframe
    SUPPORTED_INFERENCE = "supported_inference" # Contextually deduced from CV + visuals
    SPECULATIVE_HYPOTHESIS = "speculative"     # Plausible but unconfirmed


class WeatherConditionEnum(str, Enum):
    CLEAR = "clear"
    OVERCAST = "overcast"
    RAIN = "rain"
    FOG = "fog"
    UNKNOWN = "unknown"


class IlluminationConditionEnum(str, Enum):
    DAYLIGHT = "daylight"
    TWILIGHT = "twilight"
    NIGHT = "night"
    UNKNOWN = "unknown"


class RoadSurfaceConditionEnum(str, Enum):
    DRY = "dry"
    WET_REFLECTIVE = "wet_reflective"
    SNOW_ICE = "snow_ice"
    UNKNOWN = "unknown"


class EnvironmentalConditionsSchema(BaseModel):
    """Environmental and atmospheric scene parameters."""
    weather: WeatherConditionEnum = Field(..., description="Observed weather condition")
    illumination: IlluminationConditionEnum = Field(..., description="Observed ambient illumination")
    road_surface: RoadSurfaceConditionEnum = Field(..., description="Observed road surface state")
    visibility_hazards: List[str] = Field(
        default_factory=list,
        description="Visible impediments such as lens glare, spray, or smoke",
    )
    confidence: ConfidenceCategoryEnum = Field(..., description="Certainty of environmental observation")


class VisualObservationSchema(BaseModel):
    """Direct visual feature identified in a specific keyframe."""
    observation_id: str = Field(..., description="Unique observation identifier")
    frame_timestamp_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Video timestamp in seconds where visible (None if timing unknown)",
    )
    keyframe_id: Optional[str] = Field(
        default=None,
        description="Identifier of the keyframe providing direct visual evidence",
    )
    description: str = Field(..., description="Description of the visual feature")
    bounding_box_normalized: Optional[Tuple[float, float, float, float]] = Field(
        default=None,
        description="Normalized coordinates (ymin, xmin, ymax, xmax) in [0.0, 1.0]",
    )
    visual_cues: List[str] = Field(
        default_factory=list,
        description="Supporting visual cues (e.g. 'brake lights illuminated', 'pylons present')",
    )


class EventInterpretationSchema(BaseModel):
    """Contextual root-cause interpretation for a stationary or anomalous event."""
    event_id: str = Field(..., description="Identifier matching the CV stopped_vehicle event")
    track_id: int = Field(..., description="Vehicle track ID assigned by ByteTrack")
    start_timestamp_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Event start timestamp in seconds (None if timing unknown)",
    )
    end_timestamp_s: Optional[float] = Field(default=None, description="Event resolution timestamp")
    cv_measured_duration_s: float = Field(..., ge=0.0, description="Measured stationary duration in seconds")
    probable_cause: str = Field(..., description="Interpreted primary reason for stoppage")
    contextual_explanation: str = Field(..., description="Detailed narrative explaining scene factors")
    confidence: ConfidenceCategoryEnum = Field(..., description="Confidence classification of interpretation")
    supporting_observations: List[str] = Field(
        default_factory=list,
        description="References to observation_ids supporting this interpretation",
    )


class TelemetryProvenanceSchema(BaseModel):
    """
    Structured citation linking a CV fact claim directly to verified telemetry or event records.
    """
    source_type: Literal["telemetry_summary", "event"] = Field(
        ...,
        description="Source category: 'telemetry_summary' or 'event'",
    )
    record_id: Optional[str] = Field(
        default=None,
        description="Specific record ID if source_type is 'event' (e.g. 'stop_1_0.40')",
    )
    field_path: str = Field(
        ...,
        description="Attribute or metric path (e.g. 'duration_s', 'total_unique_tracks', 'line_counts.line_1', 'fps')",
    )
    numeric_value: Optional[float] = Field(
        default=None,
        description="Asserted quantitative numeric value to check against source",
    )
    text_value: Optional[str] = Field(
        default=None,
        description="Asserted text or categorical value to check against source",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Entity or track subject of the fact (e.g. 'track_1', 'scene', 'roi_main')",
    )
    predicate: Optional[str] = Field(
        default=None,
        description="Asserted property or relationship (e.g. 'stationary_duration', 'track_count', 'detection_count')",
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement for numeric_value (e.g. 'seconds', 's', 'count', 'fps', 'vpm')",
    )


class EpistemicClaimSchema(BaseModel):
    """Individual atomic claim with explicit provenance and attribution."""
    claim_id: str = Field(..., description="Unique claim identifier")
    category: ClaimCategoryEnum = Field(..., description="Epistemic status of claim")
    statement: str = Field(..., description="Factual or inferential statement")
    keyframe_ids: List[str] = Field(
        default_factory=list,
        description="Associated keyframe identifiers providing direct visual evidence",
    )
    frame_timestamps: List[float] = Field(default_factory=list, description="Associated frame timestamps")
    associated_track_ids: List[int] = Field(default_factory=list, description="Associated vehicle track IDs")
    evidence_rationale: str = Field(..., description="Basis for this claim")
    telemetry_provenance: Optional[TelemetryProvenanceSchema] = Field(
        default=None,
        description="Structured telemetry citation required for verified cv_fact claims",
    )


class ModelMetadataSchema(BaseModel):
    """Execution telemetry and provenance of the VLM inference run."""
    model_name: str = Field(..., description="Identifier of model used (e.g. Qwen3-VL-2B-Instruct)")
    provider_type: str = Field(..., description="Provider backend (e.g. openai_compatible, huggingface, mock)")
    execution_time_ms: int = Field(..., ge=0, description="Inference wall-clock duration in milliseconds")
    prompt_tokens: int = Field(..., ge=0, description="Total input tokens consumed")
    completion_tokens: int = Field(..., ge=0, description="Total generation tokens produced")
    quantization_precision: Optional[str] = Field(default=None, description="Quantization format (e.g. int4, fp16)")


class SceneInterpretationResponseSchema(BaseModel):
    """Master response schema for multimodal scene understanding."""
    job_id: str = Field(..., description="Associated processing job identifier")
    video_id: str = Field(..., description="Associated video asset identifier")
    scene_summary: str = Field(..., description="Comprehensive natural-language scene narrative")
    environmental_conditions: EnvironmentalConditionsSchema = Field(..., description="Environmental context")
    roadway_description: str = Field(..., description="Description of road layout, lane geometry, and camera view")
    visual_observations: List[VisualObservationSchema] = Field(
        default_factory=list,
        description="List of directly observed visual features",
    )
    event_interpretations: List[EventInterpretationSchema] = Field(
        default_factory=list,
        description="Interpretations of stationary vehicle events",
    )
    claims: List[EpistemicClaimSchema] = Field(
        default_factory=list,
        description="List of verified and categorized claims",
    )
    unverified_aspects: List[str] = Field(
        default_factory=list,
        description="Explicit disclosures of aspects that cannot be determined from the view",
    )
    model_metadata: ModelMetadataSchema = Field(..., description="Inference telemetry")


# ============================================================================
# 6. Provider Health & Service Result Schemas
# ============================================================================

class ProviderHealthSchema(BaseModel):
    """Health check status and connectivity diagnostics for a VLM provider."""
    provider_type: str = Field(..., description="Identifier of provider backend")
    model_name: str = Field(..., description="Configured model identifier")
    is_healthy: bool = Field(..., description="True if provider endpoint is reachable and operational")
    status_message: str = Field(..., description="Descriptive status or error message")
    latency_ms: Optional[int] = Field(default=None, description="Round-trip health check latency in ms")


class SceneInterpretationFailureSchema(BaseModel):
    """Structured error descriptor returned when VLM inference fails, isolating the failure."""
    job_id: str = Field(..., description="Associated processing job identifier")
    video_id: str = Field(..., description="Associated video asset identifier")
    error_type: str = Field(..., description="Category of error (e.g. timeout, connection, parsing, disabled)")
    message: str = Field(..., description="Sanitized human-readable error description (no secrets)")
    provider_type: str = Field(..., description="Provider backend that encountered the error")
    model_name: str = Field(default="unknown", description="Configured model identifier")
    is_retryable: bool = Field(default=False, description="Whether the failure was transient and retryable")


class SceneInterpretationResult(BaseModel):
    """Composite result container from SceneInterpretationService."""
    success: bool = Field(..., description="Whether scene interpretation succeeded")
    interpretation: Optional[SceneInterpretationResponseSchema] = Field(
        default=None,
        description="Validated scene interpretation if successful",
    )
    failure: Optional[SceneInterpretationFailureSchema] = Field(
        default=None,
        description="Failure descriptor if inference failed",
    )
