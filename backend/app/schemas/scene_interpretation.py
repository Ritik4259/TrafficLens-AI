"""
Pydantic v2 schemas and lifecycle state validation for multimodal scene interpretations.

Defines:
- Lifecycle enums (persisted statuses and virtual NOT_REQUESTED)
- State machine transition validation and exceptions
- Creation, response, error, and history contracts
- Structured environmental context, roadway description, and event interpretations
- Incomplete result protection safeguards and strict error sanitization
"""

from __future__ import annotations
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from backend.app.models.scene_interpretation import SceneInterpretationStatus
from backend.app.schemas.vlm import (
    EpistemicClaimSchema,
    TelemetryProvenanceSchema,
    ClaimCategoryEnum,
    ConfidenceCategoryEnum,
)


class SceneInterpretationLifecycleStatus(str, Enum):
    """
    Client-facing lifecycle statuses for scene interpretations.

    NOT_REQUESTED: Virtual status returned when no interpretation attempt exists for a job.
    QUEUED: Persisted status when an interpretation request is enqueued.
    PROCESSING: Persisted status when an active worker is evaluating keyframes and VLM.
    COMPLETED: Terminal persisted status when scene understanding succeeded and results are stored.
    FAILED: Terminal persisted status when inference or processing failed.
    """
    NOT_REQUESTED = "NOT_REQUESTED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# State machine allowed transitions for persisted statuses
ALLOWED_INTERPRETATION_TRANSITIONS: Dict[SceneInterpretationStatus, Set[SceneInterpretationStatus]] = {
    SceneInterpretationStatus.QUEUED: {SceneInterpretationStatus.PROCESSING},
    SceneInterpretationStatus.PROCESSING: {SceneInterpretationStatus.COMPLETED, SceneInterpretationStatus.FAILED},
    SceneInterpretationStatus.COMPLETED: set(),  # Terminal state
    SceneInterpretationStatus.FAILED: set(),     # Terminal state
}

# Analytical result fields that are ONLY permitted when status == COMPLETED
ANALYTICAL_RESULT_FIELDS: Tuple[str, ...] = (
    "summary",
    "environmental_context",
    "scene_description",
    "event_interpretations",
    "epistemic_claims",
    "unverified_aspects",
    "evidence_package_id",
)


class InvalidInterpretationStatusTransitionError(ValueError):
    """Raised when an illegal interpretation status transition is attempted."""

    def __init__(
        self,
        current_status: Any,
        target_status: Any,
        message: Optional[str] = None,
    ):
        curr_val = current_status.value if isinstance(current_status, Enum) else str(current_status)
        target_val = target_status.value if isinstance(target_status, Enum) else str(target_status)
        msg = message or (
            f"Cannot transition scene interpretation from '{curr_val}' to '{target_val}'. "
            f"Allowed transitions: QUEUED -> PROCESSING, PROCESSING -> COMPLETED | FAILED. "
            f"Terminal states (COMPLETED, FAILED) cannot be transitioned."
        )
        super().__init__(msg)
        self.current_status = current_status
        self.target_status = target_status


def validate_interpretation_status_transition(
    current_status: Union[SceneInterpretationStatus, str],
    target_status: Union[SceneInterpretationStatus, str],
) -> None:
    """
    Validates that a transition between interpretation statuses is permitted.

    Permitted transitions:
        QUEUED -> PROCESSING
        PROCESSING -> COMPLETED
        PROCESSING -> FAILED

    All other transitions (including mutations of COMPLETED or FAILED) are rejected.
    """
    try:
        curr = (
            current_status
            if isinstance(current_status, SceneInterpretationStatus)
            else SceneInterpretationStatus(current_status)
        )
    except ValueError:
        raise InvalidInterpretationStatusTransitionError(current_status, target_status, f"Invalid current status: '{current_status}'")

    try:
        target = (
            target_status
            if isinstance(target_status, SceneInterpretationStatus)
            else SceneInterpretationStatus(target_status)
        )
    except ValueError:
        raise InvalidInterpretationStatusTransitionError(current_status, target_status, f"Invalid target status: '{target_status}'")

    allowed = ALLOWED_INTERPRETATION_TRANSITIONS.get(curr, set())
    if target not in allowed:
        raise InvalidInterpretationStatusTransitionError(curr, target)


# ==============================================================================
# Error & Sanitization Utilities
# ==============================================================================

_CREDENTIAL_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z-_]{10,}", re.IGNORECASE),
    re.compile(r"sk-[0-9A-Za-z-_]{10,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[0-9A-Za-z-_.]{10,}", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^\s'\",&]+['\"]?", re.IGNORECASE),
]

_PATH_PATTERNS = [
    re.compile(r"[a-zA-Z]:\\(?:[^\s\\/:*?\"<>|\r\n]+\\)*[^\s\\/:*?\"<>|\r\n]*"),
    re.compile(r"[a-zA-Z]:/(?:[^\s\\/:*?\"<>|\r\n]+/)*[^\s\\/:*?\"<>|\r\n]*"),
    re.compile(r"(?:/(?:home|Users|tmp|var|etc|usr|opt|storage|data)[^\s\r\n]*)"),
    re.compile(r"file://[^\s]+"),
]

_TRACEBACK_PATTERN = re.compile(
    r"Traceback \(most recent call last\):.*?(?:\n[a-zA-Z0-9_.]+(?:Error|Exception):.*|\Z)",
    re.DOTALL,
)

_FILE_LINE_PATTERN = re.compile(r'File\s+"[^"]+",\s+line\s+\d+.*', re.IGNORECASE)

_RAW_JSON_PAYLOAD_PATTERN = re.compile(
    r'\{.*?"candidates"\s*:\s*\[.*?\}',
    re.DOTALL,
)


def sanitize_error_text(text: str) -> str:
    """
    Sanitizes raw error strings to remove credentials, file paths, tracebacks,
    and raw JSON dumps while preserving actionable error messages.
    """
    if not text:
        return ""

    sanitized = text

    # 1. Redact credentials
    for pattern in _CREDENTIAL_PATTERNS:
        sanitized = pattern.sub("[REDACTED_CREDENTIAL]", sanitized)

    # 2. Redact stack tracebacks
    if "Traceback (most recent call last):" in sanitized:
        sanitized = _TRACEBACK_PATTERN.sub("[TRACEBACK REDACTED]", sanitized)
    sanitized = _FILE_LINE_PATTERN.sub("[LINE REFERENCE REDACTED]", sanitized)

    # 3. Redact local filesystem paths
    for pattern in _PATH_PATTERNS:
        sanitized = pattern.sub("[REDACTED_PATH]", sanitized)

    # 4. Redact raw model response dumps
    if '"candidates"' in sanitized or '"parts"' in sanitized:
        sanitized = _RAW_JSON_PAYLOAD_PATTERN.sub("[RAW RESPONSE REDACTED]", sanitized)

    return sanitized.strip()


def sanitize_error_details(details: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Recursively sanitizes dictionary metadata, dropping sensitive debug keys."""
    if not details:
        return None

    cleaned: Dict[str, Any] = {}
    for k, v in details.items():
        # Drop raw payload or stack trace keys entirely
        if k.lower() in ("raw_response", "stack_trace", "traceback", "api_key", "secret", "token", "password"):
            continue
        if isinstance(v, str):
            cleaned[k] = sanitize_error_text(v)
        elif isinstance(v, (int, float, bool)) or v is None:
            cleaned[k] = v
        elif isinstance(v, dict):
            cleaned[k] = sanitize_error_details(v)
        elif isinstance(v, list):
            cleaned[k] = [
                sanitize_error_text(item) if isinstance(item, str) else item
                for item in v
            ]
        else:
            cleaned[k] = str(v)
    return cleaned


# ==============================================================================
# Configuration & Creation Schemas
# ==============================================================================

class SceneInterpretationCreateSchema(BaseModel):
    """Optional payload to configure on-demand scene interpretation."""
    model_config = ConfigDict(extra="forbid")

    provider_type: Optional[str] = Field(default=None, description="Target VLM provider (e.g. 'gemini')")
    model_name: Optional[str] = Field(default=None, description="Specific model ID (e.g. 'gemini-3.8-flash')")
    frame_budget: Optional[int] = Field(default=5, ge=1, le=10, description="Max keyframes to extract (K <= 10)")
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=1.0, description="Sampling temperature")


# ==============================================================================
# Sanitized Error Schema
# ==============================================================================

class SceneInterpretationErrorSchema(BaseModel):
    """Structured error descriptor for failed scene interpretations."""
    model_config = ConfigDict(extra="ignore")

    error_code: str = Field(..., description="Canonical error code string (e.g. VLM_PROVIDER_TIMEOUT)")
    error_message: str = Field(..., description="Sanitized human-readable error description")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="Contextual diagnostic metadata")

    @field_validator("error_message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        """Sanitizes sensitive tokens, paths, and tracebacks from error message strings."""
        return sanitize_error_text(v)

    @field_validator("error_details")
    @classmethod
    def sanitize_details(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sanitizes sensitive tokens, paths, and stack traces from error details."""
        return sanitize_error_details(v)


# ==============================================================================
# Structured Output Component Schemas
# ==============================================================================

class EnvironmentalContextSchema(BaseModel):
    """Environmental and ambient roadway conditions."""
    model_config = ConfigDict(extra="allow", from_attributes=True)

    weather: Optional[str] = Field(default=None, description="Observed weather condition (e.g. clear, overcast, rain)")
    lighting: Optional[str] = Field(default=None, description="Ambient illumination (e.g. daylight, twilight, night)")
    pavement: Optional[str] = Field(default=None, description="Observed road surface state (e.g. dry, wet)")
    camera_angle: Optional[str] = Field(default=None, description="Camera viewing angle perspective")
    visibility_hazards: List[str] = Field(default_factory=list, description="Visible impediments (e.g. glare, spray)")


class SceneDescriptionSchema(BaseModel):
    """Static roadway layout and traffic topology."""
    model_config = ConfigDict(extra="allow", from_attributes=True)

    roadway_topology: Optional[str] = Field(default=None, description="Roadway classification (e.g. divided_highway)")
    lane_count: Optional[int] = Field(default=None, ge=1, description="Number of observed traffic lanes")
    traffic_density_qualitative: Optional[str] = Field(default=None, description="Qualitative traffic density (e.g. light, moderate)")
    features: List[str] = Field(default_factory=list, description="Observed roadway features (e.g. shoulder, barrier)")


class TrafficEventInterpretationSchema(BaseModel):
    """Contextual interpretation of a detected traffic event."""
    model_config = ConfigDict(extra="allow", from_attributes=True)

    event_id: str = Field(..., description="Identifier matching the CV stopped_vehicle event")
    event_type: Optional[str] = Field(default=None, description="Type of event (e.g. stopped_vehicle)")
    track_id: Optional[int] = Field(default=None, description="Vehicle track ID assigned by ByteTrack")
    interpretation: str = Field(..., description="Narrative explanation of the event")
    probable_cause: Optional[str] = Field(default=None, description="Interpreted primary reason for stoppage")
    confidence: Optional[str] = Field(default=None, description="Confidence categorization")
    supporting_citations: List[str] = Field(default_factory=list, description="Keyframe or telemetry citations")


# ==============================================================================
# Master API Response Schema
# ==============================================================================

class SceneInterpretationResponseSchema(BaseModel):
    """Comprehensive API response for a scene interpretation run."""
    model_config = ConfigDict(from_attributes=True, extra="ignore")

    id: Optional[str] = Field(default=None, description="Interpretation ID (null if NOT_REQUESTED)")
    job_id: str = Field(..., description="Associated processing job identifier")
    video_id: Optional[str] = Field(default=None, description="Associated video asset identifier")
    status: SceneInterpretationLifecycleStatus = Field(..., description="Current lifecycle state")
    created_at: Optional[datetime] = Field(default=None, description="Task creation timestamp in UTC")
    started_at: Optional[datetime] = Field(default=None, description="Timestamp when worker claimed task")
    completed_at: Optional[datetime] = Field(default=None, description="Timestamp of completion or failure")
    duration_seconds: Optional[float] = Field(default=None, ge=0.0, description="Total wall-clock duration in seconds")

    provider_type: Optional[str] = Field(default=None, description="VLM provider identifier (e.g. gemini)")
    model_name: Optional[str] = Field(default=None, description="Model identifier used")
    prompt_tokens: Optional[int] = Field(default=None, ge=0, description="Prompt input tokens")
    completion_tokens: Optional[int] = Field(default=None, ge=0, description="Generated output tokens")
    total_tokens: Optional[int] = Field(default=None, ge=0, description="Total tokens consumed")

    summary: Optional[str] = Field(default=None, description="Natural language scene summary")
    environmental_context: Optional[EnvironmentalContextSchema] = Field(
        default=None,
        description="Ambient environmental conditions",
    )
    scene_description: Optional[SceneDescriptionSchema] = Field(
        default=None,
        description="Roadway geometry and layout",
    )
    event_interpretations: Optional[List[TrafficEventInterpretationSchema]] = Field(
        default=None,
        description="Traffic event root-cause interpretations",
    )
    epistemic_claims: Optional[List[EpistemicClaimSchema]] = Field(
        default=None,
        description="Categorized epistemic claims",
    )
    unverified_aspects: Optional[List[str]] = Field(
        default=None,
        description="Explicit disclosures of unverified aspects",
    )
    evidence_package_id: Optional[str] = Field(default=None, description="Referenced evidence package ID")

    error: Optional[SceneInterpretationErrorSchema] = Field(default=None, description="Error details if status is FAILED")

    @model_validator(mode="before")
    @classmethod
    def extract_from_orm(cls, data: Any) -> Any:
        """Adapts SQLAlchemy SceneInterpretation model to response schema structure."""
        if hasattr(data, "job") and hasattr(data, "id") and not isinstance(data, dict):
            job = getattr(data, "job", None)
            vid = getattr(job, "video_id", None) if job else getattr(data, "video_id", None)

            err = None
            if getattr(data, "error_code", None) or getattr(data, "error_message", None):
                err = {
                    "error_code": data.error_code or "UNKNOWN_ERROR",
                    "error_message": data.error_message or "",
                    "error_details": data.error_details,
                }

            status_val = data.status.value if hasattr(data.status, "value") else str(data.status)

            # Incomplete result protection at ORM extraction level:
            # If status is not COMPLETED, analytical result fields are strictly stripped
            is_completed = (status_val == SceneInterpretationLifecycleStatus.COMPLETED.value)
            return {
                "id": data.id,
                "job_id": data.job_id,
                "video_id": vid,
                "status": status_val,
                "created_at": data.created_at,
                "started_at": data.started_at,
                "completed_at": data.completed_at,
                "duration_seconds": data.duration_seconds,
                "provider_type": data.provider_type,
                "model_name": data.model_name,
                "prompt_tokens": data.prompt_tokens,
                "completion_tokens": data.completion_tokens,
                "total_tokens": data.total_tokens,
                "summary": data.summary if is_completed else None,
                "environmental_context": data.environmental_context if is_completed else None,
                "scene_description": data.scene_description if is_completed else None,
                "event_interpretations": data.event_interpretations if is_completed else None,
                "epistemic_claims": data.epistemic_claims if is_completed else None,
                "unverified_aspects": data.unverified_aspects if is_completed else None,
                "evidence_package_id": data.evidence_package_id if is_completed else None,
                "error": err,
            }
        return data

    @model_validator(mode="after")
    def validate_result_protection(self) -> Self:
        """
        Incomplete Result Protection Safeguard:
        1. For every status other than COMPLETED, ensures all analytical result fields are None.
        2. For COMPLETED, ensures required summary is present and non-empty, and error is None.
        """
        if self.status != SceneInterpretationLifecycleStatus.COMPLETED:
            for field_name in ANALYTICAL_RESULT_FIELDS:
                if getattr(self, field_name) is not None:
                    raise ValueError(
                        f"Incomplete interpretation with status '{self.status.value}' "
                        f"cannot expose analytical result field '{field_name}'."
                    )

        if self.status == SceneInterpretationLifecycleStatus.COMPLETED:
            if not self.summary or not self.summary.strip():
                raise ValueError("Completed interpretation must contain a non-empty summary.")
            if self.error is not None:
                raise ValueError("Completed interpretation cannot contain error details.")

        return self

    @classmethod
    def not_requested(cls, job_id: str, video_id: Optional[str] = None) -> SceneInterpretationResponseSchema:
        """Constructs a virtual NOT_REQUESTED response when no interpretation record exists."""
        return cls(
            id=None,
            job_id=job_id,
            video_id=video_id,
            status=SceneInterpretationLifecycleStatus.NOT_REQUESTED,
        )


# ==============================================================================
# Interpretation Attempt History Schema
# ==============================================================================

class SceneInterpretationHistoryResponseSchema(BaseModel):
    """Response schema returning all interpretation attempts for a given job."""
    model_config = ConfigDict(extra="ignore")

    job_id: str = Field(..., description="Unique processing job identifier")
    total_attempts: int = Field(..., ge=0, description="Total number of interpretation attempts recorded")
    attempts: List[SceneInterpretationResponseSchema] = Field(
        default_factory=list,
        description="Chronological or reverse-chronological list of interpretation attempt records",
    )
