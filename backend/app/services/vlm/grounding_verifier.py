"""
Strict Epistemic Grounding Verifier for VLM Scene Understanding (Milestone 5C).

Provides comprehensive, provider-independent epistemic grounding enforcement
across Google Gemini, OpenAI-compatible remote adapters, and local providers.

Core Invariants Enforced:
1. Visual Observations:
   - Every visual observation must cite a valid supplied keyframe ID.
   - Observations citing unknown keyframes or None are rejected from visual_observations
     and disclosed in unverified_aspects. They are NEVER silently converted to None.

2. Inference Claims:
   - Must cite valid supporting evidence (keyframes or verified track/event IDs).
   - Inferences citing no evidence or unknown evidence are demoted to UNVERIFIED.

3. CV Fact Claims (Content & Provenance Integrity):
   - Every CV fact MUST provide structured telemetry provenance (TelemetryProvenanceSchema).
   - Missing provenance is immediately demoted to UNVERIFIED, regardless of telemetry presence.
   - Field paths must strictly adhere to an explicit allowlist of approved CV measurements.
   - Private, internal, or dunder attribute paths ('_', '__') and malformed paths are rejected.
   - Missing telemetry fields are NEVER treated as zero; genuine 0.0 values pass, missing fields demote.
   - Numeric and textual provenance values are strictly type-checked and validated.
   - Units are checked for physical compatibility against referenced telemetry fields.
   - Free-text claim statements are checked for numeric and semantic consistency with provenance.
   - Unsupported measurement domains (speed in km/h or mph, compass heading/direction, collisions/accidents)
     are detected using multi-pattern regex and semantic heuristics and demoted.

4. Unverified Claims:
   - Rationale explicitly indicates that supporting evidence is insufficient or unavailable.

5. Preservation of Provenance:
   - All valid citations, track IDs, keyframe IDs, and structured provenances are preserved intact.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.app.schemas.vlm import (
    ClaimCategoryEnum,
    EpistemicClaimSchema,
    EvidencePackageSchema,
    SceneInterpretationResponseSchema,
    TelemetryProvenanceSchema,
    VisualObservationSchema,
)

logger = logging.getLogger("trafficlens.vlm.grounding_verifier")


# ============================================================================
# Field Path Allowlists and Domain Constraints
# ============================================================================

ALLOWED_SUMMARY_SCALARS: Set[str] = {
    "total_unique_tracks",
    "total_detections",
    "fps",
    "duration_s",
    "frame_count",
    "total_frames",
    "congestion_status",
}

ALLOWED_SUMMARY_DICT_ROOTS: Set[str] = {
    "line_counts",
    "flow_vpm",
    "density_rois",
}

ALLOWED_EVENT_FIELDS: Set[str] = {
    "duration_s",
    "timestamp_s",
    "track_id",
    "event_type",
    "roi_id",
    "details.status",
    "details.duration_s",
    "details.stationary_frames",
}

EVENT_ONLY_FIELDS: Set[str] = {
    "event_id",
    "event_type",
    "roi_id",
}

UNIT_ALLOWLIST: Dict[str, Set[str]] = {
    "duration_s": {"s", "sec", "secs", "second", "seconds"},
    "timestamp_s": {"s", "sec", "secs", "second", "seconds"},
    "details.duration_s": {"s", "sec", "secs", "second", "seconds"},
    "fps": {"fps", "frames per second", "hz"},
    "total_unique_tracks": {"count", "tracks", "track_ids", "identities", "units", ""},
    "total_detections": {"count", "detections", "instances", "units", ""},
    "frame_count": {"count", "frames", "units", ""},
    "details.stationary_frames": {"count", "frames", "units", ""},
    "flow_vpm": {"vpm", "vehicles per minute", "flow", "count", ""},
    "line_counts": {"count", "vehicles", "crossings", "units", ""},
}


RESERVED_OR_DANGEROUS_DICT_KEYS: Set[str] = {
    "keys", "values", "items", "get", "pop", "clear", "update", "setdefault", "copy",
    "__class__", "__dict__", "__doc__", "__init__", "__mro__", "__subclasses__",
}

ALLOWED_LINE_NAME_PATTERN = re.compile(r"^(?:line|tripwire|sensor)_[a-zA-Z0-9_-]+$", re.IGNORECASE)
ALLOWED_ROI_NAME_PATTERN = re.compile(r"^(?:roi)_[a-zA-Z0-9_-]+$", re.IGNORECASE)
ALLOWED_FLOW_NAME_PATTERN = re.compile(r"^(?:line|roi|window|corridor)_[a-zA-Z0-9_-]+$", re.IGNORECASE)

ALLOWED_LINE_METRICS: Set[str] = {
    "total", "downstream", "upstream", "count", "in", "out",
}

SUPPORTED_PREDICATES: Dict[str, Dict[str, Any]] = {
    # 1. Stationary duration (Event-level metric)
    "stationary_duration": {
        "allowed_sources": {"event"},
        "allowed_fields": {"duration_s", "details.duration_s"},
        "unit_required": True,
        "allowed_units": {"s", "sec", "secs", "second", "seconds"},
        "template_singular": "{subject} remained stationary for 1 second.",
        "template_plural": "{subject} remained stationary for {value:g} seconds.",
    },
    "stopped_duration": {
        "allowed_sources": {"event"},
        "allowed_fields": {"duration_s", "details.duration_s"},
        "unit_required": True,
        "allowed_units": {"s", "sec", "secs", "second", "seconds"},
        "template_singular": "{subject} remained stationary for 1 second.",
        "template_plural": "{subject} remained stationary for {value:g} seconds.",
    },
    # 2. Stationary frames (Event-level frame counter)
    "stationary_frames": {
        "allowed_sources": {"event"},
        "allowed_fields": {"details.stationary_frames"},
        "unit_required": False,
        "allowed_units": {"frames", "count", "units", ""},
        "template_singular": "{subject} remained stationary for 1 frame.",
        "template_plural": "{subject} remained stationary for {int_value} frames.",
    },
    # 3. Unique tracker identities (Tracker metric - NOT physical vehicle census)
    "total_unique_tracks": {
        "allowed_sources": {"telemetry_summary"},
        "allowed_fields": {"total_unique_tracks"},
        "unit_required": False,
        "allowed_units": {"tracks", "track_ids", "identities", "units", "count", ""},
        "template_singular": "{subject} recorded 1 unique tracker identity.",
        "template_plural": "{subject} recorded {int_value} unique tracker identities.",
    },
    "track_count": {
        "allowed_sources": {"telemetry_summary"},
        "allowed_fields": {"total_unique_tracks"},
        "unit_required": False,
        "allowed_units": {"tracks", "track_ids", "identities", "units", "count", ""},
        "template_singular": "{subject} recorded 1 unique tracker identity.",
        "template_plural": "{subject} recorded {int_value} unique tracker identities.",
    },
    # 4. Total detections (Bounding box detections across frames - NOT tracks)
    "total_detections": {
        "allowed_sources": {"telemetry_summary"},
        "allowed_fields": {"total_detections"},
        "unit_required": False,
        "allowed_units": {"detections", "instances", "units", "count", ""},
        "template_singular": "{subject} recorded 1 total bounding box detection.",
        "template_plural": "{subject} recorded {int_value} total bounding box detections.",
    },
    "detection_count": {
        "allowed_sources": {"telemetry_summary"},
        "allowed_fields": {"total_detections"},
        "unit_required": False,
        "allowed_units": {"detections", "instances", "units", "count", ""},
        "template_singular": "{subject} recorded 1 total bounding box detection.",
        "template_plural": "{subject} recorded {int_value} total bounding box detections.",
    },
    # 5. Sequence duration (Sequence-level time length)
    "duration_s": {
        "allowed_sources": {"telemetry_summary"},
        "allowed_fields": {"duration_s"},
        "unit_required": True,
        "allowed_units": {"s", "sec", "secs", "second", "seconds"},
        "template_singular": "{subject} duration is 1 second.",
        "template_plural": "{subject} duration is {value:g} seconds.",
    },
    "sequence_duration": {
        "allowed_sources": {"telemetry_summary"},
        "allowed_fields": {"duration_s"},
        "unit_required": True,
        "allowed_units": {"s", "sec", "secs", "second", "seconds"},
        "template_singular": "{subject} duration is 1 second.",
        "template_plural": "{subject} duration is {value:g} seconds.",
    },
    # 6. FPS (Frame rate)
    "fps": {
        "allowed_sources": {"telemetry_summary"},
        "allowed_fields": {"fps"},
        "unit_required": True,
        "allowed_units": {"fps", "frames per second", "hz"},
        "template": "{subject} frame rate is {value:g} FPS.",
    },
    "frame_rate": {
        "allowed_sources": {"telemetry_summary"},
        "allowed_fields": {"fps"},
        "unit_required": True,
        "allowed_units": {"fps", "frames per second", "hz"},
        "template": "{subject} frame rate is {value:g} FPS.",
    },
    # 7. Total frames (Video container frame count)
    "total_frames": {
        "allowed_sources": {"telemetry_summary"},
        "allowed_fields": {"total_frames", "frame_count"},
        "unit_required": False,
        "allowed_units": {"frames", "units", "count", ""},
        "template_singular": "{subject} contains 1 total frame.",
        "template_plural": "{subject} contains {int_value} total frames.",
    },
    "frame_count": {
        "allowed_sources": {"telemetry_summary"},
        "allowed_fields": {"total_frames", "frame_count"},
        "unit_required": False,
        "allowed_units": {"frames", "units", "count", ""},
        "template_singular": "{subject} contains 1 total frame.",
        "template_plural": "{subject} contains {int_value} total frames.",
    },
    # 8. Tripwire line crossings
    "line_count": {
        "allowed_sources": {"telemetry_summary"},
        "allowed_fields": None,  # dynamic check against line_counts.*
        "unit_required": False,
        "allowed_units": {"crossings", "vehicles", "units", "count", ""},
        "template_singular": "{subject} recorded 1 vehicle crossing.",
        "template_plural": "{subject} recorded {int_value} vehicle crossings.",
    },
    "crossing_count": {
        "allowed_sources": {"telemetry_summary"},
        "allowed_fields": None,
        "unit_required": False,
        "allowed_units": {"crossings", "vehicles", "units", "count", ""},
        "template_singular": "{subject} recorded 1 vehicle crossing.",
        "template_plural": "{subject} recorded {int_value} vehicle crossings.",
    },
    # 9. Congestion status
    "congestion_status": {
        "allowed_sources": {"telemetry_summary"},
        "allowed_fields": {"congestion_status"},
        "unit_required": False,
        "allowed_units": {"", None},
        "template": "{subject} congestion status is {text_value}.",
    },
    # 10. Event status
    "event_status": {
        "allowed_sources": {"event"},
        "allowed_fields": {"details.status", "event_type"},
        "unit_required": False,
        "allowed_units": {"", None},
        "template": "{subject} event status is {text_value}.",
    },
}


INCOMPATIBLE_UNITS_COMMON: Set[str] = {
    # Speed
    "km/h", "kph", "kmh", "mph", "m/s", "miles per hour", "miles/hour",
    # Weight
    "kg", "kilograms", "lbs", "pounds", "tons", "tonnes",
    # Distance
    "m", "meters", "metres", "km", "kilometers", "miles", "feet",
}

NUMBER_WORDS: Dict[str, float] = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "thirteen": 13.0,
    "fourteen": 14.0,
    "fifteen": 15.0,
    "sixteen": 16.0,
    "seventeen": 17.0,
    "eighteen": 18.0,
    "nineteen": 19.0,
    "twenty": 20.0,
    "thirty": 30.0,
    "forty": 40.0,
    "fifty": 50.0,
    "sixty": 60.0,
    "seventy": 70.0,
    "eighty": 80.0,
    "ninety": 90.0,
    "hundred": 100.0,
}

# Multi-pattern regex for unsupported CV measurement domains
UNSUPPORTED_DOMAIN_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Speed / Velocity / Acceleration
    (
        re.compile(r'\b\d+(?:\.\d+)?\s*(?:km/?h|kph|kmh|mph|miles?\s*(?:per|/)\s*hour|m/?s|met(?:er|re)s?\s*(?:per|/)\s*second)\b', re.IGNORECASE),
        "calibrated vehicle speed (km/h or mph)",
    ),
    (
        re.compile(r'\b(?:speed|speeding|velocity|velocities|accelerat\w*|decelerat\w*|cruising\s+speed)\b', re.IGNORECASE),
        "calibrated vehicle speed (km/h or mph)",
    ),
    (
        re.compile(r'\b(?:moving|traveling|travelling|cruising|driving|running)\s+at\s+(?:a\s+)?(?:\d+|rapid|fast|slow|moderate|high|low)\b', re.IGNORECASE),
        "calibrated vehicle speed (km/h or mph)",
    ),
    (
        re.compile(r'\b(?:rate\s+of\s+travel|rate\s+of\s+speed|radar\s+speed)\b', re.IGNORECASE),
        "calibrated vehicle speed (km/h or mph)",
    ),

    # Compass Heading / Trajectory / Direction
    (
        re.compile(r'\b(?:northbound|southbound|eastbound|westbound|northward|southward|eastward|westward)\b', re.IGNORECASE),
        "compass heading trajectory",
    ),
    (
        re.compile(r'\b(?:heading|bearing|traveling|moving|driving)\s+(?:due\s+)?(?:north|south|east|west)\b', re.IGNORECASE),
        "compass heading trajectory",
    ),
    (
        re.compile(r'\b(?:heading|bearing|compass)\b', re.IGNORECASE),
        "compass heading trajectory",
    ),
    (
        re.compile(r'\b\d+(?:\.\d+)?\s*(?:degrees?|deg|°)\b', re.IGNORECASE),
        "compass heading trajectory",
    ),
    (
        re.compile(r'\b(?:turning\s+left|turning\s+right|u-turn|lane\s+change|weaving)\b', re.IGNORECASE),
        "vehicle turning movement",
    ),

    # Collisions / Crashes / Incidents
    (
        re.compile(r'\b(?:collision|collide[ds]?|colliding|crash(?:es|ed|ing)?|accident|impact|struck|hit-and-run)\b', re.IGNORECASE),
        "collision or accident event",
    ),
    (
        re.compile(r'\b(?:rear-end(?:ed|ing)?|sideswipe[ds]?|side-swipe[ds]?|t-bone[ds]?|fender\s*bender|pileup|pile-up|wreck(?:age|ed)?)\b', re.IGNORECASE),
        "collision or accident event",
    ),
]


# ============================================================================
# Field Path Validation & Safe Resolution
# ============================================================================

def validate_and_resolve_field_path(
    source_type: str,
    source_obj: Any,
    path: str,
    record_id: Optional[str] = None,
) -> Tuple[bool, Any, Optional[str]]:
    """
    Safely resolves a dotted field path against approved CV telemetry structures.
    
    Security & Correctness Guarantees:
    - Rejects empty, whitespace, or malformed paths (consecutive dots, leading/trailing dots).
    - Prohibits access to private, internal, dunder, or reserved dict methods/attributes.
    - Enforces strict allowlists for both telemetry_summary and event sources.
    - Enforces subkey validation on nested dictionary roots (line_counts, flow_vpm, etc.)
      to prevent allowlisted roots from accessing arbitrary or non-metric keys.
    - Resolves 3-level paths only for approved line metrics (e.g. line_counts.line_1.total).
    - Prevents summary provenance from accessing event-only fields and vice-versa.
    - Missing fields are NEVER treated as zero; returns (False, None, error_message).
    - Genuine 0 or 0.0 values are cleanly preserved and returned as (True, 0.0, None).
    """
    if not isinstance(path, str) or not path.strip():
        return False, None, "Field path cannot be empty or non-string."

    clean_path = path.strip()

    # Reject malformed path patterns
    if clean_path.startswith(".") or clean_path.endswith(".") or ".." in clean_path:
        return False, None, f"Malformed field path '{clean_path}' contains empty segments."

    parts = clean_path.split(".")

    # Prohibit private, dunder, or reserved dict methods/keys
    for part in parts:
        if part.startswith("_") or "__" in part or part in RESERVED_OR_DANGEROUS_DICT_KEYS:
            return (
                False,
                None,
                f"Access to private, internal, or dunder attribute '{part}' in path '{clean_path}' is strictly prohibited.",
            )

    # Resolution for telemetry_summary
    if source_type == "telemetry_summary":
        if clean_path in EVENT_ONLY_FIELDS or (parts[0] in EVENT_ONLY_FIELDS):
            return (
                False,
                None,
                f"Field path '{clean_path}' is an event-only field and cannot be resolved against telemetry_summary.",
            )

        if not isinstance(source_obj, dict):
            return False, None, "Telemetry summary source object is missing or not a dictionary."

        if len(parts) == 1:
            field_name = parts[0]
            if field_name not in ALLOWED_SUMMARY_SCALARS:
                return (
                    False,
                    None,
                    f"CV fact references non-existent telemetry field '{clean_path}'.",
                )
            if field_name not in source_obj:
                return (
                    False,
                    None,
                    f"CV fact references non-existent telemetry field '{clean_path}'.",
                )
            val = source_obj[field_name]
            if val is None:
                return False, None, f"Telemetry summary field '{clean_path}' has null value."
            if isinstance(val, (dict, list)):
                return (
                    False,
                    None,
                    f"Telemetry summary field '{clean_path}' resolves to a collection, not a leaf scalar value.",
                )
            return True, val, None

        elif len(parts) == 2:
            root_dict, subkey = parts[0], parts[1]
            if root_dict not in ALLOWED_SUMMARY_DICT_ROOTS:
                return (
                    False,
                    None,
                    f"CV fact references non-existent telemetry field '{clean_path}'.",
                )
            if root_dict not in source_obj or not isinstance(source_obj[root_dict], dict):
                return (
                    False,
                    None,
                    f"Telemetry summary does not contain nested dictionary '{root_dict}'.",
                )
            nested_dict = source_obj[root_dict]

            # Validate subkey against allowlists / approved naming patterns
            if root_dict == "line_counts":
                if not ALLOWED_LINE_NAME_PATTERN.match(subkey):
                    return (
                        False,
                        None,
                        f"Subkey '{subkey}' in '{clean_path}' is not an approved line sensor identifier.",
                    )
            elif root_dict in {"density_rois", "peak_occupancy_per_roi", "average_occupancy_per_roi", "congestion_distribution_per_roi"}:
                if not ALLOWED_ROI_NAME_PATTERN.match(subkey):
                    return (
                        False,
                        None,
                        f"Subkey '{subkey}' in '{clean_path}' is not an approved ROI identifier.",
                    )
            elif root_dict == "flow_vpm":
                if not ALLOWED_FLOW_NAME_PATTERN.match(subkey):
                    return (
                        False,
                        None,
                        f"Subkey '{subkey}' in '{clean_path}' is not an approved flow identifier.",
                    )

            if subkey not in nested_dict:
                return (
                    False,
                    None,
                    f"CV fact references non-existent telemetry field '{clean_path}'.",
                )
            val = nested_dict[subkey]
            if val is None:
                return False, None, f"Telemetry summary field '{clean_path}' has null value."
            if isinstance(val, dict):
                return (
                    False,
                    None,
                    f"Field path '{clean_path}' resolves to a nested dictionary, not a leaf scalar value.",
                )
            return True, val, None

        elif len(parts) == 3:
            root_dict, subkey, subsubkey = parts[0], parts[1], parts[2]
            if root_dict != "line_counts":
                return (
                    False,
                    None,
                    f"Field path '{clean_path}' exceeds maximum permitted depth for root '{root_dict}'.",
                )
            if not ALLOWED_LINE_NAME_PATTERN.match(subkey):
                return (
                    False,
                    None,
                    f"Subkey '{subkey}' in '{clean_path}' is not an approved line sensor identifier.",
                )
            if root_dict not in source_obj or not isinstance(source_obj[root_dict], dict):
                return (
                    False,
                    None,
                    f"Telemetry summary does not contain nested dictionary '{root_dict}'.",
                )
            nested_dict = source_obj[root_dict]
            if subkey not in nested_dict:
                return (
                    False,
                    None,
                    f"CV fact references non-existent telemetry field '{clean_path}'.",
                )
            line_obj = nested_dict[subkey]
            if not isinstance(line_obj, dict):
                return (
                    False,
                    None,
                    f"Line '{subkey}' is a scalar and does not have nested metric '{subsubkey}'.",
                )
            if subsubkey not in ALLOWED_LINE_METRICS:
                return (
                    False,
                    None,
                    f"Subkey '{subsubkey}' in '{clean_path}' is not an approved line metric {sorted(ALLOWED_LINE_METRICS)}.",
                )
            if subsubkey not in line_obj:
                return (
                    False,
                    None,
                    f"Line '{subkey}' lacks metric '{subsubkey}'.",
                )
            val = line_obj[subsubkey]
            if val is None:
                return False, None, f"Field '{clean_path}' has null value."
            if isinstance(val, (dict, list)):
                return False, None, f"Field '{clean_path}' resolves to a collection, not a leaf scalar value."
            return True, val, None

        else:
            return False, None, f"Field path '{clean_path}' exceeds maximum permitted depth (3)."

    # Resolution for event records
    elif source_type == "event":
        rec_label = f"Event '{record_id}'" if record_id else "Event"

        if clean_path in ALLOWED_SUMMARY_SCALARS and clean_path not in ALLOWED_EVENT_FIELDS:
            return (
                False,
                None,
                f"Field path '{clean_path}' is a summary metric and cannot be resolved against an event record.",
            )

        if clean_path not in ALLOWED_EVENT_FIELDS:
            return False, None, f"{rec_label} lacks field '{clean_path}'."

        if len(parts) == 1:
            attr_name = parts[0]
            if not hasattr(source_obj, attr_name):
                return False, None, f"{rec_label} lacks field '{clean_path}'."
            val = getattr(source_obj, attr_name)
            if val is None:
                return False, None, f"{rec_label} field '{clean_path}' is null."
            return True, val, None

        elif len(parts) == 2 and parts[0] == "details":
            subkey = parts[1]
            if not hasattr(source_obj, "details") or not isinstance(source_obj.details, dict):
                return False, None, f"{rec_label} lacks details dictionary."
            if subkey not in source_obj.details:
                return False, None, f"{rec_label} lacks field '{clean_path}'."
            val = source_obj.details[subkey]
            if val is None:
                return False, None, f"{rec_label} details field '{clean_path}' is null."
            return True, val, None

        else:
            return False, None, f"{rec_label} lacks field '{clean_path}'."

    else:
        return False, None, f"Unknown telemetry source_type '{source_type}'."


# ============================================================================
# Numerical and Unit Verification Helpers
# ============================================================================

def check_numeric_match(
    asserted: float,
    actual: Any,
    rel_tol: float = 0.02,
    abs_tol: float = 0.05,
) -> bool:
    """
    Checks numeric equality within explicit tolerances (2% relative or 0.05 absolute).
    """
    try:
        actual_num = float(actual)
        return math.isclose(float(asserted), actual_num, rel_tol=rel_tol, abs_tol=abs_tol)
    except (ValueError, TypeError):
        return False


def check_unit_compatibility(field_path: str, unit: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Validates that asserted unit is physically and semantically compatible with the referenced field.
    """
    if unit is None or not unit.strip():
        return True, None

    clean_unit = unit.strip().lower()

    # Reject common incompatible units on duration/timestamp fields
    if ("duration" in field_path or "timestamp" in field_path) and clean_unit in INCOMPATIBLE_UNITS_COMMON:
        return (
            False,
            f"Unit '{unit}' is incompatible with temporal field '{field_path}'.",
        )

    # Check against allowlist if present
    base_field = field_path.split(".")[0] if field_path.startswith("flow_vpm") else field_path
    if base_field in UNIT_ALLOWLIST:
        allowed = UNIT_ALLOWLIST[base_field]
        if clean_unit not in allowed:
            return (
                False,
                f"Unit '{unit}' is incompatible with field '{field_path}' (expected one of {sorted(allowed)}).",
            )

    return True, None


def check_claim_content_consistency(
    statement: str,
    claim: EpistemicClaimSchema,
    prov: TelemetryProvenanceSchema,
    actual_value: Any,
) -> Tuple[bool, Optional[str]]:
    """
    Ensures that the free-text statement content strictly agrees with its structured provenance.
    
    Prevents adversarial attacks where provenance matches telemetry, but claim text asserts
    contradictory or fabricated quantities.
    """
    if prov.numeric_value is not None:
        asserted_num = float(prov.numeric_value)

        # Build list of permitted numbers in the statement:
        # - Track IDs associated with the claim
        # - Frame timestamps associated with the claim
        # - Any number close to the asserted numeric value
        permitted_numbers: List[float] = [float(t) for t in claim.associated_track_ids]
        permitted_numbers.extend([float(ts) for ts in claim.frame_timestamps])
        permitted_numbers.append(asserted_num)

        # 1. Extract freestanding numeric tokens from statement
        found_tokens = re.findall(r'(?<![a-zA-Z_])[-+]?(?:\d*\.\d+|\d+)', statement)
        for token in found_tokens:
            try:
                num_val = float(token)
                is_permitted = any(
                    math.isclose(num_val, p, rel_tol=0.02, abs_tol=0.05)
                    for p in permitted_numbers
                )
                if not is_permitted:
                    return (
                        False,
                        f"Claim statement asserts contradictory or unconfirmed numerical quantity {token} "
                        f"(provenance specifies {asserted_num}).",
                    )
            except ValueError:
                continue

        # 2. Check for written number words (e.g. 'fifteen seconds' vs 6.5)
        words = re.findall(r'\b[a-zA-Z]+\b', statement.lower())
        for word in words:
            if word in NUMBER_WORDS:
                word_val = NUMBER_WORDS[word]
                is_permitted = any(
                    math.isclose(word_val, p, rel_tol=0.02, abs_tol=0.05)
                    for p in permitted_numbers
                )
                if not is_permitted:
                    return (
                        False,
                        f"Claim statement asserts contradictory number word '{word}' ({word_val}) "
                        f"against provenance value {asserted_num}.",
                    )

        # 3. Check for attached incompatible units in statement
        if "duration" in prov.field_path:
            if re.search(r'\b\d+(?:\.\d+)?\s*(?:km/?h|kph|mph|m/?s|kg|lbs|tons?|meters?|metres?)\b', statement, re.IGNORECASE):
                return (
                    False,
                    f"Claim statement asserts incompatible units for duration field '{prov.field_path}'.",
                )

    if prov.text_value is not None:
        asserted_text = prov.text_value.strip().lower()
        actual_text = str(actual_value).strip().lower()
        if actual_text != asserted_text:
            return (
                False,
                f"Value mismatch for '{prov.field_path}'; asserted '{prov.text_value}', source telemetry is '{actual_value}'.",
            )

    return True, None


def detect_unsupported_domain_assertions(text: str) -> Tuple[bool, Optional[str]]:
    """
    Scans text for unsupported measurement domains (speed, velocity, heading, collisions).
    """
    for pattern, desc in UNSUPPORTED_DOMAIN_PATTERNS:
        if pattern.search(text):
            return True, f"Fabricated CV fact; CV pipeline does not measure {desc}."
    return False, None


# ============================================================================
# Master Grounding Verification Routine
# ============================================================================

def format_fact_subject(subject: Optional[str]) -> str:
    """Formats a structured subject into clean, authoritative text."""
    if not subject or not str(subject).strip():
        return "The entity"
    s = str(subject).strip()
    if s.isdigit():
        return f"Track {s}"
    s_lower = s.lower()
    if s_lower.startswith("track_"):
        tid = s.split("_", 1)[1]
        return f"Track {tid}"
    if s_lower.startswith("track "):
        tid = s.split(" ", 1)[1]
        return f"Track {tid}"
    if s_lower == "scene":
        return "The scene"
    if s_lower == "sequence":
        return "The sequence"
    if s_lower == "video":
        return "The video"
    if s_lower.startswith("line_"):
        lid = s.split("_", 1)[1]
        return f"Tripwire line {lid}"
    if s_lower.startswith("line "):
        lid = s.split(" ", 1)[1]
        return f"Tripwire line {lid}"
    if s_lower.startswith("roi_"):
        rid = s.split("_", 1)[1]
        return f"Region {rid}"
    return s[:1].upper() + s[1:]


def render_verified_cv_fact(
    prov: TelemetryProvenanceSchema,
    actual_value: Any,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Deterministically renders verified CV fact statements from validated structured fields.
    
    Guarantees:
    - Statement is generated strictly from mathematical CV truth and validated structured fields.
    - Model-generated free text is completely replaced with deterministic template text.
    - Handles singular vs plural grammar accurately.
    - Enforces accurate metric descriptions (tracker identities vs physical vehicles).
    - Fails and demotes if structured predicate or fields cannot be cleanly formatted.
    """
    if not prov.predicate or not str(prov.predicate).strip():
        return False, None, "Provenance lacks required non-empty 'predicate' for rendering."

    pred = str(prov.predicate).strip().lower()
    if pred not in SUPPORTED_PREDICATES:
        return False, None, f"Unknown or unsupported predicate '{prov.predicate}' cannot be rendered."

    subj = format_fact_subject(prov.subject)
    pred_def = SUPPORTED_PREDICATES[pred]

    if prov.numeric_value is not None:
        val = prov.numeric_value
        is_int_val = isinstance(val, int) or (isinstance(val, float) and val.is_integer())
        int_str = str(int(round(val)))
        float_str = f"{val:g}"

        # Determine singular vs plural
        is_singular = is_int_val and int(round(val)) == 1

        if is_singular and "template_singular" in pred_def:
            template = pred_def["template_singular"]
        elif "template_plural" in pred_def:
            template = pred_def["template_plural"]
        elif "template" in pred_def:
            template = pred_def["template"]
        else:
            return False, None, f"No rendering template configured for predicate '{pred}'."

        try:
            rendered = template.format(
                subject=subj,
                value=val,
                int_value=int_str if is_int_val else float_str,
                text_value=str(actual_value),
            )
            return True, rendered, None
        except Exception as e:
            return False, None, f"Rendering failed for predicate '{pred}': {str(e)}"

    elif prov.text_value is not None:
        template = pred_def.get("template", pred_def.get("template_plural", ""))
        if not template:
            return False, None, f"No text template configured for predicate '{pred}'."
        try:
            rendered = template.format(
                subject=subj,
                value=actual_value,
                int_value=str(actual_value),
                text_value=str(prov.text_value),
            )
            return True, rendered, None
        except Exception as e:
            return False, None, f"Rendering failed for predicate '{pred}': {str(e)}"

    return False, None, "Provenance has neither numeric_value nor text_value to render."


def verify_scene_interpretation_grounding(
    interpretation: SceneInterpretationResponseSchema,
    evidence: EvidencePackageSchema,
) -> SceneInterpretationResponseSchema:
    """
    Validates that all keyframe IDs, event IDs, visual observations, and CV facts
    in model outputs strictly correspond to genuine elements in the evidence package.
    """
    valid_keyframe_ids: Set[str] = {kf.keyframe_id for kf in evidence.keyframes}
    valid_event_ids: Set[str] = {evt.event_id for evt in evidence.events}
    events_by_id: Dict[str, Any] = {evt.event_id: evt for evt in evidence.events}

    # Determine verified track IDs from events and telemetry summary
    valid_track_ids: Set[int] = {evt.track_id for evt in evidence.events if evt.track_id is not None}
    if "total_unique_tracks" in evidence.telemetry_summary:
        total_tracks = evidence.telemetry_summary["total_unique_tracks"]
        if isinstance(total_tracks, int) and total_tracks > 0:
            valid_track_ids.update(range(1, total_tracks + 1))
    if "tracks" in evidence.telemetry_summary and isinstance(evidence.telemetry_summary["tracks"], list):
        valid_track_ids.update(t for t in evidence.telemetry_summary["tracks"] if isinstance(t, int))

    # 1. Ground Visual Observations (Rule 1 & Rule 3)
    verified_observations: List[VisualObservationSchema] = []
    for obs in interpretation.visual_observations:
        if not obs.keyframe_id:
            logger.warning(
                f"Visual observation '{obs.observation_id}' lacks a keyframe_id reference. "
                "Rejecting from visual observations list."
            )
            interpretation.unverified_aspects.append(
                f"Visual observation '{obs.observation_id}' ('{obs.description}') rejected from visual observations: "
                "no supporting keyframe ID cited (supporting evidence unavailable)."
            )
        elif obs.keyframe_id not in valid_keyframe_ids:
            logger.warning(
                f"Visual observation '{obs.observation_id}' referenced unknown keyframe '{obs.keyframe_id}'. "
                "Rejecting from visual observations list."
            )
            interpretation.unverified_aspects.append(
                f"Visual observation '{obs.observation_id}' ('{obs.description}') rejected from visual observations: "
                f"cited keyframe '{obs.keyframe_id}' is not in supplied evidence (supporting evidence unavailable)."
            )
        else:
            verified_observations.append(obs)
    interpretation.visual_observations = verified_observations

    # 2. Ground Epistemic Claims
    for claim in interpretation.claims:
        # 2a. Visual Observation Claims (Rule 1, 3, 4)
        if claim.category == ClaimCategoryEnum.VISUAL_OBSERVATION:
            if not claim.keyframe_ids:
                claim.category = ClaimCategoryEnum.UNVERIFIED
                claim.evidence_rationale = (
                    "[Unverified: Supporting visual evidence is insufficient or unavailable (no keyframes cited)] "
                    + claim.evidence_rationale
                )
                interpretation.unverified_aspects.append(
                    f"Visual observation claim '{claim.claim_id}' demoted to unverified: no supporting keyframes cited."
                )
            else:
                valid_ids = [kfid for kfid in claim.keyframe_ids if kfid in valid_keyframe_ids]
                invalid_ids = [kfid for kfid in claim.keyframe_ids if kfid not in valid_keyframe_ids]
                if invalid_ids:
                    if not valid_ids:
                        claim.category = ClaimCategoryEnum.UNVERIFIED
                        claim.keyframe_ids = []
                        claim.evidence_rationale = (
                            f"[Unverified: Supporting visual evidence is unavailable; cited keyframes {invalid_ids} not found in evidence] "
                            + claim.evidence_rationale
                        )
                        interpretation.unverified_aspects.append(
                            f"Visual observation claim '{claim.claim_id}' demoted to unverified: cited unknown keyframes {invalid_ids}."
                        )
                    else:
                        claim.keyframe_ids = valid_ids
                        interpretation.unverified_aspects.append(
                            f"Visual observation claim '{claim.claim_id}' cited unknown keyframes {invalid_ids}; "
                            f"pruned while preserving valid citations {valid_ids}."
                        )

        # 2b. Inference Claims (Rule 2, 4)
        elif claim.category == ClaimCategoryEnum.INFERENCE:
            has_evidence = bool(claim.keyframe_ids or claim.associated_track_ids)
            if not has_evidence:
                claim.category = ClaimCategoryEnum.UNVERIFIED
                claim.evidence_rationale = (
                    "[Unverified: Supporting evidence is insufficient or unavailable (no keyframes or track IDs cited)] "
                    + claim.evidence_rationale
                )
                interpretation.unverified_aspects.append(
                    f"Inference claim '{claim.claim_id}' demoted to unverified: no supporting keyframes or track IDs cited."
                )
            else:
                valid_kf = [kfid for kfid in claim.keyframe_ids if kfid in valid_keyframe_ids]
                invalid_kf = [kfid for kfid in claim.keyframe_ids if kfid not in valid_keyframe_ids]
                valid_tr = [tid for tid in claim.associated_track_ids if tid in valid_track_ids]
                invalid_tr = [tid for tid in claim.associated_track_ids if tid not in valid_track_ids]

                if (invalid_kf or invalid_tr) and (not valid_kf and not valid_tr):
                    claim.category = ClaimCategoryEnum.UNVERIFIED
                    claim.keyframe_ids = []
                    claim.associated_track_ids = []
                    claim.evidence_rationale = (
                        f"[Unverified: Supporting evidence is unavailable; cited evidence {invalid_kf + invalid_tr} not found in evidence] "
                        + claim.evidence_rationale
                    )
                    interpretation.unverified_aspects.append(
                        f"Inference claim '{claim.claim_id}' demoted to unverified: cited unknown evidence {invalid_kf + invalid_tr}."
                    )
                elif invalid_kf or invalid_tr:
                    claim.keyframe_ids = valid_kf
                    claim.associated_track_ids = valid_tr
                    interpretation.unverified_aspects.append(
                        f"Inference claim '{claim.claim_id}' cited unknown evidence {invalid_kf + invalid_tr}; "
                        "pruned while preserving valid citations."
                    )

        # 2c. CV Fact Claims (Rule 5 — Content Grounding & Provenance Integrity)
        elif claim.category == ClaimCategoryEnum.CV_FACT:
            is_demoted = False
            demote_reason = ""
            actual_val = None

            # Check 1: Verify cited track IDs
            if claim.associated_track_ids:
                invalid_tr = [tid for tid in claim.associated_track_ids if tid not in valid_track_ids]
                if invalid_tr:
                    is_demoted = True
                    demote_reason = f"Fabricated or unconfirmed CV fact; track IDs {invalid_tr} not found in CV telemetry."
                    claim.associated_track_ids = [tid for tid in claim.associated_track_ids if tid in valid_track_ids]

            # Check 2: Detect asserted measurements in free text not supported by CV telemetry
            if not is_demoted:
                claim_text = f"{claim.statement} {claim.evidence_rationale}"
                is_unsupported, reason_unsupported = detect_unsupported_domain_assertions(claim_text)
                if is_unsupported and reason_unsupported:
                    is_demoted = True
                    demote_reason = reason_unsupported

            # Check 3: Mandatory structured telemetry provenance
            if not is_demoted and claim.telemetry_provenance is None:
                is_demoted = True
                demote_reason = "CV fact claim lacks required structured telemetry provenance (telemetry_provenance is None)."

            # Check 4: Type-check and inspect basic structured telemetry provenance fields
            if not is_demoted and claim.telemetry_provenance is not None:
                prov = claim.telemetry_provenance

                # Type check numeric_value
                if prov.numeric_value is not None:
                    if not isinstance(prov.numeric_value, (int, float)) or isinstance(prov.numeric_value, bool) or math.isnan(prov.numeric_value) or math.isinf(prov.numeric_value):
                        is_demoted = True
                        demote_reason = f"Invalid numeric_value type or value '{prov.numeric_value}' in provenance."

                # Type check text_value
                if not is_demoted and prov.text_value is not None and not isinstance(prov.text_value, str):
                    is_demoted = True
                    demote_reason = f"Invalid text_value type in provenance (expected str, got {type(prov.text_value).__name__})."

                # Check at least one value provided
                if not is_demoted and prov.numeric_value is None and prov.text_value is None:
                    is_demoted = True
                    demote_reason = "Provenance must provide at least one of numeric_value or text_value."

                # Verify unit compatibility with field path if unit is provided
                if not is_demoted and prov.unit and str(prov.unit).strip():
                    unit_ok, unit_err = check_unit_compatibility(prov.field_path, prov.unit)
                    if not unit_ok and unit_err:
                        is_demoted = True
                        demote_reason = unit_err

                # Source Type: telemetry_summary
                if not is_demoted and prov.source_type == "telemetry_summary":
                    if not evidence.telemetry_summary:
                        is_demoted = True
                        demote_reason = f"Telemetry summary is missing or empty; cannot verify field '{prov.field_path}'."
                    else:
                        ok, actual_val, err_msg = validate_and_resolve_field_path(
                            "telemetry_summary",
                            evidence.telemetry_summary,
                            prov.field_path,
                        )
                        if not ok:
                            is_demoted = True
                            demote_reason = err_msg or f"CV fact references non-existent telemetry field '{prov.field_path}'."
                        else:
                            if prov.numeric_value is not None:
                                if not check_numeric_match(prov.numeric_value, actual_val):
                                    is_demoted = True
                                    demote_reason = (
                                        f"Numeric mismatch for '{prov.field_path}'; asserted {prov.numeric_value}, "
                                        f"source telemetry is {actual_val}."
                                    )
                            if not is_demoted and prov.text_value is not None:
                                if str(actual_val).strip().lower() != prov.text_value.strip().lower():
                                    is_demoted = True
                                    demote_reason = (
                                        f"Value mismatch for '{prov.field_path}'; asserted '{prov.text_value}', "
                                        f"source telemetry is '{actual_val}'."
                                    )
                            if not is_demoted:
                                consistent, consistency_err = check_claim_content_consistency(
                                    claim.statement,
                                    claim,
                                    prov,
                                    actual_val,
                                )
                                if not consistent and consistency_err:
                                    is_demoted = True
                                    demote_reason = consistency_err

                # Source Type: event
                elif not is_demoted and prov.source_type == "event":
                    if not prov.record_id or not prov.record_id.strip():
                        is_demoted = True
                        demote_reason = "Event provenance requires a non-empty record_id."
                    elif prov.record_id not in events_by_id:
                        is_demoted = True
                        demote_reason = f"Referenced event '{prov.record_id}' not found in CV event telemetry."
                    else:
                        matched_evt = events_by_id[prov.record_id]
                        if claim.associated_track_ids and matched_evt.track_id is not None and matched_evt.track_id not in claim.associated_track_ids:
                            is_demoted = True
                            demote_reason = (
                                f"Event track ID {matched_evt.track_id} does not match claim track IDs {claim.associated_track_ids}."
                            )
                        if not is_demoted:
                            ok, actual_val, err_msg = validate_and_resolve_field_path(
                                "event",
                                matched_evt,
                                prov.field_path,
                                record_id=prov.record_id,
                            )
                            if not ok:
                                is_demoted = True
                                demote_reason = err_msg or f"Event '{prov.record_id}' lacks field '{prov.field_path}'."
                            else:
                                if prov.numeric_value is not None:
                                    if not check_numeric_match(prov.numeric_value, actual_val):
                                        is_demoted = True
                                        demote_reason = (
                                            f"Numeric mismatch for event '{prov.record_id}' field '{prov.field_path}'; "
                                            f"asserted {prov.numeric_value}, source telemetry is {actual_val}."
                                        )
                                if not is_demoted and prov.text_value is not None:
                                    if str(actual_val).strip().lower() != prov.text_value.strip().lower():
                                        is_demoted = True
                                        demote_reason = (
                                            f"Value mismatch for event '{prov.record_id}' field '{prov.field_path}'; "
                                            f"asserted '{prov.text_value}', source telemetry is '{actual_val}'."
                                        )
                                if not is_demoted:
                                    consistent, consistency_err = check_claim_content_consistency(
                                        claim.statement,
                                        claim,
                                        prov,
                                        actual_val,
                                    )
                                    if not consistent and consistency_err:
                                        is_demoted = True
                                        demote_reason = consistency_err

                elif not is_demoted:
                    is_demoted = True
                    demote_reason = f"Unknown telemetry source_type '{prov.source_type}'."

                # Check 5: Structured CV Fact Requirements (Subject, Predicate, Unit)
                if not is_demoted:
                    # Check 5a: Required Subject
                    if not prov.subject or not str(prov.subject).strip():
                        is_demoted = True
                        demote_reason = "CV fact claim lacks required non-empty 'subject' in telemetry_provenance."

                    # Check 5b: Required & Supported Predicate
                    if not is_demoted:
                        if not prov.predicate or not str(prov.predicate).strip():
                            is_demoted = True
                            demote_reason = "CV fact claim lacks required non-empty 'predicate' in telemetry_provenance."
                        else:
                            clean_pred = str(prov.predicate).strip().lower()
                            if clean_pred not in SUPPORTED_PREDICATES:
                                is_demoted = True
                                demote_reason = f"Unknown or unsupported CV fact predicate '{prov.predicate}'."
                            else:
                                pred_def = SUPPORTED_PREDICATES[clean_pred]
                                # Source compatibility
                                if prov.source_type not in pred_def["allowed_sources"]:
                                    is_demoted = True
                                    demote_reason = f"Predicate '{prov.predicate}' is incompatible with source_type '{prov.source_type}'."
                                # Field path compatibility
                                elif pred_def["allowed_fields"] is not None and prov.field_path not in pred_def["allowed_fields"]:
                                    is_demoted = True
                                    demote_reason = f"Predicate '{prov.predicate}' is incompatible with field_path '{prov.field_path}'."
                                elif pred_def["allowed_fields"] is None and clean_pred in {"line_count", "crossing_count"} and not prov.field_path.startswith("line_counts."):
                                    is_demoted = True
                                    demote_reason = f"Predicate '{prov.predicate}' requires a line_counts field_path (got '{prov.field_path}')."

                    # Check 5c: Unit requirement & compatibility with predicate
                    if not is_demoted:
                        clean_pred = str(prov.predicate).strip().lower()
                        pred_def = SUPPORTED_PREDICATES[clean_pred]
                        clean_unit = prov.unit.strip().lower() if prov.unit and str(prov.unit).strip() else None

                        if pred_def["unit_required"]:
                            if not clean_unit:
                                is_demoted = True
                                demote_reason = f"CV fact predicate '{prov.predicate}' requires a valid measurement unit."
                            elif clean_unit not in pred_def["allowed_units"]:
                                is_demoted = True
                                demote_reason = f"Unit '{prov.unit}' is incompatible with predicate '{prov.predicate}'."
                        elif clean_unit:
                            if clean_unit not in pred_def["allowed_units"]:
                                is_demoted = True
                                demote_reason = f"Unit '{prov.unit}' is incompatible with predicate '{prov.predicate}'."

                # Check 6: Deterministic Verified Rendering Safeguard
                if not is_demoted:
                    render_ok, rendered_statement, render_err = render_verified_cv_fact(prov, actual_val)
                    if not render_ok or not rendered_statement:
                        is_demoted = True
                        demote_reason = render_err or "Failed to deterministically render verified CV fact."
                    else:
                        claim.statement = rendered_statement

            if is_demoted:
                claim.category = ClaimCategoryEnum.UNVERIFIED
                clean_reason = demote_reason.rstrip(".")
                claim.evidence_rationale = f"[Unverified: {clean_reason}] {claim.evidence_rationale}"
                interpretation.unverified_aspects.append(
                    f"CV fact claim '{claim.claim_id}' demoted to unverified: {clean_reason}."
                )

        # 2d. Unverified Claims (Rule 6)
        elif claim.category == ClaimCategoryEnum.UNVERIFIED:
            indicator = "insufficient or unavailable"
            if indicator not in claim.evidence_rationale.lower() and "unverified" not in claim.evidence_rationale.lower():
                claim.evidence_rationale = (
                    "[Unverified: Supporting evidence is insufficient or unavailable] "
                    + claim.evidence_rationale
                )

    # 3. Ground Event Interpretations
    for evt_interp in interpretation.event_interpretations:
        if evt_interp.event_id not in valid_event_ids:
            interpretation.unverified_aspects.append(
                f"Event interpretation cited unknown event_id '{evt_interp.event_id}'."
            )

    return interpretation
