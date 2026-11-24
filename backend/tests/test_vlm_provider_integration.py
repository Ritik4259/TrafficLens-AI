"""
Unit and integration tests for VLM Provider Interface and Remote Adapter (Milestone 5C).

Validates:
- Provider protocol conformance
- Mock provider deterministic grounded output
- OpenAI-compatible remote adapter request assembly and response parsing
- Image data URI and content-type encoding
- Bounded retries for timeouts, rate limits (429), and 5xx errors
- Fast failure for authentication errors (401/403)
- Malformed JSON and schema validation error handling
- Ungrounded keyframe reference detection and pruning
- Service orchestration and failure isolation (never crashes caller)
- Secret redaction and header sanitization
- Provider health checks
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any
import httpx
import pytest

from backend.app.core.config import Settings
from backend.app.schemas.vlm import (
    EvidencePackageSchema,
    KeyframeDetailSchema,
    EventReferenceSchema,
    ClaimCategoryEnum,
    SceneInterpretationResponseSchema,
    SceneInterpretationResult,
    ProviderHealthSchema,
)
from backend.app.services.vlm.provider import (
    VLMProvider,
    VLMError,
    VLMConfigurationError,
    VLMConnectionError,
    VLMTimeoutError,
    VLMAuthenticationError,
    VLMRateLimitError,
    VLMResponseError,
    VLMOutputParsingError,
)
from backend.app.services.vlm.prompt_manager import PromptManager
from backend.app.services.vlm.gemini_provider import GeminiVLMProvider
from backend.app.services.vlm.remote_provider import OpenAICompatibleRemoteProvider
from backend.app.services.vlm.mock_provider import MockVLMProvider
from backend.app.services.vlm.scene_interpretation_service import SceneInterpretationService


# ============================================================================
# Test Fixtures & Helpers
# ============================================================================

@pytest.fixture
def sample_evidence_package() -> EvidencePackageSchema:
    """Creates a strongly-typed test evidence package with 2 keyframes and 1 event."""
    kf1 = KeyframeDetailSchema(
        keyframe_id="kf_test_10",
        source_video_id="vid_sample_01",
        source_job_id="job_sample_01",
        frame_idx=10,
        requested_frame_idx=10,
        is_verified=True,
        timestamp_s=0.40,
        selection_reason="stopped_vehicle_start_track_1",
        associated_event_ids=["stop_1_0.40"],
        width=640,
        height=360,
        image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        content_type="image/png",
    )
    kf2 = KeyframeDetailSchema(
        keyframe_id="kf_test_25",
        source_video_id="vid_sample_01",
        source_job_id="job_sample_01",
        frame_idx=25,
        requested_frame_idx=25,
        is_verified=True,
        timestamp_s=1.00,
        selection_reason="scene_anchor_pct_50",
        associated_event_ids=[],
        width=640,
        height=360,
        image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        content_type="image/jpeg",
    )
    evt = EventReferenceSchema(
        event_id="stop_1_0.40",
        event_type="stopped_vehicle",
        track_id=1,
        timestamp_s=0.40,
        duration_s=6.5,
        roi_id="roi_main",
        details={"status": "active"},
    )
    return EvidencePackageSchema(
        evidence_package_id="evpkg_test_01",
        job_id="job_sample_01",
        video_id="vid_sample_01",
        sequence_name="sequence_demo",
        keyframes=[kf1, kf2],
        events=[evt],
        telemetry_summary={
            "total_unique_tracks": 4,
            "total_detections": 120,
            "fps": 25.0,
        },
    )


def build_mock_gemini_completion_body(
    content_dict: Dict[str, Any],
    finish_reason: str = "STOP",
    prompt_tokens: int = 420,
    completion_tokens: int = 280,
) -> Dict[str, Any]:
    """Helper formatting a raw model dict as a Google Gemini generateContent REST JSON response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(content_dict, indent=2)
                        }
                    ],
                    "role": "model",
                },
                "finishReason": finish_reason,
                "index": 0,
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                    }
                ],
            }
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": completion_tokens,
            "totalTokenCount": prompt_tokens + completion_tokens,
        },
    }


def build_mock_openai_completion_body(content_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Helper formatting a raw model dict as an OpenAI chat completion JSON response."""
    return {
        "id": "chatcmpl-test-12345",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"```json\n{json.dumps(content_dict, indent=2)}\n```",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 420,
            "completion_tokens": 280,
            "total_tokens": 700,
        },
    }


def get_valid_completion_content() -> Dict[str, Any]:
    """Provides a valid dictionary conforming to SceneInterpretationResponseSchema."""
    return {
        "scene_summary": "Active urban arterial road observed during overcast daylight conditions.",
        "environmental_conditions": {
            "weather": "overcast",
            "illumination": "daylight",
            "road_surface": "dry",
            "visibility_hazards": [],
            "confidence": "direct_observation",
        },
        "roadway_description": "Three-lane roadway with dashed white lane dividers and elevated curb.",
        "visual_observations": [
            {
                "observation_id": "obs_1",
                "keyframe_id": "kf_test_10",
                "frame_timestamp_s": 0.40,
                "description": "Vehicle stationary in right-most lane with rear brake lights illuminated.",
                "visual_cues": ["brake lights illuminated"],
            }
        ],
        "event_interpretations": [
            {
                "event_id": "stop_1_0.40",
                "track_id": 1,
                "start_timestamp_s": 0.40,
                "cv_measured_duration_s": 6.5,
                "probable_cause": "Traffic queue congestion or intersection signal stop.",
                "contextual_explanation": "Vehicle is positioned behind ahead queue without emergency flashers.",
                "confidence": "supported_inference",
                "supporting_observations": ["obs_1"],
            }
        ],
        "claims": [
            {
                "claim_id": "claim_1",
                "category": "visual_observation",
                "statement": "Brake lights are active on the stationary vehicle.",
                "keyframe_ids": ["kf_test_10"],
                "associated_track_ids": [1],
                "evidence_rationale": "Directly observed in keyframe kf_test_10.",
            },
            {
                "claim_id": "claim_2",
                "category": "cv_fact",
                "statement": "Vehicle track 1 remained stationary for 6.5 seconds.",
                "keyframe_ids": [],
                "associated_track_ids": [1],
                "evidence_rationale": "Measured directly by ByteTrack spatial duration counter.",
                "telemetry_provenance": {
                    "source_type": "event",
                    "record_id": "stop_1_0.40",
                    "field_path": "duration_s",
                    "numeric_value": 6.5,
                    "subject": "track_1",
                    "predicate": "stationary_duration",
                    "unit": "seconds",
                },
            },
        ],
        "unverified_aspects": [
            "Downstream traffic light phase is out of camera view."
        ],
    }


# ============================================================================
# 1. Provider Protocol Conformance & Mock Provider Tests
# ============================================================================

def test_provider_interface_protocol_conformance():
    """Verifies that Gemini, Mock, and Remote providers strictly satisfy the VLMProvider Protocol."""
    mock_p = MockVLMProvider()
    gemini_p = GeminiVLMProvider(api_key="test-key", model_name="gemini-3.8-flash")
    remote_p = OpenAICompatibleRemoteProvider(api_base_url="http://localhost:8000/v1")

    assert isinstance(mock_p, VLMProvider)
    assert isinstance(gemini_p, VLMProvider)
    assert isinstance(remote_p, VLMProvider)
    assert mock_p.provider_type == "mock"
    assert gemini_p.provider_type == "gemini"
    assert gemini_p.model_name == "gemini-3.8-flash"
    assert remote_p.provider_type == "openai_compatible"


@pytest.mark.asyncio
async def test_mock_provider_generates_grounded_scene_understanding(sample_evidence_package: EvidencePackageSchema):
    """Verifies that MockVLMProvider generates valid, grounded interpretations referencing real keyframes."""
    provider = MockVLMProvider(model_name="mock-qwen-test")
    res = await provider.generate_scene_understanding(sample_evidence_package)

    assert isinstance(res, SceneInterpretationResponseSchema)
    assert res.job_id == sample_evidence_package.job_id
    assert res.video_id == sample_evidence_package.video_id
    assert res.model_metadata.provider_type == "mock"
    assert len(res.visual_observations) > 0
    # Must reference the genuine keyframe ID
    assert res.visual_observations[0].keyframe_id == "kf_test_10"
    # Claims must span epistemic categories
    categories = {c.category for c in res.claims}
    assert ClaimCategoryEnum.CV_FACT in categories
    assert ClaimCategoryEnum.VISUAL_OBSERVATION in categories
    assert ClaimCategoryEnum.INFERENCE in categories
    assert ClaimCategoryEnum.UNVERIFIED in categories


@pytest.mark.asyncio
async def test_mock_provider_health_check():
    """Verifies MockVLMProvider health check response."""
    provider = MockVLMProvider(is_healthy=True)
    health = await provider.health_check()
    assert isinstance(health, ProviderHealthSchema)
    assert health.is_healthy is True

    unhealthy_provider = MockVLMProvider(is_healthy=False)
    unhealthy = await unhealthy_provider.health_check()
    assert unhealthy.is_healthy is False


# ============================================================================
# 2. Remote Provider Request Assembly & Image Encoding Tests
# ============================================================================

def test_prompt_manager_assembles_openai_vision_messages(sample_evidence_package: EvidencePackageSchema):
    """Verifies that PromptManager formats prompt with keyframe data URIs and system rules."""
    messages = PromptManager.assemble_openai_messages(sample_evidence_package)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "DO NOT ESTIMATE OR INVENT vehicle counts" in messages[0]["content"]

    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    # Check that text prompt and image data URIs are present
    image_items = [item for item in user_content if item.get("type") == "image_url"]
    assert len(image_items) == 2
    # Verify exact data URI encoding with content type
    assert image_items[0]["image_url"]["url"].startswith("data:image/png;base64,iVBORw0KGgo")
    assert image_items[1]["image_url"]["url"].startswith("data:image/jpeg;base64,iVBORw0KGgo")

    # Check keyframe ID banners in preceding text blocks
    text_items = [item["text"] for item in user_content if item.get("type") == "text"]
    assert any("kf_test_10" in t for t in text_items)
    assert any("kf_test_25" in t for t in text_items)


@pytest.mark.asyncio
async def test_remote_provider_successful_request_and_response_parsing(sample_evidence_package: EvidencePackageSchema):
    """
    Verifies that OpenAICompatibleRemoteProvider sends standard request and correctly parses
    markdown-fenced JSON responses.
    """
    captured_requests = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        if request.url.path == "/v1/chat/completions":
            body = build_mock_openai_completion_body(get_valid_completion_content())
            return httpx.Response(200, json=body)
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = OpenAICompatibleRemoteProvider(
        api_base_url="https://api.vlm-inference.test/v1",
        api_key="secret-api-key-999",
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        http_client=client,
    )

    res = await provider.generate_scene_understanding(sample_evidence_package)

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.headers["Authorization"] == "Bearer secret-api-key-999"
    assert req.headers["Content-Type"] == "application/json"

    assert isinstance(res, SceneInterpretationResponseSchema)
    assert res.job_id == "job_sample_01"
    assert res.environmental_conditions.weather == "overcast"
    assert len(res.claims) == 2
    assert res.model_metadata.prompt_tokens == 420
    assert res.model_metadata.completion_tokens == 280


# ============================================================================
# 3. Timeout, Retry, and Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_remote_provider_retries_on_transient_500_and_succeeds(sample_evidence_package: EvidencePackageSchema):
    """Verifies that transient 500 errors trigger bounded retries before succeeding."""
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(502, text="Bad Gateway")
        body = build_mock_openai_completion_body(get_valid_completion_content())
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = OpenAICompatibleRemoteProvider(
        api_base_url="https://api.vlm-inference.test/v1",
        max_retries=2,
        http_client=client,
    )

    res = await provider.generate_scene_understanding(sample_evidence_package)
    assert call_count == 2
    assert res.job_id == "job_sample_01"


@pytest.mark.asyncio
async def test_remote_provider_retries_on_rate_limit_429(sample_evidence_package: EvidencePackageSchema):
    """Verifies that HTTP 429 triggers retry and succeeds on subsequent attempt."""
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, text="Too Many Requests")
        body = build_mock_openai_completion_body(get_valid_completion_content())
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = OpenAICompatibleRemoteProvider(
        api_base_url="https://api.vlm-inference.test/v1",
        max_retries=1,
        http_client=client,
    )

    res = await provider.generate_scene_understanding(sample_evidence_package)
    assert call_count == 2
    assert res.job_id == "job_sample_01"


@pytest.mark.asyncio
async def test_remote_provider_exhausted_retries_raises_vlm_response_error(sample_evidence_package: EvidencePackageSchema):
    """Verifies that persistent 5xx errors raise VLMResponseError after exhausting retries."""
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(503, text="Service Unavailable")

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = OpenAICompatibleRemoteProvider(
        api_base_url="https://api.vlm-inference.test/v1",
        max_retries=2,
        http_client=client,
    )

    with pytest.raises(VLMResponseError) as exc_info:
        await provider.generate_scene_understanding(sample_evidence_package)

    assert exc_info.value.status_code == 503
    assert call_count == 3  # Initial + 2 retries


@pytest.mark.asyncio
async def test_remote_provider_authentication_error_fails_immediately(sample_evidence_package: EvidencePackageSchema):
    """Verifies that HTTP 401 fails immediately without retrying."""
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(401, text="Unauthorized: Invalid API key")

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = OpenAICompatibleRemoteProvider(
        api_base_url="https://api.vlm-inference.test/v1",
        max_retries=3,
        http_client=client,
    )

    with pytest.raises(VLMAuthenticationError, match="credentials rejected"):
        await provider.generate_scene_understanding(sample_evidence_package)

    assert call_count == 1  # No retries on auth failure


@pytest.mark.asyncio
async def test_remote_provider_timeout_raises_vlm_timeout_error(sample_evidence_package: EvidencePackageSchema):
    """Verifies that request timeouts raise VLMTimeoutError."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read timed out")

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = OpenAICompatibleRemoteProvider(
        api_base_url="https://api.vlm-inference.test/v1",
        timeout_s=5.0,
        max_retries=1,
        http_client=client,
    )

    with pytest.raises(VLMTimeoutError, match="timed out"):
        await provider.generate_scene_understanding(sample_evidence_package)


# ============================================================================
# 4. JSON Parsing, Schema Validation & Grounding Enforcement Tests
# ============================================================================

@pytest.mark.asyncio
async def test_malformed_json_raises_output_parsing_error(sample_evidence_package: EvidencePackageSchema):
    """Verifies that non-JSON output raises VLMOutputParsingError."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        body = {
            "choices": [{"message": {"content": "I am unable to analyze this video because of clouds."}}],
            "usage": {},
        }
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = OpenAICompatibleRemoteProvider(
        api_base_url="https://api.vlm-inference.test/v1",
        http_client=client,
    )

    with pytest.raises(VLMOutputParsingError, match="did not contain a parseable JSON object"):
        await provider.generate_scene_understanding(sample_evidence_package)


@pytest.mark.asyncio
async def test_schema_validation_error_on_missing_required_fields(sample_evidence_package: EvidencePackageSchema):
    """Verifies that valid JSON missing required schema properties raises VLMOutputParsingError."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        # Missing environmental_conditions and roadway_description
        body = build_mock_openai_completion_body({"scene_summary": "Incomplete output."})
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = OpenAICompatibleRemoteProvider(
        api_base_url="https://api.vlm-inference.test/v1",
        http_client=client,
    )

    with pytest.raises(VLMOutputParsingError, match="failed validation against SceneInterpretationResponseSchema"):
        await provider.generate_scene_understanding(sample_evidence_package)


@pytest.mark.asyncio
async def test_ungrounded_keyframe_references_are_sanitized_and_flagged(sample_evidence_package: EvidencePackageSchema):
    """
    Verifies that if a model outputs an observation or claim citing a non-existent keyframe ID,
    the pipeline unlinks it, converts visual claims to unverified, and records disclosures in unverified_aspects.
    """
    valid_content = get_valid_completion_content()
    # Inject fabricated keyframe reference
    valid_content["visual_observations"][0]["keyframe_id"] = "kf_hallucinated_999"
    valid_content["claims"][0]["keyframe_ids"] = ["kf_hallucinated_999"]

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=build_mock_openai_completion_body(valid_content))

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = OpenAICompatibleRemoteProvider(
        api_base_url="https://api.vlm-inference.test/v1",
        http_client=client,
    )

    res = await provider.generate_scene_understanding(sample_evidence_package)

    # Observation with ungrounded keyframe must be rejected from visual_observations
    assert len(res.visual_observations) == 0
    # Claim must be demoted to UNVERIFIED since its cited keyframe was non-existent
    assert res.claims[0].category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified:" in res.claims[0].evidence_rationale
    # Unverified aspects must disclose the ungrounded citation
    assert any("kf_hallucinated_999" in aspect for aspect in res.unverified_aspects)


# ============================================================================
# 5. Service Orchestration & Failure Isolation Tests
# ============================================================================

@pytest.mark.asyncio
async def test_service_disabled_behavior(sample_evidence_package: EvidencePackageSchema):
    """Verifies that when VLM_ENABLED=False, service returns safe failure descriptor without calling provider."""
    settings = Settings(VLM_ENABLED=False, VLM_PROVIDER_TYPE="openai_compatible")
    service = SceneInterpretationService(settings=settings)

    result = await service.interpret_scene(sample_evidence_package)
    assert isinstance(result, SceneInterpretationResult)
    assert result.success is False
    assert result.interpretation is None
    assert result.failure is not None
    assert result.failure.error_type == "VLMDisabled"
    assert "disabled" in result.failure.message


@pytest.mark.asyncio
async def test_service_failure_isolation_from_cv_job(sample_evidence_package: EvidencePackageSchema):
    """
    Verifies that when the provider encounters a connection or 500 error,
    SceneInterpretationService catches it, returns success=False with failure details,
    and NEVER raises an unhandled exception that could crash the CV job.
    """
    failing_provider = MockVLMProvider(
        forced_exception=VLMConnectionError("Network drop during VLM call")
    )
    service = SceneInterpretationService(provider=failing_provider)

    result = await service.interpret_scene(sample_evidence_package)

    assert result.success is False
    assert result.interpretation is None
    assert result.failure is not None
    assert result.failure.error_type == "VLMConnectionError"
    assert "Network drop" in result.failure.message
    assert result.failure.is_retryable is True


@pytest.mark.asyncio
async def test_service_successful_interpretation(sample_evidence_package: EvidencePackageSchema):
    """Verifies successful end-to-end interpretation through SceneInterpretationService."""
    provider = MockVLMProvider()
    service = SceneInterpretationService(provider=provider)

    result = await service.interpret_scene(sample_evidence_package)

    assert result.success is True
    assert result.failure is None
    assert result.interpretation is not None
    assert result.interpretation.job_id == sample_evidence_package.job_id


# ============================================================================
# 6. Configuration & Secret Redaction Tests
# ============================================================================

def test_missing_base_url_raises_configuration_error(sample_evidence_package: EvidencePackageSchema):
    """Verifies that an unconfigured base URL raises VLMConfigurationError."""
    provider = OpenAICompatibleRemoteProvider(api_base_url=None)
    with pytest.raises(VLMConfigurationError, match="base URL is not configured"):
        asyncio.run(provider.generate_scene_understanding(sample_evidence_package))


def test_secret_redaction_in_settings_and_headers():
    """Verifies that API keys are strictly redacted in string conversions and headers."""
    settings = Settings(_env_file=None, VLM_API_KEY="sk-proj-supersecretkey123456789")
    redacted = settings.get_redacted_vlm_api_key()
    assert redacted is not None
    assert "supersecretkey" not in redacted
    assert redacted.startswith("sk***")

    provider = OpenAICompatibleRemoteProvider(
        api_base_url="https://api.test/v1",
        api_key="super-secret-key-xyz",
    )
    sanitized_headers = provider._get_sanitized_headers()
    assert sanitized_headers["Authorization"] == "Bearer [REDACTED]"
    assert "super-secret-key-xyz" not in str(sanitized_headers)


@pytest.mark.asyncio
async def test_remote_provider_health_check_connectivity():
    """Verifies remote provider health check against mocked models endpoint."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3-VL-2B-Instruct"}]})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = OpenAICompatibleRemoteProvider(
        api_base_url="https://api.test/v1",
        http_client=client,
    )

    health = await provider.health_check()
    assert health.is_healthy is True
    assert "models endpoint responded HTTP 200" in health.status_message
    assert health.latency_ms is not None


# ============================================================================
# 7. Google Gemini Native VLM Provider Tests (Milestone 5C)
# ============================================================================

def test_prompt_manager_assembles_gemini_parts(sample_evidence_package: EvidencePackageSchema):
    """Verifies that PromptManager creates Gemini native multimodal parts with inlineData and banners."""
    parts = PromptManager.assemble_gemini_parts(sample_evidence_package)
    assert len(parts) >= 3  # Text prompt + 2 * (banner + inlineData)

    # First part must be text instructions
    assert "text" in parts[0]
    assert "DO NOT ESTIMATE OR INVENT vehicle counts" in parts[0]["text"]
    assert "kf_test_10" in parts[0]["text"]

    # Subsequent parts must interleave text banners and inlineData
    inline_parts = [p["inlineData"] for p in parts if "inlineData" in p]
    assert len(inline_parts) == 2
    assert inline_parts[0]["mimeType"] == "image/png"
    assert inline_parts[0]["data"].startswith("iVBORw0KGgo")
    assert inline_parts[1]["mimeType"] == "image/jpeg"
    assert inline_parts[1]["data"].startswith("iVBORw0KGgo")

    # Keyframe banner text
    text_banners = [p["text"] for p in parts if "text" in p and "EVIDENCE KEYFRAME" in p["text"]]
    assert len(text_banners) == 2
    assert "kf_test_10" in text_banners[0]
    assert "kf_test_25" in text_banners[1]

    # Schema must be dereferenced and uppercase
    schema = PromptManager.get_gemini_response_schema()
    assert schema["type"] == "OBJECT"
    assert "$defs" not in schema
    assert "scene_summary" in schema["properties"]
    assert schema["properties"]["visual_observations"]["type"] == "ARRAY"
    assert schema["properties"]["visual_observations"]["items"]["type"] == "OBJECT"


@pytest.mark.asyncio
async def test_gemini_provider_successful_generate_content(sample_evidence_package: EvidencePackageSchema):
    """
    Verifies that GeminiVLMProvider sends native generateContent request with header auth
    and parses structured JSON responses.
    """
    captured_requests = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        if request.url.path == "/v1beta/models/gemini-3.8-flash:generateContent":
            body = build_mock_gemini_completion_body(get_valid_completion_content())
            return httpx.Response(200, json=body)
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = GeminiVLMProvider(
        api_key="secret-gemini-key-12345",
        model_name="gemini-3.8-flash",
        api_base_url="https://generativelanguage.googleapis.com/v1beta",
        http_client=client,
    )

    res = await provider.generate_scene_understanding(sample_evidence_package)

    assert len(captured_requests) == 1
    req = captured_requests[0]

    # URL path and host verification
    assert req.url.host == "generativelanguage.googleapis.com"
    assert req.url.path == "/v1beta/models/gemini-3.8-flash:generateContent"
    # Never leak key in query params
    assert "key=" not in str(req.url)

    # Header authentication verification
    assert req.headers["x-goog-api-key"] == "secret-gemini-key-12345"
    assert "Authorization" not in req.headers
    assert req.headers["Content-Type"] == "application/json"

    # Request body structure verification
    sent_payload = json.loads(req.content.decode("utf-8"))
    assert "contents" in sent_payload
    assert "generationConfig" in sent_payload
    assert sent_payload["generationConfig"]["responseMimeType"] == "application/json"
    assert "responseSchema" in sent_payload["generationConfig"]

    # Response validation
    assert isinstance(res, SceneInterpretationResponseSchema)
    assert res.job_id == "job_sample_01"
    assert res.video_id == "vid_sample_01"
    assert res.model_metadata.provider_type == "gemini"
    assert res.model_metadata.model_name == "gemini-3.8-flash"
    assert res.model_metadata.prompt_tokens == 420
    assert res.model_metadata.completion_tokens == 280
    assert len(res.claims) == 2


def test_gemini_provider_model_name_prefix_stripping():
    """Verifies that 'models/' prefix is cleanly stripped if provided in model identifier."""
    p1 = GeminiVLMProvider(api_key="key", model_name="models/gemini-3.8-pro")
    assert p1.model_name == "gemini-3.8-pro"

    p2 = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash")
    assert p2.model_name == "gemini-3.8-flash"


def test_gemini_provider_missing_config_raises_configuration_error(sample_evidence_package: EvidencePackageSchema):
    """Verifies that missing API key or model name raises VLMConfigurationError."""
    # Missing API Key
    p_no_key = GeminiVLMProvider(api_key=None, model_name="gemini-3.8-flash")
    with pytest.raises(VLMConfigurationError, match="GEMINI_API_KEY is not configured"):
        asyncio.run(p_no_key.generate_scene_understanding(sample_evidence_package))

    # Missing Model
    p_no_model = GeminiVLMProvider(api_key="key", model_name=None)
    with pytest.raises(VLMConfigurationError, match="GEMINI_MODEL is not configured"):
        asyncio.run(p_no_model.generate_scene_understanding(sample_evidence_package))


@pytest.mark.asyncio
async def test_gemini_provider_auth_error_401_403_fails_fast(sample_evidence_package: EvidencePackageSchema):
    """Verifies that HTTP 401 and 403 fail immediately without retrying."""
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(401, text="API_KEY_INVALID: The provided API key is invalid.")

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = GeminiVLMProvider(
        api_key="bad-key",
        model_name="gemini-3.8-flash",
        max_retries=2,
        http_client=client,
    )

    with pytest.raises(VLMAuthenticationError, match="authentication failed"):
        await provider.generate_scene_understanding(sample_evidence_package)

    assert call_count == 1  # Fast-fail without retries


@pytest.mark.asyncio
async def test_gemini_provider_model_not_found_404_fails_fast(sample_evidence_package: EvidencePackageSchema):
    """Verifies that HTTP 404 (model not found) fails immediately without retrying."""
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(404, text="models/invalid-model is not found")

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = GeminiVLMProvider(
        api_key="good-key",
        model_name="invalid-model",
        max_retries=2,
        http_client=client,
    )

    with pytest.raises(VLMResponseError) as exc_info:
        await provider.generate_scene_understanding(sample_evidence_package)

    assert exc_info.value.status_code == 404
    assert exc_info.value.is_retryable is False
    assert call_count == 1


@pytest.mark.asyncio
async def test_gemini_provider_retries_on_rate_limit_429(sample_evidence_package: EvidencePackageSchema):
    """Verifies that HTTP 429 triggers retry and succeeds on subsequent attempt."""
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, text="RESOURCE_EXHAUSTED: Rate limit exceeded")
        body = build_mock_gemini_completion_body(get_valid_completion_content())
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = GeminiVLMProvider(
        api_key="key",
        model_name="gemini-1.5-flash",
        max_retries=1,
        http_client=client,
    )

    res = await provider.generate_scene_understanding(sample_evidence_package)
    assert call_count == 2
    assert res.job_id == "job_sample_01"


@pytest.mark.asyncio
async def test_gemini_provider_retries_on_transient_500(sample_evidence_package: EvidencePackageSchema):
    """Verifies that HTTP 500 triggers retry and succeeds on subsequent attempt."""
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(503, text="INTERNAL: Service unavailable temporarily")
        body = build_mock_gemini_completion_body(get_valid_completion_content())
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    provider = GeminiVLMProvider(
        api_key="key",
        model_name="gemini-3.8-flash",
        max_retries=2,
        http_client=client,
    )

    res = await provider.generate_scene_understanding(sample_evidence_package)
    assert call_count == 2
    assert res.job_id == "job_sample_01"


@pytest.mark.asyncio
async def test_gemini_provider_safety_block_handling(sample_evidence_package: EvidencePackageSchema):
    """Verifies that prompt-level and candidate-level safety blocks raise non-retryable VLMResponseError."""
    # 1. Prompt Feedback block
    def mock_prompt_block(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    client1 = httpx.AsyncClient(transport=httpx.MockTransport(mock_prompt_block))
    p1 = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client1)

    with pytest.raises(VLMResponseError, match="blocked by safety filters: SAFETY") as exc1:
        await p1.generate_scene_understanding(sample_evidence_package)
    assert exc1.value.is_retryable is False

    # 2. Candidate finishReason SAFETY block
    def mock_candidate_block(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "candidates": [{"finishReason": "SAFETY", "index": 0}]
        })

    client2 = httpx.AsyncClient(transport=httpx.MockTransport(mock_candidate_block))
    p2 = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client2)

    with pytest.raises(VLMResponseError, match="finishReason=SAFETY") as exc2:
        await p2.generate_scene_understanding(sample_evidence_package)
    assert exc2.value.is_retryable is False


@pytest.mark.asyncio
async def test_gemini_provider_empty_candidates_or_text(sample_evidence_package: EvidencePackageSchema):
    """Verifies that empty candidates or empty text parts raise VLMOutputParsingError."""
    def mock_empty_cands(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_empty_cands))
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    with pytest.raises(VLMOutputParsingError, match="empty 'candidates' list"):
        await provider.generate_scene_understanding(sample_evidence_package)


@pytest.mark.asyncio
async def test_gemini_provider_max_tokens_truncation(sample_evidence_package: EvidencePackageSchema):
    """Verifies that truncation under MAX_TOKENS producing invalid JSON provides explicit error."""
    def mock_truncated(request: httpx.Request) -> httpx.Response:
        # Partial truncated JSON
        return httpx.Response(200, json={
            "candidates": [
                {
                    "content": {"parts": [{"text": '{"scene_summary": "Truncated json...'}]},
                    "finishReason": "MAX_TOKENS",
                }
            ],
            "usageMetadata": {"promptTokenCount": 500, "candidatesTokenCount": 2048},
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_truncated))
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    with pytest.raises(VLMOutputParsingError):
        await provider.generate_scene_understanding(sample_evidence_package)


@pytest.mark.asyncio
async def test_gemini_provider_ungrounded_references_sanitized(sample_evidence_package: EvidencePackageSchema):
    """Verifies that ungrounded keyframe references from Gemini are rejected and recorded in unverified_aspects."""
    valid_content = get_valid_completion_content()
    # Hallucinated keyframe reference
    valid_content["visual_observations"][0]["keyframe_id"] = "kf_hallucinated_xyz"
    valid_content["claims"][0]["keyframe_ids"] = ["kf_hallucinated_xyz"]

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=build_mock_gemini_completion_body(valid_content))

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)

    # Observation with ungrounded keyframe must be rejected from visual_observations list
    assert len(res.visual_observations) == 0
    assert res.claims[0].category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified:" in res.claims[0].evidence_rationale
    assert any("kf_hallucinated_xyz" in aspect for aspect in res.unverified_aspects)


def test_gemini_secret_redaction_and_headers():
    """Verifies that Gemini API keys are redacted from logs and headers."""
    provider = GeminiVLMProvider(
        api_key="AIzaSySuperSecretKey123456",
        model_name="gemini-3.8-flash",
    )
    headers = provider._get_sanitized_headers()
    assert headers["x-goog-api-key"] == "[REDACTED]"
    assert "AIzaSy" not in str(headers)

    # Error text sanitization
    sanitized = provider._sanitize_error_text("Error occurred with AIzaSySuperSecretKey123456 on endpoint")
    assert "AIzaSySuperSecretKey123456" not in sanitized
    assert "[REDACTED]" in sanitized


@pytest.mark.asyncio
async def test_gemini_provider_health_check_variations():
    """Verifies Gemini health check handling: 200 operational, 401 auth failure, 404 not found, unconfigured."""
    # 1. Healthy (200)
    def mock_healthy(req: httpx.Request) -> httpx.Response:
        assert req.headers["x-goog-api-key"] == "valid-key"
        assert req.url.path == "/v1beta/models/gemini-3.8-flash"
        return httpx.Response(200, json={"name": "models/gemini-3.8-flash", "displayName": "Gemini 3.8 Flash"})

    client1 = httpx.AsyncClient(transport=httpx.MockTransport(mock_healthy))
    p1 = GeminiVLMProvider(api_key="valid-key", model_name="gemini-3.8-flash", http_client=client1)
    h1 = await p1.health_check()
    assert h1.is_healthy is True
    assert "operational" in h1.status_message
    assert h1.latency_ms is not None

    # 2. 401 Auth Error
    def mock_401(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="API key invalid")

    client2 = httpx.AsyncClient(transport=httpx.MockTransport(mock_401))
    p2 = GeminiVLMProvider(api_key="bad-key", model_name="gemini-3.8-flash", http_client=client2)
    h2 = await p2.health_check()
    assert h2.is_healthy is False
    assert "Authentication failed" in h2.status_message

    # 3. 404 Model Not Found
    def mock_404(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="Model not found")

    client3 = httpx.AsyncClient(transport=httpx.MockTransport(mock_404))
    p3 = GeminiVLMProvider(api_key="key", model_name="nonexistent-model", http_client=client3)
    h3 = await p3.health_check()
    assert h3.is_healthy is False
    assert "not found" in h3.status_message

    # 4. Unconfigured
    p4 = GeminiVLMProvider(api_key=None, model_name=None)
    h4 = await p4.health_check()
    assert h4.is_healthy is False
    assert "GEMINI_API_KEY is not configured" in h4.status_message


@pytest.mark.asyncio
async def test_scene_interpretation_service_with_gemini_provider(sample_evidence_package: EvidencePackageSchema):
    """Verifies SceneInterpretationService resolves GeminiVLMProvider and isolates failures."""
    # 1. Resolves GeminiVLMProvider
    settings = Settings(
        VLM_ENABLED=True,
        VLM_PROVIDER_TYPE="gemini",
        GEMINI_API_KEY="test-gemini-key",
        GEMINI_MODEL="gemini-3.8-flash",
    )
    service = SceneInterpretationService(settings=settings)
    assert isinstance(service.provider, GeminiVLMProvider)
    assert service.provider.provider_type == "gemini"
    assert service.provider.model_name == "gemini-3.8-flash"

    # 2. Failure isolation: Provider error returns failure descriptor, never crashes caller
    failing_gemini = GeminiVLMProvider(
        api_key="key",
        model_name="gemini-3.8-flash",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500, text="Internal Server Error"))),
    )
    service_isolated = SceneInterpretationService(provider=failing_gemini, settings=settings)
    result = await service_isolated.interpret_scene(sample_evidence_package)

    assert result.success is False
    assert result.interpretation is None
    assert result.failure is not None
    assert result.failure.error_type == "VLMResponseError"
    assert result.failure.provider_type == "gemini"
    assert result.failure.job_id == sample_evidence_package.job_id


# ============================================================================
# 8. Strict Evidence-Grounding Invariants Tests
# ============================================================================

@pytest.mark.asyncio
async def test_grounding_visual_observation_with_unknown_keyframe_id(sample_evidence_package: EvidencePackageSchema):
    """
    Rule 1 & 3: Visual observation citing an unknown keyframe ID must NOT be kept in visual_observations
    with keyframe_id=None, but rejected from visual_observations with an explicit diagnostic in unverified_aspects.
    Associated visual observation claims must be demoted to unverified.
    """
    content = get_valid_completion_content()
    content["visual_observations"][0]["keyframe_id"] = "kf_unknown_999"
    content["claims"][0]["keyframe_ids"] = ["kf_unknown_999"]

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)

    # Observation cannot remain in visual_observations list
    assert len(res.visual_observations) == 0
    # Must be disclosed in unverified_aspects
    assert any("kf_unknown_999" in a and "rejected" in a.lower() for a in res.unverified_aspects)
    # Claim must be demoted to UNVERIFIED
    assert res.claims[0].category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified:" in res.claims[0].evidence_rationale


@pytest.mark.asyncio
async def test_grounding_visual_observation_with_no_evidence_references(sample_evidence_package: EvidencePackageSchema):
    """
    Rule 1 & 4: Visual observation with missing keyframe reference (None) must be rejected
    from visual_observations, and visual claim with no keyframe references must be demoted to unverified.
    """
    content = get_valid_completion_content()
    content["visual_observations"][0]["keyframe_id"] = None
    content["claims"][0]["keyframe_ids"] = []

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)

    # Observation with None keyframe cannot remain in visual_observations
    assert len(res.visual_observations) == 0
    # Claim with no keyframes must be demoted to UNVERIFIED
    assert res.claims[0].category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified: Supporting visual evidence is insufficient or unavailable" in res.claims[0].evidence_rationale
    assert any("no supporting keyframe" in a.lower() for a in res.unverified_aspects)


@pytest.mark.asyncio
async def test_grounding_inference_with_unknown_evidence(sample_evidence_package: EvidencePackageSchema):
    """
    Rule 2 & 4: Inference claim citing unknown keyframe and unknown track must be demoted to unverified
    with an explicit diagnostic and evidence pruned.
    """
    content = get_valid_completion_content()
    content["claims"].append({
        "claim_id": "inf_claim_unknown",
        "category": "inference",
        "statement": "Stoppage caused by an accident on lane 3.",
        "keyframe_ids": ["kf_unknown_ghost"],
        "associated_track_ids": [777],  # Unknown track
        "evidence_rationale": "Concluded from blockage.",
    })

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)

    inf_claim = next(c for c in res.claims if c.claim_id == "inf_claim_unknown")
    assert inf_claim.category == ClaimCategoryEnum.UNVERIFIED
    assert inf_claim.keyframe_ids == []
    assert inf_claim.associated_track_ids == []
    assert "[Unverified: Supporting evidence is unavailable" in inf_claim.evidence_rationale
    assert any("inf_claim_unknown" in a and "demoted to unverified" in a for a in res.unverified_aspects)


@pytest.mark.asyncio
async def test_grounding_inference_with_no_evidence_references(sample_evidence_package: EvidencePackageSchema):
    """
    Rule 2 & 4: Inference claim citing no keyframes and no track IDs must be demoted to unverified.
    """
    content = get_valid_completion_content()
    content["claims"].append({
        "claim_id": "inf_claim_empty",
        "category": "inference",
        "statement": "Traffic slow-down could be weather-induced.",
        "keyframe_ids": [],
        "associated_track_ids": [],
        "evidence_rationale": "General hypothesis.",
    })

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)

    inf_claim = next(c for c in res.claims if c.claim_id == "inf_claim_empty")
    assert inf_claim.category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified: Supporting evidence is insufficient or unavailable" in inf_claim.evidence_rationale
    assert any("inf_claim_empty" in a and "no supporting keyframes or track IDs cited" in a for a in res.unverified_aspects)


@pytest.mark.asyncio
async def test_grounding_valid_visual_observation_with_valid_evidence(sample_evidence_package: EvidencePackageSchema):
    """
    Rule 1 & 7: Visual observation citing a genuine supplied keyframe ID is preserved exactly as supplied.
    """
    content = get_valid_completion_content()
    # "kf_test_10" is a genuine keyframe in sample_evidence_package
    content["visual_observations"][0]["keyframe_id"] = "kf_test_10"
    content["claims"][0]["keyframe_ids"] = ["kf_test_10"]
    content["claims"][0]["category"] = "visual_observation"

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)

    # Observation must be preserved
    assert len(res.visual_observations) == 1
    assert res.visual_observations[0].keyframe_id == "kf_test_10"
    # Claim must remain visual observation with keyframe preserved
    assert res.claims[0].category == ClaimCategoryEnum.VISUAL_OBSERVATION
    assert res.claims[0].keyframe_ids == ["kf_test_10"]


@pytest.mark.asyncio
async def test_grounding_fabricated_cv_fact_not_present_in_telemetry(sample_evidence_package: EvidencePackageSchema):
    """
    Rule 5 & 4: Claim asserting a CV fact with a fabricated track ID (not in supplied telemetry)
    must be demoted to unverified with an explicit diagnostic.
    """
    content = get_valid_completion_content()
    content["claims"].append({
        "claim_id": "fab_cv_claim",
        "category": "cv_fact",
        "statement": "Vehicle track 999 was speeding at 135 km/h.",
        "keyframe_ids": [],
        "associated_track_ids": [999],  # Fabricated! Only track 1 is in sample_evidence_package
        "evidence_rationale": "Measured by CV radar tracker.",
    })

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)

    cv_claim = next(c for c in res.claims if c.claim_id == "fab_cv_claim")
    assert cv_claim.category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified: Fabricated or unconfirmed CV fact; track IDs [999] not found in CV telemetry]" in cv_claim.evidence_rationale
    assert any("fab_cv_claim" in a and "demoted to unverified" in a for a in res.unverified_aspects)


@pytest.mark.asyncio
async def test_grounding_valid_cv_fact_matching_supplied_telemetry(sample_evidence_package: EvidencePackageSchema):
    """
    Rule 5 & 7: Claim asserting a genuine CV fact referencing verified telemetry track 1
    is preserved as cv_fact with evidence intact.
    """
    content = get_valid_completion_content()
    # claim_2 in get_valid_completion_content cites track 1, which exists in sample_evidence_package
    cv_claim_in = next(c for c in content["claims"] if c["category"] == "cv_fact")
    cv_claim_in["associated_track_ids"] = [1]

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)

    cv_claim_out = next(c for c in res.claims if c.claim_id == cv_claim_in["claim_id"])
    assert cv_claim_out.category == ClaimCategoryEnum.CV_FACT
    assert cv_claim_out.associated_track_ids == [1]
    assert "[Unverified:" not in cv_claim_out.evidence_rationale


# ============================================================================
# 9. Adversarial CV Fact Content-Grounding Tests
# ============================================================================

@pytest.mark.asyncio
async def test_grounding_cv_fact_fabricated_speed(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 1: Existing valid track ID (track 1) with fabricated speed measurement.
    Must be demoted to unverified because the CV pipeline does not measure calibrated speed.
    """
    content = get_valid_completion_content()
    content["claims"].append({
        "claim_id": "adv_speed_claim",
        "category": "cv_fact",
        "statement": "Vehicle track 1 was traveling at 120 km/h.",
        "keyframe_ids": [],
        "associated_track_ids": [1],
        "evidence_rationale": "Speed estimated from video flow.",
    })

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)
    claim = next(c for c in res.claims if c.claim_id == "adv_speed_claim")
    assert claim.category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified: Fabricated CV fact; CV pipeline does not measure calibrated vehicle speed" in claim.evidence_rationale
    assert any("adv_speed_claim" in a and "demoted to unverified" in a for a in res.unverified_aspects)


@pytest.mark.asyncio
async def test_grounding_cv_fact_fabricated_direction(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 2: Existing valid track ID (track 1) with fabricated compass heading/direction.
    Must be demoted to unverified because the CV pipeline does not measure compass heading.
    """
    content = get_valid_completion_content()
    content["claims"].append({
        "claim_id": "adv_direction_claim",
        "category": "cv_fact",
        "statement": "Vehicle track 1 was heading eastbound toward the intersection.",
        "keyframe_ids": [],
        "associated_track_ids": [1],
        "evidence_rationale": "Deduced from trajectory heading.",
    })

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)
    claim = next(c for c in res.claims if c.claim_id == "adv_direction_claim")
    assert claim.category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified: Fabricated CV fact; CV pipeline does not measure compass" in claim.evidence_rationale
    assert any("adv_direction_claim" in a and "demoted to unverified" in a for a in res.unverified_aspects)


@pytest.mark.asyncio
async def test_grounding_cv_fact_unsupported_event_assertion(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 3: Existing valid track ID (track 1) with unsupported collision/crash assertion.
    Must be demoted to unverified because collision events are not classified by CV telemetry.
    """
    content = get_valid_completion_content()
    content["claims"].append({
        "claim_id": "adv_collision_claim",
        "category": "cv_fact",
        "statement": "Vehicle track 1 was involved in a serious collision crash with an adjacent vehicle.",
        "keyframe_ids": [],
        "associated_track_ids": [1],
        "evidence_rationale": "CV collision detection triggered.",
    })

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)
    claim = next(c for c in res.claims if c.claim_id == "adv_collision_claim")
    assert claim.category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified: Fabricated CV fact; CV pipeline does not measure collision or accident event" in claim.evidence_rationale
    assert any("adv_collision_claim" in a and "demoted to unverified" in a for a in res.unverified_aspects)


@pytest.mark.asyncio
async def test_grounding_cv_fact_valid_telemetry_match(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 4: Valid CV fact matching an actual telemetry field and value exactly.
    Must remain classified as cv_fact with evidence intact.
    """
    content = get_valid_completion_content()
    content["claims"].append({
        "claim_id": "valid_match_claim",
        "category": "cv_fact",
        "statement": "Vehicle track 1 stopped duration is 6.5s.",
        "keyframe_ids": [],
        "associated_track_ids": [1],
        "evidence_rationale": "Directly recorded by stopped_vehicle detector stop_1_0.40.",
        "telemetry_provenance": {
            "source_type": "event",
            "record_id": "stop_1_0.40",
            "field_path": "duration_s",
            "numeric_value": 6.5,
            "subject": "track_1",
            "predicate": "stationary_duration",
            "unit": "seconds",
        },
    })

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)
    claim = next(c for c in res.claims if c.claim_id == "valid_match_claim")
    assert claim.category == ClaimCategoryEnum.CV_FACT
    assert claim.associated_track_ids == [1]
    assert "[Unverified:" not in claim.evidence_rationale
    assert claim.telemetry_provenance is not None
    assert claim.telemetry_provenance.numeric_value == 6.5


@pytest.mark.asyncio
async def test_grounding_cv_fact_nonexistent_telemetry_field(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 5: CV fact referencing a nonexistent telemetry field in provenance.
    Must be demoted to unverified.
    """
    content = get_valid_completion_content()
    content["claims"].append({
        "claim_id": "adv_nonexistent_field_claim",
        "category": "cv_fact",
        "statement": "Vehicle track 1 vehicle weight was measured at 2500 kg.",
        "keyframe_ids": [],
        "associated_track_ids": [1],
        "evidence_rationale": "CV weight sensor report.",
        "telemetry_provenance": {
            "source_type": "event",
            "record_id": "stop_1_0.40",
            "field_path": "vehicle_weight",
            "numeric_value": 2500.0,
        },
    })

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)
    claim = next(c for c in res.claims if c.claim_id == "adv_nonexistent_field_claim")
    assert claim.category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified: Event 'stop_1_0.40' lacks field 'vehicle_weight']" in claim.evidence_rationale
    assert any("adv_nonexistent_field_claim" in a and "lacks field 'vehicle_weight'" in a for a in res.unverified_aspects)


@pytest.mark.asyncio
async def test_grounding_cv_fact_missing_telemetry():
    """
    Adversarial Test 6: CV fact when evidence package contains empty telemetry.
    Must be demoted to unverified because missing telemetry cannot confirm claims.
    """
    kf = KeyframeDetailSchema(
        keyframe_id="kf_empty_01",
        source_video_id="vid_empty_01",
        source_job_id="job_empty_01",
        frame_idx=1,
        requested_frame_idx=1,
        is_verified=True,
        timestamp_s=0.04,
        selection_reason="initial_frame",
        associated_event_ids=[],
        width=640,
        height=360,
        image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        content_type="image/jpeg",
    )
    empty_telemetry_pkg = EvidencePackageSchema(
        evidence_package_id="evpkg_empty_01",
        job_id="job_empty_01",
        video_id="vid_empty_01",
        sequence_name="sequence_empty",
        keyframes=[kf],
        events=[],
        telemetry_summary={},
    )

    content = get_valid_completion_content()
    # Replace claims with a CV fact
    content["visual_observations"][0]["keyframe_id"] = "kf_empty_01"
    content["claims"] = [
        {
            "claim_id": "adv_missing_telemetry_claim",
            "category": "cv_fact",
            "statement": "Vehicle track 1 stationary count is 1.",
            "keyframe_ids": [],
            "associated_track_ids": [1],
            "evidence_rationale": "Counted by telemetry.",
            "telemetry_provenance": {
                "source_type": "event",
                "record_id": "stop_1_0.40",
                "field_path": "duration_s",
                "numeric_value": 6.5,
            },
        }
    ]

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(empty_telemetry_pkg)
    claim = next(c for c in res.claims if c.claim_id == "adv_missing_telemetry_claim")
    assert claim.category == ClaimCategoryEnum.UNVERIFIED
    assert any("adv_missing_telemetry_claim" in a and "demoted to unverified" in a for a in res.unverified_aspects)


@pytest.mark.asyncio
async def test_grounding_cv_fact_numeric_mismatch_beyond_tolerance(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 7: Numeric value mismatch beyond permitted tolerance (rel_tol=0.02, abs_tol=0.05).
    Event duration is 6.5s; claimed duration is 15.0s.
    Must be demoted to unverified.
    """
    content = get_valid_completion_content()
    content["claims"].append({
        "claim_id": "adv_mismatch_claim",
        "category": "cv_fact",
        "statement": "Vehicle track 1 remained stationary for 15.0 seconds.",
        "keyframe_ids": [],
        "associated_track_ids": [1],
        "evidence_rationale": "Measured by stopped vehicle counter.",
        "telemetry_provenance": {
            "source_type": "event",
            "record_id": "stop_1_0.40",
            "field_path": "duration_s",
            "numeric_value": 15.0,
        },
    })

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)
    claim = next(c for c in res.claims if c.claim_id == "adv_mismatch_claim")
    assert claim.category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified: Numeric mismatch for event 'stop_1_0.40' field 'duration_s'; asserted 15.0, source telemetry is 6.5]" in claim.evidence_rationale
    assert any("adv_mismatch_claim" in a and "Numeric mismatch" in a for a in res.unverified_aspects)


@pytest.mark.asyncio
async def test_grounding_cv_fact_valid_provenance_retained_unchanged(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 8: Valid CV fact referencing telemetry summary metric (total_unique_tracks=4)
    with correct provenance is retained unchanged as cv_fact.
    """
    content = get_valid_completion_content()
    content["claims"].append({
        "claim_id": "valid_summary_claim",
        "category": "cv_fact",
        "statement": "Telemetry pipeline registered 4 total unique vehicle tracks.",
        "keyframe_ids": [],
        "associated_track_ids": [1, 2, 3, 4],
        "evidence_rationale": "Summary count from ByteTrack unique track counter.",
        "telemetry_provenance": {
            "source_type": "telemetry_summary",
            "record_id": None,
            "field_path": "total_unique_tracks",
            "numeric_value": 4.0,
            "subject": "scene",
            "predicate": "total_unique_tracks",
            "unit": "tracks",
        },
    })

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)
    claim = next(c for c in res.claims if c.claim_id == "valid_summary_claim")
    assert claim.category == ClaimCategoryEnum.CV_FACT
    assert "[Unverified:" not in claim.evidence_rationale
    assert claim.telemetry_provenance is not None
    assert claim.telemetry_provenance.source_type == "telemetry_summary"
    assert claim.telemetry_provenance.field_path == "total_unique_tracks"
    assert claim.telemetry_provenance.numeric_value == 4.0


# ============================================================================
# 10. Advanced CV Fact Integrity & Content-Consistency Adversarial Suite
# ============================================================================

@pytest.mark.asyncio
async def test_integrity_claim_text_contradicts_matching_numeric_provenance(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 1: Claim statement text directly contradicts numeric provenance.
    Actual telemetry: duration_s = 6.5.
    Provenance: duration_s = 6.5.
    Statement text: "Track 1 remained stopped for 15 seconds."
    Must be demoted to unverified because the statement text contradicts the verified provenance.
    """
    content = get_valid_completion_content()
    content["claims"].append({
        "claim_id": "adv_contradict_claim",
        "category": "cv_fact",
        "statement": "Track 1 remained stopped for 15 seconds.",
        "keyframe_ids": [],
        "associated_track_ids": [1],
        "evidence_rationale": "Directly observed in telemetry event.",
        "telemetry_provenance": {
            "source_type": "event",
            "record_id": "stop_1_0.40",
            "field_path": "duration_s",
            "numeric_value": 6.5,
            "unit": "s",
        },
    })

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)
    claim = next(c for c in res.claims if c.claim_id == "adv_contradict_claim")
    assert claim.category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified: Claim statement asserts contradictory or unconfirmed numerical quantity 15" in claim.evidence_rationale
    assert any("adv_contradict_claim" in a and "contradictory" in a for a in res.unverified_aspects)


@pytest.mark.asyncio
async def test_integrity_claim_text_fabricated_number_while_provenance_matches(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 2: Claim text asserts a fabricated number while provenance cites matching telemetry.
    Statement text: "Track 1 was observed with 99 secondary violations."
    Must be demoted to unverified because the fabricated number (99) is ungrounded.
    """
    content = get_valid_completion_content()
    content["claims"].append({
        "claim_id": "adv_fabricated_num_claim",
        "category": "cv_fact",
        "statement": "Track 1 was observed with 99 secondary violations.",
        "keyframe_ids": [],
        "associated_track_ids": [1],
        "evidence_rationale": "Telemetry inspection.",
        "telemetry_provenance": {
            "source_type": "event",
            "record_id": "stop_1_0.40",
            "field_path": "duration_s",
            "numeric_value": 6.5,
        },
    })

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)
    claim = next(c for c in res.claims if c.claim_id == "adv_fabricated_num_claim")
    assert claim.category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified: Claim statement asserts contradictory or unconfirmed numerical quantity 99" in claim.evidence_rationale
    assert any("adv_fabricated_num_claim" in a and "contradictory" in a for a in res.unverified_aspects)


@pytest.mark.asyncio
async def test_integrity_cv_fact_with_no_provenance_despite_nonempty_telemetry(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 3: Claim marked as cv_fact without structured provenance, despite non-empty telemetry.
    Must be strictly demoted to unverified.
    """
    content = get_valid_completion_content()
    content["claims"].append({
        "claim_id": "adv_no_provenance_claim",
        "category": "cv_fact",
        "statement": "Vehicle track 1 stationary status was maintained.",
        "keyframe_ids": [],
        "associated_track_ids": [1],
        "evidence_rationale": "CV tracking recorded state.",
        "telemetry_provenance": None,  # Intentionally missing!
    })

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)
    claim = next(c for c in res.claims if c.claim_id == "adv_no_provenance_claim")
    assert claim.category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified: CV fact claim lacks required structured telemetry provenance (telemetry_provenance is None)]" in claim.evidence_rationale
    assert any("adv_no_provenance_claim" in a and "lacks required structured telemetry provenance" in a for a in res.unverified_aspects)


@pytest.mark.asyncio
async def test_integrity_event_provenance_missing_or_nonexistent_record_id(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 4: Event provenance with missing record_id or nonexistent record_id.
    Must be demoted to unverified in both cases.
    """
    content = get_valid_completion_content()
    content["claims"].extend([
        {
            "claim_id": "adv_missing_record_id_claim",
            "category": "cv_fact",
            "statement": "Stopped vehicle duration was 6.5s.",
            "keyframe_ids": [],
            "associated_track_ids": [1],
            "evidence_rationale": "CV event report.",
            "telemetry_provenance": {
                "source_type": "event",
                "record_id": "",  # Empty record ID!
                "field_path": "duration_s",
                "numeric_value": 6.5,
            },
        },
        {
            "claim_id": "adv_nonexistent_record_id_claim",
            "category": "cv_fact",
            "statement": "Stopped vehicle duration was 6.5s.",
            "keyframe_ids": [],
            "associated_track_ids": [1],
            "evidence_rationale": "CV event report.",
            "telemetry_provenance": {
                "source_type": "event",
                "record_id": "stop_ghost_999",  # Nonexistent record ID!
                "field_path": "duration_s",
                "numeric_value": 6.5,
            },
        },
    ])

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)
    c1 = next(c for c in res.claims if c.claim_id == "adv_missing_record_id_claim")
    assert c1.category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified: Event provenance requires a non-empty record_id]" in c1.evidence_rationale

    c2 = next(c for c in res.claims if c.claim_id == "adv_nonexistent_record_id_claim")
    assert c2.category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified: Referenced event 'stop_ghost_999' not found in CV event telemetry]" in c2.evidence_rationale


@pytest.mark.asyncio
async def test_integrity_summary_provenance_referencing_event_only_field(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 5: Summary provenance incorrectly referencing an event-only field (e.g. event_id or roi_id).
    Must be demoted to unverified.
    """
    content = get_valid_completion_content()
    content["claims"].append({
        "claim_id": "adv_summary_event_field_claim",
        "category": "cv_fact",
        "statement": "Sequence event identifier is stop_1_0.40.",
        "keyframe_ids": [],
        "associated_track_ids": [],
        "evidence_rationale": "Summary citation.",
        "telemetry_provenance": {
            "source_type": "telemetry_summary",
            "record_id": None,
            "field_path": "event_id",  # Event-only field!
            "text_value": "stop_1_0.40",
        },
    })

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)
    claim = next(c for c in res.claims if c.claim_id == "adv_summary_event_field_claim")
    assert claim.category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified: Field path 'event_id' is an event-only field and cannot be resolved against telemetry_summary]" in claim.evidence_rationale


@pytest.mark.asyncio
async def test_integrity_private_internal_or_malformed_field_path_traversal(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 6: Private/dunder attribute traversal or malformed path.
    Must be rejected with security diagnostics.
    """
    content = get_valid_completion_content()
    content["claims"].extend([
        {
            "claim_id": "adv_private_path_claim",
            "category": "cv_fact",
            "statement": "Model class name was extracted.",
            "keyframe_ids": [],
            "associated_track_ids": [],
            "evidence_rationale": "Traversal attack.",
            "telemetry_provenance": {
                "source_type": "telemetry_summary",
                "field_path": "__class__.__name__",
                "text_value": "dict",
            },
        },
        {
            "claim_id": "adv_malformed_dots_claim",
            "category": "cv_fact",
            "statement": "FPS metric was measured at 25.",
            "keyframe_ids": [],
            "associated_track_ids": [],
            "evidence_rationale": "Malformed dots path.",
            "telemetry_provenance": {
                "source_type": "telemetry_summary",
                "field_path": "telemetry..fps",
                "numeric_value": 25.0,
            },
        },
    ])

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)
    c1 = next(c for c in res.claims if c.claim_id == "adv_private_path_claim")
    assert c1.category == ClaimCategoryEnum.UNVERIFIED
    assert "Access to private, internal, or dunder attribute" in c1.evidence_rationale

    c2 = next(c for c in res.claims if c.claim_id == "adv_malformed_dots_claim")
    assert c2.category == ClaimCategoryEnum.UNVERIFIED
    assert "Malformed field path 'telemetry..fps' contains empty segments" in c2.evidence_rationale


@pytest.mark.asyncio
async def test_integrity_numeric_value_with_incompatible_units(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 7: Numeric value with incompatible units.
    Case A: Duration field asserting unit='km/h'.
    Case B: Statement asserting speed unit '6.5 km/h' for temporal duration.
    Both must be demoted to unverified.
    """
    content = get_valid_completion_content()
    content["claims"].extend([
        {
            "claim_id": "adv_incompatible_unit_schema_claim",
            "category": "cv_fact",
            "statement": "Vehicle track 1 duration is 6.5s.",
            "keyframe_ids": [],
            "associated_track_ids": [1],
            "evidence_rationale": "Duration with wrong unit.",
            "telemetry_provenance": {
                "source_type": "event",
                "record_id": "stop_1_0.40",
                "field_path": "duration_s",
                "numeric_value": 6.5,
                "unit": "km/h",  # Incompatible unit!
            },
        },
        {
            "claim_id": "adv_incompatible_unit_text_claim",
            "category": "cv_fact",
            "statement": "Vehicle track 1 stationary duration was 6.5 meters.",
            "keyframe_ids": [],
            "associated_track_ids": [1],
            "evidence_rationale": "Duration with wrong text unit.",
            "telemetry_provenance": {
                "source_type": "event",
                "record_id": "stop_1_0.40",
                "field_path": "duration_s",
                "numeric_value": 6.5,
            },
        },
    ])

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)
    c1 = next(c for c in res.claims if c.claim_id == "adv_incompatible_unit_schema_claim")
    assert c1.category == ClaimCategoryEnum.UNVERIFIED
    assert "Unit 'km/h' is incompatible with temporal field 'duration_s'" in c1.evidence_rationale

    c2 = next(c for c in res.claims if c.claim_id == "adv_incompatible_unit_text_claim")
    assert c2.category == ClaimCategoryEnum.UNVERIFIED
    assert "Claim statement asserts incompatible units for duration field 'duration_s'" in c2.evidence_rationale


@pytest.mark.asyncio
async def test_integrity_missing_telemetry_field_vs_genuine_zero_value(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 8: Missing telemetry field vs a genuine zero value.
    - Missing field must be rejected and NEVER treated as zero.
    - Genuine zero value in telemetry MUST pass verification.
    """
    # Augment evidence package with a genuine 0 value in line_counts
    sample_evidence_package.telemetry_summary["line_counts"] = {"line_north": 0}

    content = get_valid_completion_content()
    content["claims"].extend([
        {
            "claim_id": "adv_missing_field_zero_claim",
            "category": "cv_fact",
            "statement": "Non-existent flow registered 0.0.",
            "keyframe_ids": [],
            "associated_track_ids": [],
            "evidence_rationale": "Missing field assumed zero.",
            "telemetry_provenance": {
                "source_type": "telemetry_summary",
                "field_path": "nonexistent_flow",
                "numeric_value": 0.0,
            },
        },
        {
            "claim_id": "adv_genuine_zero_claim",
            "category": "cv_fact",
            "statement": "Line north registered 0 vehicle crossings.",
            "keyframe_ids": [],
            "associated_track_ids": [],
            "evidence_rationale": "Actual zero crossing in telemetry.",
            "telemetry_provenance": {
                "source_type": "telemetry_summary",
                "field_path": "line_counts.line_north",
                "numeric_value": 0.0,
                "subject": "line_north",
                "predicate": "line_count",
                "unit": "crossings",
            },
        },
    ])

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    client = httpx.AsyncClient(transport=transport)
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=client)

    res = await provider.generate_scene_understanding(sample_evidence_package)
    c1 = next(c for c in res.claims if c.claim_id == "adv_missing_field_zero_claim")
    assert c1.category == ClaimCategoryEnum.UNVERIFIED
    assert "[Unverified: CV fact references non-existent telemetry field 'nonexistent_flow']" in c1.evidence_rationale

    c2 = next(c for c in res.claims if c.claim_id == "adv_genuine_zero_claim")
    assert c2.category == ClaimCategoryEnum.CV_FACT
    assert "[Unverified:" not in c2.evidence_rationale
    assert c2.telemetry_provenance is not None
    assert c2.telemetry_provenance.numeric_value == 0.0


@pytest.mark.asyncio
async def test_integrity_valid_consistent_cv_fact_remains_cv_fact_both_providers(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 9: Valid, consistent CV fact remains CV_FACT across both Gemini and Remote providers.
    """
    content = get_valid_completion_content()
    content["claims"] = [
        {
            "claim_id": "valid_fact_both",
            "category": "cv_fact",
            "statement": "Vehicle track 1 remained stationary for 6.5s.",
            "keyframe_ids": [],
            "associated_track_ids": [1],
            "evidence_rationale": "ByteTrack stationary duration measured 6.5 seconds.",
            "telemetry_provenance": {
                "source_type": "event",
                "record_id": "stop_1_0.40",
                "field_path": "duration_s",
                "numeric_value": 6.5,
                "unit": "s",
                "subject": "track_1",
                "predicate": "stationary_duration",
            },
        }
    ]

    # Test Gemini Provider
    gemini_transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    gemini_client = httpx.AsyncClient(transport=gemini_transport)
    gemini_provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=gemini_client)

    res_gemini = await gemini_provider.generate_scene_understanding(sample_evidence_package)
    gemini_claim = next(c for c in res_gemini.claims if c.claim_id == "valid_fact_both")
    assert gemini_claim.category == ClaimCategoryEnum.CV_FACT
    assert "[Unverified:" not in gemini_claim.evidence_rationale
    assert gemini_claim.telemetry_provenance.numeric_value == 6.5

    # Test Remote OpenAI Provider
    remote_transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_openai_completion_body(content)))
    remote_client = httpx.AsyncClient(transport=remote_transport)
    remote_provider = OpenAICompatibleRemoteProvider(api_base_url="https://api.openai.com/v1", model_name="Qwen3-VL-2B", http_client=remote_client)

    res_remote = await remote_provider.generate_scene_understanding(sample_evidence_package)
    remote_claim = next(c for c in res_remote.claims if c.claim_id == "valid_fact_both")
    assert remote_claim.category == ClaimCategoryEnum.CV_FACT
    assert "[Unverified:" not in remote_claim.evidence_rationale
    assert remote_claim.telemetry_provenance.numeric_value == 6.5


@pytest.mark.asyncio
async def test_integrity_unsupported_measurements_alternative_wording_both_providers(sample_evidence_package: EvidencePackageSchema):
    """
    Adversarial Test 10: Unsupported measurements expressed using alternative wording, units, or phrases
    are demoted to UNVERIFIED across both Gemini and Remote providers.
    """
    test_cases = [
        ("alt_speed", "Vehicle track 1 was moving at a rapid pace of 85 kph.", "calibrated vehicle speed"),
        ("alt_heading", "Vehicle track 1 maintained a bearing of 180 degrees southward.", "compass heading trajectory"),
        ("alt_collision", "Vehicle track 1 rear-ended another vehicle in a minor fender bender.", "collision or accident event"),
    ]

    for claim_id, statement, expected_phrase in test_cases:
        content = get_valid_completion_content()
        content["claims"] = [
            {
                "claim_id": claim_id,
                "category": "cv_fact",
                "statement": statement,
                "keyframe_ids": [],
                "associated_track_ids": [1],
                "evidence_rationale": "Visual deductive observation.",
                "telemetry_provenance": {
                    "source_type": "event",
                    "record_id": "stop_1_0.40",
                    "field_path": "duration_s",
                    "numeric_value": 6.5,
                },
            }
        ]

        # Gemini
        gemini_transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
        gemini_client = httpx.AsyncClient(transport=gemini_transport)
        gemini_provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=gemini_client)
        res_g = await gemini_provider.generate_scene_understanding(sample_evidence_package)
        c_g = next(c for c in res_g.claims if c.claim_id == claim_id)
        assert c_g.category == ClaimCategoryEnum.UNVERIFIED
        assert expected_phrase in c_g.evidence_rationale

        # Remote
        remote_transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_openai_completion_body(content)))
        remote_client = httpx.AsyncClient(transport=remote_transport)
        remote_provider = OpenAICompatibleRemoteProvider(api_base_url="https://api.openai.com/v1", model_name="Qwen3-VL-2B", http_client=remote_client)
        res_r = await remote_provider.generate_scene_understanding(sample_evidence_package)
        c_r = next(c for c in res_r.claims if c.claim_id == claim_id)
        assert c_r.category == ClaimCategoryEnum.UNVERIFIED
        assert expected_phrase in c_r.evidence_rationale

# ============================================================================
# 11. Verified CV Fact Rendering Safeguard Tests
# ============================================================================

@pytest.mark.asyncio
async def test_verified_structured_fact_produces_deterministic_text_both_providers(sample_evidence_package: EvidencePackageSchema):
    """
    Safeguard Test 1: Verified structured fact generates deterministic statement text
    directly from validated structured fields across both providers.
    """
    content = get_valid_completion_content()
    content["claims"] = [
        {
            "claim_id": "det_render_duration",
            "category": "cv_fact",
            "statement": "Raw Gemini model prose that should be overwritten by deterministic truth.",
            "keyframe_ids": [],
            "associated_track_ids": [1],
            "evidence_rationale": "Measured by ByteTrack detector.",
            "telemetry_provenance": {
                "source_type": "event",
                "record_id": "stop_1_0.40",
                "field_path": "duration_s",
                "numeric_value": 6.5,
                "subject": "track_1",
                "predicate": "stationary_duration",
                "unit": "seconds",
            },
        }
    ]

    # Gemini
    gemini_transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    gemini_client = httpx.AsyncClient(transport=gemini_transport)
    gemini_provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=gemini_client)
    res_g = await gemini_provider.generate_scene_understanding(sample_evidence_package)
    claim_g = next(c for c in res_g.claims if c.claim_id == "det_render_duration")
    assert claim_g.category == ClaimCategoryEnum.CV_FACT
    assert claim_g.statement == "Track 1 remained stationary for 6.5 seconds."

    # Remote
    remote_transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_openai_completion_body(content)))
    remote_client = httpx.AsyncClient(transport=remote_transport)
    remote_provider = OpenAICompatibleRemoteProvider(api_base_url="https://api.openai.com/v1", model_name="Qwen3-VL-2B", http_client=remote_client)
    res_r = await remote_provider.generate_scene_understanding(sample_evidence_package)
    claim_r = next(c for c in res_r.claims if c.claim_id == "det_render_duration")
    assert claim_r.category == ClaimCategoryEnum.CV_FACT
    assert claim_r.statement == "Track 1 remained stationary for 6.5 seconds."


@pytest.mark.asyncio
async def test_contradictory_model_text_cannot_override_verified_structured_fact(sample_evidence_package: EvidencePackageSchema):
    """
    Safeguard Test 2: Model-generated text cannot launder unverified narrative or override verified structured fact.
    - If model text has contradictory numeric claims, it is demoted to UNVERIFIED.
    - If model text has alternative narrative fluff, statement is deterministically rendered from truth.
    """
    # 2a. Model attempts numeric contradiction
    content_contra = get_valid_completion_content()
    content_contra["claims"] = [
        {
            "claim_id": "contra_claim",
            "category": "cv_fact",
            "statement": "Track 1 was stalled for 25.0 seconds at the curb.",
            "keyframe_ids": [],
            "associated_track_ids": [1],
            "evidence_rationale": "Model speculation.",
            "telemetry_provenance": {
                "source_type": "event",
                "record_id": "stop_1_0.40",
                "field_path": "duration_s",
                "numeric_value": 6.5,
                "subject": "track_1",
                "predicate": "stationary_duration",
                "unit": "seconds",
            },
        }
    ]
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content_contra)))
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=httpx.AsyncClient(transport=transport))
    res = await provider.generate_scene_understanding(sample_evidence_package)
    c_contra = next(c for c in res.claims if c.claim_id == "contra_claim")
    assert c_contra.category == ClaimCategoryEnum.UNVERIFIED
    assert "contradictory or unconfirmed numerical quantity" in c_contra.evidence_rationale

    # 2b. Model provides consistent non-contradictory narrative -> statement rendered deterministically
    content_narrative = get_valid_completion_content()
    content_narrative["claims"] = [
        {
            "claim_id": "narrative_claim",
            "category": "cv_fact",
            "statement": "Vehicle track 1 had severe engine breakdown and remained stationary for 6.5 seconds.",
            "keyframe_ids": [],
            "associated_track_ids": [1],
            "evidence_rationale": "Visual inspection and timing.",
            "telemetry_provenance": {
                "source_type": "event",
                "record_id": "stop_1_0.40",
                "field_path": "duration_s",
                "numeric_value": 6.5,
                "subject": "track_1",
                "predicate": "stationary_duration",
                "unit": "seconds",
            },
        }
    ]
    transport_n = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content_narrative)))
    provider_n = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=httpx.AsyncClient(transport=transport_n))
    res_n = await provider_n.generate_scene_understanding(sample_evidence_package)
    c_narrative = next(c for c in res_n.claims if c.claim_id == "narrative_claim")
    assert c_narrative.category == ClaimCategoryEnum.CV_FACT
    assert c_narrative.statement == "Track 1 remained stationary for 6.5 seconds."
    assert "engine breakdown" not in c_narrative.statement


@pytest.mark.asyncio
async def test_missing_subject_or_predicate_causes_demotion(sample_evidence_package: EvidencePackageSchema):
    """
    Safeguard Test 3: Missing subject or predicate causes immediate demotion to UNVERIFIED.
    """
    cases = [
        ("missing_subj", None, "stationary_duration", "lacks required non-empty 'subject'"),
        ("empty_subj", "   ", "stationary_duration", "lacks required non-empty 'subject'"),
        ("missing_pred", "track_1", None, "lacks required non-empty 'predicate'"),
        ("empty_pred", "track_1", "   ", "lacks required non-empty 'predicate'"),
    ]

    for cid, subj, pred, expected_sub in cases:
        content = get_valid_completion_content()
        content["claims"] = [
            {
                "claim_id": cid,
                "category": "cv_fact",
                "statement": "Track 1 stopped for 6.5s.",
                "keyframe_ids": [],
                "associated_track_ids": [1],
                "evidence_rationale": "Measured duration.",
                "telemetry_provenance": {
                    "source_type": "event",
                    "record_id": "stop_1_0.40",
                    "field_path": "duration_s",
                    "numeric_value": 6.5,
                    "subject": subj,
                    "predicate": pred,
                    "unit": "seconds",
                },
            }
        ]
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
        provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=httpx.AsyncClient(transport=transport))
        res = await provider.generate_scene_understanding(sample_evidence_package)
        c = next(c for c in res.claims if c.claim_id == cid)
        assert c.category == ClaimCategoryEnum.UNVERIFIED
        assert expected_sub in c.evidence_rationale


@pytest.mark.asyncio
async def test_missing_or_incompatible_units_cause_demotion_where_required(sample_evidence_package: EvidencePackageSchema):
    """
    Safeguard Test 4: Missing or incompatible units cause demotion where required.
    """
    cases = [
        ("missing_unit_dur", "stationary_duration", None, "requires a valid measurement unit"),
        ("empty_unit_dur", "stationary_duration", "  ", "requires a valid measurement unit"),
        ("incompat_unit_dur", "stationary_duration", "km/h", "is incompatible with"),
        ("incompat_unit_dist", "stationary_duration", "meters", "is incompatible with"),
    ]

    for cid, pred, unit, expected_sub in cases:
        content = get_valid_completion_content()
        content["claims"] = [
            {
                "claim_id": cid,
                "category": "cv_fact",
                "statement": "Track 1 stopped for 6.5s.",
                "keyframe_ids": [],
                "associated_track_ids": [1],
                "evidence_rationale": "Measured duration.",
                "telemetry_provenance": {
                    "source_type": "event",
                    "record_id": "stop_1_0.40",
                    "field_path": "duration_s",
                    "numeric_value": 6.5,
                    "subject": "track_1",
                    "predicate": pred,
                    "unit": unit,
                },
            }
        ]
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
        provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=httpx.AsyncClient(transport=transport))
        res = await provider.generate_scene_understanding(sample_evidence_package)
        c = next(c for c in res.claims if c.claim_id == cid)
        assert c.category == ClaimCategoryEnum.UNVERIFIED
        assert expected_sub in c.evidence_rationale


@pytest.mark.asyncio
async def test_unknown_predicates_cannot_produce_authoritative_cv_facts(sample_evidence_package: EvidencePackageSchema):
    """
    Safeguard Test 5: Unknown or uncalibrated predicates cannot produce authoritative CV facts.
    """
    content = get_valid_completion_content()
    content["claims"] = [
        {
            "claim_id": "unknown_pred_claim",
            "category": "cv_fact",
            "statement": "Track 1 was stopped for 6.5s.",
            "keyframe_ids": [],
            "associated_track_ids": [1],
            "evidence_rationale": "Measured duration.",
            "telemetry_provenance": {
                "source_type": "event",
                "record_id": "stop_1_0.40",
                "field_path": "duration_s",
                "numeric_value": 6.5,
                "subject": "track_1",
                "predicate": "driver_emotional_state",
                "unit": "seconds",
            },
        }
    ]
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=httpx.AsyncClient(transport=transport))
    res = await provider.generate_scene_understanding(sample_evidence_package)
    c = next(c for c in res.claims if c.claim_id == "unknown_pred_claim")
    assert c.category == ClaimCategoryEnum.UNVERIFIED
    assert "Unknown or unsupported CV fact predicate 'driver_emotional_state'" in c.evidence_rationale


@pytest.mark.asyncio
async def test_nested_dict_resolution_rejects_arbitrary_unapproved_keys(sample_evidence_package: EvidencePackageSchema):
    """
    Safeguard Test 6: Nested dictionary field resolution prevents allowlisted roots from accessing arbitrary keys.
    - line_counts.arbitrary_key is rejected.
    - line_counts.keys is rejected (dict method).
    - line_counts.line_1.secret_metric is rejected (unapproved 3rd-level metric).
    - line_counts.line_1.total is accepted.
    """
    sample_evidence_package.telemetry_summary["line_counts"] = {
        "line_1": {"total": 12, "downstream": 12, "upstream": 0, "secret_metric": 99},
        "arbitrary_key": 55,
    }

    cases = [
        ("arb_subkey", "line_counts.arbitrary_key", 55.0, "crossings", False, "is not an approved line sensor identifier"),
        ("dict_method", "line_counts.keys", 0.0, "crossings", False, "strictly prohibited"),
        ("unapproved_metric", "line_counts.line_1.secret_metric", 99.0, "crossings", False, "is not an approved line metric"),
        ("approved_metric", "line_counts.line_1.total", 12.0, "crossings", True, None),
    ]

    for cid, path, val, unit, should_pass, expected_sub in cases:
        content = get_valid_completion_content()
        content["claims"] = [
            {
                "claim_id": cid,
                "category": "cv_fact",
                "statement": f"Tripwire line count is {val}.",
                "keyframe_ids": [],
                "associated_track_ids": [],
                "evidence_rationale": "Tripwire telemetry.",
                "telemetry_provenance": {
                    "source_type": "telemetry_summary",
                    "field_path": path,
                    "numeric_value": val,
                    "subject": "line_1",
                    "predicate": "line_count",
                    "unit": unit,
                },
            }
        ]
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
        provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=httpx.AsyncClient(transport=transport))
        res = await provider.generate_scene_understanding(sample_evidence_package)
        c = next(c for c in res.claims if c.claim_id == cid)
        if should_pass:
            assert c.category == ClaimCategoryEnum.CV_FACT
            assert c.statement == "Tripwire line 1 recorded 12 vehicle crossings."
        else:
            assert c.category == ClaimCategoryEnum.UNVERIFIED
            assert expected_sub in c.evidence_rationale


@pytest.mark.asyncio
async def test_identical_grounding_and_rendering_behavior_both_providers(sample_evidence_package: EvidencePackageSchema):
    """
    Safeguard Test 7: Both GeminiVLMProvider and OpenAICompatibleRemoteProvider produce identical
    epistemic category, rendered statement, and evidence rationale behavior.
    """
    content = get_valid_completion_content()
    content["claims"] = [
        {
            "claim_id": "parity_cv_fact",
            "category": "cv_fact",
            "statement": "Arbitrary model text.",
            "keyframe_ids": [],
            "associated_track_ids": [1],
            "evidence_rationale": "Measured by detector stop_1_0.40.",
            "telemetry_provenance": {
                "source_type": "event",
                "record_id": "stop_1_0.40",
                "field_path": "duration_s",
                "numeric_value": 6.5,
                "subject": "track_1",
                "predicate": "stationary_duration",
                "unit": "seconds",
            },
        },
        {
            "claim_id": "parity_unverified",
            "category": "cv_fact",
            "statement": "Arbitrary model text.",
            "keyframe_ids": [],
            "associated_track_ids": [1],
            "evidence_rationale": "Measured by detector stop_1_0.40.",
            "telemetry_provenance": {
                "source_type": "event",
                "record_id": "stop_1_0.40",
                "field_path": "duration_s",
                "numeric_value": 6.5,
                "subject": "track_1",
                "predicate": "unsupported_predicate",
                "unit": "seconds",
            },
        },
    ]

    # Gemini
    gemini_transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    gemini_client = httpx.AsyncClient(transport=gemini_transport)
    gemini_provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=gemini_client)
    res_g = await gemini_provider.generate_scene_understanding(sample_evidence_package)

    # Remote
    remote_transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_openai_completion_body(content)))
    remote_client = httpx.AsyncClient(transport=remote_transport)
    remote_provider = OpenAICompatibleRemoteProvider(api_base_url="https://api.openai.com/v1", model_name="Qwen3-VL-2B", http_client=remote_client)
    res_r = await remote_provider.generate_scene_understanding(sample_evidence_package)

    # Parity assertions
    assert len(res_g.claims) == len(res_r.claims)
    for c_g, c_r in zip(res_g.claims, res_r.claims):
        assert c_g.claim_id == c_r.claim_id
        assert c_g.category == c_r.category
        assert c_g.statement == c_r.statement
        assert c_g.evidence_rationale == c_r.evidence_rationale

# ============================================================================
# 12. Semantic Predicate & Metric Grounding Audit Tests
# ============================================================================

@pytest.mark.asyncio
async def test_semantic_unique_tracks_renders_tracker_identities_not_vehicles(sample_evidence_package: EvidencePackageSchema):
    """
    Semantic Audit Test 1: total_unique_tracks represents tracker identities,
    NOT a physical vehicle census. Statement must state 'unique tracker identities'.
    """
    content = get_valid_completion_content()
    content["claims"] = [
        {
            "claim_id": "audit_tracks_claim",
            "category": "cv_fact",
            "statement": "There were 4 vehicles on the road.",
            "keyframe_ids": [],
            "associated_track_ids": [],
            "evidence_rationale": "Tracking summary.",
            "telemetry_provenance": {
                "source_type": "telemetry_summary",
                "field_path": "total_unique_tracks",
                "numeric_value": 4.0,
                "subject": "scene",
                "predicate": "total_unique_tracks",
                "unit": "tracks",
            },
        }
    ]

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=httpx.AsyncClient(transport=transport))
    res = await provider.generate_scene_understanding(sample_evidence_package)
    c = next(c for c in res.claims if c.claim_id == "audit_tracks_claim")

    assert c.category == ClaimCategoryEnum.CV_FACT
    assert c.statement == "The scene recorded 4 unique tracker identities."
    assert "vehicle" not in c.statement.lower()


@pytest.mark.asyncio
async def test_semantic_unique_tracks_rejects_vehicle_unit(sample_evidence_package: EvidencePackageSchema):
    """
    Semantic Audit Test 2: total_unique_tracks cannot accept 'vehicles' as a unit,
    because tracking identities do not equal physical vehicle count.
    """
    content = get_valid_completion_content()
    content["claims"] = [
        {
            "claim_id": "audit_tracks_wrong_unit",
            "category": "cv_fact",
            "statement": "The scene recorded 4 vehicles.",
            "keyframe_ids": [],
            "associated_track_ids": [],
            "evidence_rationale": "Tracking count.",
            "telemetry_provenance": {
                "source_type": "telemetry_summary",
                "field_path": "total_unique_tracks",
                "numeric_value": 4.0,
                "subject": "scene",
                "predicate": "total_unique_tracks",
                "unit": "vehicles",  # Prohibited unit!
            },
        }
    ]

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=httpx.AsyncClient(transport=transport))
    res = await provider.generate_scene_understanding(sample_evidence_package)
    c = next(c for c in res.claims if c.claim_id == "audit_tracks_wrong_unit")

    assert c.category == ClaimCategoryEnum.UNVERIFIED
    assert "Unit 'vehicles' is incompatible with" in c.evidence_rationale


@pytest.mark.asyncio
async def test_semantic_vehicle_count_predicate_rejected(sample_evidence_package: EvidencePackageSchema):
    """
    Semantic Audit Test 3: 'vehicle_count' predicate is rejected because our CV pipeline
    tracks identities and crossings, not a physical vehicle census.
    """
    content = get_valid_completion_content()
    content["claims"] = [
        {
            "claim_id": "audit_veh_count_claim",
            "category": "cv_fact",
            "statement": "Total vehicles counted was 4.",
            "keyframe_ids": [],
            "associated_track_ids": [],
            "evidence_rationale": "Claimed vehicle count.",
            "telemetry_provenance": {
                "source_type": "telemetry_summary",
                "field_path": "total_unique_tracks",
                "numeric_value": 4.0,
                "subject": "scene",
                "predicate": "vehicle_count",  # Prohibited predicate!
                "unit": "count",
            },
        }
    ]

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=httpx.AsyncClient(transport=transport))
    res = await provider.generate_scene_understanding(sample_evidence_package)
    c = next(c for c in res.claims if c.claim_id == "audit_veh_count_claim")

    assert c.category == ClaimCategoryEnum.UNVERIFIED
    assert "Unknown or unsupported CV fact predicate 'vehicle_count'" in c.evidence_rationale


@pytest.mark.asyncio
async def test_semantic_detection_count_rendered_as_bounding_box_detections(sample_evidence_package: EvidencePackageSchema):
    """
    Semantic Audit Test 4: detection_count represents raw bounding box detections,
    distinct from unique tracked identities or physical vehicles.
    """
    content = get_valid_completion_content()
    content["claims"] = [
        {
            "claim_id": "audit_detections_claim",
            "category": "cv_fact",
            "statement": "Model detected 120 cars.",
            "keyframe_ids": [],
            "associated_track_ids": [],
            "evidence_rationale": "Detection aggregator.",
            "telemetry_provenance": {
                "source_type": "telemetry_summary",
                "field_path": "total_detections",
                "numeric_value": 120.0,
                "subject": "scene",
                "predicate": "total_detections",
                "unit": "detections",
            },
        }
    ]

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=httpx.AsyncClient(transport=transport))
    res = await provider.generate_scene_understanding(sample_evidence_package)
    c = next(c for c in res.claims if c.claim_id == "audit_detections_claim")

    assert c.category == ClaimCategoryEnum.CV_FACT
    assert c.statement == "The scene recorded 120 total bounding box detections."
    assert "track" not in c.statement.lower()
    assert "vehicle" not in c.statement.lower()


@pytest.mark.asyncio
async def test_semantic_stationary_duration_requires_event_source(sample_evidence_package: EvidencePackageSchema):
    """
    Semantic Audit Test 5: stationary_duration is strictly an event-level metric and
    cannot be asserted against telemetry_summary.
    """
    sample_evidence_package.telemetry_summary["duration_s"] = 25.0
    content = get_valid_completion_content()
    content["claims"] = [
        {
            "claim_id": "audit_stat_dur_summary_claim",
            "category": "cv_fact",
            "statement": "The scene was stationary.",
            "keyframe_ids": [],
            "associated_track_ids": [],
            "evidence_rationale": "Mismatched source.",
            "telemetry_provenance": {
                "source_type": "telemetry_summary",  # Incompatible with stationary_duration!
                "field_path": "duration_s",
                "numeric_value": 25.0,
                "subject": "scene",
                "predicate": "stationary_duration",
                "unit": "seconds",
            },
        }
    ]

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
    provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=httpx.AsyncClient(transport=transport))
    res = await provider.generate_scene_understanding(sample_evidence_package)
    c = next(c for c in res.claims if c.claim_id == "audit_stat_dur_summary_claim")

    assert c.category == ClaimCategoryEnum.UNVERIFIED
    assert "Predicate 'stationary_duration' is incompatible with source_type 'telemetry_summary'" in c.evidence_rationale


@pytest.mark.asyncio
async def test_semantic_singular_plural_grammar_accuracy(sample_evidence_package: EvidencePackageSchema):
    """
    Semantic Audit Test 6: Deterministic rendering accurately handles singular vs plural grammar.
    """
    # Augment evidence with singular values
    sample_evidence_package.telemetry_summary["total_unique_tracks"] = 1
    sample_evidence_package.telemetry_summary["total_detections"] = 1
    sample_evidence_package.telemetry_summary["total_frames"] = 1
    sample_evidence_package.telemetry_summary["duration_s"] = 1.0
    sample_evidence_package.telemetry_summary["line_counts"] = {"line_1": {"total": 1}}

    # Add singular event
    evt_singular = EventReferenceSchema(
        event_id="stop_single",
        event_type="stopped_vehicle",
        track_id=1,
        timestamp_s=0.1,
        duration_s=1.0,
        roi_id="roi_main",
        details={"status": "resolved", "stationary_frames": 1},
    )
    sample_evidence_package.events.append(evt_singular)

    cases = [
        # Predicate, source_type, record_id, field_path, val, unit, subj, expected_text
        ("stationary_duration", "event", "stop_single", "duration_s", 1.0, "s", "track_1", "Track 1 remained stationary for 1 second."),
        ("stationary_frames", "event", "stop_single", "details.stationary_frames", 1.0, "frames", "track_1", "Track 1 remained stationary for 1 frame."),
        ("total_unique_tracks", "telemetry_summary", None, "total_unique_tracks", 1.0, "tracks", "scene", "The scene recorded 1 unique tracker identity."),
        ("total_detections", "telemetry_summary", None, "total_detections", 1.0, "detections", "scene", "The scene recorded 1 total bounding box detection."),
        ("sequence_duration", "telemetry_summary", None, "duration_s", 1.0, "seconds", "sequence", "The sequence duration is 1 second."),
        ("total_frames", "telemetry_summary", None, "total_frames", 1.0, "frames", "video", "The video contains 1 total frame."),
        ("line_count", "telemetry_summary", None, "line_counts.line_1.total", 1.0, "crossings", "line_1", "Tripwire line 1 recorded 1 vehicle crossing."),
    ]

    for pred, src, rec, path, val, unit, subj, expected_text in cases:
        content = get_valid_completion_content()
        content["claims"] = [
            {
                "claim_id": f"sing_{pred}",
                "category": "cv_fact",
                "statement": "Raw text placeholder.",
                "keyframe_ids": [],
                "associated_track_ids": [1] if src == "event" else [],
                "evidence_rationale": "Singular test.",
                "telemetry_provenance": {
                    "source_type": src,
                    "record_id": rec,
                    "field_path": path,
                    "numeric_value": val,
                    "subject": subj,
                    "predicate": pred,
                    "unit": unit,
                },
            }
        ]
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
        provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=httpx.AsyncClient(transport=transport))
        res = await provider.generate_scene_understanding(sample_evidence_package)
        c = next(c for c in res.claims if c.claim_id == f"sing_{pred}")

        assert c.category == ClaimCategoryEnum.CV_FACT, f"Failed for {pred}: {c.evidence_rationale}"
        assert c.statement == expected_text, f"Mismatch for {pred}: got '{c.statement}', expected '{expected_text}'"


@pytest.mark.asyncio
async def test_semantic_unknown_or_ambiguous_predicates_rejected(sample_evidence_package: EvidencePackageSchema):
    """
    Semantic Audit Test 7: Unknown, uncalibrated, or ambiguous predicates are rejected.
    """
    ambiguous_predicates = [
        "traffic_density",
        "vehicle_speed",
        "driver_intent",
        "uncalibrated_metric",
        "headway_distance",
    ]

    for amb_pred in ambiguous_predicates:
        content = get_valid_completion_content()
        content["claims"] = [
            {
                "claim_id": f"amb_{amb_pred}",
                "category": "cv_fact",
                "statement": "Ambiguous claim.",
                "keyframe_ids": [],
                "associated_track_ids": [1],
                "evidence_rationale": "Ambiguous.",
                "telemetry_provenance": {
                    "source_type": "event",
                    "record_id": "stop_1_0.40",
                    "field_path": "duration_s",
                    "numeric_value": 6.5,
                    "subject": "track_1",
                    "predicate": amb_pred,
                    "unit": "seconds",
                },
            }
        ]
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json=build_mock_gemini_completion_body(content)))
        provider = GeminiVLMProvider(api_key="key", model_name="gemini-3.8-flash", http_client=httpx.AsyncClient(transport=transport))
        res = await provider.generate_scene_understanding(sample_evidence_package)
        c = next(c for c in res.claims if c.claim_id == f"amb_{amb_pred}")

        assert c.category == ClaimCategoryEnum.UNVERIFIED
        assert f"Unknown or unsupported CV fact predicate '{amb_pred}'" in c.evidence_rationale

