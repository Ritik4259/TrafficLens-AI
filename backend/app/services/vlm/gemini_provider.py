"""
Google Gemini Vision-Language Model (VLM) Provider Adapter for TrafficLens-AI.

Implements native Google Gemini generateContent REST API integration using async httpx,
bounded retries, header authentication, strict schema enforcement, and epistemic grounding verification.
"""

from __future__ import annotations
import asyncio
import json
import math
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import httpx
from pydantic import ValidationError

from backend.app.core.logging import get_logger
from backend.app.schemas.vlm import (
    EvidencePackageSchema,
    SceneInterpretationResponseSchema,
    ModelMetadataSchema,
    ProviderHealthSchema,
    ClaimCategoryEnum,
)
from backend.app.services.vlm.provider import (
    VLMProvider,
    VLMConfigurationError,
    VLMConnectionError,
    VLMTimeoutError,
    VLMAuthenticationError,
    VLMRateLimitError,
    VLMResponseError,
    VLMOutputParsingError,
    VLMGroundingError,
)
from backend.app.services.vlm.prompt_manager import PromptManager
from backend.app.services.vlm.grounding_verifier import verify_scene_interpretation_grounding

logger = get_logger("trafficlens.vlm.gemini_provider")


class GeminiVLMProvider:
    """
    Concrete VLM provider adapter utilizing Google's native Gemini REST API.
    
    Target endpoint:
    POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    
    Security & Architecture Rules:
    - Authenticates via 'x-goog-api-key' header (never via URL query parameter).
    - API keys are redacted in all logs, exceptions, and serialized headers.
    - Uses async httpx client with bounded retries and exponential backoff.
    - Validates responses strictly against SceneInterpretationResponseSchema.
    - Enforces epistemic grounding against the supplied EvidencePackageSchema.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        api_base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        max_output_tokens: int = 2048,
        temperature: float = 0.2,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self._api_key = api_key
        # Strip potential 'models/' prefix to maintain consistent model identifiers
        if model_name and model_name.startswith("models/"):
            self._model_name = model_name[7:]
        else:
            self._model_name = model_name

        self._api_base_url = api_base_url.rstrip("/") if api_base_url else "https://generativelanguage.googleapis.com/v1beta"
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._custom_client = http_client

    @property
    def provider_type(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name or "unknown"

    def _get_headers(self) -> Dict[str, str]:
        """Headers for live Gemini API requests with header-based authentication."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["x-goog-api-key"] = self._api_key
        return headers

    def _get_sanitized_headers(self) -> Dict[str, str]:
        """Safe headers for logging and diagnostic messages."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["x-goog-api-key"] = "[REDACTED]"
        return headers

    def _sanitize_error_text(self, text: str) -> str:
        """Strips potential API keys or sensitive query strings from error text."""
        if not text:
            return ""
        if self._api_key and self._api_key in text:
            text = text.replace(self._api_key, "[REDACTED]")
        # Redact any key=... query parameters if echoed in error text
        text = re.sub(r"(key=)[a-zA-Z0-9_\-]+", r"\1[REDACTED]", text)
        return text[:300]

    async def generate_scene_understanding(
        self,
        evidence: EvidencePackageSchema,
    ) -> SceneInterpretationResponseSchema:
        """
        Submits evidence package to Gemini generateContent REST API and validates structured output.

        Args:
            evidence: Complete evidence package with base64 keyframes and CV telemetry.

        Returns:
            SceneInterpretationResponseSchema containing grounded scene interpretation.
        """
        # 1. Configuration Validation
        if not self._api_key:
            raise VLMConfigurationError(
                "GEMINI_API_KEY is not configured. Please set the GEMINI_API_KEY environment variable."
            )
        if not self._model_name:
            raise VLMConfigurationError(
                "GEMINI_MODEL is not configured. Please set GEMINI_MODEL (e.g. 'gemini-3.8-flash')."
            )

        client = self._custom_client or httpx.AsyncClient(timeout=self._timeout_s)
        should_close_client = self._custom_client is None

        endpoint_url = f"{self._api_base_url}/models/{self._model_name}:generateContent"

        # 2. Build Gemini Multimodal Payload
        gemini_parts = PromptManager.assemble_gemini_parts(evidence)
        response_schema = PromptManager.get_gemini_response_schema()

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": gemini_parts,
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
                "maxOutputTokens": self._max_output_tokens,
                "temperature": self._temperature,
            },
        }

        start_time = time.perf_counter()
        raw_response: Optional[httpx.Response] = None
        last_error: Optional[VLMError] = None

        try:
            for attempt in range(self._max_retries + 1):
                try:
                    logger.debug(
                        f"Dispatching Gemini generateContent request (attempt {attempt + 1}/{self._max_retries + 1}) "
                        f"to {endpoint_url} with {len(evidence.keyframes)} keyframes"
                    )

                    resp = await client.post(
                        endpoint_url,
                        headers=self._get_headers(),
                        json=payload,
                    )

                    status = resp.status_code

                    # Success
                    if status == 200:
                        raw_response = resp
                        break

                    sanitized_body = self._sanitize_error_text(resp.text)

                    # Non-retryable authentication error
                    if status in (401, 403):
                        raise VLMAuthenticationError(
                            f"Gemini API authentication failed (HTTP {status}). Check GEMINI_API_KEY."
                        )

                    # Non-retryable model not found
                    if status == 404:
                        raise VLMResponseError(
                            f"Gemini model or endpoint not found (HTTP 404): '{self._model_name}'. Verify GEMINI_MODEL.",
                            status_code=404,
                            is_retryable=False,
                        )

                    # Non-retryable bad request / client error
                    if status == 400:
                        raise VLMResponseError(
                            f"Gemini API rejected request (HTTP 400): {sanitized_body}",
                            status_code=400,
                            is_retryable=False,
                        )

                    # Retryable rate limits
                    if status == 429:
                        last_error = VLMRateLimitError(
                            f"Gemini API rate limit exceeded (HTTP 429): {sanitized_body}"
                        )

                    # Retryable server errors (500, 502, 503, 504)
                    elif status in (500, 502, 503, 504):
                        last_error = VLMResponseError(
                            f"Gemini server error (HTTP {status}): {sanitized_body}",
                            status_code=status,
                            is_retryable=True,
                        )
                    else:
                        raise VLMResponseError(
                            f"Gemini API returned unhandled status HTTP {status}: {sanitized_body}",
                            status_code=status,
                            is_retryable=False,
                        )

                except (httpx.TimeoutException, TimeoutError) as exc:
                    last_error = VLMTimeoutError(
                        f"Gemini request timed out after {self._timeout_s}s on attempt {attempt + 1}"
                    )
                except (httpx.ConnectError, httpx.NetworkError) as exc:
                    last_error = VLMConnectionError(
                        f"Failed to connect to Gemini endpoint {self._api_base_url}: {str(exc)}"
                    )

                # Retry backoff
                if attempt < self._max_retries:
                    if isinstance(last_error, VLMRateLimitError):
                        backoff_delay = max(2.0 * (2 ** attempt), 3.0)
                    else:
                        backoff_delay = 0.5 * (2 ** attempt)
                    logger.warning(
                        f"Gemini attempt {attempt + 1} failed ({type(last_error).__name__}); "
                        f"retrying in {backoff_delay:.2f}s..."
                    )
                    await asyncio.sleep(backoff_delay)
                else:
                    if last_error:
                        raise last_error

            if raw_response is None or raw_response.status_code != 200:
                raise last_error or VLMResponseError("Gemini inference failed without receiving successful response.")

            execution_duration_ms = int((time.perf_counter() - start_time) * 1000)

            # 3. Parse Gemini Response Structure
            try:
                resp_json = raw_response.json()
            except Exception as exc:
                raise VLMOutputParsingError(f"Gemini endpoint returned invalid non-JSON HTTP body: {str(exc)}")

            # Check prompt feedback blocks (e.g. prompt safety violation)
            prompt_feedback = resp_json.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason")
            if block_reason:
                raise VLMResponseError(
                    f"Gemini request was blocked by safety filters: {block_reason}",
                    status_code=400,
                    is_retryable=False,
                )

            # Check candidates
            candidates = resp_json.get("candidates", [])
            if not candidates:
                raise VLMOutputParsingError("Gemini endpoint returned completion payload with empty 'candidates' list.")

            candidate = candidates[0]
            finish_reason = candidate.get("finishReason")

            # Check candidate-level safety or recitation stops
            if finish_reason == "SAFETY":
                raise VLMResponseError(
                    "Gemini generation was blocked by safety filters (finishReason=SAFETY).",
                    status_code=400,
                    is_retryable=False,
                )
            if finish_reason == "RECITATION":
                raise VLMResponseError(
                    "Gemini generation stopped due to recitation policy (finishReason=RECITATION).",
                    status_code=400,
                    is_retryable=False,
                )

            parts = candidate.get("content", {}).get("parts", [])
            if not parts:
                raise VLMOutputParsingError("Gemini candidate content contained no parts.")

            content_text = parts[0].get("text", "")
            if not content_text or not content_text.strip():
                raise VLMOutputParsingError("Gemini candidate returned empty text content in completion part.")

            usage = resp_json.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", 0)
            completion_tokens = usage.get("candidatesTokenCount", 0)

            # 4. Extract and Parse Structured JSON Output
            try:
                parsed_dict = self._extract_json_dict(content_text)
            except VLMOutputParsingError as parse_err:
                if finish_reason == "MAX_TOKENS":
                    raise VLMOutputParsingError(
                        f"Gemini completion was truncated due to MAX_TOKENS ({self._max_output_tokens}). "
                        "Output token budget was exceeded before JSON response was fully generated."
                    )
                raise parse_err

            # Ensure provenance identifiers match caller evidence
            parsed_dict["job_id"] = evidence.job_id
            parsed_dict["video_id"] = evidence.video_id

            # Inject execution telemetry into model metadata
            parsed_dict["model_metadata"] = {
                "model_name": self._model_name,
                "provider_type": self.provider_type,
                "execution_time_ms": execution_duration_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "quantization_precision": None,
            }

            # 5. Validate Output Schema via Pydantic
            try:
                scene_interpretation = SceneInterpretationResponseSchema.model_validate(parsed_dict)
            except ValidationError as val_err:
                if finish_reason == "MAX_TOKENS":
                    raise VLMOutputParsingError(
                        f"Gemini completion was truncated due to MAX_TOKENS ({self._max_output_tokens}) "
                        f"and failed schema validation: {val_err.errors()}"
                    )
                raise VLMOutputParsingError(
                    f"Gemini output JSON failed validation against SceneInterpretationResponseSchema: {val_err.errors()}"
                )

            # 6. Enforce Epistemic Grounding Verification
            self._verify_grounding(scene_interpretation, evidence)

            return scene_interpretation

        finally:
            if should_close_client:
                await client.aclose()

    async def health_check(self) -> ProviderHealthSchema:
        """
        Executes lightweight diagnostic check against Gemini models API.
        
        GET https://generativelanguage.googleapis.com/v1beta/models/{model}
        """
        if not self._api_key:
            return ProviderHealthSchema(
                provider_type=self.provider_type,
                model_name=self._model_name or "unknown",
                is_healthy=False,
                status_message="GEMINI_API_KEY is not configured.",
                latency_ms=None,
            )

        if not self._model_name:
            return ProviderHealthSchema(
                provider_type=self.provider_type,
                model_name="unknown",
                is_healthy=False,
                status_message="GEMINI_MODEL is not configured.",
                latency_ms=None,
            )

        client = self._custom_client or httpx.AsyncClient(timeout=min(self._timeout_s, 5.0))
        should_close_client = self._custom_client is None
        start_time = time.perf_counter()

        try:
            model_url = f"{self._api_base_url}/models/{self._model_name}"
            resp = await client.get(model_url, headers=self._get_headers())
            latency = int((time.perf_counter() - start_time) * 1000)

            if resp.status_code == 200:
                return ProviderHealthSchema(
                    provider_type=self.provider_type,
                    model_name=self._model_name,
                    is_healthy=True,
                    status_message=f"Gemini model '{self._model_name}' is accessible and operational.",
                    latency_ms=latency,
                )
            elif resp.status_code in (401, 403):
                return ProviderHealthSchema(
                    provider_type=self.provider_type,
                    model_name=self._model_name,
                    is_healthy=False,
                    status_message="Authentication failed (HTTP 401/403). Invalid GEMINI_API_KEY.",
                    latency_ms=latency,
                )
            elif resp.status_code == 404:
                return ProviderHealthSchema(
                    provider_type=self.provider_type,
                    model_name=self._model_name,
                    is_healthy=False,
                    status_message=f"Model '{self._model_name}' not found (HTTP 404). Verify GEMINI_MODEL.",
                    latency_ms=latency,
                )
            else:
                sanitized = self._sanitize_error_text(resp.text)
                return ProviderHealthSchema(
                    provider_type=self.provider_type,
                    model_name=self._model_name,
                    is_healthy=False,
                    status_message=f"Gemini API returned HTTP {resp.status_code}: {sanitized}",
                    latency_ms=latency,
                )
        except Exception as exc:
            latency = int((time.perf_counter() - start_time) * 1000)
            return ProviderHealthSchema(
                provider_type=self.provider_type,
                model_name=self._model_name,
                is_healthy=False,
                status_message=f"Connection failed: {str(exc)}",
                latency_ms=latency,
            )
        finally:
            if should_close_client:
                await client.aclose()

    def _extract_json_dict(self, text: str) -> Dict[str, Any]:
        """Extracts JSON object from model completion text, stripping markdown fences if present."""
        text = text.strip()

        # Handle ```json ... ``` code fences
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Attempt to find outermost curly braces {...}
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            substring = text[brace_start : brace_end + 1]
            try:
                return json.loads(substring)
            except json.JSONDecodeError as exc:
                raise VLMOutputParsingError(f"Failed to parse JSON substring from Gemini output: {str(exc)}")

        raise VLMOutputParsingError("Gemini completion did not contain a parseable JSON object.")

    def _verify_grounding(
        self,
        interpretation: SceneInterpretationResponseSchema,
        evidence: EvidencePackageSchema,
    ) -> None:
        """
        Validates that all keyframe IDs, event IDs, visual observations, and CV facts
        in model outputs strictly correspond to genuine elements in the evidence package.
        Delegates to the centralized GroundingVerifier.
        """
        verify_scene_interpretation_grounding(interpretation, evidence)
