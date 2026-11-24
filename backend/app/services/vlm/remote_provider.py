"""
OpenAI-Compatible Remote VLM Adapter for TrafficLens-AI.

Interacts with any OpenAI-compatible multimodal endpoint (vLLM, Ollama, OpenRouter,
cloud APIs) using async HTTP, bounded retries, secret redaction, and strict grounding validation.
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

logger = get_logger("trafficlens.vlm.remote_provider")


class OpenAICompatibleRemoteProvider:
    """
    Generic remote provider adapter targeting OpenAI-compatible multimodal HTTP endpoints.
    
    Compatible with:
    - Self-hosted vLLM instances serving Qwen3-VL, SmolVLM, etc.
    - Ollama multimodal instances (/v1/chat/completions)
    - Cloud multimodal vision APIs (OpenAI, Gemini via OpenAI gateway, OpenRouter, Together)
    """

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        max_output_tokens: int = 2048,
        temperature: float = 0.2,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self._api_base_url = api_base_url.rstrip("/") if api_base_url else None
        self._api_key = api_key
        self._model_name = model_name
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._custom_client = http_client

    @property
    def provider_type(self) -> str:
        return "openai_compatible"

    @property
    def model_name(self) -> str:
        return self._model_name or "unknown"

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _get_sanitized_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer [REDACTED]"
        return headers

    async def generate_scene_understanding(
        self,
        evidence: EvidencePackageSchema,
    ) -> SceneInterpretationResponseSchema:
        """
        Submits evidence package to OpenAI-compatible chat completion endpoint.
        """
        if not self._api_base_url:
            raise VLMConfigurationError(
                "VLM API base URL is not configured. Set VLM_API_BASE_URL to an OpenAI-compatible endpoint."
            )

        endpoint_url = f"{self._api_base_url}/chat/completions"
        messages = PromptManager.assemble_openai_messages(evidence)

        payload = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": self._max_output_tokens,
            "temperature": self._temperature,
        }

        client = self._custom_client or httpx.AsyncClient(timeout=self._timeout_s)
        should_close_client = self._custom_client is None

        start_time = time.perf_counter()
        raw_response: Optional[httpx.Response] = None
        last_error: Optional[Exception] = None

        logger.info(
            f"Dispatching VLM scene interpretation request to {endpoint_url} "
            f"(model: {self._model_name}, keyframes: {len(evidence.keyframes)}, timeout: {self._timeout_s}s)"
        )

        try:
            for attempt in range(self._max_retries + 1):
                try:
                    raw_response = await client.post(
                        endpoint_url,
                        headers=self._get_headers(),
                        json=payload,
                    )

                    # Evaluate HTTP status codes
                    if raw_response.status_code == 200:
                        break

                    status = raw_response.status_code
                    if status in (401, 403):
                        raise VLMAuthenticationError(
                            f"Remote provider authentication failed (HTTP {status}): credentials rejected."
                        )
                    elif status == 429:
                        err = VLMRateLimitError(f"Remote provider rate limit exceeded (HTTP 429).")
                        last_error = err
                    elif status in (500, 502, 503, 504):
                        err = VLMResponseError(
                            f"Remote provider server error (HTTP {status}).",
                            status_code=status,
                            is_retryable=True,
                        )
                        last_error = err
                    else:
                        raise VLMResponseError(
                            f"Remote provider rejected request (HTTP {status}): {raw_response.text[:200]}",
                            status_code=status,
                            is_retryable=False,
                        )

                except (httpx.TimeoutException, TimeoutError) as exc:
                    last_error = VLMTimeoutError(
                        f"VLM request timed out after {self._timeout_s}s on attempt {attempt + 1}"
                    )
                except (httpx.ConnectError, httpx.NetworkError) as exc:
                    last_error = VLMConnectionError(
                        f"Failed to connect to VLM endpoint {self._api_base_url}: {str(exc)}"
                    )

                # Retry logic
                if attempt < self._max_retries:
                    backoff_delay = 0.5 * (2 ** attempt)
                    logger.warning(
                        f"VLM attempt {attempt + 1} failed ({type(last_error).__name__}); retrying in {backoff_delay:.2f}s..."
                    )
                    await asyncio.sleep(backoff_delay)
                else:
                    if last_error:
                        raise last_error

            if raw_response is None or raw_response.status_code != 200:
                raise last_error or VLMResponseError("VLM inference failed without receiving successful response.")

            execution_duration_ms = int((time.perf_counter() - start_time) * 1000)

            # 2. Parse response completion
            try:
                resp_json = raw_response.json()
            except Exception as exc:
                raise VLMOutputParsingError(f"Remote endpoint returned invalid non-JSON HTTP body: {str(exc)}")

            choices = resp_json.get("choices", [])
            if not choices:
                raise VLMOutputParsingError("Remote endpoint returned completion payload with empty 'choices' list.")

            content_text = choices[0].get("message", {}).get("content", "")
            if not content_text or not content_text.strip():
                raise VLMOutputParsingError("Remote endpoint returned empty text content in completion message.")

            usage = resp_json.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # 3. Extract JSON object from completion text
            parsed_dict = self._extract_json_dict(content_text)

            # Inject top-level job and video IDs from evidence
            parsed_dict["job_id"] = evidence.job_id
            parsed_dict["video_id"] = evidence.video_id

            # Inject telemetry metadata
            parsed_dict["model_metadata"] = {
                "model_name": self._model_name,
                "provider_type": self.provider_type,
                "execution_time_ms": execution_duration_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "quantization_precision": None,
            }

            # 4. Validate output schema
            try:
                scene_interpretation = SceneInterpretationResponseSchema.model_validate(parsed_dict)
            except ValidationError as val_err:
                raise VLMOutputParsingError(
                    f"VLM output JSON failed validation against SceneInterpretationResponseSchema: {val_err.errors()}"
                )

            # 5. Enforce Grounding Verification
            self._verify_grounding(scene_interpretation, evidence)

            return scene_interpretation

        finally:
            if should_close_client:
                await client.aclose()

    async def health_check(self) -> ProviderHealthSchema:
        """Lightweight diagnostic connectivity check against endpoint."""
        if not self._api_base_url:
            return ProviderHealthSchema(
                provider_type=self.provider_type,
                model_name=self._model_name,
                is_healthy=False,
                status_message="VLM_API_BASE_URL is not configured.",
                latency_ms=None,
            )

        client = self._custom_client or httpx.AsyncClient(timeout=min(self._timeout_s, 5.0))
        should_close_client = self._custom_client is None
        start_time = time.perf_counter()

        try:
            # Query /models endpoint (standard in OpenAI compatible servers)
            models_url = f"{self._api_base_url}/models"
            resp = await client.get(models_url, headers=self._get_headers())
            latency = int((time.perf_counter() - start_time) * 1000)

            if resp.status_code == 200:
                return ProviderHealthSchema(
                    provider_type=self.provider_type,
                    model_name=self._model_name,
                    is_healthy=True,
                    status_message="Provider reachable and models endpoint responded HTTP 200.",
                    latency_ms=latency,
                )
            else:
                return ProviderHealthSchema(
                    provider_type=self.provider_type,
                    model_name=self._model_name,
                    is_healthy=False,
                    status_message=f"Endpoint returned HTTP {resp.status_code}: {resp.text[:120]}",
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
                raise VLMOutputParsingError(f"Failed to parse JSON substring from model output: {str(exc)}")

        raise VLMOutputParsingError("Model completion did not contain a parseable JSON object.")

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
