"""
Vision-Language Model (VLM) Provider Interface and Error Contracts for TrafficLens-AI.

Defines the abstract protocol boundary and domain exceptions governing
multimodal scene interpretation backends.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
from backend.app.schemas.vlm import (
    EvidencePackageSchema,
    SceneInterpretationResponseSchema,
    ProviderHealthSchema,
)


# ============================================================================
# 1. Domain Exceptions
# ============================================================================

class VLMError(Exception):
    """Base exception for all Vision-Language Model operations."""
    def __init__(self, message: str, is_retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.is_retryable = is_retryable


class VLMConfigurationError(VLMError):
    """Raised when provider configuration or credentials are missing, invalid, or disabled."""
    def __init__(self, message: str):
        super().__init__(message, is_retryable=False)


class VLMConnectionError(VLMError):
    """Raised when the remote provider endpoint is unreachable or network drops."""
    def __init__(self, message: str):
        super().__init__(message, is_retryable=True)


class VLMTimeoutError(VLMError):
    """Raised when inference or health check exceeds configured timeout threshold."""
    def __init__(self, message: str):
        super().__init__(message, is_retryable=True)


class VLMAuthenticationError(VLMError):
    """Raised when the remote endpoint rejects credentials (HTTP 401 / 403)."""
    def __init__(self, message: str):
        super().__init__(message, is_retryable=False)


class VLMRateLimitError(VLMError):
    """Raised when remote endpoint rate limits or throttles requests (HTTP 429)."""
    def __init__(self, message: str):
        super().__init__(message, is_retryable=True)


class VLMResponseError(VLMError):
    """Raised when remote endpoint returns a non-200 HTTP status (e.g. 500, 502, 503)."""
    def __init__(self, message: str, status_code: int = 500, is_retryable: bool = True):
        super().__init__(message, is_retryable=is_retryable)
        self.status_code = status_code


class VLMOutputParsingError(VLMError):
    """Raised when model completion is malformed, not valid JSON, or fails Pydantic schema validation."""
    def __init__(self, message: str):
        super().__init__(message, is_retryable=False)


class VLMGroundingError(VLMError):
    """Raised when model output references non-existent keyframe IDs or fabricated metadata."""
    def __init__(self, message: str):
        super().__init__(message, is_retryable=False)


# ============================================================================
# 2. Abstract Provider Protocol
# ============================================================================

@runtime_checkable
class VLMProvider(Protocol):
    """
    Abstract protocol for swappable vision-language model inference backends.

    Concrete implementations (e.g. OpenAICompatibleRemoteProvider, MockVLMProvider)
    must satisfy this protocol without coupling to vendor-specific SDKs.
    """

    @property
    def provider_type(self) -> str:
        """Identifier of the provider backend (e.g. 'openai_compatible', 'mock')."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier of the target model (e.g. 'Qwen/Qwen3-VL-2B-Instruct')."""
        ...

    async def generate_scene_understanding(
        self,
        evidence: EvidencePackageSchema,
    ) -> SceneInterpretationResponseSchema:
        """
        Processes a multimodal evidence package and returns validated structured scene understanding.

        Args:
            evidence: Complete evidence package with base64 keyframes and CV telemetry.

        Returns:
            SceneInterpretationResponseSchema containing validated observations, claims, and telemetry.

        Raises:
            VLMConfigurationError: If configuration is invalid.
            VLMConnectionError: If remote endpoint cannot be reached.
            VLMTimeoutError: If the request times out.
            VLMAuthenticationError: If API credentials are rejected.
            VLMRateLimitError: If rate limited.
            VLMResponseError: If remote server returns 5xx error.
            VLMOutputParsingError: If completion cannot be validated against output schema.
            VLMGroundingError: If output violates epistemic grounding rules.
        """
        ...

    async def health_check(self) -> ProviderHealthSchema:
        """
        Executes a lightweight diagnostic ping against the provider.

        Returns:
            ProviderHealthSchema detailing reachability, latency, and status message.
        """
        ...
