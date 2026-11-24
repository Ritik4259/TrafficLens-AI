"""
Scene Interpretation Orchestration Service for TrafficLens-AI.

Coordinates multimodal scene understanding across swappable VLM providers,
enforces strict epistemic output validation, and isolates inference failures
to ensure primary computer vision processing jobs never fail due to downstream VLM errors.
"""

from __future__ import annotations
from typing import Optional
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.schemas.vlm import (
    EvidencePackageSchema,
    SceneInterpretationResponseSchema,
    SceneInterpretationFailureSchema,
    SceneInterpretationResult,
    ProviderHealthSchema,
)
from backend.app.services.vlm.provider import VLMProvider, VLMError
from backend.app.services.vlm.gemini_provider import GeminiVLMProvider
from backend.app.services.vlm.mock_provider import MockVLMProvider
from backend.app.services.vlm.remote_provider import OpenAICompatibleRemoteProvider

logger = get_logger("trafficlens.vlm.service")


class SceneInterpretationService:
    """
    High-level domain orchestrator for Vision-Language Model scene interpretation.
    
    Provides strict failure isolation: VLM errors (timeouts, rate limits, network faults,
    schema mismatches) are caught, sanitized, and returned as structured failure descriptors
    without raising unhandled exceptions or compromising core CV processing results.
    """

    def __init__(
        self,
        provider: Optional[VLMProvider] = None,
        settings: Optional[Settings] = None,
    ):
        self._settings = settings or get_settings()
        self._provider = provider or self._resolve_provider()

    @property
    def provider(self) -> VLMProvider:
        return self._provider

    def _resolve_provider(self) -> VLMProvider:
        """Resolves active VLM provider instance based on application settings."""
        provider_type = self._settings.VLM_PROVIDER_TYPE.lower()

        if provider_type == "gemini":
            logger.info(
                f"Initializing GeminiVLMProvider (model: {self._settings.GEMINI_MODEL}, "
                f"base_url: {self._settings.GEMINI_API_BASE_URL}, "
                f"key: {self._settings.get_redacted_gemini_api_key()})"
            )
            return GeminiVLMProvider(
                api_key=self._settings.GEMINI_API_KEY,
                model_name=self._settings.GEMINI_MODEL,
                api_base_url=self._settings.GEMINI_API_BASE_URL,
                timeout_s=self._settings.VLM_REQUEST_TIMEOUT,
                max_retries=self._settings.VLM_MAX_RETRIES,
                max_output_tokens=self._settings.VLM_MAX_OUTPUT_TOKENS,
                temperature=self._settings.VLM_TEMPERATURE,
            )
        elif provider_type == "openai_compatible":
            logger.info(
                f"Initializing OpenAICompatibleRemoteProvider (base_url: {self._settings.VLM_API_BASE_URL}, "
                f"model: {self._settings.VLM_MODEL_NAME})"
            )
            return OpenAICompatibleRemoteProvider(
                api_base_url=self._settings.VLM_API_BASE_URL,
                api_key=self._settings.VLM_API_KEY,
                model_name=self._settings.VLM_MODEL_NAME,
                timeout_s=self._settings.VLM_REQUEST_TIMEOUT_S,
                max_retries=self._settings.VLM_MAX_RETRIES,
                max_output_tokens=self._settings.VLM_MAX_OUTPUT_TOKENS,
                temperature=self._settings.VLM_TEMPERATURE,
            )
        elif provider_type == "mock":
            logger.info("Initializing MockVLMProvider for offline/test operation.")
            return MockVLMProvider(model_name=self._settings.GEMINI_MODEL or self._settings.VLM_MODEL_NAME or "mock-vlm")
        else:
            logger.warning(
                f"Unknown VLM_PROVIDER_TYPE '{provider_type}'; falling back to MockVLMProvider."
            )
            return MockVLMProvider(model_name=self._settings.GEMINI_MODEL or self._settings.VLM_MODEL_NAME or "mock-vlm")

    async def interpret_scene(
        self,
        evidence_package: EvidencePackageSchema,
    ) -> SceneInterpretationResult:
        """
        Executes multimodal scene understanding on the supplied evidence package.

        Guarantees failure isolation: If VLM inference fails, returns a structured
        SceneInterpretationResult with failure details rather than raising an unhandled exception.

        Args:
            evidence_package: Package containing keyframes and CV telemetry from Milestone 5B.

        Returns:
            SceneInterpretationResult containing either a validated interpretation or failure descriptor.
        """
        job_id = evidence_package.job_id
        video_id = evidence_package.video_id

        # 1. Check if VLM is enabled
        if not self._settings.VLM_ENABLED and not isinstance(self._provider, MockVLMProvider):
            logger.info(f"Skipping VLM interpretation for job '{job_id}': VLM_ENABLED is False.")
            return SceneInterpretationResult(
                success=False,
                failure=SceneInterpretationFailureSchema(
                    job_id=job_id,
                    video_id=video_id,
                    error_type="VLMDisabled",
                    message="VLM scene interpretation is disabled in settings (VLM_ENABLED=False).",
                    provider_type=self._provider.provider_type,
                    model_name=self._provider.model_name or "unknown",
                    is_retryable=False,
                ),
            )

        # 2. Invoke Provider with Failure Isolation
        try:
            interpretation = await self._provider.generate_scene_understanding(evidence_package)
            logger.info(
                f"Successfully completed VLM scene interpretation for job '{job_id}' "
                f"via {self._provider.provider_type} ({self._provider.model_name})"
            )
            return SceneInterpretationResult(
                success=True,
                interpretation=interpretation,
            )

        except VLMError as vlm_err:
            logger.error(
                f"VLM inference error for job '{job_id}' ({type(vlm_err).__name__}): {vlm_err.message}"
            )
            return SceneInterpretationResult(
                success=False,
                failure=SceneInterpretationFailureSchema(
                    job_id=job_id,
                    video_id=video_id,
                    error_type=type(vlm_err).__name__,
                    message=vlm_err.message,
                    provider_type=self._provider.provider_type,
                    model_name=self._provider.model_name or "unknown",
                    is_retryable=vlm_err.is_retryable,
                ),
            )

        except Exception as exc:
            logger.error(
                f"Unexpected exception during VLM scene interpretation for job '{job_id}': {str(exc)}",
                exc_info=True,
            )
            return SceneInterpretationResult(
                success=False,
                failure=SceneInterpretationFailureSchema(
                    job_id=job_id,
                    video_id=video_id,
                    error_type="UnexpectedVLMError",
                    message=f"An unexpected error occurred during VLM inference: {type(exc).__name__}",
                    provider_type=self._provider.provider_type,
                    model_name=self._provider.model_name or "unknown",
                    is_retryable=False,
                ),
            )

    async def check_health(self) -> ProviderHealthSchema:
        """Queries health status of the active VLM provider backend."""
        try:
            return await self._provider.health_check()
        except Exception as exc:
            return ProviderHealthSchema(
                provider_type=self._provider.provider_type,
                model_name=self._provider.model_name,
                is_healthy=False,
                status_message=f"Health check exception: {str(exc)}",
                latency_ms=None,
            )
