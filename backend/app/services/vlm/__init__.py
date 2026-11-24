"""
VLM and Multimodal Scene Understanding Services.
"""

from .keyframe_extractor import KeyframeExtractor, ExtractedFrame, VideoReadError
from .evidence_packager import EvidencePackager
from .provider import (
    VLMProvider,
    VLMError,
    VLMConfigurationError,
    VLMConnectionError,
    VLMTimeoutError,
    VLMAuthenticationError,
    VLMRateLimitError,
    VLMResponseError,
    VLMOutputParsingError,
    VLMGroundingError,
)
from .prompt_manager import PromptManager
from .gemini_provider import GeminiVLMProvider
from .remote_provider import OpenAICompatibleRemoteProvider
from .mock_provider import MockVLMProvider
from .scene_interpretation_service import SceneInterpretationService
from .orchestrator import InterpretationOrchestratorService
from .grounding_verifier import (
    verify_scene_interpretation_grounding,
    render_verified_cv_fact,
    SUPPORTED_PREDICATES,
)

__all__ = [
    "KeyframeExtractor",
    "ExtractedFrame",
    "VideoReadError",
    "EvidencePackager",
    "VLMProvider",
    "VLMError",
    "VLMConfigurationError",
    "VLMConnectionError",
    "VLMTimeoutError",
    "VLMAuthenticationError",
    "VLMRateLimitError",
    "VLMResponseError",
    "VLMOutputParsingError",
    "VLMGroundingError",
    "PromptManager",
    "GeminiVLMProvider",
    "OpenAICompatibleRemoteProvider",
    "MockVLMProvider",
    "SceneInterpretationService",
    "InterpretationOrchestratorService",
    "verify_scene_interpretation_grounding",
    "render_verified_cv_fact",
    "SUPPORTED_PREDICATES",
]
