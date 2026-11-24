"""
Core configuration and environment settings for TrafficLens-AI backend.

Leverages Pydantic v2 Settings for type-safe environment variable parsing,
sensible development defaults, SQLite/PostgreSQL compatibility, and CV baseline parameters.
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from pydantic import Field, field_validator, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # Project Information
    PROJECT_NAME: str = "TrafficLens-AI"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = Field(default="development", description="development, testing, or production")
    DEBUG: bool = False

    # Database Configuration (SQLite default with PostgreSQL dialect support)
    DATABASE_URL: str = Field(
        default="sqlite:///./trafficlens.db",
        description="Database connection URL. SQLite default; compatible with postgresql+psycopg://...",
    )
    DB_ECHO: bool = False

    @property
    def is_sqlite(self) -> bool:
        """Indicates whether the active database is SQLite."""
        return self.DATABASE_URL.startswith("sqlite")

    # Storage and Media Paths
    BASE_DIR: Path = Path(__file__).resolve().parents[3]
    STORAGE_DIR: Path = Field(default=Path("./storage"))
    UPLOAD_DIR: Path = Field(default=Path("./storage/uploads"))
    OUTPUT_DIR: Path = Field(default=Path("./storage/outputs"))
    MAX_UPLOAD_SIZE_BYTES: int = 500 * 1024 * 1024  # 500 MB
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
    WORKER_MAX_QUEUE_SIZE: int = Field(default=100, description="Maximum pending jobs in worker queue")

    # In-Process CV Inference Configuration (reflects frozen CV baselines)
    YOLO_MODEL_PATH: str = "yolo11n.pt"
    INFERENCE_DEVICE: str = "cpu"  # "cpu", "cuda", "mps"
    CONF_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45

    # ByteTrack Tracking Parameters (reflects frozen validation sweep)
    TRACK_HIGH_THRESH: float = 0.40
    TRACK_LOW_THRESH: float = 0.10
    NEW_TRACK_THRESH: float = 0.50
    MATCH_THRESH: float = 0.70
    TRACK_BUFFER: int = 45

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # "json" or "text"

    # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Vision-Language Model (VLM) Configuration
    VLM_ENABLED: bool = Field(default=False, description="Enable multimodal VLM scene interpretation")
    VLM_PROVIDER_TYPE: str = Field(
        default="gemini",
        description="Active VLM provider type ('gemini', 'mock', 'openai_compatible')",
    )
    GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "VLM_API_KEY"),
        description="Google Gemini API key for remote multimodal inference. Redacted in logs.",
    )
    GEMINI_MODEL: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_MODEL", "VLM_MODEL_NAME"),
        description="Remote model identifier for Gemini VLM inference (e.g. 'gemini-3.8-flash'). Must be explicitly configured.",
    )
    GEMINI_API_BASE_URL: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        validation_alias=AliasChoices("GEMINI_API_BASE_URL", "VLM_API_BASE_URL"),
        description="Base URL for Google Gemini Generative Language REST API",
    )
    VLM_REQUEST_TIMEOUT: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        validation_alias=AliasChoices("VLM_REQUEST_TIMEOUT", "VLM_REQUEST_TIMEOUT_S"),
        description="HTTP request timeout in seconds for VLM inference",
    )
    VLM_MAX_RETRIES: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retry attempts for transient VLM network, 429, or 5xx errors",
    )
    INCIDENT_AGENT_TIMEOUT: float = Field(
        default=30.0,
        ge=0.1,
        le=120.0,
        description="HTTP request timeout in seconds for incident agent LLM report generation",
    )
    VLM_MAX_OUTPUT_TOKENS: int = Field(
        default=8192,
        ge=256,
        le=8192,
        description="Maximum completion token limit for VLM responses",
    )
    VLM_TEMPERATURE: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for deterministic structured extraction",
    )
    INTERPRETATION_MAX_QUEUE_SIZE: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum pending scene interpretations in worker queue before 503",
    )
    INTERPRETATION_WORKER_CONCURRENCY: int = Field(
        default=1,
        ge=1,
        le=1,
        description="Concurrent scene interpretation worker threads (strictly 1 in single-process architecture)",
    )
    MAX_INTERPRETATION_ATTEMPTS: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum total interpretation attempts per CV job",
    )

    @field_validator("INTERPRETATION_WORKER_CONCURRENCY", mode="before")
    @classmethod
    def validate_interpretation_concurrency(cls, v: Any) -> Any:
        if v is not None and int(v) != 1:
            raise ValueError(
                f"INTERPRETATION_WORKER_CONCURRENCY must be 1. "
                f"Configured value '{v}' is unsupported: TrafficLens AI operates on a single-process "
                f"bounded worker architecture (see docs/adr/0005-vlm-job-orchestration.md)."
            )
        return v

    @property
    def VLM_API_KEY(self) -> Optional[str]:
        return self.GEMINI_API_KEY

    @property
    def VLM_MODEL_NAME(self) -> Optional[str]:
        return self.GEMINI_MODEL

    @property
    def VLM_API_BASE_URL(self) -> str:
        return self.GEMINI_API_BASE_URL

    @property
    def VLM_REQUEST_TIMEOUT_S(self) -> float:
        return self.VLM_REQUEST_TIMEOUT

    def get_redacted_gemini_api_key(self) -> Optional[str]:
        """Returns safe representation of the Gemini API key for logging purposes."""
        if not self.GEMINI_API_KEY:
            return None
        if len(self.GEMINI_API_KEY) <= 6:
            return "***"
        return f"{self.GEMINI_API_KEY[:2]}***{self.GEMINI_API_KEY[-2:]}"

    def get_redacted_vlm_api_key(self) -> Optional[str]:
        """Backward compatibility helper for get_redacted_gemini_api_key."""
        return self.get_redacted_gemini_api_key()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Factory creating cached application settings."""
    return Settings()
