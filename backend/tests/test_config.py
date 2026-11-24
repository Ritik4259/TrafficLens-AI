"""
Tests for application settings and configuration management.
"""

from __future__ import annotations
import os
import pytest

from backend.app.core.config import Settings


def test_default_configuration_settings():
    """Validates sensible defaults for local development."""
    settings = Settings()
    assert settings.PROJECT_NAME == "TrafficLens-AI"
    assert settings.VERSION == "0.1.0"
    assert settings.API_V1_STR == "/api/v1"
    assert settings.ENVIRONMENT == "development"
    assert settings.is_sqlite is True
    assert settings.DATABASE_URL == "sqlite:///./trafficlens.db"
    assert settings.CONF_THRESHOLD == 0.25
    assert settings.TRACK_HIGH_THRESH == 0.40
    assert settings.TRACK_BUFFER == 45


def test_environment_variable_overrides(monkeypatch):
    """Validates that environment variables override configuration defaults."""
    monkeypatch.setenv("PROJECT_NAME", "TrafficLens-Production")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:password@localhost:5432/trafficlens_prod")
    monkeypatch.setenv("CONF_THRESHOLD", "0.35")
    monkeypatch.setenv("TRACK_BUFFER", "60")

    settings = Settings()
    assert settings.PROJECT_NAME == "TrafficLens-Production"
    assert settings.ENVIRONMENT == "production"
    assert settings.is_sqlite is False
    assert settings.DATABASE_URL == "postgresql+psycopg://user:password@localhost:5432/trafficlens_prod"
    assert settings.CONF_THRESHOLD == 0.35
    assert settings.TRACK_BUFFER == 60
