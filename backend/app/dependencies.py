"""
FastAPI dependency injection providers for TrafficLens-AI backend.
"""

from __future__ import annotations
from typing import Generator
from fastapi import Depends
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.db.session import get_db


def get_app_settings() -> Settings:
    """Dependency providing cached application settings."""
    return get_settings()


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI request-scoped database session dependency."""
    yield from get_db()
