"""
Health check endpoints for TrafficLens-AI backend.
"""

from __future__ import annotations
from typing import Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.core.config import Settings
from backend.app.dependencies import get_app_settings, get_db_session

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=Dict[str, Any])
def health_check(
    settings: Settings = Depends(get_app_settings),
    db: Session = Depends(get_db_session),
) -> Dict[str, Any]:
    """Returns service health status, version, and verified database connectivity."""
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    return {
        "status": "healthy" if db_connected else "degraded",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "database_dialect": "sqlite" if settings.is_sqlite else "postgresql",
        "database_connected": db_connected,
    }
