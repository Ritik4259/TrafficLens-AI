"""
Aggregated API router for TrafficLens-AI backend.
"""

from __future__ import annotations
from fastapi import APIRouter

from backend.app.api.routes import health, jobs, videos, interpretation, incident_reports

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(videos.router)
api_router.include_router(jobs.router)
api_router.include_router(interpretation.router)
api_router.include_router(incident_reports.router)
