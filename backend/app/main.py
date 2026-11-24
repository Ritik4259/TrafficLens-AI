"""
FastAPI application entrypoint for TrafficLens-AI backend.
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging, get_logger
from backend.app.api.router import api_router
from backend.app.services.worker import get_job_worker
from backend.app.services.vlm_worker import get_interpretation_worker


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context managing startup and shutdown routines."""
    settings = get_settings()
    setup_logging(level=settings.LOG_LEVEL, log_format=settings.LOG_FORMAT)
    logger = get_logger("trafficlens.lifecycle")
    logger.info(
        f"Starting {settings.PROJECT_NAME} v{settings.VERSION} "
        f"[env={settings.ENVIRONMENT}, db={settings.DATABASE_URL}]"
    )

    # Startup logic (storage directory creation)
    settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Start background workers with fault isolation
    try:
        worker = get_job_worker()
        worker.start()
    except Exception as e:
        logger.error(f"Error starting background CV worker: {e}", exc_info=True)

    try:
        interpretation_worker = get_interpretation_worker()
        interpretation_worker.start()
    except Exception as e:
        logger.error(f"Error starting background interpretation worker: {e}", exc_info=True)

    yield

    # Graceful shutdown with independent fault isolation
    try:
        interpretation_worker = get_interpretation_worker()
        interpretation_worker.stop(timeout=5.0)
    except Exception as e:
        logger.error(f"Error stopping interpretation worker during shutdown: {e}", exc_info=True)

    try:
        worker = get_job_worker()
        worker.stop(timeout=5.0)
    except Exception as e:
        logger.error(f"Error stopping CV worker during shutdown: {e}", exc_info=True)

    logger.info(f"Shutting down {settings.PROJECT_NAME} backend.")


def create_application() -> FastAPI:
    """Application factory for FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "Intelligent traffic understanding and analytics platform using computer vision, "
            "multi-object tracking, and spatiotemporal traffic telemetry."
        ),
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    app.include_router(api_router, prefix=settings.API_V1_STR)

    @app.get("/", tags=["Root"])
    def root():
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "docs_url": "/docs",
            "health_url": f"{settings.API_V1_STR}/health",
        }

    return app


app = create_application()
