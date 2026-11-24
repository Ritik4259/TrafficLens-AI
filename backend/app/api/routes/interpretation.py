"""
Scene interpretation endpoints for TrafficLens-AI backend.

Exposes RESTful HTTP APIs for:
- Requesting scene interpretation for completed CV jobs
- Querying interpretation status and verified results
- Retrieving interpretation attempt history
- Retrying failed interpretation attempts with strict attempt-limit enforcement
"""

from __future__ import annotations
import threading
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.core.config import Settings
from backend.app.core.logging import get_logger
from backend.app.dependencies import get_app_settings, get_db_session
from backend.app.models.job import ProcessingJob
from backend.app.models.scene_interpretation import (
    SceneInterpretation,
    SceneInterpretationStatus,
)
from backend.app.repositories.job_repository import SQLAlchemyJobRepository
from backend.app.repositories.scene_interpretation_repository import (
    SQLAlchemySceneInterpretationRepository,
)
from backend.app.schemas.job import JobStatus
from backend.app.schemas.scene_interpretation import (
    SceneInterpretationCreateSchema,
    SceneInterpretationHistoryResponseSchema,
    SceneInterpretationResponseSchema,
)
from backend.app.services.vlm_worker import (
    InterpretationWorker,
    QueueFullError,
    get_interpretation_worker,
)

logger = get_logger("trafficlens.api.interpretation")
router = APIRouter(tags=["Scene Interpretation"])

# Process-level mutex serializing creation and retry requests to prevent race conditions
_interpretation_create_lock = threading.Lock()


# ==============================================================================
# 1. Trigger / Request Scene Interpretation
# ==============================================================================

@router.post(
    "/jobs/{job_id}/interpretations",
    response_model=SceneInterpretationResponseSchema,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request multimodal scene interpretation for a completed CV job",
)
@router.post(
    "/jobs/{job_id}/interpretation",
    response_model=SceneInterpretationResponseSchema,
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=False,
)
def request_scene_interpretation(
    job_id: str,
    payload: Optional[SceneInterpretationCreateSchema] = None,
    db: Session = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> SceneInterpretationResponseSchema:
    """
    Submits a scene interpretation task for an existing completed CV processing job.
    
    Enforces:
    - CV job must exist (404 Not Found)
    - CV job must be in COMPLETED status (409 Conflict if active, 400 Bad Request if failed)
    - Maximum attempt limit per job (settings.MAX_INTERPRETATION_ATTEMPTS) (409 Conflict)
    - Single active interpretation invariant (409 Conflict if already QUEUED or PROCESSING)
    - Asynchronous execution: enqueues to InterpretationWorker without blocking the HTTP request
    - Safe queue capacity handling: returns 503 and rolls back record if queue is full
    """
    job_repo = SQLAlchemyJobRepository(db)
    interp_repo = SQLAlchemySceneInterpretationRepository(db)

    with _interpretation_create_lock:
        # 1. Verify job exists
        job = job_repo.get_job_model(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processing job '{job_id}' not found.",
            )

        # 2. Assert CV job status
        if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"CV processing is still in progress (status: {job.status.value}). "
                    "Scene interpretation can only be requested once CV processing is COMPLETED."
                ),
            )
        if job.status == JobStatus.FAILED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot interpret a failed CV job. Valid computer vision telemetry is required.",
            )

        # 3. Check attempt limit
        current_attempts = interp_repo.count_by_job_id(job_id)
        if current_attempts >= settings.MAX_INTERPRETATION_ATTEMPTS:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Maximum interpretation attempt limit ({settings.MAX_INTERPRETATION_ATTEMPTS}) "
                    f"exhausted for job '{job_id}'."
                ),
            )

        # 4. Check single active interpretation constraint
        active = interp_repo.get_active_for_job(job_id)
        if active is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"An active interpretation ('{active.id}') is already in status "
                    f"'{active.status.value}' for job '{job_id}'."
                ),
            )

        # 5. Persist new QUEUED record
        interp_id = f"interp_{uuid.uuid4().hex[:12]}"
        provider_type = (
            payload.provider_type
            if (payload and payload.provider_type)
            else settings.VLM_PROVIDER_TYPE
        )
        model_name = (
            payload.model_name
            if (payload and payload.model_name)
            else (settings.GEMINI_MODEL or settings.VLM_MODEL_NAME or "gemini-3.8-flash")
        )

        new_interp = SceneInterpretation(
            id=interp_id,
            job_id=job_id,
            status=SceneInterpretationStatus.QUEUED,
            provider_type=provider_type,
            model_name=model_name,
        )

        try:
            interp_repo.create(new_interp)
            db.commit()
            db.refresh(new_interp)
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"An active interpretation task was concurrently created for job '{job_id}'.",
            )

        # 6. Enqueue to background worker safely
        worker = get_interpretation_worker()
        try:
            worker.enqueue_interpretation(interp_id)
        except QueueFullError:
            # Roll back created record so database does not contain a stranded QUEUED row
            logger.error(
                f"Queue capacity exceeded while enqueueing interpretation '{interp_id}' for job '{job_id}'. "
                "Rolling back database record."
            )
            try:
                db.delete(new_interp)
                db.commit()
            except Exception as e:
                logger.error(f"Failed to delete orphaned record {interp_id}: {e}")

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Interpretation worker queue is full (max: {settings.INTERPRETATION_MAX_QUEUE_SIZE}). "
                    "Please retry later."
                ),
                headers={"Retry-After": "5"},
            )

        logger.info(
            f"Successfully queued scene interpretation '{interp_id}' for job '{job_id}' "
            f"(attempt {current_attempts + 1}/{settings.MAX_INTERPRETATION_ATTEMPTS})."
        )
        return SceneInterpretationResponseSchema.model_validate(new_interp)


# ==============================================================================
# 2. Get Interpretation Status and Results
# ==============================================================================

@router.get(
    "/interpretations/{interpretation_id}",
    response_model=SceneInterpretationResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Get scene interpretation status and analytical results by interpretation ID",
)
def get_interpretation_by_id(
    interpretation_id: str,
    db: Session = Depends(get_db_session),
) -> SceneInterpretationResponseSchema:
    """
    Retrieves a scene interpretation by its unique identifier.
    
    Enforces incomplete-result protection:
    - If status is COMPLETED: returns analytical fields (summary, context, events, claims).
    - If status is FAILED: returns sanitized error information.
    - If status is QUEUED or PROCESSING: analytical fields and error information are strictly None.
    """
    interp_repo = SQLAlchemySceneInterpretationRepository(db)
    interp = interp_repo.get_by_id(interpretation_id, load_job=True)
    if not interp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene interpretation '{interpretation_id}' not found.",
        )

    return SceneInterpretationResponseSchema.model_validate(interp)


@router.get(
    "/jobs/{job_id}/interpretation",
    response_model=SceneInterpretationResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Get the latest scene interpretation for a processing job",
)
@router.get(
    "/jobs/{job_id}/interpretations/latest",
    response_model=SceneInterpretationResponseSchema,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
def get_latest_interpretation_for_job(
    job_id: str,
    db: Session = Depends(get_db_session),
) -> SceneInterpretationResponseSchema:
    """
    Retrieves the most recent scene interpretation attempt for a job.
    
    If no interpretation has ever been requested for the job, returns a virtual
    NOT_REQUESTED status response without raising an error.
    """
    job_repo = SQLAlchemyJobRepository(db)
    interp_repo = SQLAlchemySceneInterpretationRepository(db)

    job = job_repo.get_job_model(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Processing job '{job_id}' not found.",
        )

    latest = interp_repo.get_latest_by_job_id(job_id)
    if not latest:
        return SceneInterpretationResponseSchema.not_requested(
            job_id=job_id,
            video_id=job.video_id,
        )

    return SceneInterpretationResponseSchema.model_validate(latest)


# ==============================================================================
# 3. Get Interpretation History for a Job
# ==============================================================================

@router.get(
    "/jobs/{job_id}/interpretations",
    response_model=SceneInterpretationHistoryResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Get all historical scene interpretation attempts for a processing job",
)
def get_interpretation_history(
    job_id: str,
    ascending: bool = True,
    db: Session = Depends(get_db_session),
) -> SceneInterpretationHistoryResponseSchema:
    """
    Retrieves the full historical list of interpretation attempts for a given job
    in chronological order (oldest first by default, or newest first if ascending=False).
    """
    job_repo = SQLAlchemyJobRepository(db)
    interp_repo = SQLAlchemySceneInterpretationRepository(db)

    job = job_repo.get_job_model(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Processing job '{job_id}' not found.",
        )

    attempts = interp_repo.list_by_job_id(job_id, ascending=ascending)
    return SceneInterpretationHistoryResponseSchema(
        job_id=job_id,
        total_attempts=len(attempts),
        attempts=[SceneInterpretationResponseSchema.model_validate(a) for a in attempts],
    )


# ==============================================================================
# 4. Retry a Failed Interpretation
# ==============================================================================

@router.post(
    "/jobs/{job_id}/interpretations/retry",
    response_model=SceneInterpretationResponseSchema,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Retry a failed scene interpretation by creating a new attempt",
)
@router.post(
    "/jobs/{job_id}/interpretation/retry",
    response_model=SceneInterpretationResponseSchema,
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=False,
)
def retry_failed_interpretation(
    job_id: str,
    db: Session = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> SceneInterpretationResponseSchema:
    """
    Retries an interpretation whose most recent attempt was FAILED.
    
    Enforces:
    - Most recent attempt must be FAILED (400 Bad Request if COMPLETED, QUEUED, or PROCESSING)
    - Total attempts cannot exceed settings.MAX_INTERPRETATION_ATTEMPTS (409 Conflict)
    - Single active interpretation invariant (409 Conflict)
    - Immutable history: previous failed records are preserved intact for auditability
    - Queue capacity handling: returns 503 if queue is full
    """
    job_repo = SQLAlchemyJobRepository(db)
    interp_repo = SQLAlchemySceneInterpretationRepository(db)

    with _interpretation_create_lock:
        # 1. Verify job exists
        job = job_repo.get_job_model(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processing job '{job_id}' not found.",
            )

        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot retry interpretation: CV job is in status '{job.status.value}', must be COMPLETED.",
            )

        # 2. Check previous attempts
        latest = interp_repo.get_latest_by_job_id(job_id)
        if not latest:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"No prior interpretation attempt found for job '{job_id}'. "
                    "Use POST /jobs/{job_id}/interpretations to initiate the first interpretation."
                ),
            )

        if latest.status != SceneInterpretationStatus.FAILED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Cannot retry interpretation: latest attempt '{latest.id}' is in status "
                    f"'{latest.status.value}'. Only FAILED interpretations can be retried."
                ),
            )

        # 3. Check attempt limit
        current_attempts = interp_repo.count_by_job_id(job_id)
        if current_attempts >= settings.MAX_INTERPRETATION_ATTEMPTS:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Maximum interpretation attempt limit ({settings.MAX_INTERPRETATION_ATTEMPTS}) "
                    f"exhausted for job '{job_id}'."
                ),
            )

        # 4. Check active interpretation
        active = interp_repo.get_active_for_job(job_id)
        if active is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"An active interpretation ('{active.id}') is already in status "
                    f"'{active.status.value}' for job '{job_id}'."
                ),
            )

        # 5. Log warning regarding remote provider retry
        logger.warning(
            f"Retrying scene interpretation for job '{job_id}' "
            f"(attempt {current_attempts + 1}/{settings.MAX_INTERPRETATION_ATTEMPTS}). "
            "This will trigger a new remote VLM inference request."
        )

        # 6. Create NEW SceneInterpretation record (immutable attempt history)
        new_interp_id = f"interp_{uuid.uuid4().hex[:12]}"
        new_interp = SceneInterpretation(
            id=new_interp_id,
            job_id=job_id,
            status=SceneInterpretationStatus.QUEUED,
            provider_type=latest.provider_type,
            model_name=latest.model_name,
        )

        try:
            interp_repo.create(new_interp)
            db.commit()
            db.refresh(new_interp)
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"An active interpretation task was concurrently created for job '{job_id}'.",
            )

        # 7. Enqueue to background worker
        worker = get_interpretation_worker()
        try:
            worker.enqueue_interpretation(new_interp_id)
        except QueueFullError:
            logger.error(
                f"Queue capacity exceeded while enqueueing retry interpretation '{new_interp_id}' for job '{job_id}'. "
                "Rolling back database record."
            )
            try:
                db.delete(new_interp)
                db.commit()
            except Exception as e:
                logger.error(f"Failed to delete orphaned retry record {new_interp_id}: {e}")

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Interpretation worker queue is full (max: {settings.INTERPRETATION_MAX_QUEUE_SIZE}). "
                    "Please retry later."
                ),
                headers={"Retry-After": "5"},
            )

        return SceneInterpretationResponseSchema.model_validate(new_interp)


@router.post(
    "/interpretations/{interpretation_id}/retry",
    response_model=SceneInterpretationResponseSchema,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Retry a specific failed scene interpretation attempt",
)
def retry_interpretation_by_id(
    interpretation_id: str,
    db: Session = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
) -> SceneInterpretationResponseSchema:
    """
    Retries a specific failed interpretation attempt by delegating to the job retry handler.
    Validates that the specified interpretation exists and is FAILED.
    """
    interp_repo = SQLAlchemySceneInterpretationRepository(db)
    interp = interp_repo.get_by_id(interpretation_id)
    if not interp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene interpretation '{interpretation_id}' not found.",
        )

    if interp.status != SceneInterpretationStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Cannot retry interpretation '{interpretation_id}': status is '{interp.status.value}'. "
                "Only FAILED interpretations can be retried."
            ),
        )

    return retry_failed_interpretation(
        job_id=interp.job_id,
        db=db,
        settings=settings,
    )
