"""
Concrete SQLAlchemy-backed SceneInterpretation repository implementation.

Provides database persistence, relationship loading, concurrency-safe conditional state
transitions, token/latency accounting, and atomic completion/failure persistence.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Union
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from backend.app.models.scene_interpretation import (
    SceneInterpretation,
    SceneInterpretationStatus,
)
from backend.app.schemas.scene_interpretation import (
    SceneInterpretationResponseSchema,
    SceneInterpretationHistoryResponseSchema,
    validate_interpretation_status_transition,
    sanitize_error_text,
    sanitize_error_details,
)


class SceneInterpretationRepository(Protocol):
    """Protocol defining the repository contract for scene interpretation persistence."""

    def create(self, interpretation: SceneInterpretation) -> SceneInterpretation:
        """Persists a new scene interpretation record."""
        ...

    def get_by_id(self, interpretation_id: str, load_job: bool = True) -> Optional[SceneInterpretation]:
        """Retrieves a scene interpretation by ID."""
        ...

    def get_latest_by_job_id(self, job_id: str) -> Optional[SceneInterpretation]:
        """Retrieves the most recent scene interpretation for a job."""
        ...

    def get_active_for_job(self, job_id: str) -> Optional[SceneInterpretation]:
        """Retrieves the active (QUEUED or PROCESSING) interpretation for a job."""
        ...

    def list_by_job_id(self, job_id: str, ascending: bool = True) -> List[SceneInterpretation]:
        """Lists all interpretation attempts for a job in chronological order."""
        ...

    def count_by_job_id(self, job_id: str) -> int:
        """Returns the total number of interpretation attempts recorded for a job."""
        ...

    def list_queued(self, limit: int = 50) -> List[SceneInterpretation]:
        """Retrieves QUEUED interpretations for worker scheduling."""
        ...

    def list_processing(self) -> List[SceneInterpretation]:
        """Retrieves PROCESSING interpretations for startup crash recovery."""
        ...

    def claim_for_processing(
        self,
        interpretation_id: str,
        started_at: Optional[datetime] = None,
    ) -> Optional[SceneInterpretation]:
        """Conditionally claims a QUEUED interpretation for PROCESSING."""
        ...

    def complete(
        self,
        interpretation_id: str,
        summary: Optional[str] = None,
        environmental_context: Optional[Dict[str, Any]] = None,
        scene_description: Optional[Dict[str, Any]] = None,
        event_interpretations: Optional[List[Dict[str, Any]]] = None,
        epistemic_claims: Optional[List[Dict[str, Any]]] = None,
        unverified_aspects: Optional[List[str]] = None,
        evidence_package_id: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        completed_at: Optional[datetime] = None,
        duration_seconds: Optional[float] = None,
    ) -> Optional[SceneInterpretation]:
        """Atomically persists validated interpretation results and completion status."""
        ...

    def fail(
        self,
        interpretation_id: str,
        error_code: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        completed_at: Optional[datetime] = None,
        duration_seconds: Optional[float] = None,
    ) -> Optional[SceneInterpretation]:
        """Atomically persists failure diagnostics and failure status."""
        ...

    def update_status(
        self,
        interpretation_id: str,
        target_status: SceneInterpretationStatus,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        duration_seconds: Optional[float] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> Optional[SceneInterpretation]:
        """Updates lifecycle status enforcing state machine validation."""
        ...


class SQLAlchemySceneInterpretationRepository:
    """
    Concrete SQLAlchemy 2.0 repository managing SceneInterpretation entities.
    """

    def __init__(self, session: Session):
        self.session = session

    # --------------------------------------------------------------------------
    # Creation & Retrieval Operations
    # --------------------------------------------------------------------------

    def create(self, interp: SceneInterpretation) -> SceneInterpretation:
        """
        Persists a new SceneInterpretation record.
        
        Guarantees that on an IntegrityError (e.g. violating the partial unique
        index uix_active_interpretation_per_job), the session is cleanly rolled back
        before re-raising the exception.
        """
        try:
            self.session.add(interp)
            self.session.flush()
            return interp
        except IntegrityError:
            self.session.rollback()
            raise

    def get_by_id(self, interpretation_id: str, load_job: bool = True) -> Optional[SceneInterpretation]:
        """Retrieves a scene interpretation by its unique identifier."""
        stmt = select(SceneInterpretation).where(SceneInterpretation.id == interpretation_id)
        if load_job:
            stmt = stmt.options(joinedload(SceneInterpretation.job))
        return self.session.scalars(stmt).first()

    def get_latest_by_job_id(self, job_id: str) -> Optional[SceneInterpretation]:
        """Retrieves the most recent scene interpretation attempt for a job."""
        stmt = (
            select(SceneInterpretation)
            .where(SceneInterpretation.job_id == job_id)
            .order_by(SceneInterpretation.created_at.desc())
            .options(joinedload(SceneInterpretation.job))
        )
        return self.session.scalars(stmt).first()

    def get_active_for_job(self, job_id: str) -> Optional[SceneInterpretation]:
        """
        Retrieves the active interpretation (QUEUED or PROCESSING) for a job.
        Returns None if no active interpretation exists.
        """
        stmt = (
            select(SceneInterpretation)
            .where(
                SceneInterpretation.job_id == job_id,
                SceneInterpretation.status.in_([
                    SceneInterpretationStatus.QUEUED,
                    SceneInterpretationStatus.PROCESSING,
                ]),
            )
            .options(joinedload(SceneInterpretation.job))
        )
        return self.session.scalars(stmt).first()

    def list_by_job_id(self, job_id: str, ascending: bool = True) -> List[SceneInterpretation]:
        """
        Lists all interpretation attempts for a job in chronological order.
        Pass ascending=False for reverse-chronological order.
        """
        order_clause = (
            SceneInterpretation.created_at.asc()
            if ascending
            else SceneInterpretation.created_at.desc()
        )
        stmt = (
            select(SceneInterpretation)
            .where(SceneInterpretation.job_id == job_id)
            .order_by(order_clause)
            .options(joinedload(SceneInterpretation.job))
        )
        return list(self.session.scalars(stmt).all())

    def count_by_job_id(self, job_id: str) -> int:
        """Returns the total number of interpretation attempts recorded for a CV job."""
        stmt = (
            select(func.count(SceneInterpretation.id))
            .where(SceneInterpretation.job_id == job_id)
        )
        return self.session.scalar(stmt) or 0

    def list_queued(self, limit: int = 50) -> List[SceneInterpretation]:
        """
        Retrieves QUEUED interpretations for worker scheduling in FIFO order.
        """
        stmt = (
            select(SceneInterpretation)
            .where(SceneInterpretation.status == SceneInterpretationStatus.QUEUED)
            .order_by(SceneInterpretation.created_at.asc())
            .limit(limit)
            .options(joinedload(SceneInterpretation.job))
        )
        return list(self.session.scalars(stmt).all())

    def list_processing(self) -> List[SceneInterpretation]:
        """
        Retrieves PROCESSING interpretations for startup crash recovery.
        """
        stmt = (
            select(SceneInterpretation)
            .where(SceneInterpretation.status == SceneInterpretationStatus.PROCESSING)
            .order_by(SceneInterpretation.started_at.asc())
            .options(joinedload(SceneInterpretation.job))
        )
        return list(self.session.scalars(stmt).all())

    # --------------------------------------------------------------------------
    # Lifecycle & Atomic State Transitions
    # --------------------------------------------------------------------------

    def claim_for_processing(
        self,
        interpretation_id: str,
        started_at: Optional[datetime] = None,
    ) -> Optional[SceneInterpretation]:
        """
        Conditionally claims a QUEUED interpretation for PROCESSING.
        Executes an atomic conditional update: WHERE id=:id AND status='QUEUED'.
        Refreshes identity map to ensure returned object never contains stale status values.
        Returns the updated SceneInterpretation model, or None if already claimed.
        """
        now = started_at or datetime.now(timezone.utc)
        stmt = (
            update(SceneInterpretation)
            .where(
                SceneInterpretation.id == interpretation_id,
                SceneInterpretation.status == SceneInterpretationStatus.QUEUED,
            )
            .values(
                status=SceneInterpretationStatus.PROCESSING,
                started_at=now,
            )
        )
        result = self.session.execute(stmt)
        self.session.flush()

        if result.rowcount == 0:
            return None

        # Expire / refresh in session identity map to ensure returned object is not stale
        existing = self.session.get(SceneInterpretation, interpretation_id)
        if existing is not None:
            self.session.refresh(existing)
            return existing
        return self.get_by_id(interpretation_id)

    def complete(
        self,
        interpretation_id: str,
        summary: Optional[str] = None,
        environmental_context: Optional[Dict[str, Any]] = None,
        scene_description: Optional[Dict[str, Any]] = None,
        event_interpretations: Optional[List[Dict[str, Any]]] = None,
        epistemic_claims: Optional[List[Dict[str, Any]]] = None,
        unverified_aspects: Optional[List[str]] = None,
        evidence_package_id: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        completed_at: Optional[datetime] = None,
        duration_seconds: Optional[float] = None,
    ) -> Optional[SceneInterpretation]:
        """
        Atomically persists validated interpretation results, token accounting,
        and transitions status to COMPLETED within the same database transaction.
        """
        interp = self.get_by_id(interpretation_id)
        if not interp:
            return None

        # Refresh from database to ensure in-memory entity reflects any external changes (e.g. shutdown abortion)
        try:
            self.session.refresh(interp)
        except Exception:
            return None

        # Enforce lifecycle transition rules
        validate_interpretation_status_transition(interp.status, SceneInterpretationStatus.COMPLETED)

        now = completed_at or datetime.now(timezone.utc)
        dur = duration_seconds
        if dur is None and interp.started_at is not None:
            dur = max(0.0, (now - interp.started_at).total_seconds())

        interp.status = SceneInterpretationStatus.COMPLETED
        interp.completed_at = now
        interp.duration_seconds = dur

        interp.summary = summary
        interp.environmental_context = environmental_context
        interp.scene_description = scene_description
        interp.event_interpretations = event_interpretations
        interp.epistemic_claims = epistemic_claims
        interp.unverified_aspects = unverified_aspects
        interp.evidence_package_id = evidence_package_id

        interp.prompt_tokens = prompt_tokens
        interp.completion_tokens = completion_tokens
        interp.total_tokens = total_tokens

        interp.error_code = None
        interp.error_message = None
        interp.error_details = None

        self.session.flush()
        return interp

    def fail(
        self,
        interpretation_id: str,
        error_code: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        completed_at: Optional[datetime] = None,
        duration_seconds: Optional[float] = None,
    ) -> Optional[SceneInterpretation]:
        """
        Atomically persists sanitized failure diagnostics and transitions status to FAILED
        within the same database transaction.
        """
        interp = self.get_by_id(interpretation_id)
        if not interp:
            return None

        # Refresh from database to ensure in-memory entity reflects any external changes
        try:
            self.session.refresh(interp)
        except Exception:
            return None

        # Enforce lifecycle transition rules
        validate_interpretation_status_transition(interp.status, SceneInterpretationStatus.FAILED)

        now = completed_at or datetime.now(timezone.utc)
        dur = duration_seconds
        if dur is None and interp.started_at is not None:
            dur = max(0.0, (now - interp.started_at).total_seconds())

        interp.status = SceneInterpretationStatus.FAILED
        interp.completed_at = now
        interp.duration_seconds = dur

        interp.error_code = error_code
        interp.error_message = sanitize_error_text(error_message)
        interp.error_details = sanitize_error_details(error_details)

        self.session.flush()
        return interp

    def update_status(
        self,
        interpretation_id: str,
        target_status: SceneInterpretationStatus,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        duration_seconds: Optional[float] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> Optional[SceneInterpretation]:
        """
        Updates lifecycle status validating state transition rules.
        Rejects invalid transitions and mutations of terminal states.
        """
        interp = self.get_by_id(interpretation_id)
        if not interp:
            return None

        validate_interpretation_status_transition(interp.status, target_status)

        interp.status = target_status
        if started_at is not None:
            interp.started_at = started_at
        if completed_at is not None:
            interp.completed_at = completed_at
        if duration_seconds is not None:
            interp.duration_seconds = duration_seconds
        if error_code is not None:
            interp.error_code = error_code
        if error_message is not None:
            interp.error_message = sanitize_error_text(error_message)
        if error_details is not None:
            interp.error_details = sanitize_error_details(error_details)

        self.session.flush()
        return interp

    # --------------------------------------------------------------------------
    # Schema Conversion Helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def to_response_schema(interp: SceneInterpretation) -> SceneInterpretationResponseSchema:
        """Converts an ORM model into SceneInterpretationResponseSchema."""
        return SceneInterpretationResponseSchema.model_validate(interp)

    def to_history_schema(
        self,
        job_id: str,
        attempts: Optional[List[SceneInterpretation]] = None,
    ) -> SceneInterpretationHistoryResponseSchema:
        """Converts a job's attempts into SceneInterpretationHistoryResponseSchema."""
        records = attempts if attempts is not None else self.list_by_job_id(job_id)
        total = self.count_by_job_id(job_id)
        return SceneInterpretationHistoryResponseSchema(
            job_id=job_id,
            total_attempts=total,
            attempts=[self.to_response_schema(m) for m in records],
        )
