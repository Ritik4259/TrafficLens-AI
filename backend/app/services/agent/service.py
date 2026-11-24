"""
Domain service coordinating the LangGraph Incident Agent and database persistence.
"""

from __future__ import annotations
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.models.incident_report import IncidentReport
from backend.app.repositories.job_repository import SQLAlchemyJobRepository
from backend.app.repositories.incident_report_repository import (
    SQLAlchemyIncidentReportRepository,
)
from backend.app.schemas.incident_report import (
    IncidentReportHistoryResponseSchema,
    IncidentReportResponseSchema,
    IncidentReportSchema,
)
from backend.app.schemas.job import JobStatus
from backend.app.services.agent.graph import create_incident_agent_graph
from backend.app.services.agent.provider import (
    IncidentReportLLMProvider,
    MockIncidentReportProvider,
    GeminiIncidentReportProvider,
)

logger = get_logger("trafficlens.agent.service")


class IncidentReportService:
    """Coordinates execution of the LangGraph incident agent and audit persistence."""

    def __init__(
        self,
        db: Session,
        settings: Optional[Settings] = None,
        provider: Optional[IncidentReportLLMProvider] = None,
    ):
        self.db = db
        self.settings = settings or get_settings()
        self.repo = SQLAlchemyIncidentReportRepository(db)
        self.job_repo = SQLAlchemyJobRepository(db)
        self.provider = provider or (
            GeminiIncidentReportProvider(settings=self.settings)
            if self.settings.GEMINI_API_KEY
            else MockIncidentReportProvider()
        )

    async def generate_incident_report(
        self,
        job_id: str,
        provider_override: Optional[IncidentReportLLMProvider] = None,
    ) -> IncidentReportResponseSchema:
        """
        Executes the LangGraph incident agent workflow and persists the generated report.

        Args:
            job_id: ProcessingJob identifier.
            provider_override: Optional custom provider (used for testing or model switching).

        Returns:
            IncidentReportResponseSchema.
        """
        # 1. Verify Job existence and status
        job = self.job_repo.get_job_model(job_id)
        if not job:
            raise KeyError(f"ProcessingJob '{job_id}' not found.")

        if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
            raise ValueError(
                f"Job '{job_id}' is currently {job.status.value}. "
                "Incident report can only be generated for COMPLETED jobs."
            )

        if job.status == JobStatus.FAILED:
            raise ValueError(
                f"Job '{job_id}' failed: {job.error_message or 'Unknown error'}. "
                "Cannot generate incident report for failed jobs."
            )

        active_provider = provider_override or self.provider
        graph = create_incident_agent_graph(provider=active_provider, settings=self.settings)

        initial_state = {
            "job_id": job_id,
            "db": self.db,
            "job_model": job,
            "video_id": job.video_id,
            "sequence_name": job.sequence_name,
            "errors": [],
        }

        timeout_s = getattr(self.settings, "INCIDENT_AGENT_TIMEOUT", 30.0)

        try:
            final_state = await asyncio.wait_for(
                graph.ainvoke(initial_state),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            err_msg = f"Incident report generation timed out after {timeout_s:.1f}s."
            logger.error(f"LangGraph execution timed out for job {job_id}")
            now = datetime.now(timezone.utc)
            rep_id = f"rep_{uuid.uuid4().hex[:12]}"
            failed_model = IncidentReport(
                id=rep_id,
                job_id=job_id,
                status="FAILED",
                created_at=now,
                summary=None,
                report_data={},
                validation_passed=False,
                validation_summary=None,
                error_message=err_msg,
            )
            try:
                self.repo.create(failed_model)
                self.db.commit()
            except Exception as db_err:
                self.db.rollback()
                logger.error(f"Failed to persist timeout failure model for job {job_id}: {db_err}")

            return IncidentReportResponseSchema(
                report_id=rep_id,
                job_id=job_id,
                video_id=job.video_id,
                sequence_name=job.sequence_name,
                status="FAILED",
                created_at=now,
                report=None,
                validation_passed=False,
                error_message=err_msg,
            )
        except Exception as exc:
            logger.error(f"LangGraph execution exception for job {job_id}: {exc}", exc_info=True)
            now = datetime.now(timezone.utc)
            rep_id = f"rep_{uuid.uuid4().hex[:12]}"
            failed_model = IncidentReport(
                id=rep_id,
                job_id=job_id,
                status="FAILED",
                created_at=now,
                summary=None,
                report_data={},
                validation_passed=False,
                validation_summary=None,
                error_message=str(exc),
            )
            try:
                self.repo.create(failed_model)
                self.db.commit()
            except Exception as db_err:
                self.db.rollback()
                logger.error(f"Failed to persist failure model for job {job_id}: {db_err}")

            return IncidentReportResponseSchema(
                report_id=rep_id,
                job_id=job_id,
                video_id=job.video_id,
                sequence_name=job.sequence_name,
                status="FAILED",
                created_at=now,
                report=None,
                validation_passed=False,
                error_message=str(exc),
            )

        errors = final_state.get("errors") or []
        validated_report: Optional[IncidentReportSchema] = final_state.get("validated_report")
        diagnostics = final_state.get("validation_diagnostics")

        now = datetime.now(timezone.utc)
        rep_id = validated_report.report_id if validated_report else f"rep_{uuid.uuid4().hex[:12]}"
        status = "COMPLETED" if (validated_report and not errors) else "FAILED"
        err_msg = "; ".join(errors) if errors else None

        report_model = IncidentReport(
            id=rep_id,
            job_id=job_id,
            status=status,
            created_at=now,
            summary=validated_report.executive_summary if validated_report else None,
            report_data=validated_report.model_dump(mode="json") if validated_report else {},
            validation_passed=diagnostics.validation_passed if diagnostics else False,
            validation_summary=diagnostics.model_dump(mode="json") if diagnostics else None,
            error_message=err_msg,
        )

        try:
            self.repo.create(report_model)
            self.db.commit()
        except Exception as persist_exc:
            self.db.rollback()
            logger.error(
                f"Failed to persist incident report '{rep_id}' for job '{job_id}': {persist_exc}",
                exc_info=True,
            )
            raise RuntimeError(f"Database persistence failure: {str(persist_exc)}") from persist_exc

        return IncidentReportResponseSchema(
            report_id=rep_id,
            job_id=job_id,
            video_id=job.video_id,
            sequence_name=job.sequence_name,
            status=status,
            created_at=now,
            report=validated_report,
            validation_passed=diagnostics.validation_passed if diagnostics else False,
            error_message=err_msg,
        )

    def get_report(self, report_id: str) -> Optional[IncidentReportResponseSchema]:
        """Retrieves an incident report by ID."""
        model = self.repo.get_by_id(report_id)
        if not model:
            return None
        return SQLAlchemyIncidentReportRepository.to_response_schema(model)

    def get_latest_report_for_job(self, job_id: str) -> Optional[IncidentReportResponseSchema]:
        """Retrieves the latest incident report for a job."""
        model = self.repo.get_latest_by_job_id(job_id)
        if not model:
            return None
        return SQLAlchemyIncidentReportRepository.to_response_schema(model)

    def list_reports_for_job(
        self,
        job_id: str,
        ascending: bool = True,
    ) -> IncidentReportHistoryResponseSchema:
        """Retrieves complete historical audit trail of incident reports for a job."""
        models = self.repo.list_by_job_id(job_id, ascending=ascending)
        reports = [
            SQLAlchemyIncidentReportRepository.to_response_schema(m)
            for m in models
        ]
        return IncidentReportHistoryResponseSchema(
            job_id=job_id,
            total_reports=len(reports),
            reports=reports,
        )
