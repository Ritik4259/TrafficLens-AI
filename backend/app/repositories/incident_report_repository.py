"""
SQLAlchemy-backed IncidentReport repository implementation.

Provides database persistence, query operations, and audit trail management
for Evidence-Grounded Traffic Incident Reports.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models.incident_report import IncidentReport
from backend.app.schemas.incident_report import (
    IncidentReportResponseSchema,
    IncidentReportSchema,
)


class IncidentReportRepository(Protocol):
    """Protocol defining the repository contract for incident report persistence."""

    def create(self, report: IncidentReport) -> IncidentReport:
        """Persists a new incident report record."""
        ...

    def get_by_id(self, report_id: str) -> Optional[IncidentReport]:
        """Retrieves an incident report by ID."""
        ...

    def get_latest_by_job_id(self, job_id: str) -> Optional[IncidentReport]:
        """Retrieves the most recent incident report for a job."""
        ...

    def list_by_job_id(self, job_id: str, ascending: bool = True) -> List[IncidentReport]:
        """Lists all incident reports for a job."""
        ...

    def count_by_job_id(self, job_id: str) -> int:
        """Counts total incident reports generated for a job."""
        ...


class SQLAlchemyIncidentReportRepository:
    """Concrete SQLAlchemy implementation for IncidentReport persistence."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, report: IncidentReport) -> IncidentReport:
        """Persists a new incident report record."""
        self.session.add(report)
        self.session.flush()
        return report

    def get_by_id(self, report_id: str) -> Optional[IncidentReport]:
        """Retrieves an incident report by ID."""
        stmt = select(IncidentReport).where(IncidentReport.id == report_id)
        return self.session.execute(stmt).scalars().first()

    def get_latest_by_job_id(self, job_id: str) -> Optional[IncidentReport]:
        """Retrieves the most recent incident report for a job by created_at DESC."""
        stmt = (
            select(IncidentReport)
            .where(IncidentReport.job_id == job_id)
            .order_by(IncidentReport.created_at.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalars().first()

    def list_by_job_id(self, job_id: str, ascending: bool = True) -> List[IncidentReport]:
        """Lists all incident reports for a job in chronological or reverse-chronological order."""
        stmt = select(IncidentReport).where(IncidentReport.job_id == job_id)
        if ascending:
            stmt = stmt.order_by(IncidentReport.created_at.asc())
        else:
            stmt = stmt.order_by(IncidentReport.created_at.desc())
        return list(self.session.execute(stmt).scalars().all())

    def count_by_job_id(self, job_id: str) -> int:
        """Counts total incident reports generated for a job."""
        stmt = select(func.count(IncidentReport.id)).where(IncidentReport.job_id == job_id)
        return self.session.execute(stmt).scalar() or 0

    @staticmethod
    def to_response_schema(
        report: IncidentReport,
        video_id: str = "unknown",
        sequence_name: str = "unknown",
    ) -> IncidentReportResponseSchema:
        """Converts an ORM IncidentReport into an API response schema."""
        parsed_report = None
        if report.report_data:
            try:
                parsed_report = IncidentReportSchema(**report.report_data)
            except Exception:
                pass

        vid = video_id
        seq = sequence_name
        if parsed_report:
            vid = parsed_report.video_id
            seq = parsed_report.sequence_name
        elif report.job:
            vid = report.job.video_id
            seq = report.job.sequence_name

        return IncidentReportResponseSchema(
            report_id=report.id,
            job_id=report.job_id,
            video_id=vid,
            sequence_name=seq,
            status=report.status,
            created_at=report.created_at,
            report=parsed_report,
            validation_passed=report.validation_passed,
            error_message=report.error_message,
        )
