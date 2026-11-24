"""
LangGraph Workflow for Evidence-Grounded Traffic Incident Report Generation.

Implements a bounded, deterministic StateGraph with 5 sequential nodes:
1. retrieve_evidence: Fetches ProcessingJob CV analytics and latest VLM interpretation
2. normalize_evidence: Merges CV metrics and VLM context into NormalizedEvidenceContext
3. extract_traffic_events: Identifies and formats significant traffic events
4. generate_report: Generates structured IncidentReportSchema via configured LLM provider
5. validate_report: Runs deterministic GroundingValidator to verify claims against evidence
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from sqlalchemy.orm import Session
from langgraph.graph import StateGraph, START, END

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.models.job import ProcessingJob
from backend.app.models.scene_interpretation import SceneInterpretation, SceneInterpretationStatus
from backend.app.repositories.job_repository import SQLAlchemyJobRepository
from backend.app.repositories.scene_interpretation_repository import (
    SQLAlchemySceneInterpretationRepository,
)
from backend.app.schemas.incident_report import (
    IncidentReportSchema,
    ReportValidationDiagnosticsSchema,
)
from backend.app.schemas.job import JobStatus
from backend.app.services.agent.evidence_normalizer import (
    EvidenceNormalizer,
    NormalizedEvidenceContext,
)
from backend.app.services.agent.grounding_validator import GroundingValidator
from backend.app.services.agent.provider import (
    GeminiIncidentReportProvider,
    IncidentReportLLMProvider,
    MockIncidentReportProvider,
)

logger = get_logger("trafficlens.agent.graph")


class IncidentAgentState(TypedDict, total=False):
    """Execution state tracked across the LangGraph incident workflow."""
    job_id: str
    db: Optional[Session]
    video_id: str
    sequence_name: str
    job_model: Optional[ProcessingJob]
    cv_results: Optional[Dict[str, Any]]
    scene_interpretation: Optional[Any]
    evidence_context: Optional[NormalizedEvidenceContext]
    extracted_events: Optional[List[Dict[str, Any]]]
    raw_report: Optional[IncidentReportSchema]
    validated_report: Optional[IncidentReportSchema]
    validation_diagnostics: Optional[ReportValidationDiagnosticsSchema]
    status: str
    errors: List[str]


def _build_nodes(
    provider: IncidentReportLLMProvider,
    settings: Optional[Settings] = None,
):
    """Builds node functions bound to the configured LLM provider and settings."""
    cfg = settings or get_settings()

    async def retrieve_evidence(state: IncidentAgentState) -> Dict[str, Any]:
        """Node 1: Retrieves CV telemetry and completed VLM scene interpretation from DB."""
        job_id = state.get("job_id", "")
        db = state.get("db")
        errors: List[str] = list(state.get("errors") or [])

        # If cv_results was pre-supplied (e.g. unit tests without full DB), use it
        if state.get("cv_results") is not None and state.get("sequence_name"):
            return {
                "status": "EVIDENCE_RETRIEVED",
                "errors": errors,
            }

        if not db:
            errors.append(f"No database session provided to retrieve evidence for job '{job_id}'.")
            return {"status": "FAILED", "errors": errors}

        job_repo = SQLAlchemyJobRepository(db)
        job = job_repo.get_job_model(job_id)
        if not job:
            errors.append(f"ProcessingJob '{job_id}' not found.")
            return {"status": "FAILED", "errors": errors}

        if job.status != JobStatus.COMPLETED:
            errors.append(
                f"ProcessingJob '{job_id}' is in status '{job.status.value}'. "
                "Incident report can only be generated for COMPLETED jobs."
            )
            return {"status": "FAILED", "errors": errors}

        # Retrieve CV results
        cv_results = (job.config_metadata or {}).get("results")
        if not cv_results:
            # Check filesystem artifacts fallback
            output_dir = cfg.OUTPUT_DIR / job.id
            if output_dir.exists():
                cv_results = {}
                for name, fname in [
                    ("analytics_summary", "analytics_summary.json"),
                    ("detection_summary", "detection_summary.json"),
                    ("tracking_summary", "tracking_summary.json"),
                ]:
                    fpath = output_dir / fname
                    if fpath.exists():
                        try:
                            cv_results[name] = json.loads(fpath.read_text(encoding="utf-8"))
                        except Exception:
                            pass

        if not cv_results:
            errors.append(f"No CV telemetry results available for completed job '{job_id}'.")
            return {"status": "FAILED", "errors": errors}

        # Retrieve latest completed SceneInterpretation
        scene_repo = SQLAlchemySceneInterpretationRepository(db)
        scene_interp = scene_repo.get_latest_by_job_id(job_id)
        # Check if completed
        if scene_interp and scene_interp.status != SceneInterpretationStatus.COMPLETED:
            logger.info(
                f"Scene interpretation for job '{job_id}' is in status '{scene_interp.status.value}', not COMPLETED. "
                "Proceeding with CV telemetry alone."
            )
            scene_interp = None

        return {
            "job_model": job,
            "video_id": job.video_id,
            "sequence_name": job.sequence_name,
            "cv_results": cv_results,
            "scene_interpretation": scene_interp,
            "status": "EVIDENCE_RETRIEVED",
            "errors": errors,
        }

    async def normalize_evidence(state: IncidentAgentState) -> Dict[str, Any]:
        """Node 2: Harmonizes CV and VLM evidence into NormalizedEvidenceContext."""
        if state.get("errors"):
            return {"status": "FAILED"}

        job_id = state["job_id"]
        video_id = state.get("video_id", "unknown")
        seq_name = state.get("sequence_name", "unknown")
        cv_results = state.get("cv_results") or {}
        scene_interp = state.get("scene_interpretation")

        evidence_context = EvidenceNormalizer.normalize(
            job_id=job_id,
            video_id=video_id,
            sequence_name=seq_name,
            cv_results=cv_results,
            scene_interpretation=scene_interp,
        )

        return {
            "evidence_context": evidence_context,
            "status": "EVIDENCE_NORMALIZED",
        }

    async def extract_traffic_events(state: IncidentAgentState) -> Dict[str, Any]:
        """Node 3: Identifies and normalizes key traffic events from evidence context."""
        if state.get("errors"):
            return {"status": "FAILED"}

        evidence = state.get("evidence_context")
        if not evidence:
            return {"status": "FAILED", "errors": ["Missing evidence context in event extraction node."]}

        events: List[Dict[str, Any]] = []
        # 1. Stopped vehicle events
        for stop in evidence.stopped_events:
            events.append({
                "type": "stopped_vehicle",
                "track_id": stop.track_id,
                "timestamp_s": stop.start_timestamp_s,
                "duration_s": stop.duration_s,
                "roi_id": stop.roi_id,
                "probable_cause": stop.vlm_probable_cause,
            })

        # 2. Peak occupancy events
        for roi_id, occ in evidence.peak_occupancy_per_roi.items():
            if occ >= 0.35:
                events.append({
                    "type": "elevated_occupancy",
                    "roi_id": roi_id,
                    "peak_occupancy": occ,
                })

        return {
            "extracted_events": events,
            "status": "EVENTS_EXTRACTED",
        }

    async def generate_report(state: IncidentAgentState) -> Dict[str, Any]:
        """Node 4: Generates structured incident report using configured LLM provider."""
        if state.get("errors"):
            return {"status": "FAILED"}

        evidence = state.get("evidence_context")
        if not evidence:
            return {"status": "FAILED", "errors": ["Missing evidence context in report generation node."]}

        errors: List[str] = list(state.get("errors") or [])
        try:
            raw_report = await provider.generate_report(evidence)
            return {
                "raw_report": raw_report,
                "status": "REPORT_GENERATED",
            }
        except Exception as e:
            logger.error(f"Incident report generation failed via provider: {e}", exc_info=True)
            errors.append(f"LLM provider failed to generate report: {str(e)}")
            return {
                "status": "FAILED",
                "errors": errors,
            }

    async def validate_report(state: IncidentAgentState) -> Dict[str, Any]:
        """Node 5: Validates and corrects report claims against ground-truth evidence."""
        if state.get("errors"):
            return {"status": "FAILED"}

        raw_report = state.get("raw_report")
        evidence = state.get("evidence_context")
        if not raw_report or not evidence:
            return {"status": "FAILED", "errors": ["Missing report or evidence in validation node."]}

        try:
            validated_report, diagnostics = GroundingValidator.validate_and_correct(
                report=raw_report,
                evidence=evidence,
            )
            return {
                "validated_report": validated_report,
                "validation_diagnostics": diagnostics,
                "status": "REPORT_VALIDATED",
            }
        except Exception as ve:
            logger.error(f"GroundingValidator exception in validate_report node: {ve}", exc_info=True)
            errs = list(state.get("errors") or [])
            errs.append(f"Grounding validation encountered an internal error: {str(ve)}")
            return {
                "status": "FAILED",
                "errors": errs,
            }

    return (
        retrieve_evidence,
        normalize_evidence,
        extract_traffic_events,
        generate_report,
        validate_report,
    )


def create_incident_agent_graph(
    provider: Optional[IncidentReportLLMProvider] = None,
    settings: Optional[Settings] = None,
) -> Any:
    """
    Constructs and compiles the bounded 5-node LangGraph Incident Agent StateGraph.
    
    Flow:
    START -> retrieve_evidence -> normalize_evidence -> extract_traffic_events -> generate_report -> validate_report -> END
    """
    cfg = settings or get_settings()
    llm_provider = provider or (
        GeminiIncidentReportProvider(settings=cfg)
        if cfg.GEMINI_API_KEY
        else MockIncidentReportProvider()
    )

    (
        retrieve_node,
        normalize_node,
        extract_node,
        generate_node,
        validate_node,
    ) = _build_nodes(llm_provider, cfg)

    builder = StateGraph(IncidentAgentState)

    builder.add_node("retrieve_evidence", retrieve_node)
    builder.add_node("normalize_evidence", normalize_node)
    builder.add_node("extract_traffic_events", extract_node)
    builder.add_node("generate_report", generate_node)
    builder.add_node("validate_report", validate_node)

    # Deterministic sequential edge sequence
    builder.add_edge(START, "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "normalize_evidence")
    builder.add_edge("normalize_evidence", "extract_traffic_events")
    builder.add_edge("extract_traffic_events", "generate_report")
    builder.add_edge("generate_report", "validate_report")
    builder.add_edge("validate_report", END)

    return builder.compile()
