"""
Multimodal Scene Interpretation Orchestrator Service for TrafficLens-AI.

Coordinates the end-to-end execution pipeline connecting:
1. CV evidence retrieval (completed ProcessingJob results & telemetry)
2. Deterministic KeyframeExtractor (temporal candidate selection & OpenCV decoding)
3. EvidencePackager (structured evidence bundle creation)
4. Vision-Language Model inference via SceneInterpretationService
5. Epistemic verification via GroundingVerifier
6. Single-transaction database persistence via SceneInterpretationRepository

Enforces strict isolation: VLM errors, timeouts, or grounding failures never alter
or fail the successfully completed primary computer vision ProcessingJob.
"""

from __future__ import annotations
import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.models.job import ProcessingJob
from backend.app.models.scene_interpretation import (
    SceneInterpretation,
    SceneInterpretationStatus,
)
from backend.app.models.video import VideoAsset
from backend.app.repositories.job_repository import SQLAlchemyJobRepository
from backend.app.repositories.scene_interpretation_repository import (
    SQLAlchemySceneInterpretationRepository,
)
from backend.app.schemas.job import JobStatus
from backend.app.schemas.scene_interpretation import (
    InvalidInterpretationStatusTransitionError,
)
from backend.app.schemas.vlm import (
    EvidencePackageSchema,
    ExtractionConfigSchema,
    SceneInterpretationResponseSchema,
)
from backend.app.services.vlm.evidence_packager import EvidencePackager
from backend.app.services.vlm.grounding_verifier import (
    verify_scene_interpretation_grounding,
)
from backend.app.services.vlm.keyframe_extractor import (
    KeyframeExtractor,
    VideoReadError,
)
from backend.app.services.vlm.provider import VLMProvider
from backend.app.services.vlm.scene_interpretation_service import (
    SceneInterpretationService,
)

logger = get_logger("trafficlens.vlm.orchestrator")


class InterpretationOrchestratorService:
    """
    Executes the multimodal scene understanding pipeline for a SceneInterpretation task.
    
    Guarantees strict failure containment, epistemic grounding verification,
    and non-blocking interaction with the primary computer vision processing jobs.
    """

    def __init__(
        self,
        db: Session,
        settings: Optional[Settings] = None,
        provider: Optional[VLMProvider] = None,
        scene_interpretation_service: Optional[SceneInterpretationService] = None,
        keyframe_extractor: Optional[KeyframeExtractor] = None,
    ):
        self.db = db
        self.settings = settings or get_settings()
        self.repo = SQLAlchemySceneInterpretationRepository(db)
        self.job_repo = SQLAlchemyJobRepository(db)
        self.extractor = keyframe_extractor or KeyframeExtractor()
        self.scene_service = scene_interpretation_service or SceneInterpretationService(
            provider=provider, settings=self.settings
        )

    async def execute_interpretation(
        self,
        interp_id: str,
        stop_event: Optional[threading.Event] = None,
    ) -> Optional[SceneInterpretation]:
        """
        Executes the end-to-end scene understanding pipeline for a given interpretation task.

        Args:
            interp_id: Unique identifier of the SceneInterpretation record.
            stop_event: Optional cooperative cancellation event.

        Returns:
            Updated SceneInterpretation model upon completion, or None if task could not be processed.
        """
        # ----------------------------------------------------------------------
        # 1. Retrieve interpretation record and verify lifecycle state
        # ----------------------------------------------------------------------
        interp = self.repo.get_by_id(interp_id, load_job=True)
        if not interp:
            logger.error(f"Cannot execute interpretation: record '{interp_id}' not found in database.")
            return None

        # Claim if QUEUED, or proceed if already claimed to PROCESSING by worker
        if interp.status == SceneInterpretationStatus.QUEUED:
            claimed = self.repo.claim_for_processing(interp_id)
            if not claimed:
                logger.warning(
                    f"Interpretation '{interp_id}' could not be claimed for processing (not in QUEUED state). "
                    "Skipping execution."
                )
                return None
            self.db.commit()
            interp = claimed
        elif interp.status != SceneInterpretationStatus.PROCESSING:
            logger.warning(
                f"Interpretation '{interp_id}' is in non-active status '{interp.status.value}'. "
                "Preserving terminal state and skipping execution."
            )
            return interp

        self._check_stop_signal(stop_event, interp_id)

        # ----------------------------------------------------------------------
        # 2. Retrieve associated ProcessingJob and verify completed status
        # ----------------------------------------------------------------------
        job = interp.job or self.job_repo.get_job_model(interp.job_id, load_relationship=True)
        if not job:
            return self._fail(
                interp_id=interp_id,
                error_code="CV_JOB_NOT_FOUND",
                error_message=f"Associated CV processing job '{interp.job_id}' not found.",
            )

        if job.status != JobStatus.COMPLETED:
            return self._fail(
                interp_id=interp_id,
                error_code="CV_JOB_NOT_COMPLETED",
                error_message=(
                    f"Associated CV processing job '{job.id}' is in status '{job.status.value}'. "
                    "Scene interpretation can only run on COMPLETED CV jobs."
                ),
            )

        self._check_stop_signal(stop_event, interp_id)

        # ----------------------------------------------------------------------
        # 3. Retrieve VideoAsset and verify filesystem availability
        # ----------------------------------------------------------------------
        video_asset = job.video_asset or self.job_repo.get_video_asset(job.video_id)
        if not video_asset or not Path(video_asset.storage_path).exists():
            storage_path = video_asset.storage_path if video_asset else "unknown"
            return self._fail(
                interp_id=interp_id,
                error_code="VIDEO_NOT_FOUND",
                error_message=f"Video asset for video_id '{job.video_id}' not found on disk at '{storage_path}'.",
            )

        video_path = Path(video_asset.storage_path)

        # ----------------------------------------------------------------------
        # 4. Extract CV results payload (telemetry, tracks, events)
        # ----------------------------------------------------------------------
        results = (job.config_metadata or {}).get("results")
        if not results:
            results = self._load_results_from_artifacts(job.id)

        if not results:
            return self._fail(
                interp_id=interp_id,
                error_code="CV_RESULTS_MISSING",
                error_message=f"No CV telemetry results available for completed job '{job.id}'.",
            )

        analytics_summary = results.get("analytics_summary") or {}
        detection_summary = results.get("detection_summary") or {}
        tracking_summary = results.get("tracking_summary") or {}
        stopped_events = (
            analytics_summary.get("stopped_vehicle_events", [])
            if isinstance(analytics_summary, dict)
            else []
        )

        self._check_stop_signal(stop_event, interp_id)

        # ----------------------------------------------------------------------
        # 5. Extract keyframes using KeyframeExtractor
        # ----------------------------------------------------------------------
        extraction_config = ExtractionConfigSchema(
            frame_budget=5,
        )

        try:
            keyframes, diagnostics = self.extractor.extract_keyframes(
                video_path=video_path,
                config=extraction_config,
                stopped_events=stopped_events,
                congestion_states=analytics_summary.get("congestion_distribution_per_roi")
                if isinstance(analytics_summary, dict)
                else None,
                peak_occupancy_per_roi=analytics_summary.get("peak_occupancy_per_roi")
                if isinstance(analytics_summary, dict)
                else None,
                source_fps=video_asset.fps,
            )
        except VideoReadError as vre:
            return self._fail(
                interp_id=interp_id,
                error_code="VIDEO_READ_ERROR",
                error_message=f"Failed to decode video frames: {str(vre)}",
            )
        except Exception as ext_exc:
            return self._fail(
                interp_id=interp_id,
                error_code="KEYFRAME_EXTRACTION_FAILED",
                error_message=f"Keyframe extraction failed: {str(ext_exc)}",
            )

        if not keyframes:
            return self._fail(
                interp_id=interp_id,
                error_code="NO_KEYFRAMES_EXTRACTED",
                error_message="Keyframe extraction produced 0 keyframes from source video.",
            )

        self._check_stop_signal(stop_event, interp_id)

        # ----------------------------------------------------------------------
        # 6. Package evidence using EvidencePackager
        # ----------------------------------------------------------------------
        try:
            evidence_package = EvidencePackager.package_evidence(
                video_id=video_asset.id,
                job_id=job.id,
                sequence_name=job.sequence_name,
                extracted_keyframes=keyframes,
                analytics_summary=analytics_summary,
                detection_summary=detection_summary,
                tracking_summary=tracking_summary,
                stopped_events=stopped_events,
                config=extraction_config,
                diagnostics=diagnostics,
            )
        except Exception as pkg_exc:
            return self._fail(
                interp_id=interp_id,
                error_code="EVIDENCE_PACKAGING_FAILED",
                error_message=f"Evidence packaging failed: {str(pkg_exc)}",
            )

        self._check_stop_signal(stop_event, interp_id)

        # ----------------------------------------------------------------------
        # 7. Invoke VLM inference via SceneInterpretationService
        # ----------------------------------------------------------------------
        result = await self.scene_service.interpret_scene(evidence_package)
        if not result.success:
            failure = result.failure
            error_type = failure.error_type if failure else "VLM_ERROR"
            error_msg = failure.message if failure else "VLM inference failed without detailed descriptor."
            return self._fail(
                interp_id=interp_id,
                error_code=self._map_vlm_error_code(error_type),
                error_message=error_msg,
            )

        raw_interpretation = result.interpretation
        if not raw_interpretation:
            return self._fail(
                interp_id=interp_id,
                error_code="EMPTY_VLM_RESPONSE",
                error_message="VLM provider returned a successful result but response payload was empty.",
            )

        self._check_stop_signal(stop_event, interp_id)

        # ----------------------------------------------------------------------
        # 8. Epistemic Grounding Verification
        # ----------------------------------------------------------------------
        try:
            verified_interpretation = verify_scene_interpretation_grounding(
                interpretation=raw_interpretation,
                evidence=evidence_package,
            )
        except Exception as g_exc:
            return self._fail(
                interp_id=interp_id,
                error_code="GROUNDING_VERIFICATION_FAILED",
                error_message=f"Epistemic grounding verification encountered an internal error: {str(g_exc)}",
            )

        self._check_stop_signal(stop_event, interp_id)

        # ----------------------------------------------------------------------
        # 9. Transform and Persist Verified Results in Database
        # ----------------------------------------------------------------------
        payloads = self._format_component_payloads(verified_interpretation)
        now = datetime.now(timezone.utc)
        duration_s = None
        if interp.started_at:
            duration_s = max(0.0, (now - interp.started_at).total_seconds())

        try:
            completed_interp = self.repo.complete(
                interpretation_id=interp_id,
                summary=payloads["summary"],
                environmental_context=payloads["environmental_context"],
                scene_description=payloads["scene_description"],
                event_interpretations=payloads["event_interpretations"],
                epistemic_claims=payloads["epistemic_claims"],
                unverified_aspects=payloads["unverified_aspects"],
                evidence_package_id=evidence_package.evidence_package_id,
                prompt_tokens=payloads["prompt_tokens"],
                completion_tokens=payloads["completion_tokens"],
                total_tokens=payloads["total_tokens"],
                completed_at=now,
                duration_seconds=duration_s,
            )
            self.db.commit()

            if completed_interp is None:
                logger.warning(
                    f"Interpretation '{interp_id}' completed execution, but database record was no longer "
                    "in PROCESSING state (e.g. aborted during shutdown timeout). Preserving terminal state."
                )
                return None

            logger.info(
                f"Scene interpretation '{interp_id}' for job '{job.id}' completed and verified successfully "
                f"in {duration_s:.2f}s."
            )
            return completed_interp

        except InvalidInterpretationStatusTransitionError as ite:
            logger.warning(
                f"Interpretation '{interp_id}' attempted invalid transition during completion: {ite}. "
                "Preserving terminal state."
            )
            self.db.rollback()
            return None
        except Exception as persist_exc:
            logger.error(
                f"Failed to persist COMPLETED status for interpretation '{interp_id}': {persist_exc}",
                exc_info=True,
            )
            self.db.rollback()
            return self._fail(
                interp_id=interp_id,
                error_code="DATABASE_PERSISTENCE_FAILED",
                error_message="Failed to persist completed interpretation results to database.",
            )

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def _fail(
        self,
        interp_id: str,
        error_code: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> Optional[SceneInterpretation]:
        """Safely marks an interpretation task FAILED, strictly preserving existing terminal states."""
        logger.error(f"Failing scene interpretation '{interp_id}' [{error_code}]: {error_message}")
        try:
            failed_interp = self.repo.fail(
                interpretation_id=interp_id,
                error_code=error_code,
                error_message=error_message,
                error_details=error_details,
            )
            self.db.commit()
            return failed_interp
        except InvalidInterpretationStatusTransitionError:
            logger.warning(
                f"Cannot mark interpretation '{interp_id}' as FAILED: already in terminal state. "
                "Preserving terminal state."
            )
            self.db.rollback()
            return self.repo.get_by_id(interp_id)
        except Exception as exc:
            logger.error(f"Failed to record failure state for interpretation '{interp_id}': {exc}", exc_info=True)
            self.db.rollback()
            return None

    def _check_stop_signal(self, stop_event: Optional[threading.Event], interp_id: str) -> None:
        """Raises InterruptedError if a shutdown cancellation signal has been set."""
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError(
                f"Scene interpretation '{interp_id}' received cooperative shutdown stop signal."
            )

    def _load_results_from_artifacts(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fallback helper to load CV results from storage output directory if not in config_metadata."""
        output_dir = self.settings.OUTPUT_DIR / job_id
        if not output_dir.exists():
            return None

        results: Dict[str, Any] = {}
        analytics_file = output_dir / "analytics_summary.json"
        if analytics_file.exists():
            try:
                results["analytics_summary"] = json.loads(analytics_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.debug(f"Could not parse analytics artifact for job {job_id}: {e}")

        det_file = output_dir / "detection_summary.json"
        if det_file.exists():
            try:
                results["detection_summary"] = json.loads(det_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.debug(f"Could not parse detection artifact for job {job_id}: {e}")

        track_file = output_dir / "tracking_summary.json"
        if track_file.exists():
            try:
                results["tracking_summary"] = json.loads(track_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.debug(f"Could not parse tracking artifact for job {job_id}: {e}")

        return results if results else None

    @staticmethod
    def _map_vlm_error_code(error_type: str) -> str:
        """Maps internal VLM exception and failure types to standardized canonical error codes."""
        mapping = {
            "VLMTimeoutError": "VLM_PROVIDER_TIMEOUT",
            "VLMRateLimitError": "VLM_PROVIDER_RATE_LIMIT",
            "VLMAuthenticationError": "VLM_PROVIDER_AUTH_ERROR",
            "VLMConnectionError": "VLM_PROVIDER_CONNECTION_ERROR",
            "VLMOutputParsingError": "VLM_OUTPUT_PARSING_ERROR",
            "VLMGroundingError": "VLM_GROUNDING_ERROR",
            "VLMConfigurationError": "VLM_CONFIG_ERROR",
            "VLMResponseError": "VLM_RESPONSE_ERROR",
            "VLMDisabled": "VLM_DISABLED",
            "ValidationError": "VLM_SCHEMA_VALIDATION_ERROR",
            "UnexpectedVLMError": "VLM_UNEXPECTED_ERROR",
        }
        return mapping.get(error_type, f"VLM_{error_type.upper()}")

    @staticmethod
    def _format_component_payloads(
        verified_interpretation: SceneInterpretationResponseSchema,
    ) -> Dict[str, Any]:
        """Transforms verified VLM schema domain objects into serializable repository dictionaries."""
        # 1. Environmental context
        env = verified_interpretation.environmental_conditions
        env_context = {
            "weather": env.weather.value if hasattr(env.weather, "value") else str(env.weather),
            "lighting": env.illumination.value if hasattr(env.illumination, "value") else str(env.illumination),
            "pavement": env.road_surface.value if hasattr(env.road_surface, "value") else str(env.road_surface),
            "visibility_hazards": list(env.visibility_hazards or []),
            "confidence": env.confidence.value if hasattr(env.confidence, "value") else str(env.confidence),
        }

        # 2. Scene description
        features = [obs.description for obs in (verified_interpretation.visual_observations or [])]
        scene_desc = {
            "roadway_topology": verified_interpretation.roadway_description,
            "features": features,
        }

        # 3. Event interpretations
        events = []
        for evt in (verified_interpretation.event_interpretations or []):
            events.append({
                "event_id": evt.event_id,
                "track_id": evt.track_id,
                "interpretation": evt.contextual_explanation,
                "probable_cause": evt.probable_cause,
                "confidence": evt.confidence.value if hasattr(evt.confidence, "value") else str(evt.confidence),
                "supporting_citations": list(evt.supporting_observations or []),
                "start_timestamp_s": evt.start_timestamp_s,
                "end_timestamp_s": evt.end_timestamp_s,
                "cv_measured_duration_s": evt.cv_measured_duration_s,
            })

        # 4. Epistemic claims
        claims = []
        for claim in (verified_interpretation.claims or []):
            claims.append(claim.model_dump(mode="json") if hasattr(claim, "model_dump") else claim)

        # 5. Token accounting
        meta = verified_interpretation.model_metadata
        prompt_tokens = meta.prompt_tokens if meta else None
        completion_tokens = meta.completion_tokens if meta else None
        total_tokens = (
            (prompt_tokens + completion_tokens)
            if (prompt_tokens is not None and completion_tokens is not None)
            else None
        )

        return {
            "summary": verified_interpretation.scene_summary,
            "environmental_context": env_context,
            "scene_description": scene_desc,
            "event_interpretations": events,
            "epistemic_claims": claims,
            "unverified_aspects": list(verified_interpretation.unverified_aspects or []),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
