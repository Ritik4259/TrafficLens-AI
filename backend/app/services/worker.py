"""
Single-process asynchronous video processing worker and recovery manager.

Coordinates background execution of CV inference and spatiotemporal analytics,
guaranteeing that the database is the source of truth, recovering interrupted jobs
on startup, and ensuring the FastAPI event loop is never blocked.
"""

from __future__ import annotations
import queue
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.db.session import SessionLocal
from backend.app.models.job import ProcessingJob
from backend.app.repositories.job_repository import SQLAlchemyJobRepository
from backend.app.schemas.job import JobStatus, ProcessingErrorSchema
from backend.app.schemas.common import CountingLineConfigSchema, ROIConfigSchema
from backend.app.services.cv_pipeline_adapter import CVPipelineAdapter
from sqlalchemy import update

logger = get_logger("trafficlens.worker")


class QueueFullError(Exception):
    """Raised when the asynchronous worker queue has reached its maximum capacity."""
    pass


class JobWorker:
    """
    Bounded, thread-based background worker processing video jobs asynchronously.
    """

    def __init__(self):
        self.settings = get_settings()
        self._queue: queue.Queue[str] = queue.Queue(maxsize=self.settings.WORKER_MAX_QUEUE_SIZE)
        self._queued_ids: set[str] = set()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running_job_id: Optional[str] = None
        self._lock = threading.Lock()
        self._last_redispatch_check: float = 0.0

    def start(self) -> None:
        """Starts the background worker thread and executes startup recovery."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="TrafficLens-Worker")
        self._thread.start()
        logger.info(
            f"Background job worker thread started (max_queue_size={self.settings.WORKER_MAX_QUEUE_SIZE})."
        )

        # Execute recovery of interrupted and pending jobs
        self.recover_jobs()

    def stop(self, timeout: float = 5.0) -> None:
        """Signals worker to terminate and waits up to timeout seconds."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    f"Background worker thread did not terminate within {timeout}s shutdown timeout."
                )
                with self._lock:
                    stuck_job_id = self._running_job_id
                if stuck_job_id:
                    try:
                        with SessionLocal() as db:
                            now = datetime.now(timezone.utc)
                            abort_stmt = (
                                update(ProcessingJob)
                                .where(ProcessingJob.id == stuck_job_id, ProcessingJob.status == JobStatus.PROCESSING)
                                .values(
                                    status=JobStatus.FAILED,
                                    completed_at=now,
                                    error_code="INTERRUPTED_BY_SHUTDOWN",
                                    error_message="Job processing was interrupted by application shutdown.",
                                )
                            )
                            db.execute(abort_stmt)
                            db.commit()
                            logger.warning(f"Marked active job {stuck_job_id} as FAILED during shutdown.")
                    except Exception as e:
                        logger.error(f"Failed to update job status during shutdown: {e}")
        logger.info("Background job worker stopped.")

    def enqueue_job(self, job_id: str) -> None:
        """Enqueues a job identifier for asynchronous processing."""
        with self._lock:
            try:
                self._queue.put_nowait(job_id)
                self._queued_ids.add(job_id)
                logger.info(f"Enqueued job {job_id} for background processing (queue size: {self._queue.qsize()}).")
            except queue.Full:
                logger.error(
                    f"Cannot enqueue job {job_id}: worker queue is full (maxsize={self.settings.WORKER_MAX_QUEUE_SIZE})."
                )
                raise QueueFullError(
                    f"Worker queue is at maximum capacity ({self.settings.WORKER_MAX_QUEUE_SIZE}). Please retry later."
                )

    def redispatch_pending_jobs(self) -> int:
        """
        Scans database for pending QUEUED jobs that are not currently in the memory queue
        and enqueues them up to available capacity, without requiring an application restart.
        """
        with self._lock:
            available_capacity = self.settings.WORKER_MAX_QUEUE_SIZE - self._queue.qsize()
            if available_capacity <= 0:
                return 0

        with SessionLocal() as db:
            from sqlalchemy import select
            stmt = (
                select(ProcessingJob.id)
                .where(ProcessingJob.status == JobStatus.QUEUED)
                .order_by(ProcessingJob.created_at.asc())
            )
            queued_ids = db.scalars(stmt).all()

        redispatched = 0
        for jid in queued_ids:
            with self._lock:
                if self._queue.qsize() >= self.settings.WORKER_MAX_QUEUE_SIZE:
                    break
                if jid in self._queued_ids or jid == self._running_job_id:
                    continue
                try:
                    self._queue.put_nowait(jid)
                    self._queued_ids.add(jid)
                    redispatched += 1
                    logger.info(f"Redispatched pending job {jid} from database to active queue.")
                except queue.Full:
                    break

        return redispatched

    def recover_jobs(self) -> None:
        """
        Executes startup recovery:
        1. Identifies jobs interrupted in PROCESSING state and marks them as FAILED.
        2. Dynamically redispatches pending QUEUED jobs up to queue capacity.
        """
        with SessionLocal() as db:
            from sqlalchemy import select
            # 1. Recover interrupted jobs
            interrupted_stmt = select(ProcessingJob).where(ProcessingJob.status == JobStatus.PROCESSING)
            interrupted_jobs = db.scalars(interrupted_stmt).all()

            for job in interrupted_jobs:
                logger.warning(
                    f"Recovering interrupted job {job.id} (found in PROCESSING state on boot). "
                    "Marking as FAILED."
                )
                now = datetime.now(timezone.utc)
                job.status = JobStatus.FAILED
                job.completed_at = now
                job.error_code = "INTERRUPTED_BY_SHUTDOWN"
                job.error_message = "Job processing was interrupted by application restart or crash."
                if job.started_at:
                    job.duration_seconds = max(0.0, (now - job.started_at).total_seconds())

            db.commit()

        # 2. Redispatch any pending QUEUED jobs up to queue capacity
        self.redispatch_pending_jobs()

    def _worker_loop(self) -> None:
        """Main consumer loop running in background thread."""
        while not self._stop_event.is_set():
            # Dynamically redispatch any persisted QUEUED jobs if queue has capacity
            now_ts = time.time()
            if now_ts - self._last_redispatch_check > 1.0:
                self._last_redispatch_check = now_ts
                try:
                    self.redispatch_pending_jobs()
                except Exception as e:
                    logger.debug(f"Redispatch scan encountered: {e}")

            try:
                job_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                with self._lock:
                    self._running_job_id = job_id
                    self._queued_ids.discard(job_id)
                self._process_single_job(job_id)
            except Exception as e:
                logger.error(f"Unexpected exception processing job {job_id}: {e}", exc_info=True)
            finally:
                with self._lock:
                    self._running_job_id = None
                self._queue.task_done()

    def _process_single_job(self, job_id: str) -> None:
        """Executes the CV pipeline for a claimed job within an isolated DB session."""
        with SessionLocal() as db:
            repo = SQLAlchemyJobRepository(db)
            now = datetime.now(timezone.utc)

            # Atomic claim: update status to PROCESSING only if currently in QUEUED state
            claim_stmt = (
                update(ProcessingJob)
                .where(ProcessingJob.id == job_id, ProcessingJob.status == JobStatus.QUEUED)
                .values(status=JobStatus.PROCESSING, started_at=now)
            )
            claim_res = db.execute(claim_stmt)
            db.commit()

            if claim_res.rowcount == 0:
                logger.warning(f"Job {job_id} could not be claimed (not in QUEUED state). Skipping execution.")
                return

            job = repo.get_job_model(job_id)
            if not job:
                logger.error(f"Cannot process job {job_id}: record not found in database.")
                return

            logger.info(f"Atomically claimed job {job_id} -> PROCESSING.")

            video_asset = repo.get_video_asset(job.video_id)
            if not video_asset or not Path(video_asset.storage_path).exists():
                err_msg = f"Video asset for video_id {job.video_id} not found on disk."
                logger.error(f"Job {job_id} failed: {err_msg}")
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now(timezone.utc)
                job.error_code = "VIDEO_NOT_FOUND"
                job.error_message = err_msg
                db.commit()
                return

            video_path = Path(video_asset.storage_path)
            output_dir = self.settings.OUTPUT_DIR / job_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Parse lines and ROIs from config_metadata if present
            lines = None
            rois = None
            if job.config_metadata:
                if "lines" in job.config_metadata:
                    lines = [CountingLineConfigSchema(**l) for l in job.config_metadata["lines"]]
                if "rois" in job.config_metadata:
                    rois = [ROIConfigSchema(**r) for r in job.config_metadata["rois"]]

            # Execute CV pipeline
            t0 = time.perf_counter()
            try:
                cv_res = CVPipelineAdapter.process_video_file(
                    video_path=video_path,
                    output_dir=output_dir,
                    sequence_name=job.sequence_name,
                    lines=lines,
                    rois=rois,
                    stop_event=self._stop_event,
                )
                if len(cv_res) == 5:
                    report, det_summary, track_summary, artifacts, annotation_diagnostics = cv_res
                else:
                    report, det_summary, track_summary, artifacts = cv_res
                    annotation_diagnostics = None

                duration = time.perf_counter() - t0

                # Mark COMPLETED conditionally on status == PROCESSING to prevent overwriting terminal states
                completed_time = datetime.now(timezone.utc)
                results_payload = {
                    "detection_summary": det_summary.model_dump(),
                    "tracking_summary": track_summary.model_dump(),
                    "analytics_summary": CVPipelineAdapter.convert_analytics_report_to_schema(report).model_dump(),
                    "artifacts": artifacts,
                    "annotation_diagnostics": annotation_diagnostics,
                }
                meta = dict(job.config_metadata or {})
                meta["results"] = results_payload

                complete_stmt = (
                    update(ProcessingJob)
                    .where(ProcessingJob.id == job_id, ProcessingJob.status == JobStatus.PROCESSING)
                    .values(
                        status=JobStatus.COMPLETED,
                        completed_at=completed_time,
                        duration_seconds=round(duration, 2),
                        config_metadata=meta,
                    )
                )
                res = db.execute(complete_stmt)
                db.commit()
                if res.rowcount == 0:
                    logger.warning(
                        f"Job {job_id} finished execution, but was no longer in PROCESSING state "
                        "(terminal state was already set, e.g. FAILED during shutdown timeout). Preserving terminal state."
                    )
                else:
                    logger.info(f"Job {job_id} completed successfully in {duration:.2f}s.")

            except InterruptedError as e:
                err_time = datetime.now(timezone.utc)
                logger.warning(f"Job {job_id} processing interrupted by shutdown: {e}")
                abort_stmt = (
                    update(ProcessingJob)
                    .where(ProcessingJob.id == job_id, ProcessingJob.status == JobStatus.PROCESSING)
                    .values(
                        status=JobStatus.FAILED,
                        completed_at=err_time,
                        duration_seconds=round((err_time - now).total_seconds(), 2),
                        error_code="INTERRUPTED_BY_SHUTDOWN",
                        error_message=str(e),
                    )
                )
                res = db.execute(abort_stmt)
                db.commit()
                if res.rowcount == 0:
                    logger.info(f"Job {job_id} terminal state already recorded during shutdown.")

            except Exception as e:
                err_time = datetime.now(timezone.utc)
                logger.error(f"CV pipeline execution failed for job {job_id}: {e}", exc_info=True)
                fail_stmt = (
                    update(ProcessingJob)
                    .where(ProcessingJob.id == job_id, ProcessingJob.status == JobStatus.PROCESSING)
                    .values(
                        status=JobStatus.FAILED,
                        completed_at=err_time,
                        duration_seconds=round((err_time - now).total_seconds(), 2),
                        error_code="PIPELINE_EXECUTION_FAILURE",
                        error_message=str(e),
                        error_details={"traceback": traceback.format_exc()},
                    )
                )
                res = db.execute(fail_stmt)
                db.commit()
                if res.rowcount == 0:
                    logger.warning(
                        f"Job {job_id} terminal state was already updated prior to exception handler. Preserving terminal state."
                    )


# Worker singleton
_worker_instance: Optional[JobWorker] = None


def get_job_worker() -> JobWorker:
    """Returns the singleton background JobWorker instance."""
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = JobWorker()
    return _worker_instance
