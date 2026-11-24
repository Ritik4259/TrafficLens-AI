"""
Single-process asynchronous scene interpretation worker and recovery manager.

Coordinates background execution of multimodal VLM scene understanding pipelines,
guaranteeing that the database is the source of truth, recovering interrupted tasks
on startup, enforcing queue capacity limits, and ensuring the CV worker and FastAPI
event loop are never blocked.
"""

from __future__ import annotations
import asyncio
import inspect
import queue
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.db.session import SessionLocal
from backend.app.models.scene_interpretation import SceneInterpretation
from backend.app.repositories.scene_interpretation_repository import (
    SQLAlchemySceneInterpretationRepository,
)
from backend.app.schemas.scene_interpretation import (
    SceneInterpretationStatus,
    InvalidInterpretationStatusTransitionError,
)

logger = get_logger("trafficlens.vlm_worker")


class QueueFullError(Exception):
    """Raised when the asynchronous scene interpretation queue has reached its maximum capacity."""
    pass


class InterpretationWorker:
    """
    Bounded, thread-based background worker processing multimodal interpretations asynchronously.
    Supports a single application process only (uvicorn --workers 1).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        handler: Optional[Callable[..., Any]] = None,
    ):
        self.settings = settings or get_settings()
        if self.settings.INTERPRETATION_WORKER_CONCURRENCY != 1:
            raise ValueError(
                f"Unsupported INTERPRETATION_WORKER_CONCURRENCY={self.settings.INTERPRETATION_WORKER_CONCURRENCY}. "
                "Only concurrency=1 is supported in the single-process in-memory queue architecture."
            )
        self._handler = handler
        self._queue: queue.Queue[str] = queue.Queue(maxsize=self.settings.INTERPRETATION_MAX_QUEUE_SIZE)
        self._queued_ids: Set[str] = set()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running_interp_id: Optional[str] = None
        self._lock = threading.Lock()
        self._last_redispatch_check: float = 0.0

    @property
    def running_interp_id(self) -> Optional[str]:
        """Returns the ID of the currently processing interpretation, if any."""
        with self._lock:
            return self._running_interp_id

    @property
    def queued_ids(self) -> Set[str]:
        """Returns a snapshot of the interpretation IDs currently queued in memory."""
        with self._lock:
            return set(self._queued_ids)

    @property
    def queue_size(self) -> int:
        """Returns current queue depth."""
        return self._queue.qsize()

    @property
    def is_alive(self) -> bool:
        """Returns True if the background worker thread is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Starts the background worker thread and executes startup recovery."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="TrafficLens-InterpretationWorker",
        )
        self._thread.start()
        logger.info(
            f"Background scene interpretation worker thread started "
            f"(max_queue_size={self.settings.INTERPRETATION_MAX_QUEUE_SIZE})."
        )

        # Execute recovery of interrupted and pending interpretations
        self.recover_interpretations()

    def stop(self, timeout: float = 5.0) -> None:
        """Signals worker to terminate and waits up to timeout seconds."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    f"Background interpretation worker thread did not terminate within {timeout}s shutdown timeout. "
                    "Thread will continue as a daemon until process exit (Python threads cannot be forcibly killed). "
                    "Active database task marked FAILED."
                )
                with self._lock:
                    stuck_id = self._running_interp_id
                if stuck_id:
                    try:
                        with SessionLocal() as db:
                            repo = SQLAlchemySceneInterpretationRepository(db)
                            rec = repo.get_by_id(stuck_id)
                            if rec and rec.status == SceneInterpretationStatus.PROCESSING:
                                repo.fail(
                                    interpretation_id=stuck_id,
                                    error_code="INTERRUPTED_BY_SHUTDOWN",
                                    error_message="Scene interpretation processing was interrupted by application shutdown.",
                                )
                                db.commit()
                                logger.warning(
                                    f"Marked active interpretation {stuck_id} as FAILED during shutdown."
                                )
                    except Exception as e:
                        logger.error(f"Failed to update interpretation status during shutdown: {e}")
        logger.info("Background scene interpretation worker stopped.")

    def enqueue_interpretation(self, interp_id: str) -> None:
        """
        Enqueues an interpretation identifier for asynchronous processing.
        Prevents duplicate scheduling if the ID is already queued or actively running.
        Raises QueueFullError if queue is at maximum capacity.
        """
        with self._lock:
            if interp_id in self._queued_ids or interp_id == self._running_interp_id:
                logger.warning(
                    f"Interpretation {interp_id} is already queued or currently running; skipping duplicate enqueue."
                )
                return

            try:
                self._queue.put_nowait(interp_id)
                self._queued_ids.add(interp_id)
                logger.info(
                    f"Enqueued scene interpretation {interp_id} for background processing "
                    f"(queue size: {self._queue.qsize()})."
                )
            except queue.Full:
                logger.error(
                    f"Cannot enqueue interpretation {interp_id}: worker queue is full "
                    f"(maxsize={self.settings.INTERPRETATION_MAX_QUEUE_SIZE})."
                )
                raise QueueFullError(
                    f"Interpretation worker queue is at maximum capacity "
                    f"({self.settings.INTERPRETATION_MAX_QUEUE_SIZE}). Please retry later."
                )

    def redispatch_pending_interpretations(self) -> int:
        """
        Scans database for pending QUEUED interpretations that are not currently in the memory queue
        and enqueues them up to available capacity, without requiring an application restart.
        """
        with self._lock:
            available_capacity = self.settings.INTERPRETATION_MAX_QUEUE_SIZE - self._queue.qsize()
            if available_capacity <= 0:
                return 0

        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)
            queued_records = repo.list_queued(limit=available_capacity)
            queued_ids = [r.id for r in queued_records]

        redispatched = 0
        for iid in queued_ids:
            with self._lock:
                if self._queue.qsize() >= self.settings.INTERPRETATION_MAX_QUEUE_SIZE:
                    break
                if iid in self._queued_ids or iid == self._running_interp_id:
                    continue
                try:
                    self._queue.put_nowait(iid)
                    self._queued_ids.add(iid)
                    redispatched += 1
                    logger.info(f"Redispatched pending scene interpretation {iid} from database to active queue.")
                except queue.Full:
                    break

        return redispatched

    def recover_interpretations(self) -> None:
        """
        Executes startup recovery:
        1. Identifies interpretations interrupted in PROCESSING state and marks them as FAILED.
        2. Dynamically redispatches pending QUEUED interpretations up to queue capacity.
        Does not automatically re-execute external API calls.
        """
        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)
            interrupted = repo.list_processing()

            for item in interrupted:
                logger.warning(
                    f"Recovering interrupted scene interpretation {item.id} "
                    "(found in PROCESSING state on boot). Marking as FAILED."
                )
                now = datetime.now(timezone.utc)
                duration = None
                if item.started_at:
                    duration = max(0.0, (now - item.started_at).total_seconds())
                repo.fail(
                    interpretation_id=item.id,
                    error_code="INTERRUPTED_BY_SHUTDOWN",
                    error_message="Scene interpretation was interrupted by application restart or crash.",
                    completed_at=now,
                    duration_seconds=duration,
                )

            db.commit()

        # Redispatch any pending QUEUED interpretations up to queue capacity
        self.redispatch_pending_interpretations()

    def _worker_loop(self) -> None:
        """Main consumer loop running in background daemon thread."""
        while not self._stop_event.is_set():
            # Periodically redispatch persisted QUEUED tasks if queue has capacity
            now_ts = time.time()
            if now_ts - self._last_redispatch_check > 1.0:
                self._last_redispatch_check = now_ts
                try:
                    self.redispatch_pending_interpretations()
                except Exception as e:
                    logger.debug(f"Interpretation redispatch scan encountered: {e}")

            try:
                interp_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                with self._lock:
                    self._running_interp_id = interp_id
                    self._queued_ids.discard(interp_id)
                self._process_single_interpretation(interp_id)
            except Exception as e:
                logger.error(
                    f"Unexpected exception processing scene interpretation {interp_id}: {e}",
                    exc_info=True,
                )
            finally:
                with self._lock:
                    self._running_interp_id = None
                self._queue.task_done()

    def _process_single_interpretation(self, interp_id: str) -> None:
        """
        Executes scene interpretation within an isolated DB session.
        Atomically claims task (QUEUED -> PROCESSING).
        Executes orchestrator pipeline or custom handler bridge.
        Catches unhandled exceptions and marks record as FAILED.
        """
        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)

            # Atomic claim: update status to PROCESSING only if currently in QUEUED state
            claimed = repo.claim_for_processing(interp_id)
            if not claimed:
                logger.warning(
                    f"Scene interpretation {interp_id} could not be claimed "
                    "(not in QUEUED state). Skipping execution."
                )
                return

            db.commit()
            logger.info(f"Atomically claimed scene interpretation {interp_id} -> PROCESSING.")

            try:
                if self._handler is not None:
                    try:
                        res = self._handler(interp_id, db=db, stop_event=self._stop_event)
                    except TypeError:
                        try:
                            res = self._handler(interp_id, db)
                        except TypeError:
                            res = self._handler(interp_id)

                    if inspect.iscoroutine(res):
                        self._run_async(res)
                else:
                    # Issue 04 will provide InterpretationOrchestratorService
                    try:
                        from backend.app.services.vlm.orchestrator import InterpretationOrchestratorService

                        orchestrator = InterpretationOrchestratorService(db=db, settings=self.settings)
                        res = orchestrator.execute_interpretation(interp_id, stop_event=self._stop_event)
                        if inspect.iscoroutine(res):
                            self._run_async(res)
                    except ImportError:
                        logger.info(
                            f"InterpretationOrchestratorService not yet available; claimed {interp_id}."
                        )
            except InterruptedError as ie:
                logger.warning(f"Scene interpretation {interp_id} interrupted by shutdown signal: {ie}")
                try:
                    with SessionLocal() as abort_db:
                        abort_repo = SQLAlchemySceneInterpretationRepository(abort_db)
                        rec = abort_repo.get_by_id(interp_id)
                        if rec and rec.status == SceneInterpretationStatus.PROCESSING:
                            abort_repo.fail(
                                interpretation_id=interp_id,
                                error_code="INTERRUPTED_BY_SHUTDOWN",
                                error_message=str(ie),
                            )
                            abort_db.commit()
                except Exception as fe:
                    logger.error(f"Failed to record shutdown abort for interpretation {interp_id}: {fe}")
            except InvalidInterpretationStatusTransitionError as ite:
                logger.warning(
                    f"Scene interpretation {interp_id} attempted invalid transition ({ite}). "
                    "Terminal state was already recorded (e.g. FAILED during shutdown timeout). Preserving terminal state."
                )
            except Exception as e:
                logger.error(
                    f"Scene interpretation execution failed for {interp_id}: {e}",
                    exc_info=True,
                )
                try:
                    with SessionLocal() as err_db:
                        err_repo = SQLAlchemySceneInterpretationRepository(err_db)
                        rec = err_repo.get_by_id(interp_id)
                        if rec and rec.status == SceneInterpretationStatus.PROCESSING:
                            err_repo.fail(
                                interpretation_id=interp_id,
                                error_code="EXECUTION_FAILURE",
                                error_message=str(e),
                                error_details={"traceback": traceback.format_exc()},
                            )
                            err_db.commit()
                except Exception as fe:
                    logger.error(f"Failed to record failure for interpretation {interp_id}: {fe}")

    def _run_async(self, coro: Any) -> Any:
        """Thread-local async execution bridge to run async orchestrator coroutines."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)


# Worker singleton
_interpretation_worker_instance: Optional[InterpretationWorker] = None
_instance_lock = threading.Lock()


def get_interpretation_worker() -> InterpretationWorker:
    """Returns the singleton background InterpretationWorker instance."""
    global _interpretation_worker_instance
    if _interpretation_worker_instance is None:
        with _instance_lock:
            if _interpretation_worker_instance is None:
                _interpretation_worker_instance = InterpretationWorker()
    return _interpretation_worker_instance


def set_interpretation_worker(worker: Optional[InterpretationWorker]) -> None:
    """Sets or resets the singleton InterpretationWorker instance (useful for testing)."""
    global _interpretation_worker_instance
    with _instance_lock:
        _interpretation_worker_instance = worker
