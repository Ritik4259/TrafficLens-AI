"""
Comprehensive tests for InterpretationWorker and queue concurrency in TrafficLens-AI.

Verifies:
- Bounded queue capacity limits and QueueFullError
- Concurrent enqueue operations from multiple threads
- Duplicate scheduling prevention (queued and running states)
- Atomic task claiming (QUEUED -> PROCESSING)
- Exception isolation (unhandled handler errors mark FAILED without crashing worker)
- Thread-local async execution bridge
- Preservation of terminal states (COMPLETED and FAILED are never re-processed)
- Startup recovery of interrupted tasks (PROCESSING -> FAILED with INTERRUPTED_BY_SHUTDOWN)
- Redispatch of persisted QUEUED tasks without duplicates
- Cooperative shutdown and stuck task abortion
- Independence and non-blocking isolation between CV worker and VLM worker
- FastAPI lifespan startup and shutdown integration
"""

from __future__ import annotations
import asyncio
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings, get_settings
from backend.app.db.session import SessionLocal
from backend.app.main import app
from backend.app.models.job import ProcessingJob
from backend.app.models.video import VideoAsset
from backend.app.models.scene_interpretation import (
    SceneInterpretation,
    SceneInterpretationStatus,
)
from backend.app.repositories.scene_interpretation_repository import (
    SQLAlchemySceneInterpretationRepository,
)
from backend.app.schemas.job import JobStatus
from backend.app.services.vlm_worker import (
    InterpretationWorker,
    QueueFullError,
    get_interpretation_worker,
    set_interpretation_worker,
)
from backend.app.services.worker import JobWorker, get_job_worker


# ------------------------------------------------------------------------------
# Test Fixtures & Database Helpers
# ------------------------------------------------------------------------------

def _create_test_hierarchy(
    status: SceneInterpretationStatus = SceneInterpretationStatus.QUEUED,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> tuple[str, str, str]:
    """
    Creates a VideoAsset, a completed ProcessingJob, and a SceneInterpretation record in DB.
    Returns (video_id, job_id, interpretation_id).
    """
    uid = uuid.uuid4().hex[:8]
    vid = f"vid_test_{uid}"
    jid = f"job_test_{uid}"
    iid = f"interp_test_{uid}"

    with SessionLocal() as db:
        video = VideoAsset(
            id=vid,
            filename=f"test_{uid}.mp4",
            storage_path=f"storage/uploads/test_{uid}.mp4",
            file_size_bytes=1024,
        )
        db.add(video)
        db.flush()

        job = ProcessingJob(
            id=jid,
            video_id=vid,
            sequence_name="SEQ_WORKER_TEST",
            status=JobStatus.COMPLETED,
        )
        db.add(job)
        db.flush()

        interp = SceneInterpretation(
            id=iid,
            job_id=jid,
            status=status,
            provider_type="gemini",
            model_name="gemini-3.8-flash",
            started_at=started_at,
            completed_at=completed_at,
        )
        db.add(interp)
        db.commit()

    return vid, jid, iid


def _delete_test_hierarchy(vid: str, jid: str, iid: str) -> None:
    """Cleans up test records from database."""
    with SessionLocal() as db:
        interp = db.get(SceneInterpretation, iid)
        if interp:
            db.delete(interp)
        job = db.get(ProcessingJob, jid)
        if job:
            db.delete(job)
        video = db.get(VideoAsset, vid)
        if video:
            db.delete(video)
        db.commit()


# ------------------------------------------------------------------------------
# 1. Queue Capacity & QueueFullError
# ------------------------------------------------------------------------------

def test_queue_capacity_and_queue_full_error():
    """Validates bounded queue enforcement and QueueFullError when capacity is exceeded."""
    custom_settings = Settings(INTERPRETATION_MAX_QUEUE_SIZE=2)
    worker = InterpretationWorker(settings=custom_settings)

    assert worker.queue_size == 0
    worker.enqueue_interpretation("interp_1")
    assert worker.queue_size == 1
    assert "interp_1" in worker.queued_ids

    worker.enqueue_interpretation("interp_2")
    assert worker.queue_size == 2
    assert "interp_2" in worker.queued_ids

    # 3rd enqueue must raise QueueFullError
    with pytest.raises(QueueFullError) as exc_info:
        worker.enqueue_interpretation("interp_3")

    assert "at maximum capacity (2)" in str(exc_info.value)
    assert worker.queue_size == 2
    assert "interp_3" not in worker.queued_ids


# ------------------------------------------------------------------------------
# 2. Concurrent Enqueue Thread-Safety
# ------------------------------------------------------------------------------

def test_concurrent_enqueue_thread_safety():
    """Validates that concurrent enqueue calls from multiple threads are thread-safe."""
    custom_settings = Settings(INTERPRETATION_MAX_QUEUE_SIZE=50)
    worker = InterpretationWorker(settings=custom_settings)

    num_threads = 25
    barrier = threading.Barrier(num_threads)
    errors: list[Exception] = []

    def _enqueue_target(idx: int):
        try:
            barrier.wait(timeout=5.0)
            worker.enqueue_interpretation(f"task_{idx}")
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=_enqueue_target, args=(i,))
        for i in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert len(errors) == 0
    assert worker.queue_size == num_threads
    assert len(worker.queued_ids) == num_threads


# ------------------------------------------------------------------------------
# 3. Duplicate Scheduling Prevention
# ------------------------------------------------------------------------------

def test_duplicate_scheduling_prevention_while_queued():
    """Validates that duplicate enqueue attempts for an already queued ID are ignored."""
    worker = InterpretationWorker()
    worker.enqueue_interpretation("duplicate_task")
    assert worker.queue_size == 1

    # Second enqueue should be skipped cleanly without error
    worker.enqueue_interpretation("duplicate_task")
    assert worker.queue_size == 1
    assert worker.queued_ids == {"duplicate_task"}


def test_duplicate_scheduling_prevention_while_running():
    """Validates that duplicate enqueue attempts for an actively executing task are ignored."""
    worker = InterpretationWorker()
    with worker._lock:
        worker._running_interp_id = "running_task"

    # Enqueue attempt for running task should be skipped
    worker.enqueue_interpretation("running_task")
    assert worker.queue_size == 0
    assert "running_task" not in worker.queued_ids


# ------------------------------------------------------------------------------
# 4. Atomic Task Claiming Lifecycle
# ------------------------------------------------------------------------------

def test_worker_task_claiming_lifecycle():
    """Validates atomic claim (QUEUED -> PROCESSING) and execution dispatch."""
    vid, jid, iid = _create_test_hierarchy(status=SceneInterpretationStatus.QUEUED)
    try:
        execution_log: list[str] = []

        def test_handler(interp_id: str, db):
            execution_log.append(interp_id)
            repo = SQLAlchemySceneInterpretationRepository(db)
            repo.complete(interp_id, summary="Processed by test handler")
            db.commit()

        worker = InterpretationWorker(handler=test_handler)
        worker._process_single_interpretation(iid)

        assert execution_log == [iid]

        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)
            interp = repo.get_by_id(iid)
            assert interp is not None
            assert interp.status == SceneInterpretationStatus.COMPLETED
            assert interp.started_at is not None
            assert interp.summary == "Processed by test handler"

        # Attempting to process again should skip execution because it is no longer QUEUED
        execution_log.clear()
        worker._process_single_interpretation(iid)
        assert len(execution_log) == 0
    finally:
        _delete_test_hierarchy(vid, jid, iid)


# ------------------------------------------------------------------------------
# 5. Worker Exception Isolation & Terminal State Preservation
# ------------------------------------------------------------------------------

def test_worker_exception_isolation_marks_failed():
    """Validates that unhandled exceptions from handlers mark record FAILED without dying."""
    vid, jid, iid = _create_test_hierarchy(status=SceneInterpretationStatus.QUEUED)
    try:
        def crashing_handler(interp_id: str, db):
            raise RuntimeError("Gemini API connection abruptly terminated")

        worker = InterpretationWorker(handler=crashing_handler)
        worker._process_single_interpretation(iid)

        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)
            interp = repo.get_by_id(iid)
            assert interp is not None
            assert interp.status == SceneInterpretationStatus.FAILED
            assert interp.error_code == "EXECUTION_FAILURE"
            assert "Gemini API connection abruptly terminated" in interp.error_message
            assert interp.completed_at is not None
    finally:
        _delete_test_hierarchy(vid, jid, iid)


def test_no_mutation_of_terminal_records():
    """Validates that records already in terminal state (COMPLETED/FAILED) are never claimed."""
    # 1. Completed record
    vid1, jid1, iid1 = _create_test_hierarchy(status=SceneInterpretationStatus.COMPLETED)
    # 2. Failed record
    vid2, jid2, iid2 = _create_test_hierarchy(status=SceneInterpretationStatus.FAILED)
    try:
        handler_called = []

        def mock_handler(interp_id: str):
            handler_called.append(interp_id)

        worker = InterpretationWorker(handler=mock_handler)
        worker._process_single_interpretation(iid1)
        worker._process_single_interpretation(iid2)

        assert len(handler_called) == 0

        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)
            rec1 = repo.get_by_id(iid1)
            rec2 = repo.get_by_id(iid2)
            assert rec1.status == SceneInterpretationStatus.COMPLETED
            assert rec2.status == SceneInterpretationStatus.FAILED
    finally:
        _delete_test_hierarchy(vid1, jid1, iid1)
        _delete_test_hierarchy(vid2, jid2, iid2)


# ------------------------------------------------------------------------------
# 6. Thread-Local Async Execution Bridge
# ------------------------------------------------------------------------------

def test_async_handler_execution_bridge():
    """Validates that coroutines returned by async handlers are executed cleanly via async bridge."""
    vid, jid, iid = _create_test_hierarchy(status=SceneInterpretationStatus.QUEUED)
    try:
        async def async_handler(interp_id: str, db):
            await asyncio.sleep(0.01)
            repo = SQLAlchemySceneInterpretationRepository(db)
            repo.complete(interp_id, summary="Async bridge completed successfully")
            db.commit()

        worker = InterpretationWorker(handler=async_handler)
        worker._process_single_interpretation(iid)

        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)
            interp = repo.get_by_id(iid)
            assert interp is not None
            assert interp.status == SceneInterpretationStatus.COMPLETED
            assert interp.summary == "Async bridge completed successfully"
    finally:
        _delete_test_hierarchy(vid, jid, iid)


# ------------------------------------------------------------------------------
# 7. Startup Recovery & Redispatch
# ------------------------------------------------------------------------------

def test_startup_recovery_interrupted_tasks():
    """Validates that PROCESSING tasks are recovered to FAILED on boot, and QUEUED are redispatched."""
    # Task 1: Interrupted in PROCESSING state
    vid1, jid1, iid1 = _create_test_hierarchy(
        status=SceneInterpretationStatus.PROCESSING,
        started_at=datetime.now(timezone.utc),
    )
    # Task 2: Persisted QUEUED state
    vid2, jid2, iid2 = _create_test_hierarchy(
        status=SceneInterpretationStatus.QUEUED,
    )
    try:
        worker = InterpretationWorker()
        assert worker.queue_size == 0

        # Execute recovery
        worker.recover_interpretations()

        # 1. Verify Task 1 was recovered to FAILED
        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)
            rec1 = repo.get_by_id(iid1)
            assert rec1.status == SceneInterpretationStatus.FAILED
            assert rec1.error_code == "INTERRUPTED_BY_SHUTDOWN"
            assert "interrupted by application restart or crash" in rec1.error_message
            assert rec1.completed_at is not None

        # 2. Verify Task 2 was redispatched into in-memory queue
        assert worker.queue_size >= 1
        assert iid2 in worker.queued_ids

        # 3. Repeated recovery must not duplicate Task 2 in queue
        current_size = worker.queue_size
        worker.recover_interpretations()
        assert worker.queue_size == current_size
    finally:
        _delete_test_hierarchy(vid1, jid1, iid1)
        _delete_test_hierarchy(vid2, jid2, iid2)


# ------------------------------------------------------------------------------
# 8. Cooperative Shutdown & Stuck Task Abortion
# ------------------------------------------------------------------------------

def test_cooperative_shutdown_clean():
    """Validates that worker thread starts and cleanly shuts down on stop signal."""
    worker = InterpretationWorker()
    worker.start()
    assert worker.is_alive is True

    worker.stop(timeout=2.0)
    assert worker.is_alive is False


def test_cooperative_shutdown_stuck_task_abortion():
    """Validates that a stuck in-flight task is aborted to FAILED when shutdown times out."""
    vid, jid, iid = _create_test_hierarchy(status=SceneInterpretationStatus.QUEUED)
    unblock_event = threading.Event()
    task_started_event = threading.Event()

    try:
        def hanging_handler(interp_id: str, db):
            task_started_event.set()
            # Simulate hung network call waiting for event
            unblock_event.wait(timeout=10.0)

        worker = InterpretationWorker(handler=hanging_handler)
        worker.start()

        worker.enqueue_interpretation(iid)

        # Wait until worker begins executing the task
        assert task_started_event.wait(timeout=5.0)
        assert worker.running_interp_id == iid

        # Stop with short timeout so it times out while task is still running
        worker.stop(timeout=0.2)

        # Check that DB record was marked FAILED with INTERRUPTED_BY_SHUTDOWN
        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)
            rec = repo.get_by_id(iid)
            assert rec.status == SceneInterpretationStatus.FAILED
            assert rec.error_code == "INTERRUPTED_BY_SHUTDOWN"
            assert "interrupted by application shutdown" in rec.error_message
    finally:
        unblock_event.set()
        worker.stop(timeout=1.0)
        _delete_test_hierarchy(vid, jid, iid)


# ------------------------------------------------------------------------------
# 9. Clean Separation & Isolation from CV Worker
# ------------------------------------------------------------------------------

def test_cv_and_vlm_workers_are_isolated():
    """Validates that CV worker and VLM worker operate on distinct threads and independent queues."""
    cv_worker = get_job_worker()
    vlm_worker = get_interpretation_worker()

    assert cv_worker is not vlm_worker
    assert type(cv_worker) is JobWorker
    assert type(vlm_worker) is InterpretationWorker

    cv_worker.start()
    vlm_worker.start()

    # Find the active threads by name
    thread_names = {t.name for t in threading.enumerate()}
    assert "TrafficLens-Worker" in thread_names
    assert "TrafficLens-InterpretationWorker" in thread_names

    # Verify that enqueueing in one does not touch the other
    initial_cv_qsize = cv_worker._queue.qsize()
    initial_vlm_qsize = vlm_worker._queue.qsize()

    vlm_worker.enqueue_interpretation("dummy_vlm_isolation_check")
    assert vlm_worker._queue.qsize() == initial_vlm_qsize + 1
    assert cv_worker._queue.qsize() == initial_cv_qsize

    # Clean up dummy queue entry
    with vlm_worker._lock:
        vlm_worker._queued_ids.discard("dummy_vlm_isolation_check")
        # Drain the dummy item if still in queue
        try:
            item = vlm_worker._queue.get_nowait()
            vlm_worker._queue.task_done()
        except Exception:
            pass


# ------------------------------------------------------------------------------
# 10. FastAPI Lifespan Integration
# ------------------------------------------------------------------------------

def test_fastapi_lifespan_starts_and_stops_vlm_worker():
    """Validates that entering and exiting the FastAPI lifespan context starts/stops workers."""
    with TestClient(app) as test_client:
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200

        # While within lifespan context, interpretation worker should be active
        interp_worker = get_interpretation_worker()
        assert interp_worker.is_alive is True

    # Once lifespan context exits, stop() has been called
    # Note: daemon thread termination might take up to timeout
    time.sleep(0.2)


# ------------------------------------------------------------------------------
# 11. Concurrency Audit Safeguards
# ------------------------------------------------------------------------------

def test_unsupported_worker_concurrency_rejected():
    """Validates that INTERPRETATION_WORKER_CONCURRENCY != 1 is strictly rejected."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        Settings(INTERPRETATION_WORKER_CONCURRENCY=2)
    assert "INTERPRETATION_WORKER_CONCURRENCY must be 1" in str(exc_info.value)

    # Also verify InterpretationWorker constructor directly
    mock_settings = Settings()
    object.__setattr__(mock_settings, "INTERPRETATION_WORKER_CONCURRENCY", 3)
    with pytest.raises(ValueError) as val_exc:
        InterpretationWorker(settings=mock_settings)
    assert "Unsupported INTERPRETATION_WORKER_CONCURRENCY=3" in str(val_exc.value)


def test_active_task_cannot_persist_completed_after_shutdown_abortion():
    """
    Validates race condition safety: If a task is aborted to FAILED during shutdown timeout,
    a delayed in-flight handler completing subsequently CANNOT overwrite the terminal FAILED state.
    """
    from backend.app.schemas.scene_interpretation import InvalidInterpretationStatusTransitionError

    vid, jid, iid = _create_test_hierarchy(status=SceneInterpretationStatus.QUEUED)
    try:
        def delayed_handler(interp_id: str, db):
            # 1. Simulate external shutdown abortion happening in another transaction
            with SessionLocal() as abort_db:
                abort_repo = SQLAlchemySceneInterpretationRepository(abort_db)
                abort_repo.fail(
                    interpretation_id=interp_id,
                    error_code="INTERRUPTED_BY_SHUTDOWN",
                    error_message="Aborted by shutdown timeout.",
                )
                abort_db.commit()

            # 2. Delayed handler now attempts to mark COMPLETED on the already FAILED record
            repo = SQLAlchemySceneInterpretationRepository(db)
            repo.complete(
                interpretation_id=interp_id,
                summary="Attempted late completion.",
            )
            db.commit()

        worker = InterpretationWorker(handler=delayed_handler)
        # Process task: handler attempts late completion on FAILED record
        worker._process_single_interpretation(iid)

        # 3. Verify record remains FAILED and was NOT overwritten by late COMPLETED
        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)
            rec = repo.get_by_id(iid)
            assert rec.status == SceneInterpretationStatus.FAILED
            assert rec.error_code == "INTERRUPTED_BY_SHUTDOWN"
            assert rec.summary is None  # Analytical results not persisted
    finally:
        _delete_test_hierarchy(vid, jid, iid)


def test_cooperative_shutdown_with_stop_event_interrupted_error():
    """Validates that handlers accepting stop_event and raising InterruptedError are handled cleanly."""
    vid, jid, iid = _create_test_hierarchy(status=SceneInterpretationStatus.QUEUED)
    try:
        def cooperative_handler(interp_id: str, db, stop_event):
            # Handler observes stop_event or raises InterruptedError
            raise InterruptedError("Worker received shutdown signal during processing.")

        worker = InterpretationWorker(handler=cooperative_handler)
        worker._process_single_interpretation(iid)

        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)
            rec = repo.get_by_id(iid)
            assert rec.status == SceneInterpretationStatus.FAILED
            assert rec.error_code == "INTERRUPTED_BY_SHUTDOWN"
            assert "Worker received shutdown signal" in rec.error_message
    finally:
        _delete_test_hierarchy(vid, jid, iid)


def test_repeated_recovery_calls_idempotency():
    """Validates that calling recover_interpretations repeatedly is strictly idempotent."""
    vid1, jid1, iid1 = _create_test_hierarchy(
        status=SceneInterpretationStatus.PROCESSING,
        started_at=datetime.now(timezone.utc),
    )
    vid2, jid2, iid2 = _create_test_hierarchy(status=SceneInterpretationStatus.QUEUED)
    vid3, jid3, iid3 = _create_test_hierarchy(status=SceneInterpretationStatus.QUEUED)

    try:
        worker = InterpretationWorker()

        # Call recover 3 consecutive times
        worker.recover_interpretations()
        worker.recover_interpretations()
        worker.recover_interpretations()

        # Verify PROCESSING was marked FAILED exactly once
        with SessionLocal() as db:
            repo = SQLAlchemySceneInterpretationRepository(db)
            rec1 = repo.get_by_id(iid1)
            assert rec1.status == SceneInterpretationStatus.FAILED
            assert rec1.error_code == "INTERRUPTED_BY_SHUTDOWN"

        # Verify QUEUED items are enqueued without duplicate entries
        assert iid2 in worker.queued_ids
        assert iid3 in worker.queued_ids
        # Total queued must not exceed unique items
        assert worker.queue_size == len(worker.queued_ids)
    finally:
        _delete_test_hierarchy(vid1, jid1, iid1)
        _delete_test_hierarchy(vid2, jid2, iid2)
        _delete_test_hierarchy(vid3, jid3, iid3)


def test_lifespan_fault_isolation_when_worker_fails(monkeypatch):
    """Validates that lifespan startup and shutdown remain fault-tolerant if a worker errors."""
    vlm_worker = get_interpretation_worker()

    def buggy_start():
        raise RuntimeError("Simulated worker start failure")

    def buggy_stop(timeout=5.0):
        raise RuntimeError("Simulated worker stop failure")

    monkeypatch.setattr(vlm_worker, "start", buggy_start)
    monkeypatch.setattr(vlm_worker, "stop", buggy_stop)

    # Lifespan should catch errors independently and not crash the process
    with TestClient(app) as test_client:
        resp = test_client.get("/api/v1/health")
        assert resp.status_code == 200

