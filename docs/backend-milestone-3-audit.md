# Backend Milestone 3: Quality Audit & Real-Video Smoke Test Report

**TrafficLens-AI Backend Quality Audit**  
**Date**: November 22, 2026  
**Status**: **PASSED**  
**Target Milestone**: Backend Milestone 3 — Video Ingestion, Storage & Job Orchestration  

---

## 1. Executive Summary

A comprehensive quality audit and end-to-end real-video smoke test was conducted on the TrafficLens-AI backend service. The audit verified worker queue bounds, concurrency controls, atomic state machine transitions, streaming upload safety, OpenCV stream validation, and crash recovery semantics.

Following the audit, a complete application workflow was executed with **real traffic video** extracted from the UA-DETRAC dataset (`MVI_20011`), driving real YOLO11n detection, ByteTrack tracking, spatiotemporal traffic telemetry calculation, and video annotation through actual HTTP endpoints.

**Final Assessment: PASS (110 / 110 tests passing, real-video smoke test verified end-to-end).**

---

## 2. Worker Queue and Concurrency Audit

### 2.1 Worker Queue Capacity & Overflow Behavior
- **Audit Finding**: In the initial Milestone 3 implementation, `self._queue = queue.Queue()` was instantiated without an explicit `maxsize` parameter (defaulting to unbounded). Under sustained high upload volume, unbounded queuing could exhaust host memory.
- **Remediation**:
  - Introduced configurable `WORKER_MAX_QUEUE_SIZE: int = 100` in [`backend/app/core/config.py`](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/backend/app/core/config.py).
  - Configured `self._queue = queue.Queue(maxsize=self.settings.WORKER_MAX_QUEUE_SIZE)` in [`JobWorker`](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/backend/app/services/worker.py).
  - Defined `QueueFullError(Exception)` raised by `enqueue_job()` upon `queue.Full`.
  - Added HTTP error translation in [`videos.py`](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/backend/app/api/routes/videos.py) and [`jobs.py`](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/backend/app/api/routes/jobs.py) to return **HTTP 503 Service Unavailable** with detail `Worker queue is at maximum capacity (100). Please retry later.`
  - Verified via integration test `test_worker_queue_capacity_limit_and_503`.

### 2.2 Concurrent Duplicate Job Scheduling Prevention
- **Audit Finding**: While the API routes queried for existing `QUEUED` or `PROCESSING` jobs before creating a new job, concurrent requests arriving simultaneously could potentially race through the read check before either committed.
- **Remediation**:
  - Introduced thread synchronization locks (`_video_job_lock` in `videos.py` and `_job_create_lock` in `jobs.py`) ensuring serialized check-and-insert execution per video asset within the process.
  - Integration test `test_concurrent_duplicate_scheduling_prevention` submits 5 simultaneous concurrent job creation requests for the same video across a `ThreadPoolExecutor(max_workers=5)`. Exactly 1 request succeeds with **HTTP 202 Accepted**, and all 4 concurrent attempts are rejected with **HTTP 409 Conflict**.

### 2.3 Atomic `QUEUED -> PROCESSING` State Claim
- **Audit Finding**: In early iterations, `_process_single_job()` read the job model, checked status, and then updated `job.status = JobStatus.PROCESSING`. In multi-worker or re-entrant scenarios, this two-step read-then-write could cause race conditions.
- **Remediation**:
  - Replaced with an atomic SQL update statement:
    ```python
    claim_stmt = (
        update(ProcessingJob)
        .where(ProcessingJob.id == job_id, ProcessingJob.status == JobStatus.QUEUED)
        .values(status=JobStatus.PROCESSING, started_at=now)
    )
    claim_res = db.execute(claim_stmt)
    db.commit()
    if claim_res.rowcount == 0:
        return  # Claim failed: another thread claimed it or state changed
    ```
  - Guarantees database-level atomicity across both SQLite and PostgreSQL dialects.

### 2.4 SQLAlchemy Session Scoping & Thread Safety
- **Audit Finding**: Confirmed that worker routines (`_process_single_job()` and `recover_jobs()`) use dedicated, short-lived database sessions via `with SessionLocal() as db:`.
- **Verification**: Sessions are opened within the worker thread, commit their transactions, and close immediately upon exiting the context block, preventing cross-thread session leakage or connection starvation.

### 2.5 Shutdown and Crash Recovery Semantics
- **Audit Finding**: Verified that application shutdown (`lifespan`) does not leave phantom `PROCESSING` jobs:
  - During graceful termination, `JobWorker.stop(timeout=5.0)` signals `_stop_event` and waits up to 5 seconds for in-flight jobs to terminate. If the thread does not exit within the timeout window, `stop()` immediately updates the active `_running_job_id` to `FAILED` with `error_code="INTERRUPTED_BY_SHUTDOWN"`.
  - On subsequent boot, `JobWorker.recover_jobs()` queries for any remaining jobs with `status == PROCESSING` and transitions them to `FAILED`, while safely re-enqueuing any valid `QUEUED` jobs.

### 2.6 Worker Shutdown Consistency & Terminal State Immutability
- **Audit Finding**: If an active job was marked `FAILED` during a shutdown timeout, the still-running inference thread could finish later and unconditionally execute `job.status = JobStatus.COMPLETED; db.commit()`, overwriting the terminal state. Furthermore, `stop_event` was not threaded into the CV frame loop.
- **Remediation**:
  - Propagated `stop_event=self._stop_event` into `CVPipelineAdapter.process_video_file()`. The frame decode loop checks `if stop_event.is_set(): raise InterruptedError(...)` and releases video handles cleanly.
  - Replaced unconstrained ORM updates with atomic conditional SQL statements:
    ```python
    update(ProcessingJob)
    .where(ProcessingJob.id == job_id, ProcessingJob.status == JobStatus.PROCESSING)
    .values(...)
    ```
  - Enforced that if `rowcount == 0`, the worker recognizes the job has already transitioned to a terminal state (e.g., `FAILED` via shutdown timeout) and strictly discards completion writes.
  - Handled `InterruptedError` explicitly to record `error_code="INTERRUPTED_BY_SHUTDOWN"`.
  - Added regression tests `test_shutdown_during_active_processing_interrupts_cleanly` and `test_terminal_state_cannot_be_overwritten_by_delayed_completion`.

### 2.7 Queue Overflow Rollback & Durable Dynamic Scheduling
- **Audit Finding**: In route handlers, `db.commit()` was executed before `enqueue_job()`. When `QueueFullError` occurred, the job was committed to the DB in `QUEUED` status and the client received HTTP 503. Subsequent client retries were rejected with **HTTP 409 Conflict** because of the stranded active job, blocking processing until an application restart.
- **Remediation**:
  - Reordered route execution in `videos.py` and `jobs.py`: `repo.create_job()` is staged with `db.flush()`. `enqueue_job()` is attempted, and `db.commit()` is executed only on success.
  - On `QueueFullError`, `db.rollback()` is invoked before raising HTTP 503. No stranded `QUEUED` jobs remain in the DB, allowing immediate clean client retries without 409 Conflict.
  - Implemented `JobWorker.redispatch_pending_jobs() -> int` and integrated it into the consumer loop and recovery routines. Any persisted `QUEUED` jobs in the DB are dynamically enqueued whenever queue capacity is available, without requiring an application restart.
  - Added regression tests `test_queue_overflow_rollback_allows_clean_retry` and `test_durable_scheduling_redispatch_pending_jobs`.

---

## 3. Upload Implementation & Defensive Security Audit

### 3.1 Streaming Upload Size Enforcement
- **Inspection**: In [`videos.py`](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/backend/app/api/routes/videos.py), files are read in 1 MB stream chunks (`await file.read(1024 * 1024)`).
- **Early Abort**: Size checks are performed iteratively *as each chunk is read*. If `bytes_written > settings.MAX_UPLOAD_SIZE_BYTES` (500 MB), an **HTTP 413 Content Too Large** exception is raised immediately, terminating the stream without reading or writing remaining payload bytes.
- **Partial File Removal**: The exception handler guarantees that partially written files on disk are unlinked immediately:
  ```python
  except HTTPException:
      if storage_path.exists():
          storage_path.unlink()
      raise
  ```

### 3.2 Real Video Decodability vs MIME/Extension Checks
- **Inspection**: The backend does **not** rely on file extensions or HTTP MIME headers as proof of video validity.
- **OpenCV Validation**: Once written to disk, the file is immediately inspected with `cv2.VideoCapture(str(storage_path))`:
  1. `cap.isOpened()` verifies valid container headers.
  2. `ret, first_frame = cap.read()` decodes the initial frame.
  3. If unreadable, empty, or corrupt, the temporary file is unlinked and rejected with **HTTP 400 Bad Request** (`CORRUPT_OR_EMPTY_VIDEO`).

### 3.3 Safe Metadata Extraction Under Boundary Conditions
- **Zero / Negative FPS**: Handled safely via `if fps <= 0.0 or fps > 240.0: fps = 25.0`.
- **Zero / Negative Frame Count**: Handled via `if frame_count <= 0: frame_count = 1`.
- **Duration Calculation**: Guards against zero division: `duration = frame_count / fps if fps > 0 else 0.0`.
- **Frame Dimensions**: Uses OpenCV properties with fallback to decoded `first_frame.shape[1]` (width) and `first_frame.shape[0]` (height), guaranteed positive.

### 3.4 Filesystem Path Containment
- **Inspection**: Raw uploads are never stored using the client-provided `filename`.
- **Sanitization**: Files are assigned a random UUID identifier: `safe_filename = f"vid_{uuid.uuid4().hex[:12]}{ext}"`.
- **Directory Containment**: Storage paths are resolved and validated with `is_relative_to()`:
  ```python
  storage_path = (settings.UPLOAD_DIR / safe_filename).resolve()
  if not storage_path.is_relative_to(settings.UPLOAD_DIR.resolve()):
      raise HTTPException(status_code=400, detail="Invalid filename: path escapes upload directory.")
  ```

---

## 4. Real-Video End-to-End Smoke Test Evidence

A real-video clip was compiled from the first 75 frames of UA-DETRAC sequence `MVI_20011` (960x540 resolution, 25.0 FPS, 3.0 seconds duration, 2.86 MB).

The complete smoke test script [`experiments/smoke_test_backend_m3.py`](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/experiments/smoke_test_backend_m3.py) was executed with real YOLO11n detection, ByteTrack tracking, spatiotemporal analytics, and video rendering (no mocks or monkeypatches).

### 4.1 Step-by-Step Workflow Audit Trail

| Step | Action | Endpoint / Method | Result | Evidence / Details |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Video Upload** | `POST /api/v1/videos` | **HTTP 201 Created** | Response time: 221.8 ms. Generated `video_id: vid_2c3acd2efa66`. Metadata extracted: 960x540 @ 25.0 FPS, 75 frames, 3.0s. |
| **2** | **DB Asset Verification** | SQLAlchemy `db.get(VideoAsset, ...)` | **VERIFIED** | Record verified in SQLite. File size: 2,994,497 bytes. Path confirmed on disk in `storage/uploads/`. |
| **3** | **Job Dispatch** | `POST /api/v1/videos/{video_id}/jobs` | **HTTP 202 Accepted** | Created job `job_743f34980672` in state `QUEUED`. Virtual line `line_mid` and ROI `roi_main_flow` registered. |
| **4** | **Status Polling** | `GET /api/v1/jobs/{job_id}` | **OBSERVED** | State transitions observed: `QUEUED` $\to$ `PROCESSING` (at +0.05s) $\to$ `COMPLETED` (at +25.00s). |
| **5** | **Results Retrieval** | `GET /api/v1/jobs/{job_id}/results` | **HTTP 200 OK** | Returned detection summary, tracking summary, analytics report, and artifacts paths. |
| **6** | **Artifact Verification** | Local filesystem & OpenCV inspection | **VERIFIED** | Validated `analytics.json`, `telemetry.csv`, and `annotated.mp4` on disk. |

### 4.2 Analytics and Telemetry Outputs

- **Total Detections**: 2,716 vehicle bounding boxes across 75 frames (mean confidence: 0.315).
- **Class Distribution**:
  - `car`: 2,442
  - `truck`: 206
  - `motorcycle`: 51
  - `bus`: 17
- **Unique Track IDs**: 16 unique vehicle trajectories tracked by ByteTrack.
- **Virtual Line Crossings (`line_mid`)**:
  - Downstream: 1 vehicle crossing detected.
  - Upstream: 0 vehicle crossings.
  - Total: 1 crossing.
- **ROI Spatial Occupancy (`roi_main_flow`)**:
  - Peak occupancy ratio: **12.93%**
  - Congestion classification: **100% low congestion** (75 frames).

### 4.3 Output Artifacts Verification

```
storage/outputs/job_743f34980672/
├── MVI_20011_SMOKE_analytics.json   # 345,765 bytes — Valid JSON, 75 frame telemetry records
├── MVI_20011_SMOKE_telemetry.csv    # 3,326 bytes — Valid CSV, 75 data rows, correct headers
└── MVI_20011_SMOKE_annotated.mp4    # 3,672,913 bytes — Playable MP4, 960x540 @ 25.0 FPS, 75 decodable frames
```
- **Video Playability**: Verified via `cv2.VideoCapture`. All 75 annotated frames decodable without dropped frames or codec artifacts. Bounding boxes, track labels, and ROI boundaries rendered correctly.

---

## 5. Performance and Resource Measurements

Measurements recorded during the real-video smoke test execution on the host machine:

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Video Duration** | 3.00 seconds | 75 frames at 25.0 FPS |
| **Frame Dimensions** | 960 $\times$ 540 pixels | Real UA-DETRAC `MVI_20011` sequence |
| **End-to-End Processing Time** | **25.00 seconds** | Wall-clock time from `QUEUED` to `COMPLETED` |
| **CV Pipeline Duration** | **24.77 seconds** | Internal inference, tracking, analytics, and video encoding |
| **Average Processing Throughput** | **3.00 FPS** | Real-time factor: ~8.3x video duration on CPU |
| **Initial Process Memory** | 328.56 MB | Base Python process with PyTorch and Ultralytics loaded |
| **Peak Process Memory** | **504.79 MB** | Measured during YOLO11n + ByteTrack + OpenCV video encoding |
| **Memory Delta** | **+176.23 MB** | Peak memory increase during video execution |
| **Physical Hardware** | **NVIDIA GeForce RTX 2050** | Laptop GPU (Driver Version: 32.0.16.1742) + Intel UHD Graphics |
| **Active Python Environment** | **PyTorch 2.15.0+cpu** | CPU wheel installed (`torch.version.cuda: None`) |
| **Inference Device** | **CPU** | PyTorch CUDA unavailable in current virtual environment; CPU used |
| **Warnings / Errors** | **0** | Clean execution without warnings or unhandled exceptions |

---

## 6. Regression Test Suite

The full repository test suite was executed covering the computer vision pipeline, dataset parsers, models, migrations, schemas, repositories, endpoints, and background worker orchestration:

```bash
$ pytest
============================ 110 passed in 26.38s =============================
```

### Breakdown by Test Suite:
- `tests/test_analytics.py`: 14 passed
- `tests/test_dataset_parser.py`: 26 passed
- `tests/test_detection.py`: 14 passed
- `tests/test_tracking.py`: 11 passed
- `backend/tests/test_api_endpoints.py`: 3 passed
- `backend/tests/test_config.py`: 2 passed
- `backend/tests/test_cv_serialization.py`: 3 passed
- `backend/tests/test_ingestion_and_orchestration.py`: 15 passed
- `backend/tests/test_job_lifecycle.py`: 9 passed
- `backend/tests/test_persistence.py`: 8 passed
- `backend/tests/test_schemas.py`: 5 passed

---

## 7. Known Architectural Limitations

1. **In-Process Single-Threaded Worker**:
   - The current worker processes jobs sequentially from a single internal queue. While ideal for local development and preventing resource thrashing on CPU, multiple concurrent jobs queue up behind one another.
2. **CPU-Bound Inference Throughput in Active Virtual Environment**:
   - The host machine physically contains an NVIDIA GeForce RTX 2050 GPU, but the virtual environment is currently configured with the CPU-only PyTorch build (`torch==2.15.0+cpu`, `torch.version.cuda: None`). Consequently, YOLO11n + ByteTrack + OpenCV video writer runs at ~3.0 FPS for 960x540 resolution on CPU. Installing a CUDA-enabled PyTorch wheel (`torch==2.x+cu12x`) will activate hardware acceleration on the RTX 2050 and provide estimated 15x–30x throughput improvement.
3. **Local Filesystem Storage**:
   - Artifacts and raw videos are stored on local disk under `storage/`. In multi-instance deployments, an S3/GCS object store adapter will be needed.

---

## 8. Final Pass/Fail Assessment

| Audit Criteria | Status | Details |
| :--- | :---: | :--- |
| Worker Queue Boundedness & 503 Overload Rejection | **PASS** | `WORKER_MAX_QUEUE_SIZE=100`, returns 503 on overflow |
| Queue Overflow Rollback & Clean Retries | **PASS** | DB transaction rolled back on full queue; retry succeeds (no 409) |
| Durable Scheduling via Dynamic Redispatch | **PASS** | DB-persisted `QUEUED` jobs dynamically redispatched without restart |
| Concurrent Duplicate Active Job Prevention | **PASS** | Thread-synchronized, returns 409 on race attempts |
| Atomic Status Claim (`QUEUED -> PROCESSING`) | **PASS** | Atomic SQL update statement with rowcount guard |
| Worker Shutdown Consistency & Terminal State Immutability | **PASS** | Stop event checked in frame loop; terminal states protected from overwrite |
| Independent SQLAlchemy Sessions per Thread | **PASS** | Scoped `SessionLocal()` context managers |
| Shutdown & Boot Crash Recovery | **PASS** | Interrupted jobs marked `FAILED`, pending re-enqueued |
| Chunked Streaming Upload & 413 Rejection | **PASS** | 1 MB streaming chunks, deletes partial files on abort |
| OpenCV Video Decodability Verification | **PASS** | Rejects non-video/corrupt payloads with 400 |
| Defensive Metadata Extraction | **PASS** | Fallbacks for 0 FPS, 0 frames, and dimensions |
| Storage Path Containment | **PASS** | Resolved `is_relative_to()` directory containment check |
| Real-Video End-to-End Smoke Test | **PASS** | 75 real frames processed, 2,716 detections, 16 tracks |
| Artifact Playability & Telemetry Integrity | **PASS** | `analytics.json`, `telemetry.csv`, `annotated.mp4` verified |
| Full Repository Regression Suite | **PASS** | All integration and unit test suites passing |

**OVERALL AUDIT VERDICT: PASSED**
