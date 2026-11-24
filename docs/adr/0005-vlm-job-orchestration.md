# ADR 0005: Asynchronous Scene Understanding Orchestration and Decoupled Lifecycle Management

## Status

Accepted

## Context

Phase 1 through Phase 4 established a robust, deterministic Computer Vision (CV) pipeline (YOLO11n detector, ByteTrack tracker, directional tripwire counting, tumbling window flow, polygonal ROI density estimation, and stationary vehicle detection). Milestones 5A through 5C developed the multimodal foundation:
- **Milestone 5A**: Established the multimodal scene understanding architecture and epistemic taxonomy ([ADR-0004](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/docs/adr/0004-vlm-integration-architecture.md)).
- **Milestone 5B**: Implemented deterministic keyframe extraction (`KeyframeExtractor`) with strict timestamp integrity and verified frame-index semantics, and structured evidence packaging (`EvidencePackager`).
- **Milestone 5C**: Integrated Google Gemini REST API (`GeminiVLMProvider`), centralized epistemic grounding verification (`GroundingVerifier`), and deterministic verified CV fact rendering.

However, these components currently exist as standalone service layers and integration libraries without persistent orchestration or public backend API exposure.

Integrating VLM scene understanding into the TrafficLens-AI backend introduces critical operational requirements and failure modes:

1. **Lifecycle Independence**:
   The primary CV pipeline is deterministic, local, and compute-heavy (OpenCV decoding, YOLO inference, ByteTrack association). In contrast, VLM scene understanding is probabilistic, remote, network-bound, and subject to external provider latencies (2–10 seconds), rate limits (HTTP 429), safety blocks, or network timeouts.
   If the VLM stage is coupled synchronously to the primary CV job lifecycle, any external VLM outage, transient network fault, or rate-limit throttle would either stall the CV worker thread or cause the primary `ProcessingJob` to fail, discarding valuable, compute-intensive CV analytics.

2. **Resource Constraints and Execution Profiles**:
   The target environment is resource-bounded (4 GB VRAM GPU, 8 GB RAM, CPU-bound video processing). Introducing heavy distributed queue infrastructure (Redis, Celery, RabbitMQ, Kafka) introduces unnecessary operational complexity, excessive memory overhead, and external dependency failure surfaces.
   At the same time, executing network-bound VLM requests inside the CPU-bound `JobWorker` thread would serialize video decoding and remote I/O, creating artificial queue starvation.

3. **On-Demand vs. Automatic Ingestion**:
   Certain operational workflows require immediate scene understanding upon video completion, whereas others require CV analytics only, with VLM scene reasoning requested on-demand by human analysts for specific completed sequences.

4. **Multi-Run History and Reprocessing**:
   Users and evaluation benchmarks may need to re-run scene interpretation with different VLM providers, updated prompt configurations, or following transient provider outages without re-running the 30-second to multi-minute CV video decoding pipeline.

## Decision

We adopt a **Decoupled, Dual-Worker Asynchronous Orchestration Architecture** governed by the following core architectural decisions:

### 1. Dedicated Domain Entity (`SceneInterpretation`)

We introduce a dedicated domain entity and database table, `scene_interpretations`, distinct from `processing_jobs`:
- **Relationship**: 1:N relationship from `ProcessingJob` to `SceneInterpretation` (`ProcessingJob.scene_interpretations`, foreign key `job_id` referencing `processing_jobs.id` with `ON DELETE CASCADE`).
- **Identity & Schema**:
  - `id`: String(36) UUID primary key (`interp_<uuid>`).
  - `job_id`: String(36) foreign key referencing `processing_jobs.id`.
  - `video_id`: String(36) referencing `video_assets.id`.
  - `status`: String enum `SceneInterpretationStatus` (`QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`).
  - `created_at`, `started_at`, `completed_at`, `duration_seconds`: Precision UTC timestamps.
  - `provider_type`: String (e.g., `gemini`, `openai_compatible`, `mock`).
  - `model_name`: String (e.g., `gemini-3.8-flash`).
  - `prompt_tokens`, `completion_tokens`, `total_tokens`: Token accounting metadata.
  - `summary`: Text summary of the scene.
  - `environmental_context`: JSON structure storing weather, lighting, road condition, and camera perspective.
  - `scene_description`: JSON structure storing road layout and vehicle composition.
  - `event_interpretations`: JSON array of grounded interpretations for detected traffic events.
  - `epistemic_claims`: JSON array of categorized and verified claims (`cv_fact`, `visual_observation`, `inference`, `unverified`).
  - `unverified_aspects`: JSON array of explicit visual or contextual uncertainties.
  - `evidence_package_id`: String identifier linking to the generated evidence package.
  - `error_code`, `error_message`, `error_details`: Structured failure descriptors when `status == FAILED`.
- **Query Semantics**:
  - The default single-entity API (`GET /api/v1/jobs/{job_id}/interpretation`) retrieves the latest interpretation by `created_at`.
  - If no interpretation has been requested for a job, the endpoint returns a virtual status payload with `status="NOT_REQUESTED"`.
  - Historical interpretations remain accessible via `GET /api/v1/jobs/{job_id}/interpretations`.

### 2. Fully Decoupled State Machines & Immutable Attempts

The lifecycle of a `ProcessingJob` and a `SceneInterpretation` are strictly decoupled:
- **`ProcessingJob` Lifecycle**:
  `QUEUED` $\to$ `PROCESSING` $\to$ `COMPLETED` | `FAILED`
  The primary job reaches `COMPLETED` the instant CV inference, tracking, spatiotemporal analytics, and disk artifact persistence finish. Its status is never held in `PROCESSING` while awaiting VLM inference, and never transitions to `FAILED` if VLM inference fails.
- **`SceneInterpretation` Lifecycle (Strictly Unidirectional & Terminal)**:
  `[NOT_REQUESTED]` $\to$ `QUEUED` $\to$ `PROCESSING` $\to$ `COMPLETED` | `FAILED`
  Both `COMPLETED` and `FAILED` are strict terminal states. A `SceneInterpretation` record is immutable once terminal.
  - When a user or system retries a failed interpretation, the system **creates a new `SceneInterpretation` record** linked to the same `ProcessingJob`.
  - All previous attempts, execution durations, and error details remain intact in the database for auditability and inspection.
  - A maximum limit of `MAX_INTERPRETATION_ATTEMPTS = 3` total records (including the initial request) is enforced per CV job.
  - At most one active interpretation (`QUEUED` or `PROCESSING`) is permitted for any CV job at any time.

```
+-----------------------------------------------------------------------------------+
|                              Primary CV Job Lifecycle                             |
|                                                                                   |
|  [ QUEUED ]  ======>  [ PROCESSING ]  ======>  [ COMPLETED ]                      |
|                              |                                                    |
|                              v                                                    |
|                         [ FAILED ]                                                |
+-----------------------------------------------------------------------------------+
                                                        |
                                                        | Triggers (Opt-in or On-Demand)
                                                        v
+-----------------------------------------------------------------------------------+
|                     Scene Interpretation Attempt Lifecycle                        |
|                                                                                   |
|  [ NOT_REQUESTED ]  -->  [ QUEUED ]  -->  [ PROCESSING ]  -->  [ COMPLETED (Term)]|
|                                                  |                                |
|                                                  v                                |
|                                             [ FAILED (Term)]                      |
|                                                  |                                |
|                                                  +-----> (Retry creates NEW row)  |
|                                                          (Up to 3 total attempts) |
+-----------------------------------------------------------------------------------+
```

### 3. Dual-Worker In-Process Concurrency (`InterpretationWorker`) & Single-Process Scope

To respect the local single-process deployment profile while preventing I/O starvation:
- We implement a dedicated `InterpretationWorker` operating independently from the video processing `JobWorker`.
- **Queue Mechanics**: Uses an in-process bounded queue (`queue.Queue[str]` with capacity `settings.INTERPRETATION_MAX_QUEUE_SIZE`, default 20) with atomic deduplication (`_queued_ids: set[str]`).
- **Thread Isolation**: Runs in a separate background daemon thread named `TrafficLens-InterpretationWorker`. This ensures that slow remote HTTP requests, exponential backoff pauses, and network timeouts never block the OpenCV/YOLO video decoding loop in `TrafficLens-Worker`.
- **Async Execution Bridge**: The worker thread executes the asynchronous `SceneInterpretationService.interpret_scene()` call within an isolated thread-local event loop (`asyncio.run()` or dedicated loop), ensuring clean integration with non-blocking HTTP clients (`httpx.AsyncClient`).
- **Single-Process Constraint**: This in-process queue, threading mutex, and daemon-worker architecture strictly assumes a **single application process** (`uvicorn --workers 1`). Multi-worker Uvicorn (`--workers > 1`) and multi-instance container horizontal scaling are explicitly **not supported** with this concurrency model. Scaling to multiple processes or nodes represents a future migration seam requiring a database-backed queue with row leases (`SELECT ... FOR UPDATE SKIP LOCKED`) or an external message broker (Phase 6).

### 4. Triggering Modes, Concurrency Guards & Retry Limits

We support both automated and on-demand execution:
1. **On-Demand Execution (Default & Recommended)**:
   - Client issues `POST /api/v1/jobs/{job_id}/interpretation`.
   - Pre-condition: The associated `ProcessingJob` must be in `COMPLETED` status. If still `PROCESSING` or `QUEUED`, the endpoint returns HTTP `409 Conflict`. If `FAILED`, returns HTTP `400 Bad Request`.
   - **Active-Task Concurrency Guard**: Guarded by a process-level mutex (`_interpretation_create_lock = threading.Lock()`) and a database-level partial unique index (`uix_active_interpretation_per_job` on `(job_id) WHERE status IN ('QUEUED', 'PROCESSING')`). If an active interpretation already exists, returns HTTP `409 Conflict`.
   - **Attempt Limit Enforcement**: If total interpretation records for `job_id` have reached `MAX_INTERPRETATION_ATTEMPTS = 3`, the request is rejected with HTTP `409 Conflict` (`"Maximum interpretation attempt limit (3) exhausted for this job."`).
2. **Automated Post-Processing (Opt-In)**:
   - Enabled via job parameter `enable_vlm_interpretation: bool = True` during job creation or globally via `AUTO_TRIGGER_VLM_INTERPRETATION=True` in backend settings.
   - Upon successfully transitioning `ProcessingJob` to `COMPLETED`, `JobWorker` calls `InterpretationWorker.enqueue_job(job_id)`.
   - If the interpretation queue is full, the failure is logged and recorded as a `FAILED` interpretation task with `error_code="QUEUE_FULL"`; the CV job remains `COMPLETED`.
3. **Manual Retry (`POST /api/v1/jobs/{job_id}/interpretation/retry`)**:
   - Creates a new `SceneInterpretation` row if the latest attempt is in `FAILED` status and total records $< 3$.
   - Rejects with HTTP `409 Conflict` if an active interpretation is currently running or if the attempt limit (3) is exhausted.
   - Rejects with HTTP `400 Bad Request` if the latest attempt is not in `FAILED` status.
   - **Ambiguous Timeout Warning**: If a previous attempt failed due to `VLM_PROVIDER_TIMEOUT`, the external provider may have processed the request after network socket closure. Initiating a retry will execute a new Gemini API request and **can incur an additional Gemini API charge**. At-least-once external execution semantics apply.

### 5. Deterministic Orchestration Pipeline & Single Atomic Commit

When `InterpretationWorker` claims an interpretation task:
1. **Atomic DB Claim**: Atomically transitions `SceneInterpretation` from `QUEUED` to `PROCESSING` with `started_at = now()` using a conditional update (`WHERE id=:id AND status='QUEUED'`).
2. **Artifact & Telemetry Retrieval**:
   - Loads CV analytics summary, detection summary, tracking summary, and stopped vehicle events from the job's persisted JSON artifacts (`data/artifacts/{job_id}/`) and database metadata.
3. **Deterministic Keyframe Extraction**:
   - Calls `KeyframeExtractor.extract_keyframes(video_path, event_timestamps)` using the verified video file.
   - Extracts at most $K = 5$ frames respecting timestamp integrity and verified OpenCV frame-index semantics.
4. **Evidence Packaging**:
   - Invokes `EvidencePackager.package_evidence()` to produce a validated `EvidencePackageSchema`.
5. **VLM Scene Understanding & Epistemic Grounding**:
   - Passes evidence package to `SceneInterpretationService.interpret_scene()`.
   - The provider executes Gemini inference with structured JSON output contracts.
   - `GroundingVerifier` validates all citations:
     - Observations must cite valid keyframes.
     - CV facts must match verified structured telemetry; free-text assertions are deterministically overwritten with authoritative rendered statements.
     - Unsupported domains or ungrounded assertions are demoted to `unverified`.
6. **Single Atomic Persistence & Completion**:
   - Persists the validated interpretation results (summary, environmental context, scene description, event interpretations, epistemic claims, unverified aspects, tokens, timestamps) and transitions status to `COMPLETED` in a **single atomic database transaction**. Partial commits are strictly prohibited.
   - If an error occurs at any point (keyframe extraction failure, provider timeout, rate limit, schema mismatch), records status as `FAILED` with explicit `error_code`, `error_message`, and `error_details` in a single atomic transaction.

### 6. Crash Recovery & Resilience

- **Database as Source of Truth**:
  The in-memory queue is only a transient scheduling mechanism.
- **Startup Recovery**:
  During application startup in `lifespan` (`main.py`), `InterpretationWorker.start()` invokes `recover_interpretations()`:
  1. Identifies all `SceneInterpretation` records in `PROCESSING` state (interrupted by server restart or crash) and marks them `FAILED` with `error_code="INTERRUPTED_BY_SHUTDOWN"`.
  2. Scans for persisted `QUEUED` interpretations and redispatches them into the active in-memory queue up to capacity.
  3. **No Automatic Retry of External Calls**: Interrupted tasks are never automatically retried by the recovery routine to avoid infinite crash loops or runaway token expenditure. At-least-once external execution semantics require explicit client intervention for retries.
- **Graceful Shutdown**:
  During `lifespan` shutdown, signals `InterpretationWorker.stop(timeout=5.0)`. If a running interpretation does not terminate within the timeout, marks it `FAILED` with `error_code="INTERRUPTED_BY_SHUTDOWN"` to prevent orphaned `PROCESSING` rows.

## Considered Options

1. **Option A: Monolithic Job Status Extension (`JobStatus.VLM_PROCESSING`)**:
   - *Rejected*: Violates single responsibility and fault isolation. A failure in Gemini (e.g., 429 quota exhaustion or network drop) would leave the entire video processing job in a failed or indeterminate state, preventing users from accessing passing CV analytics.
2. **Option B: Synchronous Inline Execution inside `POST /jobs` or CV Worker**:
   - *Rejected*: Incurring 5–15 seconds of remote VLM API latency inside the CV worker thread would block downstream video processing jobs, causing severe throughput collapse.
3. **Option C: Heavy Distributed Task Queue (Celery + Redis / RabbitMQ)**:
   - *Rejected*: Incompatible with the lightweight, zero-dependency laptop/edge deployment constraint (8 GB RAM). Introduces external broker daemon management, serialization complexity, and dual-source-of-truth reconciliation problems.
4. **Option D: Ephemeral / Non-Persisted VLM Results (Client Polling Only)**:
   - *Rejected*: Discarding expensive VLM reasoning results forces wasteful repeated remote inference calls, impairs auditability, and prevents historical comparison.

## Consequences

### Positive

- **Total Fault Isolation**: The primary CV pipeline operates completely independently; zero risk of CV job failure from external VLM errors.
- **Responsive Architecture**: Compute-heavy CV processing and I/O-bound VLM queries run on dedicated worker threads with zero queue interference.
- **Comprehensive Traceability**: Every interpretation run is persisted with its provider, model, token usage, timestamps, evidence package ID, and epistemic claims.
- **Crash Consistency**: The database is the single source of truth; server restarts cleanly fail orphaned runs and resume queued work.
- **Resource Efficiency**: Implemented purely with standard library threading, bounded in-process queues, and existing SQLAlchemy/SQLite infrastructure without external message brokers.

### Negative / Trade-offs

- **Increased Schema Complexity**: Requires an additional database table (`scene_interpretations`), Alembic migration, repository, and service classes.
- **Dual Worker Lifecycle**: Application startup and shutdown routines must coordinate both `JobWorker` and `InterpretationWorker`.
- **Storage Footprint**: Persisting structured interpretation outputs and keyframe evidence packages consumes disk space in SQLite and artifact directories.