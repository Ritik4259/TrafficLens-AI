# ADR 0002: Backend Architecture, Job Lifecycle, and Data Contracts

## Context

Phase 1 established the computer vision pipeline (YOLO11n vehicle detection, ByteTrack tracking, spatiotemporal analytics, and verified telemetry exports). In Milestones 1 and 2 of Phase 2, we are establishing the backend architecture, persistence models, and data contracts for TrafficLens-AI.

The backend must fulfill several core criteria:
1. Provide a clean HTTP interface for video ingestion, job coordination, and telemetry retrieval.
2. Prevent tight coupling between web endpoints and the computational computer vision layer.
3. Establish a predictable asynchronous job lifecycle (`QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`).
4. Expose typed, rigorously validated API contracts matching actual CV telemetry outputs without fabricating unsupported metrics (such as uncalibrated km/h).
5. Ensure zero-dependency local development via SQLite while maintaining full dialect compatibility for seamless PostgreSQL migration.
6. Provide robust schema migrations and foreign-key integrity.

## Decision

We adopt the following backend architecture and persistence design:

1. **Modular Layered Architecture**:
   The backend resides under `backend/app/` partitioned into:
   - `core/`: Environment settings (`Settings`) and structured logging.
   - `schemas/`: Pydantic v2 data contracts.
   - `models/`: SQLAlchemy 2.0 declarative models (`VideoAsset`, `ProcessingJob`).
   - `db/`: Engine configuration, SQLite PRAGMA hook, and request-scoped session generators.
   - `services/`: Domain business logic (`JobService`) and the CV pipeline adapter seam (`CVPipelineAdapter`).
   - `repositories/`: Persistence repositories (`SQLAlchemyJobRepository`) implementing explicit protocol boundaries (`JobRepository`).
   - `api/routes/`: Thin FastAPI controllers focused strictly on request/response validation and HTTP status codes.

2. **Decoupled CV Adapter Seam (`CVPipelineAdapter`)**:
   Backend controllers and services never import low-level inference scripts or model files. Instead, `CVPipelineAdapter` translates API schemas to internal CV configurations and converts native `SequenceAnalyticsReport` models into validated `TrafficAnalyticsSummarySchema` contracts.

3. **Asynchronous Processing State Machine**:
   Video processing is treated as an asynchronous file-based job with a formal state machine:
   - Initial state: `QUEUED`.
   - Allowed transitions: `QUEUED -> PROCESSING`, `QUEUED -> FAILED`, `PROCESSING -> COMPLETED`, `PROCESSING -> FAILED`.
   - Terminal states: `COMPLETED`, `FAILED` (no further transitions permitted).
   - Any disallowed transition raises `InvalidJobStatusTransitionError` and is rejected before any database write occurs.

4. **Persistence & Dual Dialect Compatibility**:
   - We standardize on **SQLAlchemy 2.0 ORM** with declarative typed mappings (`Mapped[...]`, `mapped_column(...)`).
   - Development defaults to SQLite (`sqlite:///./trafficlens.db`). SQLite connections strictly execute `PRAGMA foreign_keys = ON;` upon connection to guarantee relational integrity.
   - All models, types, and column definitions remain 100% compatible with PostgreSQL (`postgresql+psycopg://...`).
   - All datetime columns use custom `UTCDateTime` TypeDecorators ensuring timezone-aware UTC datetime instances across both SQLite and PostgreSQL dialects.
   - **Alembic** manages database migrations in `backend/alembic/` with `render_as_batch=True` for SQLite batch table alteration support.

5. **Entity Relationships & Cascade Deletion**:
   - `VideoAsset` has a one-to-many relationship with `ProcessingJob`.
   - Deleting a parent `VideoAsset` cascades and deletes all associated `ProcessingJob` rows both in the ORM (`cascade="all, delete-orphan"`) and via DDL foreign key (`ondelete="CASCADE"`).

6. **Strict Grounding of Data Contracts**:
   API schemas reflect verified outputs from the CV pipeline:
   - Directional counting lines and Vehicles Per Minute (VPM) flow.
   - Distinct labels for vehicle count and spatial occupancy ratio.
   - Speed reporting explicitly flagged as uncalibrated (`speed_kmh: Optional[float] = None`, `is_calibrated: bool = False`) until camera homographies are configured.
   - Explainable congestion states with human-readable rationale.

7. **Chunked Streaming Video Ingestion & Defensive Validation**:
   Raw video uploads are written to disk in 1 MB chunks without buffering entire payloads in memory. Upload size is capped at `MAX_UPLOAD_SIZE_BYTES` (500 MB). The stored file is immediately validated using OpenCV (`cv2.VideoCapture`) to inspect stream headers and read the initial frame. Corrupt, empty, or unreadable files are removed from disk immediately and rejected with HTTP 400 Bad Request. Safe internal naming (`storage/uploads/vid_{uuid}.ext`) completely eliminates path traversal and collision risks.

8. **Duplicate Job Scheduling Prevention**:
   A video asset can have at most one active job (`QUEUED` or `PROCESSING`) at any time. Submitting a new job for a video asset that is already active is rejected with HTTP 409 Conflict to prevent redundant GPU/CPU resource contention.

9. **Bounded Background Worker with Atomic Claim Semantics**:
   A thread-based background worker (`JobWorker`) continuously processes jobs from a thread-safe queue. The worker claims jobs atomically (`QUEUED -> PROCESSING`) before invoking the CV pipeline adapter. Frame reading and inference occur in the dedicated worker thread, ensuring the FastAPI asynchronous event loop is never blocked. Output artifacts are isolated per job in `storage/outputs/{job_id}/` (`analytics.json`, `telemetry.csv`, `annotated.mp4`).

10. **Startup Crash Recovery Policy**:
   Upon backend boot, `JobWorker.recover_jobs()` queries the database for interrupted jobs:
   - Any job left in `PROCESSING` state (due to server termination or power loss) is transitioned to `FAILED` with `error_code="INTERRUPTED_BY_SHUTDOWN"`.
   - Any job in `QUEUED` state is safely re-enqueued for background execution.

11. **No Distributed Infrastructure in Current Phase**:
    Distributed message brokers (Celery, Redis, Kafka) and microservices are explicitly rejected at this stage. In-process asynchronous task execution meets all throughput and architectural goals.

## Considered Options

1. **Synchronous Request-Response Video Processing**:
   - Rejected: Video inference takes tens of seconds to minutes. Synchronous blocking over HTTP would lead to client gateway timeouts (HTTP 504) and poor user experience.
2. **Immediate Celery / Redis Task Queue Setup**:
   - Rejected: Introduces external daemon dependencies (Redis broker, Celery worker processes) prematurely for local development and benchmarking without architectural necessity.
3. **Direct Exposure of Native CV Classes in API Controllers**:
   - Rejected: Tightly couples FastAPI controllers to PyTorch, Ultralytics, and NumPy data structures, compromising testability and maintainability.
4. **Row-per-Detection Database Persistence**:
   - Rejected: Storing tens of thousands of bounding box detections as individual relational database rows would create unnecessary write amplification and I/O bottlenecks. Aggregated sequence telemetry, line flows, and discrete events are persisted instead.

## Consequences

- The backend can be tested and verified rapidly with `pytest` without invoking full video inference runs.
- Asynchronous job status polling provides a reliable pattern for frontend integration.
- The separation between API schemas and internal CV configs allows independent evolution of both layers.
- SQLite enables instant onboarding for developers, while PostgreSQL compatibility ensures zero-friction deployment to production cloud environments.
- Alembic migrations provide auditability and reproducibility for database schema evolution.
