# TrafficLens-AI — Backend Service

The `backend/` package provides the production-grade FastAPI application, SQLAlchemy 2.0 persistence layer, Alembic database migrations, and domain services for TrafficLens-AI. It serves as the bridge between client applications and the computer vision analytics pipeline.

---

## 1. Directory Structure & Responsibilities

```
backend/
├── alembic/                 # Alembic migration environment and version scripts
│   ├── env.py               # Migration context loader (Settings + Base.metadata)
│   ├── script.py.mako       # Revision template
│   └── versions/            # Versioned migration scripts (0001_initial_schema, ...)
├── app/
│   ├── main.py              # FastAPI application entrypoint, CORS, lifespan hooks
│   ├── dependencies.py      # Dependency injection providers (Settings, Database Session)
│   ├── api/
│   │   ├── router.py        # Central API router aggregating route modules
│   │   └── routes/
│   │       ├── health.py    # Service health, environment, and dialect inspection
│   │       ├── jobs.py      # Video processing job creation, polling, telemetry, and results
│   │       └── videos.py    # Multipart video streaming upload and metadata extraction
│   ├── core/
│   │   ├── config.py        # Pydantic Settings with SQLite/PostgreSQL configuration
│   │   └── logging.py       # Structured JSON and development console logging
│   ├── db/
│   │   ├── __init__.py      # Engine, sessionmaker, and session dependencies
│   │   └── session.py       # create_db_engine, SQLite PRAGMA hook, get_db generator
│   ├── models/
│   │   ├── __init__.py      # Declarative Base, VideoAsset, ProcessingJob
│   │   ├── base.py          # Base class and UTCDateTime TypeDecorator
│   │   ├── video.py         # VideoAsset model (files, duration, FPS, dimensions)
│   │   └── job.py           # ProcessingJob model (status, errors, timestamps, config)
│   ├── schemas/
│   │   ├── common.py        # Geometric primitives (BBox, Points, Lines, ROIs)
│   │   ├── job.py           # Job lifecycle state machine and video upload contracts
│   │   └── telemetry.py     # CV detection, tracking, flow, density, and congestion schemas
│   ├── services/
│   │   ├── cv_pipeline_adapter.py # Seam decoupling backend schemas from native CV engines
│   │   ├── job_service.py   # Job lifecycle coordinator enforcing allowed state transitions
│   │   ├── media_delivery.py# RFC 9110 HTTP Range media streaming and path containment
│   │   └── worker.py        # Bounded asynchronous background worker and crash recovery
│   └── repositories/
│       ├── __init__.py
│       └── job_repository.py# SQLAlchemyJobRepository implementation & protocol
└── tests/
    ├── test_api_endpoints.py            # HTTP endpoint integration tests
    ├── test_config.py                   # Settings and environment variable tests
    ├── test_cv_serialization.py         # Real CV artifact validation tests
    ├── test_ingestion_and_orchestration.py # Video upload, worker flow, and crash recovery tests
    ├── test_job_lifecycle.py            # State machine transition and rejection tests
    ├── test_media_delivery.py           # HTTP Range streaming, path containment, artifact listing tests
    ├── test_persistence.py              # SQLAlchemy models, sessions, constraints, Alembic tests
    └── test_schemas.py                  # Pydantic schema validation tests
```

---

## 2. Database Models & Schema Design

Database persistence uses **SQLAlchemy 2.0** declarative mapping with timezone-aware UTC timestamps and foreign-key cascade deletes.

### 2.1 `video_assets` Table
Represents uploaded video files registered on disk.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `VARCHAR(36)` | Primary Key | Unique asset identifier (e.g. UUID string) |
| `filename` | `VARCHAR(255)` | NOT NULL, Index | Original uploaded filename |
| `storage_path` | `VARCHAR(512)` | NOT NULL, Unique | Relative or absolute path on disk |
| `content_type` | `VARCHAR(100)` | NOT NULL | MIME content type (e.g. `video/mp4`) |
| `file_size_bytes` | `BIGINT` | NOT NULL | Video file size in bytes |
| `duration_seconds` | `FLOAT` | NULLABLE | Video duration in seconds |
| `fps` | `FLOAT` | NULLABLE | Video frame rate |
| `width` | `INTEGER` | NULLABLE | Frame width in pixels |
| `height` | `INTEGER` | NULLABLE | Frame height in pixels |
| `created_at` | `TIMESTAMP WITH TZ` | NOT NULL, Index | UTC creation timestamp |
| `updated_at` | `TIMESTAMP WITH TZ` | NOT NULL | UTC last modification timestamp |

### 2.2 `processing_jobs` Table
Represents asynchronous CV pipeline processing runs over an associated `VideoAsset`.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `VARCHAR(36)` | Primary Key | Unique job identifier (`job_...`) |
| `video_id` | `VARCHAR(36)` | Foreign Key, NOT NULL, Index | References `video_assets.id` (`ON DELETE CASCADE`) |
| `sequence_name` | `VARCHAR(100)` | NOT NULL, Index | Logical sequence identifier (e.g. `MVI_20011`) |
| `status` | `VARCHAR(20)` | NOT NULL, Index | `QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED` |
| `created_at` | `TIMESTAMP WITH TZ` | NOT NULL, Index | UTC job creation timestamp |
| `started_at` | `TIMESTAMP WITH TZ` | NULLABLE | UTC processing start timestamp |
| `completed_at` | `TIMESTAMP WITH TZ` | NULLABLE | UTC processing termination timestamp |
| `duration_seconds` | `FLOAT` | NULLABLE | Elapsed processing wall-clock time |
| `error_code` | `VARCHAR(100)` | NULLABLE | Canonical error code if status == `FAILED` |
| `error_message` | `TEXT` | NULLABLE | Detailed error message if status == `FAILED` |
| `error_details` | `JSON` | NULLABLE | Contextual metadata/stack trace |
| `config_metadata` | `JSON` | NULLABLE | Serialized virtual line and ROI definitions |

---

## 3. Database Migrations with Alembic

Alembic manages schema evolution for both SQLite and PostgreSQL.

### Migration Commands
```bash
# Run all pending migrations
alembic upgrade head

# Roll back the most recent migration
alembic downgrade -1

# Roll back all migrations to empty database
alembic downgrade base

# Generate a new autogenerated migration script after model changes
alembic revision --autogenerate -m "describe_changes"

# View current migration version
alembic current
```

---

## 4. Video Ingestion, Storage & Job Orchestration

Milestone 3 implements the end-to-end ingestion, asynchronous execution, and telemetry results retrieval pipeline:

### 4.1 Storage Layout
```
storage/
├── uploads/                     # Uploaded raw video streams
│   └── vid_{uuid}.mp4           # Safe, collision-free internal naming
└── outputs/                     # Job-isolated output artifacts
    └── job_{uuid}/              # Isolated directory per processing run
        ├── analytics.json       # Structured spatiotemporal analytics report
        ├── telemetry.csv        # Frame-level telemetry CSV export
        └── annotated.mp4        # Rendered visualization video with bounding boxes & ROIs
```

### 4.2 Ingestion & Orchestration Endpoints
- **`POST /api/v1/videos`**: Streams multipart video files directly to disk in 1 MB chunks without buffering whole files in memory. Enforces extension allowlist (`.mp4`, `.avi`, `.mov`, `.mkv`), maximum upload limit (`MAX_UPLOAD_SIZE_BYTES`, default 500 MB), and verifies stream readability via OpenCV (`cv2.VideoCapture`). Extracts native metadata (`duration`, `fps`, `width`, `height`, `frame_count`) and registers a `VideoAsset` record. Corrupt, empty, or oversized files are rejected immediately and cleaned up from disk.
- **`GET /api/v1/videos/{video_id}`**: Retrieves metadata for an existing video asset.
- **`POST /api/v1/videos/{video_id}/jobs`**: Schedules a video asset for asynchronous CV processing. Prevents duplicate concurrent scheduling: if an active job already exists in `QUEUED` or `PROCESSING` state for the video, the endpoint rejects the request with **HTTP 409 Conflict**. Upon acceptance, returns **HTTP 202 Accepted** with job status `QUEUED` and dispatches to the background worker.
- **`GET /api/v1/jobs/{job_id}/results`**: Retrieves complete spatiotemporal results and artifact file paths. Rejects with **HTTP 409 Conflict** if the job is still in progress (`QUEUED` or `PROCESSING`) or has `FAILED`. When `COMPLETED`, returns detection summary, tracking metrics, line flow analytics, stopped vehicle events, and artifact disk paths.

### 4.3 Background Worker Architecture & Startup Recovery
- **Single-Process Worker (`JobWorker`)**: A bounded, thread-based background worker runs in a dedicated daemon thread, consuming jobs from a thread-safe queue. The FastAPI async event loop is never blocked.
- **Database as Source of Truth**: The worker atomically claims jobs by transitioning state from `QUEUED -> PROCESSING` before invoking the CV pipeline adapter.
- **Startup Crash/Restart Recovery**: On application boot (`lifespan`), `JobWorker.recover_jobs()` inspects the database:
  - Any job left in `PROCESSING` state (due to server crash or hard reboot) is marked `FAILED` with `error_code="INTERRUPTED_BY_SHUTDOWN"`.
  - Any jobs in `QUEUED` state are safely re-enqueued into the worker queue.

---

## 5. Key Architectural Decisions

1. **Decoupled CV Boundary**: HTTP routes and backend services never directly import low-level CV scripts or model weights. All interaction flows through [`CVPipelineAdapter`](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/backend/app/services/cv_pipeline_adapter.py) and typed Pydantic contracts.
2. **Streaming Video Ingestion & OpenCV Validation**: Raw video streams are streamed directly to disk in 1 MB chunks to prevent RAM bloat. Headers and decodability are validated immediately with OpenCV before registering assets.
3. **Asynchronous File-Based Processing**: Video upload registers an asset and queues a job (`status: QUEUED`). Background runners process frames asynchronously, transitioning states to `PROCESSING`, `COMPLETED`, or `FAILED`.
4. **Strict Job State Machine & Duplicate Prevention**: Status transitions are strictly validated. Duplicate scheduling on already active videos is rejected with HTTP 409 Conflict.
5. **SQLAlchemy 2.0 with Dual Dialect Compatibility**: SQLite (`sqlite:///./trafficlens.db`) is the local development default with zero external daemon requirements (enforcing `PRAGMA foreign_keys=ON;`). Configuration, queries, and types remain 100% compatible with PostgreSQL (`postgresql+psycopg://...`).
6. **Zero Premature Distributed Infrastructure**: No Celery, Redis, Kafka, or microservices. In-process asynchronous task management fulfills all latency and throughput requirements.
7. **Secure Media Delivery & RFC 9110 Range Streaming**: Source video files and rendered annotated MP4s are streamed using bounded 256 KB chunk generators with full support for single byte ranges (`206 Partial Content`), open-ended ranges, and suffix ranges. No static file mounts exist; server filesystem paths are never exposed to clients; and all requests strictly resolve through database ownership and server-side directory containment checks.

---

## 6. Configuration & Environment Variables

Settings are managed via `backend/app/core/config.py` using `pydantic-settings`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PROJECT_NAME` | `TrafficLens-AI` | Application identifier |
| `ENVIRONMENT` | `development` | Deployment environment (`development`, `testing`, `production`) |
| `API_V1_STR` | `/api/v1` | Base API routing prefix |
| `DATABASE_URL` | `sqlite:///./trafficlens.db` | SQLAlchemy connection string (SQLite or PostgreSQL) |
| `UPLOAD_DIR` | `./storage/uploads` | Path for uploaded video files |
| `OUTPUT_DIR` | `./storage/outputs` | Path for generated telemetry and annotated media |
| `MAX_UPLOAD_SIZE_BYTES` | `524288000` (500 MB) | Maximum upload payload size |
| `CONF_THRESHOLD` | `0.25` | Frozen detector operating threshold |
| `TRACK_BUFFER` | `45` | Frozen ByteTrack trajectory memory buffer |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | `json` | Log format (`json` or `text`) |

---

## 7. Running the Backend Locally

```bash
# Apply migrations
alembic upgrade head

# Start the backend server with hot-reload
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

# Access interactive documentation
# Swagger UI: http://127.0.0.1:8000/docs
# ReDoc: http://127.0.0.1:8000/redoc
# Health check: http://127.0.0.1:8000/api/v1/health
```

---

## 8. Running Tests

```bash
# Run backend persistence, ingestion, media delivery, and contract tests (67 tests)
pytest backend/tests

# Run entire repository test suite (131 tests: CV pipeline + Backend)
pytest
```
