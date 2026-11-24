# TrafficLens-AI — Project Roadmap

**TrafficLens-AI**: Intelligent traffic understanding and analytics platform leveraging Computer Vision, Multi-Object Tracking, Spatiotemporal Analytics, Multimodal Vision-Language Models, and Agentic Workflows.

---

## High-Level Project Phases

| Phase | Title | Focus & Core Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **Phase 0** | **Architecture & Scaffolding** | Repository layout, ADR-0001, GLOSSARY.md, evaluation conventions. | ✅ Completed |
| **Phase 1** | **Dataset & CV Baseline** | UA-DETRAC XML parser with ignore-region handling, 8-sequence benchmark split, YOLO11n detection baseline (AP@0.50: 93.77%, F1: 92.44%), ByteTrack MOT baseline (HOTA: 68.46%, MOTA: 81.37%, IDF1: 87.06%). | ✅ Completed |
| **Phase 2** | **Core Backend & Ingestion** | FastAPI modular monolith architecture, SQLAlchemy 2.0 models, Alembic migrations, chunked streaming video ingestion, OpenCV stream validation, bounded background worker, crash recovery, atomic status transitions, real-video smoke test. | ✅ Completed |
| **Phase 3** | **Event Engine & Core Analytics** | Spatiotemporal analytics layer (`experiments/src/analytics/`): directional tripwire counting, tumbling window flow (VPM), polygonal ROI density & spatial occupancy, explainable rule-based congestion classifier, stationary vehicle detector. | ✅ Completed |
| **Phase 4** | **Frontend Web Dashboard** | Modern React + TypeScript + Vite + Tailwind CSS dashboard. Flagship Video Analysis Workspace, synchronized dual-stream video inspection, HTTP Range media delivery, interactive telemetry visualization. | ✅ Completed |
| **Phase 5** | **Multimodal VLM & Agentic AI** | Multimodal scene understanding with Vision-Language Models, deterministic evidence packaging, grounded contextual event interpretation, LangGraph analytical query agent. | 🚀 Active |
| **Phase 6** | **Containerization & Deployment** | Multi-service `docker-compose.yml` (FastAPI backend + React frontend + PostgreSQL), production container optimization, end-to-end integration verification. | ⏳ Planned |

---

## Phase 4: Frontend Web Dashboard Milestones (Completed)

Phase 4 delivered the production-ready user interface and media streaming capabilities:

- **Milestone 1: Architecture, Design System & Data Contracts**: Obsidian Optical design system, strict TypeScript models, ADR-0003. ✅ Completed
- **Milestone 2: Project Scaffolding & Core Workstation Layout**: App shell, router, live health indicator, sample fixtures. ✅ Completed
- **Milestone 3: Video Ingestion, Library & Real-Time Job Polling**: Drag-and-drop chunked uploader, catalog table, job configuration modal, status polling. ✅ Completed
- **Milestone 4A: Backend Media Delivery Architecture**: Byte-range video streaming (`206 Partial Content`), path traversal security containment, `Cache-Control: private, no-store`. ✅ Completed
- **Milestone 4B: Video Playback & CV Visualization**: `VideoInspectionPlayer` with 60Hz rAF drift compensation ($0.10\text{ ms}$ median drift), single-frame stepping ($\pm 40\text{ ms}$), dual-stream lock, class-agnostic baseline grounding, and 166 passing automated regression tests. ✅ Completed

---

## Phase 5: Multimodal VLM & Agentic AI Milestones

Phase 5 introduces qualitative multimodal intelligence to TrafficLens-AI by pairing deterministic CV measurements with Vision-Language Models and conversational agentic workflows.

### Milestone 5A: VLM Architecture & Model Feasibility (Current)
- **Status**: ✅ Completed
- **Focus**: Research, architecture specification, model feasibility, output contracts, and evaluation methodology.
- **Deliverables**:
  - Architecture Decision Record ([ADR-0004](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/docs/adr/0004-vlm-integration-architecture.md)).
  - Comprehensive technical report ([docs/vlm-architecture-and-model-evaluation.md](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/docs/vlm-architecture-and-model-evaluation.md)).
  - Strict epistemic boundary definitions (`cv_fact`, `visual_observation`, `inference`, `unverified`).
  - Deterministic keyframe selection algorithm ($K \le 5$ frame budget, duplicate suppression, event-driven triggers).
  - Model candidate evaluation (`Qwen3.5-2B`, `Qwen3-VL-2B-Instruct`, `SmolVLM2-2.2B-Instruct`) against physical hardware constraints (4 GB RTX 2050 Laptop GPU, 8 GB system RAM, CPU-only PyTorch).
  - Primary recommendation (`Qwen3-VL-2B-Instruct` via OpenAI-compatible endpoint) and local fallback (`SmolVLM2-2.2B-Instruct`).
  - Model-agnostic `VLMProvider` protocol and Pydantic v2 output contracts.
  - Evaluation plan across 5 curated UA-DETRAC benchmark sequences.

### Milestone 5B: Multimodal Evidence Pipeline & Keyframe Extraction
- **Status**: ✅ Completed (including Timestamp Integrity & Frame-Index Semantics Audits)
- **Focus**: Core deterministic keyframe extraction, timestamp integrity verification, OpenCV frame-index semantics audit, and structured evidence packaging.
- **Deliverables**:
  - Implemented `KeyframeExtractor` (`backend/app/services/vlm/keyframe_extractor.py`) with OpenCV seek, duplicate suppression ($\Delta t < 1.0\text{s}$ or identical frame index), aspect-ratio preserving downscaling ($960\text{px}$), and $K \le 5$ frame budget enforcement.
  - Eliminated fabricated FPS defaults (removed assumed 25.0 FPS); strictly prefers stream FPS, accepts caller metadata FPS, or sets `timestamp_s` to `None` with explicit diagnostic warnings.
  - Enforced strict provenance: prohibited associating unknown-timestamp frames with timestamped events using assumed FPS; verified seek correspondence via `CAP_PROP_POS_FRAMES`.
  - Audited OpenCV `CAP_PROP_POS_FRAMES` post-read semantics: correctly derived decoded frame index as $\text{int}(\text{round}(\text{pos\_after})) - 1$ to eliminate off-by-one errors and float truncation; explicitly distinguished `requested_frame_idx` and `is_verified`.
  - Handled unavailable (`-1.0`), unadvancing (`0.0`), and out-of-range decoder reports with explicit warnings and unverified flags.
  - Implemented `EvidencePackager` (`backend/app/services/vlm/evidence_packager.py`) linking `SequenceAnalyticsReport`, events, and keyframe images with clean null-timestamp preservation.
  - Defined Pydantic v2 schemas (`backend/app/schemas/vlm.py`) with strict public separation of binary payloads and elimination of local filesystem paths.
  - Comprehensive unit test suite (`backend/tests/test_vlm_evidence_pipeline.py`): 28 / 28 passed.
  - Full backend regression suite (97 passed) and CV regression suite (65 passed) with zero failures.

### Milestone 5C: Google Gemini VLM Provider Integration
- **Status**: ✅ Completed (including CV Fact Content Grounding, Rendering Safeguards & Semantic Predicate Audit)
- **Focus**: Native Google Gemini API `generateContent` REST integration, header-based authentication, structured schema enforcement, bounded retries, epistemic grounding verification, centralized verifier seam, verified CV fact rendering safeguard, semantic predicate audit, and failure isolation.
- **Deliverables**:
  - Implemented `GeminiVLMProvider` in `backend/app/services/vlm/gemini_provider.py` leveraging Google's native Gemini REST API (`https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`).
  - Native header-based authentication via `x-goog-api-key` (never exposed in query parameters, error payloads, or logs).
  - Multimodal request assembly in `PromptManager` combining epistemic instruction rules, structured CV telemetry, and interleaved base64 `inlineData` keyframes with metadata headers.
  - OpenAPI/JSON Schema dereferencing for `generationConfig.responseSchema` with uppercase type specifications matching Gemini's Protobuf schema requirements.
  - Robust response candidate validation, safety block classification (`promptFeedback.blockReason`, `finishReason=SAFETY`, `finishReason=RECITATION`), and `MAX_TOKENS` truncation detection.
  - Centralized Grounding Verifier (`backend/app/services/vlm/grounding_verifier.py`): unified epistemic verification engine shared with 100% parity across both `GeminiVLMProvider` and `OpenAICompatibleRemoteProvider`.
  - Epistemic grounding enforcement: every visual observation must reference a valid keyframe ID; ungrounded observations are rejected from `visual_observations` and disclosed in `unverified_aspects`; claims with missing/unknown evidence (visual, inference, or CV facts) are demoted to `unverified` with explicit diagnostics; valid evidence references are preserved intact.
  - Strict CV fact content grounding & provenance audit: introduced `TelemetryProvenanceSchema` verifying field paths and values within tolerance (`rel_tol=0.02, abs_tol=0.05`); unsupported CV domains (speed in km/h or mph, compass heading/direction, and collisions/crashes) cannot be inferred from valid track IDs and are strictly demoted to `unverified`; empty telemetry cannot confirm claims.
  - Final CV fact integrity enforcement: claim-content consistency checking (`check_claim_content_consistency`) preventing contradictory or fabricated free-text numbers; mandatory structured provenance for `CV_FACT`; safe field-path allowlists blocking private/internal attributes and cross-source leakage; strict differentiation between missing telemetry fields and genuine zero values; physical unit compatibility checking (`UNIT_ALLOWLIST`); broadened unsupported domain pattern detection.
  - Verified CV Fact Rendering Safeguard: deterministic rendering engine (`render_verified_cv_fact`) overwriting model-generated free-text claims with authoritative statements formatted directly from verified structured telemetry fields; canonical predicate allowlist (`SUPPORTED_PREDICATES`); nested dictionary subkey hardening and intermediate dictionary blocking.
  - Semantic Predicate & Metric Grounding Audit: enforced distinction between tracker identities (`total_unique_tracks` states `"unique tracker identities"` and rejects `"vehicles"` as a unit) and physical vehicle census; eliminated `vehicle_count` predicate; separated raw bounding box detections (`total_detections`) from tracker identities; restricted event duration to event sources; enforced singular vs. plural grammar accuracy (e.g., `1 second` vs `{val} seconds`, `1 identity` vs `{val} identities`).
  - Resilience & error handling: fast-failure for 401/403/404/400; bounded retries with exponential backoff for 429 rate limits, 5xx server errors, and timeouts.
  - Full diagnostic health check via `GET /v1beta/models/{model}` verifying connectivity, credentials, and model availability.
  - Strict failure isolation in `SceneInterpretationService`: VLM errors are caught and returned as structured failure descriptors without crashing CV jobs.
  - Environment-driven configuration in `backend/app/core/config.py` supporting `VLM_ENABLED`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_API_BASE_URL`, `VLM_REQUEST_TIMEOUT`, `VLM_MAX_RETRIES`, and secret redaction. Documented discovery curl command for `GET /v1beta/models` with current example `gemini-3.8-flash`.
  - Comprehensive unit test suite (`backend/tests/test_vlm_provider_integration.py`): 73 / 73 passed (including 32 dedicated adversarial content, integrity, rendering safeguard, and semantic audit tests) with mocked HTTP transport (zero live network calls or credential leaks); 235 / 235 full project regression tests passed.

### Milestone 5D: Backend Scene Understanding API & Job Orchestration
- **Status**: 🚀 Active (Architecture & Implementation Plan Approved)
- **Focus**: Decoupled service orchestration, dual-worker asynchronous architecture, persistence, and RESTful API exposure.
- **Deliverables**:
  - Architecture Decision Record ([ADR-0005](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/docs/adr/0005-vlm-job-orchestration.md)): Asynchronous Scene Understanding Orchestration and Decoupled Lifecycle Management.
  - Comprehensive technical design document ([docs/vlm-backend-orchestration-design.md](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/docs/vlm-backend-orchestration-design.md)).
  - Implementation specification ([.scratch/vlm-orchestration/spec.md](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/.scratch/vlm-orchestration/spec.md)) with declarative SQLAlchemy models, Alembic migration plan, Pydantic contracts, and state transition matrices.
  - Granular task breakdown in local issue tracker (`.scratch/vlm-orchestration/issues/`):
    - [Issue 01](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/.scratch/vlm-orchestration/issues/01-db-model-and-migration.md): Database Model & Alembic Migration (`scene_interpretations` table).
    - [Issue 02](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/.scratch/vlm-orchestration/issues/02-schemas-and-repository.md): Pydantic Schemas & `SceneInterpretationRepository`.
    - [Issue 03](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/.scratch/vlm-orchestration/issues/03-worker-and-queue.md): Background `InterpretationWorker` & Crash Recovery.
    - [Issue 04](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/.scratch/vlm-orchestration/issues/04-orchestrator-service.md): `InterpretationOrchestratorService` Execution Pipeline.
    - [Issue 05](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/.scratch/vlm-orchestration/issues/05-api-endpoints.md): FastAPI HTTP Routes & Router Registration.
    - [Issue 06](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/.scratch/vlm-orchestration/issues/06-cv-worker-trigger-hook.md): Optional CV Worker Trigger Hook.
    - [Issue 07](file:///c:/Users/bhard/Desktop/TrafficLens%20AI/.scratch/vlm-orchestration/issues/07-unit-and-integration-tests.md): Unit, Concurrency & Integration Test Suite.
  - Strict lifecycle independence: `ProcessingJob` reaches `COMPLETED` immediately upon finishing video analysis; VLM failures/timeouts never fail or invalidate primary CV jobs.
  - Dedicated persistent entity `SceneInterpretation` (1:N relationship with `ProcessingJob`, cascade on delete).
  - Immutable interpretation attempts: `SceneInterpretation` follows strictly unidirectional transitions (`QUEUED -> PROCESSING -> COMPLETED | FAILED`); both `COMPLETED` and `FAILED` are terminal. Retrying creates a new attempt row while preserving audit history.
  - Maximum attempt limit: `MAX_INTERPRETATION_ATTEMPTS = 3` total interpretation records per CV job (initial attempt plus retries).
  - In-process bounded background worker (`InterpretationWorker`) running in isolated daemon thread for single-process deployments (`uvicorn --workers 1`).
  - Concurrency safeguards: Process-level creation mutex (`_interpretation_create_lock`) and database partial unique index (`uix_active_interpretation_per_job` on `(job_id) WHERE status IN ('QUEUED', 'PROCESSING')`).
  - Single atomic commit: Persistence of all structured results, token accounting, and `COMPLETED` status occurs in one database transaction.
  - REST API contracts:
    - `POST /api/v1/jobs/{job_id}/interpretation` (enqueues scene interpretation; returns 202 Accepted, 409 Conflict if active or limit exhausted, 400 Bad Request, 503 Service Unavailable).
    - `GET /api/v1/jobs/{job_id}/interpretation` (fetches latest interpretation or virtual `status="NOT_REQUESTED"`).
    - `GET /api/v1/jobs/{job_id}/interpretations` (fetches full historical interpretation runs).
    - `POST /api/v1/jobs/{job_id}/interpretation/retry` (creates new attempt record; returns 202 Accepted, 409 Conflict if active or limit exhausted, 400 Bad Request).
  - Crash recovery routine (`recover_interpretations`) reconciling interrupted `PROCESSING` tasks on startup and redispatching queued requests; no automatic retries of ambiguous external API calls.
  - Enforced epistemic grounding verification (`GroundingVerifier`) and deterministic verified CV fact rendering prior to terminal `COMPLETED` state.

### Milestone 5E: Benchmark Evaluation & Frontend Multimodal Workspace
- **Status**: ⏳ Planned
- **Focus**: Verification and user-facing presentation.
- **Deliverables**:
  - Automated benchmark execution across 5 UA-DETRAC sequences, measuring schema validity, citation grounding rate, and latency.
  - Frontend Workspace integration: Scene Understanding card in `AnalysisWorkspacePage` displaying environmental context, event interpretations, and epistemic claims with click-to-seek keyframe interaction.
  - Final documentation and verification report.
