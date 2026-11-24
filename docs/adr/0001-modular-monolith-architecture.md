# Modular Monolith Architecture and Core Decisions for TrafficLens-AI

## Context

TrafficLens-AI is an end-to-end traffic understanding and analytics system integrating computer vision (vehicle detection, multi-object tracking), spatiotemporal analytics, multimodal LLMs, and agentic workflows. As an applied ML engineering project, the architecture must support rapid experimentation, maintainable modular boundaries, and straightforward deployment without incurring distributed systems overhead or premature abstractions.

## Decision

We adopt a **Modular Monolith** architecture with a unified Python backend (FastAPI), a React frontend, and a strictly isolated experimentation directory (`experiments/`). The specific architectural choices are:

1. **Modular Monolith Shape**:
   The backend internal modules (`pipeline`, `api`, `agents`, `db`, `core`) communicate via clear Python module boundaries and typed Pydantic interfaces. ML research code and training scripts reside strictly in `experiments/` and are never imported by the production backend.

2. **Video Ingestion & Processing**:
   Prioritize asynchronous, file-based batch video processing first to ensure deterministic, reproducible benchmarking against UA-DETRAC sequences and clean telemetry generation. The processing pipeline interface is designed with a clean frame ingestion seam to remain extensible for future live RTSP/WebRTC streaming.

3. **In-Process Inference Runtime**:
   Vehicle detection and tracking run in-process inside the FastAPI backend. PyTorch inference is the initial baseline; ONNX Runtime will be evaluated only if benchmarking demonstrates a meaningful latency or throughput advantage. No separate model serving container (such as Triton or TorchServe) will be introduced.

4. **Persistence Layer**:
   Use SQLAlchemy with SQLite for zero-dependency local development, maintaining connection string and dialect compatibility with PostgreSQL for seamless future migration. We deliberately avoid custom database abstraction layers beyond standard SQLAlchemy ORM models.

5. **Phased Agent & VLM Boundary**:
   The CV pipeline owns all numerical analysis, tracking, and deterministic event detection. The LangGraph agent serves as an analytical query layer that inspects structured telemetry via tools and accesses selected keyframes via a VLM for qualitative incident interpretation. Agent workflows will not be implemented until the underlying CV and analytics pipeline is fully functional.

6. **Dataset & Evaluation Strategy**:
   Local experimentation begins with a curated UA-DETRAC subset of 5–10 representative video sequences covering varied illuminations, weather conditions, and traffic densities. Original annotations, object identities, ignore regions, and metadata are strictly preserved with reproducible train/val/test splits. Model performance is evaluated against dataset ground truth using mAP (detection) and MOTA, IDF1, and HOTA (tracking). All initial target figures are provisional benchmarks, and off-the-shelf baselines are established before any fine-tuning.

## Considered Options

1. **Microservices (Separate detector, tracker, analytics, and agent services)**:
   - Rejected: Prohibitive operational overhead, unnecessary network serialization latency over raw video frames, and excessive complexity for a student ML engineering project.
2. **Dedicated Model Serving Service (Triton / TorchServe)**:
   - Rejected: Adds infrastructure complexity without measurable benefit over optimized in-process PyTorch/ONNX inference for current project scope.
3. **Pure Script / Monolithic Notebook Structure**:
   - Rejected: Couples research artifacts with production serving, leading to brittle dependency chains and difficult testing.

## Consequences

- Direct in-process execution eliminates inter-process communication latency during video analysis.
- The separation between `experiments/` and `backend/` prevents experimental dependencies (e.g., Jupyter, matplotlib) from leaking into production builds.
- Keeping SQLite/PostgreSQL configuration unified through SQLAlchemy enables local development out of the box with zero database daemon setup.
- Phasing agent development after the CV pipeline ensures the agent has verified, real telemetry and keyframes to act upon.
