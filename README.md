# TrafficLens-AI 🚦🔍

> Intelligent traffic understanding and analytics platform powered by Computer Vision, Multi-Object Tracking, Multimodal Vision-Language Models, and Agentic AI.

---

## 📌 Project Overview

**TrafficLens-AI** transforms traffic camera video streams into actionable traffic intelligence. The platform performs vehicle detection and multi-object tracking, computes critical traffic analytics (flow rates, density, vehicle classification, dwell times), detects traffic events and anomalies, and provides an agentic natural-language query interface powered by LangGraph and multimodal vision-language models.

### Core Capabilities
- 🚗 **Vehicle Detection & Classification**: High-accuracy detection of cars, buses, vans, and trucks using modern detectors evaluated on the UA-DETRAC benchmark.
- 🎯 **Multi-Object Tracking (MOT)**: Persistent vehicle trajectory tracking with occlusion handling using ByteTrack.
- 📊 **Spatial Traffic Analytics**: Automated vehicle counting across directional tripwires, queue density calculation in polygonal Regions of Interest (ROIs), and speed estimation.
- ⚠️ **Event Detection**: Rule-based detection of traffic irregularities including congestion bottlenecks, stationary vehicles, and illegal movements.
- 🤖 **Agentic & Multimodal Intelligence**: Natural language querying and automated incident analysis using LangGraph workflows and Vision-Language Model (VLM) keyframe inspection (phased after CV pipeline is operational).
- 💻 **Interactive Dashboard**: Modern React dashboard providing video playback with telemetry overlays, analytical charts, event feeds, and an AI chat assistant.

---

## 🏗️ Architecture

TrafficLens-AI follows a **Modular Monolith** architecture designed for maintainability, high performance, and rapid engineering iteration without unnecessary microservices.

```
TrafficLens-AI/
├── docs/                         # Architecture Decision Records & Agent Guidelines
│   ├── adr/                      # ADR-0001 Modular Monolith & Core Decisions
│   └── agents/                   # Agent operational configs
├── data/                         # Local dataset & artifact storage (gitignored)
│   ├── raw/                      # Raw video / UA-DETRAC dataset archives
│   ├── processed/                # Curated subset frames, extracted labels
│   └── annotations/              # Evaluation splits, ignore regions & ground truth
├── experiments/                  # Isolated ML experimentation workspace
│   ├── notebooks/                # Exploratory Data Analysis & prototyping
│   ├── configs/                  # Training & evaluation hyperparameters (YAML)
│   ├── src/                      # UA-DETRAC parser, baseline benchmarks, eval scripts
│   └── checkpoints/              # Model weights & training logs (gitignored)
├── backend/                      # Production FastAPI Backend Application
│   ├── app/
│   │   ├── api/v1/               # REST endpoints & WebSocket handlers
│   │   ├── core/                 # Configuration, logging, settings
│   │   ├── db/                   # Database models & sessions (SQLAlchemy: SQLite/PostgreSQL)
│   │   ├── pipeline/             # In-process CV inference engine (PyTorch baseline)
│   │   │   ├── detector.py       # Object detection engine wrapper
│   │   │   ├── tracker.py        # Multi-object tracking (ByteTrack)
│   │   │   ├── analytics.py      # Tripwires, ROIs, density, velocity
│   │   │   └── events.py         # Rule-based event & anomaly detection
│   │   ├── agents/               # LangGraph workflows & VLM tools (Phase 5)
│   │   ├── schemas/              # Pydantic request/response schemas
│   │   └── main.py               # FastAPI application entrypoint
│   ├── tests/                    # Unit, integration, and pipeline tests
│   ├── Dockerfile                # Backend container definition
│   └── requirements.txt          # Production dependencies
├── frontend/                     # React + Vite + Tailwind CSS Dashboard
│   ├── src/
│   │   ├── components/           # VideoPlayer, MetricsCard, EventLog, ChatPanel
│   │   ├── pages/                # Dashboard, Analytics, VideoUpload, AIQuery
│   │   ├── services/             # API client services
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
├── docker-compose.yml            # Multi-service local orchestration
├── GLOSSARY.md                   # Canonical domain vocabulary
└── AGENTS.md                     # Agent skill instructions & repository rules
```

---

## 🔬 Dataset & Evaluation Strategy

Model development and benchmarking center on the **UA-DETRAC** benchmark:
- **Curated Subset**: A representative benchmark subset of 5–10 sequences covering diverse conditions (day, night, sunny, rainy, heavy traffic) for fast, reproducible iterations.
- **Annotation Integrity**: Original XML annotations, vehicle identities, speed attributes, and evaluation ignore regions are rigorously preserved.
- **Baseline First**: Off-the-shelf pretrained models are benchmarked as a baseline prior to any fine-tuning.
- **Evaluation Metrics**:
  - **Detection**: mAP@0.5, mAP@0.5:0.95 evaluated strictly against ground truth with ignore regions excluded.
  - **Tracking**: MOTA (Multi-Object Tracking Accuracy), IDF1, and HOTA (Higher Order Tracking Accuracy) alongside ID switches (IDSW) and fragmentation.
  - *Note*: Initial target metrics serve as provisional benchmarks, not rigid acceptance criteria.

---

## 🗺️ Development Roadmap

| Phase | Milestone | Deliverables & Acceptance Criteria | Status |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **Architecture & Scaffolding** | Repository layout, ADR-0001, GLOSSARY.md, and roadmap finalized. | ✅ Completed |
| **Phase 1** | **Dataset Curation & CV Baseline** | UA-DETRAC XML parser with ignore-region handling, 8-sequence curated split, PyTorch YOLO11n detection baseline (AP@0.50: 93.77%, F1: 92.44%), ByteTrack MOT baseline (HOTA: 68.46%, MOTA: 81.37%, IDF1: 87.06%). | ✅ Completed |
| **Phase 2** | **Core Backend & Ingestion** | FastAPI async video processing endpoints, OpenCV frame reader, in-process PyTorch pipeline runner, SQLAlchemy models with SQLite persistence, bounded job queues. | ✅ Completed |
| **Phase 3** | **Event Engine & Analytics** | Rule-based event detection (stationary vehicles, congestion onset, directional line crossings, ROI occupancy), keyframe snapshot extraction, telemetry exports. | ✅ Completed |
| **Phase 4** | **Frontend Web Dashboard** | Modern React + Vite + Tailwind dashboard, dual video player (source & annotated), real-time synchronized telemetry inspection, and ROI visualization. | ✅ Completed |
| **Phase 5** | **Multimodal VLM & Incident Agent** | Gemini 2.5 Flash VLM scene understanding with epistemic claim grounding, and LangGraph-based evidence-grounded traffic incident reporting agent. | ✅ Completed |
| **Phase 7** | **Frontend AI Integration** | Interactive Scene Understanding panel, evidence-grounded Incident Report interface, epistemic confidence badges, timestamp-based video seeking, and historical audit drawers. | ✅ Completed |

---

## 🚀 Running the Project Locally

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Node.js 18+ and npm
- (Optional) Google Gemini API Key for live multimodal VLM scene interpretation

### 2. Backend Setup
```bash
# Clone the repository
git clone <repo-url>
cd "TrafficLens AI"

# Install Python dependencies
pip install -r backend/requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env to set your GEMINI_API_KEY if testing live VLM inference

# Run database migrations
alembic upgrade head

# Start FastAPI backend server (http://127.0.0.1:8000)
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Frontend Setup
```bash
# In a new terminal, navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite development server (http://127.0.0.1:3000)
npm run dev
```

### 4. Running Automated Tests
```bash
# Backend pytest suite (382 tests)
pytest

# Frontend vitest suite (79 tests)
npm --prefix frontend test

# Production build check
npm --prefix frontend run build
```

---

## ⚠️ Known Limitations & Epistemic Boundaries

1. **Camera Calibration & Speed Measurement**:
   - The platform strictly prohibits displaying speeds in `km/h` or `mph` without camera homography matrix calibration. Speeds are reported as uncalibrated pixel displacement (`px/s`).
2. **Tracker Instance Semantics**:
   - Vehicle counts reflect tracked identities (`Track #ID`). Under severe occlusions or camera jitter, tracklet fragmentation may lead to higher track counts than unique physical vehicles.
3. **Congestion Heuristics**:
   - Spatial occupancy levels are rule-based 2D bounding-box coverage heuristics, labeled with `EXPLAINABLE HEURISTIC`, and should not be treated as certified highway capacity determinations.
4. **Advisory Decision Support**:
   - All incident report recommendations are strictly advisory and must never be connected directly to automated signal actuators without human-in-the-loop validation.
5. **Hardware & Throughput**:
   - In-process CV detection and tracking achieves ~8.6 FPS on modern CPUs. Real-time multi-stream deployment requires CUDA-enabled GPU acceleration.
6. **Live Gemini Inference**:
   - Requires setting `GEMINI_API_KEY` in `.env` and enabling `VLM_ENABLED=true`. When absent, the incident agent seamlessly operates using deterministic offline mock providers for testing.

---

## 📊 Pretrained YOLO11n Detection Baseline Results

Evaluated on the held-out test sequences (`MVI_20011` [night] & `MVI_40131` [rainy]) at optimal validation-tuned threshold ($conf=0.25$, $IoU=0.50$):

| Metric | Benchmark Score | Description |
| :--- | :--- | :--- |
| **AP@0.50** | **93.77%** | Mean Average Precision across vehicle detections (101-point COCO interpolation) |
| **AP@0.75** | **76.51%** | Strict localization threshold |
| **AP@0.50:0.95** | **64.19%** | 10-step mean Average Precision |
| **Precision@0.50** | **93.05%** | Accuracy of positive detections |
| **Recall@0.50** | **91.84%** | Ground truth coverage |
| **F1-Score@0.50** | **92.44%** | Harmonic balance between Precision and Recall |
| **Throughput** | **~8.3 FPS (121.1 ms)** | Pure model inference on CPU (batch size 16, $640 \times 640$) |

---

## 🎯 ByteTrack Multi-Object Tracking Baseline Results

Evaluated on held-out test sequences (`MVI_20011` & `MVI_40131`) using frozen validation-tuned parameters ($track\_high=0.40$, $track\_low=0.10$, $match\_thresh=0.70$, $track\_buffer=45$) via official **TrackEval** and **motmetrics**:

| Metric | Benchmark Score | Description |
| :--- | :--- | :--- |
| **HOTA** | **68.46%** | Higher Order Tracking Accuracy (unified detection & association) |
| **DetA** | **67.55%** | Detection Accuracy (mean over 19 localization thresholds) |
| **AssA** | **69.51%** | Association Accuracy (trajectory consistency over time) |
| **MOTA** | **81.37%** | Multiple Object Tracking Accuracy (CLEAR metric) |
| **IDF1** | **87.06%** | Global identity preservation |
| **ID Switches (IDSW)** | **12** | Total identity swaps across 2,309 test frames |
| **Mostly Tracked (MT)** | **93** | Vehicles tracked for $\ge 80\%$ of their trajectory |
| **Mostly Lost (ML)** | **1** | Vehicles tracked for $\le 20\%$ of their trajectory |
| **End-to-End Latency** | **116.38 ms** | Combined detection (113.37 ms) + ByteTrack update (3.01 ms) |
| **End-to-End Throughput** | **8.6 FPS** | Real-time tracking pipeline on CPU |


---

## 📜 Domain Language

All terminology adheres strictly to [GLOSSARY.md](./GLOSSARY.md). Please consult the glossary before introducing new entity definitions or domain concepts.

## ⚖️ Architecture Decisions

All non-trivial architectural decisions are recorded in [docs/adr/](./docs/adr/).
- [ADR-0001: Modular Monolith Architecture and Core Decisions](./docs/adr/0001-modular-monolith-architecture.md)
