# ADR 0003: Frontend Architecture, Design System, and Video Analysis Workspace

## Context

TrafficLens-AI transforms raw traffic video streams into structured spatiotemporal intelligence using YOLO11n vehicle detection, ByteTrack multi-object tracking, directional tripwires, polygonal ROIs, and rule-based congestion and stopped-vehicle classification.

In Phase 4, we are establishing the frontend web interface for TrafficLens-AI.

The frontend must satisfy critical requirements:
1. **Computer-Vision First User Experience**: The product is an optical traffic analytics workstation for traffic engineers and operations center personnel, not a generic SaaS metrics dashboard. The central visual element must be the video evidence stream, with data visualizations and event feeds synchronized with video playback.
2. **Deterministic, Grounded Telemetry Display**: Every chart, table, and metric must map directly to physical pixel coordinates, persistent vehicle tracklet IDs, counting lines, or spatial ROIs.
3. **Rigorous Typing & Backend Contract Alignment**: The frontend data layer must interface directly with the FastAPI backend endpoints (`/api/v1/videos`, `/api/v1/jobs`, `/api/v1/jobs/{job_id}/results`) and schemas without ungrounded mock abstractions.
4. **Predictable Asynchronous Polling**: Long-running video processing (`QUEUED` $\to$ `PROCESSING` $\to$ `COMPLETED` / `FAILED`) must be represented clearly with resilient polling, error diagnostic inspection, and zero memory leaks.
5. **Anti-Generic Design System**: Eliminate generic dashboard clichés (excessive neon gradients, glowing borders, fuzzy glassmorphism, decorative cards) in favor of high-contrast, optical clarity optimized for surveillance and traffic control environments.

## Decision

We adopt the following frontend architecture and visual standards:

### 1. Technology Foundation
- **Core Framework**: React 18+ with TypeScript (strict mode enabled).
- **Tooling & Build System**: Vite for rapid ES-module Hot Module Replacement and production bundling.
- **Styling & Theming**: Tailwind CSS with custom theme extensions defined in `tailwind.config.ts`.
- **UI Primitives**: Radix UI headless accessible components (dialogs, dropdowns, tabs, tooltips, sliders).
- **Iconography**: Lucide React for consistent technical vector iconography.
- **Data Visualization**: Recharts for composable, responsive SVG time-series charts (tumbling window Vehicles Per Minute, ROI occupancy ratio curves, class distribution breakdowns).
- **Video Controller & Canvas Layer**: Native HTML5 `<video>` element coordinated with an overlay HTML5 2D `<canvas>` layer for interactive bounding box hover, tripwire rendering, and ROI footprint drawing.
- **State & Server Synchronization**: TanStack Query (React Query v5) for server state caching and polling; Zustand for local workspace playback synchronization.

### 2. Design System Tokens: "Obsidian Optical"
- **Neutrals**: Deep obsidian canvas (`#0B0F17`), panel surfaces (`#111827`), subtle borders (`#1F2937`), and high-contrast text (`#F9FAFB`).
- **Unified Vehicle Token**: The evaluated CV pipeline operates with a unified `vehicle` class. Detections and tracks use a unified optical token (`stroke: "#06B6D4"`, `fill: "rgba(6, 182, 212, 0.15)"`, `text: "#22D3EE"`). Subtype tokens (`car`, `truck`, `bus`, `van`, `motorcycle`) are documented solely as a prospective future capability if multi-class classification is evaluated.
- **Congestion Palette**: Low (`#10B981` Emerald), Moderate (`#F59E0B` Amber), High (`#EF4444` Red). Displayed directly from backend responses; the frontend never duplicates threshold calculations. Explicitly noted as rule-based heuristics, not scientific traffic engineering standards.
- **Spatial Geometry**:
  - Tripwires: Downstream (`#10B981` Emerald), Upstream (`#06B6D4` Cyan).
  - Regions of Interest (ROI): Blueprint Indigo (`#6366F1`, 10% fill).
  - Stopped Vehicle Event: Alert Red (`#DC2626`).
- **Telemetry Semantics**:
  - Explicitly separates **spatial occupancy ratio** (polygon area fraction covered by vehicle footprints $[0.0, 1.0]$) from **tracked vehicle count** (discrete integer count). Physical density (e.g. vehicles/km) is explicitly unavailable without metric homography.
  - Speeds displayed strictly in `px/s` and labeled as uncalibrated (`speed_kmh: null`).
- **Typography**: Dual-font architecture: `Inter` for general UI text; `JetBrains Mono` / monospace for frame indices, timestamps, coordinates, and telemetry counts to eliminate layout jitter.

### 3. Application Routing & Page Structure
1. `/operations`: High-level traffic operations overview (health, recent activity, system status).
2. `/videos`: Video library and ingestion portal (streamed multipart upload, metadata inspection, job dispatch).
3. `/workspace/:jobId`: **Flagship Video Analysis Workspace** featuring a 65/35 dual-pane split:
   - Primary video evidence player with frame-stepping (`-1f`, `+1f`), speed adjustment, timeline event markers, and interactive canvas overlays.
   - Synchronized telemetry tabs (Summary KPIs, Tumbling Window Flow Rate, Spatial ROI Density, Stopped-Vehicle Event Feed).
4. `/jobs/:jobId`: Technical execution audit trail and raw artifact download links (`analytics.json`, `telemetry.csv`, `annotated.mp4`).
5. `/settings`: Backend API connection configuration, detection thresholds, and homography calibration status indicators.

### 4. Direct API Contract Grounding & Dedicated Media Serving
- The frontend client strictly uses the endpoints and JSON schemas established in ADR-0002 and `docs/backend-api-contracts.md`.
- **Upload Terminology**: The upload interface uses **streamed multipart uploads** (`multipart/form-data`) streamed to disk in 1 MB chunks with a 500 MB limit, not resumable chunked uploads.
- **Dedicated Video & Artifact Endpoints**: Video streaming and artifact downloads require dedicated backend endpoints rather than public storage mounting:
  - `GET /api/v1/videos/{video_id}/content` (source video streaming with HTTP 206 Range support).
  - `GET /api/v1/jobs/{job_id}/annotated-video` (annotated video streaming with HTTP 206 Range support, completed state verification).
  - `GET /api/v1/jobs/{job_id}/artifacts/{artifact_type}` (whitelisted discrete artifact downloads with path traversal protection).
  - *Public static mounting of `storage/` is strictly rejected to preserve path traversal protection and database asset authorization.*
- **Development Sample Strategy**: Gated under `import.meta.env.DEV`, utilizing the verified 75-frame `MVI_20011` sequence via relative repository paths (`data/samples/MVI_20011_sample.mp4`), completely isolated from production builds and standard user uploads.

## Considered Options

1. **Next.js (App Router / Fullstack SSR)**:
   - *Rejected*: The backend is already established in FastAPI (Python) as a modular monolith. Introducing Next.js would duplicate server-side logic and add unnecessary Node.js runtime hosting overhead for a student ML engineering project.
2. **Heavyweight Video Player Framework (Video.js / Plyr)**:
   - *Rejected*: Heavy third-party players introduce skinning bloat and complicate frame-accurate HTML5 canvas overlay synchronization. Native HTML5 `<video>` provides full programmatic control with zero external overhead.
3. **Canvas-Only Charting (Chart.js / Visx)**:
   - *Rejected*: Recharts offers composable, responsive React SVG components with built-in accessibility and tooltips, requiring significantly less boilerplate for standard traffic time-series telemetry.

## Consequences

- The frontend operates as a clean, decoupled Single Page Application (SPA) communicating over standard HTTP REST endpoints with the FastAPI backend.
- The dual-pane workspace architecture ensures that visual video evidence remains front-and-center, directly reinforcing the computer vision domain.
- Synchronous time-scrubbing between the video player and telemetry charts delivers an intuitive, interactive analytical experience.
- Strict design tokens prevent visual inconsistency and ensure professional accessibility across all display resolutions.
