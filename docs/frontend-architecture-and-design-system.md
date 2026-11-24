# Frontend Architecture & Design System Specification

**Project**: TrafficLens AI  
**Document**: Architectural Blueprint & Design System  
**Status**: APPROVED / SPECIFICATION  
**Target Phase**: Phase 4 — Frontend Web Dashboard (Milestone 1)  

---

## 1. Executive Summary & Product Vision

**TrafficLens AI** is an intelligent traffic understanding and analytics platform. The frontend web interface serves as the primary visual and operational portal for transportation engineers, Traffic Operations Center (TOC) operators, and urban mobility researchers.

### 1.1 Computer-Vision First Principle
Unlike generic SaaS business dashboards—which typically present grids of disconnected KPI cards, abstract metrics, and decorative gradients—TrafficLens AI is fundamentally an **optical, visual-evidence-first product**.

- **Visual Grounding**: Every numerical telemetry point (vehicle count, flow rate in Vehicles Per Minute, spatial occupancy ratio, stationary event) maps directly to a physical spatial coordinate $(x, y)$, a persistent vehicle tracklet ID, an oriented virtual tripwire, or a polygonal Region of Interest (ROI) visible on the video camera view.
- **Evidence-Centricity**: The primary focus of user attention is the video viewport. Data visualizations, metric curves, and event logs do not compete with the video; they synchronize with it across time and space.
- **Explainability & Rigor**: Traffic classifications (e.g., congestion levels) are presented alongside their deterministic, rule-based reasoning criteria (e.g., "Occupancy: 12.9%, Stopped Vehicles: 0, Mean Speed: 42 px/s"), avoiding opaque or ungrounded AI claims.

---

## 2. Technology Stack Evaluation & Selection

| Layer | Selected Technology | Version | Rationale & Trade-offs |
| :--- | :--- | :---: | :--- |
| **Framework & Language** | **React** with **TypeScript** | `18.3+` / `5.5+` | Industry standard for complex, stateful analytical applications. Strict type-safety directly mirrored from backend Pydantic schemas. |
| **Build & Tooling** | **Vite** | `5.x+` | Extremely fast development server and Hot Module Replacement (HMR). Native ES modules, zero complex configuration, highly optimized production rollups. |
| **Styling & Theming** | **Tailwind CSS** + **Tailwind Merge** | `3.4+` | Zero-runtime CSS with declarative utility classes. Fully customizable design tokens (`tailwind.config.ts`) enforcing spacing scales, optical color palettes, and typographic hierarchy. |
| **Primitive Components** | **Radix UI Primitives** | `1.x` | Unstyled, fully accessible (WAI-ARIA compliant) headless components for dialogs, dropdowns, tabs, tooltips, and sliders, styled seamlessly with Tailwind. |
| **Iconography** | **Lucide React** | `0.4x+` | Clean, geometric, consistent 24x24 vector stroke icons designed for modern engineering and surveillance software. |
| **Data Visualization** | **Recharts** | `2.12+` | Declarative, composable SVG charting library. Excels at responsive time-series flow curves (VPM), ROI occupancy lines, and class distribution charts with synchronized crosshairs and tooltips. |
| **Video & Overlay Layer** | **HTML5 Video** + **HTML5 2D Canvas** | Native | Direct control over frame-stepping, custom playback rate, canvas bounding box synchronization, and ROI rendering without heavy, inflexible third-party video players. |
| **Server State & Polling** | **TanStack Query** (React Query) | `v5+` | Manages asynchronous server state, automatic background polling (`refetchInterval`) during `QUEUED` and `PROCESSING` job phases, caching, and retry policies. |
| **Client Workspace State**| **Zustand** | `4.5+` | Minimalist, boilerplate-free state store for playback timestamp synchronization, active track/ROI selection, and layer visibility toggles. |
| **Routing** | **React Router DOM** | `v6+` | Declarative client-side routing with nested layouts and dynamic URL parameters (`/workspace/:jobId`). |
| **Testing & Quality** | **Vitest** + **React Testing Library** + **Playwright** | Latest | Unit testing for components and utilities via Vitest; end-to-end integration and smoke testing via Playwright. |

---

## 3. Design System & Optical Token Hierarchy

To communicate that TrafficLens AI is a high-precision computer vision workstation, the interface employs an **"Obsidian Optical"** theme—a dark, high-contrast palette calibrated for video surveillance environments, with an accessible, high-clarity daylight theme.

### 3.1 Color Palette & Semantic Token Mappings

#### 3.1.1 Structural Neutrals
```typescript
export const neutralTokens = {
  canvas: {
    base: "#0B0F17",     // Deep obsidian root background
    subtle: "#111827",   // Card, panel, and sidebar background (gray-900)
    surface: "#1A2234",  // Elevated controls, dropdown menus, modals (gray-850)
    hover: "#243048",    // Hover state for interactive cards and rows
  },
  border: {
    subtle: "#1F2937",   // Subtle card borders, dividers (gray-800)
    default: "#374151",  // Interactive control borders, inputs (gray-700)
    focused: "#0EA5E9",  // Active focus ring (cyan-500)
  },
  text: {
    primary: "#F9FAFB",   // High-contrast headings and values (gray-50)
    secondary: "#9CA3AF", // Labels, metadata, axis titles (gray-400)
    muted: "#6B7280",     // Timestamps, disabled states (gray-500)
  }
};
```

#### 3.1.2 Vehicle Classification Palette
The currently evaluated computer vision pipeline operates with a unified `vehicle` class (mapping COCO categories `car`, `truck`, `bus`, `van`, `motorcycle` to a single ground-truth class). Detections, multi-object tracking, and spatiotemporal analytics evaluate all vehicles as a single unified category.

Consequently, the active, validated visualization token is the unified vehicle token:
```typescript
// Active validated visualization token
export const activeVehicleToken = {
  stroke: "#06B6D4",                  // Primary Optical Cyan
  fill: "rgba(6, 182, 212, 0.15)",   // Translucent Cyan footprint fill
  text: "#22D3EE",
  label: "Vehicle",
};

// Prospective future subtype tokens (unvalidated by current CV baseline):
// Documented solely as a future capability if a fine-tuned or evaluated
// multi-class vehicle classifier is integrated into the CV pipeline.
export const prospectiveSubtypeTokens: Record<string, { stroke: string; fill: string; text: string }> = {
  car: {
    stroke: "#0EA5E9",  // Sky Cyan
    fill: "rgba(14, 165, 233, 0.15)",
    text: "#38BDF8",
  },
  truck: {
    stroke: "#F59E0B",  // Industrial Amber
    fill: "rgba(245, 158, 11, 0.15)",
    text: "#FBBF24",
  },
  bus: {
    stroke: "#8B5CF6",  // Transit Violet
    fill: "rgba(139, 92, 246, 0.15)",
    text: "#A78BFA",
  },
  van: {
    stroke: "#14B8A6",  // Utility Teal
    fill: "rgba(20, 184, 166, 0.15)",
    text: "#2DD4BF",
  },
  motorcycle: {
    stroke: "#F43F5E",  // Rapid Crimson
    fill: "rgba(244, 63, 94, 0.15)",
    text: "#FB7185",
  },
};
```

#### 3.1.3 Traffic State & Heuristic Congestion Levels
Directly corresponding to backend `CongestionLevelEnum`.

> [!NOTE]
> Congestion levels (`low`, `moderate`, `high`) are classified strictly by the backend's rule-based heuristic logic (`backend/app/schemas/telemetry.py: CongestionLevelEnum`, `CongestionStateSchema`). The frontend displays the classification directly as received from the backend response (`congestion_distribution_per_roi` or `CongestionStateSchema.level` along with `reason`). The frontend must **never** duplicate, re-implement, or hardcode classification thresholds in client code.
> 
> Furthermore, these heuristic classifications are rule-based operational indicators designed for demonstration and proof-of-concept monitoring, **not** scientifically validated traffic engineering standards (e.g., Highway Capacity Manual Level of Service).

```typescript
export const congestionTokens = {
  low: {
    color: "#10B981",    // Emerald Green
    bg: "rgba(16, 185, 129, 0.12)",
    border: "#059669",
    label: "Low Congestion (Free Flow)",
  },
  moderate: {
    color: "#F59E0B",    // Amber Orange
    bg: "rgba(245, 158, 11, 0.12)",
    border: "#D97706",
    label: "Moderate Congestion (Dense)",
  },
  high: {
    color: "#EF4444",    // Crimson Red
    bg: "rgba(239, 68, 68, 0.15)",
    border: "#DC2626",
    label: "High Congestion (Bottleneck)",
  }
};
```

#### 3.1.4 Spatial Geometry Elements (Tripwires & ROIs)
```typescript
export const spatialGeometryTokens = {
  tripwire: {
    downstream: "#10B981", // Directional positive normal (Emerald)
    upstream: "#06B6D4",   // Directional negative normal (Cyan)
    strokeWidth: 2,
    dashPattern: [6, 4],
  },
  roi: {
    stroke: "#6366F1",     // Blueprint Indigo
    fill: "rgba(99, 102, 241, 0.10)",
    activeStroke: "#818CF8",
    activeFill: "rgba(99, 102, 241, 0.22)",
    strokeWidth: 2,
  },
  stoppedEvent: {
    marker: "#DC2626",     // Alert Red
    halo: "rgba(220, 38, 38, 0.35)",
    pulseDuration: "2s",
  }
};
```

### 3.2 Typography Scale
- **Primary Interface Font**: `Inter`, `-apple-system`, `BlinkMacSystemFont`, `Segoe UI`, `Roboto`, `sans-serif`.
- **Numerical & Telemetry Font**: `JetBrains Mono`, `Fira Code`, `ui-monospace`, `Consolas`, `monospace`.
- Frame numbers, timestamps, coordinates, confidence scores, and VPM throughput figures strictly use the monospace font to prevent tabular jitter during playback.

| Token | Size | Weight | Line Height | Usage |
| :--- | :---: | :---: | :---: | :--- |
| `text-xs` | 11px | Regular / Medium | 14px | Micro metadata, frame indices, coordinates, tags |
| `text-sm` | 13px | Regular / Medium | 18px | Telemetry labels, tabular cells, button text |
| `text-base` | 14px | Regular / Medium | 20px | Standard body text, form inputs, list items |
| `text-lg` | 16px | Semi-Bold | 24px | Section headers, modal titles, panel tabs |
| `text-xl` | 20px | Bold | 28px | KPI values, sequence titles, primary headings |
| `text-2xl` | 24px | Bold | 32px | Page titles, major dashboard headers |

### 3.3 Elevation, Borders & Radii
- **Radii**:
  - `rounded-sm`: 4px (tags, tooltips, frame badges)
  - `rounded-md`: 6px (buttons, input fields, control groups)
  - `rounded-lg`: 8px (cards, video viewport container, panels)
  - `rounded-xl`: 12px (modal dialogs, drawers)
- **Shadows**:
  - Low-profile ambient occlusion shadows (`shadow-sm`, `shadow-md` in `#000000 30%`), explicitly avoiding neon blurs or uncalibrated colored glow effects.

---

## 4. Frontend Directory Structure

```
frontend/
├── index.html
├── package.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
├── tailwind.config.ts
├── postcss.config.js
└── src/
    ├── main.tsx                         # App bootstrap and React root
    ├── App.tsx                          # Router provider & global layout
    ├── assets/                          # Static branding, placeholder icons
    ├── types/                           # TypeScript types matching backend schemas
    │   ├── api.ts                       # Health, API error, pagination types
    │   ├── video.ts                     # VideoMetadataSchema, VideoUploadResponse
    │   ├── job.ts                       # ProcessingJob, JobStatus, ProcessingJobCreate
    │   ├── telemetry.ts                 # LineFlow, ROIDensity, StoppedEvent, DetectionSummary
    │   └── common.ts                    # Point2D, BBox, CountingLineConfig, ROIConfig
    ├── core/                            # Configuration, constants, design tokens
    │   ├── config.ts                    # API base URL, timeout, polling intervals
    │   ├── tokens.ts                    # Exported design tokens (colors, classes, states)
    │   └── routes.ts                    # Route path constants
    ├── services/                        # Strongly-typed API client modules
    │   ├── apiClient.ts                 # Axios / Fetch client with interceptors
    │   ├── videoService.ts              # uploadVideo, getVideo, listVideos
    │   ├── jobService.ts                # createJob, getJob, getJobResults, getTelemetry
    │   └── healthService.ts             # getHealth
    ├── hooks/                           # Reusable React & TanStack Query hooks
    │   ├── useHealth.ts                 # Backend operational status polling
    │   ├── useVideo.ts                  # Fetch single video asset metadata
    │   ├── useJobStatus.ts              # Smart polling hook for in-progress jobs
    │   ├── useJobResults.ts             # Fetch completed job telemetry & artifacts
    │   ├── useVideoPlayback.ts          # Synchronized time, frame, play/pause controls
    │   └── useCanvasOverlay.ts          # 2D canvas drawing logic for bounding boxes & ROIs
    ├── state/                           # Zustand stores for workspace interaction
    │   ├── useWorkspaceStore.ts         # Active frame, selected vehicle, layer visibility
    │   └── useFilterStore.ts            # Video library filters, date ranges
    ├── components/                      # Reusable component library
    │   ├── common/                      # Generic atomic UI primitives
    │   │   ├── Button.tsx
    │   │   ├── Badge.tsx
    │   │   ├── Card.tsx
    │   │   ├── Input.tsx
    │   │   ├── Modal.tsx
    │   │   ├── Tabs.tsx
    │   │   └── Tooltip.tsx
    │   ├── feedback/                    # Status indicators, spinners, error alerts
    │   │   ├── StatusBadge.tsx          # QUEUED, PROCESSING, COMPLETED, FAILED badge
    │   │   ├── ProgressBar.tsx          # Upload / processing progress bar
    │   │   ├── LoadingSkeleton.tsx      # Skeleton loader for charts and video
    │   │   ├── EmptyState.tsx           # Empty video library / zero events view
    │   │   └── ErrorAlert.tsx           # Error callout with diagnostic details
    │   ├── layout/                      # Application shell components
    │   │   ├── AppHeader.tsx            # Global topbar (brand, navigation, health indicator)
    │   │   ├── AppSidebar.tsx           # Navigation drawer for routes
    │   │   └── PageContainer.tsx        # Standard padded page wrapper
    │   ├── video/                       # Core CV video player & overlay components
    │   │   ├── VideoPlayer.tsx          # HTML5 video element with custom HUD
    │   │   ├── CanvasOverlay.tsx        # Drawing layer for BBoxes, tripwires, ROIs
    │   │   ├── PlaybackControls.tsx     # Frame step, speed selector, time scrubber
    │   │   ├── LayerControls.tsx        # Toggles for bounding boxes, tracks, ROIs
    │   │   └── VideoUploader.tsx        # Drag-and-drop streamed multipart uploader
    │   └── analytics/                   # Data visualization & telemetry widgets
    │       ├── MetricCard.tsx           # Monospace KPI card with delta indicator
    │       ├── FlowChart.tsx            # Tumbling window Vehicles Per Minute line chart
    │       ├── OccupancyChart.tsx       # ROI spatial occupancy ratio area chart
    │       ├── ClassDistributionBar.tsx # Car, truck, bus, van breakdown bar
    │       ├── CongestionBadge.tsx      # Low/Moderate/High congestion status badge
    │       └── StoppedEventFeed.tsx     # Stationary vehicle event list with keyframe jump
    └── pages/                           # Application page views
        ├── OperationsOverviewPage.tsx   # Global traffic operations summary
        ├── VideoLibraryPage.tsx         # Video asset catalog, upload trigger
        ├── AnalysisWorkspacePage.tsx    # Flagship split-pane video + telemetry workstation
        ├── JobDetailsPage.tsx           # Execution audit trail, artifact downloads
        └── SettingsPage.tsx             # System configuration, API endpoint, homography
```

---

## 5. Application Routing & Page Architecture

```
                               ┌───────────────────┐
                               │    AppHeader      │
                               │  (System Status)  │
                               └─────────┬─────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│  /operations         │      │  /videos             │      │  /settings           │
│  Traffic Operations  │      │  Video Library       │      │  System Config &     │
│  Overview            │      │  & Ingestion         │      │  Calibration         │
└──────────────────────┘      └──────────┬───────────┘      └──────────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
              ┌──────────────────────┐        ┌──────────────────────┐
              │  /workspace/:jobId   │        │  /jobs/:jobId        │
              │  Video Analysis      │        │  Job Execution &     │
              │  Workstation         │        │  Artifacts Audit     │
              └──────────────────────┘        └──────────────────────┘
```

### 5.1 Route Specifications

| Path | Page Name | Primary Objective & Functionality |
| :--- | :--- | :--- |
| `/` or `/operations` | **Traffic Operations Overview** | High-level operations center: active backend health, recent video processing activity, overall vehicle detection count, fleet class distribution, and quick links to recent analyses. |
| `/videos` | **Video Library** | Video asset management: catalog of uploaded videos with dimensions, duration, FPS, and file size. Upload modal for streaming new MP4 videos. Action button to trigger processing job with default or custom tripwires. |
| `/workspace/:jobId` | **Video Analysis Workspace** | **The flagship workstation**: Split-screen layout. Left side: annotated video playback with interactive canvas overlays, frame-stepping, and timeline scrubbing. Right side: synchronized telemetry tabs (Overview, Line Flow, ROIs, Stationary Events). |
| `/jobs/:jobId` | **Processing Job Details** | Execution monitoring and technical audit trail: state transitions (`QUEUED` $\to$ `PROCESSING` $\to$ `COMPLETED`/`FAILED`), processing duration, pipeline execution breakdown, raw artifact file download links (`analytics.json`, `telemetry.csv`, `annotated.mp4`). |
| `/settings` | **Settings & Calibration** | API connection configuration, detection confidence floor threshold inspector, camera homography calibration status indicators (explicitly clarifying uncalibrated speed status). |

---

## 6. Flagship View: Video Analysis Workspace Layout

The **Video Analysis Workspace** (`/workspace/:jobId`) provides the primary analytical experience:

```
+------------------------------------------------------------------------------------------------------------------+
| [Header] TrafficLens AI  |  Seq: MVI_20011  |  Status: [COMPLETED]  |  Duration: 3.0s (75f @ 25fps)  | [Export ▼] |
+------------------------------------------------------------------------------------+-----------------------------+
|                                                                                    | TELEMETRY WORKSPACE (35%)   |
|  PRIMARY VIDEO EVIDENCE PLAYER (65%)                                               | [Overview] [Flow] [ROIs] [!] |
|  +------------------------------------------------------------------------------+  |-----------------------------|
|  | [Annotated MP4 Video Stream]                                                  |  | SUMMARY KPIS                |
|  | [Canvas Layer: Tripwires (line_mid), ROIs (roi_main_flow), Vehicle BBoxes]   |  | * Total Vehicles: 16 unique |
|  |                                                                              |  | * Total Detections: 2,716   |
|  |  Bounding Box Tooltip on Hover:                                              |  | * Line Crossings: 1         |
|  |  +-----------------------------------------------+                             |  | * Peak ROI Occ: 12.93%      |
|  |  | Vehicle #12 (Unified Vehicle)                 |                             |  | * Active Congestion: Low    |
|  |  | Conf: 0.91 | Speed: 38.4 px/s (Uncalibrated) |                             |  |-----------------------------|
|  |  +-----------------------------------------------+                             |  | TRIPWIRE FLOW RATE (VPM)    |
|  |                                                                              |  | Line: line_mid              |
+---------------------------------------------------------------------------------+  | [  Recharts Flow Windows   ]|
|  | Controls HUD:                                                                |  | 20 VPM |        _/\_         |
|  | [▶ Play] [⏮ -1f] [⏭ +1f]  00:01.80 / 00:03.00 (Frame: 45 / 75)   [1.0x ▼]   |  | 10 VPM |_______/    \________|
|  | ---------------------------------------------------------------------------- |  |        0.0s   1.5s   3.0s   |
|  | Timeline Scrubber with Event Marker Pins: [▲ Crossing @ 1.8s]                |  |-----------------------------|
|  | Overlays: [x] BBoxes  [x] Track IDs  [x] Tripwires  [x] ROIs  [ ] Velocities  |  | SPATIAL ROI DENSITY         |
|  +------------------------------------------------------------------------------+  | ROI: roi_main_flow          |
|                                                                                    | Occupancy Ratio: 12.93%     |
|  FRAME METRICS INSPECTOR (Collapsible Bottom Drawer)                               | Congestion: [Low - 100%]    |
|  Frame #45: Active Tracks: 14 | Instantaneous Occupancy: 11.2% | Crossing: Track #12 | Stopped Vehicles: 0         |
+------------------------------------------------------------------------------------+-----------------------------+
```

### 6.1 Telemetry Synchronization Principle
1. When the user plays or scrubs the video, the active timestamp $t$ and frame index $f$ are broadcast to the Zustand workspace store (`currentTime`, `currentFrame`).
2. The Recharts time-series cursor advances synchronously with the video scrubber.
3. Clicking on any event row (e.g., a tripwire crossing event or stopped vehicle alert) automatically seeks the video player to that event's exact timestamp (`start_timestamp_s`), highlighting the target vehicle track ID.

### 6.2 Analytics Semantics & Scientific Grounding

To preserve scientific integrity and avoid misleading claims, all analytical values rendered in the workspace adhere strictly to the following domain definitions:

1. **Clear Distinction of Spatial Density Metrics**:
   - **Spatial Occupancy Ratio (`occupancy_ratio`)**: The dimensionless spatial fraction $[0.0, 1.0]$ of an ROI polygon covered by estimated 2D vehicle bounding-box footprints (bounding box base). Displayed as a percentage (e.g., $12.93\%$).
   - **Tracked Vehicle Count (`vehicle_count`)**: The discrete integer count of active unique vehicle tracklets whose bottom-center reference coordinate lies within the ROI polygon at frame $t$.
   - **Physical Density**: True physical density (e.g., vehicles per lane-kilometer or PCU/km) is **not supported** by the current baseline because it requires metric camera homography calibration and lane geometry mapping. The UI must never label spatial occupancy ratio as "physical density" or "traffic density".

2. **Speed Telemetry Reporting**:
   - Speed is measured and rendered strictly in **pixels per second** (`px/s`).
   - Telemetry displays must visibly label speed as **Uncalibrated** (`is_calibrated: false`, `speed_kmh: null`) with a tooltip disclaimer: *"Uncalibrated pixel displacement rate. Metric speed (km/h) is unavailable without camera homography calibration."*

3. **Heuristic Congestion Classification**:
   - The frontend renders congestion states (`low`, `moderate`, `high`) strictly from backend responses (`congestion_distribution_per_roi` or `CongestionStateSchema.level` along with `reason`).
   - The frontend must **never duplicate, hardcode, or re-calculate congestion thresholds** in client JavaScript/TypeScript.
   - The UI includes an explicit technical disclaimer: *"Congestion states reflect rule-based operational heuristics (combining spatial occupancy ratio, vehicle count, and pixel displacement thresholds), not scientifically validated Highway Capacity Manual (HCM) Level of Service (LOS) standards."*

---

## 7. API Integration Strategy & TypeScript Data Contracts

The frontend integrates directly with the existing backend endpoints without intermediary mock servers or artificial payload wrapping.

### 7.1 Backend Endpoint Mapping Matrix

| Backend Route | HTTP Method | Request Payload | Response Schema | Client Hook |
| :--- | :---: | :--- | :--- | :--- |
| `/api/v1/health` | `GET` | *None* | `HealthCheckSchema` | `useHealth()` |
| `/api/v1/videos` | `POST` | `multipart/form-data` (`file`) | `VideoUploadResponseSchema` | `useUploadVideo()` |
| `/api/v1/videos/{video_id}` | `GET` | *None* | `VideoMetadataSchema` | `useVideo(videoId)` |
| `/api/v1/videos/{video_id}/jobs` | `POST` | `ProcessingJobCreateSchema` | `ProcessingJobResponseSchema` | `useCreateJob()` |
| `/api/v1/jobs` | `POST` | `ProcessingJobCreateSchema` | `ProcessingJobResponseSchema` | `useCreateJob()` |
| `/api/v1/jobs/{job_id}` | `GET` | *None* | `ProcessingJobResponseSchema` | `useJobStatus(jobId)` |
| `/api/v1/jobs/{job_id}/results`| `GET` | *None* | `JobResultsResponseSchema` | `useJobResults(jobId)` |
| `/api/v1/jobs/{job_id}/telemetry`| `GET`| *None* | `JobTelemetryResponseSchema` | `useJobTelemetry(jobId)` |

### 7.2 Core TypeScript Definitions (Grounded in Backend Schemas)

```typescript
// types/common.ts
export interface Point2D {
  x: number;
  y: number;
}

export interface BBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface CountingLineConfig {
  line_id: string;
  name: string;
  p1: [number, number];
  p2: [number, number];
  positive_direction_label?: string;
  negative_direction_label?: string;
  cooldown_frames?: number;
  min_track_age?: number;
}

export interface ROIConfig {
  roi_id: string;
  name: string;
  polygon: Array<[number, number]>;
  description?: string;
}

// types/job.ts
export type JobStatus = "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED";

export interface ProcessingError {
  error_code: string;
  message: string;
  details?: Record<string, any> | null;
  occurred_at: string;
}

export interface ProcessingJob {
  job_id: string;
  video_id: string;
  sequence_name: string;
  status: JobStatus;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  duration_seconds?: number | null;
  error?: ProcessingError | null;
}

// types/telemetry.ts
export interface LineFlow {
  line_id: string;
  window_start_s: number;
  window_end_s: number;
  total_count: number;
  flow_rate_vpm: number;
  direction_counts: Record<string, number>;
  direction_flows_vpm: Record<string, number>;
}

export interface StoppedVehicleEvent {
  track_id: number;
  start_frame: number;
  start_timestamp_s: number;
  end_frame?: number | null;
  end_timestamp_s?: number | null;
  duration_s: number;
  location: [number, number];
  roi_id?: string | null;
  status: "active" | "resolved";
}

export interface DetectionSummary {
  total_detections: number;
  mean_confidence: number;
  class_distribution: Record<string, number>;
}

export interface TrackingSummary {
  total_unique_tracks: number;
  mostly_tracked_count?: number | null;
  mostly_lost_count?: number | null;
  id_switches?: number | null;
  mota?: number | null;
  hota?: number | null;
  idf1?: number | null;
}

export interface TrafficAnalyticsSummary {
  sequence_name: string;
  total_frames: number;
  duration_s: number;
  fps: number;
  total_unique_vehicles: number;
  line_counts: Record<string, Record<string, number>>;
  flow_windows: LineFlow[];
  stopped_vehicle_events: StoppedVehicleEvent[];
  peak_occupancy_per_roi: Record<string, number>;
  average_occupancy_per_roi: Record<string, number>;
  congestion_distribution_per_roi: Record<string, Record<string, number>>;
}

export interface JobResults {
  job_id: string;
  video_id: string;
  sequence_name: string;
  status: string;
  duration_seconds?: number | null;
  detection_summary?: DetectionSummary | null;
  tracking_summary?: TrackingSummary | null;
  analytics_summary?: TrafficAnalyticsSummary | null;
  artifacts: {
    analytics_json?: string;
    telemetry_csv?: string;
    annotated_video?: string;
  };
}
```

### 7.3 Asynchronous Job Polling Hook Implementation Strategy

Using TanStack Query, the frontend handles asynchronous processing gracefully without manual intervals or memory leaks:

```typescript
// hooks/useJobStatus.ts
import { useQuery } from "@tanstack/react-query";
import { jobService } from "../services/jobService";
import { JobStatus, ProcessingJob } from "../types/job";

export function useJobStatus(jobId: string | undefined) {
  return useQuery<ProcessingJob, Error>({
    queryKey: ["job", jobId],
    queryFn: () => jobService.getJob(jobId!),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      // Fast polling (1.5s) while queued or processing; stop polling once terminal
      if (status === "QUEUED" || status === "PROCESSING") {
        return 1500;
      }
      return false;
    },
    refetchIntervalInBackground: true,
    staleTime: 1000,
  });
}
```

### 7.4 Dedicated Video and Artifact Serving Endpoints (Required Backend Dependency)

To support HTML5 video playback, frame scrubbing, and artifact inspection, the frontend architecture specifies dedicated backend endpoints as the preferred design:

1. **Source Video Content**: `GET /api/v1/videos/{video_id}/content`
   - Streams the raw uploaded video file for the specified `video_id`.
   - **HTTP Range Support**: Implements `Range: bytes=start-end` returning HTTP 206 Partial Content, enabling frame-accurate scrubbing in the HTML5 `<video>` player.
   - **Ownership & Path Containment**: Validates database `VideoAsset` record and strictly verifies path containment within `settings.UPLOAD_DIR` (`is_relative_to()`).

2. **Annotated Output Video**: `GET /api/v1/jobs/{job_id}/annotated-video`
   - Streams the processed, bounding-box annotated MP4 video (`{sequence_name}_annotated.mp4`).
   - **HTTP Range Support**: Full byte-range seeking capability.
   - **State Validation**: Confirms the target job exists and has reached terminal status `JobStatus.COMPLETED` before serving.
   - **Path Containment**: Verifies file existence strictly within `settings.OUTPUT_DIR / job_id`.

3. **Discrete Job Artifacts**: `GET /api/v1/jobs/{job_id}/artifacts/{artifact_type}`
   - Serves structured export files (`analytics_json` $\to$ `.json`, `telemetry_csv` $\to$ `.csv`).
   - **Security**: Strictly rejects directory traversal characters (`..`, `/`, `\`), enforces an enum whitelist (`analytics_json`, `telemetry_csv`), and sets appropriate `Content-Disposition` headers.

> [!CAUTION]
> **No Public Storage Directory Mounting**: The backend storage directory (`storage/`) must **never** be mounted as a public static directory. Public static file serving exposes raw server directory hierarchies, bypasses database ownership checks, and introduces security and path traversal vulnerabilities.
> 
> *Note*: These endpoints are **not** implemented in this milestone; they are recorded as a required backend integration dependency for the upcoming frontend milestone.

---

## 8. State Handling, Feedback & Error Diagnostics

The application design defines structured, unambiguous visual treatments for every state:

### 8.1 State Matrix

| State Type | Visual Representation | User Guidance / Actions |
| :--- | :--- | :--- |
| **Video Uploading** | Progress ring / linear bar with uploaded MB / total MB and percentage. | "Streaming video to server. Validating container readability..." |
| **Job QUEUED** | Subtle pulsing blue badge with queue indicator. | "Job submitted to processing queue. Waiting for background worker slot..." |
| **Job PROCESSING** | Active cyan spinner, elapsed wall-clock timer, frame animation. | "Running YOLO11n detection, ByteTrack tracking, and traffic telemetry..." |
| **Job COMPLETED** | Solid emerald badge, green success summary banner. | "Processing finished in X.Xs. Results and annotated video available." |
| **Job FAILED** | Crimson alert box with canonical error code and expandable diagnostics. | "Processing failed: [error_code]. Cause: [message]. [Retry Job]" |
| **Empty Library** | Clean illustration of traffic camera with dotted upload zone. | "No video assets ingested yet. Drag and drop an MP4 video or select a sample." |
| **Empty Events** | Calibrated green shield icon. | "No traffic events or stationary vehicle anomalies detected in this sequence." |

### 8.2 Error Diagnostics & HTTP Status Handling

| HTTP Code | Backend Trigger | UI Presentation & Action |
| :---: | :--- | :--- |
| **400 Bad Request** | Unreadable video container, corrupt frame 0, empty 0-byte file. | Red form warning: "Invalid video stream. File cannot be decoded by OpenCV. Please ensure valid H.264/MP4 encoding." |
| **409 Conflict** | Video already has an active `QUEUED` or `PROCESSING` job. | Warning banner: "Video already has an active processing job. View existing job status in Workspace." |
| **413 Content Too Large** | Video file exceeds configured `MAX_UPLOAD_SIZE_BYTES` (500 MB). | Error callout: "Upload exceeds maximum permitted size of 500 MB. Please compress or clip the video." |
| **415 Unsupported Media**| File extension is not in `.mp4`, `.avi`, `.mov`, `.mkv`. | Error callout: "Unsupported video format. Allowed formats: .mp4, .avi, .mov, .mkv." |
| **503 Service Unavailable**| Worker queue is full (`WORKER_MAX_QUEUE_SIZE = 100`). | Amber alert: "Processing queue is currently at maximum capacity. Please retry shortly." |

---

## 9. Responsive Layout Strategy & Breakpoints

TrafficLens AI operates as an engineering workstation. While desktop monitors ($1440\text{px}+$ and $1920\text{px}\times1080\text{px}$) represent the primary deployment target, the layout degrades gracefully:

| Breakpoint | Window Width | Layout Adaptations |
| :--- | :---: | :--- |
| **Ultra-Wide / Desktop** (`2xl`, `xl`) | $\ge 1280\text{px}$ | Full dual-pane workspace: 65% video evidence viewport + 35% sticky telemetry & event sidebar. Collapsible bottom frame inspector drawer. |
| **Laptop / Compact Desktop** (`lg`) | $1024\text{px} - 1279\text{px}$ | Stacked dual-pane: 55% video viewport + 45% sidebar. Telemetry tabs collapse to 2-column KPI grid. |
| **Tablet / Field Inspection** (`md`) | $768\text{px} - 1023\text{px}$ | Single-column stacked layout: Video player fixed on top; telemetry widgets and event log scroll beneath. Frame inspector accessible via modal sheet. |
| **Mobile / Alert View** (`sm`) | $< 768\text{px}$ | Compact view optimized for reviewing completed job statuses, summary KPIs, and event notifications. Interactive video canvas overlays hidden. |

---

## 10. Quality Assurance & Testing Strategy

```
                          ┌────────────────────────┐
                          │     Playwright E2E     │  (Full Upload, Polling, Results)
                          └───────────┬────────────┘
                                      │
                          ┌───────────┴────────────┐
                          │   Component Testing    │  (RTL: VideoPlayer, Charts, Badges)
                          └───────────┬────────────┘
                                      │
                          ┌───────────┴────────────┐
                          │    Unit & Logic Tests  │  (Vitest: Coordinate math, schemas)
                          └────────────────────────┘
```

1. **Unit Tests (Vitest)**:
   - Coordinate transformation utilities (pixel space to normalized viewport ratios).
   - Timestamp-to-frame calculation helpers.
   - Design token resolution and vehicle class color mapping.
2. **Component Integration Tests (React Testing Library)**:
   - `VideoPlayer`: Verifies frame-stepping calculations, play/pause controls, and HUD overlay formatting.
   - `StatusBadge`: Validates correct color and label for `QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`.
   - `FlowChart`: Validates rendering of tumbling window VPM curves.
   - `StoppedEventFeed`: Verifies click-to-seek callback firing.
3. **End-to-End Smoke Tests (Playwright)**:
   - Video upload workflow with synthetic MP4 $\to$ job submission $\to$ status polling simulation $\to$ results display.

---

## 11. Implementation Roadmap: Next Frontend Milestones

| Milestone | Phase | Scope & Deliverables |
| :--- | :---: | :--- |
| **Milestone 1** (Current) | **Architecture & Design System** | Technology stack selection, design tokens, component hierarchy, page layouts, API contracts, ADR-0003. |
| **Milestone 2** | **Project Scaffolding & Core Shell** | Setup `frontend/` directory with Vite + React + TS + Tailwind, Radix primitives, Lucide icons, AppHeader, AppSidebar, and routing skeleton. |
| **Milestone 3** | **Video Ingestion & Library** | Video upload dropzone with streamed multipart upload, video asset catalog table, job creation modal, and real-time status polling badges. |
| **Milestone 4** | **Video Analysis Workspace** | VideoEvidencePlayer with synchronized HTML5 Canvas, frame-stepping HUD, Recharts VPM flow curve, ROI occupancy cards, and stopped event inspector. |
| **Milestone 5** | **Polish, Testing & Verification** | Keyboard shortcuts, responsive testing across breakpoints, Vitest + Playwright test suite, final documentation. |

---

## 12. Development-Only Sample Sequence Strategy

To facilitate rapid, reproducible frontend development, UI prototyping, and visual testing without requiring repetitive manual video uploads:

1. **Target Benchmark Fixture**:
   - The verified 75-frame clip from UA-DETRAC sequence `MVI_20011` (960×540 resolution, 25.0 FPS, 3.0 seconds duration, 2,716 detections, 16 tracks).
2. **Repository Path**:
   - Relative path: `data/samples/MVI_20011_sample.mp4` (or `experiments/fixtures/`).
   - Hardcoded machine-specific absolute paths are strictly prohibited across all code and fixtures.
3. **Strict Development Gating**:
   - The sample loader is strictly gated under development mode:
     ```typescript
     if (import.meta.env.DEV) {
       // Render "[Load Development Sample: MVI_20011]" developer action
     }
     ```
4. **Isolation from Production Workflows**:
   - The developer sample action submits the sample clip through the standard client service to exercise standard ingestion pipelines.
   - The sample button and fixture logic are completely omitted from production builds, keeping standard end-user video upload workflows clean and production-ready.
