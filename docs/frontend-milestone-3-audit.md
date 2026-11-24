# TrafficLens AI — Frontend Milestone 3 Audit & Verification Report
**Video Ingestion, Library & Job Orchestration**

---

## 1. Executive Summary

Frontend Milestone 3 establishes the first authentic end-to-end operational pipeline in TrafficLens AI:

$$\text{Upload Video Stream} \longrightarrow \text{Persist Asset} \longrightarrow \text{Configure Processing} \longrightarrow \text{Dispatch Job} \longrightarrow \text{Track State} \longrightarrow \text{Live Telemetry Workspace}$$

This audit thoroughly validates implementation correctness, telemetry provenance, local artifact security boundaries, development fixture isolation, and failure-mode handling across both automated API integration and genuine browser end-to-end automation.

---

## 2. Telemetry Provenance Audit

Every displayed workspace metric was audited directly against the computer vision pipeline (`backend/app/services/cv_pipeline_adapter.py`), the Pydantic schemas (`backend/app/schemas/telemetry.py`), and the structured results endpoint (`GET /api/v1/jobs/{job_id}/results`).

### 2.1 Workspace Metric Provenance Matrix

| Displayed UI Metric | Exact Backend Field / Source | Computation Mechanism in CV Engine | Status / Live Reliability |
| :--- | :--- | :--- | :--- |
| **Tracked Vehicle Identities**<br>*(Unique Track IDs)* | `analytics_summary.total_unique_vehicles` | `len(analytics_pipeline._unique_track_ids)` assigned by ByteTrack across all frames. | **Genuinely Computed (Tracker Trajectories, Not Physical Census)**: Grounded in ByteTrack track ID assignments (e.g. 16 trajectories). Clarified in UI and documentation that track fragmentation, occlusions, and re-identifications can produce multiple IDs for one physical vehicle; does not represent a guaranteed physical vehicle census. |
| **Total Detections** | `detection_summary.total_detections` | Running sum of raw vehicle bounding boxes output by YOLO11n (`len(predictions)`). | **Genuinely Computed**: Verified live per sequence (e.g. 2,716 detections). |
| **Mean Confidence** | `detection_summary.mean_confidence` | `float(np.mean(conf_scores))` from all YOLO11n prediction confidences. | **Genuinely Computed**: Verified live per sequence (e.g. 78.4%). |
| **Line Crossings** | `analytics_summary.line_counts` | Vector crossing test ($p_{\text{prev}} \times p_{\text{curr}}$ normal dot-product) with directional classification. | **Genuinely Computed**: Summed across configured virtual tripwires. |
| **Peak Occupancy** | `analytics_summary.peak_occupancy_per_roi` | Maximum 2D bounding box footprint area ratio over ROI polygon area observed across frames. | **Genuinely Computed**: Grounded in geometric footprint overlap (e.g. 10.3%). |
| **Average Occupancy** | `analytics_summary.average_occupancy_per_roi` | Time-averaged spatial occupancy ratio per ROI across sequence duration. | **Genuinely Computed**: Grounded in geometric footprint overlap. |
| **Flow Rates (VPM)** | `analytics_summary.flow_windows` | Tumbling window throughput: $\frac{\text{count}}{\Delta t} \times 60$ per line and direction. | **Genuinely Computed**: Grounded in time-window aggregations. |
| **Stopped Vehicles** | `analytics_summary.stopped_vehicle_events` | Stationary centroid displacement detector with duration and frame span. | **Genuinely Computed**: Grounded in spatial tracking history. |
| **Speed Telemetry** | `SpeedMeasurementSchema.speed_px_per_s` | Inter-frame pixel centroid Euclidean displacement rate (`px/s`). | **Caveat Preserved**: Explicitly labeled **uncalibrated** (`px/s`); metric speed in km/h requires camera homography calibration. |
| **Heuristic Congestion** | `congestion_distribution_per_roi` | Rule-based classification (`low`, `moderate`, `high`) based on occupancy ratio and stopped vehicle count. | **Caveat Preserved**: Labeled heuristic/rule-based, matching backend classifier. |
| **MOTA, HOTA, IDF1** | `tracking_summary.mota/hota/idf1` | Requires ground-truth annotation files (offline TrackEval / MOTMetrics evaluation). | **Corrected**: Removed from live job telemetry. Added explicit notice explaining that benchmark metrics belong to offline evaluation and are undefined for live unannotated inference. |

---

## 3. Artifact Handling & Security Boundary Audit

### 3.1 Upload Filename & Path Traversal Protections
- **UUID Renaming**: Uploaded files are never written using user-supplied filenames. The backend generates a random identifier (`vid_{uuid.uuid4().hex[:12]}{ext}`) and verifies `storage_path.is_relative_to(settings.UPLOAD_DIR.resolve())`.
- **Directory Traversal Prevention**: File writes escaping `storage/uploads/` raise HTTP 400 Bad Request. Empty files (0 bytes) and oversized streams (> 500 MB) are automatically purged from disk before registration.
- **No Static Mount**: Verified `backend/app/main.py`. The backend **does not mount** `storage/` or `uploads/` via FastAPI `StaticFiles`. The local filesystem is completely private and unexposed.

### 3.2 Artifact Presentation & Future Serving Boundaries
- **No Local Filesystem Path Exposure**: The frontend completely avoids rendering absolute or relative host filesystem paths (`storage\outputs\...`). Server paths are retained internally as database records, while the UI presents only clean metadata:
  - **Filename**: Clean extracted basename (e.g. `analytics.json`, `telemetry.csv`, `annotated.mp4`).
  - **Artifact Type**: Domain description (`Structured JSON`, `Tabular CSV`, `Rendered Video MP4`).
  - **Availability Status**: Clear badge indicating whether the artifact is queryable (`Available via API`) or persisted pending future endpoints (`Persisted (No download endpoint)`, `Persisted (No streaming endpoint)`).
- **No Premature Claims**: The UI does not claim or imply that video streaming or CSV downloading is currently available. Media download/stream actions are disabled until dedicated serving routes exist.
- **REST Telemetry Action**: The results action button is accurately labeled `View Results JSON`, invoking the verified `GET /api/v1/jobs/{job_id}/results` endpoint.
- **No False Claims of Authentication**: Endpoint access is not described as authenticated, accurately reflecting that user authentication has not yet been introduced into TrafficLens AI.
- **Policy Notice**: `JobDetailsPage.tsx` displays an informational banner explaining backend storage retention and noting that dedicated serving endpoints will be introduced in Milestone 4.

---

## 4. Development Fixture Isolation Audit

### 4.1 Production Bundle Cleanliness
- **Elimination of Public Duplication**: The copied file `frontend/public/samples/MVI_20011_sample.mp4` has been completely deleted.
- **Zero Production Bloat**: Verified `dist/` after `npm run build`. The production distribution contains **zero video files** and no `samples/` directory (total bundle size: index.html 0.94 kB, CSS 41.78 kB, JS 395.31 kB).

### 4.2 Canonical Development Serving Mechanism
- **Vite Middleware**: Added `devSampleFixturePlugin` in `frontend/vite.config.ts` with `apply: 'serve'`. This middleware only activates during `npm run dev` and streams directly from `data/samples/MVI_20011_sample.mp4`.
- **Real Ingestion Pipeline**: In development (`APP_CONFIG.isDev`), clicking "Load Sample Clip" loads the clip into a native `File` object and submits it through the identical `validateVideoFile` $\rightarrow$ `apiClient.uploadWithProgress` $\rightarrow$ `POST /api/v1/videos` pipeline as any user file. It creates genuine database records and authentic IDs.

---

## 5. Distinction: API Integration vs. Genuine Browser E2E Testing

To ensure complete technical honesty, two distinct test levels are maintained and verified:

```
+-------------------------------------------------------------------------------+
|                        TESTING TAXONOMY & VERIFICATION                        |
+-------------------------------------------------------------------------------+
|  1. Backend API Integration Smoke Test (scratch/e2e_smoke_test.py)            |
|     - Uses Python requests to hit http://127.0.0.1:8000/api/v1 directly.       |
|     - Verifies HTTP contracts, status codes (201, 202, 200), and serialization.|
|     - Does NOT execute JavaScript, render React components, or test DOM state.|
+-------------------------------------------------------------------------------+
|  2. Genuine Browser End-to-End Workflow (scratch/browser_e2e_workflow.py)     |
|     - Uses Playwright to drive Google Chrome (headless=True, channel="chrome").|
|     - Navigates http://127.0.0.1:3000, clicks buttons, types in inputs.       |
|     - Validates React state, TanStack Query cache invalidations, and DOM HUD.  |
|     - Captures real visual screenshots of UI components during execution.     |
+-------------------------------------------------------------------------------+
```

### 5.1 Browser E2E Test Execution Log (Playwright + Chrome)
1. **Navigated to `/videos`**: Verified header and table catalog.
2. **Opened `VideoUploader` Modal**: Clicked `Load Sample Clip` $\rightarrow$ Validated `MVI_20011_sample.mp4` (2.86 MB).
3. **Streamed Upload**: Clicked `Upload Video Stream` $\rightarrow$ Observed progress $\rightarrow$ Confirmed OpenCV metadata extraction $\rightarrow$ Persisted `vid_a09fa74ab6c9`.
4. **Library Update**: Closed modal $\rightarrow$ Confirmed newly persisted `vid_a09fa74ab6c9` dynamically appeared in the table.
5. **Configured Job**: Clicked `Process Video` $\rightarrow$ Verified `ProcessingConfigModal` with midline tripwire and polygonal ROI.
6. **Dispatched Job**: Clicked `Submit Processing Job` $\rightarrow$ Created `job_2d5b32c46880`.
7. **Navigated to Workspace**: Automated URL transition to `/workspace/job_2d5b32c46880`.
8. **Monitored Polling**: Observed live status transitions (`QUEUED` $\rightarrow$ `PROCESSING` $\rightarrow$ `COMPLETED`).
9. **Validated HUD Telemetry**: Verified **16 Tracked Vehicle Identities** (Unique Track IDs), **2,716 Total Detections**, and grounded tracking execution note.
10. **Navigated to Job Details**: Clicked `View Audit Trail` $\rightarrow$ Verified execution timeline and artifact security notice.

---

## 6. Failure Handling & Query Cache Verification

### 6.1 Failure Modes Verified
- **Unsupported File Extensions**: Client-side rejection in `validateVideoFile` and backend HTTP 415 in `videos.py`.
- **Zero-Byte File**: Immediate rejection in `validateVideoFile` ("Uploaded file is empty (0 bytes)").
- **Oversized Video (> 500 MB)**: Client-side validation prevents upload; backend streams up to 500 MB and raises HTTP 413 with partial file cleanup.
- **Network Upload Interruption**: `xhr.onerror` catches network drops and surfaces an actionable error banner.
- **HTTP 409 Active Job Conflict**: Handled in `ProcessingConfigModal.tsx` by regex-extracting the active `job_id` and rendering a direct link to the running job.
- **HTTP 503 Worker Queue Full**: Handled in `ProcessingConfigModal.tsx` with guidance to wait for active jobs to complete.
- **Non-Existent Job ID (404)**: `AnalysisWorkspacePage` and `JobDetailsPage` render dedicated error cards with retry and library navigation options.
- **Job Failure (FAILED State)**: State machine halts polling immediately; renders structured error code and message.
- **Results Endpoint Unavailable**: Added fallback error card with "Retry Loading Results" action in `AnalysisWorkspacePage`.
- **Page Refresh During Processing**: URL `/workspace/:jobId` persists state; TanStack Query resumes polling until terminal state.

### 6.2 Polling & Query Cache Behavior
- `useJobStatus.ts`: Configured with `refetchIntervalInBackground: false`. Polling is strictly bounded to `QUEUED` and `PROCESSING` ($\sim 1.5\text{s}$) and permanently halts on `COMPLETED` or `FAILED`.
- Query invalidation: Creating a job in `ProcessingConfigModal` invalidates `["job", jobId]` and `["videos"]`. Ingestion in `VideoLibraryPage` invalidates `["videos"]`.
- Results query: Strictly gated by `enabled: Boolean(isRealJob && isCompleted)`. Results are never queried while a job is running.

---

## 7. Final Verification Results

```
================================================================================
                               VERIFICATION SUITE
================================================================================
[PASS] TypeScript Strict Compilation (tsc -b): Zero errors, zero warnings
[PASS] Production Bundle Build (vite build): 2,212 modules transformed in 1.48s
[PASS] Bundle Isolation: Zero video files in frontend/dist/
[PASS] Frontend Unit Tests (vitest): 16 / 16 passed in 4.19s
       - validation.test.ts (7 tests)
       - jobStatus.test.ts (3 tests)
       - failureHandling.test.ts (6 tests)
[PASS] Full Backend Regression Suite (pytest): 111 / 111 passed in 36.85s
[PASS] Genuine Browser End-to-End Test: 10 / 10 stages passed in Google Chrome
[PASS] API Integration Smoke Test: All 6 stages passed
================================================================================
```

---

## 8. Remaining Limitations & Milestone 4 Boundaries

In strict compliance with project governance:
1. **Video Playback & Canvas Overlays**: Canvas-based bounding box rendering, track trail scrubbing, and frame-seeking are scheduled for **Frontend Milestone 4**.
2. **Dedicated Media Serving Endpoints**: HTTP Range-enabled video streaming and dedicated artifact download endpoints will be introduced in Milestone 4 (access controls will be defined when user authentication is introduced).
3. **Camera Homography Calibration**: Metric vehicle speed (km/h) remains uncalibrated; pixel displacement (`px/s`) is reported.
4. **VLM & Agentic Incident Reports**: Reserved for later milestones.
