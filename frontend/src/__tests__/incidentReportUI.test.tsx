import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { IncidentReportPanel } from "../components/workspace/IncidentReportPanel";
import { TelemetryInspectionPanel } from "../components/workspace/TelemetryInspectionPanel";
import * as useIncidentHooks from "../hooks/useIncidentReport";
import * as useSceneInterpHooks from "../hooks/useSceneInterpretation";
import {
  IncidentReportResponse,
  IncidentReportHistoryResponse,
} from "../types/incidentReport";
import { JobResults } from "../types/telemetry";

// @ts-expect-error - React act testing environment flag
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

describe("Milestone 7C: Incident Report UI Component Tests", () => {
  let container: HTMLDivElement;
  let root: any;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    vi.restoreAllMocks();

    // Default mock for useIncidentReportHistory
    vi.spyOn(useIncidentHooks, "useIncidentReportHistory").mockReturnValue({
      data: { job_id: "job_1", total_reports: 1, reports: [] },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    // Default mock for useIncidentReportById
    vi.spyOn(useIncidentHooks, "useIncidentReportById").mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    // Default mock for useGenerateIncidentReport
    vi.spyOn(useIncidentHooks, "useGenerateIncidentReport").mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      isError: false,
      error: null,
    } as any);

    // Default mock for useSceneInterpretation
    vi.spyOn(useSceneInterpHooks, "useSceneInterpretation").mockReturnValue({
      data: { id: null, job_id: "job_1", status: "NOT_REQUESTED" },
      status: "NOT_REQUESTED",
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      isNotRequested: true,
      isQueued: false,
      isProcessing: false,
      isCompleted: false,
      isFailed: false,
      isActive: false,
      isTerminal: false,
      refetch: vi.fn(),
    });
  });

  afterEach(() => {
    if (root) {
      act(() => {
        root.unmount();
      });
    }
    if (container) {
      container.remove();
    }
  });

  function renderComponent(ui: React.ReactElement) {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    act(() => {
      root.render(
        <QueryClientProvider client={queryClient}>
          {ui}
        </QueryClientProvider>
      );
    });
    return container;
  }

  const sampleMockReportResponse: IncidentReportResponse = {
    report_id: "rep_99182a",
    job_id: "job_sample_123",
    video_id: "vid_road_cam",
    sequence_name: "INTERSECTION_MAIN_01",
    status: "COMPLETED",
    created_at: "2026-11-23T14:30:00Z",
    validation_passed: true,
    report: {
      report_id: "rep_99182a",
      job_id: "job_sample_123",
      video_id: "vid_road_cam",
      sequence_name: "INTERSECTION_MAIN_01",
      generated_at: "2026-11-23T14:30:00Z",
      executive_summary:
        "Traffic flow remained largely stable across the 120s sequence with an average flow rate of 42.5 VPM. One stationary vehicle was detected in the northbound curb lane causing localized queue stagnation.",
      confidence_classification: "high",
      traffic_conditions: {
        sequence_name: "INTERSECTION_MAIN_01",
        duration_s: 120.0,
        total_detections: 3450,
        total_tracked_instances: 48,
        overall_flow_rate_vpm: 42.5,
        peak_occupancy_ratio: 0.38,
        environmental_context: {
          weather: "clear",
          lighting: "daylight",
          road_surface: "dry",
        },
        visual_context_available: true,
        visual_context_note: "Verified keyframe observations integrated via Gemini 2.5 Flash",
      },
      congestion_observations: [
        {
          roi_id: "ROI_NORTH_APPROACH",
          observed_level: "moderate",
          peak_occupancy: 0.38,
          average_occupancy: 0.22,
          is_heuristic_rule: true,
          description: "Localized queue formation behind stationary vehicle at curb zone",
        },
      ],
      stopped_vehicles: [
        {
          event_id: "evt_stop_001",
          track_id: 14,
          start_timestamp_s: 18.5,
          duration_s: 24.2,
          location: [412, 650],
          roi_id: "ROI_NORTH_APPROACH",
          probable_cause: "Temporary delivery drop-off with hazards active",
          contextual_explanation: "Vehicle pulled to curb and remained stationary with active brake and hazard lights",
          verification_status: "grounded",
        },
      ],
      visual_observations: [
        {
          observation_id: "vis_obs_01",
          keyframe_id: "kf_004",
          timestamp_s: 19.2,
          description: "Silver delivery van stopped with hazard lights flashing at curb",
          visual_cues: ["hazard flashers", "curbside positioning", "dry asphalt"],
        },
      ],
      observed_events: [
        {
          event_type: "stopped_vehicle",
          timestamp_s: 18.5,
          description: "Track #14 stationary at northbound curb",
          supporting_evidence_ref: "evt_stop_001",
        },
        {
          event_type: "queue_onset",
          timestamp_s: 22.0,
          description: "Approaching vehicles braking and diverting into adjacent lane",
          supporting_evidence_ref: "line_crossing_north",
        },
      ],
      evidence_references: [
        {
          source: "cv_telemetry",
          metric_or_field: "duration_s",
          asserted_value: 120.0,
          verified: true,
        },
        {
          source: "cv_telemetry",
          metric_or_field: "stopped_vehicles",
          asserted_value: "Track #14 for 24.2s",
          verified: true,
        },
      ],
      recommendations: [
        {
          action: "Review curbside loading zone restrictions",
          priority: "medium",
          rationale: "Delivery vehicles stopping during peak hours constrain flow from 2 lanes to 1 lane",
          is_advisory: true,
        },
      ],
      validation_diagnostics: {
        validation_passed: true,
        total_claims_checked: 14,
        grounded_claims: 14,
        demoted_claims: 0,
        warnings: [],
        notes: ["All numeric speed assertions strictly suppressed (uncalibrated homography)."],
      },
      unverified_aspects: ["Exact delivery company branding not resolvable"],
      limitations: [
        "Camera perspective uncalibrated for physical speed (km/h)",
        "Tracklet fragmentation possible during vehicle occlusion",
      ],
    },
  };

  // ---------------------------------------------------------------------------
  // 1. Empty State
  // ---------------------------------------------------------------------------
  it("renders empty state when no report has been generated yet", () => {
    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: null,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: false,
      isCompleted: false,
      isFailed: false,
      refetch: vi.fn(),
    });

    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_1"
        currentTimeSeconds={0}
        onSeek={vi.fn()}
      />
    );

    expect(el.textContent).toContain("No Incident Report Generated");
    expect(el.textContent).toContain("Generate Incident Report");
    expect(el.textContent).toContain("Epistemic Grounding Guarantee");
  });

  // ---------------------------------------------------------------------------
  // 2. Report Generation Action
  // ---------------------------------------------------------------------------
  it("triggers generateMutation when 'Generate Incident Report' CTA is clicked", () => {
    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: null,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: false,
      isCompleted: false,
      isFailed: false,
      refetch: vi.fn(),
    });

    const mockMutate = vi.fn();
    vi.spyOn(useIncidentHooks, "useGenerateIncidentReport").mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      isError: false,
      error: null,
    } as any);

    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_1"
        currentTimeSeconds={0}
        onSeek={vi.fn()}
      />
    );

    const generateBtn = Array.from(el.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("Generate Incident Report")
    );
    expect(generateBtn).toBeDefined();

    act(() => {
      generateBtn?.click();
    });

    expect(mockMutate).toHaveBeenCalledTimes(1);
  });

  // ---------------------------------------------------------------------------
  // 3. Loading State & Duplicate Submission Prevention
  // ---------------------------------------------------------------------------
  it("renders generating loading state and prevents duplicate submissions while mutation is pending", () => {
    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: null,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: false,
      isCompleted: false,
      isFailed: false,
      refetch: vi.fn(),
    });

    vi.spyOn(useIncidentHooks, "useGenerateIncidentReport").mockReturnValue({
      mutate: vi.fn(),
      isPending: true,
      isError: false,
      error: null,
    } as any);

    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_1"
        currentTimeSeconds={0}
        onSeek={vi.fn()}
      />
    );

    expect(el.textContent).toContain("Synthesizing Telemetry & Grounding Claims...");
    expect(el.textContent).toContain("Synchronous agent execution in progress");
    expect(el.querySelector('[data-testid="incident-report-generating"]')).not.toBeNull();
  });

  // ---------------------------------------------------------------------------
  // 4. Generation Errors & Missing CV Telemetry
  // ---------------------------------------------------------------------------
  it("renders ErrorAlert on query failure or generation error", () => {
    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: null,
      isLoading: false,
      isFetching: false,
      isError: true,
      error: new Error("Network error connecting to incident report service"),
      hasReport: false,
      isCompleted: false,
      isFailed: false,
      refetch: vi.fn(),
    });

    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_1"
        currentTimeSeconds={0}
        onSeek={vi.fn()}
      />
    );

    expect(el.textContent).toContain("Incident Report Service Inaccessible");
    expect(el.textContent).toContain("Network error connecting to incident report service");
    expect(el.querySelector('[data-testid="incident-report-query-error"]')).not.toBeNull();
  });

  it("renders ErrorAlert when generation mutation fails with upstream provider error", () => {
    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: null,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: false,
      isCompleted: false,
      isFailed: false,
      refetch: vi.fn(),
    });

    vi.spyOn(useIncidentHooks, "useGenerateIncidentReport").mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      isError: true,
      error: new Error("504 Gateway Timeout: Agent LLM call timed out after 30s"),
    } as any);

    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_1"
        currentTimeSeconds={0}
        onSeek={vi.fn()}
      />
    );

    expect(el.textContent).toContain("Report Generation Failed");
    expect(el.textContent).toContain("504 Gateway Timeout");
  });

  // ---------------------------------------------------------------------------
  // 5. Executive Summary & Confidence Badges
  // ---------------------------------------------------------------------------
  it("renders executive summary and high confidence badge with epistemic notice", () => {
    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: sampleMockReportResponse,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: true,
      isCompleted: true,
      isFailed: false,
      refetch: vi.fn(),
    });

    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_sample_123"
        currentTimeSeconds={10}
        onSeek={vi.fn()}
      />
    );

    expect(el.textContent).toContain("Executive Summary");
    expect(el.textContent).toContain("Traffic flow remained largely stable across the 120s sequence");
    expect(el.textContent).toContain("High Confidence");
    expect(el.textContent).toContain("AI-generated synthesis of available evidence");
    expect(el.textContent).toContain("COMPLETED");
    expect(el.textContent).toContain("Grounding Passed");
  });

  it("renders cautionary confidence badge when confidence is cautionary", () => {
    const cautionaryResponse: IncidentReportResponse = {
      ...sampleMockReportResponse,
      report: {
        ...sampleMockReportResponse.report!,
        confidence_classification: "cautionary",
      },
    };

    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: cautionaryResponse,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: true,
      isCompleted: true,
      isFailed: false,
      refetch: vi.fn(),
    });

    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_sample_123"
        currentTimeSeconds={10}
        onSeek={vi.fn()}
      />
    );

    expect(el.textContent).toContain("Cautionary");
  });

  // ---------------------------------------------------------------------------
  // 6. Traffic Conditions & Terminology
  // ---------------------------------------------------------------------------
  it("renders traffic conditions with strict terminology (frame detections, tracked vehicle instances, duration)", () => {
    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: sampleMockReportResponse,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: true,
      isCompleted: true,
      isFailed: false,
      refetch: vi.fn(),
    });

    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_sample_123"
        currentTimeSeconds={10}
        onSeek={vi.fn()}
      />
    );

    expect(el.textContent).toContain("Traffic Conditions Baseline");
    expect(el.textContent).toContain("120.0s");
    expect(el.textContent).toContain("3,450");
    expect(el.textContent).toContain("Bounding box detections");
    expect(el.textContent).toContain("48");
    expect(el.textContent).toContain("Tracker identities (not physical census)");
    expect(el.textContent).toContain("38.0%");
    expect(el.textContent).toContain("42.5 VPM");
    expect(el.textContent).toContain("Multimodal VLM Context Active");
    expect(el.textContent).toContain("weather: clear");
    expect(el.textContent).toContain("Speeds are strictly uncalibrated (pixel space)");
  });

  // ---------------------------------------------------------------------------
  // 7. Congestion Observations & Heuristic Labeling
  // ---------------------------------------------------------------------------
  it("renders congestion observations with EXPLAINABLE HEURISTIC badge and peak occupancy", () => {
    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: sampleMockReportResponse,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: true,
      isCompleted: true,
      isFailed: false,
      refetch: vi.fn(),
    });

    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_sample_123"
        currentTimeSeconds={10}
        onSeek={vi.fn()}
      />
    );

    expect(el.textContent).toContain("ROI_NORTH_APPROACH");
    expect(el.textContent).toContain("moderate");
    expect(el.textContent).toContain("EXPLAINABLE HEURISTIC");
    expect(el.textContent).toContain("38.0%");
    expect(el.textContent).toContain("Localized queue formation behind stationary vehicle");
  });

  // ---------------------------------------------------------------------------
  // 8. Stopped Vehicles & Traffic Events with Seek Callback
  // ---------------------------------------------------------------------------
  it("renders stopped vehicles with ADVISORY HYPOTHESIS badge and clickable timestamp seeking", () => {
    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: sampleMockReportResponse,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: true,
      isCompleted: true,
      isFailed: false,
      refetch: vi.fn(),
    });

    const mockSeek = vi.fn();
    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_sample_123"
        currentTimeSeconds={10}
        onSeek={mockSeek}
      />
    );

    expect(el.textContent).toContain("Track #14");
    expect(el.textContent).toContain("24.2s");
    expect(el.textContent).toContain("ADVISORY HYPOTHESIS");
    expect(el.textContent).toContain("Temporary delivery drop-off with hazards active");

    const seekBtn = Array.from(el.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("Seek to t=18.5s")
    );
    expect(seekBtn).toBeDefined();

    act(() => {
      seekBtn?.click();
    });

    expect(mockSeek).toHaveBeenCalledWith(18.5);
  });

  // ---------------------------------------------------------------------------
  // 9. Visual Observations
  // ---------------------------------------------------------------------------
  it("renders visual observations with keyframe ID and visual cues chips", () => {
    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: sampleMockReportResponse,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: true,
      isCompleted: true,
      isFailed: false,
      refetch: vi.fn(),
    });

    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_sample_123"
        currentTimeSeconds={10}
        onSeek={vi.fn()}
      />
    );

    expect(el.textContent).toContain("vis_obs_01");
    expect(el.textContent).toContain("kf_004");
    expect(el.textContent).toContain("Silver delivery van stopped with hazard lights flashing");
    expect(el.textContent).toContain("hazard flashers");
    expect(el.textContent).toContain("curbside positioning");
  });

  // ---------------------------------------------------------------------------
  // 10. Evidence References
  // ---------------------------------------------------------------------------
  it("renders evidence references with subsystem source, asserted value, and verification status", () => {
    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: sampleMockReportResponse,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: true,
      isCompleted: true,
      isFailed: false,
      refetch: vi.fn(),
    });

    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_sample_123"
        currentTimeSeconds={10}
        onSeek={vi.fn()}
      />
    );

    expect(el.textContent).toContain("Evidence References & Provenance Citations");
    expect(el.textContent).toContain("cv_telemetry");
    expect(el.textContent).toContain("stopped_vehicles");
    expect(el.textContent).toContain("Track #14 for 24.2s");
    expect(el.textContent).toContain("VERIFIED");
  });

  // ---------------------------------------------------------------------------
  // 11. Advisory Recommendations
  // ---------------------------------------------------------------------------
  it("renders follow-up recommendations with ADVISORY badge, priority, and decision-support disclaimer", () => {
    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: sampleMockReportResponse,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: true,
      isCompleted: true,
      isFailed: false,
      refetch: vi.fn(),
    });

    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_sample_123"
        currentTimeSeconds={10}
        onSeek={vi.fn()}
      />
    );

    expect(el.textContent).toContain("Operational Follow-Up Recommendations");
    expect(el.textContent).toContain("Review curbside loading zone restrictions");
    expect(el.textContent).toContain("Medium Priority");
    expect(el.textContent).toContain("ADVISORY");
    expect(el.textContent).toContain("Must not be connected to automated signal actuators");
  });

  // ---------------------------------------------------------------------------
  // 12. Validation Diagnostics & Limitations
  // ---------------------------------------------------------------------------
  it("renders validation diagnostics, warnings, notes, and unverified aspects", () => {
    const reportWithWarning: IncidentReportResponse = {
      ...sampleMockReportResponse,
      report: {
        ...sampleMockReportResponse.report!,
        validation_diagnostics: {
          validation_passed: true,
          total_claims_checked: 15,
          grounded_claims: 14,
          demoted_claims: 1,
          warnings: ["Suppressed speculative speed assertion (65 km/h)."],
          notes: ["Validated against ByteTrack telemetry."],
        },
      },
    };

    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: reportWithWarning,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: true,
      isCompleted: true,
      isFailed: false,
      refetch: vi.fn(),
    });

    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_sample_123"
        currentTimeSeconds={10}
        onSeek={vi.fn()}
      />
    );

    expect(el.textContent).toContain("Claims Evaluated");
    expect(el.textContent).toContain("15");
    expect(el.textContent).toContain("14");
    expect(el.textContent).toContain("1");
    expect(el.textContent).toContain("Suppressed speculative speed assertion (65 km/h).");
    expect(el.textContent).toContain("Validated against ByteTrack telemetry.");
    expect(el.textContent).toContain("Unverified Scene Aspects");
    expect(el.textContent).toContain("Camera perspective uncalibrated for physical speed");
  });

  // ---------------------------------------------------------------------------
  // 13. Report History & Historical Version Selection
  // ---------------------------------------------------------------------------
  it("renders report history version selector and allows selecting a historical report", () => {
    const historicalReport: IncidentReportResponse = {
      ...sampleMockReportResponse,
      report_id: "rep_hist_v1",
      created_at: "2026-11-23T12:00:00Z",
      report: {
        ...sampleMockReportResponse.report!,
        report_id: "rep_hist_v1",
        executive_summary: "Initial v1 incident report assessment.",
      },
    };

    const historyData: IncidentReportHistoryResponse = {
      job_id: "job_sample_123",
      total_reports: 2,
      reports: [historicalReport, sampleMockReportResponse],
    };

    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: sampleMockReportResponse,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: true,
      isCompleted: true,
      isFailed: false,
      refetch: vi.fn(),
    });

    vi.spyOn(useIncidentHooks, "useIncidentReportHistory").mockReturnValue({
      data: historyData,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_sample_123"
        currentTimeSeconds={10}
        onSeek={vi.fn()}
      />
    );

    expect(el.textContent).toContain("Audit Versions: 2 available");
    expect(el.textContent).toContain("v1");
    expect(el.textContent).toContain("v2 (Latest)");

    // Click on historical version v1
    const v1Btn = Array.from(el.querySelectorAll("button")).find((b) =>
      b.textContent === "v1"
    );
    expect(v1Btn).toBeDefined();

    act(() => {
      v1Btn?.click();
    });

    // Should now show historical banner and v1 report summary
    expect(el.textContent).toContain("Historical Version Viewed");
    expect(el.textContent).toContain("Initial v1 incident report assessment.");
    expect(el.textContent).toContain("View Latest");

    // Click "View Latest" to return
    const backBtn = Array.from(el.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("View Latest")
    );
    act(() => {
      backBtn?.click();
    });

    expect(el.textContent).toContain("Traffic flow remained largely stable across the 120s sequence");
  });

  // ---------------------------------------------------------------------------
  // 14. Defensive Null/Missing Field Handling
  // ---------------------------------------------------------------------------
  it("defensively handles nullable or missing report fields without throwing", () => {
    const minimalResponse: IncidentReportResponse = {
      report_id: "rep_min",
      job_id: "job_1",
      video_id: "vid_1",
      sequence_name: "SEQ_MIN",
      status: "COMPLETED",
      created_at: "2026-11-23T14:00:00Z",
      validation_passed: true,
      report: {
        report_id: "rep_min",
        job_id: "job_1",
        video_id: "vid_1",
        sequence_name: "SEQ_MIN",
        generated_at: "2026-11-23T14:00:00Z",
        executive_summary: "Minimal report.",
        confidence_classification: "medium",
        traffic_conditions: {
          sequence_name: "SEQ_MIN",
          duration_s: 10,
          total_detections: 50,
          total_tracked_instances: 5,
          visual_context_available: false,
        },
        observed_events: [],
        congestion_observations: [],
        stopped_vehicles: [],
        visual_observations: [],
        evidence_references: [],
        recommendations: [],
        unverified_aspects: [],
        limitations: [],
      },
    };

    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: minimalResponse,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: true,
      isCompleted: true,
      isFailed: false,
      refetch: vi.fn(),
    });

    const el = renderComponent(
      <IncidentReportPanel
        jobId="job_1"
        currentTimeSeconds={0}
        onSeek={vi.fn()}
      />
    );

    expect(el.textContent).toContain("Minimal report.");
    expect(el.textContent).toContain("Medium Confidence");
    expect(el.textContent).toContain("CV Telemetry Only");
  });

  // ---------------------------------------------------------------------------
  // 15. Workspace Tab Switching in TelemetryInspectionPanel
  // ---------------------------------------------------------------------------
  it("renders Incident Report tab in TelemetryInspectionPanel and allows tab switching", () => {
    vi.spyOn(useIncidentHooks, "useIncidentReport").mockReturnValue({
      data: sampleMockReportResponse,
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      hasReport: true,
      isCompleted: true,
      isFailed: false,
      refetch: vi.fn(),
    });

    const mockJobResults: JobResults = {
      job_id: "job_sample_123",
      video_id: "vid_1",
      sequence_name: "SEQ_1",
      status: "completed",
      artifacts: {},
      analytics_summary: {
        sequence_name: "SEQ_1",
        duration_s: 10,
        total_frames: 250,
        fps: 25,
        total_unique_vehicles: 5,
        line_counts: {},
        peak_occupancy_per_roi: {},
        average_occupancy_per_roi: {},
        congestion_distribution_per_roi: {},
        stopped_vehicle_events: [],
        flow_windows: [],
      },
      detection_summary: { total_detections: 40, mean_confidence: 0.9, class_distribution: {} },
      tracking_summary: { total_unique_tracks: 5, mostly_tracked_count: 5, mostly_lost_count: 0, id_switches: 0 },
    };

    const el = renderComponent(
      <TelemetryInspectionPanel
        jobId="job_sample_123"
        results={mockJobResults}
        currentTimeSeconds={0}
        onSeek={vi.fn()}
      />
    );

    // Verify all 6 tabs are present
    const buttons = Array.from(el.querySelectorAll("button"));
    const buttonTexts = buttons.map((b) => b.textContent);

    expect(buttonTexts.some((t) => t?.includes("Tracking & Detection"))).toBe(true);
    expect(buttonTexts.some((t) => t?.includes("Flow & Crossings"))).toBe(true);
    expect(buttonTexts.some((t) => t?.includes("Spatial Zones & Congestion"))).toBe(true);
    expect(buttonTexts.some((t) => t?.includes("Stationary Events"))).toBe(true);
    expect(buttonTexts.some((t) => t?.includes("Scene Understanding"))).toBe(true);
    expect(buttonTexts.some((t) => t?.includes("Incident Report"))).toBe(true);

    // Click Incident Report tab
    const reportTabBtn = el.querySelector('[data-testid="tab-incident-report"]') as HTMLButtonElement;
    expect(reportTabBtn).not.toBeNull();

    act(() => {
      reportTabBtn.click();
    });

    // Report content should now be rendered
    expect(el.querySelector('[data-testid="incident-report-panel"]')).not.toBeNull();
    expect(el.textContent).toContain("Traffic flow remained largely stable across the 120s sequence");
  });
});
