import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { SceneInterpretationPanel } from "../components/workspace/SceneInterpretationPanel";
import { TelemetryInspectionPanel } from "../components/workspace/TelemetryInspectionPanel";
import * as useSceneInterpHooks from "../hooks/useSceneInterpretation";
import { SceneInterpretationResponse } from "../types/interpretation";

// @ts-expect-error - React act testing environment flag
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

describe("Milestone 7B: Scene Understanding UI Component Tests", () => {
  let container: HTMLDivElement;
  let root: any;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    vi.restoreAllMocks();
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

  // ---------------------------------------------------------------------------
  // 1. Rendering Lifecycle States
  // ---------------------------------------------------------------------------
  it("renders NOT_REQUESTED state with EmptyState and trigger CTA", () => {
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

    vi.spyOn(useSceneInterpHooks, "useSceneInterpretationHistory").mockReturnValue({
      data: { job_id: "job_1", total_attempts: 0, attempts: [] },
    } as any);

    const el = renderComponent(
      <SceneInterpretationPanel jobId="job_1" currentTimeSeconds={0} onSeek={vi.fn()} />
    );

    expect(el.querySelector('[data-testid="vlm-not-requested"]')).not.toBeNull();
    expect(el.textContent).toContain("Scene Understanding Not Requested");
    expect(el.textContent).toContain("Run Scene Understanding");
  });

  it("renders QUEUED state with worker queue notice", () => {
    vi.spyOn(useSceneInterpHooks, "useSceneInterpretation").mockReturnValue({
      data: { id: "interp_q", job_id: "job_1", status: "QUEUED" },
      status: "QUEUED",
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      isNotRequested: false,
      isQueued: true,
      isProcessing: false,
      isCompleted: false,
      isFailed: false,
      isActive: true,
      isTerminal: false,
      refetch: vi.fn(),
    });

    vi.spyOn(useSceneInterpHooks, "useSceneInterpretationHistory").mockReturnValue({
      data: { job_id: "job_1", total_attempts: 1, attempts: [] },
    } as any);

    const el = renderComponent(
      <SceneInterpretationPanel jobId="job_1" currentTimeSeconds={0} onSeek={vi.fn()} />
    );

    expect(el.querySelector('[data-testid="vlm-queued"]')).not.toBeNull();
    expect(el.textContent).toContain("Task Enqueued in Interpretation Worker");
  });

  it("renders PROCESSING state with active model indicator", () => {
    vi.spyOn(useSceneInterpHooks, "useSceneInterpretation").mockReturnValue({
      data: {
        id: "interp_p",
        job_id: "job_1",
        status: "PROCESSING",
        model_name: "gemini-2.5-flash",
      },
      status: "PROCESSING",
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      isNotRequested: false,
      isQueued: false,
      isProcessing: true,
      isCompleted: false,
      isFailed: false,
      isActive: true,
      isTerminal: false,
      refetch: vi.fn(),
    });

    vi.spyOn(useSceneInterpHooks, "useSceneInterpretationHistory").mockReturnValue({
      data: { job_id: "job_1", total_attempts: 1, attempts: [] },
    } as any);

    const el = renderComponent(
      <SceneInterpretationPanel jobId="job_1" currentTimeSeconds={0} onSeek={vi.fn()} />
    );

    expect(el.querySelector('[data-testid="vlm-processing"]')).not.toBeNull();
    expect(el.textContent).toContain("VLM Multimodal Reasoning in Progress");
    expect(el.textContent).toContain("gemini-2.5-flash");
  });

  // ---------------------------------------------------------------------------
  // 2. Starting Interpretation
  // ---------------------------------------------------------------------------
  it("triggers requestMutation when Run Scene Understanding is clicked", () => {
    const mutateSpy = vi.fn();
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

    vi.spyOn(useSceneInterpHooks, "useCreateSceneInterpretation").mockReturnValue({
      mutate: mutateSpy,
      isPending: false,
      isError: false,
    } as any);

    const el = renderComponent(
      <SceneInterpretationPanel jobId="job_1" currentTimeSeconds={0} onSeek={vi.fn()} />
    );

    const runBtn = el.querySelector("button");
    expect(runBtn).not.toBeNull();
    act(() => {
      runBtn!.click();
    });
    expect(mutateSpy).toHaveBeenCalled();
  });

  // ---------------------------------------------------------------------------
  // 3 & 4. FAILED State & Retry Limit Handling
  // ---------------------------------------------------------------------------
  it("renders FAILED state with error message, code, and active retry button when attempts < 3", () => {
    const retrySpy = vi.fn();
    vi.spyOn(useSceneInterpHooks, "useSceneInterpretation").mockReturnValue({
      data: {
        id: "interp_f",
        job_id: "job_1",
        status: "FAILED",
        error: {
          error_code: "VLM_PROVIDER_TIMEOUT",
          error_message: "Gemini provider timed out after 30 seconds.",
        },
      },
      status: "FAILED",
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      isNotRequested: false,
      isQueued: false,
      isProcessing: false,
      isCompleted: false,
      isFailed: true,
      isActive: false,
      isTerminal: true,
      refetch: vi.fn(),
    });

    vi.spyOn(useSceneInterpHooks, "useSceneInterpretationHistory").mockReturnValue({
      data: { job_id: "job_1", total_attempts: 1, attempts: [] },
    } as any);

    vi.spyOn(useSceneInterpHooks, "useRetrySceneInterpretation").mockReturnValue({
      mutate: retrySpy,
      isPending: false,
      isError: false,
    } as any);

    const el = renderComponent(
      <SceneInterpretationPanel jobId="job_1" currentTimeSeconds={0} onSeek={vi.fn()} />
    );

    expect(el.querySelector('[data-testid="vlm-failed"]')).not.toBeNull();
    expect(el.textContent).toContain("VLM_PROVIDER_TIMEOUT");
    expect(el.textContent).toContain("Gemini provider timed out");
    expect(el.textContent).toContain("Attempt 1 of 3");

    const retryBtn = el.querySelector('[data-testid="vlm-retry-button"]') as HTMLButtonElement;
    expect(retryBtn).not.toBeNull();
    expect(retryBtn.disabled).toBe(false);
    expect(retryBtn.textContent).toContain("Retry Interpretation");

    act(() => {
      retryBtn.click();
    });
    expect(retrySpy).toHaveBeenCalled();
  });

  it("disables retry button and renders limit message when maximum attempts (3) are reached", () => {
    vi.spyOn(useSceneInterpHooks, "useSceneInterpretation").mockReturnValue({
      data: {
        id: "interp_f3",
        job_id: "job_1",
        status: "FAILED",
        error: {
          error_code: "VLM_RATE_LIMITED",
          error_message: "Rate limit exhausted.",
        },
      },
      status: "FAILED",
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      isNotRequested: false,
      isQueued: false,
      isProcessing: false,
      isCompleted: false,
      isFailed: true,
      isActive: false,
      isTerminal: true,
      refetch: vi.fn(),
    });

    vi.spyOn(useSceneInterpHooks, "useSceneInterpretationHistory").mockReturnValue({
      data: { job_id: "job_1", total_attempts: 3, attempts: [] },
    } as any);

    const el = renderComponent(
      <SceneInterpretationPanel jobId="job_1" currentTimeSeconds={0} onSeek={vi.fn()} />
    );

    const retryBtn = el.querySelector('[data-testid="vlm-retry-button"]') as HTMLButtonElement;
    expect(retryBtn.disabled).toBe(true);
    expect(retryBtn.textContent).toContain("Limit Reached");
    expect(el.textContent).toContain("Maximum 3 attempts reached for this job.");
  });

  // ---------------------------------------------------------------------------
  // 5. Environmental Context & Roadway Description
  // ---------------------------------------------------------------------------
  it("renders COMPLETED state with environmental context and roadway layout", () => {
    const mockCompletedData: SceneInterpretationResponse = {
      id: "interp_c1",
      job_id: "job_1",
      status: "COMPLETED",
      summary: "Clear daytime traffic on an urban divided corridor with steady throughput.",
      duration_seconds: 2.14,
      model_name: "gemini-2.5-flash",
      environmental_context: {
        weather: "clear",
        lighting: "daylight",
        pavement: "dry",
        camera_angle: "elevated south-facing viewpoint",
        visibility_hazards: ["slight windshield glare"],
      },
      scene_description: {
        roadway_topology: "divided_highway",
        lane_count: 4,
        traffic_density_qualitative: "moderate",
        features: ["paved shoulder", "concrete median barrier", "overpass"],
      },
      event_interpretations: [],
      epistemic_claims: [],
      unverified_aspects: [],
    };

    vi.spyOn(useSceneInterpHooks, "useSceneInterpretation").mockReturnValue({
      data: mockCompletedData,
      status: "COMPLETED",
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      isNotRequested: false,
      isQueued: false,
      isProcessing: false,
      isCompleted: true,
      isFailed: false,
      isActive: false,
      isTerminal: true,
      refetch: vi.fn(),
    });

    vi.spyOn(useSceneInterpHooks, "useSceneInterpretationHistory").mockReturnValue({
      data: { job_id: "job_1", total_attempts: 1, attempts: [mockCompletedData] },
    } as any);

    const el = renderComponent(
      <SceneInterpretationPanel jobId="job_1" currentTimeSeconds={1.5} onSeek={vi.fn()} />
    );

    expect(el.querySelector('[data-testid="vlm-completed"]')).not.toBeNull();
    expect(el.textContent).toContain("Clear daytime traffic on an urban divided corridor");
    expect(el.textContent).toContain("clear");
    expect(el.textContent).toContain("daylight");
    expect(el.textContent).toContain("dry");
    expect(el.textContent).toContain("slight windshield glare");
    expect(el.textContent).toContain("divided highway");
    expect(el.textContent).toContain("4 lanes");
    expect(el.textContent).toContain("moderate");
    expect(el.textContent).toContain("concrete median barrier");
  });

  // ---------------------------------------------------------------------------
  // 6. Traffic Event Interpretations
  // ---------------------------------------------------------------------------
  it("renders event interpretations with probable cause labeled as hypothesis", () => {
    const mockEventData: SceneInterpretationResponse = {
      id: "interp_c2",
      job_id: "job_1",
      status: "COMPLETED",
      summary: "Sequence exhibits stationary vehicle event.",
      event_interpretations: [
        {
          event_id: "stop_4_3.20",
          event_type: "stopped_vehicle",
          track_id: 4,
          interpretation: "Vehicle remained stationary due to temporary traffic queue formation.",
          probable_cause: "Downstream intersection queue spillback",
          confidence: "direct_observation",
          supporting_citations: ["keyframe_001", "t=3.2s"],
        },
      ],
    };

    vi.spyOn(useSceneInterpHooks, "useSceneInterpretation").mockReturnValue({
      data: mockEventData,
      status: "COMPLETED",
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      isNotRequested: false,
      isQueued: false,
      isProcessing: false,
      isCompleted: true,
      isFailed: false,
      isActive: false,
      isTerminal: true,
      refetch: vi.fn(),
    });

    vi.spyOn(useSceneInterpHooks, "useSceneInterpretationHistory").mockReturnValue({
      data: { job_id: "job_1", total_attempts: 1, attempts: [] },
    } as any);

    const el = renderComponent(
      <SceneInterpretationPanel jobId="job_1" currentTimeSeconds={0} onSeek={vi.fn()} />
    );

    const evtsSection = el.querySelector('[data-testid="vlm-event-interpretations"]');
    expect(evtsSection).not.toBeNull();
    expect(evtsSection!.textContent).toContain("Track #4");
    expect(evtsSection!.textContent).toContain("Interpreted Probable Cause:");
    expect(evtsSection!.textContent).toContain("Downstream intersection queue spillback");
    expect(evtsSection!.textContent).toContain("Hypotheses (Advisory)");
  });

  // ---------------------------------------------------------------------------
  // 7. Epistemic Claims and Provenance Badges
  // ---------------------------------------------------------------------------
  it("renders categorized epistemic claims with provenance citations and clickable timestamps", () => {
    const seekSpy = vi.fn();
    const mockClaimsData: SceneInterpretationResponse = {
      id: "interp_c3",
      job_id: "job_1",
      status: "COMPLETED",
      summary: "Scene analysis complete.",
      epistemic_claims: [
        {
          claim_id: "cl_1",
          category: "cv_fact",
          statement: "8 unique vehicle tracklets were tracked across 24.5 seconds.",
          keyframe_ids: [],
          frame_timestamps: [1.2, 5.0],
          associated_track_ids: [1, 2, 3],
          evidence_rationale: "Directly measured from ByteTrack tracking summary.",
          telemetry_provenance: {
            source_type: "telemetry_summary",
            field_path: "total_unique_tracks",
            numeric_value: 8,
            unit: "tracks",
          },
        },
        {
          claim_id: "cl_2",
          category: "visual_observation",
          statement: "Roadway exhibits dry asphalt surface.",
          keyframe_ids: ["kf_1"],
          frame_timestamps: [2.0],
          associated_track_ids: [],
          evidence_rationale: "Directly visible in keyframe 1.",
        },
        {
          claim_id: "cl_3",
          category: "inference",
          statement: "Vehicle 4 stopped due to queue congestion.",
          keyframe_ids: [],
          frame_timestamps: [],
          associated_track_ids: [4],
          evidence_rationale: "Deduced from stoppage duration and upstream traffic.",
        },
      ],
    };

    vi.spyOn(useSceneInterpHooks, "useSceneInterpretation").mockReturnValue({
      data: mockClaimsData,
      status: "COMPLETED",
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      isNotRequested: false,
      isQueued: false,
      isProcessing: false,
      isCompleted: true,
      isFailed: false,
      isActive: false,
      isTerminal: true,
      refetch: vi.fn(),
    });

    vi.spyOn(useSceneInterpHooks, "useSceneInterpretationHistory").mockReturnValue({
      data: { job_id: "job_1", total_attempts: 1, attempts: [] },
    } as any);

    const el = renderComponent(
      <SceneInterpretationPanel jobId="job_1" currentTimeSeconds={0} onSeek={seekSpy} />
    );

    const claimsSection = el.querySelector('[data-testid="vlm-epistemic-claims"]');
    expect(claimsSection).not.toBeNull();
    expect(claimsSection!.textContent).toContain("CV Fact");
    expect(claimsSection!.textContent).toContain("Visual Observation");
    expect(claimsSection!.textContent).toContain("Inference");
    expect(claimsSection!.textContent).toContain("total_unique_tracks");
    expect(claimsSection!.textContent).toContain("8 tracks");

    // Click timestamp button
    const timeBtn = claimsSection!.querySelector("button");
    expect(timeBtn).not.toBeNull();
    act(() => {
      timeBtn!.click();
    });
    expect(seekSpy).toHaveBeenCalledWith(1.2);
  });

  // ---------------------------------------------------------------------------
  // 8. Nullable / Missing Analytical Fields Graceful Handling
  // ---------------------------------------------------------------------------
  it("gracefully omits missing sections without throwing or fabricating observations", () => {
    const minimalCompleted: SceneInterpretationResponse = {
      id: "interp_min",
      job_id: "job_1",
      status: "COMPLETED",
      summary: "Minimal summary without optional context.",
      environmental_context: null,
      scene_description: null,
      event_interpretations: null,
      epistemic_claims: null,
      unverified_aspects: ["No lane markers detectable due to night darkness."],
    };

    vi.spyOn(useSceneInterpHooks, "useSceneInterpretation").mockReturnValue({
      data: minimalCompleted,
      status: "COMPLETED",
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      isNotRequested: false,
      isQueued: false,
      isProcessing: false,
      isCompleted: true,
      isFailed: false,
      isActive: false,
      isTerminal: true,
      refetch: vi.fn(),
    });

    vi.spyOn(useSceneInterpHooks, "useSceneInterpretationHistory").mockReturnValue({
      data: { job_id: "job_1", total_attempts: 1, attempts: [] },
    } as any);

    const el = renderComponent(
      <SceneInterpretationPanel jobId="job_1" currentTimeSeconds={0} onSeek={vi.fn()} />
    );

    expect(el.querySelector('[data-testid="vlm-environmental-context"]')).toBeNull();
    expect(el.querySelector('[data-testid="vlm-scene-description"]')).toBeNull();
    expect(el.querySelector('[data-testid="vlm-event-interpretations"]')).toBeNull();
    expect(el.querySelector('[data-testid="vlm-epistemic-claims"]')).toBeNull();

    // Limitations section must show the unverified aspect
    const limSection = el.querySelector('[data-testid="vlm-limitations-section"]');
    expect(limSection).not.toBeNull();
    expect(limSection!.textContent).toContain("No lane markers detectable due to night darkness.");
  });

  // ---------------------------------------------------------------------------
  // 9. API Error Handling
  // ---------------------------------------------------------------------------
  it("renders ErrorAlert when useSceneInterpretation encounters query failure", () => {
    const refetchSpy = vi.fn();
    vi.spyOn(useSceneInterpHooks, "useSceneInterpretation").mockReturnValue({
      data: undefined,
      status: undefined,
      isLoading: false,
      isFetching: false,
      isError: true,
      error: new Error("Network connection lost"),
      isNotRequested: false,
      isQueued: false,
      isProcessing: false,
      isCompleted: false,
      isFailed: false,
      isActive: false,
      isTerminal: false,
      refetch: refetchSpy,
    });

    const el = renderComponent(
      <SceneInterpretationPanel jobId="job_1" currentTimeSeconds={0} onSeek={vi.fn()} />
    );

    expect(el.querySelector('[data-testid="vlm-error-state"]')).not.toBeNull();
    expect(el.textContent).toContain("Network connection lost");

    const retryBtn = el.querySelector("button");
    act(() => {
      retryBtn!.click();
    });
    expect(refetchSpy).toHaveBeenCalled();
  });

  // ---------------------------------------------------------------------------
  // 10. Preserving Existing Workspace Tabs in TelemetryInspectionPanel
  // ---------------------------------------------------------------------------
  it("renders Scene Understanding tab in TelemetryInspectionPanel and allows tab switching", () => {
    vi.spyOn(useSceneInterpHooks, "useSceneInterpretation").mockReturnValue({
      data: { id: null, job_id: "job_test", status: "NOT_REQUESTED" },
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

    const el = renderComponent(
      <TelemetryInspectionPanel
        jobId="job_test"
        results={{
          job_id: "job_test",
          video_id: "video_1",
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
        }}
        currentTimeSeconds={0}
        onSeek={vi.fn()}
      />
    );

    // Check that all 5 tab buttons are present
    const buttons = Array.from(el.querySelectorAll("button"));
    const buttonTexts = buttons.map((b) => b.textContent);

    expect(buttonTexts.some((t) => t?.includes("Tracking & Detection"))).toBe(true);
    expect(buttonTexts.some((t) => t?.includes("Flow & Crossings"))).toBe(true);
    expect(buttonTexts.some((t) => t?.includes("Spatial Zones & Congestion"))).toBe(true);
    expect(buttonTexts.some((t) => t?.includes("Stationary Events"))).toBe(true);
    expect(buttonTexts.some((t) => t?.includes("Scene Understanding"))).toBe(true);

    // Click Scene Understanding tab
    const vlmTabBtn = el.querySelector('[data-testid="tab-scene-understanding"]') as HTMLButtonElement;
    expect(vlmTabBtn).not.toBeNull();
    act(() => {
      vlmTabBtn.click();
    });

    // Subpanel should switch to SceneInterpretationPanel
    expect(el.querySelector('[data-testid="scene-interpretation-panel"]')).not.toBeNull();
    expect(el.textContent).toContain("Multimodal Scene Understanding");
  });
});
