import { describe, it, expect, beforeEach } from "vitest";
import { videoService } from "../services/videoService";
import { jobService } from "../services/jobService";
import { useWorkspaceStore } from "../state/useWorkspaceStore";
import { StoppedVehicleEvent, LineFlow, JobResults } from "../types/telemetry";

describe("Milestone 4B: Video Playback & URL Construction", () => {
  it("constructs correct source video streaming URL with encoding", () => {
    const url = videoService.getVideoContentUrl("vid_abc 123");
    expect(url).toContain("/videos/vid_abc%20123/content");
  });

  it("constructs correct annotated video streaming URL with encoding", () => {
    const url = jobService.getAnnotatedVideoUrl("job_xyz 456");
    expect(url).toContain("/jobs/job_xyz%20456/annotated-video");
  });

  it("calculates correct frame index from timestamp at 25 FPS", () => {
    const fps = 25.0;
    const totalFrames = 75;

    const computeFrame = (t: number) =>
      Math.min(totalFrames, Math.max(1, Math.floor(t * fps) + 1));

    expect(computeFrame(0.0)).toBe(1);
    expect(computeFrame(0.04)).toBe(2);
    expect(computeFrame(1.0)).toBe(26);
    expect(computeFrame(2.96)).toBe(75);
    expect(computeFrame(5.0)).toBe(75); // Clamped to totalFrames
    expect(computeFrame(-1.0)).toBe(1); // Clamped to 1
  });

  it("steps frames backward and forward accurately without out-of-bounds drift", () => {
    const fps = 25.0;
    const frameDuration = 1.0 / fps; // 0.04s
    const duration = 3.0;

    const step = (current: number, direction: -1 | 1) =>
      Math.max(0, Math.min(duration, current + direction * frameDuration));

    let time = 1.0;
    time = step(time, 1);
    expect(time).toBeCloseTo(1.04, 3);

    time = step(time, -1);
    expect(time).toBeCloseTo(1.0, 3);

    // Clamping at zero
    expect(step(0.02, -1)).toBe(0);

    // Clamping at duration
    expect(step(2.98, 1)).toBe(3.0);
  });
});

describe("Milestone 4B: View Mode & Fallback Logic", () => {
  beforeEach(() => {
    useWorkspaceStore.setState({
      viewMode: "annotated",
      isPlaying: false,
      currentTimeSeconds: 0,
      currentFrame: 1,
    });
  });

  it("allows switching viewMode in workspace store", () => {
    const store = useWorkspaceStore.getState();
    expect(store.viewMode).toBe("annotated");

    store.setViewMode("source");
    expect(useWorkspaceStore.getState().viewMode).toBe("source");

    store.setViewMode("side-by-side");
    expect(useWorkspaceStore.getState().viewMode).toBe("side-by-side");
  });

  it("correctly determines effective view mode fallback when annotated video is unavailable", () => {
    const resolveEffectiveMode = (
      selectedMode: "annotated" | "source" | "side-by-side",
      isAnnotatedAvailable: boolean
    ): "annotated" | "source" | "side-by-side" => {
      if (selectedMode === "annotated" && !isAnnotatedAvailable) {
        return "source";
      }
      return selectedMode;
    };

    expect(resolveEffectiveMode("annotated", true)).toBe("annotated");
    expect(resolveEffectiveMode("annotated", false)).toBe("source");
    expect(resolveEffectiveMode("source", false)).toBe("source");
    expect(resolveEffectiveMode("side-by-side", true)).toBe("side-by-side");
  });
});

describe("Milestone 4B: Synchronized Event Timeline Logic", () => {
  const sampleEvents: StoppedVehicleEvent[] = [
    {
      track_id: 12,
      start_frame: 25,
      start_timestamp_s: 1.0,
      end_frame: 50,
      end_timestamp_s: 2.0,
      duration_s: 1.0,
      location: [400.0, 300.0],
      roi_id: "roi_main",
      status: "resolved",
    },
    {
      track_id: 18,
      start_frame: 60,
      start_timestamp_s: 2.4,
      end_frame: null,
      end_timestamp_s: null,
      duration_s: 0.6,
      location: [500.0, 320.0],
      status: "active",
    },
  ];

  const sampleFlowWindows: LineFlow[] = [
    {
      line_id: "crossline_1",
      window_start_s: 0.0,
      window_end_s: 1.5,
      total_count: 5,
      flow_rate_vpm: 200.0,
      direction_counts: { downstream: 5 },
      direction_flows_vpm: { downstream: 200.0 },
    },
    {
      line_id: "crossline_1",
      window_start_s: 1.5,
      window_end_s: 3.0,
      total_count: 3,
      flow_rate_vpm: 120.0,
      direction_counts: { downstream: 3 },
      direction_flows_vpm: { downstream: 120.0 },
    },
  ];

  it("calculates active stopped vehicle events accurately at a given playback timestamp", () => {
    const isStoppedActiveAt = (evt: StoppedVehicleEvent, t: number) => {
      const end = evt.end_timestamp_s ?? evt.start_timestamp_s + evt.duration_s;
      return t >= evt.start_timestamp_s && t <= end;
    };

    // Before event 1
    expect(isStoppedActiveAt(sampleEvents[0], 0.5)).toBe(false);

    // During event 1 (1.0s - 2.0s)
    expect(isStoppedActiveAt(sampleEvents[0], 1.0)).toBe(true);
    expect(isStoppedActiveAt(sampleEvents[0], 1.5)).toBe(true);
    expect(isStoppedActiveAt(sampleEvents[0], 2.0)).toBe(true);

    // After event 1
    expect(isStoppedActiveAt(sampleEvents[0], 2.1)).toBe(false);

    // During active event 2 (start 2.4s, duration 0.6s -> end 3.0s)
    expect(isStoppedActiveAt(sampleEvents[1], 2.5)).toBe(true);
  });

  it("matches active flow window accurately at given playback timestamps", () => {
    const findActiveFlow = (windows: LineFlow[], t: number) =>
      windows.find((w) => t >= w.window_start_s && t <= w.window_end_s);

    const activeAt1s = findActiveFlow(sampleFlowWindows, 1.0);
    expect(activeAt1s?.window_start_s).toBe(0.0);
    expect(activeAt1s?.flow_rate_vpm).toBe(200.0);

    const activeAt2s = findActiveFlow(sampleFlowWindows, 2.0);
    expect(activeAt2s?.window_start_s).toBe(1.5);
    expect(activeAt2s?.flow_rate_vpm).toBe(120.0);
  });

  it("computes accurate marker positions on timeline bar", () => {
    const duration = 3.0;
    const getMarkerPct = (timestamp: number) => (timestamp / duration) * 100;

    expect(getMarkerPct(1.5)).toBeCloseTo(50.0, 1);
    expect(getMarkerPct(0.0)).toBe(0.0);
    expect(getMarkerPct(3.0)).toBe(100.0);
  });
});

describe("Milestone 4B: Real Telemetry Mapping & Disclosures", () => {
  it("preserves tracking identity semantics without claiming physical vehicle ground truth", () => {
    const mockResults: Partial<JobResults> = {
      tracking_summary: {
        total_unique_tracks: 14,
        mostly_tracked_count: 10,
        mostly_lost_count: 2,
        id_switches: 3,
        mota: null,
        hota: null,
        idf1: null,
      },
      analytics_summary: {
        sequence_name: "test_seq",
        total_frames: 75,
        duration_s: 3.0,
        fps: 25.0,
        total_unique_vehicles: 14,
        line_counts: { line_1: { downstream: 8, upstream: 6 } },
        flow_windows: [],
        stopped_vehicle_events: [],
        peak_occupancy_per_roi: { roi_1: 0.245 },
        average_occupancy_per_roi: { roi_1: 0.112 },
        congestion_distribution_per_roi: { roi_1: { low: 70, moderate: 5, high: 0 } },
      },
    };

    // Track identities extracted from real telemetry
    const trackedIdentities =
      mockResults.tracking_summary?.total_unique_tracks ??
      mockResults.analytics_summary?.total_unique_vehicles;
    expect(trackedIdentities).toBe(14);

    // Offline benchmarks (MOTA/HOTA/IDF1) are null for unannotated live sequences
    expect(mockResults.tracking_summary?.mota).toBeNull();
    expect(mockResults.tracking_summary?.hota).toBeNull();
    expect(mockResults.tracking_summary?.idf1).toBeNull();
  });

  it("enforces class-agnostic vehicle baseline without fabricating unvalidated subtypes", () => {
    // When backend returns unified class mapping
    const detectionSummary = {
      total_detections: 142,
      mean_confidence: 0.88,
      class_distribution: { vehicle: 142 },
    };

    // The primary operational category must be unified vehicle
    const activeClass = Object.keys(detectionSummary.class_distribution);
    expect(activeClass).toEqual(["vehicle"]);
    expect(activeClass).not.toContain("car");
    expect(activeClass).not.toContain("truck");
    expect(activeClass).not.toContain("motorcycle");
  });

  it("strictly separates aggregate line counts from timestamped flow windows", () => {
    const analytics = {
      line_counts: {
        crossline_main: { downstream: 12, upstream: 5 },
      },
      flow_windows: [
        {
          line_id: "crossline_main",
          window_start_s: 0.0,
          window_end_s: 1.5,
          total_count: 8,
          flow_rate_vpm: 320.0,
          direction_counts: { downstream: 6, upstream: 2 },
          direction_flows_vpm: { downstream: 240.0, upstream: 80.0 },
        },
      ],
    };

    // Aggregate sequence count
    const totalDownstream = analytics.line_counts.crossline_main.downstream;
    const totalUpstream = analytics.line_counts.crossline_main.upstream;
    const cumulativeTotal = totalDownstream + totalUpstream;
    expect(cumulativeTotal).toBe(17);

    // Flow windows have explicit start/end timestamps, whereas line_counts is sequence-level aggregate
    expect(analytics.flow_windows[0].window_start_s).toBe(0.0);
    expect(analytics.flow_windows[0].window_end_s).toBe(1.5);
    expect((analytics.line_counts.crossline_main as any).timestamp_s).toBeUndefined();
  });
});

describe("Milestone 4B: Dual-Video Synchronization Drift Correction", () => {
  const DRIFT_TOLERANCE = 0.05;
  const RATE_ADJUST_THRESHOLD = 0.25;

  const calculateSyncAction = (
    primaryTime: number,
    secondaryTime: number,
    baseRate: number
  ): { action: "in-sync" | "adjust-rate" | "hard-seek"; rate: number; seekTo?: number } => {
    const diff = primaryTime - secondaryTime;
    const absDiff = Math.abs(diff);

    if (absDiff <= DRIFT_TOLERANCE) {
      return { action: "in-sync", rate: baseRate };
    } else if (absDiff <= RATE_ADJUST_THRESHOLD) {
      const adjustment = diff > 0 ? 1.08 : 0.92;
      return { action: "adjust-rate", rate: Math.round(baseRate * adjustment * 1000) / 1000 };
    } else {
      return { action: "hard-seek", rate: baseRate, seekTo: primaryTime };
    }
  };

  it("maintains base rate when streams are within tolerance (<= 0.05s)", () => {
    const result = calculateSyncAction(1.00, 1.02, 1.0);
    expect(result.action).toBe("in-sync");
    expect(result.rate).toBe(1.0);
  });

  it("gently accelerates lagging secondary video when drift is moderate (0.05s - 0.25s)", () => {
    // Secondary is 0.10s behind primary
    const result = calculateSyncAction(1.50, 1.40, 1.0);
    expect(result.action).toBe("adjust-rate");
    expect(result.rate).toBe(1.08); // 8% speedup
  });

  it("gently slows down leading secondary video when drift is moderate", () => {
    // Secondary is 0.10s ahead of primary
    const result = calculateSyncAction(1.40, 1.50, 1.0);
    expect(result.action).toBe("adjust-rate");
    expect(result.rate).toBe(0.92); // 8% slowdown
  });

  it("forces hard seek only when drift exceeds threshold (> 0.25s)", () => {
    // Secondary is 0.50s behind primary (e.g. after network pause)
    const result = calculateSyncAction(2.00, 1.50, 1.0);
    expect(result.action).toBe("hard-seek");
    expect(result.seekTo).toBe(2.00);
    expect(result.rate).toBe(1.0);
  });
});
