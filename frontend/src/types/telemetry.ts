/**
 * Telemetry and analytics schemas matching backend/app/schemas/telemetry.py
 */

export type CongestionLevel = "low" | "moderate" | "high";

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

export interface SpeedMeasurement {
  track_id: number;
  frame_idx: number;
  speed_px_per_s: number;
  speed_kmh?: number | null;
  is_calibrated: boolean;
  calibration_note: string;
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

export interface AnnotationDiagnostics {
  status: "COMPLETED" | "FAILED" | "UNAVAILABLE";
  selected_codec?: string | null;
  input_metadata?: Record<string, any>;
  output_metadata?: Record<string, any>;
  frames_written?: number;
  writer_initialized?: boolean;
  writer_finalized?: boolean;
  verification_passed?: boolean;
  error_message?: string | null;
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
  annotation_diagnostics?: AnnotationDiagnostics | null;
}
