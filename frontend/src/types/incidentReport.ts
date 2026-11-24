/**
 * TypeScript contracts for Evidence-Grounded Traffic Incident Reports
 * Matching backend/app/schemas/incident_report.py
 */

export type ReportConfidence = "high" | "medium" | "low" | "cautionary";

export interface TrafficConditions {
  sequence_name: string;
  duration_s: number;
  total_detections: number;
  total_tracked_instances: number;
  overall_flow_rate_vpm?: number | null;
  peak_occupancy_ratio?: number | null;
  environmental_context?: Record<string, any> | null;
  visual_context_available: boolean;
  visual_context_note?: string | null;
}

export interface CongestionObservation {
  roi_id: string;
  observed_level: string;
  peak_occupancy: number;
  average_occupancy?: number | null;
  is_heuristic_rule: boolean;
  description: string;
}

export interface StoppedVehicleReportItem {
  event_id: string;
  track_id: number;
  start_timestamp_s: number;
  duration_s: number;
  location?: [number, number] | null;
  roi_id?: string | null;
  probable_cause?: string | null;
  contextual_explanation?: string | null;
  verification_status: string;
}

export interface VisualObservationReportItem {
  observation_id: string;
  keyframe_id?: string | null;
  timestamp_s?: number | null;
  description: string;
  visual_cues: string[];
}

export interface ObservedEvent {
  event_type: string;
  timestamp_s?: number | null;
  description: string;
  supporting_evidence_ref: string;
}

export interface EvidenceReference {
  source: string;
  metric_or_field: string;
  asserted_value: any;
  verified: boolean;
}

export interface FollowUpRecommendation {
  action: string;
  priority: string;
  rationale: string;
  is_advisory: boolean;
}

export interface ReportValidationDiagnostics {
  validation_passed: boolean;
  total_claims_checked: number;
  grounded_claims: number;
  demoted_claims: number;
  warnings: string[];
  notes: string[];
}

export interface IncidentReport {
  report_id: string;
  job_id: string;
  video_id: string;
  sequence_name: string;
  generated_at: string;
  executive_summary: string;
  traffic_conditions: TrafficConditions;
  observed_events: ObservedEvent[];
  congestion_observations: CongestionObservation[];
  stopped_vehicles: StoppedVehicleReportItem[];
  visual_observations: VisualObservationReportItem[];
  evidence_references: EvidenceReference[];
  confidence_classification: ReportConfidence;
  unverified_aspects: string[];
  limitations: string[];
  recommendations: FollowUpRecommendation[];
  validation_diagnostics?: ReportValidationDiagnostics | null;
}

export interface IncidentReportResponse {
  report_id: string;
  job_id: string;
  video_id: string;
  sequence_name: string;
  status: "COMPLETED" | "FAILED";
  created_at: string;
  report?: IncidentReport | null;
  validation_passed: boolean;
  error_message?: string | null;
}

export interface IncidentReportHistoryResponse {
  job_id: string;
  total_reports: number;
  reports: IncidentReportResponse[];
}
