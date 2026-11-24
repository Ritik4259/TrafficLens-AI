/**
/**
 * TypeScript contracts for Multimodal Scene Interpretation (VLM)
 * Matching backend/app/schemas/scene_interpretation.py and vlm.py
 */

export type SceneInterpretationLifecycleStatus =
  | "NOT_REQUESTED"
  | "QUEUED"
  | "PROCESSING"
  | "COMPLETED"
  | "FAILED";

export interface SceneInterpretationCreate {
  provider_type?: string;
  model_name?: string;
  frame_budget?: number;
  temperature?: number;
}

export interface SceneInterpretationError {
  error_code: string;
  error_message: string;
  error_details?: Record<string, any> | null;
}

export interface EnvironmentalContext {
  weather?: string | null;
  lighting?: string | null;
  pavement?: string | null;
  camera_angle?: string | null;
  visibility_hazards: string[];
}

export interface SceneDescription {
  roadway_topology?: string | null;
  lane_count?: number | null;
  traffic_density_qualitative?: string | null;
  features: string[];
}

export interface TrafficEventInterpretation {
  event_id: string;
  event_type?: string | null;
  track_id?: number | null;
  interpretation: string;
  probable_cause?: string | null;
  confidence?: string | null;
  supporting_citations: string[];
}

export type EpistemicCategory =
  | "cv_fact"
  | "visual_observation"
  | "inference"
  | "unverified";

export type ConfidenceCategory =
  | "direct_observation"
  | "supported_inference"
  | "speculative";

export interface TelemetryProvenance {
  source_type: "telemetry_summary" | "event";
  record_id?: string | null;
  field_path: string;
  numeric_value?: number | null;
  text_value?: string | null;
  subject?: string | null;
  predicate?: string | null;
  unit?: string | null;
}

export interface EpistemicClaim {
  claim_id: string;
  category: EpistemicCategory;
  statement: string;
  keyframe_ids: string[];
  frame_timestamps: number[];
  associated_track_ids: number[];
  evidence_rationale: string;
  telemetry_provenance?: TelemetryProvenance | null;
}

export interface SceneInterpretationResponse {
  id: string | null;
  job_id: string;
  video_id?: string | null;
  status: SceneInterpretationLifecycleStatus;
  created_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  duration_seconds?: number | null;
  provider_type?: string | null;
  model_name?: string | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  summary?: string | null;
  environmental_context?: EnvironmentalContext | null;
  scene_description?: SceneDescription | null;
  event_interpretations?: TrafficEventInterpretation[] | null;
  epistemic_claims?: EpistemicClaim[] | null;
  unverified_aspects?: string[] | null;
  evidence_package_id?: string | null;
  error?: SceneInterpretationError | null;
}

export interface SceneInterpretationHistoryResponse {
  job_id: string;
  total_attempts: number;
  attempts: SceneInterpretationResponse[];
}
