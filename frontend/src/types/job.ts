/**
 * Processing job schemas matching backend/app/schemas/job.py
 */

import { CountingLineConfig, ROIConfig } from "./common";

export type JobStatus = "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED";

export interface ProcessingError {
  error_code: string;
  message: string;
  details?: Record<string, any> | null;
  occurred_at: string;
}

export interface ProcessingJobCreate {
  video_id?: string;
  sequence_name?: string;
  lines?: CountingLineConfig[];
  rois?: ROIConfig[];
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

export interface JobArtifactItem {
  name: string;
  filename: string;
  artifact_type: string;
  is_available: boolean;
  file_size_bytes?: number | null;
  serving_url?: string | null;
}

export interface JobArtifactsResponse {
  job_id: string;
  status: JobStatus;
  artifacts: JobArtifactItem[];
}

