/**
 * Video asset schemas matching backend/app/schemas/job.py
 */

export interface VideoMetadata {
  video_id: string;
  filename: string;
  content_type: string;
  file_size_bytes: number;
  duration_seconds?: number | null;
  fps?: number | null;
  width?: number | null;
  height?: number | null;
  frame_count?: number | null;
  uploaded_at: string;
}

export interface VideoUploadResponse {
  video_id: string;
  metadata: VideoMetadata;
  message: string;
}
