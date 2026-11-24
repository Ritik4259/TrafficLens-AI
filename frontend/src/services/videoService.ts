/**
 * Video service querying backend /api/v1/videos
 */

import { apiClient } from "./apiClient";
import { VideoMetadata, VideoUploadResponse } from "../types/video";
import { ProcessingJob, ProcessingJobCreate } from "../types/job";

export const videoService = {
  async listVideos(): Promise<VideoMetadata[]> {
    return apiClient.get<VideoMetadata[]>("/videos");
  },

  async getVideo(videoId: string): Promise<VideoMetadata> {
    return apiClient.get<VideoMetadata>(`/videos/${encodeURIComponent(videoId)}`);
  },

  getVideoContentUrl(videoId: string): string {
    return apiClient.buildUrl(`/videos/${encodeURIComponent(videoId)}/content`);
  },

  async uploadVideo(
    file: File,
    onProgress?: (percent: number) => void
  ): Promise<VideoUploadResponse> {
    const formData = new FormData();
    formData.append("file", file, file.name);
    return apiClient.uploadWithProgress<VideoUploadResponse>("/videos", formData, onProgress);
  },

  async createVideoJob(
    videoId: string,
    payload?: ProcessingJobCreate
  ): Promise<ProcessingJob> {
    return apiClient.post<ProcessingJob>(
      `/videos/${encodeURIComponent(videoId)}/jobs`,
      payload
    );
  },
};
