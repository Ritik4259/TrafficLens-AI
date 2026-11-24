/**
 * Service client for Multimodal Scene Interpretation (VLM)
 * Interacts with backend /api/v1/jobs/{job_id}/interpretations and /api/v1/interpretations
 */

import { apiClient } from "./apiClient";
import {
  SceneInterpretationCreate,
  SceneInterpretationHistoryResponse,
  SceneInterpretationResponse,
} from "../types/interpretation";

export const interpretationService = {
  /**
   * Submits a request for multimodal scene interpretation for a completed job.
   */
  async requestInterpretation(
    jobId: string,
    payload?: SceneInterpretationCreate
  ): Promise<SceneInterpretationResponse> {
    return apiClient.post<SceneInterpretationResponse>(
      `/jobs/${encodeURIComponent(jobId)}/interpretations`,
      payload
    );
  },

  /**
   * Retrieves the most recent scene interpretation for a job.
   * Returns virtual NOT_REQUESTED (200 OK) if none was ever requested.
   */
  async getLatestInterpretation(jobId: string): Promise<SceneInterpretationResponse> {
    return apiClient.get<SceneInterpretationResponse>(
      `/jobs/${encodeURIComponent(jobId)}/interpretation`
    );
  },

  /**
   * Retrieves a specific scene interpretation by its unique ID.
   */
  async getInterpretationById(
    interpretationId: string
  ): Promise<SceneInterpretationResponse> {
    return apiClient.get<SceneInterpretationResponse>(
      `/interpretations/${encodeURIComponent(interpretationId)}`
    );
  },

  /**
   * Retrieves the full chronological attempt history for a job.
   */
  async getInterpretationHistory(
    jobId: string,
    ascending: boolean = true
  ): Promise<SceneInterpretationHistoryResponse> {
    return apiClient.get<SceneInterpretationHistoryResponse>(
      `/jobs/${encodeURIComponent(jobId)}/interpretations?ascending=${ascending}`
    );
  },

  /**
   * Retries an interpretation for a job whose latest attempt was FAILED.
   * Creates a new attempt record up to settings.MAX_INTERPRETATION_ATTEMPTS.
   */
  async retryInterpretation(jobId: string): Promise<SceneInterpretationResponse> {
    return apiClient.post<SceneInterpretationResponse>(
      `/jobs/${encodeURIComponent(jobId)}/interpretations/retry`
    );
  },

  /**
   * Retries a specific failed interpretation attempt by ID.
   */
  async retryInterpretationById(
    interpretationId: string
  ): Promise<SceneInterpretationResponse> {
    return apiClient.post<SceneInterpretationResponse>(
      `/interpretations/${encodeURIComponent(interpretationId)}/retry`
    );
  },
};
