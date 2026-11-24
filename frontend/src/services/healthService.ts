/**
 * Health service querying backend /api/v1/health
 */

import { apiClient } from "./apiClient";
import { HealthCheckResponse } from "../types/api";

export const healthService = {
  async getHealth(): Promise<HealthCheckResponse> {
    return apiClient.get<HealthCheckResponse>("/health");
  },
};
