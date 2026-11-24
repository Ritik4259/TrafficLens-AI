/**
 * Core API and Health schemas matching backend/app/schemas/ and GET /api/v1/health
 */

export interface HealthCheckResponse {
  status: "healthy" | "degraded" | "unhealthy" | string;
  service: string;
  version: string;
  environment: string;
  database_dialect: string;
  database_connected?: boolean;
}

export interface ApiError {
  status: number;
  message: string;
  detail?: string | Record<string, any>;
}
