/**
 * Hook querying backend health status via TanStack Query
 */

import { useQuery } from "@tanstack/react-query";
import { healthService } from "../services/healthService";
import { HealthCheckResponse } from "../types/api";
import { APP_CONFIG } from "../core/config";

export type HealthStatusMode = "online" | "offline" | "checking";

export function useHealth() {
  const query = useQuery<HealthCheckResponse, Error>({
    queryKey: ["health"],
    queryFn: () => healthService.getHealth(),
    refetchInterval: APP_CONFIG.healthPollingIntervalMs,
    refetchIntervalInBackground: true,
    retry: 2,
    staleTime: 5000,
  });

  const mode: HealthStatusMode = query.isLoading
    ? "checking"
    : query.isSuccess && query.data?.status === "healthy"
      ? "online"
      : "offline";

  return {
    ...query,
    mode,
    health: query.data,
  };
}
