/**
 * TanStack Query hooks for Multimodal Scene Interpretation (VLM)
 *
 * Provides:
 * - useSceneInterpretation: auto-polling latest interpretation for a job
 * - useSceneInterpretationHistory: chronological attempt history
 * - useCreateSceneInterpretation: mutation to submit new task
 * - useRetrySceneInterpretation: mutation to retry a failed attempt
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { interpretationService } from "../services/interpretationService";
import {
  SceneInterpretationCreate,
  SceneInterpretationResponse,
  SceneInterpretationHistoryResponse,
  SceneInterpretationLifecycleStatus,
} from "../types/interpretation";
import { QUERY_KEYS } from "../core/queryKeys";
import { APP_CONFIG } from "../core/config";

export interface UseSceneInterpretationResult {
  data?: SceneInterpretationResponse;
  status?: SceneInterpretationLifecycleStatus;
  isLoading: boolean;
  isFetching: boolean;
  isError: boolean;
  error: Error | null;
  isNotRequested: boolean;
  isQueued: boolean;
  isProcessing: boolean;
  isCompleted: boolean;
  isFailed: boolean;
  isActive: boolean;
  isTerminal: boolean;
  refetch: () => void;
}

/**
 * Queries the latest scene interpretation for a job.
 * Automatically polls at ~2s intervals while status is QUEUED or PROCESSING.
 * Automatically halts polling once terminal (COMPLETED, FAILED) or NOT_REQUESTED.
 */
export function useSceneInterpretation(
  jobId: string | undefined
): UseSceneInterpretationResult {
  const isEnabled = Boolean(jobId && jobId !== "preview" && jobId !== "audit");

  const query = useQuery({
    queryKey: QUERY_KEYS.interpretation.latest(jobId),
    queryFn: () => interpretationService.getLatestInterpretation(jobId!),
    enabled: isEnabled,
    refetchInterval: (q) => {
      const currentStatus = q.state.data?.status;
      if (currentStatus === "QUEUED" || currentStatus === "PROCESSING") {
        return APP_CONFIG.interpretationPollingIntervalMs || 2000;
      }
      return false; // Stop polling on terminal states and NOT_REQUESTED
    },
    refetchIntervalInBackground: false,
    retry: 2,
    staleTime: 2000,
  });

  const interp = query.data;
  const interpStatus = interp?.status;
  const isTerminal = interpStatus === "COMPLETED" || interpStatus === "FAILED";
  const isActive = interpStatus === "QUEUED" || interpStatus === "PROCESSING";

  return {
    data: interp,
    status: interpStatus,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    isError: query.isError,
    error: query.error as Error | null,
    isNotRequested: interpStatus === "NOT_REQUESTED",
    isQueued: interpStatus === "QUEUED",
    isProcessing: interpStatus === "PROCESSING",
    isCompleted: interpStatus === "COMPLETED",
    isFailed: interpStatus === "FAILED",
    isActive,
    isTerminal,
    refetch: () => query.refetch(),
  };
}

/**
 * Queries the full chronological interpretation attempt history for a job.
 */
export function useSceneInterpretationHistory(
  jobId: string | undefined,
  ascending: boolean = true
) {
  const isEnabled = Boolean(jobId && jobId !== "preview" && jobId !== "audit");

  return useQuery<SceneInterpretationHistoryResponse>({
    queryKey: QUERY_KEYS.interpretation.history(jobId, ascending),
    queryFn: () => interpretationService.getInterpretationHistory(jobId!, ascending),
    enabled: isEnabled,
    staleTime: 10000,
  });
}

/**
 * Mutation to submit a new scene interpretation request for a completed CV job.
 * Invalidates latest interpretation and history queries on success.
 */
export function useCreateSceneInterpretation(jobId: string | undefined) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (config?: SceneInterpretationCreate) => {
      if (!jobId) {
        throw new Error("Cannot request interpretation without a valid jobId.");
      }
      return interpretationService.requestInterpretation(jobId, config);
    },
    onSuccess: (data) => {
      queryClient.setQueryData(QUERY_KEYS.interpretation.latest(jobId), data);
      queryClient.invalidateQueries({
        queryKey: ["sceneInterpretationHistory", jobId],
      });
    },
  });
}

/**
 * Mutation to retry a failed scene interpretation attempt for a job.
 * Invalidates latest interpretation and history queries on success.
 */
export function useRetrySceneInterpretation(jobId: string | undefined) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => {
      if (!jobId) {
        throw new Error("Cannot retry interpretation without a valid jobId.");
      }
      return interpretationService.retryInterpretation(jobId);
    },
    onSuccess: (data) => {
      queryClient.setQueryData(QUERY_KEYS.interpretation.latest(jobId), data);
      queryClient.invalidateQueries({
        queryKey: ["sceneInterpretationHistory", jobId],
      });
    },
  });
}
