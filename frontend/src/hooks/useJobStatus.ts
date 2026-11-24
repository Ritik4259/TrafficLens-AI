/**
 * Asynchronous job status polling hook using TanStack Query.
 *
 * Polls at ~1.5s intervals when status is QUEUED or PROCESSING.
 * Automatically stops polling when job reaches a terminal state (COMPLETED or FAILED).
 */

import { useQuery } from "@tanstack/react-query";
import { jobService } from "../services/jobService";
import { ProcessingJob, JobStatus } from "../types/job";
import { APP_CONFIG } from "../core/config";

export interface UseJobStatusResult {
  job?: ProcessingJob;
  status?: JobStatus;
  isLoading: boolean;
  isFetching: boolean;
  isError: boolean;
  error: Error | null;
  isTerminal: boolean;
  isCompleted: boolean;
  isFailed: boolean;
  isProcessing: boolean;
  isQueued: boolean;
  refetch: () => void;
}

export function useJobStatus(jobId: string | undefined): UseJobStatusResult {
  const isEnabled = Boolean(jobId && jobId !== "preview" && jobId !== "audit");

  const query = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => jobService.getJob(jobId!),
    enabled: isEnabled,
    refetchInterval: (q) => {
      const currentStatus = q.state.data?.status;
      if (currentStatus === "QUEUED" || currentStatus === "PROCESSING") {
        return APP_CONFIG.jobPollingIntervalMs || 1500;
      }
      return false; // Stop polling on terminal states (COMPLETED, FAILED)
    },
    refetchIntervalInBackground: false,
    retry: 2,
    staleTime: 1000,
  });

  const job = query.data;
  const status = job?.status;
  const isTerminal = status === "COMPLETED" || status === "FAILED";

  return {
    job,
    status,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    isError: query.isError,
    error: query.error as Error | null,
    isTerminal,
    isCompleted: status === "COMPLETED",
    isFailed: status === "FAILED",
    isProcessing: status === "PROCESSING",
    isQueued: status === "QUEUED",
    refetch: () => query.refetch(),
  };
}
