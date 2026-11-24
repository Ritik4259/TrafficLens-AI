/**
 * TanStack Query hooks for Evidence-Grounded Traffic Incident Reports
 *
 * Provides:
 * - useIncidentReport: queries the latest report for a job (handles 404 as null)
 * - useIncidentReportHistory: audit trail of historical reports
 * - useIncidentReportById: queries a specific report by ID
 * - useGenerateIncidentReport: mutation to trigger agent report generation
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { incidentReportService } from "../services/incidentReportService";
import {
  IncidentReportResponse,
  IncidentReportHistoryResponse,
} from "../types/incidentReport";
import { QUERY_KEYS } from "../core/queryKeys";

export interface UseIncidentReportResult {
  data: IncidentReportResponse | null;
  isLoading: boolean;
  isFetching: boolean;
  isError: boolean;
  error: Error | null;
  hasReport: boolean;
  isCompleted: boolean;
  isFailed: boolean;
  refetch: () => void;
}

/**
 * Queries the latest incident report for a job.
 * Treats backend HTTP 404 cleanly as a valid empty state (null) rather than an error banner.
 */
export function useIncidentReport(
  jobId: string | undefined
): UseIncidentReportResult {
  const isEnabled = Boolean(jobId && jobId !== "preview" && jobId !== "audit");

  const query = useQuery<IncidentReportResponse | null>({
    queryKey: QUERY_KEYS.incidentReport.latest(jobId),
    queryFn: async () => {
      try {
        return await incidentReportService.getLatestReport(jobId!);
      } catch (err: any) {
        // Backend returns 404 when no incident report has been generated yet
        if (err?.status === 404) {
          return null;
        }
        throw err;
      }
    },
    enabled: isEnabled,
    staleTime: Infinity, // Completed incident reports are immutable
    retry: false, // Do not retry 404 or generation errors automatically
  });

  const reportResponse = query.data ?? null;
  const isCompleted = reportResponse?.status === "COMPLETED";
  const isFailed = reportResponse?.status === "FAILED";

  return {
    data: reportResponse,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    isError: query.isError,
    error: query.error as Error | null,
    hasReport: Boolean(reportResponse && isCompleted),
    isCompleted,
    isFailed,
    refetch: () => query.refetch(),
  };
}

/**
 * Queries all historical incident reports generated for a job.
 */
export function useIncidentReportHistory(
  jobId: string | undefined,
  ascending: boolean = true
) {
  const isEnabled = Boolean(jobId && jobId !== "preview" && jobId !== "audit");

  return useQuery<IncidentReportHistoryResponse>({
    queryKey: QUERY_KEYS.incidentReport.history(jobId, ascending),
    queryFn: () => incidentReportService.listReportsForJob(jobId!, ascending),
    enabled: isEnabled,
    staleTime: 10000,
  });
}

/**
 * Queries a specific incident report by its unique ID.
 */
export function useIncidentReportById(reportId: string | undefined) {
  const isEnabled = Boolean(reportId);

  return useQuery<IncidentReportResponse>({
    queryKey: QUERY_KEYS.incidentReport.byId(reportId),
    queryFn: () => incidentReportService.getReportById(reportId!),
    enabled: isEnabled,
    staleTime: Infinity,
  });
}

/**
 * Mutation to trigger generation of an evidence-grounded incident report.
 * Updates query cache with returned report and invalidates report history.
 */
export function useGenerateIncidentReport(jobId: string | undefined) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => {
      if (!jobId) {
        throw new Error("Cannot generate incident report without a valid jobId.");
      }
      return incidentReportService.generateReport(jobId);
    },
    onSuccess: (data) => {
      // Prime the latest report cache with the returned report
      queryClient.setQueryData(QUERY_KEYS.incidentReport.latest(jobId), data);
      queryClient.invalidateQueries({
        queryKey: ["incidentReportHistory", jobId],
      });
    },
  });
}
