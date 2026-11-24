/**
 * Service client for Evidence-Grounded Traffic Incident Reports
 * Interacts with backend /api/v1/jobs/{job_id}/incident-reports and /api/v1/incident-reports
 */

import { apiClient } from "./apiClient";
import {
  IncidentReportHistoryResponse,
  IncidentReportResponse,
} from "../types/incidentReport";

export const incidentReportService = {
  /**
   * Triggers the LangGraph incident agent workflow to generate a grounded report.
   */
  async generateReport(jobId: string): Promise<IncidentReportResponse> {
    return apiClient.post<IncidentReportResponse>(
      `/jobs/${encodeURIComponent(jobId)}/incident-reports`
    );
  },

  /**
   * Retrieves the latest incident report for a job.
   * Note: Throws 404 ApiError if no report has been generated yet for this job.
   */
  async getLatestReport(jobId: string): Promise<IncidentReportResponse> {
    return apiClient.get<IncidentReportResponse>(
      `/jobs/${encodeURIComponent(jobId)}/incident-report`
    );
  },

  /**
   * Retrieves a specific incident report by its unique ID.
   */
  async getReportById(reportId: string): Promise<IncidentReportResponse> {
    return apiClient.get<IncidentReportResponse>(
      `/incident-reports/${encodeURIComponent(reportId)}`
    );
  },

  /**
   * Retrieves complete historical audit trail of incident reports for a job.
   */
  async listReportsForJob(
    jobId: string,
    ascending: boolean = true
  ): Promise<IncidentReportHistoryResponse> {
    return apiClient.get<IncidentReportHistoryResponse>(
      `/jobs/${encodeURIComponent(jobId)}/incident-reports?ascending=${ascending}`
    );
  },
};
