import { describe, it, expect, vi, beforeEach } from "vitest";
import { interpretationService } from "../services/interpretationService";
import { incidentReportService } from "../services/incidentReportService";
import { apiClient } from "../services/apiClient";
import { QUERY_KEYS } from "../core/queryKeys";
import { APP_CONFIG } from "../core/config";
import {
  SceneInterpretationResponse,
  SceneInterpretationLifecycleStatus,
} from "../types/interpretation";
import { IncidentReportResponse } from "../types/incidentReport";

describe("Milestone 7A: Multimodal AI Services and Hooks Logic", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  // ---------------------------------------------------------------------------
  // 1. Interpretation Service Endpoints
  // ---------------------------------------------------------------------------
  describe("interpretationService", () => {
    it("requestInterpretation posts to /jobs/{jobId}/interpretations with config payload", async () => {
      const postSpy = vi.spyOn(apiClient, "post").mockResolvedValue({
        id: "interp_123",
        job_id: "job_test",
        status: "QUEUED",
      } as SceneInterpretationResponse);

      const payload = { frame_budget: 7, temperature: 0.3 };
      const res = await interpretationService.requestInterpretation("job_test", payload);

      expect(postSpy).toHaveBeenCalledWith("/jobs/job_test/interpretations", payload);
      expect(res.status).toBe("QUEUED");
      expect(res.id).toBe("interp_123");
    });

    it("getLatestInterpretation fetches /jobs/{jobId}/interpretation", async () => {
      const getSpy = vi.spyOn(apiClient, "get").mockResolvedValue({
        id: null,
        job_id: "job_test",
        status: "NOT_REQUESTED",
      } as SceneInterpretationResponse);

      const res = await interpretationService.getLatestInterpretation("job_test");
      expect(getSpy).toHaveBeenCalledWith("/jobs/job_test/interpretation");
      expect(res.status).toBe("NOT_REQUESTED");
      expect(res.id).toBeNull();
    });

    it("getInterpretationById fetches /interpretations/{id}", async () => {
      const getSpy = vi.spyOn(apiClient, "get").mockResolvedValue({
        id: "interp_abc",
        job_id: "job_test",
        status: "COMPLETED",
      } as SceneInterpretationResponse);

      const res = await interpretationService.getInterpretationById("interp_abc");
      expect(getSpy).toHaveBeenCalledWith("/interpretations/interp_abc");
      expect(res.status).toBe("COMPLETED");
    });

    it("getInterpretationHistory queries /jobs/{jobId}/interpretations with ascending param", async () => {
      const getSpy = vi.spyOn(apiClient, "get").mockResolvedValue({
        job_id: "job_test",
        total_attempts: 2,
        attempts: [],
      });

      await interpretationService.getInterpretationHistory("job_test", false);
      expect(getSpy).toHaveBeenCalledWith("/jobs/job_test/interpretations?ascending=false");
    });

    it("retryInterpretation posts to /jobs/{jobId}/interpretations/retry", async () => {
      const postSpy = vi.spyOn(apiClient, "post").mockResolvedValue({
        id: "interp_retry",
        job_id: "job_test",
        status: "QUEUED",
      } as SceneInterpretationResponse);

      const res = await interpretationService.retryInterpretation("job_test");
      expect(postSpy).toHaveBeenCalledWith("/jobs/job_test/interpretations/retry");
      expect(res.status).toBe("QUEUED");
    });

    it("retryInterpretationById posts to /interpretations/{id}/retry", async () => {
      const postSpy = vi.spyOn(apiClient, "post").mockResolvedValue({
        id: "interp_retry_2",
        job_id: "job_test",
        status: "QUEUED",
      } as SceneInterpretationResponse);

      const res = await interpretationService.retryInterpretationById("interp_old");
      expect(postSpy).toHaveBeenCalledWith("/interpretations/interp_old/retry");
      expect(res.status).toBe("QUEUED");
    });
  });

  // ---------------------------------------------------------------------------
  // 2. Incident Report Service Endpoints
  // ---------------------------------------------------------------------------
  describe("incidentReportService", () => {
    it("generateReport posts to /jobs/{jobId}/incident-reports", async () => {
      const postSpy = vi.spyOn(apiClient, "post").mockResolvedValue({
        report_id: "rep_123",
        job_id: "job_test",
        status: "COMPLETED",
        validation_passed: true,
      } as IncidentReportResponse);

      const res = await incidentReportService.generateReport("job_test");
      expect(postSpy).toHaveBeenCalledWith("/jobs/job_test/incident-reports");
      expect(res.status).toBe("COMPLETED");
      expect(res.report_id).toBe("rep_123");
    });

    it("getLatestReport fetches /jobs/{jobId}/incident-report", async () => {
      const getSpy = vi.spyOn(apiClient, "get").mockResolvedValue({
        report_id: "rep_latest",
        job_id: "job_test",
        status: "COMPLETED",
        validation_passed: true,
      } as IncidentReportResponse);

      const res = await incidentReportService.getLatestReport("job_test");
      expect(getSpy).toHaveBeenCalledWith("/jobs/job_test/incident-report");
      expect(res.report_id).toBe("rep_latest");
    });

    it("getReportById fetches /incident-reports/{reportId}", async () => {
      const getSpy = vi.spyOn(apiClient, "get").mockResolvedValue({
        report_id: "rep_target",
        job_id: "job_test",
        status: "COMPLETED",
        validation_passed: true,
      } as IncidentReportResponse);

      const res = await incidentReportService.getReportById("rep_target");
      expect(getSpy).toHaveBeenCalledWith("/incident-reports/rep_target");
      expect(res.report_id).toBe("rep_target");
    });

    it("listReportsForJob queries /jobs/{jobId}/incident-reports with sorting param", async () => {
      const getSpy = vi.spyOn(apiClient, "get").mockResolvedValue({
        job_id: "job_test",
        total_reports: 3,
        reports: [],
      });

      await incidentReportService.listReportsForJob("job_test", true);
      expect(getSpy).toHaveBeenCalledWith("/jobs/job_test/incident-reports?ascending=true");
    });
  });

  // ---------------------------------------------------------------------------
  // 3. Polling Logic and Termination Rules
  // ---------------------------------------------------------------------------
  describe("Scene Interpretation Polling Rules", () => {
    function getInterpretationPollingInterval(
      status?: SceneInterpretationLifecycleStatus
    ): number | false {
      if (status === "QUEUED" || status === "PROCESSING") {
        return APP_CONFIG.interpretationPollingIntervalMs || 2000;
      }
      return false; // Stop polling on terminal states and NOT_REQUESTED
    }

    it("schedules polling at ~2000ms while interpretation is active (QUEUED or PROCESSING)", () => {
      expect(getInterpretationPollingInterval("QUEUED")).toBe(2000);
      expect(getInterpretationPollingInterval("PROCESSING")).toBe(2000);
    });

    it("disables polling (returns false) when interpretation enters terminal state", () => {
      expect(getInterpretationPollingInterval("COMPLETED")).toBe(false);
      expect(getInterpretationPollingInterval("FAILED")).toBe(false);
    });

    it("disables polling when interpretation is in virtual NOT_REQUESTED state", () => {
      expect(getInterpretationPollingInterval("NOT_REQUESTED")).toBe(false);
      expect(getInterpretationPollingInterval(undefined)).toBe(false);
    });

    it("derives correct boolean flags for lifecycle states", () => {
      const checkFlags = (status?: SceneInterpretationLifecycleStatus) => ({
        isNotRequested: status === "NOT_REQUESTED",
        isQueued: status === "QUEUED",
        isProcessing: status === "PROCESSING",
        isCompleted: status === "COMPLETED",
        isFailed: status === "FAILED",
        isActive: status === "QUEUED" || status === "PROCESSING",
        isTerminal: status === "COMPLETED" || status === "FAILED",
      });

      expect(checkFlags("QUEUED")).toMatchObject({
        isActive: true,
        isTerminal: false,
        isQueued: true,
      });

      expect(checkFlags("PROCESSING")).toMatchObject({
        isActive: true,
        isTerminal: false,
        isProcessing: true,
      });

      expect(checkFlags("COMPLETED")).toMatchObject({
        isActive: false,
        isTerminal: true,
        isCompleted: true,
      });

      expect(checkFlags("FAILED")).toMatchObject({
        isActive: false,
        isTerminal: true,
        isFailed: true,
      });

      expect(checkFlags("NOT_REQUESTED")).toMatchObject({
        isActive: false,
        isTerminal: false,
        isNotRequested: true,
      });
    });
  });

  // ---------------------------------------------------------------------------
  // 4. Incident Report 404 Handling (Clean Empty State)
  // ---------------------------------------------------------------------------
  describe("Incident Report 404 Empty State Handling", () => {
    it("handles 404 ApiError as null empty state instead of throwing an unhandled exception", async () => {
      const fetchLatestSafe = async (jobId: string) => {
        try {
          return await incidentReportService.getLatestReport(jobId);
        } catch (err: any) {
          if (err?.status === 404) {
            return null;
          }
          throw err;
        }
      };

      vi.spyOn(incidentReportService, "getLatestReport").mockRejectedValue({
        status: 404,
        message: "No incident report has been generated for job 'job_test'.",
      });

      const result = await fetchLatestSafe("job_test");
      expect(result).toBeNull();
    });

    it("re-throws non-404 errors (500, 502, 504) as expected", async () => {
      const fetchLatestSafe = async (jobId: string) => {
        try {
          return await incidentReportService.getLatestReport(jobId);
        } catch (err: any) {
          if (err?.status === 404) {
            return null;
          }
          throw err;
        }
      };

      vi.spyOn(incidentReportService, "getLatestReport").mockRejectedValue({
        status: 504,
        message: "Incident report generation timed out",
      });

      await expect(fetchLatestSafe("job_test")).rejects.toMatchObject({
        status: 504,
      });
    });
  });

  // ---------------------------------------------------------------------------
  // 5. Query Keys and Invalidation Targets
  // ---------------------------------------------------------------------------
  describe("Centralized Query Keys", () => {
    it("generates stable, consistent query keys for interpretation", () => {
      expect(QUERY_KEYS.interpretation.latest("job_1")).toEqual([
        "sceneInterpretation",
        "job_1",
      ]);
      expect(QUERY_KEYS.interpretation.byId("interp_1")).toEqual([
        "sceneInterpretationById",
        "interp_1",
      ]);
      expect(QUERY_KEYS.interpretation.history("job_1", true)).toEqual([
        "sceneInterpretationHistory",
        "job_1",
        { ascending: true },
      ]);
    });

    it("generates stable, consistent query keys for incident reports", () => {
      expect(QUERY_KEYS.incidentReport.latest("job_1")).toEqual([
        "incidentReport",
        "job_1",
      ]);
      expect(QUERY_KEYS.incidentReport.byId("rep_1")).toEqual([
        "incidentReportById",
        "rep_1",
      ]);
      expect(QUERY_KEYS.incidentReport.history("job_1", false)).toEqual([
        "incidentReportHistory",
        "job_1",
        { ascending: false },
      ]);
    });
  });
});
