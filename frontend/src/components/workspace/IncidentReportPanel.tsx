import React, { useState } from "react";
import {
  FileText,
  RotateCw,
  AlertTriangle,
  Eye,
  Layers,
  ShieldAlert,
  Navigation,
  CheckCircle2,
  Clock,
  Sparkles,
  Activity,
  History,
  AlertCircle,
  Info,
} from "lucide-react";
import { Card } from "../common/Card";
import { Button } from "../common/Button";
import { EmptyState } from "../feedback/EmptyState";
import { ErrorAlert } from "../feedback/ErrorAlert";
import { LoadingSkeleton } from "../feedback/LoadingSkeleton";
import {
  useIncidentReport,
  useIncidentReportHistory,
  useIncidentReportById,
  useGenerateIncidentReport,
} from "../../hooks/useIncidentReport";
import {
  IncidentReport,
  IncidentReportResponse,
  ReportConfidence,
} from "../../types/incidentReport";

export interface IncidentReportPanelProps {
  jobId?: string;
  currentTimeSeconds: number;
  onSeek: (timeSeconds: number) => void;
}

export const IncidentReportPanel: React.FC<IncidentReportPanelProps> = ({
  jobId,
  currentTimeSeconds,
  onSeek,
}) => {
  const {
    data: latestReportResponse,
    isLoading: isLatestLoading,
    isError: isLatestError,
    error: latestError,
    refetch: refetchLatest,
  } = useIncidentReport(jobId);

  const {
    data: historyData,
  } = useIncidentReportHistory(jobId);

  const generateMutation = useGenerateIncidentReport(jobId);

  // Selected historical report version (if null, display latest)
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null);

  // If a historical report is selected, load its detailed data if needed
  const {
    data: historicalReportResponse,
    isLoading: isHistoricalLoading,
  } = useIncidentReportById(selectedReportId || undefined);

  // Determine active report response
  const activeReportResponse: IncidentReportResponse | null = selectedReportId
    ? historicalReportResponse ||
      historyData?.reports.find((r) => r.report_id === selectedReportId) ||
      null
    : latestReportResponse;

  const isHistoricalView = Boolean(selectedReportId && selectedReportId !== latestReportResponse?.report_id);
  const report: IncidentReport | undefined = activeReportResponse?.report ?? undefined;
  const isCompleted = activeReportResponse?.status === "COMPLETED";
  const isFailed = activeReportResponse?.status === "FAILED";

  // 1. Initial Query Loading Skeleton
  if ((isLatestLoading && !latestReportResponse) || isHistoricalLoading) {
    return (
      <div className="space-y-4 font-mono text-xs" data-testid="incident-report-loading">
        <LoadingSkeleton className="h-16 w-full" />
        <LoadingSkeleton className="h-32 w-full" />
        <LoadingSkeleton className="h-48 w-full" />
      </div>
    );
  }

  // 2. Query Error State
  if (isLatestError) {
    return (
      <div className="space-y-3 font-mono text-xs" data-testid="incident-report-query-error">
        <ErrorAlert
          title="Incident Report Service Inaccessible"
          message={latestError?.message || "Failed to retrieve incident report status from backend."}
        />
        <Button variant="outline" size="sm" onClick={() => refetchLatest()}>
          Retry Status Query
        </Button>
      </div>
    );
  }

  // 3. In-flight Generation Loading State
  if (generateMutation.isPending) {
    return (
      <div
        className="p-8 rounded-xl bg-[#0E1524] border border-cyan-800/60 font-mono text-xs text-center space-y-4 shadow-xl"
        data-testid="incident-report-generating"
      >
        <div className="w-12 h-12 rounded-xl bg-cyan-950/80 border border-cyan-600/60 mx-auto flex items-center justify-center animate-spin">
          <RotateCw className="w-6 h-6 text-cyan-400" />
        </div>
        <div>
          <h3 className="text-base font-bold text-white tracking-wide">
            Synthesizing Telemetry & Grounding Claims...
          </h3>
          <p className="text-xs text-[#9CA3AF] max-w-md mx-auto mt-1 leading-relaxed">
            The LangGraph incident agent is assembling CV analytics, multimodal scene interpretations,
            and executing epistemic grounding validation.
          </p>
        </div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded bg-[#161F30] border border-[#2B3B55] text-cyan-300 text-[11px]">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
          <span>Synchronous agent execution in progress (expected ~10–25s)</span>
        </div>
      </div>
    );
  }

  // 4. Empty State (No Report Generated Yet)
  if (!activeReportResponse) {
    return (
      <div className="space-y-4 font-mono text-xs" data-testid="incident-report-empty-state">
        {generateMutation.isError && (
          <ErrorAlert
            title="Report Generation Failed"
            message={
              generateMutation.error?.message ||
              "Failed to generate incident report. Ensure CV processing has completed."
            }
            onRetry={() => generateMutation.mutate()}
          />
        )}
        <EmptyState
          icon={<FileText className="w-6 h-6 text-cyan-400" />}
          title="No Incident Report Generated"
          description="Generate an evidence-grounded traffic incident report synthesizing quantitative CV detections, tracking telemetry, and multimodal visual context."
          actionLabel="Generate Incident Report"
          onAction={() => generateMutation.mutate()}
        />
        <div className="p-3 rounded-lg bg-[#0E1524]/60 border border-[#1F2937] text-[11px] text-[#9CA3AF] flex items-start gap-2">
          <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
          <div>
            <strong>Epistemic Grounding Guarantee:</strong> The report agent references only measured
            CV telemetry and validated keyframe observations. Unsupported collision accusations, uncalibrated
            speeds, and ungrounded claims are strictly suppressed.
          </div>
        </div>
      </div>
    );
  }

  // 5. Failed Report State
  if (isFailed && !report) {
    return (
      <div className="space-y-4 font-mono text-xs" data-testid="incident-report-failed">
        <ErrorAlert
          title="Incident Report Generation Failed"
          message={
            activeReportResponse.error_message ||
            "The LangGraph agent encountered an error during synthesis or validation."
          }
          errorCode="REPORT_AGENT_FAILED"
          onRetry={() => generateMutation.mutate()}
        />
        <div className="flex items-center gap-3">
          <Button
            variant="primary"
            size="sm"
            onClick={() => generateMutation.mutate()}
            icon={<RotateCw className="w-3.5 h-3.5" />}
          >
            Retry Report Generation
          </Button>
          {isHistoricalView && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSelectedReportId(null)}
            >
              Back to Latest
            </Button>
          )}
        </div>
      </div>
    );
  }

  // Helper for Confidence Badge
  const renderConfidenceBadge = (confidence?: ReportConfidence) => {
    switch (confidence) {
      case "high":
        return (
          <span className="px-2.5 py-0.5 rounded text-[10px] font-bold uppercase bg-emerald-950/80 text-emerald-300 border border-emerald-700/60 flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3 text-emerald-400" />
            High Confidence
          </span>
        );
      case "medium":
        return (
          <span className="px-2.5 py-0.5 rounded text-[10px] font-bold uppercase bg-cyan-950/80 text-cyan-300 border border-cyan-700/60 flex items-center gap-1">
            <Activity className="w-3 h-3 text-cyan-400" />
            Medium Confidence
          </span>
        );
      case "low":
        return (
          <span className="px-2.5 py-0.5 rounded text-[10px] font-bold uppercase bg-amber-950/80 text-amber-300 border border-amber-700/60 flex items-center gap-1">
            <AlertTriangle className="w-3 h-3 text-amber-400" />
            Low Confidence
          </span>
        );
      case "cautionary":
      default:
        return (
          <span className="px-2.5 py-0.5 rounded text-[10px] font-bold uppercase bg-red-950/80 text-red-300 border border-red-700/60 flex items-center gap-1">
            <ShieldAlert className="w-3 h-3 text-red-400" />
            Cautionary
          </span>
        );
    }
  };

  // Helper for Priority styling
  const renderPriorityBadge = (priority: string) => {
    const p = priority.toLowerCase();
    if (p === "high") {
      return (
        <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-red-950/80 text-red-300 border border-red-700/60">
          High Priority
        </span>
      );
    }
    if (p === "medium") {
      return (
        <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-amber-950/80 text-amber-300 border border-amber-700/60">
          Medium Priority
        </span>
      );
    }
    return (
      <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-blue-950/80 text-blue-300 border border-blue-700/60">
        Low Priority
      </span>
    );
  };

  const totalReports = historyData?.total_reports ?? (latestReportResponse ? 1 : 0);

  return (
    <div className="space-y-4 font-mono text-xs" data-testid="incident-report-panel">
      {/* 1. TOP HEADER & METADATA BAR */}
      <div className="p-3.5 rounded-lg bg-[#0E1524] border border-[#1F2937] flex flex-wrap items-center justify-between gap-3 shadow-md">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <FileText className="w-4 h-4 text-cyan-400" />
            <span className="font-bold text-base text-white">Traffic Incident Report</span>
            {report?.confidence_classification &&
              renderConfidenceBadge(report.confidence_classification)}
            <span
              className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase border ${
                isCompleted
                  ? "bg-emerald-950/80 text-emerald-400 border-emerald-800"
                  : "bg-red-950/80 text-red-400 border-red-800"
              }`}
            >
              {activeReportResponse.status}
            </span>
            {activeReportResponse.validation_passed && (
              <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                Grounding Passed
              </span>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-3 text-[11px] text-[#9CA3AF]">
            {activeReportResponse.report_id && (
              <span>
                Report ID: <code className="text-cyan-300 font-bold">{activeReportResponse.report_id}</code>
              </span>
            )}
            {report?.generated_at && (
              <span>
                Generated: <strong className="text-white">{new Date(report.generated_at).toLocaleString()}</strong>
              </span>
            )}
            {isHistoricalView && (
              <span className="px-2 py-0.5 rounded bg-purple-950/80 text-purple-300 border border-purple-700/60 font-semibold">
                Historical Version Viewed
              </span>
            )}
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-[10px] text-[#9CA3AF]">
            <Clock className="w-3 h-3 text-cyan-400" />
            <span>Time: <strong className="text-cyan-400">{currentTimeSeconds.toFixed(2)}s</strong></span>
          </div>
          {isHistoricalView && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSelectedReportId(null)}
            >
              View Latest
            </Button>
          )}
          <Button
            variant="primary"
            size="sm"
            onClick={() => generateMutation.mutate()}
            disabled={generateMutation.isPending}
            icon={<RotateCw className={`w-3.5 h-3.5 ${generateMutation.isPending ? "animate-spin" : ""}`} />}
          >
            {generateMutation.isPending ? "Generating..." : "Regenerate Report"}
          </Button>
        </div>
      </div>

      {/* Generation Error Banner if any */}
      {generateMutation.isError && (
        <ErrorAlert
          title="Regeneration Failed"
          message={generateMutation.error?.message || "Failed to generate report."}
          onRetry={() => generateMutation.mutate()}
        />
      )}

      {/* 2. REPORT VERSION HISTORY SELECTOR */}
      {historyData && historyData.reports.length > 1 && (
        <div className="p-2.5 rounded-lg bg-[#0B0F17] border border-[#1F2937] flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-[11px] text-[#9CA3AF]">
            <History className="w-3.5 h-3.5 text-cyan-400" />
            <span>Audit Versions: <strong className="text-white">{totalReports}</strong> available</span>
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            {historyData.reports.map((r, idx) => {
              const isSelected = selectedReportId
                ? r.report_id === selectedReportId
                : r.report_id === latestReportResponse?.report_id;
              const versionLabel = `v${idx + 1}${idx === historyData.reports.length - 1 ? " (Latest)" : ""}`;
              return (
                <button
                  key={r.report_id}
                  type="button"
                  onClick={() => setSelectedReportId(r.report_id)}
                  className={`px-2.5 py-1 rounded text-[10px] font-bold transition-all border ${
                    isSelected
                      ? "bg-cyan-900/60 text-cyan-200 border-cyan-500 shadow-sm"
                      : "bg-[#161F30] text-[#9CA3AF] border-[#2B3B55] hover:text-white hover:bg-[#1E293B]"
                  }`}
                >
                  {versionLabel}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* 3. EXECUTIVE SUMMARY */}
      {report?.executive_summary && (
        <Card className="p-4 rounded-xl bg-[#0E1524] border border-[#1F2937] space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <span>Executive Summary</span>
            </h4>
            <span className="text-[10px] text-[#6B7280] uppercase tracking-wider font-semibold">
              Evidence-Grounded Synthesis
            </span>
          </div>
          <p className="text-xs text-[#E5E7EB] leading-relaxed bg-[#0B0F17]/80 p-3 rounded-lg border border-[#1F2937]/80">
            {report.executive_summary}
          </p>
          <div className="flex items-center gap-2 p-2 rounded bg-amber-950/20 border border-amber-800/40 text-[10px] text-amber-300/90">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
            <span>
              <strong>Epistemic Notice:</strong> AI-generated synthesis of available evidence. This summary is
              produced by an automated agent pipeline and does not constitute an independently certified incident determination.
            </span>
          </div>
        </Card>
      )}

      {/* 4. TRAFFIC CONDITIONS (STRUCTURED BASELINE) */}
      {report?.traffic_conditions && (
        <Card className="p-4 rounded-xl bg-[#0E1524] border border-[#1F2937] space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>Traffic Conditions Baseline</span>
            </h4>
            <span className="text-[10px] text-[#9CA3AF]">
              Sequence: <strong className="text-white">{report.traffic_conditions.sequence_name}</strong>
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            <div className="p-2.5 rounded-lg bg-[#0B0F17] border border-[#1F2937] space-y-1">
              <div className="text-[10px] text-[#9CA3AF] uppercase">Video Duration</div>
              <div className="text-sm font-bold text-white">
                {report.traffic_conditions.duration_s.toFixed(1)}s
              </div>
            </div>

            <div className="p-2.5 rounded-lg bg-[#0B0F17] border border-[#1F2937] space-y-1">
              <div className="text-[10px] text-[#9CA3AF] uppercase">Frame Detections</div>
              <div className="text-sm font-bold text-cyan-300">
                {report.traffic_conditions.total_detections.toLocaleString()}
              </div>
              <div className="text-[9px] text-[#6B7280]">Bounding box detections</div>
            </div>

            <div className="p-2.5 rounded-lg bg-[#0B0F17] border border-[#1F2937] space-y-1">
              <div className="text-[10px] text-[#9CA3AF] uppercase">Tracked Instances</div>
              <div className="text-sm font-bold text-purple-300">
                {report.traffic_conditions.total_tracked_instances.toLocaleString()}
              </div>
              <div className="text-[9px] text-[#6B7280]">Tracker identities (not physical census)</div>
            </div>

            <div className="p-2.5 rounded-lg bg-[#0B0F17] border border-[#1F2937] space-y-1">
              <div className="text-[10px] text-[#9CA3AF] uppercase">Peak Spatial Occupancy</div>
              <div className="text-sm font-bold text-amber-300">
                {report.traffic_conditions.peak_occupancy_ratio != null
                  ? `${(report.traffic_conditions.peak_occupancy_ratio * 100).toFixed(1)}%`
                  : "N/A"}
              </div>
              <div className="text-[9px] text-[#6B7280]">Max ROI area ratio</div>
            </div>
          </div>

          {/* Flow rate & Multimodal Context row */}
          <div className="flex flex-wrap items-center justify-between gap-2 p-2.5 rounded-lg bg-[#0B0F17] border border-[#1F2937] text-[11px]">
            <div className="flex items-center gap-3">
              <span>
                Overall Flow Rate:{" "}
                <strong className="text-cyan-400">
                  {report.traffic_conditions.overall_flow_rate_vpm != null
                    ? `${report.traffic_conditions.overall_flow_rate_vpm.toFixed(1)} VPM`
                    : "Not calculated"}
                </strong>
              </span>
            </div>

            <div className="flex items-center gap-2">
              {report.traffic_conditions.visual_context_available ? (
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950/80 text-emerald-300 border border-emerald-700/60 flex items-center gap-1">
                  <Eye className="w-3 h-3 text-emerald-400" />
                  Multimodal VLM Context Active
                </span>
              ) : (
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-[#1F2937] text-[#9CA3AF] border border-[#374151] flex items-center gap-1">
                  CV Telemetry Only
                </span>
              )}
            </div>
          </div>

          {report.traffic_conditions.visual_context_note && (
            <p className="text-[10px] text-[#9CA3AF] italic">
              {report.traffic_conditions.visual_context_note}
            </p>
          )}

          {/* Environmental Context if provided */}
          {report.traffic_conditions.environmental_context && (
            <div className="p-2.5 rounded-lg bg-[#0B0F17]/90 border border-[#1F2937] space-y-1.5">
              <span className="text-[10px] text-[#9CA3AF] uppercase font-semibold">
                Environmental & Ambient Parameters
              </span>
              <div className="flex flex-wrap gap-2 text-[10px]">
                {Object.entries(report.traffic_conditions.environmental_context).map(([k, v]) => (
                  <span
                    key={k}
                    className="px-2 py-0.5 rounded bg-[#161F30] text-[#D1D5DB] border border-[#2B3B55]"
                  >
                    <strong>{k.replace(/_/g, " ")}:</strong> {String(v)}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="p-2 rounded bg-[#0B0F17] text-[10px] text-[#6B7280]">
            <strong>Measurement Provenance:</strong> Numerical metrics are directly measured by YOLO11n + ByteTrack.
            Tracker instances represent assigned tracklet IDs and may exceed unique vehicles due to tracker fragmentation or occlusions.
            Speeds are strictly uncalibrated (pixel space).
          </div>
        </Card>
      )}

      {/* 5. CONGESTION OBSERVATIONS */}
      {report?.congestion_observations && report.congestion_observations.length > 0 && (
        <Card className="p-4 rounded-xl bg-[#0E1524] border border-[#1F2937] space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-amber-400" />
              <span>Congestion & Spatial Occupancy Observations</span>
            </h4>
            <span className="text-[10px] text-[#9CA3AF]">
              {report.congestion_observations.length} Zones Evaluated
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
            {report.congestion_observations.map((obs, idx) => (
              <div
                key={`congestion-${idx}`}
                className="p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937] space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-white text-xs">{obs.roi_id}</span>
                  <div className="flex items-center gap-1.5">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                        obs.observed_level.toLowerCase() === "high"
                          ? "bg-red-950/80 text-red-300 border border-red-700/60"
                          : obs.observed_level.toLowerCase() === "moderate"
                          ? "bg-amber-950/80 text-amber-300 border border-amber-700/60"
                          : "bg-cyan-950/80 text-cyan-300 border border-cyan-700/60"
                      }`}
                    >
                      {obs.observed_level}
                    </span>
                    {obs.is_heuristic_rule && (
                      <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-[#1F2937] text-[#9CA3AF] border border-[#374151]">
                        EXPLAINABLE HEURISTIC
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-3 text-[11px] text-[#9CA3AF]">
                  <span>
                    Peak Occupancy: <strong className="text-amber-400">{(obs.peak_occupancy * 100).toFixed(1)}%</strong>
                  </span>
                  {obs.average_occupancy != null && (
                    <span>
                      Avg: <strong className="text-white">{(obs.average_occupancy * 100).toFixed(1)}%</strong>
                    </span>
                  )}
                </div>

                <p className="text-[11px] text-[#D1D5DB] leading-relaxed">
                  {obs.description}
                </p>
              </div>
            ))}
          </div>

          <p className="text-[10px] text-[#6B7280]">
            * Congestion levels are calculated via 2D bounding box spatial occupancy heuristic rules,
            not certified highway capacity determinations.
          </p>
        </Card>
      )}

      {/* 6. TRAFFIC EVENTS & STOPPED VEHICLES */}
      {((report?.observed_events && report.observed_events.length > 0) ||
        (report?.stopped_vehicles && report.stopped_vehicles.length > 0)) && (
        <Card className="p-4 rounded-xl bg-[#0E1524] border border-[#1F2937] space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-400" />
              <span>Traffic Events & Stationary Vehicles</span>
            </h4>
            <span className="text-[10px] text-[#9CA3AF]">
              {report?.stopped_vehicles?.length ?? 0} Stationary Records • {report?.observed_events?.length ?? 0} Events
            </span>
          </div>

          {/* Stopped Vehicles Items */}
          {report?.stopped_vehicles && report.stopped_vehicles.length > 0 && (
            <div className="space-y-2">
              <span className="text-[10px] text-[#9CA3AF] uppercase font-semibold">
                Documented Stationary Vehicle Incidents
              </span>
              <div className="space-y-2">
                {report.stopped_vehicles.map((item, idx) => (
                  <div
                    key={`stopped-${idx}`}
                    className="p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937] hover:border-red-800/60 transition-colors space-y-2"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-red-300">Track #{item.track_id}</span>
                        <span className="text-[10px] text-[#9CA3AF]">({item.event_id})</span>
                        <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                          {item.verification_status}
                        </span>
                        {item.roi_id && (
                          <span className="text-[10px] text-[#9CA3AF]">in {item.roi_id}</span>
                        )}
                      </div>

                      <button
                        type="button"
                        onClick={() => onSeek(item.start_timestamp_s)}
                        className="px-2 py-0.5 rounded bg-red-950/60 hover:bg-red-900/60 text-red-300 border border-red-800/80 flex items-center gap-1 text-[10px] font-semibold transition-colors"
                      >
                        <Navigation className="w-3 h-3 rotate-45" />
                        <span>Seek to t={item.start_timestamp_s.toFixed(1)}s</span>
                      </button>
                    </div>

                    <div className="text-[11px] text-[#9CA3AF]">
                      Stationary duration: <strong className="text-white">{item.duration_s.toFixed(1)}s</strong> •
                      Start: {item.start_timestamp_s.toFixed(2)}s
                      {item.location && (
                        <span> • Location: [{item.location[0].toFixed(0)}, {item.location[1].toFixed(0)}] px</span>
                      )}
                    </div>

                    {item.probable_cause && (
                      <div className="p-2 rounded bg-[#161F30] border border-[#2B3B55] text-[11px] space-y-1">
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] font-bold text-amber-300">Interpreted Probable Cause:</span>
                          <span className="px-1.5 py-0.2 rounded text-[9px] font-bold bg-amber-950/80 text-amber-300 border border-amber-800">
                            ADVISORY HYPOTHESIS
                          </span>
                        </div>
                        <p className="text-[#E5E7EB]">{item.probable_cause}</p>
                      </div>
                    )}

                    {item.contextual_explanation && (
                      <p className="text-[11px] text-[#9CA3AF] italic">
                        {item.contextual_explanation}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Observed Events List */}
          {report?.observed_events && report.observed_events.length > 0 && (
            <div className="space-y-2 pt-2">
              <span className="text-[10px] text-[#9CA3AF] uppercase font-semibold">
                Classified Traffic Events & Operational Irregularities
              </span>
              <div className="space-y-2">
                {report.observed_events.map((evt, idx) => (
                  <div
                    key={`obs-evt-${idx}`}
                    className="p-2.5 rounded-lg bg-[#0B0F17] border border-[#1F2937] flex flex-wrap items-center justify-between gap-2"
                  >
                    <div className="space-y-1 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-purple-950/80 text-purple-300 border border-purple-800">
                          {evt.event_type.replace(/_/g, " ")}
                        </span>
                        {evt.supporting_evidence_ref && (
                          <span className="text-[10px] text-[#6B7280]">
                            Ref: <code>{evt.supporting_evidence_ref}</code>
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-[#E5E7EB]">{evt.description}</p>
                    </div>

                    {evt.timestamp_s != null && (
                      <button
                        type="button"
                        onClick={() => onSeek(evt.timestamp_s!)}
                        className="px-2 py-0.5 rounded bg-cyan-950/60 hover:bg-cyan-900/60 text-cyan-300 border border-cyan-800/80 flex items-center gap-1 text-[10px] font-semibold transition-colors"
                      >
                        <Navigation className="w-3 h-3 rotate-45" />
                        <span>t={evt.timestamp_s.toFixed(1)}s</span>
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>
      )}

      {/* 7. VISUAL OBSERVATIONS (VLM) */}
      {report?.visual_observations && report.visual_observations.length > 0 && (
        <Card className="p-4 rounded-xl bg-[#0E1524] border border-[#1F2937] space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <Eye className="w-4 h-4 text-emerald-400" />
              <span>Direct Visual Keyframe Observations</span>
            </h4>
            <span className="text-[10px] text-[#9CA3AF]">
              {report.visual_observations.length} Observations
            </span>
          </div>

          <div className="space-y-2">
            {report.visual_observations.map((obs, idx) => (
              <div
                key={`vis-obs-${idx}`}
                className="p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937] space-y-2"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-emerald-300">{obs.observation_id}</span>
                    {obs.keyframe_id && (
                      <span className="text-[10px] text-[#9CA3AF]">Frame: {obs.keyframe_id}</span>
                    )}
                  </div>

                  {obs.timestamp_s != null && (
                    <button
                      type="button"
                      onClick={() => onSeek(obs.timestamp_s!)}
                      className="px-2 py-0.5 rounded bg-emerald-950/60 hover:bg-emerald-900/60 text-emerald-300 border border-emerald-800/80 flex items-center gap-1 text-[10px] font-semibold transition-colors"
                    >
                      <Navigation className="w-3 h-3 rotate-45" />
                      <span>Seek to t={obs.timestamp_s.toFixed(1)}s</span>
                    </button>
                  )}
                </div>

                <p className="text-xs text-[#E5E7EB] leading-relaxed">
                  {obs.description}
                </p>

                {obs.visual_cues && obs.visual_cues.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 pt-1">
                    {obs.visual_cues.map((cue, cIdx) => (
                      <span
                        key={cIdx}
                        className="px-2 py-0.5 rounded text-[10px] bg-[#161F30] text-emerald-300 border border-[#2B3B55]"
                      >
                        {cue}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          <p className="text-[10px] text-[#6B7280]">
            * Visual observations are extracted from keyframes via multimodal VLM reasoning. They provide
            qualitative context, not replacement measurements for CV telemetry.
          </p>
        </Card>
      )}

      {/* 8. EVIDENCE REFERENCES & PROVENANCE CITATIONS */}
      {report?.evidence_references && report.evidence_references.length > 0 && (
        <Card className="p-4 rounded-xl bg-[#0E1524] border border-[#1F2937] space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-cyan-400" />
              <span>Evidence References & Provenance Citations</span>
            </h4>
            <span className="text-[10px] text-[#9CA3AF]">
              {report.evidence_references.length} Citations Verified
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-[11px]">
              <thead>
                <tr className="border-b border-[#1F2937] text-[#9CA3AF]">
                  <th className="py-2 px-2 font-semibold">Subsystem Source</th>
                  <th className="py-2 px-2 font-semibold">Metric / Field Cited</th>
                  <th className="py-2 px-2 font-semibold">Asserted Value</th>
                  <th className="py-2 px-2 font-semibold">Grounding Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1F2937]/60">
                {report.evidence_references.map((ref, idx) => (
                  <tr key={`ev-ref-${idx}`} className="hover:bg-[#161F30]/40">
                    <td className="py-2 px-2 font-mono text-cyan-300">{ref.source}</td>
                    <td className="py-2 px-2 font-mono text-purple-300">{ref.metric_or_field}</td>
                    <td className="py-2 px-2 font-mono text-white">
                      {typeof ref.asserted_value === "object"
                        ? JSON.stringify(ref.asserted_value)
                        : String(ref.asserted_value)}
                    </td>
                    <td className="py-2 px-2">
                      {ref.verified ? (
                        <span className="px-2 py-0.5 rounded text-[9px] font-bold bg-emerald-950/80 text-emerald-400 border border-emerald-800">
                          VERIFIED
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded text-[9px] font-bold bg-amber-950/80 text-amber-400 border border-amber-800">
                          UNVERIFIED
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* 9. ADVISORY RECOMMENDATIONS */}
      {report?.recommendations && report.recommendations.length > 0 && (
        <Card className="p-4 rounded-xl bg-[#0E1524] border border-[#1F2937] space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-purple-400" />
              <span>Operational Follow-Up Recommendations</span>
            </h4>
            <span className="text-[10px] text-[#9CA3AF]">
              {report.recommendations.length} Actions
            </span>
          </div>

          <div className="space-y-2.5">
            {report.recommendations.map((rec, idx) => (
              <div
                key={`rec-${idx}`}
                className="p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937] space-y-2"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-white text-xs">{rec.action}</span>
                    {renderPriorityBadge(rec.priority)}
                    {rec.is_advisory && (
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-950/80 text-purple-300 border border-purple-700/60">
                        ADVISORY
                      </span>
                    )}
                  </div>
                </div>

                <p className="text-xs text-[#D1D5DB] leading-relaxed">
                  <strong>Rationale:</strong> {rec.rationale}
                </p>
              </div>
            ))}
          </div>

          <div className="p-2.5 rounded bg-purple-950/20 border border-purple-800/40 text-[10px] text-purple-300/90 flex items-center gap-2">
            <AlertTriangle className="w-3.5 h-3.5 text-purple-400 shrink-0" />
            <span>
              <strong>Advisory Safeguard:</strong> Decision-support recommendation. Must not be connected to
              automated signal actuators without human validation.
            </span>
          </div>
        </Card>
      )}

      {/* 10. VALIDATION DIAGNOSTICS & LIMITATIONS */}
      <Card className="p-4 rounded-xl bg-[#0E1524] border border-[#1F2937] space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-bold text-white flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-cyan-400" />
            <span>Epistemic Grounding Diagnostics & Limitations</span>
          </h4>
          {report?.validation_diagnostics && (
            <span
              className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase border ${
                report.validation_diagnostics.validation_passed
                  ? "bg-emerald-950/80 text-emerald-400 border-emerald-800"
                  : "bg-red-950/80 text-red-400 border-red-800"
              }`}
            >
              {report.validation_diagnostics.validation_passed ? "Validation Passed" : "Grounding Flags Detected"}
            </span>
          )}
        </div>

        {report?.validation_diagnostics && (
          <div className="grid grid-cols-3 gap-2.5 text-center">
            <div className="p-2 rounded bg-[#0B0F17] border border-[#1F2937]">
              <div className="text-[10px] text-[#9CA3AF]">Claims Evaluated</div>
              <div className="text-sm font-bold text-white">
                {report.validation_diagnostics.total_claims_checked}
              </div>
            </div>
            <div className="p-2 rounded bg-[#0B0F17] border border-[#1F2937]">
              <div className="text-[10px] text-[#9CA3AF]">Grounded Claims</div>
              <div className="text-sm font-bold text-emerald-400">
                {report.validation_diagnostics.grounded_claims}
              </div>
            </div>
            <div className="p-2 rounded bg-[#0B0F17] border border-[#1F2937]">
              <div className="text-[10px] text-[#9CA3AF]">Demoted Claims</div>
              <div className="text-sm font-bold text-amber-400">
                {report.validation_diagnostics.demoted_claims}
              </div>
            </div>
          </div>
        )}

        {/* Validation Warnings */}
        {report?.validation_diagnostics?.warnings &&
          report.validation_diagnostics.warnings.length > 0 && (
            <div className="p-3 rounded-lg bg-amber-950/30 border border-amber-800/60 space-y-1.5">
              <span className="text-[10px] font-bold text-amber-300 uppercase flex items-center gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5" />
                Grounding Warnings & Suppressed Assertions
              </span>
              <ul className="list-disc list-inside text-[11px] text-amber-200/90 space-y-1">
                {report.validation_diagnostics.warnings.map((w, idx) => (
                  <li key={`warn-${idx}`}>{w}</li>
                ))}
              </ul>
            </div>
          )}

        {/* Validation Notes */}
        {report?.validation_diagnostics?.notes &&
          report.validation_diagnostics.notes.length > 0 && (
            <div className="p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937] space-y-1">
              <span className="text-[10px] font-bold text-[#9CA3AF] uppercase">
                Grounding Validator Notes
              </span>
              <ul className="list-disc list-inside text-[11px] text-[#D1D5DB] space-y-1">
                {report.validation_diagnostics.notes.map((n, idx) => (
                  <li key={`note-${idx}`}>{n}</li>
                ))}
              </ul>
            </div>
          )}

        {/* Unverified Aspects */}
        {report?.unverified_aspects && report.unverified_aspects.length > 0 && (
          <div className="p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937] space-y-1">
            <span className="text-[10px] font-bold text-amber-400 uppercase">
              Unverified Scene Aspects
            </span>
            <ul className="list-disc list-inside text-[11px] text-[#9CA3AF] space-y-1">
              {report.unverified_aspects.map((u, idx) => (
                <li key={`unver-${idx}`}>{u}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Operational Limitations */}
        {report?.limitations && report.limitations.length > 0 && (
          <div className="p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937] space-y-1">
            <span className="text-[10px] font-bold text-[#9CA3AF] uppercase">
              Operational Limitations & Calibration Caveats
            </span>
            <ul className="list-disc list-inside text-[11px] text-[#6B7280] space-y-1">
              {report.limitations.map((lim, idx) => (
                <li key={`lim-${idx}`}>{lim}</li>
              ))}
            </ul>
          </div>
        )}
      </Card>
    </div>
  );
};
