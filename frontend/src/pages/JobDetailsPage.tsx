import React from "react";
import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  FileText,
  CheckCircle2,
  ArrowLeft,
  RotateCw,
  AlertTriangle,
  Clock,
  Video,
  MonitorPlay,
} from "lucide-react";
import { PageHeader } from "../components/common/PageHeader";
import { SectionCard } from "../components/common/SectionCard";
import { Card } from "../components/common/Card";
import { Badge } from "../components/common/Badge";
import { Button } from "../components/common/Button";
import { StatusBadge } from "../components/common/StatusBadge";
import { EmptyState } from "../components/feedback/EmptyState";
import { LoadingSkeleton } from "../components/feedback/LoadingSkeleton";
import { useJobStatus } from "../hooks/useJobStatus";
import { jobService } from "../services/jobService";
import { ROUTES } from "../core/routes";

const getArtifactFilename = (pathStr?: string, defaultName: string = ""): string => {
  if (!pathStr) return defaultName;
  const parts = pathStr.split(/[/\\]/);
  return parts[parts.length - 1] || defaultName;
};

export const JobDetailsPage: React.FC = () => {
  const { jobId } = useParams<{ jobId?: string }>();
  const isRealJob = Boolean(jobId && jobId !== "audit");

  const {
    job,
    status,
    isLoading: isJobLoading,
    isError: isJobError,
    error: jobError,
    isCompleted,
    isProcessing,
    isFailed,
    refetch,
  } = useJobStatus(jobId);

  // Fetch results to get generated artifact records
  const { data: results } = useQuery({
    queryKey: ["jobResults", jobId],
    queryFn: () => jobService.getJobResults(jobId!),
    enabled: Boolean(isRealJob && isCompleted),
    staleTime: Infinity,
  });

  if (!isRealJob) {
    return (
      <div className="space-y-6">
        <PageHeader
          category="Orchestration & Audit"
          title="Job Audit Trail"
          subtitle="Asynchronous job execution lifecycle, timestamps, and structured telemetry artifacts"
          actions={
            <Link to={ROUTES.VIDEOS}>
              <Button variant="primary" size="sm" icon={<Video className="w-4 h-4" />}>
                Go to Video Library
              </Button>
            </Link>
          }
        />
        <Card className="p-8 border-[#1F2937] bg-[#111827]">
          <EmptyState
            icon={<Clock className="w-8 h-8 text-cyan-400" />}
            title="No Job Selected for Audit"
            description="Select or initiate a processing job from the Video Library to inspect its execution lifecycle, worker timing, and telemetry artifacts."
            actionLabel="View Video Library"
            onAction={() => (window.location.href = ROUTES.VIDEOS)}
          />
        </Card>
      </div>
    );
  }

  if (isJobLoading) {
    return (
      <div className="space-y-4 font-mono">
        <LoadingSkeleton lines={1} className="h-20 w-full" />
        <LoadingSkeleton lines={1} className="h-48 w-full" />
        <LoadingSkeleton lines={1} className="h-64 w-full" />
      </div>
    );
  }

  if (isJobError) {
    return (
      <div className="space-y-6 font-mono text-xs">
        <div className="p-6 rounded-xl bg-red-950/30 border border-red-800/60 space-y-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div>
              <h2 className="text-base font-bold text-white">Job Inaccessible or Not Found</h2>
              <p className="mt-1 text-red-300">
                {(jobError as any)?.message || `Job '${jobId}' does not exist in the database.`}
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <Link to={ROUTES.VIDEOS}>
              <Button variant="outline" size="sm" icon={<ArrowLeft className="w-3.5 h-3.5" />}>
                Video Library
              </Button>
            </Link>
            <Button variant="primary" size="sm" onClick={() => refetch()}>
              Retry Fetch
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 font-mono">
      <PageHeader
        category="Orchestration & Audit"
        title={`Processing Job: ${jobId}`}
        subtitle="Asynchronous execution lifecycle, pipeline duration, and structured telemetry artifacts"
        actions={
          <div className="flex items-center gap-2">
            <Link to={ROUTES.VIDEOS}>
              <Button variant="outline" size="sm" icon={<ArrowLeft className="w-3.5 h-3.5" />}>
                Videos
              </Button>
            </Link>
            <Button
              variant="outline"
              size="sm"
              onClick={() => refetch()}
              icon={<RotateCw className="w-3.5 h-3.5 text-cyan-400" />}
            >
              Refresh
            </Button>
            <Link to={`/workspace/${jobId}`}>
              <Button variant="primary" size="sm" icon={<MonitorPlay className="w-3.5 h-3.5" />}>
                Open Workspace
              </Button>
            </Link>
          </div>
        }
      />

      {/* Lifecycle Status & Timing Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4 bg-[#111827] border-[#1F2937]">
          <div className="flex items-center justify-between text-xs text-[#9CA3AF]">
            <span>Current Status</span>
          </div>
          <div className="mt-2">
            {status && <StatusBadge status={status} />}
          </div>
          <div className="mt-2 text-[11px] text-[#6B7280]">
            State: {status} {isCompleted || isFailed ? "(Terminal)" : "(Active)"}
          </div>
        </Card>

        <Card className="p-4 bg-[#111827] border-[#1F2937]">
          <div className="flex items-center justify-between text-xs text-[#9CA3AF]">
            <span>Wall-Clock Duration</span>
          </div>
          <div className="mt-2 text-xl font-bold text-cyan-400">
            {job?.duration_seconds != null ? `${job.duration_seconds.toFixed(2)}s` : isProcessing ? "Running..." : "--"}
          </div>
          <div className="mt-2 text-[11px] text-[#6B7280]">
            Total pipeline execution time
          </div>
        </Card>

        <Card className="p-4 bg-[#111827] border-[#1F2937]">
          <div className="flex items-center justify-between text-xs text-[#9CA3AF]">
            <span>Target Video</span>
          </div>
          <div className="mt-2 text-sm font-bold text-[#F9FAFB] truncate">
            {job?.video_id}
          </div>
          <div className="mt-2 text-[11px] text-[#6B7280]">
            Sequence: {job?.sequence_name}
          </div>
        </Card>

        <Card className="p-4 bg-[#111827] border-[#1F2937]">
          <div className="flex items-center justify-between text-xs text-[#9CA3AF]">
            <span>Creation Timestamp</span>
          </div>
          <div className="mt-2 text-xs font-bold text-[#F9FAFB]">
            {job?.created_at ? new Date(job.created_at).toLocaleTimeString() : "--"}
          </div>
          <div className="mt-2 text-[11px] text-[#6B7280]">
            {job?.created_at ? new Date(job.created_at).toLocaleDateString() : "--"}
          </div>
        </Card>
      </div>

      {/* Failure Details if FAILED */}
      {isFailed && job?.error && (
        <div className="p-4 rounded-xl bg-red-950/30 border border-red-800/60 text-xs space-y-2">
          <div className="flex items-center gap-2 text-red-300 font-bold">
            <AlertTriangle className="w-4 h-4 text-red-400" />
            <span>Job Execution Failure Details</span>
          </div>
          <p className="text-white text-xs">{job.error.message}</p>
          <div className="text-[11px] text-red-400 flex gap-4">
            <span>Code: <code>{job.error.error_code}</code></span>
            <span>Occurred: {new Date(job.error.occurred_at).toLocaleTimeString()}</span>
          </div>
        </div>
      )}

      {/* Execution Lifecycle Audit Trail */}
      <SectionCard
        title="Execution Lifecycle Audit Trail"
        description="Deterministic state transitions logged during job orchestration"
      >
        <div className="space-y-3 text-xs">
          {/* Step 1: Created & Queued */}
          <div className="flex items-center gap-3 p-3 rounded bg-[#1A2234] border border-[#374151]">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <div className="flex-1 flex flex-col sm:flex-row sm:items-center justify-between gap-1">
              <div>
                <span className="font-bold text-white">CREATED → QUEUED</span>
                <span className="text-[#9CA3AF] ml-2">Job persisted in database and dispatched to worker queue</span>
              </div>
              <span className="text-[#6B7280]">
                {job?.created_at ? new Date(job.created_at).toLocaleTimeString() : "--"}
              </span>
            </div>
          </div>

          {/* Step 2: Started / Processing */}
          {(isProcessing || isCompleted || isFailed) && (
            <div className="flex items-center gap-3 p-3 rounded bg-[#1A2234] border border-[#374151]">
              <CheckCircle2
                className={`w-4 h-4 shrink-0 ${
                  job?.started_at ? "text-emerald-400" : "text-cyan-400 animate-pulse"
                }`}
              />
              <div className="flex-1 flex flex-col sm:flex-row sm:items-center justify-between gap-1">
                <div>
                  <span className="font-bold text-white">QUEUED → PROCESSING</span>
                  <span className="text-[#9CA3AF] ml-2">Claimed by background JobWorker thread</span>
                </div>
                <span className="text-[#6B7280]">
                  {job?.started_at ? new Date(job.started_at).toLocaleTimeString() : "Executing..."}
                </span>
              </div>
            </div>
          )}

          {/* Step 3: Terminal State */}
          {(isCompleted || isFailed) && (
            <div className="flex items-center gap-3 p-3 rounded bg-[#1A2234] border border-[#374151]">
              <CheckCircle2
                className={`w-4 h-4 shrink-0 ${
                  isCompleted ? "text-emerald-400" : "text-red-400"
                }`}
              />
              <div className="flex-1 flex flex-col sm:flex-row sm:items-center justify-between gap-1">
                <div>
                  <span className="font-bold text-white">
                    PROCESSING → {status}
                  </span>
                  <span className="text-[#9CA3AF] ml-2">
                    {isCompleted
                      ? "Inference, tracking, analytics, and video encoding finished"
                      : "Processing aborted due to pipeline error"}
                  </span>
                </div>
                <span className="text-[#6B7280]">
                  {job?.completed_at ? new Date(job.completed_at).toLocaleTimeString() : "--"}
                </span>
              </div>
            </div>
          )}
        </div>
      </SectionCard>

      {/* Generated Telemetry Artifacts Section */}
      <SectionCard
        title="Generated Telemetry Artifacts"
        description="Structured data outputs and annotated video generated by the pipeline"
      >
        {/* Artifact Storage & Access Policy */}
        <div className="mb-4 p-3 rounded-lg bg-[#161F30] border border-[#1F2937] text-xs text-[#9CA3AF] space-y-1">
          <div className="flex items-center gap-2 text-cyan-300 font-bold">
            <AlertTriangle className="w-3.5 h-3.5 text-cyan-400" />
            <span>Artifact Storage & Access Policy</span>
          </div>
          <p className="text-[11px] leading-relaxed">
            Telemetry artifacts are retained internally in the backend storage directory. Direct static directory exposure and arbitrary path browsing are disabled. Dedicated artifact downloading and range-seekable video streaming endpoints will be introduced in Milestone 4.
          </p>
        </div>

        {isCompleted && results?.artifacts ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Artifact 1: analytics.json */}
            <div className="p-4 rounded-lg bg-[#1A2234] border border-[#374151] flex flex-col justify-between">
              <div>
                <div className="flex items-center gap-2 text-sm font-bold text-white">
                  <FileText className="w-4 h-4 text-cyan-400" />
                  <span>analytics.json</span>
                </div>
                <p className="text-xs text-[#9CA3AF] mt-1">
                  Full spatiotemporal report, tumbling window flow, and per-frame telemetry records.
                </p>
                <div className="mt-2.5 pt-2 border-t border-[#1F2937]/60 space-y-1 text-[11px] font-mono">
                  <div className="flex justify-between">
                    <span className="text-[#6B7280]">Filename:</span>
                    <span className="text-white truncate max-w-[160px]" title={getArtifactFilename(results.artifacts.analytics_json, "analytics.json")}>
                      {getArtifactFilename(results.artifacts.analytics_json, "analytics.json")}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#6B7280]">Type:</span>
                    <span className="text-cyan-400">Structured JSON</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#6B7280]">Status:</span>
                    <span className="text-emerald-400">Available via API</span>
                  </div>
                </div>
              </div>
              <div className="mt-4 pt-3 border-t border-[#1F2937] flex items-center justify-between">
                <Badge variant="cyan" size="sm">GENERATED</Badge>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => window.open(`/api/v1/jobs/${jobId}/results`, '_blank')}
                  icon={<FileText className="w-3 h-3 text-cyan-400" />}
                >
                  View Results JSON
                </Button>
              </div>
            </div>

            {/* Artifact 2: telemetry.csv */}
            <div className="p-4 rounded-lg bg-[#1A2234] border border-[#374151] flex flex-col justify-between">
              <div>
                <div className="flex items-center gap-2 text-sm font-bold text-white">
                  <FileText className="w-4 h-4 text-cyan-400" />
                  <span>telemetry.csv</span>
                </div>
                <p className="text-xs text-[#9CA3AF] mt-1">
                  Frame-by-frame tabular telemetry for external analysis and spreadsheets.
                </p>
                <div className="mt-2.5 pt-2 border-t border-[#1F2937]/60 space-y-1 text-[11px] font-mono">
                  <div className="flex justify-between">
                    <span className="text-[#6B7280]">Filename:</span>
                    <span className="text-white truncate max-w-[160px]" title={getArtifactFilename(results.artifacts.telemetry_csv, "telemetry.csv")}>
                      {getArtifactFilename(results.artifacts.telemetry_csv, "telemetry.csv")}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#6B7280]">Type:</span>
                    <span className="text-cyan-400">Tabular CSV</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#6B7280]">Status:</span>
                    <span className="text-white">Persisted (No download endpoint)</span>
                  </div>
                </div>
              </div>
              <div className="mt-4 pt-3 border-t border-[#1F2937] flex items-center justify-between">
                <Badge variant={results.artifacts.telemetry_csv ? "cyan" : "outline"} size="sm">
                  {results.artifacts.telemetry_csv ? "GENERATED" : "NOT GENERATED"}
                </Badge>
                <span className="text-[10px] text-[#6B7280]">Serving endpoint deferred to M4</span>
              </div>
            </div>

            {/* Artifact 3: annotated.mp4 */}
            <div className="p-4 rounded-lg bg-[#1A2234] border border-[#374151] flex flex-col justify-between">
              <div>
                <div className="flex items-center gap-2 text-sm font-bold text-white">
                  <FileText className="w-4 h-4 text-cyan-400" />
                  <span>annotated.mp4</span>
                </div>
                <p className="text-xs text-[#9CA3AF] mt-1">
                  Rendered video stream with bounding boxes, persistent track IDs, and ROI overlays.
                </p>
                <div className="mt-2.5 pt-2 border-t border-[#1F2937]/60 space-y-1 text-[11px] font-mono">
                  <div className="flex justify-between">
                    <span className="text-[#6B7280]">Filename:</span>
                    <span className="text-white truncate max-w-[160px]" title={getArtifactFilename(results.artifacts.annotated_video, "annotated.mp4")}>
                      {getArtifactFilename(results.artifacts.annotated_video, "annotated.mp4")}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#6B7280]">Type:</span>
                    <span className="text-cyan-400">Rendered Video MP4</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#6B7280]">Status:</span>
                    <span className="text-white">Persisted (No streaming endpoint)</span>
                  </div>
                </div>
              </div>
              <div className="mt-4 pt-3 border-t border-[#1F2937] flex items-center justify-between">
                <Badge variant={results.artifacts.annotated_video ? "cyan" : "outline"} size="sm">
                  {results.artifacts.annotated_video ? "GENERATED" : "PENDING (M4)"}
                </Badge>
                <span className="text-[10px] text-[#6B7280]">Streaming endpoint deferred to M4</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-6 rounded-lg bg-[#161F30]/40 border border-[#1F2937] text-center text-xs text-[#9CA3AF]">
            {isCompleted
              ? "Loading artifact records from backend..."
              : "Artifacts are published once pipeline execution reaches COMPLETED status."}
          </div>
        )}
      </SectionCard>
    </div>
  );
};