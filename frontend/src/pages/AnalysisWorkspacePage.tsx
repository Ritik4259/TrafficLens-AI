import React from "react";
import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  RotateCw,
  AlertTriangle,
  Video,
  ArrowLeft,
  FileText,
  FileCode,
  Film,
} from "lucide-react";
import { Card } from "../components/common/Card";
import { Badge } from "../components/common/Badge";
import { Button } from "../components/common/Button";
import { StatusBadge } from "../components/common/StatusBadge";
import { EmptyState } from "../components/feedback/EmptyState";
import { LoadingSkeleton } from "../components/feedback/LoadingSkeleton";
import { useWorkspaceStore } from "../state/useWorkspaceStore";
import { useJobStatus } from "../hooks/useJobStatus";
import { jobService } from "../services/jobService";
import { ROUTES } from "../core/routes";
import { VideoInspectionPlayer } from "../components/video/VideoInspectionPlayer";
import { SynchronizedTimeline } from "../components/workspace/SynchronizedTimeline";
import { TelemetryInspectionPanel } from "../components/workspace/TelemetryInspectionPanel";

export const AnalysisWorkspacePage: React.FC = () => {
  const { jobId } = useParams<{ jobId?: string }>();
  const isRealJob = Boolean(jobId && jobId !== "preview");

  const {
    job,
    status,
    isLoading: isJobLoading,
    isError: isJobError,
    error: jobError,
    isQueued,
    isProcessing,
    isCompleted,
    isFailed,
    refetch: refetchJob,
  } = useJobStatus(jobId);

  // Results query: Only fetched when job status is COMPLETED
  const {
    data: results,
    isLoading: isResultsLoading,
    isError: isResultsError,
    error: resultsError,
    refetch: refetchResults,
  } = useQuery({
    queryKey: ["jobResults", jobId],
    queryFn: () => jobService.getJobResults(jobId!),
    enabled: Boolean(isRealJob && isCompleted),
    staleTime: Infinity,
  });

  // Artifacts query: Verify physical availability of generated media and logs
  const { data: artifactsData } = useQuery({
    queryKey: ["jobArtifacts", jobId],
    queryFn: () => jobService.getJobArtifacts(jobId!),
    enabled: Boolean(isRealJob && isCompleted),
    staleTime: Infinity,
  });

  const {
    currentTimeSeconds,
    setTime,
  } = useWorkspaceStore();

  // If no job or "preview" route, show guidance to select a real job
  if (!isRealJob) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between pb-3 border-b border-[#1F2937]">
          <div>
            <h1 className="text-xl font-bold font-mono text-[#F9FAFB] tracking-tight">
              Video Analysis Workspace
            </h1>
            <p className="text-xs text-[#9CA3AF] font-mono mt-0.5">
              Optical computer vision inspection and spatiotemporal traffic telemetry
            </p>
          </div>
          <Link to={ROUTES.VIDEOS}>
            <Button variant="primary" size="sm" icon={<Video className="w-4 h-4" />}>
              Select Video from Library
            </Button>
          </Link>
        </div>

        <Card className="p-8 border-[#1F2937] bg-[#111827]">
          <EmptyState
            icon={<Video className="w-8 h-8 text-cyan-400" />}
            title="No Active Video Processing Job Selected"
            description="To view real optical video playback, CV detections, tracking metrics, and spatial occupancy, select an ingested video from the Video Library and initiate processing."
            actionLabel="Go to Video Library"
            onAction={() => (window.location.href = ROUTES.VIDEOS)}
          />
        </Card>
      </div>
    );
  }

  // Loading Job Record
  if (isJobLoading) {
    return (
      <div className="space-y-4 font-mono">
        <LoadingSkeleton className="h-24 w-full" />
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          <div className="lg:col-span-8">
            <LoadingSkeleton className="aspect-video w-full" />
          </div>
          <div className="lg:col-span-4">
            <LoadingSkeleton className="h-96 w-full" />
          </div>
        </div>
      </div>
    );
  }

  // Job Query Error (e.g., 404 Not Found)
  if (isJobError) {
    return (
      <div className="space-y-6 font-mono text-xs">
        <div className="p-6 rounded-xl bg-red-950/30 border border-red-800/60 space-y-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div>
              <h2 className="text-base font-bold text-white">Job Inaccessible or Not Found</h2>
              <p className="mt-1 text-red-300">
                {(jobError as any)?.message ||
                  `Job '${jobId}' does not exist in the backend persistence database.`}
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <Link to={ROUTES.VIDEOS}>
              <Button variant="outline" size="sm" icon={<ArrowLeft className="w-3.5 h-3.5" />}>
                Video Library
              </Button>
            </Link>
            <Button variant="primary" size="sm" onClick={() => refetchJob()}>
              Retry Fetch
            </Button>
          </div>
        </div>
      </div>
    );
  }

  const sequenceName = job?.sequence_name || `Job ${jobId}`;
  const totalFrames = results?.analytics_summary?.total_frames || 75;
  const durationSec = results?.analytics_summary?.duration_s || job?.duration_seconds || 3.0;
  const fps = results?.analytics_summary?.fps || 25.0;

  // Check physical availability of annotated video artifact
  const annotatedArtifact = artifactsData?.artifacts?.find(
    (a) => a.name === "annotated_video" || a.artifact_type === "video"
  );
  const isAnnotatedAvailable = Boolean(
    annotatedArtifact ? annotatedArtifact.is_available : Boolean(results?.artifacts?.annotated_video)
  );

  const stoppedEvents = results?.analytics_summary?.stopped_vehicle_events || [];
  const flowWindows = results?.analytics_summary?.flow_windows || [];

  return (
    <div className="space-y-4 font-mono">
      {/* Workspace Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-3 border-b border-[#1F2937]">
        <div className="flex items-center gap-3">
          <Link
            to={ROUTES.VIDEOS}
            className="p-1.5 rounded-md text-[#9CA3AF] hover:text-white hover:bg-[#1A2234] transition-colors"
            title="Back to Video Library"
          >
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-[#F9FAFB] tracking-tight">
                {sequenceName}
              </h1>
              {status && <StatusBadge status={status} size="sm" />}
              <Badge variant="outline" size="sm">
                Target: {job?.video_id}
              </Badge>
            </div>
            <p className="text-xs text-[#9CA3AF] mt-0.5">
              Job ID: <span className="text-cyan-400">{jobId}</span> • Created:{" "}
              {job?.created_at ? new Date(job.created_at).toLocaleTimeString() : "--"}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <Link to={`/jobs/${jobId}`}>
            <Button variant="outline" size="sm">
              View Audit Trail
            </Button>
          </Link>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              refetchJob();
              if (isCompleted) refetchResults();
            }}
            icon={<RotateCw className="w-3.5 h-3.5 text-cyan-400" />}
          >
            Refresh
          </Button>
        </div>
      </div>

      {/* Asynchronous Queued Banner */}
      {isQueued && (
        <div className="p-4 rounded-lg bg-cyan-950/30 border border-cyan-800/60 flex items-center justify-between gap-4 text-xs">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 rounded-full bg-cyan-400 animate-ping" />
            <div>
              <span className="font-bold text-cyan-300 uppercase tracking-wide">
                Job Queued in Worker Queue
              </span>
              <p className="text-[#9CA3AF] text-[11px] mt-0.5">
                Waiting for background inference thread to claim execution. Original source video is available below.
              </p>
            </div>
          </div>
          <Badge variant="cyan" size="sm">
            STATUS: QUEUED
          </Badge>
        </div>
      )}

      {/* Asynchronous Processing Banner */}
      {isProcessing && (
        <div className="p-4 rounded-lg bg-cyan-950/40 border border-cyan-500/50 flex items-center justify-between gap-4 text-xs">
          <div className="flex items-center gap-3">
            <RotateCw className="w-4 h-4 text-cyan-400 animate-spin" />
            <div>
              <span className="font-bold text-cyan-300 uppercase tracking-wide">
                CV Pipeline Processing in Progress
              </span>
              <p className="text-[#9CA3AF] text-[11px] mt-0.5">
                YOLO11n vehicle detection, ByteTrack tracking, and spatiotemporal analytics executing on worker thread.
              </p>
            </div>
          </div>
          <Badge variant="cyan" size="sm">
            PROCESSING
          </Badge>
        </div>
      )}

      {/* Asynchronous Failure Banner */}
      {isFailed && (
        <div className="p-4 rounded-lg bg-red-950/40 border border-red-800/70 text-xs space-y-2">
          <div className="flex items-center gap-2 text-red-300 font-bold">
            <AlertTriangle className="w-4 h-4 text-red-400" />
            <span>Processing Pipeline Failed</span>
          </div>
          <p className="text-[#D1D5DB] text-[11px]">
            {job?.error?.message || "An unhandled error occurred during CV pipeline execution."}
          </p>
          {job?.error?.error_code && (
            <div className="text-[10px] text-red-400">
              Error Code: <code className="bg-black/40 px-1 py-0.5 rounded">{job.error.error_code}</code>
            </div>
          )}
        </div>
      )}

      {/* Flagship Dual-Pane Workstation Layout: 65% Video Player / 35% Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* LEFT COLUMN: Primary Video Inspection & Timeline Viewport (60-65%) */}
        <div className="lg:col-span-7 xl:col-span-8 space-y-3.5">
          {job?.video_id && (
            <VideoInspectionPlayer
              videoId={job.video_id}
              jobId={jobId!}
              fps={fps}
              totalFrames={totalFrames}
              durationSeconds={durationSec}
              isAnnotatedAvailable={isAnnotatedAvailable}
              annotationDiagnostics={results?.annotation_diagnostics}
              stoppedEvents={stoppedEvents}
              flowWindows={flowWindows}
            />
          )}

          {/* Synchronized Timeline Navigation Bar */}
          <SynchronizedTimeline
            durationSeconds={durationSec}
            currentTimeSeconds={currentTimeSeconds}
            fps={fps}
            stoppedEvents={stoppedEvents}
            flowWindows={flowWindows}
            lineCounts={results?.analytics_summary?.line_counts}
            onSeek={(t) => setTime(t)}
          />

          {/* Verified Artifact Actions HUD */}
          {isCompleted && (
            <div className="p-3 rounded-xl bg-[#111827] border border-[#1F2937] flex flex-wrap items-center justify-between gap-3 text-xs">
              <span className="text-[#9CA3AF] text-[11px] font-semibold flex items-center gap-1.5">
                <FileCode className="w-3.5 h-3.5 text-cyan-400" />
                Verified Server Artifacts:
              </span>

              <div className="flex flex-wrap items-center gap-2">
                <a
                  href={`/api/v1/jobs/${encodeURIComponent(jobId!)}/results`}
                  target="_blank"
                  rel="noreferrer"
                  className="px-2.5 py-1 rounded bg-[#161F30] hover:bg-[#1A2234] border border-[#374151] text-cyan-300 transition-colors flex items-center gap-1.5 text-[11px]"
                  title="Inspect raw structured JSON telemetry response"
                >
                  <FileText className="w-3 h-3 text-cyan-400" />
                  <span>Results JSON</span>
                </a>

                {isAnnotatedAvailable && (
                  <a
                    href={`/api/v1/jobs/${encodeURIComponent(jobId!)}/annotated-video`}
                    target="_blank"
                    rel="noreferrer"
                    className="px-2.5 py-1 rounded bg-cyan-950/40 hover:bg-cyan-900/60 border border-cyan-700/60 text-cyan-200 transition-colors flex items-center gap-1.5 text-[11px]"
                    title="Open annotated MP4 media stream in new tab"
                  >
                    <Film className="w-3 h-3 text-cyan-400" />
                    <span>Annotated MP4 Stream</span>
                  </a>
                )}

                <a
                  href={`/api/v1/videos/${encodeURIComponent(job?.video_id || "")}/content`}
                  target="_blank"
                  rel="noreferrer"
                  className="px-2.5 py-1 rounded bg-[#161F30] hover:bg-[#1A2234] border border-[#374151] text-[#D1D5DB] transition-colors flex items-center gap-1.5 text-[11px]"
                  title="Open original source MP4 media stream in new tab"
                >
                  <Video className="w-3 h-3 text-[#9CA3AF]" />
                  <span>Source MP4 Stream</span>
                </a>
              </div>
            </div>
          )}
        </div>

        {/* RIGHT COLUMN: Real Telemetry Inspection Panel (35-40%) */}
        <div className="lg:col-span-5 xl:col-span-4 space-y-4">
          {!isCompleted ? (
            <Card className="p-6 border-[#1F2937] bg-[#111827] text-center space-y-3 font-mono">
              <RotateCw className="w-6 h-6 animate-spin mx-auto text-cyan-400" />
              <div className="text-xs text-[#9CA3AF]">
                {isQueued
                  ? "Job is queued in the worker queue. Telemetry will render automatically once processing completes."
                  : isProcessing
                  ? "Inference in progress on background worker thread. Telemetry will populate upon completion."
                  : "Telemetry results unavailable for failed job."}
              </div>
            </Card>
          ) : isResultsLoading ? (
            <div className="space-y-3 font-mono">
              <LoadingSkeleton className="h-44 w-full" />
              <LoadingSkeleton className="h-64 w-full" />
            </div>
          ) : isResultsError ? (
            <div className="p-4 rounded-xl bg-red-950/30 border border-red-800/60 text-xs font-mono space-y-2">
              <div className="flex items-center gap-2 text-red-300 font-bold">
                <AlertTriangle className="w-4 h-4 text-red-400" />
                <span>Failed to Fetch Results</span>
              </div>
              <p className="text-[11px] text-red-200">
                {(resultsError as any)?.message || "Endpoint error retrieving job results."}
              </p>
              <Button variant="outline" size="sm" onClick={() => refetchResults()}>
                Retry
              </Button>
            </div>
          ) : (
            <TelemetryInspectionPanel
              jobId={jobId}
              results={results}
              currentTimeSeconds={currentTimeSeconds}
              onSeek={(t) => setTime(t)}
            />
          )}
        </div>
      </div>
    </div>
  );
};
