import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import {
  Cpu,
  AlertTriangle,
  RotateCw,
  X,
  Plus,
  Trash2,
  CheckCircle2,
  ArrowRight,
} from "lucide-react";
import { Button } from "../common/Button";
import { Badge } from "../common/Badge";
import { videoService } from "../../services/videoService";
import { VideoMetadata } from "../../types/video";
import { CountingLineConfig, ROIConfig } from "../../types/common";
import { ProcessingJobCreate } from "../../types/job";
import { formatDuration } from "../../core/validation";

interface ProcessingConfigModalProps {
  video: VideoMetadata;
  onClose: () => void;
}

export const ProcessingConfigModal: React.FC<ProcessingConfigModalProps> = ({
  video,
  onClose,
}) => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [sequenceName, setSequenceName] = useState(
    video.filename.replace(/\.[^/.]+$/, "") || `seq_${video.video_id}`
  );

  // Counting Lines State (default to a sensible midline across frame)
  const defaultLineY = video.height ? Math.round(video.height * 0.55) : 300;
  const defaultLineWidth = video.width ? Math.round(video.width * 0.85) : 800;
  const defaultLineStartX = video.width ? Math.round(video.width * 0.1) : 100;

  const [enableLines, setEnableLines] = useState(true);
  const [lines, setLines] = useState<CountingLineConfig[]>([
    {
      line_id: "line_mid",
      name: "Midline Crossline",
      p1: [defaultLineStartX, defaultLineY],
      p2: [defaultLineStartX + defaultLineWidth, defaultLineY],
      positive_direction_label: "downstream",
      negative_direction_label: "upstream",
      cooldown_frames: 15,
      min_track_age: 2,
    },
  ]);

  // Polygonal ROI State (default to a 4-point polygon covering road area)
  const [enableRois, setEnableRois] = useState(true);
  const roiWidth = video.width || 960;
  const roiHeight = video.height || 540;
  const [rois] = useState<ROIConfig[]>([
    {
      roi_id: "roi_main",
      name: "Road Surface Zone",
      polygon: [
        [Math.round(roiWidth * 0.1), Math.round(roiHeight * 0.2)],
        [Math.round(roiWidth * 0.9), Math.round(roiHeight * 0.2)],
        [Math.round(roiWidth * 0.95), Math.round(roiHeight * 0.9)],
        [Math.round(roiWidth * 0.05), Math.round(roiHeight * 0.9)],
      ],
      description: "Primary traffic monitoring ROI",
    },
  ]);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [activeConflictJobId, setActiveConflictJobId] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setErrorMessage(null);
    setActiveConflictJobId(null);

    const payload: ProcessingJobCreate = {
      sequence_name: sequenceName.trim() || undefined,
      lines: enableLines && lines.length > 0 ? lines : undefined,
      rois: enableRois && rois.length > 0 ? rois : undefined,
    };

    try {
      const job = await videoService.createVideoJob(video.video_id, payload);
      // Invalidate relevant queries so cache reflects new active job
      queryClient.invalidateQueries({ queryKey: ["job", job.job_id] });
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      // Navigate to the real workspace with real job ID
      navigate(`/workspace/${job.job_id}`);
    } catch (err: any) {
      if (err.status === 409) {
        // Active job already exists
        const msg = err.detail || err.message || "Video already has an active processing job.";
        setErrorMessage(msg);

        // Attempt to extract existing job ID from detail string (e.g. "active processing job (job_...)")
        const match = typeof msg === "string" ? msg.match(/job_[a-f0-9]+/) : null;
        if (match) {
          setActiveConflictJobId(match[0]);
        }
      } else if (err.status === 503) {
        setErrorMessage(
          "Worker queue is full (capacity limit reached). Please wait for active jobs to complete."
        );
      } else {
        const msg = err.detail || err.message || "Failed to schedule processing job.";
        setErrorMessage(typeof msg === "string" ? msg : JSON.stringify(msg));
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const addLine = () => {
    const nextIdx = lines.length + 1;
    setLines([
      ...lines,
      {
        line_id: `line_${nextIdx}`,
        name: `Crossline #${nextIdx}`,
        p1: [100, 200 + nextIdx * 50],
        p2: [800, 200 + nextIdx * 50],
        positive_direction_label: "downstream",
        negative_direction_label: "upstream",
        cooldown_frames: 15,
        min_track_age: 2,
      },
    ]);
  };

  const removeLine = (index: number) => {
    setLines(lines.filter((_, i) => i !== index));
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xs flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-[#111827] border border-[#1F2937] rounded-xl max-w-2xl w-full p-6 text-[#E5E7EB] shadow-2xl space-y-5 my-8">
        {/* Modal Header */}
        <div className="flex items-center justify-between pb-4 border-b border-[#1F2937]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-cyan-950/80 border border-cyan-800/80 flex items-center justify-center text-cyan-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold font-mono text-white tracking-tight">
                Configure Processing Job
              </h3>
              <p className="text-xs text-[#9CA3AF] font-mono mt-0.5">
                Target: {video.filename} ({video.width}x{video.height} @ {video.fps} FPS, {formatDuration(video.duration_seconds)})
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-md text-[#9CA3AF] hover:text-white hover:bg-[#1A2234] transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Video Asset Metadata Banner */}
        <div className="p-3 rounded-lg bg-[#161F30] border border-[#1F2937] grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
          <div>
            <span className="text-[#6B7280] block text-[10px]">VIDEO ID</span>
            <span className="text-cyan-400 font-bold">{video.video_id}</span>
          </div>
          <div>
            <span className="text-[#6B7280] block text-[10px]">FRAMES</span>
            <span className="text-white font-bold">{video.frame_count || "Auto"}</span>
          </div>
          <div>
            <span className="text-[#6B7280] block text-[10px]">DETECTION MODEL</span>
            <span className="text-[#D1D5DB]">YOLO11n (Vehicle)</span>
          </div>
          <div>
            <span className="text-[#6B7280] block text-[10px]">TRACKER</span>
            <span className="text-[#D1D5DB]">ByteTrack Baseline</span>
          </div>
        </div>

        {/* Error Alert */}
        {errorMessage && (
          <div className="p-3.5 rounded-lg bg-amber-950/30 border border-amber-800/60 flex items-start justify-between gap-3 font-mono text-xs text-amber-200">
            <div className="flex items-start gap-2.5">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div>
                <span className="font-bold">Execution Notice:</span>
                <p className="mt-0.5 text-[11px] text-[#D1D5DB] leading-relaxed">
                  {errorMessage}
                </p>
              </div>
            </div>
            {activeConflictJobId && (
              <Button
                variant="primary"
                size="sm"
                onClick={() => navigate(`/workspace/${activeConflictJobId}`)}
                icon={<ArrowRight className="w-3.5 h-3.5" />}
              >
                Track Job
              </Button>
            )}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4 font-mono text-xs">
          {/* Sequence Name Field */}
          <div>
            <label className="block text-[#9CA3AF] text-xs font-semibold mb-1">
              Sequence Name / Logical Label
            </label>
            <input
              type="text"
              value={sequenceName}
              onChange={(e) => setSequenceName(e.target.value)}
              placeholder="e.g., MVI_20011"
              required
              className="w-full bg-[#1A2234] border border-[#374151] rounded-lg px-3 py-2 text-white placeholder-[#6B7280] focus:outline-none focus:border-cyan-500 font-mono"
            />
          </div>

          {/* Virtual Counting Lines Configuration */}
          <div className="border border-[#1F2937] rounded-lg p-3 bg-[#0E1420]/70 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="enableLines"
                  checked={enableLines}
                  onChange={(e) => setEnableLines(e.target.checked)}
                  className="rounded border-[#374151] bg-[#1A2234] text-cyan-500 focus:ring-0"
                />
                <label htmlFor="enableLines" className="text-white font-bold cursor-pointer">
                  Virtual Counting Lines
                </label>
                <Badge variant="cyan" size="sm">
                  {lines.length} Line{lines.length !== 1 ? "s" : ""}
                </Badge>
              </div>
              {enableLines && (
                <button
                  type="button"
                  onClick={addLine}
                  className="text-cyan-400 hover:text-cyan-300 text-[11px] flex items-center gap-1 font-semibold"
                >
                  <Plus className="w-3 h-3" /> Add Line
                </button>
              )}
            </div>

            {enableLines && (
              <div className="space-y-2 max-h-40 overflow-y-auto pr-1">
                {lines.map((line, idx) => (
                  <div
                    key={line.line_id || idx}
                    className="p-2.5 rounded bg-[#161F30] border border-[#1F2937] flex items-center justify-between gap-3 text-[11px]"
                  >
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 flex-1">
                      <div>
                        <span className="text-[#6B7280] block text-[9px]">ID & LABEL</span>
                        <span className="text-white font-bold">{line.name}</span>
                        <span className="text-[#9CA3AF] text-[10px] block">({line.line_id})</span>
                      </div>
                      <div>
                        <span className="text-[#6B7280] block text-[9px]">COORDINATES (p1 → p2)</span>
                        <span className="text-cyan-300">
                          [{line.p1[0]}, {line.p1[1]}] → [{line.p2[0]}, {line.p2[1]}]
                        </span>
                      </div>
                      <div>
                        <span className="text-[#6B7280] block text-[9px]">DIRECTIONS</span>
                        <span className="text-emerald-400">{line.positive_direction_label}</span> /{" "}
                        <span className="text-amber-400">{line.negative_direction_label}</span>
                      </div>
                    </div>
                    {lines.length > 1 && (
                      <button
                        type="button"
                        onClick={() => removeLine(idx)}
                        className="text-[#9CA3AF] hover:text-red-400 p-1"
                        title="Remove line"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Polygonal ROI Configuration */}
          <div className="border border-[#1F2937] rounded-lg p-3 bg-[#0E1420]/70 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="enableRois"
                  checked={enableRois}
                  onChange={(e) => setEnableRois(e.target.checked)}
                  className="rounded border-[#374151] bg-[#1A2234] text-cyan-500 focus:ring-0"
                />
                <label htmlFor="enableRois" className="text-white font-bold cursor-pointer">
                  Polygonal Regions of Interest (ROIs)
                </label>
                <Badge variant="cyan" size="sm">
                  {rois.length} Zone{rois.length !== 1 ? "s" : ""}
                </Badge>
              </div>
            </div>

            {enableRois && (
              <div className="space-y-2">
                {rois.map((roi) => (
                  <div
                    key={roi.roi_id}
                    className="p-2.5 rounded bg-[#161F30] border border-[#1F2937] space-y-1 text-[11px]"
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-white font-bold">
                        {roi.name} <span className="text-[#9CA3AF]">({roi.roi_id})</span>
                      </span>
                      <span className="text-[#6B7280]">{roi.polygon.length} vertices</span>
                    </div>
                    <div className="text-[10px] text-cyan-400/90 truncate">
                      Polygon: {JSON.stringify(roi.polygon)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Modal Actions */}
          <div className="pt-3 border-t border-[#1F2937] flex items-center justify-between">
            <p className="text-[11px] text-[#6B7280]">
              Dispatched to background worker queue (max concurrency: 1).
            </p>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={onClose} disabled={isSubmitting}>
                Cancel
              </Button>
              <Button
                variant="primary"
                size="sm"
                type="submit"
                disabled={isSubmitting}
                icon={
                  isSubmitting ? (
                    <RotateCw className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <CheckCircle2 className="w-3.5 h-3.5" />
                  )
                }
              >
                {isSubmitting ? "Queueing Job..." : "Submit Processing Job"}
              </Button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};
