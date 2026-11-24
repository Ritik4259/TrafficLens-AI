import React, { useState } from "react";
import {
  Activity,
  TrendingUp,
  Grid,
  AlertCircle,
  HelpCircle,
  Car,
  Layers,
  Navigation,
  Sparkles,
  FileText,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { JobResults, LineFlow, StoppedVehicleEvent } from "../../types/telemetry";
import { SceneInterpretationPanel } from "./SceneInterpretationPanel";
import { IncidentReportPanel } from "./IncidentReportPanel";
import { useSceneInterpretation } from "../../hooks/useSceneInterpretation";
import { useIncidentReport } from "../../hooks/useIncidentReport";

export interface TelemetryInspectionPanelProps {
  jobId?: string;
  results?: JobResults | null;
  currentTimeSeconds: number;
  onSeek: (timeSeconds: number) => void;
}

export const TelemetryInspectionPanel: React.FC<TelemetryInspectionPanelProps> = ({
  jobId,
  results,
  currentTimeSeconds,
  onSeek,
}) => {
  const [activeTab, setActiveTab] = useState<
    "overview" | "flow" | "rois" | "events" | "vlm" | "report"
  >("overview");

  const { status: vlmStatus } = useSceneInterpretation(jobId);
  const { hasReport: hasIncidentReport } = useIncidentReport(jobId);

  const analytics = results?.analytics_summary;
  const detection = results?.detection_summary;
  const tracking = results?.tracking_summary;

  // Extract real telemetry metrics
  const totalUniqueTracks =
    tracking?.total_unique_tracks ?? analytics?.total_unique_vehicles ?? 0;
  const totalDetections = detection?.total_detections ?? 0;
  const meanConfidence = detection?.mean_confidence ?? 0;

  const lineCounts = analytics?.line_counts || {};
  const flowWindows: LineFlow[] = analytics?.flow_windows || [];
  const stoppedEvents: StoppedVehicleEvent[] = analytics?.stopped_vehicle_events || [];
  const peakOccupancyPerRoi = analytics?.peak_occupancy_per_roi || {};
  const avgOccupancyPerRoi = analytics?.average_occupancy_per_roi || {};
  const congestionDistPerRoi = analytics?.congestion_distribution_per_roi || {};

  // Cumulative line crossings
  const totalCrossings = Object.values(lineCounts).reduce(
    (acc, dirs) =>
      acc + Object.values(dirs).reduce((s: number, c: number) => s + c, 0),
    0
  );

  // Prepare chart data for Flow Windows
  const flowChartData = flowWindows.map((win, idx) => ({
    name: `W${idx + 1} (${win.window_start_s}s)`,
    start: win.window_start_s,
    end: win.window_end_s,
    flowVpm: Math.round(win.flow_rate_vpm * 10) / 10,
    count: win.total_count,
  }));

  // Prepare chart data for ROI Occupancy
  const roiChartData = Object.keys(peakOccupancyPerRoi).map((roiId) => ({
    roiId,
    peakPct: Math.round((peakOccupancyPerRoi[roiId] || 0) * 1000) / 10,
    avgPct: Math.round((avgOccupancyPerRoi[roiId] || 0) * 1000) / 10,
  }));

  return (
    <div className="rounded-xl bg-[#111827] border border-[#1F2937] overflow-hidden flex flex-col font-mono text-xs shadow-lg">
      {/* Tab Navigation Header */}
      <div className="flex border-b border-[#1F2937] bg-[#0B0F17]/80 overflow-x-auto text-xs">
        <button
          type="button"
          onClick={() => setActiveTab("overview")}
          className={`px-3.5 py-2.5 font-semibold transition-all border-b-2 flex items-center gap-1.5 whitespace-nowrap ${
            activeTab === "overview"
              ? "border-cyan-400 text-cyan-300 bg-[#161F30]"
              : "border-transparent text-[#9CA3AF] hover:text-white hover:bg-[#1A2234]"
          }`}
        >
          <Activity className="w-3.5 h-3.5" />
          <span>Tracking & Detection</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("flow")}
          className={`px-3.5 py-2.5 font-semibold transition-all border-b-2 flex items-center gap-1.5 whitespace-nowrap ${
            activeTab === "flow"
              ? "border-cyan-400 text-cyan-300 bg-[#161F30]"
              : "border-transparent text-[#9CA3AF] hover:text-white hover:bg-[#1A2234]"
          }`}
        >
          <TrendingUp className="w-3.5 h-3.5" />
          <span>Flow & Crossings</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("rois")}
          className={`px-3.5 py-2.5 font-semibold transition-all border-b-2 flex items-center gap-1.5 whitespace-nowrap ${
            activeTab === "rois"
              ? "border-cyan-400 text-cyan-300 bg-[#161F30]"
              : "border-transparent text-[#9CA3AF] hover:text-white hover:bg-[#1A2234]"
          }`}
        >
          <Grid className="w-3.5 h-3.5" />
          <span>Spatial Zones & Congestion</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("events")}
          className={`px-3.5 py-2.5 font-semibold transition-all border-b-2 flex items-center gap-1.5 whitespace-nowrap ${
            activeTab === "events"
              ? "border-red-400 text-red-300 bg-red-950/20"
              : "border-transparent text-[#9CA3AF] hover:text-white hover:bg-[#1A2234]"
          }`}
        >
          <AlertCircle className="w-3.5 h-3.5 text-red-400" />
          <span>Stationary Events ({stoppedEvents.length})</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("vlm")}
          className={`px-3.5 py-2.5 font-semibold transition-all border-b-2 flex items-center gap-1.5 whitespace-nowrap ${
            activeTab === "vlm"
              ? "border-purple-400 text-purple-300 bg-purple-950/20"
              : "border-transparent text-[#9CA3AF] hover:text-white hover:bg-[#1A2234]"
          }`}
          data-testid="tab-scene-understanding"
        >
          <Sparkles className="w-3.5 h-3.5 text-purple-400" />
          <span>Scene Understanding</span>
          {vlmStatus && vlmStatus !== "NOT_REQUESTED" && (
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                vlmStatus === "COMPLETED"
                  ? "bg-emerald-400"
                  : vlmStatus === "FAILED"
                  ? "bg-red-400"
                  : "bg-cyan-400 animate-pulse"
              }`}
            />
          )}
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("report")}
          className={`px-3.5 py-2.5 font-semibold transition-all border-b-2 flex items-center gap-1.5 whitespace-nowrap ${
            activeTab === "report"
              ? "border-cyan-400 text-cyan-300 bg-cyan-950/20"
              : "border-transparent text-[#9CA3AF] hover:text-white hover:bg-[#1A2234]"
          }`}
          data-testid="tab-incident-report"
        >
          <FileText className="w-3.5 h-3.5 text-cyan-400" />
          <span>Incident Report</span>
          {hasIncidentReport && (
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          )}
        </button>
      </div>

      {/* Tab Contents */}
      <div className="p-4 space-y-4">
        {/* TAB 1: OVERVIEW & TRACKING */}
        {activeTab === "overview" && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              <div className="p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937]">
                <div className="text-[11px] text-[#9CA3AF]">Tracked Vehicle Identities</div>
                <div className="text-xl font-bold text-cyan-400 mt-1">{totalUniqueTracks}</div>
                <div className="text-[10px] text-[#6B7280] mt-0.5">Unique ByteTrack IDs</div>
              </div>

              <div className="p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937]">
                <div className="text-[11px] text-[#9CA3AF]">Total Detections</div>
                <div className="text-xl font-bold text-white mt-1">{totalDetections}</div>
                <div className="text-[10px] text-[#6B7280] mt-0.5">YOLO11n Bounding Boxes</div>
              </div>

              <div className="p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937]">
                <div className="text-[11px] text-[#9CA3AF]">Mean Confidence</div>
                <div className="text-xl font-bold text-emerald-400 mt-1">
                  {(meanConfidence * 100).toFixed(1)}%
                </div>
                <div className="text-[10px] text-[#6B7280] mt-0.5">Avg Model Score</div>
              </div>

              <div className="p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937]">
                <div className="text-[11px] text-[#9CA3AF]">Total Crossings</div>
                <div className="text-xl font-bold text-amber-400 mt-1">{totalCrossings}</div>
                <div className="text-[10px] text-[#6B7280] mt-0.5">Line Count Events</div>
              </div>
            </div>

            {/* Tracking Quality & Benchmark Disclosure */}
            <div className="p-3.5 rounded-lg bg-[#0E1524] border border-[#1F2937] space-y-2.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-[#E5E7EB] flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-cyan-400" />
                  Multi-Object Tracking Lifecycle Metrics
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/60 border border-[#374151] text-[#9CA3AF]">
                  Provenance: ByteTrack Kalman
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="p-2 rounded bg-[#0B0F17] border border-[#1F2937]">
                  <div className="text-[10px] text-[#9CA3AF]">Mostly Tracked (&gt;=80%)</div>
                  <div className="text-base font-bold text-white mt-0.5">
                    {tracking?.mostly_tracked_count !== null && tracking?.mostly_tracked_count !== undefined
                      ? tracking.mostly_tracked_count
                      : "N/A"}
                  </div>
                </div>

                <div className="p-2 rounded bg-[#0B0F17] border border-[#1F2937]">
                  <div className="text-[10px] text-[#9CA3AF]">Mostly Lost (&lt;=20%)</div>
                  <div className="text-base font-bold text-white mt-0.5">
                    {tracking?.mostly_lost_count !== null && tracking?.mostly_lost_count !== undefined
                      ? tracking.mostly_lost_count
                      : "N/A"}
                  </div>
                </div>

                <div className="p-2 rounded bg-[#0B0F17] border border-[#1F2937]">
                  <div className="text-[10px] text-[#9CA3AF]">ID Switches (IDSW)</div>
                  <div className="text-base font-bold text-amber-400 mt-0.5">
                    {tracking?.id_switches !== null && tracking?.id_switches !== undefined
                      ? tracking.id_switches
                      : "N/A"}
                  </div>
                </div>
              </div>

              {/* MOTA / HOTA / IDF1 disclosure badge */}
              <div className="p-2 rounded bg-[#161F30] border border-cyan-800/40 text-[11px] text-[#9CA3AF] flex items-start gap-2">
                <HelpCircle className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold text-cyan-300">
                    MOTA / HOTA / IDF1 Reference Disclosure:
                  </span>{" "}
                  MOTA, HOTA, and IDF1 are sequence-level ground-truth evaluation metrics computed against annotated test splits (e.g. UA-DETRAC benchmark). They are not available as live per-job values on unannotated user video inputs.
                </div>
              </div>
            </div>

            {/* Unified Vehicle Classification Architecture */}
            <div className="p-3.5 rounded-lg bg-[#0E1524] border border-[#1F2937] space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-[#E5E7EB] flex items-center gap-1.5">
                  <Car className="w-3.5 h-3.5 text-cyan-400" />
                  Vehicle Classification Architecture
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-700/60 text-cyan-300 font-bold uppercase">
                  Class-Agnostic Baseline
                </span>
              </div>

              <div className="p-2.5 rounded bg-[#0B0F17] border border-[#1F2937] flex items-center justify-between text-xs">
                <div>
                  <span className="text-white font-semibold">Active Operational Class:</span>
                  <span className="text-cyan-400 font-bold ml-2">Vehicle (Unified)</span>
                </div>
                <span className="text-[11px] px-2 py-0.5 rounded bg-[#161F30] text-[#9CA3AF]">
                  Total: <strong className="text-white">{totalDetections}</strong> detections
                </span>
              </div>

              <div className="p-2.5 rounded bg-[#111827] border border-[#1F2937] text-[11px] text-[#9CA3AF] leading-relaxed">
                <span className="text-amber-300/90 font-semibold">Classification Provenance:</span> The computer vision baseline unifies all road traffic detections into a single class-agnostic <code className="text-white">vehicle</code> category. Fine-grained vehicle subtypes (such as car, truck, bus, or motorcycle) are unvalidated under this baseline and are strictly not inferred from visual appearance.
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: FLOW & LINE CROSSINGS */}
        {activeTab === "flow" && (
          <div className="space-y-4">
            {/* Line Counts Breakdown */}
            <div className="space-y-2">
              <div className="text-xs font-semibold text-[#E5E7EB]">
                Virtual Counting Lines
              </div>
              {Object.keys(lineCounts).length === 0 ? (
                <div className="p-3 rounded bg-[#0B0F17] border border-[#1F2937] text-center text-[#6B7280]">
                  No virtual counting lines configured for this job.
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                  {Object.entries(lineCounts).map(([lineId, directions]) => (
                    <div
                      key={lineId}
                      className="p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937] space-y-1.5"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-cyan-400">{lineId}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#1F2937] text-[#9CA3AF]">
                          Total:{" "}
                          {Object.values(directions).reduce((s: number, c: number) => s + c, 0)}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-[11px] text-[#9CA3AF] pt-1 border-t border-[#1F2937]">
                        <span>Downstream: <strong className="text-emerald-400">{directions.downstream ?? 0}</strong></span>
                        <span>Upstream: <strong className="text-cyan-400">{directions.upstream ?? 0}</strong></span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Tumbling Flow Chart */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-[#E5E7EB] flex items-center gap-1.5">
                  <TrendingUp className="w-3.5 h-3.5 text-cyan-400" />
                  Traffic Throughput over Time (Vehicles per Minute)
                </span>
                <span className="text-[10px] text-[#9CA3AF]">
                  Cursor at: <strong className="text-cyan-400">{currentTimeSeconds.toFixed(2)}s</strong>
                </span>
              </div>

              {flowChartData.length === 0 ? (
                <div className="p-4 rounded bg-[#0B0F17] border border-[#1F2937] text-center text-[#6B7280]">
                  No flow windows computed. Video duration may be shorter than the tumbling window threshold.
                </div>
              ) : (
                <div className="h-48 w-full p-2 rounded-lg bg-[#0B0F17] border border-[#1F2937]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={flowChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" vertical={false} />
                      <XAxis
                        dataKey="name"
                        stroke="#6B7280"
                        fontSize={10}
                        tickLine={false}
                      />
                      <YAxis
                        stroke="#6B7280"
                        fontSize={10}
                        tickLine={false}
                        axisLine={false}
                        unit=" VPM"
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#0B0F17",
                          borderColor: "#374151",
                          borderRadius: "8px",
                          fontSize: "11px",
                          fontFamily: "monospace",
                        }}
                      />
                      <Bar
                        dataKey="flowVpm"
                        name="Throughput (VPM)"
                        fill="#06B6D4"
                        radius={[4, 4, 0, 0]}
                        onClick={(entry: any) => onSeek(entry.start)}
                        cursor="pointer"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 3: SPATIAL ZONES & CONGESTION */}
        {activeTab === "rois" && (
          <div className="space-y-4">
            <div className="p-3 rounded bg-amber-950/20 border border-amber-800/40 text-[11px] text-amber-200/90 flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <span>
                <strong>Heuristic Notice:</strong> Congestion states and occupancy ratios are derived rule-based estimates calculated from 2D bounding-box spatial coverage and pixel displacement.
              </span>
            </div>

            {/* ROI Density Chart */}
            <div className="space-y-2">
              <div className="text-xs font-semibold text-[#E5E7EB]">
                ROI Spatial Occupancy (Peak vs Average)
              </div>
              {roiChartData.length === 0 ? (
                <div className="p-4 rounded bg-[#0B0F17] border border-[#1F2937] text-center text-[#6B7280]">
                  No regions of interest (ROIs) registered for this video.
                </div>
              ) : (
                <div className="h-44 w-full p-2 rounded-lg bg-[#0B0F17] border border-[#1F2937]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={roiChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" vertical={false} />
                      <XAxis dataKey="roiId" stroke="#6B7280" fontSize={10} tickLine={false} />
                      <YAxis
                        stroke="#6B7280"
                        fontSize={10}
                        tickLine={false}
                        axisLine={false}
                        unit="%"
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#0B0F17",
                          borderColor: "#374151",
                          borderRadius: "8px",
                          fontSize: "11px",
                          fontFamily: "monospace",
                        }}
                      />
                      <Bar dataKey="peakPct" name="Peak Occupancy %" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="avgPct" name="Average Occupancy %" fill="#06B6D4" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            {/* Congestion Level Distribution Breakdown */}
            {Object.keys(congestionDistPerRoi).length > 0 && (
              <div className="space-y-2">
                <div className="text-xs font-semibold text-[#E5E7EB]">
                  Congestion Distribution by ROI
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                  {Object.entries(congestionDistPerRoi).map(([roiId, levels]) => (
                    <div
                      key={roiId}
                      className="p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937] space-y-1.5"
                    >
                      <div className="font-bold text-white text-xs">{roiId}</div>
                      <div className="grid grid-cols-3 gap-1.5 text-center text-[10px]">
                        <div className="p-1 rounded bg-emerald-950/40 border border-emerald-800 text-emerald-300">
                          Low: {levels.low ?? 0}f
                        </div>
                        <div className="p-1 rounded bg-amber-950/40 border border-amber-800 text-amber-300">
                          Mod: {levels.moderate ?? 0}f
                        </div>
                        <div className="p-1 rounded bg-red-950/40 border border-red-800 text-red-300">
                          High: {levels.high ?? 0}f
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 4: STATIONARY VEHICLE EVENTS */}
        {activeTab === "events" && (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs text-[#9CA3AF]">
              <span>Stationary Vehicle Log</span>
              <span>Total Events: <strong className="text-white">{stoppedEvents.length}</strong></span>
            </div>

            {stoppedEvents.length === 0 ? (
              <div className="p-4 rounded-lg bg-[#0B0F17] border border-[#1F2937] text-center text-[#6B7280]">
                No stationary vehicle events recorded for this job.
              </div>
            ) : (
              <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
                {stoppedEvents.map((evt, idx) => {
                  const end = evt.end_timestamp_s ?? evt.start_timestamp_s + evt.duration_s;
                  const isCurrent =
                    currentTimeSeconds >= evt.start_timestamp_s && currentTimeSeconds <= end;
                  return (
                    <div
                      key={`stop-row-${idx}`}
                      className={`p-2.5 rounded-lg border transition-all flex flex-wrap items-center justify-between gap-3 ${
                        isCurrent
                          ? "bg-red-950/40 border-red-500 ring-1 ring-red-500"
                          : "bg-[#0B0F17] border-[#1F2937] hover:border-[#374151]"
                      }`}
                    >
                      <div className="space-y-0.5">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-red-300">Track #{evt.track_id}</span>
                          <span
                            className={`px-1.5 py-0.2 rounded text-[10px] uppercase font-bold ${
                              evt.status === "active"
                                ? "bg-amber-950/80 text-amber-400 border border-amber-800"
                                : "bg-emerald-950/80 text-emerald-400 border border-emerald-800"
                            }`}
                          >
                            {evt.status}
                          </span>
                          {evt.roi_id && (
                            <span className="text-[10px] text-[#9CA3AF]">in {evt.roi_id}</span>
                          )}
                        </div>
                        <div className="text-[11px] text-[#9CA3AF]">
                          Stationary duration: <strong className="text-white">{evt.duration_s}s</strong> • From{" "}
                          {evt.start_timestamp_s.toFixed(2)}s to {end.toFixed(2)}s (Frame {evt.start_frame})
                        </div>
                      </div>

                      <button
                        type="button"
                        onClick={() => onSeek(evt.start_timestamp_s)}
                        className="px-2.5 py-1 rounded bg-red-900/40 hover:bg-red-800 text-red-200 border border-red-700/60 transition-colors flex items-center gap-1.5 text-[11px] font-semibold"
                      >
                        <Navigation className="w-3 h-3 rotate-45" />
                        <span>Seek to Event</span>
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* TAB 5: SCENE UNDERSTANDING (VLM) */}
        {activeTab === "vlm" && (
          <SceneInterpretationPanel
            jobId={jobId}
            currentTimeSeconds={currentTimeSeconds}
            onSeek={onSeek}
          />
        )}

        {/* TAB 6: EVIDENCE-GROUNDED INCIDENT REPORT */}
        {activeTab === "report" && (
          <IncidentReportPanel
            jobId={jobId}
            currentTimeSeconds={currentTimeSeconds}
            onSeek={onSeek}
          />
        )}
      </div>
    </div>
  );
};
