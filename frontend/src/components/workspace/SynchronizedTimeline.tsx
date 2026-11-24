import React from "react";
import { Clock, AlertTriangle, TrendingUp, Navigation } from "lucide-react";
import { StoppedVehicleEvent, LineFlow } from "../../types/telemetry";

export interface SynchronizedTimelineProps {
  durationSeconds: number;
  currentTimeSeconds: number;
  fps: number;
  stoppedEvents: StoppedVehicleEvent[];
  flowWindows: LineFlow[];
  lineCounts?: Record<string, Record<string, number>>;
  onSeek: (timeSeconds: number) => void;
}

export const SynchronizedTimeline: React.FC<SynchronizedTimelineProps> = ({
  durationSeconds,
  currentTimeSeconds,
  fps,
  stoppedEvents,
  flowWindows,
  lineCounts,
  onSeek,
}) => {
  const currentFrame = Math.max(1, Math.floor(currentTimeSeconds * fps) + 1);

  // Active stopped events at current playback time
  const activeStoppedNow = stoppedEvents.filter((e) => {
    const end = e.end_timestamp_s ?? e.start_timestamp_s + e.duration_s;
    return currentTimeSeconds >= e.start_timestamp_s && currentTimeSeconds <= end;
  });

  // Active flow window at current playback time
  const activeFlowNow = flowWindows.find(
    (w) =>
      currentTimeSeconds >= w.window_start_s && currentTimeSeconds <= w.window_end_s
  );

  return (
    <div className="rounded-xl bg-[#111827] border border-[#1F2937] p-3.5 space-y-3 font-mono text-xs">
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-[#1F2937]">
        <div className="flex items-center gap-2 text-cyan-300 font-bold">
          <Clock className="w-4 h-4 text-cyan-400" />
          <span>SYNCHRONIZED EVENT TIMELINE</span>
        </div>
        <div className="flex items-center gap-3 text-[11px]">
          <span className="text-[#9CA3AF]">
            Time: <span className="text-white font-semibold">{currentTimeSeconds.toFixed(2)}s</span> / {durationSeconds.toFixed(2)}s
          </span>
          <span className="text-[#4B5563]">|</span>
          <span className="text-[#9CA3AF]">
            Frame: <span className="text-cyan-400 font-semibold">{currentFrame}</span>
          </span>
        </div>
      </div>

      {/* Active telemetry callouts at current timestamp */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[11px]">
        {/* Active Stopped Event Notice */}
        <div
          className={`p-2 rounded-lg border flex items-center justify-between gap-2 transition-all ${
            activeStoppedNow.length > 0
              ? "bg-red-950/40 border-red-800 text-red-300"
              : "bg-[#0B0F17]/60 border-[#1F2937] text-[#6B7280]"
          }`}
        >
          <div className="flex items-center gap-2">
            <AlertTriangle
              className={`w-3.5 h-3.5 ${
                activeStoppedNow.length > 0 ? "text-red-400 animate-pulse" : "text-[#4B5563]"
              }`}
            />
            <span>
              {activeStoppedNow.length > 0
                ? `${activeStoppedNow.length} Stationary Vehicle(s) Active`
                : "No Stationary Vehicles Active"}
            </span>
          </div>
          {activeStoppedNow.length > 0 && (
            <span className="px-1.5 py-0.5 rounded bg-red-900/60 text-red-200 text-[10px] font-bold">
              ID {activeStoppedNow.map((e) => `#${e.track_id}`).join(", ")}
            </span>
          )}
        </div>

        {/* Active Flow Window Throughput Notice */}
        <div
          className={`p-2 rounded-lg border flex items-center justify-between gap-2 transition-all ${
            activeFlowNow
              ? "bg-cyan-950/30 border-cyan-800/60 text-cyan-300"
              : "bg-[#0B0F17]/60 border-[#1F2937] text-[#6B7280]"
          }`}
        >
          <div className="flex items-center gap-2">
            <TrendingUp
              className={`w-3.5 h-3.5 ${
                activeFlowNow ? "text-cyan-400" : "text-[#4B5563]"
              }`}
            />
            <span>
              {activeFlowNow
                ? `Window: ${activeFlowNow.window_start_s}s - ${activeFlowNow.window_end_s}s`
                : "Awaiting Flow Window"}
            </span>
          </div>
          {activeFlowNow && (
            <span className="px-1.5 py-0.5 rounded bg-cyan-900/50 text-cyan-200 text-[10px] font-bold">
              {activeFlowNow.flow_rate_vpm.toFixed(1)} VPM
            </span>
          )}
        </div>
      </div>

      {/* Stopped-Vehicle Event Navigator Track */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-[11px] text-[#9CA3AF]">
          <span className="font-semibold text-red-400 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-red-500" />
            Stationary Vehicle Events ({stoppedEvents.length})
          </span>
          <span className="text-[10px] text-[#6B7280]">Click event to seek video</span>
        </div>

        {stoppedEvents.length === 0 ? (
          <div className="p-2 rounded bg-[#0B0F17] border border-[#1F2937] text-[#6B7280] text-center text-[11px]">
            No stationary vehicle events detected in this sequence.
          </div>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {stoppedEvents.map((evt, idx) => {
              const end = evt.end_timestamp_s ?? evt.start_timestamp_s + evt.duration_s;
              const isCurrent =
                currentTimeSeconds >= evt.start_timestamp_s && currentTimeSeconds <= end;
              return (
                <button
                  key={`stop-evt-${idx}`}
                  type="button"
                  onClick={() => onSeek(evt.start_timestamp_s)}
                  className={`px-2 py-1 rounded text-[11px] font-medium transition-all flex items-center gap-1.5 border ${
                    isCurrent
                      ? "bg-red-950 border-red-500 text-red-200 ring-1 ring-red-500 shadow-md"
                      : "bg-[#0B0F17] border-[#374151] text-[#D1D5DB] hover:bg-[#1F2937] hover:border-red-400"
                  }`}
                  title={`Seek to ${evt.start_timestamp_s.toFixed(2)}s (Duration: ${evt.duration_s}s)`}
                >
                  <Navigation className="w-3 h-3 text-red-400 rotate-45" />
                  <span>Track #{evt.track_id}</span>
                  <span className="text-[#9CA3AF]">({evt.duration_s}s @ {evt.start_timestamp_s.toFixed(1)}s)</span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Flow Windows Navigator Track */}
      {flowWindows.length > 0 && (
        <div className="space-y-1.5 pt-1">
          <div className="flex items-center justify-between text-[11px] text-[#9CA3AF]">
            <span className="font-semibold text-cyan-400 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              Tumbling Flow Windows ({flowWindows.length})
            </span>
            <span className="text-[10px] text-[#6B7280]">Click window to seek video</span>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {flowWindows.map((win, idx) => {
              const isCurrent =
                currentTimeSeconds >= win.window_start_s &&
                currentTimeSeconds <= win.window_end_s;
              return (
                <button
                  key={`flow-win-${idx}`}
                  type="button"
                  onClick={() => onSeek(win.window_start_s)}
                  className={`px-2 py-1 rounded text-[11px] font-medium transition-all flex items-center gap-1.5 border ${
                    isCurrent
                      ? "bg-cyan-950 border-cyan-400 text-cyan-200 ring-1 ring-cyan-400 shadow-md"
                      : "bg-[#0B0F17] border-[#374151] text-[#D1D5DB] hover:bg-[#1F2937] hover:border-cyan-400"
                  }`}
                  title={`Seek to window start at ${win.window_start_s}s`}
                >
                  <TrendingUp className="w-3 h-3 text-cyan-400" />
                  <span>
                    W{idx + 1}: {win.window_start_s}s-{win.window_end_s}s
                  </span>
                  <span className="text-cyan-300 font-bold">
                    {win.flow_rate_vpm.toFixed(1)} VPM
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Aggregate Line Crossings (Sequence Total) */}
      {lineCounts && Object.keys(lineCounts).length > 0 && (
        <div className="space-y-1.5 pt-2 border-t border-[#1F2937]/70">
          <div className="flex items-center justify-between text-[11px] text-[#9CA3AF]">
            <span className="font-semibold text-emerald-400 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              Aggregate Virtual Line Crossings (Sequence Total)
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/60 border border-[#374151] text-[#9CA3AF]">
              Sequence Counts (No Frame Timestamps)
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[11px]">
            {Object.entries(lineCounts).map(([lineId, directions]) => {
              const total = Object.values(directions).reduce((a, b) => a + b, 0);
              return (
                <div
                  key={lineId}
                  className="p-2 rounded bg-[#0B0F17] border border-[#1F2937] flex items-center justify-between"
                >
                  <div>
                    <span className="font-bold text-cyan-400">{lineId}:</span>
                    <span className="text-[#9CA3AF] ml-2">
                      Downstream: <strong className="text-emerald-400">{directions.downstream ?? 0}</strong> • Upstream: <strong className="text-cyan-300">{directions.upstream ?? 0}</strong>
                    </span>
                  </div>
                  <span className="px-2 py-0.5 rounded bg-[#161F30] font-bold text-white">
                    {total} Total
                  </span>
                </div>
              );
            })}
          </div>

          <p className="text-[10px] text-[#6B7280] leading-relaxed">
            Note: Cumulative line crossings are aggregated across all video frames by the CV pipeline. Individual frame-level crossing timestamps are not exposed in the results summary and are not fabricated; throughput is summarized in the tumbling flow windows above.
          </p>
        </div>
      )}
    </div>
  );
};
