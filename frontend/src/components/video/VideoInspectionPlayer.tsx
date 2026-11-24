import React, { useRef, useState, useEffect, useCallback } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
  Maximize2,
  Minimize2,
  AlertTriangle,
  RotateCw,
  Film,
  SplitSquareHorizontal,
  Video,
} from "lucide-react";
import { useWorkspaceStore } from "../../state/useWorkspaceStore";
import { videoService } from "../../services/videoService";
import { jobService } from "../../services/jobService";
import { StoppedVehicleEvent, LineFlow, AnnotationDiagnostics } from "../../types/telemetry";

export interface VideoInspectionPlayerProps {
  videoId: string;
  jobId: string;
  fps?: number;
  totalFrames?: number;
  durationSeconds?: number;
  isAnnotatedAvailable: boolean;
  annotationDiagnostics?: AnnotationDiagnostics | null;
  stoppedEvents?: StoppedVehicleEvent[];
  flowWindows?: LineFlow[];
  onTimeUpdate?: (timeSeconds: number, frameIdx: number) => void;
}

export const VideoInspectionPlayer: React.FC<VideoInspectionPlayerProps> = ({
  videoId,
  jobId,
  fps = 25.0,
  totalFrames = 75,
  durationSeconds,
  isAnnotatedAvailable,
  annotationDiagnostics,
  stoppedEvents = [],
  flowWindows = [],
  onTimeUpdate,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const primaryVideoRef = useRef<HTMLVideoElement>(null);
  const secondaryVideoRef = useRef<HTMLVideoElement>(null);

  const {
    currentFrame,
    currentTimeSeconds,
    isPlaying,
    playbackRate,
    viewMode,
    setFrame,
    setTime,
    setIsPlaying,
    setPlaybackRate,
    setViewMode,
  } = useWorkspaceStore();

  const [duration, setDuration] = useState<number>(durationSeconds || 3.0);
  const [isBuffering, setIsBuffering] = useState<boolean>(false);
  const [primaryError, setPrimaryError] = useState<string | null>(null);
  const [secondaryError, setSecondaryError] = useState<string | null>(null);
  const [volume, setVolume] = useState<number>(1.0);
  const [isMuted, setIsMuted] = useState<boolean>(true);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [bufferedPercent, setBufferedPercent] = useState<number>(0);
  const [hoverTime, setHoverTime] = useState<number | null>(null);

  // Synchronization tolerance thresholds for side-by-side mode:
  // - Target: <= 0.040s (1 frame at 25fps)
  // - Drift-compensation activation: > 0.030s (smooth proportional playbackRate nudge)
  // - Hard-seek snap threshold: > 0.250s (re-synchronizes currentTime if severe desync occurs)
  // - Rate-restore deadband: <= 0.015s (returns secondary playbackRate to base rate)
  const DRIFT_ACTIVATE_THRESHOLD = 0.030;
  const HARD_SEEK_THRESHOLD = 0.250;
  const RATE_RESTORE_THRESHOLD = 0.015;

  const sourceVideoUrl = videoService.getVideoContentUrl(videoId);
  const annotatedVideoUrl = jobService.getAnnotatedVideoUrl(jobId);

  // If user selected "annotated" view but the artifact is physically unavailable,
  // we gracefully fallback to source video.
  const activeEffectiveMode =
    viewMode === "annotated" && !isAnnotatedAvailable ? "source" : viewMode;

  // Format seconds to mm:ss.SS
  const formatTimecode = (sec: number): string => {
    if (!Number.isFinite(sec) || sec < 0) return "00:00.00";
    const minutes = Math.floor(sec / 60);
    const seconds = Math.floor(sec % 60);
    const centiseconds = Math.floor((sec % 1) * 100);
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(centiseconds).padStart(2, "0")}`;
  };

  // Synchronize playback rate
  useEffect(() => {
    if (primaryVideoRef.current) {
      primaryVideoRef.current.playbackRate = playbackRate;
    }
    if (secondaryVideoRef.current) {
      secondaryVideoRef.current.playbackRate = playbackRate;
    }
  }, [playbackRate]);

  // Cleanup resources and pause playback on unmount or mode switch
  useEffect(() => {
    return () => {
      if (primaryVideoRef.current) primaryVideoRef.current.pause();
      if (secondaryVideoRef.current) secondaryVideoRef.current.pause();
    };
  }, [activeEffectiveMode]);

  // Synchronize play/pause
  useEffect(() => {
    const primary = primaryVideoRef.current;
    const secondary = secondaryVideoRef.current;
    if (!primary) return;

    if (isPlaying) {
      primary.play().catch((err) => {
        // Autoplay policy or interrupt
        console.warn("Primary video play interrupted:", err);
        setIsPlaying(false);
      });
      if (secondary && activeEffectiveMode === "side-by-side") {
        secondary.play().catch(() => {});
      }
    } else {
      primary.pause();
      if (secondary) secondary.pause();
    }
  }, [isPlaying, activeEffectiveMode, setIsPlaying]);

  // Dedicated high-frequency drift correction loop for side-by-side mode (runs at display refresh rate ~60Hz)
  useEffect(() => {
    if (!isPlaying || activeEffectiveMode !== "side-by-side") {
      return;
    }

    let animationFrameId: number;

    const syncLoop = () => {
      const primary = primaryVideoRef.current;
      const secondary = secondaryVideoRef.current;

      if (primary && secondary && !primary.paused && !secondary.paused) {
        // diff > 0 means primary is ahead (secondary is lagging) -> speed up secondary
        // diff < 0 means secondary is ahead (secondary is leading) -> slow down secondary
        const diff = primary.currentTime - secondary.currentTime;
        const absDiff = Math.abs(diff);

        if (absDiff > HARD_SEEK_THRESHOLD) {
          // Hard-seek snap threshold: snap secondary to primary currentTime
          secondary.currentTime = primary.currentTime;
          secondary.playbackRate = playbackRate;
        } else if (absDiff > DRIFT_ACTIVATE_THRESHOLD) {
          // Drift-compensation activation threshold: proportional nudge (up to ±15%)
          const nudge = Math.max(-0.15, Math.min(0.15, diff * 1.5));
          secondary.playbackRate = playbackRate * (1 + nudge);
        } else if (absDiff <= RATE_RESTORE_THRESHOLD) {
          // In-sync deadband: restore secondary rate to match user playbackRate
          if (secondary.playbackRate !== playbackRate) {
            secondary.playbackRate = playbackRate;
          }
        }
      }

      animationFrameId = requestAnimationFrame(syncLoop);
    };

    animationFrameId = requestAnimationFrame(syncLoop);

    return () => {
      cancelAnimationFrame(animationFrameId);
      if (secondaryVideoRef.current && secondaryVideoRef.current.playbackRate !== playbackRate) {
        secondaryVideoRef.current.playbackRate = playbackRate;
      }
    };
  }, [isPlaying, activeEffectiveMode, playbackRate]);

  // Coordinated waiting and buffering handlers for multi-stream lock
  const handleWaiting = (stream: "primary" | "secondary") => {
    setIsBuffering(true);
    // If one stream stalls during side-by-side playback, pause the other to prevent runaway drift
    if (activeEffectiveMode === "side-by-side") {
      if (stream === "primary" && secondaryVideoRef.current && !secondaryVideoRef.current.paused) {
        secondaryVideoRef.current.pause();
      } else if (stream === "secondary" && primaryVideoRef.current && !primaryVideoRef.current.paused) {
        primaryVideoRef.current.pause();
      }
    }
  };

  const handlePlayingOrCanPlay = () => {
    const primary = primaryVideoRef.current;
    const secondary = secondaryVideoRef.current;
    const primaryReady = primary ? primary.readyState >= 3 : false;
    const secondaryReady =
      activeEffectiveMode !== "side-by-side" || (secondary ? secondary.readyState >= 3 : true);

    if (primaryReady && secondaryReady) {
      setIsBuffering(false);
      if (isPlaying) {
        primary?.play().catch(() => {});
        if (activeEffectiveMode === "side-by-side") {
          secondary?.play().catch(() => {});
        }
      }
    }
  };

  // Video end handler: cleanly stops playback when either stream concludes
  const handleEnded = () => {
    setIsPlaying(false);
    if (primaryVideoRef.current) primaryVideoRef.current.pause();
    if (secondaryVideoRef.current) secondaryVideoRef.current.pause();
  };

  // Handle media time update for UI timeline, frame counters, and buffer progress
  const handleTimeUpdate = useCallback(() => {
    const video = primaryVideoRef.current;
    if (!video) return;

    const time = video.currentTime;
    const frame = Math.min(
      totalFrames,
      Math.max(1, Math.floor(time * fps) + 1)
    );

    setTime(time);
    setFrame(frame);
    if (onTimeUpdate) {
      onTimeUpdate(time, frame);
    }

    // Update buffer progress
    if (video.buffered.length > 0 && duration > 0) {
      try {
        const bufferedEnd = video.buffered.end(video.buffered.length - 1);
        setBufferedPercent(Math.min(100, (bufferedEnd / duration) * 100));
      } catch {
        // Buffered range read error
      }
    }
  }, [fps, totalFrames, duration, setTime, setFrame, onTimeUpdate]);

  // Synchronize external seeks (e.g. user clicked on timeline marker or telemetry row)
  useEffect(() => {
    const primary = primaryVideoRef.current;
    if (!primary) return;
    if (Math.abs(primary.currentTime - currentTimeSeconds) > 0.15) {
      const clamped = Math.max(0, Math.min(duration, currentTimeSeconds));
      primary.currentTime = clamped;
      if (secondaryVideoRef.current) {
        secondaryVideoRef.current.currentTime = clamped;
      }
    }
  }, [currentTimeSeconds, duration]);

  const handleLoadedMetadata = () => {
    const primary = primaryVideoRef.current;
    if (!primary) return;
    if (Number.isFinite(primary.duration) && primary.duration > 0) {
      setDuration(primary.duration);
    }
    setPrimaryError(null);
    setIsBuffering(false);
  };

  const handleSeek = (targetTime: number) => {
    const clamped = Math.max(0, Math.min(duration, targetTime));
    if (primaryVideoRef.current) {
      primaryVideoRef.current.currentTime = clamped;
    }
    if (secondaryVideoRef.current) {
      secondaryVideoRef.current.currentTime = clamped;
    }
    const frame = Math.min(
      totalFrames,
      Math.max(1, Math.floor(clamped * fps) + 1)
    );
    setTime(clamped);
    setFrame(frame);
    if (onTimeUpdate) {
      onTimeUpdate(clamped, frame);
    }
  };

  const stepFrame = (direction: -1 | 1) => {
    const frameDuration = 1.0 / fps;
    const nextTime = currentTimeSeconds + direction * frameDuration;
    handleSeek(nextTime);
  };

  const togglePlay = () => {
    setIsPlaying(!isPlaying);
  };

  const handleScrubberClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    const targetTime = pos * duration;
    handleSeek(targetTime);
  };

  const handleScrubberMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const pos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    setHoverTime(pos * duration);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setVolume(val);
    setIsMuted(val === 0);
    if (primaryVideoRef.current) {
      primaryVideoRef.current.volume = val;
      primaryVideoRef.current.muted = val === 0;
    }
    if (secondaryVideoRef.current) {
      secondaryVideoRef.current.volume = val;
      secondaryVideoRef.current.muted = val === 0;
    }
  };

  const toggleMute = () => {
    const nextMuted = !isMuted;
    setIsMuted(nextMuted);
    if (primaryVideoRef.current) {
      primaryVideoRef.current.muted = nextMuted;
    }
    if (secondaryVideoRef.current) {
      secondaryVideoRef.current.muted = nextMuted;
    }
  };

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
    }
  };

  // Keyboard controls (Space to play/pause, Left/Right for frame stepping)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input
      if (["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }
      if (e.code === "Space") {
        e.preventDefault();
        togglePlay();
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        stepFrame(-1);
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        stepFrame(1);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isPlaying, currentTimeSeconds, duration, fps]);

  return (
    <div
      ref={containerRef}
      className="relative rounded-xl bg-black border border-[#1F2937] overflow-hidden flex flex-col justify-between shadow-2xl font-mono select-none"
    >
      {/* Top HUD bar: View Mode Switcher & Stream Badges */}
      <div className="p-3 bg-gradient-to-b from-black/95 via-black/60 to-transparent flex flex-wrap items-center justify-between gap-3 z-20 text-xs border-b border-[#1F2937]/50">
        <div className="flex items-center gap-2.5">
          <span
            className={`w-2.5 h-2.5 rounded-full ${
              isPlaying
                ? "bg-emerald-400 animate-pulse"
                : "bg-cyan-400 shadow-[0_0_8px_rgba(6,182,212,0.6)]"
            }`}
          />
          <span className="font-bold tracking-wider text-cyan-300">
            {activeEffectiveMode === "side-by-side"
              ? "DUAL SYNC VIEW"
              : activeEffectiveMode === "annotated"
              ? "ANNOTATED CV VIEW"
              : "SOURCE VIDEO VIEW"}
          </span>
          <span className="text-[#4B5563]">|</span>
          <span className="text-[#9CA3AF] text-[11px]">
            {totalFrames} frames • {fps.toFixed(1)} FPS
          </span>
        </div>

        {/* View Mode Switcher Pills */}
        <div className="flex items-center gap-1 bg-[#111827]/90 p-1 rounded-lg border border-[#374151]">
          <button
            type="button"
            onClick={() => setViewMode("annotated")}
            disabled={!isAnnotatedAvailable}
            className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-all flex items-center gap-1.5 ${
              viewMode === "annotated" && isAnnotatedAvailable
                ? "bg-cyan-500 text-black shadow-sm"
                : isAnnotatedAvailable
                ? "text-[#D1D5DB] hover:text-white hover:bg-[#1F2937]"
                : "text-[#6B7280] opacity-50 cursor-not-allowed"
            }`}
            title={
              isAnnotatedAvailable
                ? "Inspect YOLO11n + ByteTrack rendered detections and trajectories"
                : "Annotated video artifact not available for this job"
            }
          >
            <Film className="w-3.5 h-3.5" />
            <span>Annotated CV</span>
          </button>

          <button
            type="button"
            onClick={() => setViewMode("source")}
            className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-all flex items-center gap-1.5 ${
              viewMode === "source"
                ? "bg-cyan-500 text-black shadow-sm"
                : "text-[#D1D5DB] hover:text-white hover:bg-[#1F2937]"
            }`}
            title="Inspect clean unannotated original source video"
          >
            <Video className="w-3.5 h-3.5" />
            <span>Original Source</span>
          </button>

          <button
            type="button"
            onClick={() => setViewMode("side-by-side")}
            disabled={!isAnnotatedAvailable}
            className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-all flex items-center gap-1.5 ${
              viewMode === "side-by-side" && isAnnotatedAvailable
                ? "bg-cyan-500 text-black shadow-sm"
                : isAnnotatedAvailable
                ? "text-[#D1D5DB] hover:text-white hover:bg-[#1F2937]"
                : "text-[#6B7280] opacity-50 cursor-not-allowed"
            }`}
            title={
              isAnnotatedAvailable
                ? "Synchronized side-by-side comparison"
                : "Side-by-side requires both source and annotated video"
            }
          >
            <SplitSquareHorizontal className="w-3.5 h-3.5" />
            <span>Side-by-Side</span>
          </button>
        </div>
      </div>

      {/* Main Video Viewport Canvas */}
      <div className="relative flex-1 bg-[#05070A] min-h-[360px] md:min-h-[460px] flex items-center justify-center overflow-hidden">
        {/* Fallback Notice if annotated was requested but unavailable */}
        {viewMode === "annotated" && !isAnnotatedAvailable && (
          <div className="absolute top-4 left-4 right-4 z-20 p-2.5 rounded-lg bg-amber-950/80 border border-amber-600/70 text-xs text-amber-200 flex items-center justify-between gap-2 backdrop-blur-md">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
              <span>
                {annotationDiagnostics?.error_message
                  ? `Annotated video rendering unavailable (${annotationDiagnostics.error_message}). Optical detections and traffic telemetry completed successfully.`
                  : "Annotated video artifact is not available. Displaying original source video stream instead."}
              </span>
            </div>
            <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-amber-900/60 border border-amber-500/40 text-amber-300 font-semibold shrink-0">
              CV Analytics Intact
            </span>
          </div>
        )}

        {/* Viewport Rendering: Single or Dual View */}
        {activeEffectiveMode === "side-by-side" ? (
          <div className="w-full h-full grid grid-cols-1 md:grid-cols-2 gap-2 p-2 bg-[#090D14]">
            {/* Left: Original Source */}
            <div className="relative rounded-lg overflow-hidden bg-black border border-[#1F2937] flex flex-col justify-center">
              <div className="absolute top-2 left-2 z-10 px-2 py-0.5 rounded bg-black/75 border border-[#374151] text-[10px] text-[#9CA3AF] font-bold">
                ORIGINAL SOURCE
              </div>
              <video
                ref={secondaryVideoRef}
                src={sourceVideoUrl}
                playsInline
                muted={isMuted}
                preload="metadata"
                className="w-full h-full object-contain cursor-pointer"
                onClick={togglePlay}
                onWaiting={() => handleWaiting("secondary")}
                onPlaying={handlePlayingOrCanPlay}
                onCanPlay={handlePlayingOrCanPlay}
                onEnded={handleEnded}
                onError={() => setSecondaryError("Failed to stream source video")}
              />
              {secondaryError && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/90 p-4 text-center text-xs text-red-400">
                  <AlertTriangle className="w-5 h-5 mx-auto mb-1 text-red-500" />
                  {secondaryError}
                </div>
              )}
            </div>

            {/* Right: Annotated CV */}
            <div className="relative rounded-lg overflow-hidden bg-black border border-cyan-900/60 flex flex-col justify-center">
              <div className="absolute top-2 left-2 z-10 px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-600/70 text-[10px] text-cyan-300 font-bold">
                ANNOTATED CV (YOLO11n + BYTETRACK)
              </div>
              <video
                ref={primaryVideoRef}
                src={annotatedVideoUrl}
                playsInline
                muted={isMuted}
                preload="metadata"
                className="w-full h-full object-contain cursor-pointer"
                onClick={togglePlay}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onWaiting={() => handleWaiting("primary")}
                onPlaying={handlePlayingOrCanPlay}
                onCanPlay={handlePlayingOrCanPlay}
                onEnded={handleEnded}
                onError={() => setPrimaryError("Failed to stream annotated video")}
              />
              {primaryError && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/90 p-4 text-center text-xs text-red-400">
                  <AlertTriangle className="w-5 h-5 mx-auto mb-1 text-red-500" />
                  {primaryError}
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Single Viewport Mode (either Annotated or Source) */
          <div className="relative w-full h-full flex items-center justify-center bg-black">
            <video
              ref={primaryVideoRef}
              src={activeEffectiveMode === "annotated" ? annotatedVideoUrl : sourceVideoUrl}
              playsInline
              muted={isMuted}
              preload="metadata"
              className="w-full h-full object-contain cursor-pointer max-h-[580px]"
              onClick={togglePlay}
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onWaiting={() => handleWaiting("primary")}
              onPlaying={handlePlayingOrCanPlay}
              onCanPlay={handlePlayingOrCanPlay}
              onEnded={handleEnded}
              onError={() =>
                setPrimaryError(
                  activeEffectiveMode === "annotated"
                    ? "Annotated video stream failed to load"
                    : "Source video stream failed to load"
                )
              }
            />

            {/* Buffering Overlay */}
            {isBuffering && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-[2px] z-10">
                <div className="flex flex-col items-center gap-2 p-4 rounded-xl bg-black/80 border border-[#374151] text-xs text-cyan-300">
                  <RotateCw className="w-6 h-6 animate-spin text-cyan-400" />
                  <span>Buffering video via HTTP Range...</span>
                </div>
              </div>
            )}

            {/* Error Overlay */}
            {primaryError && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/90 p-6 text-center z-10 space-y-3">
                <div className="w-12 h-12 rounded-full bg-red-950/80 border border-red-500 flex items-center justify-center text-red-400">
                  <AlertTriangle className="w-6 h-6" />
                </div>
                <div className="max-w-md">
                  <h3 className="text-sm font-bold text-white">Media Playback Error</h3>
                  <p className="text-xs text-red-300 mt-1">{primaryError}</p>
                </div>
                {activeEffectiveMode === "annotated" && (
                  <button
                    type="button"
                    onClick={() => {
                      setPrimaryError(null);
                      setViewMode("source");
                    }}
                    className="px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-black text-xs font-bold transition-colors"
                  >
                    Switch to Source Video
                  </button>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Bottom HUD: Scrubber, Timeline Event Markers & Transport Controls */}
      <div className="p-3 bg-[#111827]/98 border-t border-[#1F2937] z-20 space-y-2.5">
        {/* Interactive Scrubber with Event Markers */}
        <div className="relative group pt-1 pb-1">
          <div
            className="relative w-full h-2 bg-[#1F2937] rounded-full overflow-hidden cursor-pointer transition-all group-hover:h-3"
            onClick={handleScrubberClick}
            onMouseMove={handleScrubberMouseMove}
            onMouseLeave={() => setHoverTime(null)}
          >
            {/* Buffered progress track */}
            <div
              className="absolute top-0 bottom-0 left-0 bg-[#374151] transition-all"
              style={{ width: `${bufferedPercent}%` }}
              title={`Buffered: ${bufferedPercent.toFixed(0)}%`}
            />

            {/* Current playback playhead */}
            <div
              className="absolute top-0 bottom-0 left-0 bg-gradient-to-r from-cyan-600 to-cyan-400 shadow-[0_0_8px_rgba(6,182,212,0.8)]"
              style={{
                width: `${duration > 0 ? (currentTimeSeconds / duration) * 100 : 0}%`,
              }}
            />

            {/* Flow window markers on scrubber */}
            {flowWindows.map((win, idx) => {
              if (duration <= 0) return null;
              const leftPct = (win.window_start_s / duration) * 100;
              const widthPct = Math.max(0.6, ((win.window_end_s - win.window_start_s) / duration) * 100);
              return (
                <div
                  key={`scrub-flow-${idx}`}
                  className="absolute top-0 bottom-0 bg-cyan-400/25 border-l border-cyan-400 z-10 cursor-pointer pointer-events-none"
                  style={{
                    left: `${Math.min(99, Math.max(0, leftPct))}%`,
                    width: `${Math.min(100 - leftPct, widthPct)}%`,
                  }}
                  title={`Flow W${idx + 1}: ${win.flow_rate_vpm.toFixed(1)} VPM`}
                />
              );
            })}

            {/* Stopped vehicle event markers on scrubber */}
            {stoppedEvents.map((evt, idx) => {
              if (duration <= 0) return null;
              const leftPct = (evt.start_timestamp_s / duration) * 100;
              const widthPct = Math.max(0.8, (evt.duration_s / duration) * 100);
              return (
                <div
                  key={`scrub-stop-${idx}`}
                  className="absolute top-0 bottom-0 bg-red-500/80 hover:bg-red-400 z-10 cursor-pointer"
                  style={{
                    left: `${Math.min(99, Math.max(0, leftPct))}%`,
                    width: `${Math.min(100 - leftPct, widthPct)}%`,
                  }}
                  title={`Track #${evt.track_id} Stopped: ${evt.duration_s}s at ${evt.start_timestamp_s.toFixed(2)}s`}
                />
              );
            })}
          </div>

          {/* Hover preview tooltip */}
          {hoverTime !== null && (
            <div
              className="absolute -top-7 px-1.5 py-0.5 rounded bg-black/90 border border-cyan-500/60 text-[10px] text-cyan-300 pointer-events-none transform -translate-x-1/2"
              style={{
                left: `${duration > 0 ? (hoverTime / duration) * 100 : 0}%`,
              }}
            >
              {formatTimecode(hoverTime)}
            </div>
          )}
        </div>

        {/* Transport Controls Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
          {/* Left Transport: Play/Pause, Frame Step, Timecode */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={togglePlay}
              className="p-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-black transition-colors shadow-sm"
              title={isPlaying ? "Pause (Space)" : "Play (Space)"}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-black" />}
            </button>

            <button
              type="button"
              onClick={() => stepFrame(-1)}
              className="p-1.5 rounded-md text-[#9CA3AF] hover:text-white hover:bg-[#1F2937] transition-colors"
              title="Previous Frame (Arrow Left)"
            >
              <SkipBack className="w-3.5 h-3.5" />
            </button>

            <button
              type="button"
              onClick={() => stepFrame(1)}
              className="p-1.5 rounded-md text-[#9CA3AF] hover:text-white hover:bg-[#1F2937] transition-colors"
              title="Next Frame (Arrow Right)"
            >
              <SkipForward className="w-3.5 h-3.5" />
            </button>

            {/* Timecode and Frame Number readout */}
            <div className="ml-2 pl-3 border-l border-[#374151] flex items-center gap-2">
              <span className="font-bold text-white text-xs">
                {formatTimecode(currentTimeSeconds)}
              </span>
              <span className="text-[#6B7280]">/</span>
              <span className="text-[#9CA3AF] text-xs">
                {formatTimecode(duration)}
              </span>
              <span className="px-2 py-0.5 rounded bg-black/60 border border-[#374151] text-[11px] text-cyan-400 font-semibold">
                F{currentFrame}/{totalFrames}
              </span>
            </div>
          </div>

          {/* Right Transport: Speed, Volume, Fullscreen */}
          <div className="flex items-center gap-3">
            {/* Playback speed selector */}
            <div className="flex items-center gap-1 bg-black/40 px-2 py-0.5 rounded border border-[#374151] text-[11px]">
              <span className="text-[#6B7280]">Speed:</span>
              <select
                value={playbackRate}
                onChange={(e) => setPlaybackRate(parseFloat(e.target.value))}
                className="bg-transparent text-cyan-300 font-bold focus:outline-none cursor-pointer"
              >
                <option value={0.5} className="bg-[#111827]">0.5x</option>
                <option value={1.0} className="bg-[#111827]">1.0x</option>
                <option value={1.5} className="bg-[#111827]">1.5x</option>
                <option value={2.0} className="bg-[#111827]">2.0x</option>
              </select>
            </div>

            {/* Volume & Mute */}
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={toggleMute}
                className="p-1.5 rounded text-[#9CA3AF] hover:text-white hover:bg-[#1F2937] transition-colors"
                title={isMuted ? "Unmute" : "Mute"}
              >
                {isMuted || volume === 0 ? (
                  <VolumeX className="w-3.5 h-3.5 text-red-400" />
                ) : (
                  <Volume2 className="w-3.5 h-3.5 text-[#9CA3AF]" />
                )}
              </button>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={isMuted ? 0 : volume}
                onChange={handleVolumeChange}
                className="w-16 h-1 bg-[#374151] rounded-lg appearance-none cursor-pointer accent-cyan-400"
                title={`Volume: ${Math.round((isMuted ? 0 : volume) * 100)}%`}
              />
            </div>

            {/* Fullscreen button */}
            <button
              type="button"
              onClick={toggleFullscreen}
              className="p-1.5 rounded text-[#9CA3AF] hover:text-white hover:bg-[#1F2937] transition-colors"
              title="Toggle Fullscreen"
            >
              {isFullscreen ? (
                <Minimize2 className="w-3.5 h-3.5" />
              ) : (
                <Maximize2 className="w-3.5 h-3.5" />
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
