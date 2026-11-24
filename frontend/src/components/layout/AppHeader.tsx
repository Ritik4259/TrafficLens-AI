import React from "react";
import { Link, useLocation } from "react-router-dom";
import { Cpu, Radio, Menu } from "lucide-react";
import { useHealth } from "../../hooks/useHealth";
import { ROUTES } from "../../core/routes";

interface AppHeaderProps {
  onToggleSidebar?: () => void;
}

export const AppHeader: React.FC<AppHeaderProps> = ({ onToggleSidebar }) => {
  const { mode, health } = useHealth();
  const location = useLocation();

  // Simple route name mapper for header breadcrumb
  const getBreadcrumb = () => {
    const path = location.pathname;
    if (path.startsWith("/operations")) return "Traffic Operations";
    if (path.startsWith("/videos")) return "Video Library & Ingestion";
    if (path.startsWith("/workspace")) return "Video Analysis Workspace";
    if (path.startsWith("/jobs")) return "Processing Job Audit";
    if (path.startsWith("/settings")) return "Settings & Calibration";
    return "Operations Center";
  };

  return (
    <header className="h-16 bg-[#111827] border-b border-[#1F2937] px-4 md:px-6 flex items-center justify-between sticky top-0 z-40">
      <div className="flex items-center gap-3">
        {onToggleSidebar && (
          <button
            onClick={onToggleSidebar}
            className="p-1.5 rounded-md text-[#9CA3AF] hover:text-white hover:bg-[#1A2234] md:hidden focus:outline-none"
            aria-label="Toggle navigation menu"
          >
            <Menu className="w-5 h-5" />
          </button>
        )}

        <Link to={ROUTES.OPERATIONS} className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-cyan-950/80 border border-cyan-500/50 flex items-center justify-center text-cyan-400 shadow-sm group-hover:border-cyan-400 transition-colors">
            <Radio className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-bold tracking-tight text-white font-mono text-base">
                TrafficLens<span className="text-cyan-400 font-extrabold">AI</span>
              </span>
              <span className="text-[10px] font-mono px-1 py-0.2 rounded bg-cyan-950/60 border border-cyan-800/80 text-cyan-400 font-semibold uppercase">
                CV Core
              </span>
            </div>
            <p className="text-[10px] text-[#6B7280] font-mono tracking-tight hidden sm:block">
              Intelligent Traffic Understanding Workstation
            </p>
          </div>
        </Link>

        <div className="h-5 w-px bg-[#1F2937] mx-2 hidden lg:block" />
        <div className="text-xs font-mono text-[#9CA3AF] hidden lg:flex items-center gap-1.5">
          <span className="text-[#6B7280]">View:</span>
          <span className="text-[#E5E7EB] font-medium">{getBreadcrumb()}</span>
        </div>
      </div>

      {/* Right side: Backend health indicator & Engine status */}
      <div className="flex items-center gap-3">
        {/* Backend health status pill */}
        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-md border font-mono text-xs transition-colors ${
            mode === "online"
              ? "bg-emerald-950/40 border-emerald-800/60 text-emerald-400"
              : mode === "checking"
                ? "bg-cyan-950/40 border-cyan-800/60 text-cyan-400"
                : "bg-red-950/40 border-red-800/60 text-red-400"
          }`}
          title={
            mode === "online"
              ? `Backend v${health?.version || "0.1.0"} online (${health?.database_dialect || "SQLite"})`
              : "Backend unreachable on /api/v1/health"
          }
        >
          <span className="relative flex h-2 w-2">
            {mode === "online" && (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            )}
            <span
              className={`relative inline-flex rounded-full h-2 w-2 ${
                mode === "online"
                  ? "bg-emerald-400"
                  : mode === "checking"
                    ? "bg-cyan-400 animate-pulse"
                    : "bg-red-400"
              }`}
            />
          </span>
          <span className="font-semibold tracking-wider uppercase text-[11px]">
            {mode === "online" ? "BACKEND ONLINE" : mode === "checking" ? "CHECKING" : "OFFLINE"}
          </span>
          {health?.database_dialect && (
            <span className="text-[10px] px-1 py-0.2 rounded bg-black/30 border border-emerald-800/40 text-emerald-300 hidden md:inline">
              {health.database_dialect}
              {health.database_connected !== undefined && (health.database_connected ? " • DB OK" : " • DB ERR")}
            </span>
          )}
        </div>

        {/* Pipeline target specification pill (explicitly labeled as SPEC, not live engine status) */}
        <div
          className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-[#1A2234] border border-[#374151] font-mono text-[11px] text-[#9CA3AF]"
          title="Target CV Architecture Specification (Frozen baseline, model loaded per job)"
        >
          <span className="text-[9px] font-semibold text-[#6B7280] tracking-wider uppercase">TARGET SPEC:</span>
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span>YOLO11n + ByteTrack</span>
        </div>
      </div>
    </header>
  );
};
