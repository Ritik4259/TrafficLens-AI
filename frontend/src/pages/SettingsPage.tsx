import React from "react";
import { Info } from "lucide-react";
import { PageHeader } from "../components/common/PageHeader";
import { SectionCard } from "../components/common/SectionCard";
import { Badge } from "../components/common/Badge";
import { useHealth } from "../hooks/useHealth";
import { APP_CONFIG } from "../core/config";

export const SettingsPage: React.FC = () => {
  const { mode, health } = useHealth();

  return (
    <div className="space-y-6">
      <PageHeader
        category="Workstation Settings"
        title="Settings & Optics Configuration"
        subtitle="System endpoints, camera calibration status, and frozen computer vision baselines"
      />

      {/* Backend API Connection Status */}
      <SectionCard
        title="Backend API Configuration"
        description="FastAPI service endpoints and database persistence status"
      >
        <div className="p-4 rounded-lg bg-[#1A2234] border border-[#374151] space-y-3 font-mono text-xs">
          <div className="flex items-center justify-between pb-2 border-b border-[#1F2937]">
            <span className="text-[#9CA3AF]">API Base URL:</span>
            <span className="text-cyan-400 font-bold">{APP_CONFIG.apiBaseUrl}</span>
          </div>
          <div className="flex items-center justify-between pb-2 border-b border-[#1F2937]">
            <span className="text-[#9CA3AF]">Service Health:</span>
            <span className={mode === "online" ? "text-emerald-400 font-bold" : "text-red-400 font-bold"}>
              {mode === "online" ? "ONLINE (200 OK)" : "OFFLINE"}
            </span>
          </div>
          <div className="flex items-center justify-between pb-2 border-b border-[#1F2937]">
            <span className="text-[#9CA3AF]">Database Connectivity:</span>
            <span className={health?.database_connected ? "text-emerald-400 font-bold" : "text-amber-400 font-bold"}>
              {health?.database_connected ? "VERIFIED (SELECT 1 OK)" : "UNVERIFIED / PENDING"}
            </span>
          </div>
          <div className="flex items-center justify-between pb-2 border-b border-[#1F2937]">
            <span className="text-[#9CA3AF]">Database Dialect:</span>
            <span className="text-[#F9FAFB]">{health?.database_dialect || "SQLite"}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[#9CA3AF]">Backend Version:</span>
            <span className="text-[#F9FAFB]">v{health?.version || "0.1.0"} ({health?.environment || "development"})</span>
          </div>
        </div>
      </SectionCard>

      {/* Camera Homography & Metric Speed Calibration */}
      <SectionCard
        title="Camera Homography & Metric Speed Calibration"
        description="Geometric projection parameters for transforming pixel displacement to real-world metric speed (km/h)"
        badge={<Badge variant="amber" size="sm">UNCALIBRATED</Badge>}
      >
        <div className="p-4 rounded-lg bg-amber-950/20 border border-amber-800/40 space-y-2">
          <div className="flex items-center gap-2 text-xs font-mono font-bold text-amber-300">
            <Info className="w-4 h-4 text-amber-400 shrink-0" />
            <span>Homography Calibration Unavailable</span>
          </div>
          <p className="text-xs font-mono text-[#9CA3AF] leading-relaxed">
            The UA-DETRAC dataset does not supply metric 3D road homography matrices for fixed camera positions. Therefore, vehicle displacement is measured and reported strictly in <span className="text-[#F9FAFB] font-bold">pixels per second (px/s)</span>. Metric speed in km/h is set to <code className="text-amber-300">null</code>.
          </p>
          <div className="text-[11px] font-mono text-[#6B7280] pt-1">
            Status: Grounded in backend SpeedMeasurementSchema (`is_calibrated: false`, `speed_kmh: null`).
          </div>
        </div>
      </SectionCard>

      {/* Frozen Computer Vision Baselines */}
      <SectionCard
        title="Frozen Computer Vision Baselines"
        description="Detector and multi-object tracker parameters verified on UA-DETRAC benchmark"
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-xs">
          <div className="p-4 rounded-lg bg-[#1A2234] border border-[#374151] space-y-2">
            <div className="text-cyan-400 font-bold uppercase tracking-wider text-[11px]">
              YOLO11n Vehicle Detector
            </div>
            <div className="flex justify-between text-[#9CA3AF]">
              <span>Model Architecture:</span>
              <span className="text-[#F9FAFB]">Ultralytics YOLO11n (COCO)</span>
            </div>
            <div className="flex justify-between text-[#9CA3AF]">
              <span>Confidence Threshold:</span>
              <span className="text-[#F9FAFB]">0.25 (Validation-Tuned)</span>
            </div>
            <div className="flex justify-between text-[#9CA3AF]">
              <span>NMS IoU Threshold:</span>
              <span className="text-[#F9FAFB]">0.45</span>
            </div>
            <div className="flex justify-between text-[#9CA3AF]">
              <span>Class Mapping:</span>
              <span className="text-cyan-400">Unified Vehicle</span>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-[#1A2234] border border-[#374151] space-y-2">
            <div className="text-cyan-400 font-bold uppercase tracking-wider text-[11px]">
              ByteTrack Multi-Object Tracker
            </div>
            <div className="flex justify-between text-[#9CA3AF]">
              <span>High Confidence Track:</span>
              <span className="text-[#F9FAFB]">0.40</span>
            </div>
            <div className="flex justify-between text-[#9CA3AF]">
              <span>Low Confidence Associate:</span>
              <span className="text-[#F9FAFB]">0.10</span>
            </div>
            <div className="flex justify-between text-[#9CA3AF]">
              <span>Track Buffer Lifespan:</span>
              <span className="text-[#F9FAFB]">45 frames (1.8s)</span>
            </div>
            <div className="flex justify-between text-[#9CA3AF]">
              <span>Match Threshold:</span>
              <span className="text-[#F9FAFB]">0.70</span>
            </div>
          </div>
        </div>
      </SectionCard>
    </div>
  );
};
