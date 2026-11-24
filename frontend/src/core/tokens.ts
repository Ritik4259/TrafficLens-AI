/**
 * Centralized Obsidian Optical Design Tokens for TrafficLens AI
 *
 * Implements the approved design tokens:
 * - Structural neutrals (obsidian base, surface, borders)
 * - Validated unified vehicle token
 * - Traffic-state semantic colors (low, moderate, high)
 * - Spatial geometry (tripwires, ROIs)
 * - Typography (Inter for UI, JetBrains Mono for telemetry)
 */

export const OBSIDIAN_TOKENS = {
  canvas: {
    base: "#0B0F17",       // Deep obsidian root background
    subtle: "#111827",     // Card, panel, and sidebar background (gray-900)
    surface: "#1A2234",    // Elevated controls, dropdown menus, modals (gray-850)
    hover: "#243048",      // Hover state for interactive rows and cards
  },
  border: {
    subtle: "#1F2937",     // Subtle card borders, dividers (gray-800)
    default: "#374151",    // Interactive control borders, inputs (gray-700)
    focused: "#0EA5E9",    // Active focus ring (cyan-500)
  },
  text: {
    primary: "#F9FAFB",    // High-contrast headings and values (gray-50)
    secondary: "#9CA3AF",  // Labels, metadata, axis titles (gray-400)
    muted: "#6B7280",      // Timestamps, disabled states (gray-500)
  },
  vehicle: {
    // Active validated vehicle visualization token for current unified CV baseline
    active: {
      stroke: "#06B6D4",                  // Primary Optical Cyan
      fill: "rgba(6, 182, 212, 0.15)",   // Translucent Cyan footprint fill
      text: "#22D3EE",
      label: "Vehicle",
    },
    // Prospective future subtype tokens (unvalidated by current baseline; documented as future capability)
    prospectiveSubtypes: {
      car: { stroke: "#0EA5E9", fill: "rgba(14, 165, 233, 0.15)", text: "#38BDF8" },
      truck: { stroke: "#F59E0B", fill: "rgba(245, 158, 11, 0.15)", text: "#FBBF24" },
      bus: { stroke: "#8B5CF6", fill: "rgba(139, 92, 246, 0.15)", text: "#A78BFA" },
      van: { stroke: "#14B8A6", fill: "rgba(20, 184, 166, 0.15)", text: "#2DD4BF" },
      motorcycle: { stroke: "#F43F5E", fill: "rgba(244, 63, 94, 0.15)", text: "#FB7185" },
    },
  },
  congestion: {
    // Grounded directly in backend CongestionLevelEnum heuristics
    low: {
      color: "#10B981",
      bg: "rgba(16, 185, 129, 0.12)",
      border: "#059669",
      label: "Low Congestion (Free Flow)",
      badgeBg: "bg-emerald-950/40",
      badgeText: "text-emerald-400",
      badgeBorder: "border-emerald-800/60",
    },
    moderate: {
      color: "#F59E0B",
      bg: "rgba(245, 158, 11, 0.12)",
      border: "#D97706",
      label: "Moderate Congestion (Dense)",
      badgeBg: "bg-amber-950/40",
      badgeText: "text-amber-400",
      badgeBorder: "border-amber-800/60",
    },
    high: {
      color: "#EF4444",
      bg: "rgba(239, 68, 68, 0.15)",
      border: "#DC2626",
      label: "High Congestion (Bottleneck)",
      badgeBg: "bg-red-950/40",
      badgeText: "text-red-400",
      badgeBorder: "border-red-800/60",
    },
  },
  geometry: {
    tripwireDownstream: "#10B981", // Emerald
    tripwireUpstream: "#06B6D4",   // Cyan
    roiBorder: "#6366F1",          // Indigo
    roiFill: "rgba(99, 102, 241, 0.10)",
    stoppedVehicleAlert: "#DC2626",
  },
  status: {
    QUEUED: {
      label: "QUEUED",
      bg: "bg-blue-950/40",
      text: "text-blue-400",
      border: "border-blue-800/60",
      dot: "bg-blue-400",
    },
    PROCESSING: {
      label: "PROCESSING",
      bg: "bg-cyan-950/40",
      text: "text-cyan-400",
      border: "border-cyan-800/60",
      dot: "bg-cyan-400 animate-pulse",
    },
    COMPLETED: {
      label: "COMPLETED",
      bg: "bg-emerald-950/40",
      text: "text-emerald-400",
      border: "border-emerald-800/60",
      dot: "bg-emerald-400",
    },
    FAILED: {
      label: "FAILED",
      bg: "bg-red-950/40",
      text: "text-red-400",
      border: "border-red-800/60",
      dot: "bg-red-400",
    },
  },
} as const;
