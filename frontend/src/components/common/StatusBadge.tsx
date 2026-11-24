import React from "react";
import { JobStatus } from "../../types/job";
import { OBSIDIAN_TOKENS } from "../../core/tokens";

export interface StatusBadgeProps {
  status: JobStatus | string;
  size?: "sm" | "md";
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = "md" }) => {
  const normStatus = (status || "QUEUED").toUpperCase() as JobStatus;
  const cfg = OBSIDIAN_TOKENS.status[normStatus] || OBSIDIAN_TOKENS.status.QUEUED;

  const sizeClass = size === "sm" ? "text-[10px] px-1.5 py-0.5" : "text-xs px-2 py-0.5";

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-mono font-medium rounded border ${cfg.bg} ${cfg.text} ${cfg.border} ${sizeClass}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
      {cfg.label}
    </span>
  );
};
