import React from "react";
import { Button } from "../common/Button";

export interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  actionLabel,
  onAction,
  className = "",
}) => {
  return (
    <div
      className={`flex flex-col items-center justify-center p-8 text-center rounded-lg border border-dashed border-[#1F2937] bg-[#111827]/40 ${className}`}
    >
      {icon && (
        <div className="w-12 h-12 rounded-lg bg-[#1A2234] border border-[#374151] flex items-center justify-center text-[#9CA3AF] mb-4">
          {icon}
        </div>
      )}
      <h3 className="text-base font-semibold text-[#F9FAFB]">{title}</h3>
      <p className="text-sm text-[#9CA3AF] max-w-sm mt-1.5 mb-5">{description}</p>
      {actionLabel && onAction && (
        <Button variant="primary" size="sm" onClick={onAction}>
          {actionLabel}
        </Button>
      )}
    </div>
  );
};
