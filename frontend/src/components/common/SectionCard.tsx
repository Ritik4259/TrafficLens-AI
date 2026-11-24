import React from "react";
import { Card } from "./Card";

export interface SectionCardProps {
  title: string;
  description?: string;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

export const SectionCard: React.FC<SectionCardProps> = ({
  title,
  description,
  badge,
  actions,
  children,
  className = "",
}) => {
  return (
    <Card className={`overflow-hidden ${className}`}>
      <div className="px-5 py-4 border-b border-[#1F2937] flex items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h2 className="text-sm font-semibold tracking-wide text-[#F9FAFB] uppercase font-mono">
              {title}
            </h2>
            {badge}
          </div>
          {description && <p className="text-xs text-[#9CA3AF] mt-0.5">{description}</p>}
        </div>
        {actions && <div className="shrink-0">{actions}</div>}
      </div>
      <div className="p-5">{children}</div>
    </Card>
  );
};
