import React from "react";

export interface PageHeaderProps {
  title: string;
  subtitle?: string;
  category?: string;
  actions?: React.ReactNode;
}

export const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  subtitle,
  category,
  actions,
}) => {
  return (
    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-6 border-b border-[#1F2937]">
      <div>
        {category && (
          <div className="text-xs font-mono tracking-wider uppercase text-cyan-400 font-semibold mb-1">
            {category}
          </div>
        )}
        <h1 className="text-2xl font-bold tracking-tight text-[#F9FAFB]">{title}</h1>
        {subtitle && <p className="text-sm text-[#9CA3AF] mt-1">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2.5 shrink-0">{actions}</div>}
    </div>
  );
};
