import React from "react";

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "outline" | "vehicle" | "cyan" | "emerald" | "amber" | "red";
  size?: "sm" | "md";
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = "default",
  size = "md",
  className = "",
  ...props
}) => {
  const base = "inline-flex items-center font-mono font-medium rounded tracking-wide border";
  
  const sizes = {
    sm: "text-[10px] px-1.5 py-0.5 gap-1",
    md: "text-xs px-2 py-0.5 gap-1.5",
  };

  const variants = {
    default: "bg-[#1F2937] text-[#E5E7EB] border-[#374151]",
    outline: "bg-transparent text-[#9CA3AF] border-[#374151]",
    vehicle: "bg-cyan-950/40 text-cyan-400 border-cyan-800/60",
    cyan: "bg-cyan-950/40 text-cyan-400 border-cyan-800/60",
    emerald: "bg-emerald-950/40 text-emerald-400 border-emerald-800/60",
    amber: "bg-amber-950/40 text-amber-400 border-amber-800/60",
    red: "bg-red-950/40 text-red-400 border-red-800/60",
  };

  return (
    <span className={`${base} ${sizes[size]} ${variants[variant]} ${className}`} {...props}>
      {children}
    </span>
  );
};
