import React from "react";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
  icon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = "secondary",
  size = "md",
  icon,
  className = "",
  disabled,
  ...props
}) => {
  const base =
    "inline-flex items-center justify-center font-medium transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 disabled:opacity-50 disabled:cursor-not-allowed select-none rounded-md";

  const sizes = {
    sm: "text-xs px-2.5 py-1.5 gap-1.5",
    md: "text-sm px-3.5 py-2 gap-2",
    lg: "text-base px-4 py-2.5 gap-2.5",
  };

  const variants = {
    primary: "bg-cyan-600 hover:bg-cyan-500 text-white font-semibold shadow-sm border border-cyan-500",
    secondary: "bg-[#1A2234] hover:bg-[#243048] text-[#F9FAFB] border border-[#374151]",
    outline: "bg-transparent hover:bg-[#1A2234] text-[#E5E7EB] border border-[#374151]",
    ghost: "bg-transparent hover:bg-[#1A2234] text-[#9CA3AF] hover:text-[#F9FAFB]",
    danger: "bg-red-950/60 hover:bg-red-900/60 text-red-300 border border-red-800/80",
  };

  return (
    <button
      className={`${base} ${sizes[size]} ${variants[variant]} ${className}`}
      disabled={disabled}
      {...props}
    >
      {icon && <span className="shrink-0">{icon}</span>}
      {children}
    </button>
  );
};
