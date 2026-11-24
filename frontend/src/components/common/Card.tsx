import React from "react";

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "surface" | "interactive";
}

export const Card: React.FC<CardProps> = ({
  children,
  variant = "default",
  className = "",
  ...props
}) => {
  const base = "rounded-lg border transition-colors";

  const variants = {
    default: "bg-[#111827] border-[#1F2937] text-[#F9FAFB]",
    surface: "bg-[#1A2234] border-[#374151] text-[#F9FAFB]",
    interactive: "bg-[#111827] border-[#1F2937] hover:border-[#374151] hover:bg-[#151D2E] text-[#F9FAFB] cursor-pointer",
  };

  return (
    <div className={`${base} ${variants[variant]} ${className}`} {...props}>
      {children}
    </div>
  );
};
