import React from "react";

export interface LoadingSkeletonProps {
  className?: string;
  lines?: number;
}

export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({ className = "h-4 w-full", lines = 1 }) => {
  return (
    <div className="space-y-2">
      {Array.from({ length: lines }).map((_, idx) => (
        <div
          key={idx}
          className={`animate-pulse bg-[#1F2937]/80 rounded ${className}`}
        />
      ))}
    </div>
  );
};
