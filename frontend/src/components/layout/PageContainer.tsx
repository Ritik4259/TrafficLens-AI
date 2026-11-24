import React from "react";

export interface PageContainerProps {
  children: React.ReactNode;
  className?: string;
  fullWidth?: boolean;
}

export const PageContainer: React.FC<PageContainerProps> = ({
  children,
  className = "",
  fullWidth = false,
}) => {
  return (
    <main
      className={`flex-1 p-4 md:p-6 lg:p-8 bg-[#0B0F17] overflow-y-auto ${
        fullWidth ? "w-full" : "max-w-7xl mx-auto w-full"
      } ${className}`}
    >
      {children}
    </main>
  );
};
