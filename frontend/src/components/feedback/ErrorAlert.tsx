import React from "react";
import { AlertCircle, RefreshCw } from "lucide-react";
import { Button } from "../common/Button";

export interface ErrorAlertProps {
  title?: string;
  message: string;
  errorCode?: string;
  onRetry?: () => void;
  className?: string;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({
  title = "Processing Failure",
  message,
  errorCode,
  onRetry,
  className = "",
}) => {
  return (
    <div
      className={`p-4 rounded-lg bg-red-950/30 border border-red-800/60 text-red-200 flex items-start gap-3 ${className}`}
    >
      <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <h4 className="text-sm font-semibold text-red-300">{title}</h4>
          {errorCode && (
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-red-900/60 border border-red-700/60 text-red-200">
              {errorCode}
            </span>
          )}
        </div>
        <p className="text-xs text-red-300/90 mt-1 leading-relaxed font-mono">{message}</p>
      </div>
      {onRetry && (
        <Button
          variant="outline"
          size="sm"
          onClick={onRetry}
          icon={<RefreshCw className="w-3.5 h-3.5" />}
          className="shrink-0 text-red-300 border-red-800/80 hover:bg-red-900/40"
        >
          Retry
        </Button>
      )}
    </div>
  );
};
