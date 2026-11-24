/**
 * Video file and format validation utilities matching backend constraints.
 */

export const ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"] as const;
export const MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024; // 500 MB (524,288,000 bytes)

export interface ValidationResult {
  valid: boolean;
  error?: string;
}

/**
 * Validates file extension, non-zero size, and 500 MB maximum constraint.
 */
export function validateVideoFile(file: File): ValidationResult {
  if (!file) {
    return { valid: false, error: "No file selected." };
  }

  if (file.size === 0) {
    return { valid: false, error: "Uploaded file is empty (0 bytes)." };
  }

  if (file.size > MAX_UPLOAD_SIZE_BYTES) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    return {
      valid: false,
      error: `File size (${sizeMb} MB) exceeds maximum upload limit of 500 MB.`,
    };
  }

  const name = file.name || "";
  const extIndex = name.lastIndexOf(".");
  if (extIndex === -1) {
    return {
      valid: false,
      error: `File has no extension. Allowed extensions: ${ALLOWED_VIDEO_EXTENSIONS.join(", ")}`,
    };
  }

  const ext = name.slice(extIndex).toLowerCase();
  if (!ALLOWED_VIDEO_EXTENSIONS.includes(ext as any)) {
    return {
      valid: false,
      error: `Unsupported video extension '${ext}'. Allowed extensions: ${ALLOWED_VIDEO_EXTENSIONS.join(", ")}`,
    };
  }

  return { valid: true };
}

/**
 * Human-readable byte formatting (B, KB, MB, GB).
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

/**
 * Formats duration in seconds to "00:00.00" or "Xm Ys".
 */
export function formatDuration(seconds?: number | null): string {
  if (seconds == null || isNaN(seconds) || seconds < 0) {
    return "--:--";
  }
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (mins === 0) {
    return `${secs.toFixed(2)}s`;
  }
  return `${mins}m ${secs.toFixed(1)}s`;
}
