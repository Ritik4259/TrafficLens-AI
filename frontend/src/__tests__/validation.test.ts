import { describe, it, expect } from "vitest";
import {
  validateVideoFile,
  formatBytes,
  formatDuration,
  MAX_UPLOAD_SIZE_BYTES,
  ALLOWED_VIDEO_EXTENSIONS,
} from "../core/validation";

describe("Video File Validation", () => {
  it("accepts supported extensions (.mp4, .avi, .mov, .mkv) with valid non-zero size", () => {
    ALLOWED_VIDEO_EXTENSIONS.forEach((ext) => {
      const file = new File(["dummy video data content"], `test_video${ext}`, {
        type: "video/mp4",
      });
      const result = validateVideoFile(file);
      expect(result.valid).toBe(true);
      expect(result.error).toBeUndefined();
    });
  });

  it("rejects files with unsupported extensions", () => {
    const invalidFiles = [
      "document.txt",
      "records.csv",
      "photo.jpg",
      "diagram.png",
      "executable.exe",
      "archive.tar.gz",
    ];
    invalidFiles.forEach((filename) => {
      const file = new File(["some data"], filename, { type: "text/plain" });
      const result = validateVideoFile(file);
      expect(result.valid).toBe(false);
      expect(result.error).toContain("Unsupported video extension");
    });
  });

  it("rejects files with 0 bytes (empty file)", () => {
    const emptyFile = new File([], "empty_video.mp4", { type: "video/mp4" });
    const result = validateVideoFile(emptyFile);
    expect(result.valid).toBe(false);
    expect(result.error).toContain("Uploaded file is empty (0 bytes)");
  });

  it("rejects files exceeding the backend 500 MB limit", () => {
    const oversizedFile = new File(["x"], "huge_traffic.mp4", { type: "video/mp4" });
    Object.defineProperty(oversizedFile, "size", {
      value: MAX_UPLOAD_SIZE_BYTES + 1024,
    });

    const result = validateVideoFile(oversizedFile);
    expect(result.valid).toBe(false);
    expect(result.error).toContain("exceeds maximum upload limit of 500 MB");
  });

  it("passes files within the 500 MB limit", () => {
    const limitFile = new File(["x"], "limit_traffic.mp4", { type: "video/mp4" });
    Object.defineProperty(limitFile, "size", {
      value: MAX_UPLOAD_SIZE_BYTES,
    });

    const result = validateVideoFile(limitFile);
    expect(result.valid).toBe(true);
    expect(result.error).toBeUndefined();
  });
});

describe("Format Utilities", () => {
  it("formats bytes accurately with proper units", () => {
    expect(formatBytes(0)).toBe("0 Bytes");
    expect(formatBytes(1024)).toBe("1 KB");
    expect(formatBytes(1048576)).toBe("1 MB");
    expect(formatBytes(524288000)).toBe("500 MB");
    expect(formatBytes(1073741824)).toBe("1 GB");
  });

  it("formats duration in seconds", () => {
    expect(formatDuration(0)).toBe("0.00s");
    expect(formatDuration(5.2)).toBe("5.20s");
    expect(formatDuration(65.5)).toBe("1m 5.5s");
    expect(formatDuration(null)).toBe("--:--");
    expect(formatDuration(undefined)).toBe("--:--");
  });
});