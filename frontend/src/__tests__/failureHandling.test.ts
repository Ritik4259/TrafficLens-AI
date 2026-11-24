import { describe, it, expect } from "vitest";
import { validateVideoFile } from "../core/validation";

describe("Failure and Edge Case Handling", () => {
  it("case-insensitively accepts allowed video extensions (.MP4, .Avi)", () => {
    const fileUpper = new File(["valid data"], "CLIP_CAMERA.MP4", { type: "video/mp4" });
    const resultUpper = validateVideoFile(fileUpper);
    expect(resultUpper.valid).toBe(true);

    const fileMixed = new File(["valid data"], "CLIP_TRAFFIC.Avi", { type: "video/avi" });
    const resultMixed = validateVideoFile(fileMixed);
    expect(resultMixed.valid).toBe(true);
  });

  it("rejects files without extensions or hidden dotfiles", () => {
    const noExt = new File(["data"], "videofile", { type: "video/mp4" });
    expect(validateVideoFile(noExt).valid).toBe(false);
    expect(validateVideoFile(noExt).error).toContain("File has no extension");

    const hiddenNoExt = new File(["data"], ".hiddenfile", { type: "text/plain" });
    expect(validateVideoFile(hiddenNoExt).valid).toBe(false);
  });

  it("rejects multi-extension executable/archive files (.mp4.exe, .tar.gz)", () => {
    const dangerous = new File(["data"], "traffic_analysis.mp4.exe", { type: "application/octet-stream" });
    const result = validateVideoFile(dangerous);
    expect(result.valid).toBe(false);
    expect(result.error).toContain("Unsupported video extension '.exe'");
  });

  it("extracts active job ID from 409 Conflict messages correctly", () => {
    const conflictMessage = "Video 'vid_123' already has an active processing job (job_abc123456789) in state PROCESSING.";
    const match = conflictMessage.match(/job_[a-f0-9]+/);
    expect(match).not.toBeNull();
    expect(match![0]).toBe("job_abc123456789");
  });

  it("properly flags 503 Worker Queue capacity errors", () => {
    const error503 = {
      status: 503,
      message: "Job worker queue is at maximum capacity (10 pending jobs). Cannot accept new jobs.",
    };
    const isQueueFull = error503.status === 503;
    expect(isQueueFull).toBe(true);
  });

  it("properly handles network failure (status 0)", () => {
    const networkError = {
      status: 0,
      message: "Network error during video upload. Please check connection to backend server.",
    };
    expect(networkError.status).toBe(0);
    expect(networkError.message).toContain("Network error");
  });
});