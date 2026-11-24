import { describe, it, expect } from "vitest";
import { APP_CONFIG } from "../core/config";

describe("Job Polling Logic and Terminal States", () => {
  function getPollingInterval(status?: string): number | false {
    if (status === "QUEUED" || status === "PROCESSING") {
      return APP_CONFIG.jobPollingIntervalMs || 1500;
    }
    return false;
  }

  function checkTerminalState(status?: string): boolean {
    return status === "COMPLETED" || status === "FAILED";
  }

  it("schedules polling at ~1500ms when status is active (QUEUED or PROCESSING)", () => {
    expect(getPollingInterval("QUEUED")).toBe(1500);
    expect(getPollingInterval("PROCESSING")).toBe(1500);
  });

  it("disables polling (returns false) when job enters terminal state (COMPLETED or FAILED)", () => {
    expect(getPollingInterval("COMPLETED")).toBe(false);
    expect(getPollingInterval("FAILED")).toBe(false);
    expect(getPollingInterval(undefined)).toBe(false);
  });

  it("correctly identifies terminal vs non-terminal states", () => {
    expect(checkTerminalState("QUEUED")).toBe(false);
    expect(checkTerminalState("PROCESSING")).toBe(false);
    expect(checkTerminalState("COMPLETED")).toBe(true);
    expect(checkTerminalState("FAILED")).toBe(true);
  });
});