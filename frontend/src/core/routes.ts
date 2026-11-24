/**
 * Application Route Paths
 */

export const ROUTES = {
  ROOT: "/",
  OPERATIONS: "/operations",
  VIDEOS: "/videos",
  WORKSPACE: "/workspace/:jobId",
  WORKSPACE_DEFAULT: "/workspace",
  JOB_DETAILS: "/jobs/:jobId",
  SETTINGS: "/settings",
} as const;

export const buildWorkspaceUrl = (jobId: string) => `/workspace/${jobId}`;
export const buildJobDetailsUrl = (jobId: string) => `/jobs/${jobId}`;
