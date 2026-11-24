/**
 * Application environment configuration
 */

export const APP_CONFIG = {
  // Base URL for backend API; defaults to relative /api/v1 (proxied by Vite in dev)
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || "/api/v1",
  
  // Health polling interval in milliseconds
  healthPollingIntervalMs: 10000,
  
  // Job status polling interval in milliseconds
  jobPollingIntervalMs: 1500,

  // Scene interpretation status polling interval in milliseconds
  interpretationPollingIntervalMs: 2000,

  // Development mode flag
  isDev: import.meta.env.DEV,

  // Benchmark sample sequence fixture metadata
  sampleSequence: {
    name: "MVI_20011",
    filename: "MVI_20011_sample.mp4",
    frames: 75,
    fps: 25.0,
    durationSeconds: 3.0,
    resolution: "960x540",
    relativePath: "data/samples/MVI_20011_sample.mp4",
  },
} as const;
