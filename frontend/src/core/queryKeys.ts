/**
 * Centralized TanStack Query Keys for TrafficLens AI
 */

export const QUERY_KEYS = {
  // Processing Jobs
  job: (jobId: string | undefined) => ["job", jobId] as const,
  jobResults: (jobId: string | undefined) => ["jobResults", jobId] as const,
  jobArtifacts: (jobId: string | undefined) => ["jobArtifacts", jobId] as const,
  jobTelemetry: (jobId: string | undefined) => ["jobTelemetry", jobId] as const,

  // Multimodal Scene Interpretation (VLM)
  interpretation: {
    latest: (jobId: string | undefined) => ["sceneInterpretation", jobId] as const,
    byId: (id: string | undefined) => ["sceneInterpretationById", id] as const,
    history: (jobId: string | undefined, ascending: boolean = true) =>
      ["sceneInterpretationHistory", jobId, { ascending }] as const,
  },

  // Evidence-Grounded Traffic Incident Reports
  incidentReport: {
    latest: (jobId: string | undefined) => ["incidentReport", jobId] as const,
    byId: (id: string | undefined) => ["incidentReportById", id] as const,
    history: (jobId: string | undefined, ascending: boolean = true) =>
      ["incidentReportHistory", jobId, { ascending }] as const,
  },
} as const;
