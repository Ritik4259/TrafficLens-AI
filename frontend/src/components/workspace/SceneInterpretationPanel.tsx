import React from "react";
import {
  Sparkles,
  RotateCw,
  AlertTriangle,
  HelpCircle,
  Eye,
  Layers,
  Sun,
  ShieldAlert,
  Compass,
  FileText,
  Clock,
} from "lucide-react";
import { Card } from "../common/Card";
import { Button } from "../common/Button";
import { EmptyState } from "../feedback/EmptyState";
import { ErrorAlert } from "../feedback/ErrorAlert";
import { LoadingSkeleton } from "../feedback/LoadingSkeleton";
import {
  useSceneInterpretation,
  useSceneInterpretationHistory,
  useCreateSceneInterpretation,
  useRetrySceneInterpretation,
} from "../../hooks/useSceneInterpretation";
import { EpistemicCategory } from "../../types/interpretation";

export interface SceneInterpretationPanelProps {
  jobId?: string;
  currentTimeSeconds: number;
  onSeek: (timeSeconds: number) => void;
}

const MAX_ATTEMPTS = 3;

export const SceneInterpretationPanel: React.FC<SceneInterpretationPanelProps> = ({
  jobId,
  currentTimeSeconds,
  onSeek,
}) => {
  const {
    data: interpretation,
    status,
    isLoading,
    isError,
    error,
    isNotRequested,
    isQueued,
    isProcessing,
    isCompleted,
    isFailed,
    refetch,
  } = useSceneInterpretation(jobId);

  const { data: historyData } = useSceneInterpretationHistory(jobId);
  const createMutation = useCreateSceneInterpretation(jobId);
  const retryMutation = useRetrySceneInterpretation(jobId);

  const totalAttempts = historyData?.total_attempts ?? (interpretation?.id ? 1 : 0);
  const isMaxAttemptsReached = totalAttempts >= MAX_ATTEMPTS;

  // 1. Initial Loading Skeleton
  if (isLoading) {
    return (
      <div className="space-y-4 font-mono text-xs" data-testid="vlm-loading-skeleton">
        <LoadingSkeleton className="h-20 w-full" />
        <LoadingSkeleton className="h-32 w-full" />
        <LoadingSkeleton className="h-44 w-full" />
      </div>
    );
  }

  // 2. Query Error State
  if (isError) {
    return (
      <div className="space-y-3 font-mono text-xs" data-testid="vlm-error-state">
        <ErrorAlert
          title="Scene Understanding Inaccessible"
          message={error?.message || "Failed to retrieve scene interpretation status from backend."}
        />
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          Retry Status Query
        </Button>
      </div>
    );
  }

  // Helper for Epistemic Claim Category styling
  const renderClaimBadge = (category: EpistemicCategory) => {
    switch (category) {
      case "cv_fact":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-cyan-950/80 text-cyan-300 border border-cyan-700/60 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            CV Fact
          </span>
        );
      case "visual_observation":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-emerald-950/80 text-emerald-300 border border-emerald-700/60 flex items-center gap-1">
            <Eye className="w-3 h-3 text-emerald-400" />
            Visual Observation
          </span>
        );
      case "inference":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-purple-950/80 text-purple-300 border border-purple-700/60 flex items-center gap-1">
            <Sparkles className="w-3 h-3 text-purple-400" />
            Inference
          </span>
        );
      case "unverified":
      default:
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-amber-950/80 text-amber-300 border border-amber-700/60 flex items-center gap-1">
            <AlertTriangle className="w-3 h-3 text-amber-400" />
            Unverified
          </span>
        );
    }
  };

  return (
    <div className="space-y-4 font-mono text-xs" data-testid="scene-interpretation-panel">
      {/* Top Banner & Attempt Lifecycle Tracker */}
      <div className="p-3 rounded-lg bg-[#0E1524] border border-[#1F2937] flex flex-wrap items-center justify-between gap-2.5">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-purple-400" />
          <span className="font-bold text-[#E5E7EB]">Multimodal Scene Understanding</span>
          {status && (
            <span
              className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase border ${
                isCompleted
                  ? "bg-emerald-950/80 text-emerald-400 border-emerald-800"
                  : isFailed
                  ? "bg-red-950/80 text-red-400 border-red-800"
                  : isQueued || isProcessing
                  ? "bg-cyan-950/80 text-cyan-300 border-cyan-800 animate-pulse"
                  : "bg-[#1F2937] text-[#9CA3AF] border-[#374151]"
              }`}
            >
              {status}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] text-[#6B7280]">
            Time: <strong className="text-cyan-400">{currentTimeSeconds.toFixed(2)}s</strong>
          </span>
          {totalAttempts > 0 && (
            <span
              className="text-[10px] px-2 py-0.5 rounded bg-black/60 border border-[#374151] text-[#9CA3AF]"
              data-testid="vlm-attempt-counter"
            >
              Attempt <strong className="text-white">{totalAttempts}</strong> of {MAX_ATTEMPTS}
            </span>
          )}
        </div>
      </div>

      {/* ---------------------------------------------------------------------- */}
      {/* STATE 1: NOT_REQUESTED */}
      {/* ---------------------------------------------------------------------- */}
      {isNotRequested && (
        <Card className="p-6 border-[#1F2937] bg-[#111827] text-center space-y-4" data-testid="vlm-not-requested">
          <EmptyState
            icon={<Sparkles className="w-8 h-8 text-purple-400" />}
            title="Scene Understanding Not Requested"
            description="Extract candidate keyframes from this sequence and run Vision-Language Model inference to identify environmental conditions, roadway geometry, event root causes, and epistemic claims."
            actionLabel={createMutation.isPending ? "Submitting..." : "Run Scene Understanding"}
            onAction={() => createMutation.mutate(undefined)}
          />

          {createMutation.isError && (
            <ErrorAlert
              title="Failed to Request Scene Understanding"
              message={(createMutation.error as any)?.message || "Endpoint error enqueuing interpretation."}
            />
          )}
        </Card>
      )}

      {/* ---------------------------------------------------------------------- */}
      {/* STATE 2: QUEUED */}
      {/* ---------------------------------------------------------------------- */}
      {isQueued && (
        <div
          className="p-4 rounded-lg bg-purple-950/20 border border-purple-800/60 flex items-start gap-3.5"
          data-testid="vlm-queued"
        >
          <div className="w-3.5 h-3.5 rounded-full bg-purple-400 animate-ping mt-1 shrink-0" />
          <div className="space-y-1">
            <span className="font-bold text-purple-300 uppercase tracking-wide">
              Task Enqueued in Interpretation Worker
            </span>
            <p className="text-[#9CA3AF] text-[11px] leading-relaxed">
              The scene interpretation task is waiting for the background worker thread. Keyframes will be evaluated and grounded automatically.
            </p>
          </div>
        </div>
      )}

      {/* ---------------------------------------------------------------------- */}
      {/* STATE 3: PROCESSING */}
      {/* ---------------------------------------------------------------------- */}
      {isProcessing && (
        <div
          className="p-4 rounded-lg bg-purple-950/30 border border-purple-500/50 space-y-2.5"
          data-testid="vlm-processing"
        >
          <div className="flex items-center gap-2.5 text-purple-300 font-bold">
            <RotateCw className="w-4 h-4 text-purple-400 animate-spin" />
            <span>VLM Multimodal Reasoning in Progress</span>
          </div>
          <p className="text-[#9CA3AF] text-[11px] leading-relaxed">
            Extracting candidate keyframes, querying the Gemini vision provider, and evaluating epistemic grounding claims.
          </p>
          <div className="text-[10px] text-cyan-400">
            Model: <code className="bg-black/40 px-1 py-0.5 rounded">{interpretation?.model_name || "Gemini VLM"}</code>
          </div>
        </div>
      )}

      {/* ---------------------------------------------------------------------- */}
      {/* STATE 4: FAILED */}
      {/* ---------------------------------------------------------------------- */}
      {isFailed && (
        <div className="space-y-3" data-testid="vlm-failed">
          <div className="p-4 rounded-lg bg-red-950/40 border border-red-800/70 space-y-2.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-red-300 font-bold">
                <AlertTriangle className="w-4 h-4 text-red-400" />
                <span>Scene Understanding Failed</span>
              </div>
              <span className="text-[10px] text-red-400">
                Attempt {totalAttempts}/{MAX_ATTEMPTS}
              </span>
            </div>

            <p className="text-red-200 text-[11px]">
              {interpretation?.error?.error_message || "An error occurred during multimodal scene interpretation."}
            </p>

            {interpretation?.error?.error_code && (
              <div className="text-[10px] text-[#9CA3AF]">
                Error Code: <code className="bg-black/50 text-red-300 px-1 py-0.5 rounded">{interpretation.error.error_code}</code>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between gap-3 p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937]">
            <span className="text-[11px] text-[#9CA3AF]">
              {isMaxAttemptsReached
                ? "Maximum 3 attempts reached for this job."
                : "You may retry scene interpretation up to 3 total attempts."}
            </span>

            <Button
              variant="primary"
              size="sm"
              disabled={isMaxAttemptsReached || retryMutation.isPending}
              onClick={() => retryMutation.mutate()}
              icon={<RotateCw className={`w-3.5 h-3.5 ${retryMutation.isPending ? "animate-spin" : ""}`} />}
              data-testid="vlm-retry-button"
            >
              {retryMutation.isPending ? "Retrying..." : isMaxAttemptsReached ? "Limit Reached" : "Retry Interpretation"}
            </Button>
          </div>

          {retryMutation.isError && (
            <ErrorAlert
              title="Retry Failed"
              message={(retryMutation.error as any)?.message || "Failed to submit retry attempt."}
            />
          )}
        </div>
      )}

      {/* ---------------------------------------------------------------------- */}
      {/* STATE 5: COMPLETED */}
      {/* ---------------------------------------------------------------------- */}
      {isCompleted && interpretation && (
        <div className="space-y-4" data-testid="vlm-completed">
          {/* Executive Scene Summary */}
          {interpretation.summary && (
            <div className="p-3.5 rounded-lg bg-[#0E1524] border border-[#1F2937] space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-[#E5E7EB] flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5 text-purple-400" />
                  Visual Scene Summary
                </span>
                <span className="text-[10px] text-[#6B7280]">Gemini Vision Synthesis</span>
              </div>
              <p className="text-white text-xs leading-relaxed" data-testid="vlm-summary-text">
                {interpretation.summary}
              </p>
              <div className="pt-2 border-t border-[#1F2937] flex flex-wrap items-center justify-between gap-2 text-[10px] text-[#9CA3AF]">
                <span>
                  Model: <strong className="text-white">{interpretation.model_name || "VLM"}</strong>
                </span>
                <span>
                  Duration: <strong className="text-white">{interpretation.duration_seconds?.toFixed(2)}s</strong>
                </span>
                {interpretation.total_tokens && (
                  <span>
                    Tokens: <strong className="text-white">{interpretation.total_tokens}</strong>
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Environmental Context Section */}
          {interpretation.environmental_context && (
            <div className="p-3.5 rounded-lg bg-[#0E1524] border border-[#1F2937] space-y-2.5" data-testid="vlm-environmental-context">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-[#E5E7EB] flex items-center gap-1.5">
                  <Sun className="w-3.5 h-3.5 text-amber-400" />
                  Environmental & Ambient Context
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/60 border border-[#374151] text-[#9CA3AF]">
                  Visual Direct Observation
                </span>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-[11px]">
                {interpretation.environmental_context.weather && (
                  <div className="p-2 rounded bg-[#0B0F17] border border-[#1F2937]">
                    <div className="text-[10px] text-[#6B7280]">Weather</div>
                    <div className="font-bold text-white mt-0.5 capitalize">
                      {interpretation.environmental_context.weather}
                    </div>
                  </div>
                )}

                {interpretation.environmental_context.lighting && (
                  <div className="p-2 rounded bg-[#0B0F17] border border-[#1F2937]">
                    <div className="text-[10px] text-[#6B7280]">Illumination</div>
                    <div className="font-bold text-white mt-0.5 capitalize">
                      {interpretation.environmental_context.lighting}
                    </div>
                  </div>
                )}

                {interpretation.environmental_context.pavement && (
                  <div className="p-2 rounded bg-[#0B0F17] border border-[#1F2937]">
                    <div className="text-[10px] text-[#6B7280]">Road Surface</div>
                    <div className="font-bold text-white mt-0.5 capitalize">
                      {interpretation.environmental_context.pavement}
                    </div>
                  </div>
                )}

                {interpretation.environmental_context.camera_angle && (
                  <div className="p-2 rounded bg-[#0B0F17] border border-[#1F2937] col-span-2 md:col-span-3">
                    <div className="text-[10px] text-[#6B7280]">Camera Angle / Perspective</div>
                    <div className="font-semibold text-cyan-300 mt-0.5">
                      {interpretation.environmental_context.camera_angle}
                    </div>
                  </div>
                )}
              </div>

              {interpretation.environmental_context.visibility_hazards?.length > 0 && (
                <div className="p-2 rounded bg-amber-950/20 border border-amber-800/40 text-[11px] text-amber-300 space-y-1">
                  <div className="font-bold flex items-center gap-1.5">
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                    Visibility Hazards Identified:
                  </div>
                  <ul className="list-disc list-inside space-y-0.5 text-[10px] text-amber-200">
                    {interpretation.environmental_context.visibility_hazards.map((h, i) => (
                      <li key={i}>{h}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Roadway & Scene Description */}
          {interpretation.scene_description && (
            <div className="p-3.5 rounded-lg bg-[#0E1524] border border-[#1F2937] space-y-2.5" data-testid="vlm-scene-description">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-[#E5E7EB] flex items-center gap-1.5">
                  <Compass className="w-3.5 h-3.5 text-cyan-400" />
                  Roadway Geometry & Layout
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-purple-950/80 border border-purple-700/60 text-purple-300">
                  Visual Assessment
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-[11px]">
                {interpretation.scene_description.roadway_topology && (
                  <div className="p-2 rounded bg-[#0B0F17] border border-[#1F2937]">
                    <div className="text-[10px] text-[#6B7280]">Roadway Topology</div>
                    <div className="font-bold text-white mt-0.5 capitalize">
                      {interpretation.scene_description.roadway_topology.replace(/_/g, " ")}
                    </div>
                  </div>
                )}

                {interpretation.scene_description.lane_count != null && (
                  <div className="p-2 rounded bg-[#0B0F17] border border-[#1F2937]">
                    <div className="text-[10px] text-[#6B7280]">Observed Lanes</div>
                    <div className="font-bold text-white mt-0.5">
                      {interpretation.scene_description.lane_count} lanes
                    </div>
                  </div>
                )}
              </div>

              {interpretation.scene_description.traffic_density_qualitative && (
                <div className="p-2 rounded bg-[#0B0F17] border border-[#1F2937] flex items-center justify-between text-[11px]">
                  <div>
                    <span className="text-[#9CA3AF]">Visual Density Assessment:</span>
                    <span className="font-bold text-purple-300 ml-2 capitalize">
                      {interpretation.scene_description.traffic_density_qualitative}
                    </span>
                  </div>
                  <span className="text-[10px] text-[#6B7280]">Qualitative • Separate from CV</span>
                </div>
              )}

              {interpretation.scene_description.features?.length > 0 && (
                <div className="space-y-1 text-[11px]">
                  <span className="text-[10px] text-[#6B7280]">Physical Roadway Features:</span>
                  <div className="flex flex-wrap gap-1.5">
                    {interpretation.scene_description.features.map((feat, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-0.5 rounded bg-[#161F30] border border-[#374151] text-xs text-[#E5E7EB]"
                      >
                        {feat}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Traffic Event Contextual Interpretations */}
          {interpretation.event_interpretations && interpretation.event_interpretations.length > 0 && (
            <div className="p-3.5 rounded-lg bg-[#0E1524] border border-[#1F2937] space-y-3" data-testid="vlm-event-interpretations">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-[#E5E7EB] flex items-center gap-1.5">
                  <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
                  Traffic Event Interpretations ({interpretation.event_interpretations.length})
                </span>
                <span className="text-[10px] text-amber-300 bg-amber-950/40 px-1.5 py-0.5 rounded border border-amber-800">
                  Hypotheses (Advisory)
                </span>
              </div>

              <div className="space-y-2">
                {interpretation.event_interpretations.map((evt, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937] space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-amber-300">
                          {evt.event_type || "Stationary Event"}
                        </span>
                        {evt.track_id != null && (
                          <span className="px-1.5 py-0.2 rounded text-[10px] bg-[#161F30] text-cyan-300 border border-cyan-800/40">
                            Track #{evt.track_id}
                          </span>
                        )}
                        <span className="text-[10px] text-[#6B7280]">ID: {evt.event_id}</span>
                      </div>
                      {evt.confidence && (
                        <span className="text-[10px] px-1.5 py-0.2 rounded bg-black/60 text-[#9CA3AF] border border-[#374151]">
                          Confidence: {evt.confidence}
                        </span>
                      )}
                    </div>

                    {evt.probable_cause && (
                      <div className="p-2 rounded bg-amber-950/20 border border-amber-800/40 text-[11px]">
                        <span className="text-amber-400 font-bold">Interpreted Probable Cause:</span>{" "}
                        <span className="text-amber-100">{evt.probable_cause}</span>
                      </div>
                    )}

                    <p className="text-white text-[11px] leading-relaxed">
                      {evt.interpretation}
                    </p>

                    {evt.supporting_citations?.length > 0 && (
                      <div className="text-[10px] text-[#6B7280] pt-1 border-t border-[#1F2937]">
                        Supporting Citations: {evt.supporting_citations.join(", ")}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Epistemic Claims and Provenance */}
          {interpretation.epistemic_claims && interpretation.epistemic_claims.length > 0 && (
            <div className="p-3.5 rounded-lg bg-[#0E1524] border border-[#1F2937] space-y-3" data-testid="vlm-epistemic-claims">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-[#E5E7EB] flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-cyan-400" />
                  Epistemic Claims & Provenance ({interpretation.epistemic_claims.length})
                </span>
                <span className="text-[10px] text-[#9CA3AF]">Categorized Attribution</span>
              </div>

              {/* Taxonomy Guide */}
              <div className="p-2 rounded bg-[#0B0F17] border border-[#1F2937] flex flex-wrap gap-2 text-[10px]">
                <div className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-cyan-400" />
                  <span className="text-[#9CA3AF]">CV Fact: Measured Telemetry</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <span className="text-[#9CA3AF]">Visual Observation: Keyframe</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-purple-400" />
                  <span className="text-[#9CA3AF]">Inference: Contextual</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-amber-400" />
                  <span className="text-[#9CA3AF]">Unverified: Hypothesis</span>
                </div>
              </div>

              <div className="space-y-2">
                {interpretation.epistemic_claims.map((claim) => (
                  <div
                    key={claim.claim_id}
                    className="p-3 rounded-lg bg-[#0B0F17] border border-[#1F2937] space-y-2"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        {renderClaimBadge(claim.category)}
                        <span className="text-[10px] text-[#6B7280]">{claim.claim_id}</span>
                      </div>

                      {claim.frame_timestamps?.length > 0 && (
                        <div className="flex items-center gap-1 text-[10px] text-cyan-300">
                          <Clock className="w-3 h-3 text-cyan-400" />
                          <div className="flex items-center gap-1">
                            {claim.frame_timestamps.map((t, i) => (
                              <button
                                key={i}
                                type="button"
                                onClick={() => onSeek(t)}
                                className="px-1.5 py-0.5 rounded bg-[#161F30] hover:bg-cyan-950/80 hover:text-cyan-200 border border-[#374151] transition-colors cursor-pointer text-[10px]"
                                title={`Seek video to t=${t.toFixed(2)}s`}
                              >
                                {t.toFixed(1)}s
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    <p className="text-white text-xs font-semibold leading-relaxed">
                      {claim.statement}
                    </p>

                    <div className="text-[11px] text-[#9CA3AF] bg-[#111827] p-2 rounded border border-[#1F2937]/60">
                      <span className="text-purple-300 font-semibold">Evidence Rationale:</span>{" "}
                      {claim.evidence_rationale}
                    </div>

                    {/* Telemetry Provenance Citation */}
                    {claim.telemetry_provenance && (
                      <div className="text-[10px] p-2 rounded bg-cyan-950/20 border border-cyan-800/40 text-cyan-200 flex flex-wrap items-center justify-between gap-2">
                        <span>
                          Cited Field: <code className="text-white bg-black/40 px-1 py-0.5 rounded">{claim.telemetry_provenance.field_path}</code>
                        </span>
                        {claim.telemetry_provenance.numeric_value != null && (
                          <span>
                            Measured Value: <strong className="text-white">{claim.telemetry_provenance.numeric_value} {claim.telemetry_provenance.unit || ""}</strong>
                          </span>
                        )}
                        <span className="text-[#9CA3AF]">
                          Source: {claim.telemetry_provenance.source_type}
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Limitations and Disclosures */}
          <div className="p-3.5 rounded-lg bg-[#111827] border border-[#1F2937] space-y-2" data-testid="vlm-limitations-section">
            <div className="flex items-center gap-1.5 text-xs font-bold text-amber-300">
              <HelpCircle className="w-3.5 h-3.5 text-amber-400" />
              <span>Epistemic Boundaries & Limitations</span>
            </div>

            <ul className="list-disc list-inside space-y-1 text-[11px] text-[#9CA3AF] leading-relaxed">
              <li>
                <strong>Camera Homography:</strong> Camera perspective is uncalibrated. Vehicle speeds cannot be determined in physical units (km/h or mph).
              </li>
              <li>
                <strong>Tracker Fragmentation:</strong> Vehicle track IDs reflect tracker trajectories rather than a guaranteed physical vehicle census.
              </li>
              <li>
                <strong>Advisory Interpretations:</strong> Root-cause explanations represent multimodal hypotheses and must not be used as automated control directives.
              </li>
              {interpretation.unverified_aspects?.map((aspect, idx) => (
                <li key={idx} className="text-amber-200">
                  <strong>Unverified Aspect:</strong> {aspect}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};
