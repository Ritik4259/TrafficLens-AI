/**
 * Zustand store managing interactive workspace state
 */

import { create } from "zustand";

interface WorkspaceState {
  // Current playback state
  currentFrame: number;
  currentTimeSeconds: number;
  isPlaying: boolean;
  playbackRate: number;
  viewMode: "annotated" | "source" | "side-by-side";

  // Selected telemetry elements
  selectedTrackId: number | null;
  selectedRoiId: string | null;
  selectedLineId: string | null;

  // Overlay layer visibility toggles
  layers: {
    showBoundingBoxes: boolean;
    showTrackIds: boolean;
    showTripwires: boolean;
    showRois: boolean;
    showSpeedBadges: boolean;
  };

  // Actions
  setFrame: (frame: number) => void;
  setTime: (time: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setPlaybackRate: (rate: number) => void;
  setViewMode: (mode: "annotated" | "source" | "side-by-side") => void;
  selectTrack: (trackId: number | null) => void;
  selectRoi: (roiId: string | null) => void;
  selectLine: (lineId: string | null) => void;
  toggleLayer: (layerKey: keyof WorkspaceState["layers"]) => void;
}

export const useWorkspaceStore = create<WorkspaceState>((set) => ({
  currentFrame: 1,
  currentTimeSeconds: 0.0,
  isPlaying: false,
  playbackRate: 1.0,
  viewMode: "annotated",

  selectedTrackId: null,
  selectedRoiId: null,
  selectedLineId: null,

  layers: {
    showBoundingBoxes: true,
    showTrackIds: true,
    showTripwires: true,
    showRois: true,
    showSpeedBadges: false,
  },

  setFrame: (frame) => set({ currentFrame: frame }),
  setTime: (time) => set({ currentTimeSeconds: time }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),
  setPlaybackRate: (rate) => set({ playbackRate: rate }),
  setViewMode: (mode) => set({ viewMode: mode }),
  selectTrack: (trackId) => set({ selectedTrackId: trackId }),
  selectRoi: (roiId) => set({ selectedRoiId: roiId }),
  selectLine: (lineId) => set({ selectedLineId: lineId }),
  toggleLayer: (layerKey) =>
    set((state) => ({
      layers: {
        ...state.layers,
        [layerKey]: !state.layers[layerKey],
      },
    })),
}));
