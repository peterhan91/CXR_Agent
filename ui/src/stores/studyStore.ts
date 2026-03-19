import { create } from "zustand";

interface StudyViewState {
  // Image controls
  brightness: number;
  contrast: number;
  zoom: number;
  panX: number;
  panY: number;

  // Report view
  selectedModel: string;
  showGroundTruth: boolean;

  // Trajectory
  expandedSteps: Set<number>;

  // Actions
  setBrightness: (v: number) => void;
  setContrast: (v: number) => void;
  setZoom: (v: number) => void;
  setPan: (x: number, y: number) => void;
  resetView: () => void;
  setSelectedModel: (m: string) => void;
  toggleGroundTruth: () => void;
  toggleStep: (i: number) => void;
}

export const useStudyStore = create<StudyViewState>((set) => ({
  brightness: 100,
  contrast: 100,
  zoom: 1,
  panX: 0,
  panY: 0,
  selectedModel: "agent_initial",
  showGroundTruth: false,
  expandedSteps: new Set(),

  setBrightness: (v) => set({ brightness: v }),
  setContrast: (v) => set({ contrast: v }),
  setZoom: (v) => set({ zoom: v }),
  setPan: (x, y) => set({ panX: x, panY: y }),
  resetView: () =>
    set({ brightness: 100, contrast: 100, zoom: 1, panX: 0, panY: 0 }),
  setSelectedModel: (m) => set({ selectedModel: m }),
  toggleGroundTruth: () =>
    set((s) => ({ showGroundTruth: !s.showGroundTruth })),
  toggleStep: (i) =>
    set((s) => {
      const next = new Set(s.expandedSteps);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return { expandedSteps: next };
    }),
}));
