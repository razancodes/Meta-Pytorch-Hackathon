import { create } from "zustand";
import type { MemexStore, AGUIState } from "./types";
import {
  DEMO_STEPS,
  DEMO_EPISODE_META,
  INITIAL_NODES,
  INITIAL_EDGES,
  ENTITY_PROFILES,
  TRANSACTIONS,
  WATCHLIST_RESULTS,
} from "./demo-data";

const EMPTY_AGUI: AGUIState = {
  ram_usage: { capacity: "0/2 observations", active_context: [] },
  disk_storage: [],
  async_jobs: [],
  kernel_directives: ["You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert."],
};

export const useMemexStore = create<MemexStore>((set, get) => ({
  // Flow
  currentFlow: "triage",
  setFlow: (flow) => set({ currentFlow: flow }),

  // Episode data
  episodeMeta: null,
  steps: [],
  currentStepIndex: -1,

  // AGUI state
  aguiState: EMPTY_AGUI,

  // Graph
  nodes: [],
  edges: [],
  selectedNodeId: null,

  // Entity data
  entityProfiles: ENTITY_PROFILES,
  transactions: TRANSACTIONS,
  watchlistResults: WATCHLIST_RESULTS,

  // Playback
  isPlaying: false,
  playbackSpeed: 2000,

  // Page fault
  pageFaultActive: false,

  // Actions
  loadEpisode: () => {
    set({
      episodeMeta: DEMO_EPISODE_META,
      steps: DEMO_STEPS,
      currentStepIndex: 0,
      aguiState: DEMO_STEPS[0].agui_state,
      nodes: INITIAL_NODES,
      edges: INITIAL_EDGES,
      currentFlow: "investigation",
    });
  },

  advanceStep: () => {
    const { steps, currentStepIndex } = get();
    const nextIndex = currentStepIndex + 1;
    if (nextIndex >= steps.length) {
      set({ isPlaying: false });
      return;
    }
    const step = steps[nextIndex];

    // Check for page fault (RAM at capacity and eviction happening)
    const prevCapacity = get().aguiState.ram_usage.capacity;
    const newCapacity = step.agui_state.ram_usage.capacity;
    const wasFull = prevCapacity.startsWith("2/2");
    const isFull = newCapacity.startsWith("2/2");
    const contextChanged = wasFull && isFull &&
      JSON.stringify(get().aguiState.ram_usage.active_context) !==
      JSON.stringify(step.agui_state.ram_usage.active_context);

    set({
      currentStepIndex: nextIndex,
      aguiState: step.agui_state,
      pageFaultActive: contextChanged,
    });

    // Clear page fault after animation
    if (contextChanged) {
      setTimeout(() => set({ pageFaultActive: false }), 600);
    }

    // Check if episode is done
    if (step.observation.done) {
      set({ isPlaying: false, currentFlow: "terminal" });
    }
  },

  goToStep: (index) => {
    const { steps } = get();
    if (index < 0 || index >= steps.length) return;
    const step = steps[index];
    set({
      currentStepIndex: index,
      aguiState: step.agui_state,
      currentFlow: step.observation.done ? "terminal" : "investigation",
    });
  },

  selectNode: (id) => set({ selectedNodeId: id, currentFlow: id ? "entity-detail" : "investigation" }),

  setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),

  togglePlayback: () => {
    const { isPlaying } = get();
    set({ isPlaying: !isPlaying });
  },

  triggerPageFault: () => {
    set({ pageFaultActive: true });
    setTimeout(() => set({ pageFaultActive: false }), 600);
  },

  injectContext: (data) => {
    const { aguiState } = get();
    set({
      aguiState: {
        ...aguiState,
        kernel_directives: [
          ...aguiState.kernel_directives,
          `[HUMAN OVERRIDE]: ${data}`,
        ],
      },
    });
  },
}));
