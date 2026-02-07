import { create } from 'zustand';
import type {
  ExperimentConfig,
  ExperimentResult,
  ExperimentSummary,
  SweepSummary,
  SweepDetail,
  InterventionSpec,
  Insight,
  ComponentType,
  InterventionType,
} from '../api/types';
import * as api from '../api/client';

function defaultIntervention(): InterventionSpec {
  return {
    target_layer: 0,
    target_component: 'mlp_output' as ComponentType,
    target_head: null,
    target_position: null,
    target_neuron: null,
    intervention_type: 'zero' as InterventionType,
    intervention_params: {},
  };
}

function defaultConfig(): ExperimentConfig {
  return {
    name: '',
    base_input: '',
    source_input: null,
    interventions: [defaultIntervention()],
    seed: 42,
    model_path: 'LLM',
    max_new_tokens: 1,
    temperature: 0.0,
    capture_targets: [],
  };
}

interface ExperimentState {
  config: ExperimentConfig;
  currentResult: ExperimentResult | null;
  sweepResults: ExperimentResult[];
  insights: Insight[];
  history: ExperimentSummary[];
  running: boolean;
  error: string | null;

  // Sweep history
  sweepHistory: SweepSummary[];
  selectedSweep: SweepDetail | null;
  sweepLoading: boolean;

  updateConfig: (partial: Partial<ExperimentConfig>) => void;
  updateIntervention: (index: number, partial: Partial<InterventionSpec>) => void;
  addIntervention: () => void;
  removeIntervention: (index: number) => void;
  resetConfig: () => void;
  runExperiment: () => Promise<void>;
  runSweep: (layers?: number[]) => Promise<void>;
  loadHistory: () => Promise<void>;
  selectResult: (id: string) => Promise<void>;
  loadSweepHistory: () => Promise<void>;
  loadSweep: (id: string) => Promise<void>;
  deleteSweep: (id: string) => Promise<void>;
}

export const useExperimentStore = create<ExperimentState>((set, get) => ({
  config: defaultConfig(),
  currentResult: null,
  sweepResults: [],
  insights: [],
  history: [],
  running: false,
  error: null,
  sweepHistory: [],
  selectedSweep: null,
  sweepLoading: false,

  updateConfig: (partial) => {
    set((s) => ({ config: { ...s.config, ...partial } }));
  },

  updateIntervention: (index, partial) => {
    set((s) => {
      const interventions = [...s.config.interventions];
      interventions[index] = { ...interventions[index], ...partial };
      return { config: { ...s.config, interventions } };
    });
  },

  addIntervention: () => {
    set((s) => ({
      config: {
        ...s.config,
        interventions: [...s.config.interventions, defaultIntervention()],
      },
    }));
  },

  removeIntervention: (index) => {
    set((s) => ({
      config: {
        ...s.config,
        interventions: s.config.interventions.filter((_, i) => i !== index),
      },
    }));
  },

  resetConfig: () => set({ config: defaultConfig() }),

  runExperiment: async () => {
    const { config } = get();
    set({ running: true, error: null, insights: [] });
    try {
      const { result, insights } = await api.runExperiment(config);
      set({ currentResult: result, insights, running: false });
    } catch (e) {
      set({ running: false, error: (e as Error).message });
    }
  },

  runSweep: async (layers) => {
    const { config } = get();
    set({ running: true, error: null, sweepResults: [], insights: [] });
    try {
      const { results, insights } = await api.runSweep(config, layers);
      set({ sweepResults: results, insights, running: false });
      // Refresh sweep history in background
      get().loadSweepHistory();
    } catch (e) {
      set({ running: false, error: (e as Error).message });
    }
  },

  loadHistory: async () => {
    try {
      const history = await api.listExperiments();
      set({ history });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  selectResult: async (id) => {
    try {
      const result = await api.getExperiment(id);
      set({ currentResult: result, insights: [] });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  loadSweepHistory: async () => {
    try {
      const sweepHistory = await api.listSweeps();
      set({ sweepHistory });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  loadSweep: async (id) => {
    set({ sweepLoading: true, error: null });
    try {
      const sweep = await api.getSweep(id);
      set({ selectedSweep: sweep, sweepLoading: false });
    } catch (e) {
      set({ sweepLoading: false, error: (e as Error).message });
    }
  },

  deleteSweep: async (id) => {
    try {
      await api.deleteSweep(id);
      set((s) => ({
        sweepHistory: s.sweepHistory.filter((sw) => sw.id !== id),
        selectedSweep: s.selectedSweep?.id === id ? null : s.selectedSweep,
      }));
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },
}));
