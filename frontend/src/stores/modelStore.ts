import { create } from 'zustand';
import type { ModelInfo } from '../api/types';
import * as api from '../api/client';

interface ModelState {
  info: ModelInfo | null;
  loaded: boolean;
  loading: boolean;
  error: string | null;
  moduleTree: Record<string, string> | null;

  fetchInfo: () => Promise<void>;
  loadModel: (path?: string, device?: string) => Promise<void>;
  unloadModel: () => Promise<void>;
  fetchModuleTree: () => Promise<void>;
}

export const useModelStore = create<ModelState>((set) => ({
  info: null,
  loaded: false,
  loading: false,
  error: null,
  moduleTree: null,

  fetchInfo: async () => {
    try {
      const result = await api.getModelInfo();
      if ('loaded' in result && result.loaded === false) {
        set({ info: null, loaded: false });
      } else {
        set({ info: result as ModelInfo, loaded: true });
      }
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  loadModel: async (path = 'LLM', device = 'cuda') => {
    set({ loading: true, error: null });
    try {
      const info = await api.loadModel(path, device);
      set({ info, loaded: true, loading: false });
    } catch (e) {
      set({ loading: false, error: (e as Error).message });
    }
  },

  unloadModel: async () => {
    try {
      await api.unloadModel();
      set({ info: null, loaded: false, moduleTree: null });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  fetchModuleTree: async () => {
    try {
      const tree = await api.getModuleTree();
      set({ moduleTree: tree });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },
}));
