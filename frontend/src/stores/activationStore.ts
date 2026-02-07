import { create } from 'zustand';
import type { CaptureResult } from '../api/types';
import * as api from '../api/client';

interface ActivationState {
  captures: CaptureResult[];
  loading: boolean;
  error: string | null;

  captureActivations: (
    inputText: string,
    targets: Array<Record<string, unknown>>,
  ) => Promise<void>;
  clear: () => void;
}

export const useActivationStore = create<ActivationState>((set) => ({
  captures: [],
  loading: false,
  error: null,

  captureActivations: async (inputText, targets) => {
    set({ loading: true, error: null });
    try {
      const captures = await api.captureActivations(inputText, targets);
      set({ captures, loading: false });
    } catch (e) {
      set({ loading: false, error: (e as Error).message });
    }
  },

  clear: () => set({ captures: [], error: null }),
}));
