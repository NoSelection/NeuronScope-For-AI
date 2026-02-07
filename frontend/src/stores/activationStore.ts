import { create } from 'zustand';
import type { CaptureResult, StreamMessage } from '../api/types';
import * as api from '../api/client';

interface ActivationState {
  captures: CaptureResult[];
  loading: boolean;
  streaming: boolean;
  streamProgress: number;
  error: string | null;

  captureActivations: (
    inputText: string,
    targets: Array<Record<string, unknown>>,
  ) => Promise<void>;
  streamActivations: (
    inputText: string,
    targets: Array<Record<string, unknown>>,
  ) => void;
  clear: () => void;
}

export const useActivationStore = create<ActivationState>((set, get) => ({
  captures: [],
  loading: false,
  streaming: false,
  streamProgress: 0,
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

  streamActivations: (inputText, targets) => {
    set({ streaming: true, streamProgress: 0, error: null, captures: [] });

    const ws = api.streamActivations(
      inputText,
      targets,
      (msg: StreamMessage) => {
        if (msg.type === 'activation' && msg.target_key && msg.shape && msg.stats) {
          const capture: CaptureResult = {
            target_key: msg.target_key,
            shape: msg.shape,
            stats: msg.stats,
            values: msg.values ?? null,
          };
          set((state) => ({
            captures: [...state.captures, capture],
            streamProgress: state.streamProgress + 1,
          }));
        } else if (msg.type === 'done') {
          set({ streaming: false });
          ws.close();
        } else if (msg.type === 'error') {
          set({ streaming: false, error: msg.detail ?? 'Stream error' });
          ws.close();
        }
      },
      () => {
        set({ streaming: false, error: 'WebSocket connection failed' });
      },
    );
  },

  clear: () => set({ captures: [], error: null, streamProgress: 0 }),
}));
