import { create } from 'zustand';
import { WALKTHROUGH_STEPS } from '../education/walkthrough';

interface WalkthroughState {
  /** Whether the walkthrough is currently active */
  active: boolean;
  /** Current step index */
  stepIndex: number;
  /** Whether user has dismissed the walkthrough permanently */
  dismissed: boolean;

  start: () => void;
  next: () => void;
  prev: () => void;
  dismiss: () => void;
  reset: () => void;
}

const STORAGE_KEY = 'neuronscope_walkthrough_dismissed';

export const useWalkthroughStore = create<WalkthroughState>((set, get) => ({
  active: false,
  stepIndex: 0,
  dismissed: localStorage.getItem(STORAGE_KEY) === 'true',

  start: () => set({ active: true, stepIndex: 0 }),

  next: () => {
    const { stepIndex } = get();
    if (stepIndex < WALKTHROUGH_STEPS.length - 1) {
      set({ stepIndex: stepIndex + 1 });
    } else {
      // Finished all steps
      set({ active: false, stepIndex: 0, dismissed: true });
      localStorage.setItem(STORAGE_KEY, 'true');
    }
  },

  prev: () => {
    const { stepIndex } = get();
    if (stepIndex > 0) {
      set({ stepIndex: stepIndex - 1 });
    }
  },

  dismiss: () => {
    set({ active: false, stepIndex: 0, dismissed: true });
    localStorage.setItem(STORAGE_KEY, 'true');
  },

  reset: () => {
    set({ active: false, stepIndex: 0, dismissed: false });
    localStorage.removeItem(STORAGE_KEY);
  },
}));
