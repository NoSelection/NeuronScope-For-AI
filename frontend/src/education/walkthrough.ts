/**
 * Step definitions for the guided walkthrough.
 * Each step targets a DOM element and explains what it does.
 */

export interface WalkthroughStep {
  /** Unique step identifier */
  id: string;
  /** CSS selector for the element to highlight */
  target: string;
  /** Step title */
  title: string;
  /** 2-3 sentence explanation */
  body: string;
  /** Which side to show the tooltip on */
  placement: 'top' | 'bottom' | 'left' | 'right';
}

export const WALKTHROUGH_STEPS: WalkthroughStep[] = [
  {
    id: 'welcome',
    target: '[data-tour="header"]',
    title: 'Welcome to NeuronScope',
    body: 'This tool lets you run causal experiments on a real AI language model. You\'ll disable specific parts of the model and see exactly how its predictions change. Think of it as brain surgery for AI — but reversible!',
    placement: 'bottom',
  },
  {
    id: 'model-panel',
    target: '[data-tour="model-panel"]',
    title: 'Step 1: Load the Model',
    body: 'First, you need to load the AI model into GPU memory. This is a real neural network with 34 layers and 2.5 billion parameters. Click "Load Model" to start — it takes a moment because the model is large.',
    placement: 'right',
  },
  {
    id: 'base-input',
    target: '[data-tour="base-input"]',
    title: 'Step 2: Write a Prompt',
    body: 'Type a sentence for the model to complete. Choose something with an obvious next word, like "The Eiffel Tower is in" (→ Paris) or "Water freezes at zero degrees" (→ Celsius). This makes it easy to see when your intervention changes the answer.',
    placement: 'right',
  },
  {
    id: 'intervention',
    target: '[data-tour="intervention"]',
    title: 'Step 3: Choose What to Break',
    body: 'This is where the science happens. Pick a layer (0-33), a component (like MLP Output), and an intervention type (like Zero Ablation). You\'re telling NeuronScope: "Disable this specific part and show me what happens." Start with Layer 0, MLP Output, Zero Ablation.',
    placement: 'right',
  },
  {
    id: 'run-button',
    target: '[data-tour="run-button"]',
    title: 'Step 4: Run the Experiment',
    body: 'Click "Run Experiment" to execute. The model will run twice: once normally, once with your intervention applied. Then you\'ll see exactly how the output changed. For a broader view, try "Sweep All Layers" to test every layer at once.',
    placement: 'top',
  },
  {
    id: 'results-area',
    target: '[data-tour="results-area"]',
    title: 'Step 5: Read the Results',
    body: 'Results appear here. Look at KL Divergence (bigger = bigger effect), whether the top token changed, and the probability shifts. The insights panel will explain what the results mean in plain language. Congratulations — you just did mechanistic interpretability!',
    placement: 'left',
  },
];
