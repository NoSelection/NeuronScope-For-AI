import { useState } from 'react';
import { useActivationStore } from '../stores/activationStore';
import { useModelStore } from '../stores/modelStore';
import { Panel } from '../components/common/Panel';
import { InfoLabel, InfoTip } from '../components/common/InfoTip';
import type { ComponentType } from '../api/types';
import { COMPONENT_LABELS } from '../api/types';

const CAPTURABLE: ComponentType[] = [
  'residual_pre',
  'attn_output',
  'mlp_output',
  'residual_post',
];

export function ActivationExplorer() {
  const { loaded, info } = useModelStore();
  const { captures, loading, error, captureActivations, clear } = useActivationStore();
  const [inputText, setInputText] = useState('');
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedComponent, setSelectedComponent] = useState<ComponentType>('residual_post');

  if (!loaded) {
    return (
      <Panel title="Activation Explorer">
        <p className="text-sm text-zinc-500">Load a model first.</p>
      </Panel>
    );
  }

  const numLayers = info?.num_layers ?? 34;

  const handleCapture = () => {
    if (!inputText.trim()) return;
    captureActivations(inputText, [
      { layer: selectedLayer, component: selectedComponent },
    ]);
  };

  const handleCaptureAllLayers = () => {
    if (!inputText.trim()) return;
    const targets = Array.from({ length: numLayers }, (_, i) => ({
      layer: i,
      component: selectedComponent,
    }));
    captureActivations(inputText, targets);
  };

  return (
    <div className="space-y-5">
      <Panel title="Activation Explorer">
        <div className="space-y-4">
          <div className="mb-2 rounded-lg border border-blue-500/20 bg-blue-500/5 p-3">
            <p className="text-xs leading-relaxed text-zinc-400">
              Activations are the numbers computed inside the model as it processes text.
              Capture them here to inspect what the model is &quot;thinking&quot; at any layer and component.
            </p>
          </div>

          <div>
            <InfoLabel topic="activations">Input Text</InfoLabel>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="The Eiffel Tower is in"
              rows={2}
              className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:border-blue-500 focus:outline-none"
            />
          </div>

          <div className="flex gap-3">
            <div className="flex-1">
              <InfoLabel topic="layer">Layer</InfoLabel>
              <select
                value={selectedLayer}
                onChange={(e) => setSelectedLayer(parseInt(e.target.value))}
                className="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-200"
              >
                {Array.from({ length: numLayers }, (_, i) => (
                  <option key={i} value={i}>Layer {i}</option>
                ))}
              </select>
            </div>
            <div className="flex-1">
              <InfoLabel topic="component">Component</InfoLabel>
              <select
                value={selectedComponent}
                onChange={(e) => setSelectedComponent(e.target.value as ComponentType)}
                className="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-200"
              >
                {CAPTURABLE.map((c) => (
                  <option key={c} value={c}>{COMPONENT_LABELS[c]}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="flex gap-3">
            <button
              onClick={handleCapture}
              disabled={loading || !inputText.trim()}
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-40"
            >
              {loading ? 'Capturing...' : 'Capture Layer'}
            </button>
            <button
              onClick={handleCaptureAllLayers}
              disabled={loading || !inputText.trim()}
              className="rounded-lg border border-zinc-600 px-4 py-2 text-sm font-medium text-zinc-300 hover:bg-zinc-800 disabled:opacity-40"
            >
              Capture All Layers
            </button>
            {captures.length > 0 && (
              <button
                onClick={clear}
                className="rounded-lg px-4 py-2 text-sm text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
              >
                Clear
              </button>
            )}
          </div>

          {error && (
            <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">
              {error}
            </div>
          )}
        </div>
      </Panel>

      {/* Captured Activations */}
      {captures.length > 0 && (
        <Panel title={`Captured Activations (${captures.length})`}>
          <div className="mb-3 flex items-center gap-1.5 text-xs text-zinc-500">
            <InfoTip topic="tensor_stats">What do these statistics mean?</InfoTip>
          </div>
          <div className="space-y-3">
            {captures.map((cap) => (
              <div
                key={cap.target_key}
                className="rounded-lg border border-zinc-700/50 bg-zinc-800/30 p-4"
              >
                <div className="mb-2 flex items-center justify-between">
                  <span className="font-mono text-sm text-blue-400">{cap.target_key}</span>
                  <span className="text-xs text-zinc-500">
                    Shape: [{cap.shape.join(', ')}]
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-5">
                  <StatItem label="Mean" value={cap.stats.mean.toFixed(4)} />
                  <StatItem label="Std" value={cap.stats.std.toFixed(4)} />
                  <StatItem label="Min" value={cap.stats.min.toFixed(4)} />
                  <StatItem label="Max" value={cap.stats.max.toFixed(4)} />
                  <StatItem label="Norm" value={cap.stats.norm.toFixed(2)} />
                </div>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}

function StatItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-zinc-500">{label}: </span>
      <span className="font-mono text-zinc-300">{value}</span>
    </div>
  );
}
