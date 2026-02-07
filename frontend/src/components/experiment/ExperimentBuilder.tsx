import { useExperimentStore } from '../../stores/experimentStore';
import { useModelStore } from '../../stores/modelStore';
import { InterventionSelector } from './InterventionSelector';
import { Panel } from '../common/Panel';

export function ExperimentBuilder() {
  const { config, updateConfig, running, error, runExperiment, runSweep, resetConfig } =
    useExperimentStore();
  const { loaded, info } = useModelStore();

  const canRun = loaded && config.base_input.trim() && config.interventions.length > 0;

  return (
    <Panel
      title="Experiment Builder"
      actions={
        <button
          onClick={resetConfig}
          className="rounded px-3 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        >
          Reset
        </button>
      }
    >
      <div className="space-y-4">
        {/* Experiment Name */}
        <div>
          <label className="mb-1 block text-xs text-zinc-400">Experiment Name</label>
          <input
            type="text"
            value={config.name}
            onChange={(e) => updateConfig({ name: e.target.value })}
            placeholder="e.g., zero_ablate_L17_mlp"
            className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:border-blue-500 focus:outline-none"
          />
        </div>

        {/* Base Input */}
        <div>
          <label className="mb-1 block text-xs text-zinc-400">Base Input</label>
          <textarea
            value={config.base_input}
            onChange={(e) => updateConfig({ base_input: e.target.value })}
            placeholder="The Eiffel Tower is in"
            rows={2}
            className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:border-blue-500 focus:outline-none"
          />
        </div>

        {/* Source Input (for activation patching) */}
        {config.interventions.some((i) => i.intervention_type === 'patch') && (
          <div>
            <label className="mb-1 block text-xs text-zinc-400">
              Source Input (for Activation Patching)
            </label>
            <textarea
              value={config.source_input ?? ''}
              onChange={(e) =>
                updateConfig({ source_input: e.target.value || null })
              }
              placeholder="The Colosseum is in"
              rows={2}
              className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:border-blue-500 focus:outline-none"
            />
          </div>
        )}

        {/* Interventions */}
        <div>
          <label className="mb-2 block text-xs text-zinc-400">Interventions</label>
          <InterventionSelector numLayers={info?.num_layers ?? 34} />
        </div>

        {/* Seed */}
        <div className="flex gap-4">
          <div className="flex-1">
            <label className="mb-1 block text-xs text-zinc-400">Seed</label>
            <input
              type="number"
              value={config.seed}
              onChange={(e) => updateConfig({ seed: parseInt(e.target.value) || 0 })}
              className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
            />
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">
            {error}
          </div>
        )}

        {/* Run Buttons */}
        <div className="flex gap-3">
          <button
            onClick={runExperiment}
            disabled={!canRun || running}
            className="flex-1 rounded-lg bg-blue-600 py-2.5 text-sm font-medium text-white transition-colors hover:bg-blue-500 disabled:opacity-40"
          >
            {running ? 'Running...' : 'Run Experiment'}
          </button>
          <button
            onClick={() => runSweep()}
            disabled={!canRun || running}
            className="rounded-lg border border-zinc-600 px-4 py-2.5 text-sm font-medium text-zinc-300 transition-colors hover:bg-zinc-800 disabled:opacity-40"
          >
            {running ? '...' : 'Sweep All Layers'}
          </button>
        </div>
      </div>
    </Panel>
  );
}
