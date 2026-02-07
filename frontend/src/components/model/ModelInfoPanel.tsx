import { useModelStore } from '../../stores/modelStore';
import { Panel } from '../common/Panel';
import { InfoTip } from '../common/InfoTip';
import type { EducationKey } from '../../education/content';

export function ModelInfoPanel() {
  const { info, loaded, loading, error, loadModel, unloadModel } = useModelStore();

  if (!loaded) {
    return (
      <Panel title="Model">
        <div className="space-y-3" data-tour="model-panel">
          <p className="text-sm text-zinc-400">No model loaded.</p>
          <button
            onClick={() => loadModel()}
            disabled={loading}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-500 disabled:opacity-50"
          >
            {loading ? 'Loading...' : 'Load Model'}
          </button>
          {error && <p className="text-sm text-red-400">{error}</p>}
        </div>
      </Panel>
    );
  }

  return (
    <Panel
      title="Model"
      actions={
        <button
          onClick={unloadModel}
          className="rounded px-3 py-1 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
        >
          Unload
        </button>
      }
    >
      <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm" data-tour="model-panel">
        <Stat label="Architecture" value={info!.architecture} topic="architecture" />
        <Stat label="Layers" value={info!.num_layers} topic="num_layers" />
        <Stat label="Hidden Size" value={info!.hidden_size} topic="hidden_size" />
        <Stat label="Intermediate" value={info!.intermediate_size} topic="intermediate_size" />
        <Stat label="Vocab Size" value={info!.vocab_size.toLocaleString()} topic="vocab_size" />
        <Stat label="Dtype" value={info!.dtype} topic="dtype" />
        <Stat label="Device" value={info!.device} />
        {info!.sliding_window && (
          <Stat label="Sliding Window" value={info!.sliding_window} />
        )}
        {info!.has_vision && <Stat label="Vision" value="Yes" />}
      </div>
    </Panel>
  );
}

function Stat({
  label,
  value,
  topic,
}: {
  label: string;
  value: string | number;
  topic?: EducationKey;
}) {
  return (
    <div className="flex items-center gap-1">
      <span className="text-zinc-500">{label}: </span>
      <span className="font-mono text-zinc-200">{value}</span>
      {topic && <InfoTip topic={topic} iconOnly />}
    </div>
  );
}
