import { useEffect } from 'react';
import { useExperimentStore } from '../../stores/experimentStore';
import { Panel } from '../common/Panel';

export function ExperimentHistory() {
  const { history, loadHistory, selectResult } = useExperimentStore();

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  if (history.length === 0) {
    return (
      <Panel title="Experiment History">
        <p className="text-sm text-zinc-500">No experiments run yet.</p>
      </Panel>
    );
  }

  return (
    <Panel
      title="Experiment History"
      actions={
        <button
          onClick={loadHistory}
          className="rounded px-3 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        >
          Refresh
        </button>
      }
    >
      <div className="max-h-64 overflow-y-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-700/50 text-left text-xs text-zinc-500">
              <th className="px-3 py-2">Name</th>
              <th className="px-3 py-2">KL Div</th>
              <th className="px-3 py-2">Changed</th>
              <th className="px-3 py-2">Clean</th>
              <th className="px-3 py-2">Intervened</th>
            </tr>
          </thead>
          <tbody>
            {history.map((exp) => (
              <tr
                key={exp.id}
                onClick={() => selectResult(exp.id)}
                className="cursor-pointer border-b border-zinc-800/50 transition-colors hover:bg-zinc-800/50"
              >
                <td className="px-3 py-2 text-zinc-200">{exp.name || exp.id}</td>
                <td className="px-3 py-2 font-mono text-zinc-400">
                  {exp.kl_divergence.toFixed(4)}
                </td>
                <td className="px-3 py-2">
                  {exp.top_token_changed ? (
                    <span className="text-amber-400">YES</span>
                  ) : (
                    <span className="text-zinc-500">no</span>
                  )}
                </td>
                <td className="px-3 py-2 font-mono text-zinc-300">{exp.clean_token}</td>
                <td className="px-3 py-2 font-mono text-zinc-300">
                  {exp.intervention_token}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
