import { useState, useEffect } from 'react';
import { useExperimentStore } from '../../stores/experimentStore';
import { Panel } from '../common/Panel';
import { DiffHeatmap } from '../activation/DiffHeatmap';
import { DistributionPlot } from '../activation/DistributionPlot';
import * as api from '../../api/client';
import type { ExperimentResult, TokenPrediction, CaptureResult } from '../../api/types';

export function ComparisonView() {
  const { history, loadHistory } = useExperimentStore();
  const [leftId, setLeftId] = useState<string>('');
  const [rightId, setRightId] = useState<string>('');
  const [leftResult, setLeftResult] = useState<ExperimentResult | null>(null);
  const [rightResult, setRightResult] = useState<ExperimentResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [leftCaptures, setLeftCaptures] = useState<CaptureResult[]>([]);
  const [rightCaptures, setRightCaptures] = useState<CaptureResult[]>([]);
  const [capturingDiff, setCapturingDiff] = useState(false);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    if (!leftId) { setLeftResult(null); return; }
    setLoading(true);
    api.getExperiment(leftId)
      .then((r) => setLeftResult(r))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [leftId]);

  useEffect(() => {
    if (!rightId) { setRightResult(null); return; }
    setLoading(true);
    api.getExperiment(rightId)
      .then((r) => setRightResult(r))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [rightId]);

  return (
    <div className="space-y-5">
      <Panel title="Experiment Comparison">
        <div className="space-y-4">
          <p className="text-xs leading-relaxed text-zinc-400">
            Select two experiments to compare their results side-by-side.
            Differences in metrics and predictions are highlighted.
          </p>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="mb-1 block text-xs text-zinc-400">Experiment A</label>
              <select
                value={leftId}
                onChange={(e) => setLeftId(e.target.value)}
                className="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-200"
              >
                <option value="">Select experiment...</option>
                {history.map((h) => (
                  <option key={h.id} value={h.id}>
                    {h.name || h.id.slice(0, 8)} — {h.clean_token} → {h.intervention_token}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs text-zinc-400">Experiment B</label>
              <select
                value={rightId}
                onChange={(e) => setRightId(e.target.value)}
                className="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-200"
              >
                <option value="">Select experiment...</option>
                {history.map((h) => (
                  <option key={h.id} value={h.id}>
                    {h.name || h.id.slice(0, 8)} — {h.clean_token} → {h.intervention_token}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {error && (
            <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">
              {error}
            </div>
          )}

          {loading && <p className="text-sm text-zinc-500">Loading...</p>}

          {history.length === 0 && (
            <p className="text-sm text-zinc-500">
              No experiments in history. Run some experiments first.
            </p>
          )}
        </div>
      </Panel>

      {leftResult && rightResult && (
        <>
          {/* Metric Comparison */}
          <Panel title="Metric Comparison">
            <div className="space-y-3">
              <MetricRow
                label="KL Divergence"
                leftVal={leftResult.kl_divergence}
                rightVal={rightResult.kl_divergence}
                format={(v) => v.toFixed(4)}
                higherIsBigger
              />
              <MetricRow
                label="Top Token Changed"
                leftVal={leftResult.top_token_changed ? 1 : 0}
                rightVal={rightResult.top_token_changed ? 1 : 0}
                format={(v) => (v ? 'YES' : 'NO')}
              />
              <MetricRow
                label="Effect Size"
                leftVal={leftResult.effect_size ?? 0}
                rightVal={rightResult.effect_size ?? 0}
                format={(v) => v.toFixed(4)}
                higherIsBigger
              />
              <MetricRow
                label="Clean Output Prob"
                leftVal={leftResult.clean_output_prob}
                rightVal={rightResult.clean_output_prob}
                format={(v) => `${(v * 100).toFixed(1)}%`}
              />
              <MetricRow
                label="Intervention Output Prob"
                leftVal={leftResult.intervention_output_prob}
                rightVal={rightResult.intervention_output_prob}
                format={(v) => `${(v * 100).toFixed(1)}%`}
              />
              <MetricRow
                label="Duration"
                leftVal={leftResult.duration_seconds}
                rightVal={rightResult.duration_seconds}
                format={(v) => `${v.toFixed(2)}s`}
              />
            </div>
          </Panel>

          {/* Config Diff */}
          <Panel title="Config Differences">
            <div className="space-y-2 text-sm">
              <ConfigDiffRow
                label="Base Input"
                left={leftResult.config.base_input}
                right={rightResult.config.base_input}
              />
              <ConfigDiffRow
                label="Source Input"
                left={leftResult.config.source_input ?? '(none)'}
                right={rightResult.config.source_input ?? '(none)'}
              />
              <ConfigDiffRow
                label="Layer"
                left={String(leftResult.config.interventions[0]?.target_layer ?? '—')}
                right={String(rightResult.config.interventions[0]?.target_layer ?? '—')}
              />
              <ConfigDiffRow
                label="Component"
                left={leftResult.config.interventions[0]?.target_component ?? '—'}
                right={rightResult.config.interventions[0]?.target_component ?? '—'}
              />
              <ConfigDiffRow
                label="Intervention"
                left={leftResult.config.interventions[0]?.intervention_type ?? '—'}
                right={rightResult.config.interventions[0]?.intervention_type ?? '—'}
              />
            </div>
          </Panel>

          {/* Top-K Side by Side */}
          <Panel title="Top-K Predictions">
            <div className="grid grid-cols-2 gap-6">
              <div>
                <h3 className="mb-2 text-xs uppercase tracking-wider text-zinc-400">
                  Experiment A — Clean
                </h3>
                <TopKTable predictions={leftResult.clean_top_k} />
                <h3 className="mb-2 mt-4 text-xs uppercase tracking-wider text-zinc-400">
                  Experiment A — Intervention
                </h3>
                <TopKTable predictions={leftResult.intervention_top_k} />
              </div>
              <div>
                <h3 className="mb-2 text-xs uppercase tracking-wider text-zinc-400">
                  Experiment B — Clean
                </h3>
                <TopKTable predictions={rightResult.clean_top_k} />
                <h3 className="mb-2 mt-4 text-xs uppercase tracking-wider text-zinc-400">
                  Experiment B — Intervention
                </h3>
                <TopKTable predictions={rightResult.intervention_top_k} />
              </div>
            </div>
          </Panel>

          {/* Activation Diff */}
          <Panel title="Activation Difference">
            <p className="mb-3 text-xs text-zinc-400">
              Capture activations from both experiments at the intervention layer to visualize element-wise differences.
            </p>
            <button
              onClick={async () => {
                setCapturingDiff(true);
                try {
                  const layerA = leftResult.config.interventions[0]?.target_layer ?? 0;
                  const compA = leftResult.config.interventions[0]?.target_component ?? 'mlp_output';
                  const layerB = rightResult.config.interventions[0]?.target_layer ?? 0;
                  const compB = rightResult.config.interventions[0]?.target_component ?? 'mlp_output';
                  const [lCaps, rCaps] = await Promise.all([
                    api.captureActivations(leftResult.config.base_input, [{ layer: layerA, component: compA }]),
                    api.captureActivations(rightResult.config.base_input, [{ layer: layerB, component: compB }]),
                  ]);
                  setLeftCaptures(lCaps);
                  setRightCaptures(rCaps);
                } catch (e) {
                  setError((e as Error).message);
                } finally {
                  setCapturingDiff(false);
                }
              }}
              disabled={capturingDiff}
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-40"
            >
              {capturingDiff ? 'Capturing...' : 'Capture & Compare Activations'}
            </button>
            {leftCaptures.length > 0 && rightCaptures.length > 0 && leftCaptures[0].values && rightCaptures[0].values && (
              <>
                <DiffHeatmap
                  cleanValues={leftCaptures[0].values}
                  interventionValues={rightCaptures[0].values}
                  shape={leftCaptures[0].shape}
                />
                <DistributionPlot
                  values={leftCaptures[0].values}
                  comparisonValues={rightCaptures[0].values}
                  label="Activation Distribution Comparison"
                />
              </>
            )}
          </Panel>
        </>
      )}
    </div>
  );
}

function MetricRow({
  label,
  leftVal,
  rightVal,
  format,
  higherIsBigger,
}: {
  label: string;
  leftVal: number;
  rightVal: number;
  format: (v: number) => string;
  higherIsBigger?: boolean;
}) {
  const delta = rightVal - leftVal;
  const absDelta = Math.abs(delta);
  const showDelta = absDelta > 0.0001;

  let deltaColor = 'text-zinc-500';
  if (showDelta && higherIsBigger) {
    deltaColor = delta > 0 ? 'text-green-400' : 'text-red-400';
  } else if (showDelta) {
    deltaColor = delta > 0 ? 'text-red-400' : 'text-green-400';
  }

  return (
    <div className="flex items-center justify-between rounded-lg border border-zinc-700/50 bg-zinc-800/30 px-4 py-2.5">
      <span className="text-xs uppercase tracking-wider text-zinc-400">{label}</span>
      <div className="flex items-center gap-6">
        <span className="font-mono text-sm text-zinc-200">{format(leftVal)}</span>
        {showDelta && (
          <span className={`font-mono text-xs ${deltaColor}`}>
            {delta > 0 ? '+' : ''}{format(delta)}
          </span>
        )}
        <span className="font-mono text-sm text-zinc-200">{format(rightVal)}</span>
      </div>
    </div>
  );
}

function ConfigDiffRow({ label, left, right }: { label: string; left: string; right: string }) {
  const differs = left !== right;

  return (
    <div
      className={`flex items-start justify-between rounded-lg border px-4 py-2 ${
        differs
          ? 'border-amber-500/30 bg-amber-500/5'
          : 'border-zinc-700/50 bg-zinc-800/30'
      }`}
    >
      <span className="text-xs uppercase tracking-wider text-zinc-400">{label}</span>
      <div className="flex gap-6 text-right">
        <span className={`max-w-[200px] truncate font-mono text-xs ${differs ? 'text-amber-300' : 'text-zinc-300'}`}>
          {left}
        </span>
        <span className={`max-w-[200px] truncate font-mono text-xs ${differs ? 'text-amber-300' : 'text-zinc-300'}`}>
          {right}
        </span>
      </div>
    </div>
  );
}

function TopKTable({ predictions }: { predictions: TokenPrediction[] }) {
  return (
    <div className="rounded-lg border border-zinc-700/50 bg-zinc-800/30">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-700/50 text-left text-xs text-zinc-500">
            <th className="px-3 py-2">#</th>
            <th className="px-3 py-2">Token</th>
            <th className="px-3 py-2">Logit</th>
            <th className="px-3 py-2">Prob</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((p, i) => (
            <tr key={p.token_id} className="border-b border-zinc-800/50">
              <td className="px-3 py-1.5 text-zinc-500">{i + 1}</td>
              <td className="px-3 py-1.5 font-mono text-zinc-200">{p.token}</td>
              <td className="px-3 py-1.5 text-zinc-400">{p.logit.toFixed(2)}</td>
              <td className="px-3 py-1.5 text-zinc-400">{(p.prob * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
