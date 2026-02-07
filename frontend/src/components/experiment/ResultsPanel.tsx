import type { ExperimentResult, TokenPrediction } from '../../api/types';
import { MetricCard } from '../common/MetricCard';
import { Panel } from '../common/Panel';
import { InfoTip } from '../common/InfoTip';

interface Props {
  result: ExperimentResult;
}

export function ResultsPanel({ result }: Props) {
  return (
    <Panel title={`Results: ${result.config.name || result.id}`}>
      <div className="space-y-5">
        {/* Key Metrics */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MetricCard
            label="KL Divergence"
            value={result.kl_divergence}
            subtitle="Higher = larger causal effect"
            highlight={result.kl_divergence > 1.0}
            topic="kl_divergence"
          />
          <MetricCard
            label="Top Token Changed"
            value={result.top_token_changed ? 'YES' : 'NO'}
            highlight={result.top_token_changed}
            topic="top_token_changed"
          />
          <MetricCard
            label="Clean Output"
            value={`"${result.clean_output_token}"`}
            subtitle={`p=${result.clean_output_prob.toFixed(4)}`}
            topic="clean_output"
          />
          <MetricCard
            label="Intervention Output"
            value={`"${result.intervention_output_token}"`}
            subtitle={`p=${result.intervention_output_prob.toFixed(4)}`}
            topic="intervention_output"
          />
        </div>

        {/* Top-K Comparison */}
        <div className="grid grid-cols-2 gap-4">
          <TopKTable title="Clean Run" predictions={result.clean_top_k} />
          <TopKTable title="Intervention Run" predictions={result.intervention_top_k} />
        </div>

        {/* Rank Changes */}
        {Object.keys(result.rank_changes).length > 0 && (
          <div>
            <h3 className="mb-2 flex items-center gap-1.5 text-xs uppercase tracking-wider text-zinc-400">
              Rank Changes (Top-20)
              <InfoTip topic="rank_changes" iconOnly />
            </h3>
            <div className="max-h-48 overflow-y-auto rounded-lg border border-zinc-700/50 bg-zinc-800/30">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-700/50 text-left text-xs text-zinc-500">
                    <th className="px-3 py-2">Token</th>
                    <th className="px-3 py-2">Clean Rank</th>
                    <th className="px-3 py-2">Intervention Rank</th>
                    <th className="px-3 py-2">Delta</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.rank_changes)
                    .sort((a, b) => Math.abs(b[1].rank_delta) - Math.abs(a[1].rank_delta))
                    .slice(0, 20)
                    .map(([token, rc]) => (
                      <tr key={token} className="border-b border-zinc-800/50">
                        <td className="px-3 py-1.5 font-mono text-zinc-200">
                          {token}
                        </td>
                        <td className="px-3 py-1.5 text-zinc-400">{rc.clean_rank}</td>
                        <td className="px-3 py-1.5 text-zinc-400">
                          {rc.intervention_rank}
                        </td>
                        <td
                          className={`px-3 py-1.5 font-mono ${
                            rc.rank_delta > 0
                              ? 'text-red-400'
                              : rc.rank_delta < 0
                                ? 'text-green-400'
                                : 'text-zinc-500'
                          }`}
                        >
                          {rc.rank_delta > 0 ? '+' : ''}
                          {rc.rank_delta}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Metadata */}
        <div className="flex gap-4 text-xs text-zinc-500">
          <span>Duration: {result.duration_seconds.toFixed(2)}s</span>
          <span>Device: {result.device}</span>
          <span className="inline-flex items-center gap-1">
            Hash: {result.config_hash}
            <InfoTip topic="config_hash" iconOnly />
          </span>
        </div>
      </div>
    </Panel>
  );
}

function TopKTable({
  title,
  predictions,
}: {
  title: string;
  predictions: TokenPrediction[];
}) {
  return (
    <div>
      <h3 className="mb-2 text-xs uppercase tracking-wider text-zinc-400">{title}</h3>
      <div className="rounded-lg border border-zinc-700/50 bg-zinc-800/30">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-700/50 text-left text-xs text-zinc-500">
              <th className="px-3 py-2">#</th>
              <th className="px-3 py-2">Token</th>
              <th className="px-3 py-2">
                <span className="inline-flex items-center gap-1">
                  Logit <InfoTip topic="logit" iconOnly />
                </span>
              </th>
              <th className="px-3 py-2">
                <span className="inline-flex items-center gap-1">
                  Prob <InfoTip topic="probability" iconOnly />
                </span>
              </th>
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
    </div>
  );
}
