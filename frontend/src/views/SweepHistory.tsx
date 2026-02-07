import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { Panel } from '../components/common/Panel';
import { InsightsPanel } from '../components/experiment/InsightsPanel';
import { useExperimentStore } from '../stores/experimentStore';
import * as api from '../api/client';
import type { ExperimentResult } from '../api/types';

export function SweepHistory() {
  const {
    sweepHistory,
    selectedSweep,
    sweepLoading,
    loadSweepHistory,
    loadSweep,
    deleteSweep,
    error,
  } = useExperimentStore();

  useEffect(() => {
    loadSweepHistory();
  }, [loadSweepHistory]);

  return (
    <div className="space-y-6">
      <Panel title="Sweep History">
        {sweepHistory.length === 0 ? (
          <p className="text-sm text-zinc-500">
            No sweeps recorded yet. Run a layer sweep from the Experiment Workbench to see it here.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-700/50 text-left text-xs uppercase tracking-wider text-zinc-500">
                  <th className="px-3 py-2">Name</th>
                  <th className="px-3 py-2">Layers</th>
                  <th className="px-3 py-2">Peak KL</th>
                  <th className="px-3 py-2">Peak Layer</th>
                  <th className="px-3 py-2">Changed</th>
                  <th className="px-3 py-2">Date</th>
                  <th className="px-3 py-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {sweepHistory.map((sweep) => (
                  <SweepRow
                    key={sweep.id}
                    sweep={sweep}
                    isSelected={selectedSweep?.id === sweep.id}
                    onSelect={() => loadSweep(sweep.id)}
                    onDelete={() => deleteSweep(sweep.id)}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {sweepLoading && (
        <div className="flex items-center gap-2 text-sm text-zinc-400">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
          Loading sweep results...
        </div>
      )}

      {selectedSweep && !sweepLoading && (
        <>
          <SweepChart results={selectedSweep.results} sweepId={selectedSweep.id} />
          {selectedSweep.insights.length > 0 && (
            <InsightsPanel insights={selectedSweep.insights} />
          )}
        </>
      )}
    </div>
  );
}

function SweepRow({
  sweep,
  isSelected,
  onSelect,
  onDelete,
}: {
  sweep: { id: string; name: string; num_layers: number; peak_kl: number; peak_layer: number; layers_changed: number; timestamp: string };
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
}) {
  const [confirming, setConfirming] = useState(false);

  const date = new Date(sweep.timestamp);
  const dateStr = date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <tr
      onClick={onSelect}
      className={`cursor-pointer border-b border-zinc-800/50 transition-colors hover:bg-zinc-800/40 ${
        isSelected ? 'bg-zinc-800/60' : ''
      }`}
    >
      <td className="px-3 py-2.5 font-medium text-zinc-200">{sweep.name}</td>
      <td className="px-3 py-2.5 font-mono text-zinc-400">{sweep.num_layers}</td>
      <td className="px-3 py-2.5 font-mono text-amber-400">{sweep.peak_kl.toFixed(4)}</td>
      <td className="px-3 py-2.5 font-mono text-zinc-400">{sweep.peak_layer}</td>
      <td className="px-3 py-2.5 font-mono text-zinc-400">
        {sweep.layers_changed}/{sweep.num_layers}
      </td>
      <td className="px-3 py-2.5 text-zinc-500">{dateStr}</td>
      <td className="px-3 py-2.5">
        <div className="flex gap-2" onClick={(e) => e.stopPropagation()}>
          {confirming ? (
            <>
              <button
                onClick={onDelete}
                className="rounded px-2 py-0.5 text-xs text-red-400 transition-colors hover:bg-red-500/20"
              >
                Confirm
              </button>
              <button
                onClick={() => setConfirming(false)}
                className="rounded px-2 py-0.5 text-xs text-zinc-500 transition-colors hover:bg-zinc-700"
              >
                Cancel
              </button>
            </>
          ) : (
            <button
              onClick={() => setConfirming(true)}
              className="rounded px-2 py-0.5 text-xs text-zinc-500 transition-colors hover:bg-zinc-700 hover:text-red-400"
            >
              Delete
            </button>
          )}
        </div>
      </td>
    </tr>
  );
}

function SweepChart({ results, sweepId }: { results: ExperimentResult[]; sweepId: string }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    if (!svgRef.current || results.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    const width = svgRef.current.clientWidth - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const data = results.map((r) => ({
      layer: r.config.interventions[0]?.target_layer ?? 0,
      kl: r.kl_divergence,
      changed: r.top_token_changed,
    }));

    const x = d3
      .scaleBand()
      .domain(data.map((d) => d.layer.toString()))
      .range([0, width])
      .padding(0.2);

    const y = d3
      .scaleLinear()
      .domain([0, d3.max(data, (d) => d.kl) ?? 1])
      .nice()
      .range([height, 0]);

    g.selectAll('.bar')
      .data(data)
      .join('rect')
      .attr('class', 'bar')
      .attr('x', (d) => x(d.layer.toString()) ?? 0)
      .attr('y', (d) => y(d.kl))
      .attr('width', x.bandwidth())
      .attr('height', (d) => height - y(d.kl))
      .attr('fill', (d) => (d.changed ? '#f59e0b' : '#3b82f6'))
      .attr('rx', 2);

    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(
        d3.axisBottom(x).tickValues(
          data
            .filter((_, i) => i % Math.ceil(data.length / 17) === 0)
            .map((d) => d.layer.toString()),
        ),
      )
      .selectAll('text')
      .attr('fill', '#a1a1aa')
      .style('font-size', '11px');

    g.append('text')
      .attr('x', width / 2)
      .attr('y', height + 35)
      .attr('text-anchor', 'middle')
      .attr('fill', '#71717a')
      .style('font-size', '12px')
      .text('Layer');

    g.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .selectAll('text')
      .attr('fill', '#a1a1aa')
      .style('font-size', '11px');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -45)
      .attr('text-anchor', 'middle')
      .attr('fill', '#71717a')
      .style('font-size', '12px')
      .text('KL Divergence');

    svg.selectAll('.domain, .tick line').attr('stroke', '#3f3f46');
  }, [results]);

  if (results.length === 0) return null;

  const maxKL = Math.max(...results.map((r) => r.kl_divergence));
  const maxLayer = results.find((r) => r.kl_divergence === maxKL);
  const changedCount = results.filter((r) => r.top_token_changed).length;

  const handleExportPDF = async () => {
    setExporting(true);
    try {
      await api.downloadSweepReportById(sweepId);
    } catch (e) {
      console.error('PDF export failed:', e);
    } finally {
      setExporting(false);
    }
  };

  return (
    <Panel
      title="Sweep Results"
      actions={
        <button
          onClick={handleExportPDF}
          disabled={exporting}
          className="rounded-lg border border-zinc-600 px-3 py-1 text-xs font-medium text-zinc-300 transition-colors hover:border-blue-500 hover:text-blue-400 disabled:opacity-40"
        >
          {exporting ? 'Generating...' : 'Export PDF'}
        </button>
      }
    >
      <div className="space-y-4">
        <div className="flex gap-4 text-sm">
          <span className="inline-flex items-center gap-1 text-zinc-400">
            Peak KL:{' '}
            <span className="font-mono text-amber-400">{maxKL.toFixed(4)}</span> at layer{' '}
            {maxLayer?.config.interventions[0]?.target_layer}
          </span>
          <span className="text-zinc-400">
            Top token changed in{' '}
            <span className="font-mono text-amber-400">
              {changedCount}/{results.length}
            </span>{' '}
            layers
          </span>
        </div>

        <svg ref={svgRef} width="100%" height={300} />

        <div className="flex items-center gap-4 text-xs text-zinc-500">
          <span className="flex items-center gap-1">
            <span className="inline-block h-3 w-3 rounded-sm bg-amber-500" /> Top token changed
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-3 w-3 rounded-sm bg-blue-500" /> Top token preserved
          </span>
        </div>
      </div>
    </Panel>
  );
}
