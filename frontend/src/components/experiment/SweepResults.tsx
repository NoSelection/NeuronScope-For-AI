import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { ExperimentResult } from '../../api/types';
import { Panel } from '../common/Panel';
import { InfoTip } from '../common/InfoTip';

interface Props {
  results: ExperimentResult[];
}

export function SweepResults({ results }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

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

    // Bars
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

    // X axis
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x).tickValues(
        data.filter((_, i) => i % Math.ceil(data.length / 17) === 0).map((d) => d.layer.toString())
      ))
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

    // Y axis
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

    // Style axis lines
    svg.selectAll('.domain, .tick line').attr('stroke', '#3f3f46');
  }, [results]);

  if (results.length === 0) return null;

  const maxKL = Math.max(...results.map((r) => r.kl_divergence));
  const maxLayer = results.find((r) => r.kl_divergence === maxKL);
  const changedCount = results.filter((r) => r.top_token_changed).length;

  return (
    <Panel title="Layer Sweep Results">
      <div className="space-y-4">
        <div className="mb-1 flex items-center gap-1.5 text-xs text-zinc-500">
          <InfoTip topic="layer_sweep">What is a layer sweep?</InfoTip>
        </div>
        <div className="flex gap-4 text-sm">
          <span className="inline-flex items-center gap-1 text-zinc-400">
            Peak KL: <span className="font-mono text-amber-400">{maxKL.toFixed(4)}</span> at
            layer {maxLayer?.config.interventions[0]?.target_layer}
            <InfoTip topic="peak_kl" iconOnly />
          </span>
          <span className="text-zinc-400">
            Top token changed in{' '}
            <span className="font-mono text-amber-400">{changedCount}/{results.length}</span>{' '}
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
