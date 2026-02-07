import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface Props {
  values: number[];
  comparisonValues?: number[];
  label?: string;
}

export function DistributionPlot({ values, comparisonValues, label }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || values.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Layout
    const margin = { top: 16, right: 16, bottom: 32, left: 48 };
    const width = 460;
    const height = 140;
    const totalWidth = width + margin.left + margin.right;
    const totalHeight = height + margin.top + margin.bottom;

    svg.attr('width', totalWidth).attr('height', totalHeight);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Determine domain from all data
    const allValues = comparisonValues ? [...values, ...comparisonValues] : values;
    const xMin = d3.min(allValues) ?? 0;
    const xMax = d3.max(allValues) ?? 1;
    const padding = (xMax - xMin) * 0.05 || 0.5;

    const x = d3.scaleLinear().domain([xMin - padding, xMax + padding]).range([0, width]);

    // Create histogram bins
    const numBins = 50;
    const binGenerator = d3.bin<number, number>()
      .domain(x.domain() as [number, number])
      .thresholds(numBins);

    const bins = binGenerator(values);
    const compBins = comparisonValues ? binGenerator(comparisonValues) : null;

    // Y scale based on max count
    const maxCount = Math.max(
      d3.max(bins, (b) => b.length) ?? 0,
      compBins ? (d3.max(compBins, (b) => b.length) ?? 0) : 0,
    );
    const y = d3.scaleLinear().domain([0, maxCount]).range([height, 0]).nice();

    // Draw primary distribution
    g.selectAll('.bar-primary')
      .data(bins)
      .enter()
      .append('rect')
      .attr('x', (d) => x(d.x0 ?? 0))
      .attr('y', (d) => y(d.length))
      .attr('width', (d) => Math.max(0, x(d.x1 ?? 0) - x(d.x0 ?? 0) - 1))
      .attr('height', (d) => height - y(d.length))
      .attr('fill', '#3b82f6')
      .attr('opacity', compBins ? 0.5 : 0.7);

    // Draw comparison distribution
    if (compBins) {
      g.selectAll('.bar-comparison')
        .data(compBins)
        .enter()
        .append('rect')
        .attr('x', (d) => x(d.x0 ?? 0))
        .attr('y', (d) => y(d.length))
        .attr('width', (d) => Math.max(0, x(d.x1 ?? 0) - x(d.x0 ?? 0) - 1))
        .attr('height', (d) => height - y(d.length))
        .attr('fill', '#ef4444')
        .attr('opacity', 0.45);
    }

    // X axis
    const xAxis = d3.axisBottom(x).ticks(6);
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(xAxis)
      .selectAll('text')
      .attr('fill', '#a1a1aa')
      .style('font-size', '9px');

    g.selectAll('.domain').attr('stroke', '#3f3f46');
    g.selectAll('.tick line').attr('stroke', '#3f3f46');

    // X axis label
    g.append('text')
      .attr('x', width / 2)
      .attr('y', height + 28)
      .attr('text-anchor', 'middle')
      .attr('fill', '#71717a')
      .style('font-size', '10px')
      .text('Activation Value');

    // Y axis
    const yAxis = d3.axisLeft(y).ticks(4);
    g.append('g')
      .call(yAxis)
      .selectAll('text')
      .attr('fill', '#a1a1aa')
      .style('font-size', '9px');

    // Y axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -36)
      .attr('text-anchor', 'middle')
      .attr('fill', '#71717a')
      .style('font-size', '10px')
      .text('Count');

    // Legend
    if (compBins) {
      const legendG = svg.append('g').attr('transform', `translate(${margin.left + width - 110},${margin.top})`);

      legendG.append('rect').attr('x', 0).attr('y', 0).attr('width', 10).attr('height', 10).attr('fill', '#3b82f6').attr('opacity', 0.6);
      legendG.append('text').attr('x', 14).attr('y', 9).attr('fill', '#a1a1aa').style('font-size', '9px').text('Clean');

      legendG.append('rect').attr('x', 50).attr('y', 0).attr('width', 10).attr('height', 10).attr('fill', '#ef4444').attr('opacity', 0.55);
      legendG.append('text').attr('x', 64).attr('y', 9).attr('fill', '#a1a1aa').style('font-size', '9px').text('Intervention');
    }

    // Title
    if (label) {
      svg.append('text')
        .attr('x', totalWidth / 2)
        .attr('y', 12)
        .attr('text-anchor', 'middle')
        .attr('fill', '#a1a1aa')
        .style('font-size', '10px')
        .text(label);
    }
  }, [values, comparisonValues, label]);

  if (values.length === 0) return null;

  return (
    <div className="mt-2 overflow-x-auto rounded-lg border border-zinc-700/50 bg-zinc-800/20 p-2">
      <svg ref={svgRef} />
    </div>
  );
}
