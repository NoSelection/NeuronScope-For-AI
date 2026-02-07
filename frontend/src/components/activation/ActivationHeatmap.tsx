import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { CaptureResult } from '../../api/types';

interface Props {
  capture: CaptureResult;
}

export function ActivationHeatmap({ capture }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const { values, shape } = capture;

  useEffect(() => {
    if (!svgRef.current || !values || values.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Determine 2D dimensions from shape
    let rows: number;
    let cols: number;

    if (shape.length === 3) {
      // [batch, seq, hidden] — take batch=0, display seq x hidden
      rows = shape[1];
      cols = shape[2];
    } else if (shape.length === 2) {
      rows = shape[0];
      cols = shape[1];
    } else {
      // 1D — display as single row
      rows = 1;
      cols = values.length;
    }

    // Downsample columns if too large
    const maxCols = 256;
    let displayData: number[][];
    let displayCols: number;

    if (cols > maxCols) {
      displayCols = maxCols;
      const binSize = cols / maxCols;
      displayData = [];
      for (let r = 0; r < rows; r++) {
        const row: number[] = [];
        for (let c = 0; c < maxCols; c++) {
          const start = Math.floor(c * binSize);
          const end = Math.floor((c + 1) * binSize);
          let sum = 0;
          let count = 0;
          for (let k = start; k < end && k < cols; k++) {
            const idx = r * cols + k;
            if (idx < values.length) {
              sum += values[idx];
              count++;
            }
          }
          row.push(count > 0 ? sum / count : 0);
        }
        displayData.push(row);
      }
    } else {
      displayCols = cols;
      displayData = [];
      for (let r = 0; r < rows; r++) {
        const row: number[] = [];
        for (let c = 0; c < cols; c++) {
          const idx = r * cols + c;
          row.push(idx < values.length ? values[idx] : 0);
        }
        displayData.push(row);
      }
    }

    // Layout
    const margin = { top: 8, right: 60, bottom: 30, left: 50 };
    const cellWidth = Math.max(1, Math.min(4, Math.floor(500 / displayCols)));
    const cellHeight = Math.max(4, Math.min(20, Math.floor(200 / rows)));
    const width = displayCols * cellWidth;
    const height = rows * cellHeight;
    const totalWidth = width + margin.left + margin.right;
    const totalHeight = height + margin.top + margin.bottom;

    svg.attr('width', totalWidth).attr('height', totalHeight);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Color scale
    const allVals = displayData.flat();
    const minVal = d3.min(allVals) ?? 0;
    const maxVal = d3.max(allVals) ?? 1;
    const colorScale = d3
      .scaleSequential(d3.interpolateViridis)
      .domain([minVal, maxVal]);

    // Draw cells
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < displayCols; c++) {
        g.append('rect')
          .attr('x', c * cellWidth)
          .attr('y', r * cellHeight)
          .attr('width', cellWidth)
          .attr('height', cellHeight)
          .attr('fill', colorScale(displayData[r][c]))
          .on('mouseover', function (event) {
            if (tooltipRef.current) {
              const val = displayData[r][c];
              tooltipRef.current.style.display = 'block';
              tooltipRef.current.style.left = `${event.offsetX + 10}px`;
              tooltipRef.current.style.top = `${event.offsetY - 30}px`;
              tooltipRef.current.textContent = `[${r}, ${c}] = ${val.toFixed(4)}`;
            }
          })
          .on('mouseout', function () {
            if (tooltipRef.current) {
              tooltipRef.current.style.display = 'none';
            }
          });
      }
    }

    // X axis label
    g.append('text')
      .attr('x', width / 2)
      .attr('y', height + 22)
      .attr('text-anchor', 'middle')
      .attr('fill', '#71717a')
      .style('font-size', '10px')
      .text(cols > maxCols ? `Neuron (binned ${cols} → ${maxCols})` : 'Neuron');

    // Y axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -35)
      .attr('text-anchor', 'middle')
      .attr('fill', '#71717a')
      .style('font-size', '10px')
      .text('Position');

    // Color legend
    const legendWidth = 12;
    const legendHeight = height;
    const legendG = svg
      .append('g')
      .attr('transform', `translate(${margin.left + width + 10},${margin.top})`);

    const legendScale = d3.scaleLinear().domain([minVal, maxVal]).range([legendHeight, 0]);

    const numStops = 64;
    for (let i = 0; i < numStops; i++) {
      const val = minVal + ((maxVal - minVal) * i) / (numStops - 1);
      legendG
        .append('rect')
        .attr('x', 0)
        .attr('y', legendScale(val))
        .attr('width', legendWidth)
        .attr('height', legendHeight / numStops + 1)
        .attr('fill', colorScale(val));
    }

    // Legend axis
    const legendAxis = d3.axisRight(legendScale).ticks(4);
    legendG
      .append('g')
      .attr('transform', `translate(${legendWidth},0)`)
      .call(legendAxis)
      .selectAll('text')
      .attr('fill', '#a1a1aa')
      .style('font-size', '9px');

    legendG.selectAll('.domain, .tick line').attr('stroke', '#3f3f46');
  }, [values, shape]);

  if (!values || values.length === 0) return null;

  return (
    <div className="relative mt-3">
      <div className="overflow-x-auto rounded-lg border border-zinc-700/50 bg-zinc-800/20 p-2">
        <svg ref={svgRef} />
        <div
          ref={tooltipRef}
          className="pointer-events-none absolute hidden rounded border border-zinc-600 bg-zinc-800 px-2 py-1 font-mono text-[10px] text-zinc-200 shadow-lg"
        />
      </div>
    </div>
  );
}
