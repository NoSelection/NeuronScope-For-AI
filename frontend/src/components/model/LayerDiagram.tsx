import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useModelStore } from '../../stores/modelStore';
import { useActivationStore } from '../../stores/activationStore';

interface Props {
  selectedLayer?: number;
  onSelectLayer?: (layer: number) => void;
}

export function LayerDiagram({ selectedLayer, onSelectLayer }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const { info } = useModelStore();

  const numLayers = info?.num_layers ?? 34;

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Layout constants
    const margin = { top: 20, right: 10, bottom: 20, left: 40 };
    const layerHeight = 28;
    const layerGap = 4;
    const blockWidth = 200;
    const componentWidth = 56;
    const componentGap = 4;
    const totalHeight = numLayers * (layerHeight + layerGap) + margin.top + margin.bottom;
    const totalWidth = blockWidth + margin.left + margin.right;

    svg.attr('width', totalWidth).attr('height', totalHeight);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Component colors (matching ModuleTree badge colors)
    const componentColors = {
      attn: { fill: '#1e3a5f', stroke: '#3b82f6', text: '#93c5fd' },  // blue
      mlp: { fill: '#14532d', stroke: '#22c55e', text: '#86efac' },   // green
      norm: { fill: '#27272a', stroke: '#71717a', text: '#a1a1aa' },  // zinc
      residual: { fill: '#3b1f54', stroke: '#a855f7', text: '#d8b4fe' }, // purple
    };

    // Draw layers from bottom (layer 0) to top (layer N-1), but render top-down
    for (let i = 0; i < numLayers; i++) {
      const y = i * (layerHeight + layerGap);
      const isSelected = i === selectedLayer;

      // Layer background
      const layerG = g.append('g')
        .attr('transform', `translate(0,${y})`)
        .style('cursor', 'pointer')
        .on('click', () => onSelectLayer?.(i));

      layerG.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', blockWidth)
        .attr('height', layerHeight)
        .attr('rx', 4)
        .attr('fill', isSelected ? '#1e293b' : '#18181b')
        .attr('stroke', isSelected ? '#3b82f6' : '#27272a')
        .attr('stroke-width', isSelected ? 1.5 : 0.5);

      // Layer index
      g.append('text')
        .attr('x', -6)
        .attr('y', y + layerHeight / 2 + 4)
        .attr('text-anchor', 'end')
        .attr('fill', isSelected ? '#93c5fd' : '#71717a')
        .style('font-size', '9px')
        .style('font-family', 'monospace')
        .text(`${i}`);

      // Components inside the layer
      const components = [
        { label: 'Norm', color: componentColors.norm },
        { label: 'Attn', color: componentColors.attn },
        { label: 'Norm', color: componentColors.norm },
        { label: 'MLP', color: componentColors.mlp },
      ];

      // Residual connection arrow (background line)
      const startX = 4;
      const endX = blockWidth - 4;
      layerG.append('line')
        .attr('x1', startX)
        .attr('y1', layerHeight / 2)
        .attr('x2', endX)
        .attr('y2', layerHeight / 2)
        .attr('stroke', componentColors.residual.stroke)
        .attr('stroke-width', 0.5)
        .attr('stroke-dasharray', '2,2')
        .attr('opacity', 0.3);

      // Draw component blocks
      const totalComponentWidth = components.length * componentWidth + (components.length - 1) * componentGap;
      const startOffset = (blockWidth - totalComponentWidth) / 2;

      components.forEach((comp, ci) => {
        const cx = startOffset + ci * (componentWidth + componentGap);
        const cy = 3;
        const ch = layerHeight - 6;

        layerG.append('rect')
          .attr('x', cx)
          .attr('y', cy)
          .attr('width', componentWidth)
          .attr('height', ch)
          .attr('rx', 3)
          .attr('fill', comp.color.fill)
          .attr('stroke', comp.color.stroke)
          .attr('stroke-width', 0.5);

        layerG.append('text')
          .attr('x', cx + componentWidth / 2)
          .attr('y', cy + ch / 2 + 3.5)
          .attr('text-anchor', 'middle')
          .attr('fill', comp.color.text)
          .style('font-size', '8px')
          .style('font-weight', '500')
          .text(comp.label);
      });
    }

    // Title
    svg.append('text')
      .attr('x', totalWidth / 2)
      .attr('y', 12)
      .attr('text-anchor', 'middle')
      .attr('fill', '#a1a1aa')
      .style('font-size', '10px')
      .text('Layer Architecture');

  }, [numLayers, selectedLayer, onSelectLayer]);

  return (
    <div className="mt-3 overflow-y-auto rounded-lg border border-zinc-700/50 bg-zinc-900/50 p-2" style={{ maxHeight: '320px' }}>
      <svg ref={svgRef} />
    </div>
  );
}
