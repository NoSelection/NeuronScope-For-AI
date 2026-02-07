import type { TokenInfo } from '../../api/types';

interface TokenDisplayProps {
  tokens: TokenInfo[];
  selectedPosition: number | null;
  onSelectPosition?: (pos: number) => void;
  highlightPositions?: Set<number>;
}

export function TokenDisplay({
  tokens,
  selectedPosition,
  onSelectPosition,
  highlightPositions,
}: TokenDisplayProps) {
  return (
    <div className="flex flex-wrap gap-0.5">
      {tokens.map((t) => {
        const isSelected = selectedPosition === t.position;
        const isHighlighted = highlightPositions?.has(t.position);

        return (
          <button
            key={t.position}
            onClick={() => onSelectPosition?.(t.position)}
            className={`rounded px-1.5 py-0.5 font-mono text-sm transition-colors ${
              isSelected
                ? 'bg-blue-600 text-white'
                : isHighlighted
                  ? 'bg-amber-600/30 text-amber-200'
                  : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
            }`}
            title={`Position ${t.position} | ID ${t.token_id}`}
          >
            {t.token.replace(/ /g, '\u00B7')}
          </button>
        );
      })}
    </div>
  );
}
