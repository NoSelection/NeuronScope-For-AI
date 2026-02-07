import type { Insight } from '../../api/types';
import { Panel } from '../common/Panel';
import { InfoTip } from '../common/InfoTip';

interface Props {
  insights: Insight[];
}

const TYPE_STYLES = {
  critical: {
    border: 'border-red-500/40',
    bg: 'bg-red-500/10',
    badge: 'bg-red-500/20 text-red-300',
    badgeLabel: 'Important',
  },
  notable: {
    border: 'border-amber-500/40',
    bg: 'bg-amber-500/10',
    badge: 'bg-amber-500/20 text-amber-300',
    badgeLabel: 'Notable',
  },
  info: {
    border: 'border-blue-500/30',
    bg: 'bg-blue-500/5',
    badge: 'bg-blue-500/20 text-blue-300',
    badgeLabel: 'Info',
  },
};

export function InsightsPanel({ insights }: Props) {
  if (insights.length === 0) return null;

  return (
    <Panel title="What This Means">
      <div className="mb-3 flex items-center gap-1.5 text-xs text-zinc-500">
        <InfoTip topic="insights">What are insights?</InfoTip>
      </div>
      <div className="space-y-3">
        {insights.map((insight, i) => {
          const style = TYPE_STYLES[insight.type];
          return (
            <div
              key={i}
              className={`rounded-lg border ${style.border} ${style.bg} p-4`}
            >
              <div className="mb-1.5 flex items-center gap-2">
                <span
                  className={`rounded-full px-2 py-0.5 text-xs font-medium ${style.badge}`}
                >
                  {style.badgeLabel}
                </span>
                <span className="text-sm font-medium text-zinc-100">
                  {insight.title}
                </span>
              </div>
              <p className="text-sm leading-relaxed text-zinc-300">
                {insight.detail}
              </p>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}
