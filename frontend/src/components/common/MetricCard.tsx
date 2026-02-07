import { InfoTip } from './InfoTip';
import type { EducationKey } from '../../education/content';

interface MetricCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  highlight?: boolean;
  topic?: EducationKey;
}

export function MetricCard({ label, value, subtitle, highlight, topic }: MetricCardProps) {
  return (
    <div
      className={`rounded-lg border p-4 ${
        highlight
          ? 'border-amber-500/50 bg-amber-500/10'
          : 'border-zinc-700 bg-zinc-800/50'
      }`}
    >
      <div className="flex items-center gap-1.5 text-xs uppercase tracking-wider text-zinc-400">
        {label}
        {topic && <InfoTip topic={topic} iconOnly />}
      </div>
      <div className="mt-1 text-2xl font-semibold text-zinc-100">
        {typeof value === 'number' ? value.toFixed(4) : value}
      </div>
      {subtitle && <div className="mt-1 text-xs text-zinc-500">{subtitle}</div>}
    </div>
  );
}
