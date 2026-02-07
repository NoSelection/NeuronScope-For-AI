interface MetricCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  highlight?: boolean;
}

export function MetricCard({ label, value, subtitle, highlight }: MetricCardProps) {
  return (
    <div
      className={`rounded-lg border p-4 ${
        highlight
          ? 'border-amber-500/50 bg-amber-500/10'
          : 'border-zinc-700 bg-zinc-800/50'
      }`}
    >
      <div className="text-xs uppercase tracking-wider text-zinc-400">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-zinc-100">
        {typeof value === 'number' ? value.toFixed(4) : value}
      </div>
      {subtitle && <div className="mt-1 text-xs text-zinc-500">{subtitle}</div>}
    </div>
  );
}
