import type { ReactNode } from 'react';

interface PanelProps {
  title: string;
  children: ReactNode;
  actions?: ReactNode;
  className?: string;
}

export function Panel({ title, children, actions, className = '' }: PanelProps) {
  return (
    <div className={`rounded-xl border border-zinc-700/50 bg-zinc-900/80 ${className}`}>
      <div className="flex items-center justify-between border-b border-zinc-700/50 px-5 py-3">
        <h2 className="text-sm font-medium uppercase tracking-wider text-zinc-400">
          {title}
        </h2>
        {actions && <div className="flex gap-2">{actions}</div>}
      </div>
      <div className="p-5">{children}</div>
    </div>
  );
}
