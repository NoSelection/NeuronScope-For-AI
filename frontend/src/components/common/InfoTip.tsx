import { useState, useRef, useEffect, type ReactNode } from 'react';
import { EDUCATION, type EducationKey } from '../../education/content';

interface InfoTipProps {
  /** Key into the EDUCATION dictionary */
  topic: EducationKey;
  /** Optional children — if provided, renders inline next to the (?) icon */
  children?: ReactNode;
  /** If true, render just the icon (no label text) */
  iconOnly?: boolean;
}

/**
 * Educational tooltip that shows a (?) icon. Hover or click to reveal
 * a plain-language explanation. Designed for CS students / ML beginners.
 */
export function InfoTip({ topic, children, iconOnly }: InfoTipProps) {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const entry = EDUCATION[topic];

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setExpanded(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [open]);

  return (
    <span className="inline-flex items-center gap-1.5" ref={ref}>
      {!iconOnly && children}
      <span
        role="button"
        tabIndex={0}
        onClick={() => setOpen(!open)}
        onMouseEnter={() => setOpen(true)}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen(!open); } }}
        className="relative inline-flex h-4 w-4 flex-shrink-0 cursor-pointer items-center justify-center rounded-full border border-zinc-600 text-[10px] font-bold leading-none text-zinc-500 transition-colors hover:border-blue-500 hover:text-blue-400"
        aria-label={`Learn about ${topic.replace(/_/g, ' ')}`}
      >
        ?
        {open && (
          <div
            className="absolute bottom-full left-1/2 z-50 mb-2 w-72 -translate-x-1/2 rounded-lg border border-zinc-600 bg-zinc-800 p-3 text-left shadow-xl shadow-black/50"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Arrow */}
            <div className="absolute -bottom-1.5 left-1/2 h-3 w-3 -translate-x-1/2 rotate-45 border-b border-r border-zinc-600 bg-zinc-800" />

            <p className="text-xs font-normal leading-relaxed text-zinc-200">
              {entry.short}
            </p>

            {!expanded ? (
              <span
                role="button"
                tabIndex={0}
                onClick={(e) => {
                  e.stopPropagation();
                  setExpanded(true);
                }}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setExpanded(true); } }}
                className="mt-2 block cursor-pointer text-[11px] font-medium text-blue-400 hover:text-blue-300"
              >
                Tell me more...
              </span>
            ) : (
              <div className="mt-2 space-y-2">
                <p className="text-[11px] leading-relaxed text-zinc-300">
                  {entry.long}
                </p>
                {'analogy' in entry && entry.analogy && (
                  <p className="text-[11px] italic leading-relaxed text-zinc-400">
                    Analogy: {entry.analogy}
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </span>
    </span>
  );
}

/**
 * A label with an integrated InfoTip. Replaces bare <label> elements
 * throughout the UI to add educational context.
 */
export function InfoLabel({
  topic,
  children,
  className = '',
}: {
  topic: EducationKey;
  children: ReactNode;
  className?: string;
}) {
  return (
    <label className={`mb-1 flex items-center gap-1.5 text-xs text-zinc-400 ${className}`}>
      {children}
      <InfoTip topic={topic} iconOnly />
    </label>
  );
}
