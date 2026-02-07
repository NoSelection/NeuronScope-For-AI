import { useEffect, useState } from 'react';
import { useWalkthroughStore } from '../../stores/walkthroughStore';
import { WALKTHROUGH_STEPS } from '../../education/walkthrough';

/**
 * Full-screen overlay that highlights one element at a time
 * and shows a tooltip explaining what it does.
 */
export function WalkthroughOverlay() {
  const { active, stepIndex, next, prev, dismiss } = useWalkthroughStore();
  const [rect, setRect] = useState<DOMRect | null>(null);

  const step = WALKTHROUGH_STEPS[stepIndex];
  const isFirst = stepIndex === 0;
  const isLast = stepIndex === WALKTHROUGH_STEPS.length - 1;

  // Find and scroll to the target element
  useEffect(() => {
    if (!active || !step) return;

    const el = document.querySelector(step.target);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      // Small delay for scroll to complete
      const timer = setTimeout(() => {
        setRect(el.getBoundingClientRect());
      }, 300);
      return () => clearTimeout(timer);
    } else {
      setRect(null);
    }
  }, [active, step, stepIndex]);

  // Recalculate on scroll/resize
  useEffect(() => {
    if (!active || !step) return;

    function update() {
      const el = document.querySelector(step.target);
      if (el) setRect(el.getBoundingClientRect());
    }

    window.addEventListener('scroll', update, true);
    window.addEventListener('resize', update);
    return () => {
      window.removeEventListener('scroll', update, true);
      window.removeEventListener('resize', update);
    };
  }, [active, step]);

  if (!active || !step) return null;

  const pad = 8;
  const tooltipStyle = computeTooltipPosition(rect, step.placement, pad);

  return (
    <div className="fixed inset-0 z-[100]">
      {/* Semi-transparent backdrop */}
      <div
        className="absolute inset-0 bg-black/60"
        onClick={dismiss}
      />

      {/* Highlight cutout */}
      {rect && (
        <div
          className="absolute rounded-lg border-2 border-blue-500 shadow-[0_0_0_9999px_rgba(0,0,0,0.6)]"
          style={{
            top: rect.top - pad,
            left: rect.left - pad,
            width: rect.width + pad * 2,
            height: rect.height + pad * 2,
            pointerEvents: 'none',
          }}
        />
      )}

      {/* Tooltip card */}
      <div
        className="absolute z-[101] w-80 rounded-xl border border-zinc-600 bg-zinc-900 p-5 shadow-2xl shadow-black/50"
        style={tooltipStyle}
      >
        {/* Step counter */}
        <div className="mb-2 flex items-center gap-2">
          <span className="rounded-full bg-blue-600 px-2.5 py-0.5 text-[11px] font-semibold text-white">
            {stepIndex + 1} / {WALKTHROUGH_STEPS.length}
          </span>
        </div>

        <h3 className="mb-2 text-sm font-semibold text-zinc-100">{step.title}</h3>
        <p className="text-[13px] leading-relaxed text-zinc-300">{step.body}</p>

        {/* Navigation */}
        <div className="mt-4 flex items-center justify-between">
          <button
            onClick={dismiss}
            className="text-xs text-zinc-500 hover:text-zinc-300"
          >
            Skip tour
          </button>
          <div className="flex gap-2">
            {!isFirst && (
              <button
                onClick={prev}
                className="rounded-lg border border-zinc-600 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-800"
              >
                Back
              </button>
            )}
            <button
              onClick={next}
              className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-500"
            >
              {isLast ? 'Finish' : 'Next'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function computeTooltipPosition(
  rect: DOMRect | null,
  placement: string,
  pad: number,
): React.CSSProperties {
  if (!rect) {
    return { top: '50%', left: '50%', transform: 'translate(-50%, -50%)' };
  }

  const gap = 16;
  switch (placement) {
    case 'bottom':
      return { top: rect.bottom + pad + gap, left: rect.left + rect.width / 2 - 160 };
    case 'top':
      return { bottom: window.innerHeight - rect.top + pad + gap, left: rect.left + rect.width / 2 - 160 };
    case 'right':
      return { top: rect.top + rect.height / 2 - 80, left: rect.right + pad + gap };
    case 'left':
      return { top: rect.top + rect.height / 2 - 80, right: window.innerWidth - rect.left + pad + gap };
    default:
      return { top: rect.bottom + pad + gap, left: rect.left };
  }
}
