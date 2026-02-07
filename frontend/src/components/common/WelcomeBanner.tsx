import { useWalkthroughStore } from '../../stores/walkthroughStore';

/**
 * Shows a welcome banner for first-time users.
 * Once dismissed (or walkthrough started), it won't appear again.
 */
export function WelcomeBanner() {
  const { dismissed, start, dismiss } = useWalkthroughStore();

  if (dismissed) return null;

  return (
    <div className="mb-5 rounded-xl border border-blue-500/30 bg-gradient-to-r from-blue-500/10 via-indigo-500/10 to-purple-500/10 p-5">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-base font-semibold text-zinc-100">
            Welcome to NeuronScope
          </h2>
          <p className="mt-1 max-w-2xl text-sm leading-relaxed text-zinc-300">
            This tool lets you run real causal experiments on an AI language model.
            You&apos;ll disable or modify specific internal components and observe
            exactly how the model&apos;s predictions change. No prior ML experience needed.
          </p>
          <div className="mt-3 flex gap-3">
            <button
              onClick={start}
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-500"
            >
              Take the Guided Tour
            </button>
            <button
              onClick={dismiss}
              className="rounded-lg border border-zinc-600 px-4 py-2 text-sm font-medium text-zinc-300 transition-colors hover:bg-zinc-800"
            >
              I know what I&apos;m doing
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
