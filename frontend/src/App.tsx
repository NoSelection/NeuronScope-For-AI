import { useState } from 'react';
import { ExperimentWorkbench } from './views/ExperimentWorkbench';
import { ActivationExplorer } from './views/ActivationExplorer';
import { WalkthroughOverlay } from './components/common/WalkthroughOverlay';
import { WelcomeBanner } from './components/common/WelcomeBanner';
import { useWalkthroughStore } from './stores/walkthroughStore';

type View = 'workbench' | 'activations';

function App() {
  const [view, setView] = useState<View>('workbench');
  const { dismissed, start: startTour } = useWalkthroughStore();

  return (
    <div className="min-h-screen bg-[#0a0a0f]">
      {/* Guided walkthrough overlay */}
      <WalkthroughOverlay />

      {/* Header */}
      <header className="border-b border-zinc-800 bg-zinc-900/50" data-tour="header">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold text-zinc-100">NeuronScope</h1>
            <span className="rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400">
              v0.1.0
            </span>
          </div>

          <nav className="flex items-center gap-1">
            <NavButton
              active={view === 'workbench'}
              onClick={() => setView('workbench')}
            >
              Experiment Workbench
            </NavButton>
            <NavButton
              active={view === 'activations'}
              onClick={() => setView('activations')}
            >
              Activation Explorer
            </NavButton>
            {dismissed && (
              <button
                onClick={startTour}
                className="ml-2 rounded-lg border border-zinc-700 px-2.5 py-1 text-xs text-zinc-500 transition-colors hover:border-blue-500 hover:text-blue-400"
                title="Restart guided tour"
              >
                ? Tour
              </button>
            )}
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-6 py-6">
        {view === 'workbench' && (
          <>
            <WelcomeBanner />
            <ExperimentWorkbench />
          </>
        )}
        {view === 'activations' && <ActivationExplorer />}
      </main>
    </div>
  );
}

function NavButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-lg px-3 py-1.5 text-sm transition-colors ${
        active
          ? 'bg-zinc-800 text-zinc-100'
          : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'
      }`}
    >
      {children}
    </button>
  );
}

export default App;
