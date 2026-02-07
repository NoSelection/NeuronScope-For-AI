import { useExperimentStore } from '../stores/experimentStore';
import { ModelInfoPanel } from '../components/model/ModelInfoPanel';
import { ExperimentBuilder } from '../components/experiment/ExperimentBuilder';
import { ResultsPanel } from '../components/experiment/ResultsPanel';
import { SweepResults } from '../components/experiment/SweepResults';
import { InsightsPanel } from '../components/experiment/InsightsPanel';
import { ExperimentHistory } from '../components/experiment/ExperimentHistory';

export function ExperimentWorkbench() {
  const { currentResult, sweepResults, insights } = useExperimentStore();

  const hasResults = currentResult || sweepResults.length > 0;

  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
      {/* Left column: model + builder */}
      <div className="space-y-5 lg:col-span-5">
        <ModelInfoPanel />
        <ExperimentBuilder />
        <ExperimentHistory />
      </div>

      {/* Right column: results + insights */}
      <div className="space-y-5 lg:col-span-7">
        {insights.length > 0 && <InsightsPanel insights={insights} />}
        {currentResult && <ResultsPanel result={currentResult} />}
        {sweepResults.length > 0 && <SweepResults results={sweepResults} />}
        {!hasResults && (
          <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-zinc-700/50 text-zinc-500">
            <div className="text-center">
              <p className="text-lg">No results yet</p>
              <p className="mt-1 text-sm">
                Configure an experiment and click Run to see causal effects
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
