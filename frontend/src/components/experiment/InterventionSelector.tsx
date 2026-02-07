import { useExperimentStore } from '../../stores/experimentStore';
import { COMPONENT_LABELS, INTERVENTION_LABELS } from '../../api/types';
import type { ComponentType, InterventionType } from '../../api/types';

const COMPONENTS = Object.entries(COMPONENT_LABELS) as [ComponentType, string][];
const INTERVENTION_TYPES = Object.entries(INTERVENTION_LABELS) as [InterventionType, string][];

interface Props {
  numLayers: number;
}

export function InterventionSelector({ numLayers }: Props) {
  const { config, updateIntervention, addIntervention, removeIntervention } =
    useExperimentStore();

  return (
    <div className="space-y-3">
      {config.interventions.map((intervention, idx) => (
        <div
          key={idx}
          className="rounded-lg border border-zinc-700/50 bg-zinc-800/50 p-3"
        >
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs font-medium text-zinc-400">
              Intervention {idx + 1}
            </span>
            {config.interventions.length > 1 && (
              <button
                onClick={() => removeIntervention(idx)}
                className="text-xs text-red-400 hover:text-red-300"
              >
                Remove
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3">
            {/* Layer */}
            <div>
              <label className="mb-1 block text-xs text-zinc-500">Layer</label>
              <select
                value={intervention.target_layer}
                onChange={(e) =>
                  updateIntervention(idx, {
                    target_layer: parseInt(e.target.value),
                  })
                }
                className="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-200"
              >
                {Array.from({ length: numLayers }, (_, i) => (
                  <option key={i} value={i}>
                    Layer {i}
                  </option>
                ))}
              </select>
            </div>

            {/* Component */}
            <div>
              <label className="mb-1 block text-xs text-zinc-500">Component</label>
              <select
                value={intervention.target_component}
                onChange={(e) =>
                  updateIntervention(idx, {
                    target_component: e.target.value as ComponentType,
                  })
                }
                className="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-200"
              >
                {COMPONENTS.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </div>

            {/* Intervention Type */}
            <div>
              <label className="mb-1 block text-xs text-zinc-500">Type</label>
              <select
                value={intervention.intervention_type}
                onChange={(e) =>
                  updateIntervention(idx, {
                    intervention_type: e.target.value as InterventionType,
                  })
                }
                className="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-200"
              >
                {INTERVENTION_TYPES.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </div>

            {/* Token Position (optional) */}
            <div>
              <label className="mb-1 block text-xs text-zinc-500">
                Token Position
              </label>
              <input
                type="number"
                value={intervention.target_position ?? ''}
                onChange={(e) =>
                  updateIntervention(idx, {
                    target_position: e.target.value
                      ? parseInt(e.target.value)
                      : null,
                  })
                }
                placeholder="All"
                min={0}
                className="w-full rounded border border-zinc-600 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-200 placeholder-zinc-600"
              />
            </div>
          </div>
        </div>
      ))}

      <button
        onClick={addIntervention}
        className="w-full rounded-lg border border-dashed border-zinc-600 py-2 text-sm text-zinc-400 hover:border-zinc-500 hover:text-zinc-300"
      >
        + Add Intervention
      </button>
    </div>
  );
}
