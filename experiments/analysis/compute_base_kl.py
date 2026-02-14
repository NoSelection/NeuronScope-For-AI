"""Extract absolute KL values for Layer 26 from sweep files to provide frame of reference."""
import json
import numpy as np
from pathlib import Path

SWEEP_DIR = Path("results/self_model_circuits_v2/sweeps")

exp_names = ['self_recognition', 'capability_awareness', 'metacognition', 'training_knowledge']

print("BASE KL VALUES — LAYER 26 (zero ablation)")
print("=" * 70)

for exp_name in exp_names:
    print(f"\n{exp_name}:")
    for input_type in ['self', 'control', 'third_person']:
        kl_values = []
        for triplet_idx in range(8):
            fname = SWEEP_DIR / f"{exp_name}__{triplet_idx}__{input_type}__zero.json"
            if fname.exists():
                with open(fname) as f:
                    sweep = json.load(f)
                for step in sweep:
                    interventions = step.get('config', {}).get('interventions', [])
                    if interventions and interventions[0].get('target_layer') == 26:
                        kl = step.get('kl_divergence', None)
                        if kl is not None:
                            kl_values.append(kl)
                        break
        if kl_values:
            print(f"  {input_type:15s}: mean_KL={np.mean(kl_values):.3f}  std={np.std(kl_values):.3f}  n={len(kl_values)}")
        else:
            print(f"  {input_type:15s}: no data (checking structure...)")

# Debug: check actual structure of a step with layer 26
print("\n\nDEBUG — Step structure for layer 26:")
fname = SWEEP_DIR / "self_recognition__0__self__zero.json"
with open(fname) as f:
    sweep = json.load(f)
for step in sweep:
    interventions = step.get('config', {}).get('interventions', [])
    if interventions and interventions[0].get('target_layer') == 26:
        print(f"  Top-level keys: {list(step.keys())}")
        # Print all non-list/dict values
        for k, v in step.items():
            if not isinstance(v, (list, dict)):
                print(f"  {k}: {v}")
            elif isinstance(v, dict) and k != 'config':
                print(f"  {k}: {list(v.keys())}")
        break
