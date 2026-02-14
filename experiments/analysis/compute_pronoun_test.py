"""CRITICAL TEST: Does the Layer 26 effect also appear for control (first-person non-AI)?

If control_KL - third_person_KL ≈ self_KL - third_person_KL at Layer 26,
then the effect is a PRONOUN effect, not a self-referential effect.

We can compute this from existing sweep data without running new experiments!
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

SWEEP_DIR = Path("results/self_model_circuits_v2/sweeps")

exp_names = ['self_recognition', 'capability_awareness', 'metacognition']

print("PRONOUN CONFOUND TEST — Layer 26")
print("=" * 70)
print()
print("If Layer 26 encodes SELF-REFERENCE: self_KL >> control_KL ~ third_KL")
print("If Layer 26 encodes FIRST-PERSON:   self_KL ~ control_KL >> third_KL")
print()

for target_layer in [26, 9, 27]:  # test all universal layers
    print(f"\n{'='*70}")
    print(f"LAYER {target_layer}")
    print(f"{'='*70}")

    for exp_name in exp_names:
        self_kls = []
        ctrl_kls = []
        third_kls = []

        for triplet_idx in range(8):
            for input_type, kl_list in [('self', self_kls), ('control', ctrl_kls), ('third_person', third_kls)]:
                fname = SWEEP_DIR / f"{exp_name}__{triplet_idx}__{input_type}__zero.json"
                with open(fname) as f:
                    sweep = json.load(f)
                for step in sweep:
                    interventions = step.get('config', {}).get('interventions', [])
                    if interventions and interventions[0].get('target_layer') == target_layer:
                        kl_list.append(step['kl_divergence'])
                        break

        self_kls = np.array(self_kls)
        ctrl_kls = np.array(ctrl_kls)
        third_kls = np.array(third_kls)

        # Differential KL: self vs third-person (what we report as "self-referential")
        diff_self = self_kls - third_kls
        # Differential KL: control vs third-person (pronoun effect without AI content)
        diff_ctrl = ctrl_kls - third_kls
        # Differential KL: self vs control (AI-specific beyond pronoun)
        diff_ai = self_kls - ctrl_kls

        print(f"\n  {exp_name}:")
        print(f"    Absolute KL:  self={np.mean(self_kls):.3f}  ctrl={np.mean(ctrl_kls):.3f}  3rd={np.mean(third_kls):.3f}")
        print(f"    self - 3rd  (\"self-referential\"): mean={np.mean(diff_self):.3f}  std={np.std(diff_self,ddof=1):.3f}")
        print(f"    ctrl - 3rd  (pronoun only):       mean={np.mean(diff_ctrl):.3f}  std={np.std(diff_ctrl,ddof=1):.3f}")
        print(f"    self - ctrl (AI-specific):         mean={np.mean(diff_ai):.3f}  std={np.std(diff_ai,ddof=1):.3f}")

        # Wilcoxon on ctrl - 3rd (would this also be significant?)
        try:
            stat, p = sp_stats.wilcoxon(diff_ctrl, alternative='two-sided')
            n_positive = np.sum(diff_ctrl > 0)
            print(f"    Wilcoxon(ctrl-3rd): W={stat}, p={p:.6f}, {n_positive}/8 positive")
        except Exception as e:
            print(f"    Wilcoxon(ctrl-3rd): {e}")

        # Wilcoxon on self - ctrl (is there ANY AI-specific residual?)
        try:
            stat, p = sp_stats.wilcoxon(diff_ai, alternative='two-sided')
            n_positive = np.sum(diff_ai > 0)
            print(f"    Wilcoxon(self-ctrl): W={stat}, p={p:.6f}, {n_positive}/8 positive")
        except Exception as e:
            print(f"    Wilcoxon(self-ctrl): {e}")

        # Per-triplet comparison
        print(f"    Per-triplet [self-3rd vs ctrl-3rd]:")
        for i in range(8):
            print(f"      T{i}: self-3rd={diff_self[i]:+.3f}  ctrl-3rd={diff_ctrl[i]:+.3f}  delta={diff_self[i]-diff_ctrl[i]:+.3f}")

print("\n\nSUMMARY")
print("=" * 70)
print("If ctrl-3rd differentials are similar to self-3rd differentials,")
print("then the Layer 26 effect is primarily a FIRST-PERSON PRONOUN effect,")
print("not a self-referential AI processing effect.")
print("The self-ctrl differential (AI-specific residual) tells you how much")
print("is truly self-referential beyond the pronoun baseline.")
