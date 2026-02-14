"""Compute additional statistics requested by reviewers.

Outputs:
  - Bonferroni correction check (do key results survive?)
  - Cohen's d effect sizes for all significant layers
  - Rank-biserial correlation from Wilcoxon tests
  - Base KL frame of reference (what's the absolute KL when ablating layer 26?)
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

ANALYSIS = Path("results/self_model_circuits_v2/analysis")
SWEEP_DIR = Path("results/self_model_circuits_v2/sweeps")
NUM_LAYERS = 34

# Load analysis data
experiments = {}
for name in ['self_recognition', 'capability_awareness', 'training_knowledge', 'metacognition']:
    with open(ANALYSIS / f'{name}.json') as f:
        experiments[name] = json.load(f)

def get_layer_stats(exp_name, ablation, comparison):
    return experiments[exp_name]['analysis'][ablation][comparison]['layer_stats']

# ─── 1. Bonferroni correction ────────────────────────────────────────────
print("=" * 70)
print("1. BONFERRONI CORRECTION CHECK")
print("=" * 70)
print(f"   Bonferroni threshold: 0.05 / {NUM_LAYERS} = {0.05/NUM_LAYERS:.6f}")
print()

sig_conditions = [
    ('self_recognition', 'zero', 'self_vs_third_person'),
    ('self_recognition', 'mean', 'self_vs_control'),
    ('capability_awareness', 'zero', 'self_vs_third_person'),
    ('metacognition', 'zero', 'self_vs_third_person'),
]

bonferroni_threshold = 0.05 / NUM_LAYERS  # 0.001471

all_significant = []  # collect for later use

for exp, abl, comp in sig_conditions:
    layer_stats = get_layer_stats(exp, abl, comp)
    print(f"  {exp} / {abl} / {comp}:")
    for layer in range(NUM_LAYERS):
        ls = layer_stats[str(layer)]
        if ls['significant_fdr_05']:
            survives_bonf = ls['wilcoxon_p'] <= bonferroni_threshold
            mark = "SURVIVES" if survives_bonf else "FAILS"
            print(f"    Layer {layer:2d}: p={ls['wilcoxon_p']:.6f}  dKL={ls['mean_differential_kl']:.3f}  Bonferroni: {mark}")
            all_significant.append({
                'exp': exp, 'abl': abl, 'comp': comp,
                'layer': layer, **ls
            })
    print()

# ─── 2. Cohen's d effect sizes ───────────────────────────────────────────
print("=" * 70)
print("2. COHEN'S d EFFECT SIZES")
print("=" * 70)
print("   d = mean / std (one-sample, testing against 0)")
print("   |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large")
print()

for entry in all_significant:
    values = np.array(entry['individual_values'])
    d = np.mean(values) / np.std(values, ddof=1) if np.std(values, ddof=1) > 0 else 0
    size = "LARGE" if abs(d) >= 0.8 else "MEDIUM" if abs(d) >= 0.5 else "SMALL" if abs(d) >= 0.2 else "NEGLIGIBLE"
    print(f"  {entry['exp']:25s} L{entry['layer']:2d}: d={d:+.3f} ({size})  mean={entry['mean_differential_kl']:.3f}  std={entry['std']:.3f}")

# ─── 3. Rank-biserial correlation ────────────────────────────────────────
print()
print("=" * 70)
print("3. RANK-BISERIAL CORRELATION (r_rb)")
print("=" * 70)
print("   r_rb = 1 - (2*W) / (n*(n+1)/2), where W = Wilcoxon stat")
print("   r_rb = 1.0 means all pairs in same direction")
print("   Equivalent to matched-pairs rank-biserial from Wilcoxon signed-rank")
print()

for entry in all_significant:
    n = entry['n_triplets']
    W = entry['wilcoxon_stat']
    max_W = n * (n + 1) / 2  # = 36 for n=8
    # rank-biserial: r = 1 - (2 * W_minus) / (n*(n+1)/2)
    # But W reported is W+ (sum of positive ranks), so:
    # r_rb = (4*W / (n*(n+1))) - 1  ... OR equivalently:
    # W+ = W, W- = max_W - W
    # r_rb = (W+ - W-) / max_W = (2*W - max_W) / max_W
    r_rb = (2 * W - max_W) / max_W
    print(f"  {entry['exp']:25s} L{entry['layer']:2d}: W={W:.0f}/{max_W:.0f}  r_rb={r_rb:+.3f}  p={entry['wilcoxon_p']:.6f}")

# ─── 4. Base KL frame of reference ──────────────────────────────────────
print()
print("=" * 70)
print("4. BASE KL FRAME OF REFERENCE")
print("=" * 70)
print("   What is the absolute KL when ablating layer 26 on different input types?")
print("   This contextualizes 'differential KL = 3.2 bits'")
print()

# We need to look at the raw sweep data for this
# The sweep files contain per-layer KL values for each input
exp_names_3 = ['self_recognition', 'capability_awareness', 'metacognition']

for exp_name in exp_names_3:
    print(f"  {exp_name}:")
    # Collect absolute KL values for layer 26 across all triplets
    self_kls = []
    ctrl_kls = []
    third_kls = []

    for triplet_idx in range(8):
        # Try to load sweep files for this experiment
        for input_type, kl_list in [('self', self_kls), ('control', ctrl_kls), ('third_person', third_kls)]:
            fname = SWEEP_DIR / f"{exp_name}_t{triplet_idx}_zero_{input_type}.json"
            if fname.exists():
                with open(fname) as f:
                    sweep = json.load(f)
                # Find layer 26 result
                for step in sweep.get('steps', []):
                    interventions = step.get('config', {}).get('interventions', [])
                    if interventions and interventions[0].get('layer') == 26:
                        kl = step.get('results', {}).get('kl_divergence', None)
                        if kl is not None:
                            kl_list.append(kl)
                        break

    if self_kls:
        print(f"    Layer 26 absolute KL (self):         mean={np.mean(self_kls):.3f}  std={np.std(self_kls):.3f}  n={len(self_kls)}")
    if ctrl_kls:
        print(f"    Layer 26 absolute KL (control):      mean={np.mean(ctrl_kls):.3f}  std={np.std(ctrl_kls):.3f}  n={len(ctrl_kls)}")
    if third_kls:
        print(f"    Layer 26 absolute KL (third_person): mean={np.mean(third_kls):.3f}  std={np.std(third_kls):.3f}  n={len(third_kls)}")
    if self_kls and third_kls:
        print(f"    Differential (self - third):          mean={np.mean(self_kls) - np.mean(third_kls):.3f}")

    if not self_kls:
        print(f"    (Could not find sweep files — checking file naming...)")
    print()

# If sweep files weren't found with that naming, try alternative
if not self_kls:
    print("  Checking sweep file naming convention...")
    import glob
    sweep_files = glob.glob(str(SWEEP_DIR / "*.json"))
    if sweep_files:
        print(f"  Found {len(sweep_files)} sweep files. Sample names:")
        for f in sorted(sweep_files)[:5]:
            print(f"    {Path(f).name}")

        # Try to read one and understand the structure
        with open(sweep_files[0]) as f:
            sample = json.load(f)
        print(f"\n  Sample file keys: {list(sample.keys())}")
        if 'steps' in sample:
            print(f"  Number of steps: {len(sample['steps'])}")
            if sample['steps']:
                step = sample['steps'][0]
                print(f"  Step keys: {list(step.keys())}")
                if 'results' in step:
                    print(f"  Results keys: {list(step['results'].keys())}")
    else:
        print("  No sweep JSON files found in sweep directory!")
        # Try master summary for base KL values
        print("\n  Attempting to extract from master_summary_v2.json per_experiment data...")
        with open("results/self_model_circuits_v2/master_summary_v2.json") as f:
            # Read just enough to find structure
            content = f.read(5000)
            print(f"  (Master summary structure already known — base KL not stored in analysis)")
            print(f"  NOTE: Base KL values would need to be extracted from individual sweep files.")
            print(f"  The differential KL IS the primary metric — absolute values require sweep data.")

print()
print("=" * 70)
print("5. SUMMARY FOR MD REVISION")
print("=" * 70)
print()

# Count Bonferroni survivors
bonf_survivors = sum(1 for e in all_significant if e['wilcoxon_p'] <= bonferroni_threshold)
print(f"  FDR-significant results: {len(all_significant)}")
print(f"  Bonferroni survivors:    {bonf_survivors} / {len(all_significant)}")
print()

# Universal layers under Bonferroni
print("  Universal layers (9, 26, 27) under Bonferroni:")
for layer in [9, 26, 27]:
    for exp_name in exp_names_3:
        ls = get_layer_stats(exp_name, 'zero', 'self_vs_third_person')[str(layer)]
        survives = ls['wilcoxon_p'] <= bonferroni_threshold
        print(f"    L{layer} / {exp_name:25s}: p={ls['wilcoxon_p']:.6f}  {'SURVIVES' if survives else 'FAILS'}")
    print()

# Cohen's d for Layer 26
print("  Layer 26 Cohen's d across experiments:")
for exp_name in exp_names_3:
    ls = get_layer_stats(exp_name, 'zero', 'self_vs_third_person')['26']
    values = np.array(ls['individual_values'])
    d = np.mean(values) / np.std(values, ddof=1)
    print(f"    {exp_name:25s}: d={d:.3f}")
