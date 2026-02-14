# Experiment Scripts

All scripts are run from the **project root** (`NeuronScope-For-AI/`), not from this directory.

## Experiment Runners

| Script | What it does | Runtime | Output |
|--------|-------------|---------|--------|
| `run_v1_self_model.py` | v1 pilot: 4 prompt pairs, MLP ablation (zero + mean), 34 layers | ~1.5 hours | `results/self_model_circuits/` |
| `run_v2_self_model.py` | v2 replication: 32 prompt triplets (self/control/third-person), MLP ablation, stats | ~14 hours | `results/self_model_circuits_v2/` |
| `run_v3_head_sweep.py` | v3 attention heads: per-head ablation on layers 9/26/27, 1,728 runs | ~3.5 hours | `results/self_model_circuits_v3/` |

### How to run

```bash
# From project root:
python experiments/run_v3_head_sweep.py              # full run
python experiments/run_v3_head_sweep.py --dry-run    # preview plan, no GPU
python experiments/run_v3_head_sweep.py --analyze    # re-analyze existing data
```

All scripts checkpoint after every sweep, so they can be interrupted and resumed safely.

## Analysis Scripts

Post-hoc analysis on v2 sweep data. Also run from project root.

| Script | What it does |
|--------|-------------|
| `analysis/compute_base_kl.py` | Extract absolute KL values at Layer 26 for frame of reference |
| `analysis/compute_extra_stats.py` | Bonferroni correction, Cohen's d, rank-biserial correlation |
| `analysis/compute_pronoun_test.py` | Test whether Layer 26 effect is a pronoun confound (I vs It) |

```bash
python experiments/analysis/compute_pronoun_test.py
```
