# Experiment Scripts

All commands are run from the **project root** (`NeuronScope-For-AI/`), not from this directory.

This folder is the experiment layer of the repo. If you are looking for:

- current result bundles, see [results/README.md](../results/README.md)
- historical `v1` status, see [results/V1_PROVENANCE_NOTE.md](../results/V1_PROVENANCE_NOTE.md)
- overall repo orientation, see [START_HERE.md](../START_HERE.md)
- a local manuscript bundle, use the local `paper/` folder if it exists in your checkout

## Canonical Current Outputs

The live corrected result trees are:

- [results/self_model_circuits_v2](../results/self_model_circuits_v2)
- [results/self_model_circuits_v3](../results/self_model_circuits_v3)

Historical pre-erratum bundles and rerun logs may exist locally under `results/_pre_erratum_archives/` and `results/_rerun_logs/`, but they are not the current canonical outputs.

## Experiment Runners

| Script | What it does | Runtime | Output |
|--------|-------------|---------|--------|
| `run_v1_self_model.py` | v1 pilot: 4 prompt pairs, MLP ablation (zero + mean), 34 layers | ~1.5 hours | historical only |
| `run_v2_self_model.py` | v2 corrected triplet study: 32 prompt triplets, MLP ablation, stats | ~14 hours | `results/self_model_circuits_v2/` |
| `run_v3_head_sweep.py` | v3 corrected head sweep: attention-head ablation on layers 9/26/27 | ~3.5 hours | `results/self_model_circuits_v3/` |

## How To Run

Recommended invocation style:

```bash
python -m experiments.run_v2_self_model
python -m experiments.run_v2_self_model --dry-run
python -m experiments.run_v2_self_model --analyze

python -m experiments.run_v3_head_sweep
python -m experiments.run_v3_head_sweep --dry-run
python -m experiments.run_v3_head_sweep --analyze
```

All scripts checkpoint after every sweep, so they can be interrupted and resumed safely.

## Full Erratum Rerun Helper

For the full Windows rerun workflow used for the corrected artifact rebuild:

- [scripts/rerun_erratum_v2_v3.ps1](../scripts/rerun_erratum_v2_v3.ps1)

That helper:

- archives the current live `v2`/`v3` result folders
- writes a manifest and status log
- runs `v2` and then `v3`
- stores logs under local `results/_rerun_logs/`

## Analysis Scripts

Post-hoc analysis helpers on `v2` sweep data. Also run from project root.

| Script | What it does |
|--------|-------------|
| `analysis/compute_base_kl.py` | Extract absolute KL values at Layer 26 for frame of reference |
| `analysis/compute_extra_stats.py` | Bonferroni correction, Cohen's d, rank-biserial correlation |
| `analysis/compute_pronoun_test.py` | Test whether the Layer 26 effect is a pronoun confound |

```bash
python experiments/analysis/compute_pronoun_test.py
```

## Important Clarification

The current corrected `v2` rerun rebuilds the JSON analysis tree and master summary, but it does **not** regenerate the older markdown summary, notebooks, and charts that may exist in local historical archives.
