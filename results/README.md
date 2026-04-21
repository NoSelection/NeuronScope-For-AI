# Results Layout

This folder contains **three different categories of artifacts**.

## 1. Live corrected outputs

These are the current canonical results for the paper:

- [self_model_circuits_v2](self_model_circuits_v2)
- [self_model_circuits_v3](self_model_circuits_v3)

Each live result tree contains:

- `sweeps/`: per-sweep checkpoint JSONs
- `analysis/`: per-domain analysis summaries
- `master_summary_*.json`: top-level aggregate summary

## 2. Historical context

- [V1_PROVENANCE_NOTE.md](V1_PROVENANCE_NOTE.md)

That note explains the limited `v1` historical claims still referenced in the manuscript and how they relate to the current corrected result trees.

If your local checkout includes `results/_pre_erratum_archives/`, treat that folder as a preserved historical layer, not as a second set of canonical current results.

Important caveat:

- Later archive folders with multiple April 19 timestamps are operational snapshots from rerun attempts.
- They are useful for audit history, but they should not be mistaken for separate finalized scientific versions.

## 3. Rerun logs and manifests

If your local checkout includes `results/_rerun_logs/`, that folder is the operational log layer.

Most important successful rerun log, if present locally:

- `results/_rerun_logs/20260419_234141`

That directory contains:

- `manifest.txt`: environment and git-state snapshot
- `status.txt`: phase-level run status
- `v2.log` / `v3.log`: stdout logs
- `v2.stderr.log` / `v3.stderr.log`: stderr logs

## Naming Conventions

### `v2`

Example:

- `self_recognition__0__self__zero.json`

Meaning:

- experiment domain = `self_recognition`
- triplet index = `0`
- input type = `self`
- ablation type = `zero`

### `v3`

Example:

- `metacognition__0__self__L26.json`

Meaning:

- experiment domain = `metacognition`
- triplet index = `0`
- input type = `self`
- target layer = `26`

## Important Clarification

The live corrected `v2` rerun rebuilds the JSON analysis tree, but it does **not** regenerate the older markdown summary, notebook, and chart bundle that may exist in local historical archives. If you need those older charts/notebooks, treat them as archived historical artifacts rather than current canonical outputs.
