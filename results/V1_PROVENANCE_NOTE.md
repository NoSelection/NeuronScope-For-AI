# V1 Provenance Note

This note exists to make the manuscript's limited `v1` references easier to audit.

## What `v1` Was

`v1` was an earlier internal pilot that used prompt pairs rather than the corrected `v2` triplet design. It is **not** a like-for-like replication target for `v2`, and the manuscript now treats it as historical context only.

## What The Current Manuscript Still Uses From `v1`

The current manuscript uses only the **historical ordering** of the earlier zero-ablation pilot:

1. training knowledge
2. self-recognition
3. capability awareness
4. metacognition

This ordering is used only to motivate the `v1 -> v2` inversion discussion. It is not used as a current statistical result, and it should not be read as inferential evidence on the same footing as corrected `v2`/`v3`.

## Where That Historical Ordering Came From

The ordering came from the archived zero-ablation pilot retained in project history:

- `experiments/run_v1_self_model.py`
- the removed historical artifact `results/self_model_circuits/V1_master_summary.json` in git history
- local historical reports preserved in pre-erratum archive material, if present in the checkout

Because the live corrected repo surface is centered on `v2` and `v3`, the `v1` material is intentionally treated as archival provenance rather than a canonical current result tree.

## How To Read `v1` Relative To `v2`

The `v1 -> v2` comparison is heuristic rather than like-for-like:

- `v1` is summarized descriptively by historical zero-ablation peak ordering
- corrected `v2` is summarized by FDR-screened layer counts under the tighter triplet design

So the manuscript's `v1` discussion should be read as a cautionary historical contrast, not as a direct statistical replication analysis.
