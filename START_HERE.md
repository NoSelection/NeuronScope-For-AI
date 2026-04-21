# Start Here

This file answers the question:

**"Which files in this repo are current, which ones are historical, and which ones are just local work artifacts?"**

## Canonical Tracked Paths

| What you want | Canonical path |
|---|---|
| Current corrected `v2` results | [results/self_model_circuits_v2](results/self_model_circuits_v2) |
| Current corrected `v3` results | [results/self_model_circuits_v3](results/self_model_circuits_v3) |
| Historical `v1` source-status note | [results/V1_PROVENANCE_NOTE.md](results/V1_PROVENANCE_NOTE.md) |
| Experiment runners | [experiments](experiments) |
| Overall result layout note | [results/README.md](results/README.md) |

If your local checkout also contains a `paper/` folder, the current manuscript files are:

- `paper/main.tex`
- `paper/main.pdf`

Those manuscript files are local workspace artifacts and may be absent on public branches. The public manuscript record lives on Zenodo:

- latest version DOI: [10.5281/zenodo.19364446](https://doi.org/10.5281/zenodo.19364446)
- current corrected `v2` DOI: [10.5281/zenodo.19672904](https://doi.org/10.5281/zenodo.19672904)

## Live Vs Historical Vs Operational

### Live / current

- [results/self_model_circuits_v2](results/self_model_circuits_v2)
- [results/self_model_circuits_v3](results/self_model_circuits_v3)
- local `paper/main.tex` and `paper/main.pdf`, if present

### Historical / archived

- [results/V1_PROVENANCE_NOTE.md](results/V1_PROVENANCE_NOTE.md)
- local `results/_pre_erratum_archives/`, if present

The `v1` note explains what historical claims from the earlier pilot are still used and where they came from.

### Operational / debugging

- local `results/_rerun_logs/`, if present
- [scripts/rerun_erratum_v2_v3.ps1](scripts/rerun_erratum_v2_v3.ps1)
- local scratch such as `.codex_backup/`, `tmp/`, or paper-side render/audit folders

These are useful for reruns and audit trails, but they are not the canonical scientific result set.

## Paper Folder Notes

Inside a local `paper/` folder, the intended split is:

- `main.tex`, `main.pdf`, `references.bib`, `source_pdfs/` = canonical manuscript bundle
- `_archive/` = older packaging artifacts and internal notes
- `_work/` = render checks, audits, and scratch outputs

If you see older named PDFs or zip files, treat them as packaging artifacts unless they were explicitly regenerated from `main.tex`.

## Results Folder Notes

Inside [results](results):

- `self_model_circuits_v2/` and `self_model_circuits_v3/` are the live corrected outputs
- `V1_PROVENANCE_NOTE.md` explains the historical `v1` ordering references still used in the manuscript
- local `_pre_erratum_archives/` and `_rerun_logs/` folders are archive/log layers, not current result bundles

## Recommended Reading Order

1. [START_HERE.md](START_HERE.md)
2. [results/V1_PROVENANCE_NOTE.md](results/V1_PROVENANCE_NOTE.md)
3. local `paper/main.tex` if present
4. [results/self_model_circuits_v2/master_summary_v2.json](results/self_model_circuits_v2/master_summary_v2.json)
5. [results/self_model_circuits_v3/master_summary_v3.json](results/self_model_circuits_v3/master_summary_v3.json)
