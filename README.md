# NeuronScope

Mechanistic interpretability through causal intervention.

This repository currently serves **three roles**, but only two of them are guaranteed to be present on public branches:

| What it is | Where to start | Canonical outputs |
|---|---|---|
| Core NeuronScope Python package and frontend workbench | [README.md](README.md) and [START_HERE.md](START_HERE.md) | `neuronscope/`, `frontend/` |
| Experiment runners and corrected JSON result bundles for the Gemma 3 paper | [experiments/README.md](experiments/README.md) and [results/README.md](results/README.md) | `experiments/`, `results/self_model_circuits_v2/`, `results/self_model_circuits_v3/` |
| Local manuscript bundle, packaging artifacts, and audit scratch | local `paper/` folder if present in your checkout | `paper/main.tex`, `paper/main.pdf` |

If you only read one file first, read [START_HERE.md](START_HERE.md).

## If You Are Here For The Paper

Treat these as the source of truth:

- Public manuscript record on Zenodo (latest version DOI): [10.5281/zenodo.19364446](https://doi.org/10.5281/zenodo.19364446)
- Current corrected manuscript version on Zenodo (`v2`): [10.5281/zenodo.19672904](https://doi.org/10.5281/zenodo.19672904)
- Current corrected `v2` results: [results/self_model_circuits_v2](results/self_model_circuits_v2)
- Current corrected `v3` results: [results/self_model_circuits_v3](results/self_model_circuits_v3)
- Historical `v1` provenance note: [results/V1_PROVENANCE_NOTE.md](results/V1_PROVENANCE_NOTE.md)
- Local manuscript source and PDF, if present in your checkout: `paper/main.tex`, `paper/main.pdf`

Important distinction:

- The **live corrected results** are in `results/self_model_circuits_v2/` and `results/self_model_circuits_v3/`.
- Historical pre-erratum bundles and rerun logs may exist locally under `results/_pre_erratum_archives/` and `results/_rerun_logs/`, but they are not the current canonical result set.
- Local audit notes such as `issues.md` are workspace-only and should not be treated as public canonical documentation.

## Repo Layout

```text
NeuronScope-For-AI/
|- neuronscope/                 Core Python package
|- frontend/                    React workbench
|- experiments/                 Experiment runners and analysis helpers
|- results/                     Canonical corrected JSON results plus provenance note
|- scripts/                     Helper scripts for reruns / maintenance
|- tests/                       Automated tests
|- START_HERE.md                Canonical orientation doc
|- README.md                    Top-level repo map
```

If your local checkout also contains a `paper/` folder, that folder is the manuscript workspace. The public repository intentionally tracks the code and corrected result bundles, while the public manuscript PDF lives on Zenodo.

## Two Different Ways To Use This Repo

### 1. App / Workbench Mode

Use this mode if you want the NeuronScope backend/frontend experience.

Requirements:

- Python 3.10+
- Node.js 18+
- NVIDIA GPU with enough VRAM for your target model

```bash
git clone https://github.com/NoSelection/NeuronScope-For-AI.git
cd NeuronScope-For-AI

# Place model weights in LLM/
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .

cd frontend && npm install && cd ..

uvicorn neuronscope.main:app --host localhost --port 8000 --reload
cd frontend && npm run dev
```

Open `http://localhost:5173`.

### 2. Experiment / Result Reproduction Mode

Use this mode if you want to reproduce the corrected Gemma 3 paper analyses.

Requirements:

- Python 3.10+
- Local HuggingFace-format model snapshot in [LLM](LLM)
- CUDA-capable NVIDIA GPU

From the repo root:

```bash
python -m experiments.run_v2_self_model
python -m experiments.run_v3_head_sweep
```

For the full Windows rerun workflow, use:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/rerun_erratum_v2_v3.ps1
```

More detail lives in [experiments/README.md](experiments/README.md).

## Important Storage Distinction

This repo has **two different result-storage stories**:

- The **interactive app** is designed around persistent app state and database-backed workflows.
- The **paper experiment scripts** write checkpoint JSONs and summary JSONs into [results](results).

If you are auditing the paper, trust the JSON artifacts under `results/`, not the generic app description.

## Supported Interventions

| Intervention | What It Tests |
|---|---|
| Zero Ablation | Is this component necessary? Remove it entirely. |
| Activation Patching | Does this component carry the information that causes a specific behavior? Swap it from another input. |
| Mean Ablation | Replace the component with a paired source activation rather than zeroing it. |
| Additive Perturbation | Can you steer behavior by adding a direction to the residual stream? |

## Model Scope

The corrected result bundles in this repo are for **Gemma 3 4B** specifically.

The core hook infrastructure may work with other HuggingFace-compatible models, but the current manuscript/result story should be read as a **Gemma 3 paper repo first**. The public manuscript PDF is hosted on Zenodo rather than committed to this repository.

## License

See [LICENSE](LICENSE).
