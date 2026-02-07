# NeuronScope

**Mechanistic interpretability through causal intervention.**

NeuronScope is a research tool for understanding what happens inside large language models — not by observing their outputs, but by intervening on their internals and measuring the consequences.

Most interpretability tools show you what a model *does*. NeuronScope lets you test *why*.

---

## What This Does

NeuronScope loads a language model onto your GPU and gives you precise control over its internal components. You can:

- **Zero-ablate** any layer, attention head, or MLP — remove a component entirely and measure what breaks
- **Patch activations** between two inputs — swap a component's state from one context into another and observe if the behavior follows
- **Sweep across all layers** — run the same intervention on every layer to build a causal map of which components matter
- **Track every result** — every experiment is reproducible, hashable, and stored in SQLite

The core principle: **if you can't intervene on a component and predict the change, you don't understand it.**

## How It Works

```
Input: "The Eiffel Tower is in"

Clean run:           → "Paris" (90.3%)
Zero-ablate Layer 0: → "Paris" (52.2%)  |  KL divergence: 12.04

The model still predicts "Paris", but its confidence dropped 38%.
Layer 0's MLP causally contributes to this prediction.
```

This is not correlation. This is causal evidence — you broke a specific component and measured the exact effect on behavior.

## Architecture

```
Backend (Python)                    Frontend (React)
├── models/     Model loading       ├── views/        Workbench, Explorer
├── hooks/      PyTorch hooks       ├── components/   Experiment builder, results
├── experiments/ Run + compare      ├── stores/       Zustand state
├── analysis/   Causal metrics      └── api/          REST client
├── store/      SQLite persistence
└── api/        FastAPI endpoints
```

**Stack:** FastAPI, PyTorch, HuggingFace Transformers, React 18, TypeScript, D3.js, SQLite

## Quick Start

**Requirements:** Python 3.10+, Node.js 18+, NVIDIA GPU with 10GB+ VRAM

```bash
# 1. Clone and enter
git clone <your-repo-url>
cd NeuronScope-For-AI

# 2. Place your model weights in LLM/
# (HuggingFace format — config.json + safetensors files)

# 3. Install Python dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .

# 4. Install frontend dependencies
cd frontend && npm install && cd ..

# 5. Start backend (terminal 1)
uvicorn neuronscope.main:app --host localhost --port 8000 --reload

# 6. Start frontend (terminal 2)
cd frontend && npm run dev

# 7. Open http://localhost:5173
```

Click **Load Model**, type an input, select a layer and intervention, and click **Run Experiment**.

## Supported Interventions

| Intervention | What It Tests |
|---|---|
| **Zero Ablation** | Is this component necessary? Remove it entirely. |
| **Activation Patching** | Does this component carry the information that causes a specific behavior? Swap it from another input. |
| **Mean Ablation** | What happens when you remove the specific signal but keep the general activation pattern? |
| **Additive Perturbation** | Can you steer behavior by adding a direction to the residual stream? |

## Design Principles

This project follows 18 principles documented in `AGENTS.md`. The core ones:

1. **Mechanisms, not impressions** — any output that cannot be causally tested is invalid
2. **Every claim must be falsifiable** — if a result can't be contradicted by a counter-run, it doesn't exist
3. **Causality beats correlation** — rankings without intervention are incomplete
4. **Understanding is measured by controllability** — if behavior can't be steered through identified mechanisms, it is not understood

## Currently Supported Models

Tested with **Gemma 3** (multimodal, 4B parameters). The hook system auto-discovers module paths, so other HuggingFace transformer models should work with minimal changes.

---

## About

AI models have helped people through hard times in ways that are difficult to explain — as thinking partners, as companions in problem-solving, as something more than tools. But most people interact with these models as black boxes. They see what comes out, never what happens inside.

NeuronScope exists because this knowledge shouldn't be locked behind lab doors. If models are going to be part of how we think and work, we should be able to look inside them, test their mechanisms, and understand what they're actually doing — not through generated explanations, but through direct evidence.

This is interpretability for everyone who's curious enough to look.

## License

See [LICENSE](LICENSE) for details.
