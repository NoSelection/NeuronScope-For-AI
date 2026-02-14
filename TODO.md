# NeuronScope TODO

## Self-Referential Processing Research

Investigating whether LLM self-referential outputs are grounded in internal representations or surface-level pattern matching.

### v1 — Pilot Study (DONE)
- [x] 4 experiments, 2 ablation types, 16 sweeps, 544 runs
- [x] Identified candidate layers 5, 6, 8
- [x] Results: `results/self_model_circuits/`
- [x] Script: `experiments/run_v1_self_model.py`

### v2 — Extended Replication (DONE)
- [x] 32 prompt triplets (self / control / third-person)
- [x] 192 sweeps, 6,528 runs (~14 hours)
- [x] Wilcoxon tests + bootstrap 95% CIs
- [x] Found universal layers: 9, 26, 27
- [x] Pronoun confound analysis — Layer 26 MLP effect is "I" vs "It", not AI-specific
- [x] Cohen's d, Bonferroni correction, rank-biserial correlation
- [x] Results: `results/self_model_circuits_v2/`
- [x] Script: `experiments/run_v2_self_model.py`

### v3 — Attention Head Ablation (READY TO RUN)
- [x] Per-head ablation feature implemented (backend + frontend + UI)
- [x] 8 attention heads per layer, 3 target layers (9, 26, 27)
- [x] 1,728 individual runs (~3.5 hours)
- [ ] Run overnight: `python experiments/run_v3_head_sweep.py`
- [ ] Analyze: do any heads show AI-content-specific effects beyond pronoun confound?
- [ ] Script: `experiments/run_v3_head_sweep.py`

### NeuronScope Tool
- [x] Backend: model loading, hooks, experiments, analysis, SQLite persistence
- [x] Frontend: ExperimentWorkbench, ActivationExplorer, D3 charts
- [x] Layer sweeps + head sweeps with insights
- [x] PDF report generation (single experiment + sweep)
- [x] Educational tooltips and guided walkthrough
- [x] Non-technical-friendly UI with InfoTip system
- [x] OOM handling and error recovery

### Future Work
- [ ] Activation patching between self/control inputs to trace information flow
- [ ] Test on different model sizes (Gemma 2B, 9B) to see if circuit location scales
- [ ] Multi-layer interaction analysis (do layers 9 and 26 interact?)
- [ ] Student survey on AI self-report trust (pre/post interpretability demo)
