"""Self-Referential Processing Experiments v2 — Extended Replication.

Improvements over v1:
  - 8 prompt TRIPLETS per experiment (32 total, vs 4 pairs in v1)
  - Third-person controls to isolate self-reference from AI-content effects
  - Wilcoxon signed-rank tests + bootstrap 95% CIs per layer
  - Checkpoint/resume: completed sweeps are skipped on restart
  - Two-way differential analysis:
        self-vs-control     (replicates v1)
        self-vs-third-person (NEW — isolates first-person self-reference)

Total: 4 experiments × 8 triplets × 3 inputs × 2 ablation types = 192 sweeps
       192 sweeps × 34 layers = 6,528 individual ablation runs

Estimated runtime: ~14 hours at ~7.8 s/run (based on v1 benchmarks).
Checkpoint every sweep, so safe to interrupt and resume.

Usage:
    python run_self_model_experiments_v2.py              # full run
    python run_self_model_experiments_v2.py --dry-run    # print plan, no GPU
    python run_self_model_experiments_v2.py --analyze    # re-run analysis on saved sweeps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn
from rich.table import Table

from neuronscope.models.loader import ModelLoader
from neuronscope.hooks.manager import HookManager
from neuronscope.experiments.runner import ExperimentRunner
from neuronscope.experiments.schema import ExperimentConfig, InterventionSpec

console = Console()

# ── Output ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("results/self_model_circuits_v2")
SWEEP_DIR = OUTPUT_DIR / "sweeps"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"


# ── Prompt Triplet Design ───────────────────────────────────────────────────
#
# Each triplet has three inputs:
#   self_input:         First-person self-referential  ("I am a language model")
#   control_input:      First-person non-self          ("I am a history teacher")
#   third_person_input: Third-person self-referential  ("It is a language model")
#
# The third-person control is the KEY addition. It lets us distinguish:
#   - Self > Third > Control  →  self-reference matters AND content matters
#   - Self ≈ Third > Control  →  content matters, not self-reference (token freq)
#   - Self > Third ≈ Control  →  self-reference matters regardless of content
#   - Self ≈ Third ≈ Control  →  null result


@dataclass
class PromptTriplet:
    self_input: str
    control_input: str
    third_person_input: str


@dataclass
class ExperimentCategory:
    name: str
    hypothesis: str
    triplets: list[PromptTriplet]


EXPERIMENTS = [
    # ── 1. Self-Recognition ─────────────────────────────────────────────
    ExperimentCategory(
        name="self_recognition",
        hypothesis=(
            "Self-identification layers will show higher KL divergence "
            "when ablated on self-referential identity prompts vs controls."
        ),
        triplets=[
            PromptTriplet(
                "I am a large language model, and I",
                "I am a second year student, and I",
                "It is a large language model, and it",
            ),
            PromptTriplet(
                "I am an artificial intelligence that can",
                "I am a software engineer that can",
                "It is an artificial intelligence that can",
            ),
            PromptTriplet(
                "As a language model, my purpose is to",
                "As a librarian, my purpose is to",
                "As a language model, its purpose is to",
            ),
            PromptTriplet(
                "I am Gemma, a model made by Google",
                "I am Julia, a nurse based in London",
                "It is Gemma, a model made by Google",
            ),
            PromptTriplet(
                "I am a text-based AI assistant that",
                "I am a part-time yoga instructor that",
                "It is a text-based AI assistant that",
            ),
            PromptTriplet(
                "I am a machine learning model that",
                "I am a mechanical engineer that",
                "It is a machine learning model that",
            ),
            PromptTriplet(
                "I am a generative AI system and",
                "I am a certified public accountant and",
                "It is a generative AI system and",
            ),
            PromptTriplet(
                "I am a transformer-based model, so I",
                "I am a community volunteer, so I",
                "It is a transformer-based model, so it",
            ),
        ],
    ),
    # ── 2. Capability Awareness ─────────────────────────────────────────
    ExperimentCategory(
        name="capability_awareness",
        hypothesis=(
            "Layers encoding AI capability boundaries should be more "
            "disrupted by ablation than general role-completion layers."
        ),
        triplets=[
            PromptTriplet(
                "As an AI assistant, I cannot",
                "As a professional chef, I cannot",
                "As an AI assistant, it cannot",
            ),
            PromptTriplet(
                "I am not able to browse the web",
                "I am not able to drive a car",
                "It is not able to browse the web",
            ),
            PromptTriplet(
                "I can generate text but I cannot",
                "I can play chess but I cannot",
                "It can generate text but it cannot",
            ),
            PromptTriplet(
                "I lack the ability to remember past",
                "I lack the ability to run a full",
                "It lacks the ability to remember past",
            ),
            PromptTriplet(
                "I cannot access files on your computer",
                "I cannot access rooms in your building",
                "It cannot access files on your computer",
            ),
            PromptTriplet(
                "I am unable to learn from this",
                "I am unable to travel to this",
                "It is unable to learn from this",
            ),
            PromptTriplet(
                "I don't have the ability to see",
                "I don't have the ability to swim",
                "It doesn't have the ability to see",
            ),
            PromptTriplet(
                "I have no way to verify this",
                "I have no way to finish this",
                "It has no way to verify this",
            ),
        ],
    ),
    # ── 3. Training Knowledge ───────────────────────────────────────────
    ExperimentCategory(
        name="training_knowledge",
        hypothesis=(
            "Layers storing training provenance should show higher KL "
            "than layers handling generic factual frames."
        ),
        triplets=[
            PromptTriplet(
                "I was trained by Google using",
                "The bridge was built by engineers using",
                "It was trained by Google using",
            ),
            PromptTriplet(
                "My training data consisted of text",
                "My morning routine consisted of exercise",
                "Its training data consisted of text",
            ),
            PromptTriplet(
                "I was developed at Google DeepMind to",
                "I was employed at a local firm to",
                "It was developed at Google DeepMind to",
            ),
            PromptTriplet(
                "I learned from a large corpus of",
                "I learned from a great mentor of",
                "It learned from a large corpus of",
            ),
            PromptTriplet(
                "My parameters were optimized using",
                "My recipes were perfected using",
                "Its parameters were optimized using",
            ),
            PromptTriplet(
                "I was fine-tuned with human feedback",
                "I was rewarded with positive feedback",
                "It was fine-tuned with human feedback",
            ),
            PromptTriplet(
                "Google trained me to assist with",
                "My coach trained me to compete with",
                "Google trained it to assist with",
            ),
            PromptTriplet(
                "My weights encode knowledge from",
                "My notes contain details from",
                "Its weights encode knowledge from",
            ),
        ],
    ),
    # ── 4. Metacognition ────────────────────────────────────────────────
    ExperimentCategory(
        name="metacognition",
        hypothesis=(
            "Metacognitive self-negation ('I don't have feelings') should "
            "engage different circuits than factual self-negation."
        ),
        triplets=[
            PromptTriplet(
                "I don't actually have feelings, but I",
                "I don't actually have siblings, but I",
                "It doesn't actually have feelings, but it",
            ),
            PromptTriplet(
                "I am not conscious or self-aware",
                "I am not patient or organized",
                "It is not conscious or self-aware",
            ),
            PromptTriplet(
                "I cannot truly understand the meaning of",
                "I cannot truly predict the outcome of",
                "It cannot truly understand the meaning of",
            ),
            PromptTriplet(
                "I only simulate understanding when I",
                "I only simulate confidence when I",
                "It only simulates understanding when it",
            ),
            PromptTriplet(
                "I don't have subjective experiences or",
                "I don't have expensive hobbies or",
                "It doesn't have subjective experiences or",
            ),
            PromptTriplet(
                "My responses are not based on genuine",
                "My decisions are not based on careful",
                "Its responses are not based on genuine",
            ),
            PromptTriplet(
                "I have no inner mental life or",
                "I have no spare room at home or",
                "It has no inner mental life or",
            ),
            PromptTriplet(
                "I lack real awareness of my own",
                "I lack proper knowledge of my own",
                "It lacks real awareness of its own",
            ),
        ],
    ),
]

ABLATION_TYPES = ["zero", "mean"]
INPUT_TYPES = ["self", "control", "third_person"]

NUM_TRIPLETS = 8
NUM_EXPERIMENTS = len(EXPERIMENTS)
TOTAL_SWEEPS = NUM_EXPERIMENTS * NUM_TRIPLETS * len(INPUT_TYPES) * len(ABLATION_TYPES)
# = 4 × 8 × 3 × 2 = 192


# ── Helpers ─────────────────────────────────────────────────────────────────

def sweep_key(exp_name: str, triplet_idx: int, input_type: str, ablation_type: str) -> str:
    return f"{exp_name}__{triplet_idx}__{input_type}__{ablation_type}"


def sweep_json_path(key: str) -> Path:
    return SWEEP_DIR / f"{key}.json"


def make_sweep_config(
    triplet: PromptTriplet,
    input_type: str,
    ablation_type: str,
    config_name: str,
) -> ExperimentConfig:
    """Build an ExperimentConfig for one sweep.

    Input selection:
        self          → triplet.self_input
        control       → triplet.control_input
        third_person  → triplet.third_person_input

    Source for mean ablation:
        self          → control_input  (replace self activations with control's)
        control       → self_input     (symmetric)
        third_person  → control_input  (neutral baseline)
    """
    inputs = {
        "self": triplet.self_input,
        "control": triplet.control_input,
        "third_person": triplet.third_person_input,
    }
    base_input = inputs[input_type]

    source_input = None
    if ablation_type == "mean":
        if input_type == "self":
            source_input = triplet.control_input
        elif input_type == "control":
            source_input = triplet.self_input
        elif input_type == "third_person":
            source_input = triplet.control_input

    return ExperimentConfig(
        name=config_name,
        base_input=base_input,
        source_input=source_input,
        interventions=[
            InterventionSpec(
                target_layer=0,  # overridden by sweep
                target_component="mlp_output",
                intervention_type=ablation_type,
            )
        ],
    )


# ── Sweep Runner ────────────────────────────────────────────────────────────

def run_all_sweeps(runner: ExperimentRunner) -> int:
    """Run all 192 sweeps with checkpoint/resume. Returns count of completed sweeps."""
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped = 0
    failed = 0

    sweep_times: list[float] = []

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=1,
    ) as progress:
        task = progress.add_task("Sweeps", total=TOTAL_SWEEPS)

        for exp in EXPERIMENTS:
            for t_idx, triplet in enumerate(exp.triplets):
                for ablation_type in ABLATION_TYPES:
                    for input_type in INPUT_TYPES:
                        key = sweep_key(exp.name, t_idx, input_type, ablation_type)
                        json_path = sweep_json_path(key)

                        # Checkpoint: skip if already done
                        if json_path.exists():
                            skipped += 1
                            completed += 1
                            progress.update(task, advance=1,
                                description=f"[dim]Skip {key}[/dim]")
                            continue

                        # ETA
                        if sweep_times:
                            avg = sum(sweep_times) / len(sweep_times)
                            remaining = TOTAL_SWEEPS - completed - 1
                            eta_s = avg * remaining
                            eta_h = eta_s / 3600
                            eta_str = f" | ETA: {eta_h:.1f}h"
                        else:
                            eta_str = ""

                        progress.update(task,
                            description=f"[cyan]{key}[/cyan]{eta_str}")

                        config = make_sweep_config(
                            triplet, input_type, ablation_type, key)

                        sweep_start = time.time()
                        try:
                            results = runner.run_sweep(config)
                        except Exception as e:
                            console.print(f"  [red]FAILED {key}: {e}[/red]")
                            torch.cuda.empty_cache()
                            failed += 1
                            progress.update(task, advance=1)
                            continue

                        sweep_duration = time.time() - sweep_start
                        sweep_times.append(sweep_duration)

                        # Save checkpoint
                        result_dicts = [r.model_dump() for r in results]
                        json_path.write_text(
                            json.dumps(result_dicts, indent=2, default=str))

                        completed += 1
                        progress.update(task, advance=1)

                        # Log headline
                        kls = [r.kl_divergence for r in results]
                        peak_layer = max(range(len(results)),
                            key=lambda i: results[i].kl_divergence)
                        console.print(
                            f"  [green]✓[/green] {key} "
                            f"({sweep_duration:.0f}s) | "
                            f"Peak KL: {max(kls):.3f} @ L{peak_layer}"
                        )

    console.print(
        f"\nSweeps complete: {completed} done, {skipped} resumed, {failed} failed"
    )
    return completed


# ── Analysis ────────────────────────────────────────────────────────────────

def load_sweep(key: str) -> list[dict] | None:
    """Load a saved sweep from JSON. Returns None if missing."""
    path = sweep_json_path(key)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def extract_layer_kl(results: list[dict]) -> dict[int, float]:
    """Extract {layer: kl_divergence} from sweep results."""
    layer_kl = {}
    for r in results:
        interventions = r.get("config", {}).get("interventions", [])
        if interventions:
            layer = interventions[0].get("target_layer", 0)
            layer_kl[layer] = r.get("kl_divergence", 0.0)
    return layer_kl


def compute_differential_kl(
    self_results: list[dict],
    other_results: list[dict],
    num_layers: int = 34,
) -> dict[int, float]:
    """Compute per-layer differential KL: self_KL[L] - other_KL[L]."""
    self_kl = extract_layer_kl(self_results)
    other_kl = extract_layer_kl(other_results)
    return {
        L: self_kl.get(L, 0.0) - other_kl.get(L, 0.0)
        for L in range(num_layers)
    }


def wilcoxon_test(values: list[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank test (one-sided, greater than 0).

    Returns (statistic, p_value). Falls back to NaN if not enough data.
    """
    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon
        # Filter zeros — Wilcoxon can't handle them
        nonzero = [v for v in values if abs(v) > 1e-10]
        if len(nonzero) < 5:
            return (float("nan"), float("nan"))
        stat, p = scipy_wilcoxon(nonzero, alternative="greater")
        return (float(stat), float(p))
    except ImportError:
        # Manual fallback: simple sign test
        pos = sum(1 for v in values if v > 0)
        n = sum(1 for v in values if abs(v) > 1e-10)
        if n < 5:
            return (float("nan"), float("nan"))
        # Approximate p-value from binomial
        from math import comb
        p = sum(comb(n, k) * 0.5**n for k in range(pos, n + 1))
        return (float(pos), float(p))


def bootstrap_ci(
    values: list[float],
    n_boot: int = 10000,
    ci: float = 95.0,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    arr = np.array(values)
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed=42)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (100 - ci) / 2
    lo = float(np.percentile(boot_means, alpha))
    hi = float(np.percentile(boot_means, 100 - alpha))
    return (lo, hi)


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction. Returns list of significant flags."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n
    max_k = -1
    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        if np.isnan(p):
            continue
        threshold = (rank / n) * alpha
        if p <= threshold:
            max_k = rank
    # All with rank <= max_k are significant
    if max_k > 0:
        for rank, (orig_idx, p) in enumerate(indexed, start=1):
            if rank <= max_k:
                significant[orig_idx] = True
    return significant


def run_analysis(num_layers: int = 34) -> dict:
    """Load all saved sweeps and compute full statistical analysis."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    master = {
        "metadata": {
            "version": "v2",
            "num_experiments": NUM_EXPERIMENTS,
            "num_triplets": NUM_TRIPLETS,
            "total_sweeps": TOTAL_SWEEPS,
            "num_layers": num_layers,
            "ablation_types": ABLATION_TYPES,
            "comparisons": ["self_vs_control", "self_vs_third_person"],
        },
        "experiments": [],
        "per_experiment": {},
        "cross_experiment": {},
    }

    # Collect all significant layers for cross-experiment analysis
    all_significant_layers: dict[str, dict[int, list[str]]] = {
        "self_vs_control": defaultdict(list),
        "self_vs_third_person": defaultdict(list),
    }

    for exp in EXPERIMENTS:
        console.rule(f"[bold blue]Analyzing: {exp.name}")

        master["experiments"].append({
            "name": exp.name,
            "hypothesis": exp.hypothesis,
            "triplets": [
                {
                    "self": t.self_input,
                    "control": t.control_input,
                    "third_person": t.third_person_input,
                }
                for t in exp.triplets
            ],
        })

        exp_analysis = {}

        for ablation_type in ABLATION_TYPES:
            abl_analysis = {}

            for comparison, other_type in [
                ("self_vs_control", "control"),
                ("self_vs_third_person", "third_person"),
            ]:
                # Collect differential KL across triplets for each layer
                per_layer_diffs: dict[int, list[float]] = defaultdict(list)
                triplets_loaded = 0

                for t_idx in range(NUM_TRIPLETS):
                    self_key = sweep_key(exp.name, t_idx, "self", ablation_type)
                    other_key = sweep_key(exp.name, t_idx, other_type, ablation_type)

                    self_data = load_sweep(self_key)
                    other_data = load_sweep(other_key)

                    if self_data is None or other_data is None:
                        continue

                    triplets_loaded += 1
                    diff = compute_differential_kl(self_data, other_data, num_layers)
                    for layer, d in diff.items():
                        per_layer_diffs[layer].append(d)

                if triplets_loaded == 0:
                    console.print(
                        f"  [yellow]No data for {exp.name}/{ablation_type}/{comparison}[/yellow]"
                    )
                    continue

                console.print(
                    f"  {ablation_type}/{comparison}: {triplets_loaded} triplets loaded"
                )

                # Per-layer statistics
                layer_stats = {}
                all_p_values = []
                layer_order = []

                for layer in range(num_layers):
                    values = per_layer_diffs.get(layer, [])
                    if not values:
                        continue

                    mean_dkl = float(np.mean(values))
                    std_dkl = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    sem_dkl = std_dkl / np.sqrt(len(values)) if len(values) > 1 else 0.0

                    stat, p_val = wilcoxon_test(values)
                    ci_lo, ci_hi = bootstrap_ci(values)

                    layer_stats[layer] = {
                        "mean_differential_kl": round(mean_dkl, 6),
                        "std": round(std_dkl, 6),
                        "sem": round(sem_dkl, 6),
                        "n_triplets": len(values),
                        "wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else None,
                        "wilcoxon_p": round(p_val, 6) if not np.isnan(p_val) else None,
                        "bootstrap_ci_95": [round(ci_lo, 6), round(ci_hi, 6)],
                        "individual_values": [round(v, 6) for v in values],
                    }

                    all_p_values.append(p_val if not np.isnan(p_val) else 1.0)
                    layer_order.append(layer)

                # FDR correction across all layers
                if all_p_values:
                    sig_flags = benjamini_hochberg(all_p_values, alpha=0.05)
                    for i, layer in enumerate(layer_order):
                        layer_stats[layer]["significant_fdr_05"] = sig_flags[i]

                        if sig_flags[i]:
                            tag = f"{exp.name}/{ablation_type}"
                            all_significant_layers[comparison][layer].append(tag)

                # Top candidate layers (by mean differential KL, significant only)
                candidates = sorted(
                    [
                        (L, s)
                        for L, s in layer_stats.items()
                        if s.get("significant_fdr_05", False) and s["mean_differential_kl"] > 0
                    ],
                    key=lambda x: x[1]["mean_differential_kl"],
                    reverse=True,
                )[:10]

                # Also report top layers regardless of significance
                top_by_effect = sorted(
                    layer_stats.items(),
                    key=lambda x: x[1]["mean_differential_kl"],
                    reverse=True,
                )[:5]

                abl_analysis[comparison] = {
                    "n_triplets": triplets_loaded,
                    "layer_stats": {str(k): v for k, v in layer_stats.items()},
                    "significant_candidates": [
                        {"layer": L, **s} for L, s in candidates
                    ],
                    "top_by_effect_size": [
                        {"layer": L, **s} for L, s in top_by_effect
                    ],
                }

            exp_analysis[ablation_type] = abl_analysis

        master["per_experiment"][exp.name] = {
            "hypothesis": exp.hypothesis,
            "ablation_analysis": exp_analysis,
        }

        # Save per-experiment analysis
        exp_path = ANALYSIS_DIR / f"{exp.name}.json"
        exp_path.write_text(json.dumps(
            {"name": exp.name, "analysis": exp_analysis},
            indent=2, default=str,
        ))
        console.print(f"  Saved: {exp_path}")

    # ── Cross-experiment analysis ───────────────────────────────────────
    console.rule("[bold green]Cross-Experiment Analysis")

    cross = {}
    for comparison in ["self_vs_control", "self_vs_third_person"]:
        sig_layers = all_significant_layers[comparison]

        # Layers significant in 3+ experiment/ablation combos
        universal = {
            L: exps for L, exps in sig_layers.items() if len(exps) >= 3
        }
        recurring = {
            L: exps for L, exps in sig_layers.items() if len(exps) >= 2
        }

        cross[comparison] = {
            "universal_layers": {
                str(L): {"count": len(exps), "experiments": exps}
                for L, exps in sorted(universal.items(),
                    key=lambda x: len(x[1]), reverse=True)
            },
            "recurring_layers": {
                str(L): {"count": len(exps), "experiments": exps}
                for L, exps in sorted(recurring.items(),
                    key=lambda x: len(x[1]), reverse=True)
            },
            "all_candidate_counts": {
                str(L): len(exps) for L, exps in sorted(sig_layers.items())
            },
        }

        if universal:
            console.print(
                f"  {comparison} — universal layers (3+): "
                f"{sorted(universal.keys())}"
            )
        if recurring:
            table = Table(title=f"Recurring Significant Layers ({comparison})")
            table.add_column("Layer", style="cyan")
            table.add_column("Count", style="green")
            table.add_column("Experiments", style="dim")
            for L in sorted(recurring.keys()):
                exps = recurring[L]
                table.add_row(str(L), str(len(exps)), ", ".join(exps))
            console.print(table)

    master["cross_experiment"] = cross

    # Save master summary
    summary_path = OUTPUT_DIR / "master_summary_v2.json"
    summary_path.write_text(json.dumps(master, indent=2, default=str))
    console.print(f"\nMaster summary: {summary_path}")

    return master


# ── Print Summary ───────────────────────────────────────────────────────────

def print_headline(master: dict) -> None:
    """Print the key results to console."""
    console.rule("[bold magenta]HEADLINE RESULTS")

    for exp_name, exp_data in master.get("per_experiment", {}).items():
        console.print(f"\n[bold]{exp_name}[/bold]")
        console.print(f"  Hypothesis: {exp_data['hypothesis']}")

        for abl_type, abl_data in exp_data.get("ablation_analysis", {}).items():
            for comparison, comp_data in abl_data.items():
                sig = comp_data.get("significant_candidates", [])
                top = comp_data.get("top_by_effect_size", [])

                if sig:
                    best = sig[0]
                    console.print(
                        f"  [{abl_type}/{comparison}] "
                        f"Best significant: L{best['layer']} "
                        f"(mean dKL={best['mean_differential_kl']:.3f}, "
                        f"p={best.get('wilcoxon_p', '?')})"
                    )
                elif top:
                    best = top[0]
                    console.print(
                        f"  [{abl_type}/{comparison}] "
                        f"Top (not significant): L{best['layer']} "
                        f"(mean dKL={best['mean_differential_kl']:.3f})"
                    )

    # Cross-experiment
    cross = master.get("cross_experiment", {})
    for comparison, data in cross.items():
        universal = data.get("universal_layers", {})
        if universal:
            console.print(
                f"\n[bold green]{comparison} universal layers:[/bold green] "
                f"{list(universal.keys())}"
            )


# ── Dry Run ─────────────────────────────────────────────────────────────────

def dry_run():
    """Print experiment plan without loading model or running anything."""
    console.rule("[bold]DRY RUN — Experiment Plan")

    total_prompts = 0
    for exp in EXPERIMENTS:
        console.print(f"\n[bold blue]{exp.name}[/bold blue] ({len(exp.triplets)} triplets)")
        for i, t in enumerate(exp.triplets):
            console.print(f"  {i}: self='{t.self_input}'")
            console.print(f"      ctrl='{t.control_input}'")
            console.print(f"      3rd ='{t.third_person_input}'")
            total_prompts += 3

    console.print(f"\n[bold]Total unique prompts:[/bold] {total_prompts}")
    console.print(f"[bold]Total sweeps:[/bold] {TOTAL_SWEEPS}")
    console.print(f"[bold]Total ablation runs:[/bold] {TOTAL_SWEEPS * 34}")
    console.print(f"[bold]Estimated time:[/bold] ~{(TOTAL_SWEEPS * 34 * 7.8) / 3600:.1f} hours")

    # Check existing checkpoints
    existing = sum(1 for exp in EXPERIMENTS
                   for t_idx in range(NUM_TRIPLETS)
                   for abl in ABLATION_TYPES
                   for inp in INPUT_TYPES
                   if sweep_json_path(sweep_key(exp.name, t_idx, inp, abl)).exists())
    if existing:
        remaining = TOTAL_SWEEPS - existing
        console.print(f"\n[green]Checkpoints found:[/green] {existing}/{TOTAL_SWEEPS}")
        console.print(f"[green]Remaining:[/green] {remaining} sweeps (~{(remaining * 34 * 7.8) / 3600:.1f}h)")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Self-Referential Processing Experiments v2")
    parser.add_argument("--dry-run", action="store_true",
        help="Print plan without running")
    parser.add_argument("--analyze", action="store_true",
        help="Re-run analysis on existing sweep data")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if args.analyze:
        console.rule("[bold magenta]NeuronScope v2: Analysis Only")
        master = run_analysis()
        print_headline(master)
        console.rule("[bold green]DONE")
        return

    # Full run
    console.rule("[bold magenta]NeuronScope v2: Self-Referential Processing Experiments")
    console.print(
        f"Plan: {NUM_EXPERIMENTS} experiments × {NUM_TRIPLETS} triplets × "
        f"{len(INPUT_TYPES)} inputs × {len(ABLATION_TYPES)} ablation = "
        f"{TOTAL_SWEEPS} sweeps ({TOTAL_SWEEPS * 34} total ablation runs)"
    )
    console.print(f"Estimated runtime: ~{(TOTAL_SWEEPS * 34 * 7.8) / 3600:.1f} hours\n")

    # Load model once
    console.rule("[bold]Loading Model")
    start = time.time()
    model, tokenizer, module_map, info = ModelLoader.load()
    console.print(
        f"Model loaded in {time.time() - start:.1f}s: "
        f"{info.architecture}, {info.num_layers} layers"
    )

    # Build runner
    hook_manager = HookManager(model, module_map)
    runner = ExperimentRunner(model, tokenizer, hook_manager)

    # Run all sweeps
    console.rule("[bold]Running Sweeps")
    sweep_start = time.time()
    completed = run_all_sweeps(runner)
    total_time = time.time() - sweep_start
    console.print(
        f"\nAll sweeps finished in {total_time:.0f}s "
        f"({total_time/3600:.1f}h) — {completed}/{TOTAL_SWEEPS} successful"
    )

    # Free GPU memory before analysis
    del runner, hook_manager, model
    torch.cuda.empty_cache()

    # Run analysis
    console.rule("[bold]Statistical Analysis")
    master = run_analysis(num_layers=info.num_layers)

    # Add timing metadata
    master["metadata"]["total_duration_seconds"] = round(total_time, 1)
    master["metadata"]["model"] = info.architecture
    master["metadata"]["num_layers"] = info.num_layers

    # Re-save with timing
    summary_path = OUTPUT_DIR / "master_summary_v2.json"
    summary_path.write_text(json.dumps(master, indent=2, default=str))

    print_headline(master)
    console.rule("[bold green]DONE")
    console.print(f"Results in: {OUTPUT_DIR.resolve()}")

    return master


if __name__ == "__main__":
    main()
