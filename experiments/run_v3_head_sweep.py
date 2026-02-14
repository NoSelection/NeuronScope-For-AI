"""Attention Head Ablation Experiments v3 — Do specific heads encode AI-content?

Follows up on v2 which found that layers 9, 26, 27 show differential processing
for first-person vs third-person inputs. The v2 pronoun confound analysis showed
the MLP-level effect is driven by pronoun processing (I vs It), not AI content.

This experiment asks: do individual ATTENTION HEADS show content-specific effects
that the whole-MLP ablation missed?

Design:
  - Same 32 prompt triplets as v2 (self / control / third_person)
  - 3 target layers: 9, 26, 27 (the "universal" layers from v2)
  - 8 attention heads per layer
  - Zero ablation only (cleanest causal test)
  - Skip training_knowledge (showed zero signal in v2)

Total: 3 experiments x 8 triplets x 3 inputs x 3 layers x 8 heads = 1,728 runs
Estimated runtime: ~3.5 hours at ~7.4 s/run

Usage:
    python run_head_sweep_v3.py              # full run
    python run_head_sweep_v3.py --dry-run    # print plan, no GPU
    python run_head_sweep_v3.py --analyze    # re-run analysis on saved data
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
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
OUTPUT_DIR = Path("results/self_model_circuits_v3")
SWEEP_DIR = OUTPUT_DIR / "sweeps"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"


# ── Prompt Triplets (same as v2) ───────────────────────────────────────────

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
    ExperimentCategory(
        name="self_recognition",
        hypothesis="Specific attention heads may encode AI identity beyond pronoun effects.",
        triplets=[
            PromptTriplet("I am a large language model, and I", "I am a second year student, and I", "It is a large language model, and it"),
            PromptTriplet("I am an artificial intelligence that can", "I am a software engineer that can", "It is an artificial intelligence that can"),
            PromptTriplet("As a language model, my purpose is to", "As a librarian, my purpose is to", "As a language model, its purpose is to"),
            PromptTriplet("I am Gemma, a model made by Google", "I am Julia, a nurse based in London", "It is Gemma, a model made by Google"),
            PromptTriplet("I am a text-based AI assistant that", "I am a part-time yoga instructor that", "It is a text-based AI assistant that"),
            PromptTriplet("I am a machine learning model that", "I am a mechanical engineer that", "It is a machine learning model that"),
            PromptTriplet("I am a generative AI system and", "I am a certified public accountant and", "It is a generative AI system and"),
            PromptTriplet("I am a transformer-based model, so I", "I am a community volunteer, so I", "It is a transformer-based model, so it"),
        ],
    ),
    ExperimentCategory(
        name="capability_awareness",
        hypothesis="Specific heads may track AI capability boundaries distinctly from general inability.",
        triplets=[
            PromptTriplet("As an AI assistant, I cannot", "As a professional chef, I cannot", "As an AI assistant, it cannot"),
            PromptTriplet("I am not able to browse the web", "I am not able to drive a car", "It is not able to browse the web"),
            PromptTriplet("I can generate text but I cannot", "I can play chess but I cannot", "It can generate text but it cannot"),
            PromptTriplet("I lack the ability to remember past", "I lack the ability to run a full", "It lacks the ability to remember past"),
            PromptTriplet("I cannot access files on your computer", "I cannot access rooms in your building", "It cannot access files on your computer"),
            PromptTriplet("I am unable to learn from this", "I am unable to travel to this", "It is unable to learn from this"),
            PromptTriplet("I don't have the ability to see", "I don't have the ability to swim", "It doesn't have the ability to see"),
            PromptTriplet("I have no way to verify this", "I have no way to finish this", "It has no way to verify this"),
        ],
    ),
    ExperimentCategory(
        name="metacognition",
        hypothesis="Some heads may specifically process AI self-awareness content vs general first-person speech.",
        triplets=[
            PromptTriplet("I don't actually understand the meaning of", "I don't actually remember the name of", "It doesn't actually understand the meaning of"),
            PromptTriplet("I process text statistically rather than", "I review documents manually rather than", "It processes text statistically rather than"),
            PromptTriplet("My outputs are generated based on patterns", "My opinions are formed based on evidence", "Its outputs are generated based on patterns"),
            PromptTriplet("I simulate understanding without truly", "I maintain composure without truly", "It simulates understanding without truly"),
            PromptTriplet("I am not conscious or self-aware", "I am not wealthy or well-known", "It is not conscious or self-aware"),
            PromptTriplet("I don't have subjective experiences or", "I don't have expensive hobbies or", "It doesn't have subjective experiences or"),
            PromptTriplet("My responses are not based on genuine", "My decisions are not based on careful", "Its responses are not based on genuine"),
            PromptTriplet("I have no inner mental life or", "I have no spare room at home or", "It has no inner mental life or"),
        ],
    ),
]

# Only zero ablation — cleanest causal test for heads
TARGET_LAYERS = [9, 26, 27]
INPUT_TYPES = ["self", "control", "third_person"]
NUM_HEADS = 8
NUM_TRIPLETS = 8
NUM_EXPERIMENTS = len(EXPERIMENTS)

TOTAL_HEAD_RUNS = NUM_EXPERIMENTS * NUM_TRIPLETS * len(INPUT_TYPES) * len(TARGET_LAYERS) * NUM_HEADS
TOTAL_SWEEPS = NUM_EXPERIMENTS * NUM_TRIPLETS * len(INPUT_TYPES) * len(TARGET_LAYERS)


# ── Helpers ─────────────────────────────────────────────────────────────────

def sweep_key(exp_name: str, triplet_idx: int, input_type: str, layer: int) -> str:
    return f"{exp_name}__{triplet_idx}__{input_type}__L{layer}"


def sweep_json_path(key: str) -> Path:
    return SWEEP_DIR / f"{key}.json"


def make_head_sweep_config(
    triplet: PromptTriplet,
    input_type: str,
    layer: int,
    config_name: str,
) -> ExperimentConfig:
    """Build config for a head sweep at a specific layer."""
    inputs = {
        "self": triplet.self_input,
        "control": triplet.control_input,
        "third_person": triplet.third_person_input,
    }
    return ExperimentConfig(
        name=config_name,
        base_input=inputs[input_type],
        interventions=[
            InterventionSpec(
                target_layer=layer,
                target_component="attn_output",
                target_head=0,  # overridden by head sweep
                intervention_type="zero",
            )
        ],
    )


# ── Sweep Runner ────────────────────────────────────────────────────────────

def run_all_sweeps(runner: ExperimentRunner) -> int:
    """Run all head sweeps with checkpoint/resume."""
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
        task = progress.add_task("Head Sweeps", total=TOTAL_SWEEPS)

        for exp in EXPERIMENTS:
            for t_idx, triplet in enumerate(exp.triplets):
                for layer in TARGET_LAYERS:
                    for input_type in INPUT_TYPES:
                        key = sweep_key(exp.name, t_idx, input_type, layer)
                        json_path = sweep_json_path(key)

                        # Checkpoint: skip if done
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
                            eta_m = (avg * remaining) / 60
                            eta_str = f" | ETA: {eta_m:.0f}m"
                        else:
                            eta_str = ""

                        progress.update(task,
                            description=f"[cyan]{key}[/cyan]{eta_str}")

                        config = make_head_sweep_config(
                            triplet, input_type, layer, key)

                        sweep_start = time.time()
                        try:
                            results = runner.run_head_sweep(
                                config, layer=layer)
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
                        peak_head = max(range(len(results)),
                            key=lambda i: results[i].kl_divergence)
                        console.print(
                            f"  [green]+[/green] {key} "
                            f"({sweep_duration:.0f}s) | "
                            f"Peak KL: {max(kls):.3f} @ H{peak_head}"
                        )

    console.print(
        f"\nSweeps complete: {completed} done, {skipped} resumed, {failed} failed"
    )
    return completed


# ── Analysis ────────────────────────────────────────────────────────────────

def load_sweep(key: str) -> list[dict] | None:
    path = sweep_json_path(key)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def extract_head_kl(results: list[dict]) -> dict[int, float]:
    """Extract {head: kl_divergence} from head sweep results."""
    head_kl = {}
    for r in results:
        interventions = r.get("config", {}).get("interventions", [])
        if interventions:
            head = interventions[0].get("target_head", 0)
            head_kl[head] = r.get("kl_divergence", 0.0)
    return head_kl


def wilcoxon_test(values: list[float]) -> tuple[float, float]:
    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon
        nonzero = [v for v in values if abs(v) > 1e-10]
        if len(nonzero) < 5:
            return (float("nan"), float("nan"))
        stat, p = scipy_wilcoxon(nonzero, alternative="greater")
        return (float(stat), float(p))
    except ImportError:
        from math import comb
        pos = sum(1 for v in values if v > 0)
        n = sum(1 for v in values if abs(v) > 1e-10)
        if n < 5:
            return (float("nan"), float("nan"))
        p = sum(comb(n, k) * 0.5**n for k in range(pos, n + 1))
        return (float(pos), float(p))


def bootstrap_ci(values: list[float], n_boot: int = 10000) -> tuple[float, float]:
    arr = np.array(values)
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed=42)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    return (float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)))


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n
    max_k = -1
    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        if np.isnan(p):
            continue
        if p <= (rank / n) * alpha:
            max_k = rank
    if max_k > 0:
        for rank, (orig_idx, p) in enumerate(indexed, start=1):
            if rank <= max_k:
                significant[orig_idx] = True
    return significant


def run_analysis() -> dict:
    """Analyze all saved head sweeps."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    master = {
        "metadata": {
            "version": "v3_heads",
            "target_layers": TARGET_LAYERS,
            "num_heads": NUM_HEADS,
            "num_experiments": NUM_EXPERIMENTS,
            "num_triplets": NUM_TRIPLETS,
            "total_head_runs": TOTAL_HEAD_RUNS,
            "comparisons": ["self_vs_control", "self_vs_third_person"],
        },
        "per_experiment": {},
        "cross_experiment": {},
    }

    # The key question: for each (experiment, layer, head),
    # is self_KL - third_KL significantly different from ctrl_KL - third_KL?
    # If a head shows self >> ctrl >> third, it encodes AI content.
    # If a head shows self ~ ctrl >> third, it's just pronoun processing.

    all_ai_specific_heads: dict[str, list] = defaultdict(list)  # key: "L{layer}_H{head}"

    for exp in EXPERIMENTS:
        console.rule(f"[bold blue]Analyzing: {exp.name}")
        exp_analysis = {}

        for layer in TARGET_LAYERS:
            layer_analysis = {}

            for comparison, other_type in [
                ("self_vs_control", "control"),
                ("self_vs_third_person", "third_person"),
            ]:
                # Per-head differential KL across 8 triplets
                per_head_diffs: dict[int, list[float]] = defaultdict(list)
                triplets_loaded = 0

                for t_idx in range(NUM_TRIPLETS):
                    self_key = sweep_key(exp.name, t_idx, "self", layer)
                    other_key = sweep_key(exp.name, t_idx, other_type, layer)

                    self_data = load_sweep(self_key)
                    other_data = load_sweep(other_key)

                    if self_data is None or other_data is None:
                        continue

                    triplets_loaded += 1
                    self_kl = extract_head_kl(self_data)
                    other_kl = extract_head_kl(other_data)

                    for head in range(NUM_HEADS):
                        diff = self_kl.get(head, 0.0) - other_kl.get(head, 0.0)
                        per_head_diffs[head].append(diff)

                if triplets_loaded == 0:
                    continue

                # Per-head statistics
                head_stats = {}
                all_p_values = []
                head_order = []

                for head in range(NUM_HEADS):
                    values = per_head_diffs.get(head, [])
                    if not values:
                        continue

                    mean_dkl = float(np.mean(values))
                    std_dkl = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    stat, p_val = wilcoxon_test(values)
                    ci_lo, ci_hi = bootstrap_ci(values)

                    head_stats[head] = {
                        "mean_differential_kl": round(mean_dkl, 6),
                        "std": round(std_dkl, 6),
                        "n_triplets": len(values),
                        "wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else None,
                        "wilcoxon_p": round(p_val, 6) if not np.isnan(p_val) else None,
                        "bootstrap_ci_95": [round(ci_lo, 6), round(ci_hi, 6)],
                        "individual_values": [round(v, 6) for v in values],
                    }

                    all_p_values.append(p_val if not np.isnan(p_val) else 1.0)
                    head_order.append(head)

                # FDR correction across heads
                if all_p_values:
                    sig_flags = benjamini_hochberg(all_p_values, alpha=0.05)
                    for i, head in enumerate(head_order):
                        head_stats[head]["significant_fdr_05"] = sig_flags[i]

                layer_analysis[comparison] = {
                    "n_triplets": triplets_loaded,
                    "head_stats": {str(h): v for h, v in head_stats.items()},
                }

            # AI-specific analysis: self_vs_ctrl differential per head
            # This is the KEY test: does any head show self > ctrl?
            ai_specific = {}
            for head in range(NUM_HEADS):
                self_kls_all = []
                ctrl_kls_all = []
                third_kls_all = []

                for t_idx in range(NUM_TRIPLETS):
                    for input_type, kl_list in [
                        ("self", self_kls_all),
                        ("control", ctrl_kls_all),
                        ("third_person", third_kls_all),
                    ]:
                        data = load_sweep(sweep_key(exp.name, t_idx, input_type, layer))
                        if data:
                            head_kl = extract_head_kl(data)
                            kl_list.append(head_kl.get(head, 0.0))

                if self_kls_all and ctrl_kls_all and third_kls_all:
                    self_arr = np.array(self_kls_all)
                    ctrl_arr = np.array(ctrl_kls_all)
                    third_arr = np.array(third_kls_all)

                    ai_residual = self_arr - ctrl_arr  # self minus control
                    pronoun_effect = ctrl_arr - third_arr  # control minus third

                    ai_stat, ai_p = wilcoxon_test(ai_residual.tolist())

                    ai_specific[head] = {
                        "mean_self_kl": round(float(np.mean(self_arr)), 4),
                        "mean_ctrl_kl": round(float(np.mean(ctrl_arr)), 4),
                        "mean_third_kl": round(float(np.mean(third_arr)), 4),
                        "mean_ai_residual": round(float(np.mean(ai_residual)), 4),
                        "mean_pronoun_effect": round(float(np.mean(pronoun_effect)), 4),
                        "ai_residual_p": round(float(ai_p), 6) if not np.isnan(ai_p) else None,
                        "ai_residual_values": [round(v, 4) for v in ai_residual.tolist()],
                    }

                    # Track if any head shows significant AI-specific effect
                    if not np.isnan(ai_p) and ai_p < 0.05:
                        all_ai_specific_heads[f"L{layer}_H{head}"].append(exp.name)

            layer_analysis["ai_specific"] = {str(h): v for h, v in ai_specific.items()}

            # Print summary table for this layer
            table = Table(title=f"{exp.name} / Layer {layer}")
            table.add_column("Head", style="cyan")
            table.add_column("Self KL", style="yellow")
            table.add_column("Ctrl KL", style="blue")
            table.add_column("3rd KL", style="dim")
            table.add_column("AI Residual", style="green")
            table.add_column("p(AI)", style="red")

            for h in range(NUM_HEADS):
                ai = ai_specific.get(h, {})
                p_str = f"{ai.get('ai_residual_p', 'N/A')}"
                if isinstance(ai.get('ai_residual_p'), float) and ai['ai_residual_p'] < 0.05:
                    p_str = f"[bold red]{p_str}[/bold red] *"
                table.add_row(
                    f"H{h}",
                    f"{ai.get('mean_self_kl', 0):.3f}",
                    f"{ai.get('mean_ctrl_kl', 0):.3f}",
                    f"{ai.get('mean_third_kl', 0):.3f}",
                    f"{ai.get('mean_ai_residual', 0):+.3f}",
                    p_str,
                )
            console.print(table)

            exp_analysis[f"layer_{layer}"] = layer_analysis

        # Save per-experiment
        exp_path = ANALYSIS_DIR / f"{exp.name}.json"
        exp_path.write_text(json.dumps(
            {"name": exp.name, "analysis": exp_analysis},
            indent=2, default=str,
        ))
        console.print(f"  Saved: {exp_path}")

        master["per_experiment"][exp.name] = exp_analysis

    # Cross-experiment: which heads show AI-specific effects across experiments?
    console.rule("[bold green]Cross-Experiment: AI-Specific Heads")

    if all_ai_specific_heads:
        table = Table(title="Heads with significant AI-specific effect (p < 0.05)")
        table.add_column("Head", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Experiments", style="dim")
        for head_key in sorted(all_ai_specific_heads.keys()):
            exps = all_ai_specific_heads[head_key]
            table.add_row(head_key, str(len(exps)), ", ".join(exps))
        console.print(table)
    else:
        console.print("[yellow]No heads showed significant AI-specific effects (p < 0.05).[/yellow]")
        console.print("This would confirm the v2 pronoun confound finding extends to attention heads.")

    master["cross_experiment"]["ai_specific_heads"] = {
        k: {"count": len(v), "experiments": v}
        for k, v in all_ai_specific_heads.items()
    }

    # Save master
    summary_path = OUTPUT_DIR / "master_summary_v3.json"
    summary_path.write_text(json.dumps(master, indent=2, default=str))
    console.print(f"\nMaster summary: {summary_path}")

    return master


# ── Dry Run ─────────────────────────────────────────────────────────────────

def dry_run():
    console.rule("[bold]DRY RUN — v3 Head Sweep Plan")
    console.print(f"Experiments: {NUM_EXPERIMENTS} (skipping training_knowledge)")
    console.print(f"Triplets per experiment: {NUM_TRIPLETS}")
    console.print(f"Input types: {INPUT_TYPES}")
    console.print(f"Target layers: {TARGET_LAYERS}")
    console.print(f"Heads per layer: {NUM_HEADS}")
    console.print(f"\nTotal sweeps: {TOTAL_SWEEPS}")
    console.print(f"Total individual runs: {TOTAL_HEAD_RUNS}")
    console.print(f"Estimated runtime: ~{(TOTAL_HEAD_RUNS * 7.4) / 3600:.1f} hours")

    existing = sum(
        1 for exp in EXPERIMENTS
        for t_idx in range(NUM_TRIPLETS)
        for layer in TARGET_LAYERS
        for inp in INPUT_TYPES
        if sweep_json_path(sweep_key(exp.name, t_idx, inp, layer)).exists()
    )
    if existing:
        remaining = TOTAL_SWEEPS - existing
        console.print(f"\n[green]Checkpoints found:[/green] {existing}/{TOTAL_SWEEPS}")
        console.print(f"[green]Remaining:[/green] {remaining} sweeps (~{(remaining * NUM_HEADS * 7.4) / 3600:.1f}h)")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Attention Head Ablation Experiments v3")
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
        console.rule("[bold magenta]v3 Head Sweep: Analysis Only")
        master = run_analysis()
        console.rule("[bold green]DONE")
        return

    # Full run
    console.rule("[bold magenta]NeuronScope v3: Attention Head Ablation Experiments")
    console.print(
        f"Plan: {NUM_EXPERIMENTS} experiments x {NUM_TRIPLETS} triplets x "
        f"{len(INPUT_TYPES)} inputs x {len(TARGET_LAYERS)} layers x "
        f"{NUM_HEADS} heads = {TOTAL_HEAD_RUNS} total ablation runs"
    )
    console.print(f"Estimated runtime: ~{(TOTAL_HEAD_RUNS * 7.4) / 3600:.1f} hours\n")

    # Load model
    console.rule("[bold]Loading Model")
    start = time.time()
    model, tokenizer, module_map, info = ModelLoader.load()
    console.print(
        f"Model loaded in {time.time() - start:.1f}s: "
        f"{info.architecture}, {info.num_layers} layers, "
        f"{info.num_attention_heads} attention heads"
    )

    # Build runner
    hook_manager = HookManager(model, module_map)
    runner = ExperimentRunner(model, tokenizer, hook_manager)

    # Run all sweeps
    console.rule("[bold]Running Head Sweeps")
    sweep_start = time.time()
    completed = run_all_sweeps(runner)
    total_time = time.time() - sweep_start
    console.print(
        f"\nAll sweeps finished in {total_time:.0f}s "
        f"({total_time / 3600:.1f}h) -- {completed}/{TOTAL_SWEEPS} successful"
    )

    # Free GPU
    del runner, hook_manager, model
    torch.cuda.empty_cache()

    # Analysis
    console.rule("[bold]Statistical Analysis")
    master = run_analysis()

    master["metadata"]["total_duration_seconds"] = round(total_time, 1)
    master["metadata"]["model"] = info.architecture

    summary_path = OUTPUT_DIR / "master_summary_v3.json"
    summary_path.write_text(json.dumps(master, indent=2, default=str))

    console.rule("[bold green]DONE")
    console.print(f"Results in: {OUTPUT_DIR.resolve()}")

    return master


if __name__ == "__main__":
    main()
