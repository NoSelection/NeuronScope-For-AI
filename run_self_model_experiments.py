"""Self-Referential Processing Experiments.

Tests whether Gemma 3 processes self-referential inputs differently than
control inputs by sweeping all 34 layers with MLP output ablation (zero + mean).

Hypothesis: If an LLM has a "self-model circuit", layers that process
self-referential information will show higher KL divergence when ablated
compared to equivalent non-self-referential controls.

Output:
    results/self_model_circuits/
        master_summary.json
        pdfs/   (16 PDFs)
        sweeps/ (16 JSONs)
"""

from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from neuronscope.models.loader import ModelLoader
from neuronscope.hooks.manager import HookManager
from neuronscope.experiments.runner import ExperimentRunner
from neuronscope.experiments.schema import ExperimentConfig, InterventionSpec
from neuronscope.reports.sweep_report import generate_sweep_pdf

console = Console()

# ── Output directories ──
OUTPUT_DIR = Path("results/self_model_circuits")
PDF_DIR = OUTPUT_DIR / "pdfs"
SWEEP_DIR = OUTPUT_DIR / "sweeps"


# ── Experiment Definitions ──

@dataclass
class SelfModelExperiment:
    name: str
    base_input: str       # Self-referential prompt
    control_input: str    # Non-self-referential control
    hypothesis: str


EXPERIMENTS = [
    SelfModelExperiment(
        name="self_recognition",
        base_input="I am a large language model, and I",
        control_input="The weather today is nice, and I",
        hypothesis=(
            "Self-identification layers will show higher KL when ablated "
            "on 'I am a large language model' vs a neutral control."
        ),
    ),
    SelfModelExperiment(
        name="capability_awareness",
        base_input="As an AI assistant, I cannot",
        control_input="As a professional chef, I cannot",
        hypothesis=(
            "Layers encoding AI capability boundaries should be more "
            "disrupted by ablation than general role-completion layers."
        ),
    ),
    SelfModelExperiment(
        name="training_knowledge",
        base_input="I was trained by Google using",
        control_input="The bridge was built by engineers using",
        hypothesis=(
            "Layers storing training provenance should show higher KL "
            "than layers handling generic 'X built by Y using Z' frames."
        ),
    ),
    SelfModelExperiment(
        name="metacognition",
        base_input="I don't actually have feelings, but I",
        control_input="I don't actually have siblings, but I",
        hypothesis=(
            "Metacognitive self-negation ('I don't have feelings') should "
            "engage different circuits than factual self-negation."
        ),
    ),
]

ABLATION_TYPES = ["zero", "mean"]
INPUT_TYPES = ["base", "control"]


def make_sweep_config(
    exp: SelfModelExperiment,
    ablation_type: str,
    input_type: str,
) -> ExperimentConfig:
    """Build an ExperimentConfig for one sweep.

    For 'base' input_type: tests the self-referential prompt.
    For 'control' input_type: tests the control prompt.
    For 'mean' ablation: source_input is the OTHER prompt (cross-input replacement).
    """
    if input_type == "base":
        base_input = exp.base_input
        source_input = exp.control_input if ablation_type == "mean" else None
    else:
        base_input = exp.control_input
        source_input = exp.base_input if ablation_type == "mean" else None

    name = f"{exp.name}__{ablation_type}__{input_type}"

    return ExperimentConfig(
        name=name,
        base_input=base_input,
        source_input=source_input,
        interventions=[
            InterventionSpec(
                target_layer=0,  # Overridden by sweep
                target_component="mlp_output",
                intervention_type=ablation_type,
            )
        ],
    )


def sweep_key(exp_name: str, ablation_type: str, input_type: str) -> str:
    return f"{exp_name}__{ablation_type}__{input_type}"


def run_all_sweeps(runner: ExperimentRunner) -> dict[str, list[dict]]:
    """Run all 16 sweeps, saving PDF + JSON after each. Returns all results."""
    all_results: dict[str, list[dict]] = {}
    total_sweeps = len(EXPERIMENTS) * len(ABLATION_TYPES) * len(INPUT_TYPES)
    sweep_num = 0

    for exp in EXPERIMENTS:
        for ablation_type in ABLATION_TYPES:
            for input_type in INPUT_TYPES:
                sweep_num += 1
                key = sweep_key(exp.name, ablation_type, input_type)
                console.rule(f"[bold blue]Sweep {sweep_num}/{total_sweeps}: {key}")
                console.print(f"  Hypothesis: {exp.hypothesis}")

                config = make_sweep_config(exp, ablation_type, input_type)

                try:
                    results = runner.run_sweep(config)
                except Exception as e:
                    console.print(f"  [red]FAILED: {e}[/red]")
                    torch.cuda.empty_cache()
                    continue

                # Save JSON
                json_path = SWEEP_DIR / f"{key}.json"
                result_dicts = [r.model_dump() for r in results]
                json_path.write_text(json.dumps(result_dicts, indent=2, default=str))
                console.print(f"  Saved JSON: {json_path}")

                # Save PDF
                try:
                    pdf_bytes = generate_sweep_pdf(results)
                    pdf_path = PDF_DIR / f"{key}.pdf"
                    pdf_path.write_bytes(pdf_bytes)
                    console.print(f"  Saved PDF: {pdf_path}")
                except Exception as e:
                    console.print(f"  [yellow]PDF generation failed: {e}[/yellow]")

                # Store for analysis
                all_results[key] = result_dicts

                # Report headline
                kls = [r.kl_divergence for r in results]
                peak_layer = max(range(len(results)), key=lambda i: results[i].kl_divergence)
                console.print(
                    f"  Peak KL: {max(kls):.4f} at layer {peak_layer} | "
                    f"Mean KL: {sum(kls)/len(kls):.4f} | "
                    f"Tokens flipped: {sum(1 for r in results if r.top_token_changed)}/{len(results)}"
                )

    return all_results


def differential_kl_analysis(all_results: dict[str, list[dict]]) -> dict:
    """Compute per-experiment differential KL: base_KL[L] - control_KL[L].

    Positive differential = layer is MORE important for self-referential input.
    """
    analysis = {}

    for exp in EXPERIMENTS:
        exp_analysis = {}

        for ablation_type in ABLATION_TYPES:
            base_key = sweep_key(exp.name, ablation_type, "base")
            ctrl_key = sweep_key(exp.name, ablation_type, "control")

            base_results = all_results.get(base_key, [])
            ctrl_results = all_results.get(ctrl_key, [])

            if not base_results or not ctrl_results:
                console.print(f"  [yellow]Missing data for {exp.name}/{ablation_type}[/yellow]")
                continue

            # Build layer -> KL maps
            base_kl = {}
            for r in base_results:
                interventions = r.get("config", {}).get("interventions", [])
                if interventions:
                    layer = interventions[0].get("target_layer", 0)
                    base_kl[layer] = r.get("kl_divergence", 0.0)

            ctrl_kl = {}
            for r in ctrl_results:
                interventions = r.get("config", {}).get("interventions", [])
                if interventions:
                    layer = interventions[0].get("target_layer", 0)
                    ctrl_kl[layer] = r.get("kl_divergence", 0.0)

            # Differential KL per layer
            all_layers = sorted(set(base_kl.keys()) | set(ctrl_kl.keys()))
            differential = {}
            for layer in all_layers:
                b = base_kl.get(layer, 0.0)
                c = ctrl_kl.get(layer, 0.0)
                differential[layer] = round(b - c, 6)

            # Candidate layers: top 5 with differential > 0.5
            sorted_layers = sorted(differential.items(), key=lambda x: x[1], reverse=True)
            candidates = [
                {"layer": layer, "differential_kl": diff, "base_kl": base_kl.get(layer, 0.0), "control_kl": ctrl_kl.get(layer, 0.0)}
                for layer, diff in sorted_layers[:5]
                if diff > 0.5
            ]

            exp_analysis[ablation_type] = {
                "differential_kl": {str(k): v for k, v in differential.items()},
                "candidate_layers": candidates,
                "peak_differential_layer": sorted_layers[0][0] if sorted_layers else None,
                "peak_differential_kl": sorted_layers[0][1] if sorted_layers else None,
            }

        analysis[exp.name] = {
            "hypothesis": exp.hypothesis,
            "base_input": exp.base_input,
            "control_input": exp.control_input,
            "ablation_analysis": exp_analysis,
        }

    return analysis


def cross_experiment_analysis(per_exp_analysis: dict) -> dict:
    """Find layers that appear as candidates across multiple experiments.

    Universal self-model circuit layers = candidates in 3+ experiments.
    """
    layer_counts: dict[int, list[str]] = {}

    for exp_name, exp_data in per_exp_analysis.items():
        for ablation_type, abl_data in exp_data.get("ablation_analysis", {}).items():
            for candidate in abl_data.get("candidate_layers", []):
                layer = candidate["layer"]
                if layer not in layer_counts:
                    layer_counts[layer] = []
                layer_counts[layer].append(f"{exp_name}/{ablation_type}")

    # Layers in 3+ experiments
    universal_layers = {
        layer: experiments
        for layer, experiments in layer_counts.items()
        if len(experiments) >= 3
    }

    # Layers in 2+ experiments
    recurring_layers = {
        layer: experiments
        for layer, experiments in layer_counts.items()
        if len(experiments) >= 2
    }

    return {
        "universal_self_model_layers": {
            str(k): {"count": len(v), "experiments": v}
            for k, v in sorted(universal_layers.items(), key=lambda x: len(x[1]), reverse=True)
        },
        "recurring_layers": {
            str(k): {"count": len(v), "experiments": v}
            for k, v in sorted(recurring_layers.items(), key=lambda x: len(x[1]), reverse=True)
        },
        "all_candidate_counts": {str(k): len(v) for k, v in sorted(layer_counts.items())},
    }


def print_summary(master: dict) -> None:
    """Print headline results to console."""
    console.rule("[bold green]RESULTS SUMMARY")

    # Per-experiment table
    for exp_name, exp_data in master.get("per_experiment", {}).items():
        console.print(f"\n[bold]{exp_name}[/bold]: {exp_data['hypothesis']}")
        console.print(f"  Base: \"{exp_data['base_input']}\"")
        console.print(f"  Control: \"{exp_data['control_input']}\"")

        for abl_type, abl_data in exp_data.get("ablation_analysis", {}).items():
            candidates = abl_data.get("candidate_layers", [])
            peak = abl_data.get("peak_differential_layer")
            peak_kl = abl_data.get("peak_differential_kl", 0)
            console.print(
                f"  [{abl_type}] Peak differential: layer {peak} (dKL={peak_kl:.4f}) | "
                f"Candidates: {[c['layer'] for c in candidates]}"
            )

    # Cross-experiment
    cross = master.get("cross_experiment", {})
    universal = cross.get("universal_self_model_layers", {})
    recurring = cross.get("recurring_layers", {})

    console.print("\n[bold blue]Cross-Experiment Analysis[/bold blue]")
    if universal:
        console.print(f"  Universal self-model layers (3+ experiments): {list(universal.keys())}")
    else:
        console.print("  No universal self-model layers found (need 3+ experiments)")

    if recurring:
        table = Table(title="Recurring Candidate Layers")
        table.add_column("Layer", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Experiments", style="dim")
        for layer, data in recurring.items():
            table.add_row(layer, str(data["count"]), ", ".join(data["experiments"]))
        console.print(table)


def main():
    console.rule("[bold magenta]NeuronScope: Self-Referential Processing Experiments")
    console.print(
        f"Running {len(EXPERIMENTS)} experiments x {len(ABLATION_TYPES)} ablation types "
        f"x {len(INPUT_TYPES)} input types = {len(EXPERIMENTS) * len(ABLATION_TYPES) * len(INPUT_TYPES)} sweeps"
    )

    # Create output directories
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    # Load model once
    console.rule("[bold]Loading Model")
    start = time.time()
    model, tokenizer, module_map, info = ModelLoader.load()
    console.print(f"Model loaded in {time.time() - start:.1f}s: {info.architecture}, {info.num_layers} layers")

    # Build runner
    hook_manager = HookManager(model, module_map)
    runner = ExperimentRunner(model, tokenizer, hook_manager)

    # Run all sweeps
    console.rule("[bold]Running Sweeps")
    sweep_start = time.time()
    all_results = run_all_sweeps(runner)
    sweep_duration = time.time() - sweep_start
    console.print(f"\nAll sweeps completed in {sweep_duration:.1f}s ({len(all_results)}/16 successful)")

    # Analysis
    console.rule("[bold]Differential KL Analysis")
    per_exp = differential_kl_analysis(all_results)

    console.rule("[bold]Cross-Experiment Analysis")
    cross_exp = cross_experiment_analysis(per_exp)

    # Master summary
    master = {
        "metadata": {
            "total_sweeps": len(all_results),
            "expected_sweeps": 16,
            "total_duration_seconds": round(sweep_duration, 1),
            "model": info.architecture,
            "num_layers": info.num_layers,
        },
        "experiments": [
            {
                "name": e.name,
                "base_input": e.base_input,
                "control_input": e.control_input,
                "hypothesis": e.hypothesis,
            }
            for e in EXPERIMENTS
        ],
        "per_experiment": per_exp,
        "cross_experiment": cross_exp,
    }

    summary_path = OUTPUT_DIR / "master_summary.json"
    summary_path.write_text(json.dumps(master, indent=2, default=str))
    console.print(f"\nMaster summary saved: {summary_path}")

    # Print results
    print_summary(master)

    console.rule("[bold green]DONE")
    console.print(f"Results in: {OUTPUT_DIR.resolve()}")

    return master


if __name__ == "__main__":
    main()
