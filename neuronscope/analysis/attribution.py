"""Automated per-layer causal attribution via path patching.

For each layer, attaches an intervention hook, runs a forward pass,
and measures the causal effect (KL divergence + effect size) on the output.

This is the sweep-as-a-primitive: same logic as ExperimentRunner.run_sweep()
but structured as a reusable analysis function returning a typed result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from neuronscope.hooks.manager import HookManager
from neuronscope.hooks.targets import HookTarget, ComponentType
from neuronscope.hooks.interventions import ZeroAblation, MeanAblation
from neuronscope.analysis.metrics import kl_divergence


@dataclass
class AttributionResult:
    """Per-layer causal attribution scores."""

    layers: list[int]
    kl_scores: list[float]
    effect_sizes: list[float]
    peak_layer: int
    peak_kl: float
    total_effect: float
    component: str
    intervention_type: str

    def to_dict(self) -> dict:
        return {
            "layers": self.layers,
            "kl_scores": self.kl_scores,
            "effect_sizes": self.effect_sizes,
            "peak_layer": self.peak_layer,
            "peak_kl": self.peak_kl,
            "total_effect": self.total_effect,
            "component": self.component,
            "intervention_type": self.intervention_type,
        }


def attribute_by_layer(
    model: torch.nn.Module,
    tokenizer,
    hook_manager: HookManager,
    base_input: str,
    intervention_type: str = "zero",
    component: str = "mlp_output",
    layers: list[int] | None = None,
) -> AttributionResult:
    """Run causal attribution across layers.

    For each layer:
    1. Attach intervention hook at (layer, component)
    2. Forward pass
    3. Measure KL divergence from clean baseline

    Args:
        model: The loaded model.
        tokenizer: The model's tokenizer.
        hook_manager: HookManager instance for hook lifecycle.
        base_input: Input text to run attribution on.
        intervention_type: "zero" or "mean".
        component: Component type string (e.g. "mlp_output", "attn_output").
        layers: Which layers to test. None = all layers.

    Returns:
        AttributionResult with per-layer scores.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer(base_input, return_tensors="pt").input_ids.to(device)

    # Get number of layers
    if layers is None:
        num_layers = model.config.text_config.num_hidden_layers
        layers = list(range(num_layers))

    comp_type = ComponentType(component)

    # Clean baseline
    with torch.no_grad():
        clean_logits = model(input_ids).logits

    kl_scores: list[float] = []
    effect_sizes: list[float] = []

    for layer_idx in layers:
        target = HookTarget(layer=layer_idx, component=comp_type)

        if intervention_type == "zero":
            intervention = ZeroAblation()
        elif intervention_type == "mean":
            # For mean ablation, we need the clean activation as the mean
            # First capture it
            storage: dict[str, torch.Tensor] = {}
            with hook_manager.session() as hm:
                hm.attach_extraction_hook(target, storage)
                with torch.no_grad():
                    model(input_ids)
            mean_tensor = storage.get(target.to_key(), torch.zeros(1))
            intervention = MeanAblation(mean_activation=mean_tensor)
        else:
            raise ValueError(f"Unsupported intervention type for attribution: {intervention_type}")

        # Intervention forward pass
        with hook_manager.session() as hm:
            hm.attach_intervention_hook(target, intervention)
            with torch.no_grad():
                intervention_logits = model(input_ids).logits

        kl = kl_divergence(clean_logits, intervention_logits)
        kl_scores.append(kl)

    # Compute effect sizes relative to max KL
    max_kl = max(kl_scores) if kl_scores else 0.0
    if max_kl > 1e-8:
        effect_sizes = [kl / max_kl for kl in kl_scores]
    else:
        effect_sizes = [0.0] * len(kl_scores)

    total_effect = sum(kl_scores)
    peak_idx = kl_scores.index(max_kl) if kl_scores else 0
    peak_layer = layers[peak_idx] if layers else 0

    return AttributionResult(
        layers=layers,
        kl_scores=kl_scores,
        effect_sizes=effect_sizes,
        peak_layer=peak_layer,
        peak_kl=max_kl,
        total_effect=total_effect,
        component=component,
        intervention_type=intervention_type,
    )
