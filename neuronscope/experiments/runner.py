from __future__ import annotations

import time
import threading
import uuid

import torch
from rich.console import Console

from neuronscope.hooks.manager import HookManager
from neuronscope.hooks.targets import HookTarget, ComponentType
from neuronscope.hooks.interventions import (
    Intervention,
    ZeroAblation,
    MeanAblation,
    ActivationPatching,
    AdditivePerturbation,
)
from neuronscope.experiments.schema import (
    ExperimentConfig,
    ExperimentResult,
    InterventionSpec,
    CaptureSpec,
)
from neuronscope.experiments.comparator import (
    kl_divergence,
    top_k_predictions,
    rank_changes,
)
from neuronscope.experiments.reproducibility import set_all_seeds

console = Console()

# Global lock — one experiment at a time on GPU
_gpu_lock = threading.Lock()


class ExperimentRunner:
    """Executes experiments: clean run -> intervention run -> comparison.

    This is where the core scientific method lives:
    - Hypothesis: the intervention spec
    - Experiment: forward passes with/without intervention
    - Measurement: output comparison
    """

    def __init__(self, model, tokenizer, hook_manager: HookManager):
        self.model = model
        self.tokenizer = tokenizer
        self.hook_manager = hook_manager

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize input text and move to model device."""
        tokens = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        return tokens.input_ids.to(device)

    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return logits. Always no_grad."""
        with torch.no_grad():
            outputs = self.model(input_ids)
            return outputs.logits

    def _get_head_dim(self) -> int:
        """Get head_dim from model config for attention head slicing."""
        text_config = getattr(self.model.config, "text_config", self.model.config)
        head_dim_cfg = getattr(text_config, "head_dim", None)
        if head_dim_cfg is not None:
            return head_dim_cfg
        num_heads = getattr(text_config, "num_attention_heads", 8)
        hidden = getattr(text_config, "hidden_size", 2560)
        return hidden // num_heads

    def _spec_to_target(self, spec: InterventionSpec) -> HookTarget:
        """Convert an InterventionSpec to a HookTarget."""
        head_dim = self._get_head_dim() if spec.target_head is not None else None
        return HookTarget(
            layer=spec.target_layer,
            component=ComponentType(spec.target_component),
            head=spec.target_head,
            token_position=spec.target_position,
            neuron_index=spec.target_neuron,
            head_dim=head_dim,
        )

    def _capture_to_target(self, spec: CaptureSpec) -> HookTarget:
        """Convert a CaptureSpec to a HookTarget."""
        head_dim = self._get_head_dim() if spec.head is not None else None
        return HookTarget(
            layer=spec.layer,
            component=ComponentType(spec.component),
            head=spec.head,
            token_position=spec.token_position,
            neuron_index=spec.neuron_index,
            head_dim=head_dim,
        )

    def _build_intervention(
        self,
        spec: InterventionSpec,
        source_activations: dict[str, torch.Tensor],
    ) -> Intervention:
        """Build an Intervention object from a spec."""
        if spec.intervention_type == "zero":
            return ZeroAblation()
        elif spec.intervention_type == "mean":
            mean_key = self._spec_to_target(spec).to_key()
            mean_tensor = source_activations.get(mean_key, torch.zeros(1))
            return MeanAblation(mean_activation=mean_tensor)
        elif spec.intervention_type == "patch":
            target = self._spec_to_target(spec)
            source_key = target.to_key()
            source_tensor = source_activations.get(source_key)
            if source_tensor is None:
                raise ValueError(
                    f"Activation patching requires source activations for {source_key}. "
                    "Set source_input in the experiment config."
                )
            return ActivationPatching(source_activation=source_tensor)
        elif spec.intervention_type == "additive":
            direction = torch.randn(spec.intervention_params.get("dim", 1))
            magnitude = spec.intervention_params.get("magnitude", 1.0)
            return AdditivePerturbation(direction=direction, magnitude=magnitude)
        else:
            raise ValueError(f"Unknown intervention type: {spec.intervention_type}")

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a complete experiment.

        Steps:
        1. Clean run — no intervention, capture baseline
        2. Source run — if activation patching, get source activations
        3. Intervention run — apply interventions, capture modified output
        4. Compare clean vs. intervention outputs
        """
        with _gpu_lock:
            try:
                return self._run_locked(config)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "GPU ran out of memory during experiment. "
                    "Try: (1) shorter input text, (2) unload and reload the model, "
                    "or (3) close other GPU-using applications. "
                    "The GPU cache has been cleared automatically."
                )
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"GPU error: {e}. The CUDA cache has been cleared. "
                        "Try running the experiment again."
                    )
                raise

    def _run_locked(self, config: ExperimentConfig) -> ExperimentResult:
        start = time.time()
        device = str(next(self.model.parameters()).device)

        console.print(f"[bold]Running experiment: {config.name}[/bold]")

        # -- Phase 1: Clean run --
        set_all_seeds(config.seed)
        input_ids = self._tokenize(config.base_input)

        clean_activations: dict[str, torch.Tensor] = {}
        with self.hook_manager.session() as hm:
            # Attach extraction hooks for capture targets
            for spec in config.capture_targets:
                target = self._capture_to_target(spec)
                hm.attach_extraction_hook(target, clean_activations)

            # Also capture at intervention points for comparison
            for spec in config.interventions:
                target = self._spec_to_target(spec)
                hm.attach_extraction_hook(target, clean_activations)

            clean_logits = self._forward(input_ids)

        console.print(f"  Clean run complete. Captured {len(clean_activations)} activations.")

        # -- Phase 2: Source run (if needed for patching) --
        source_activations: dict[str, torch.Tensor] = {}
        if config.source_input:
            set_all_seeds(config.seed)
            source_ids = self._tokenize(config.source_input)

            with self.hook_manager.session() as hm:
                for spec in config.interventions:
                    target = self._spec_to_target(spec)
                    hm.attach_extraction_hook(target, source_activations)

                self._forward(source_ids)

            console.print(
                f"  Source run complete. Captured {len(source_activations)} activations."
            )

        # -- Phase 3: Intervention run --
        set_all_seeds(config.seed)  # Reset seeds for identical conditions

        with self.hook_manager.session() as hm:
            for spec in config.interventions:
                target = self._spec_to_target(spec)
                intervention = self._build_intervention(spec, source_activations)
                hm.attach_intervention_hook(target, intervention)

            intervention_logits = self._forward(input_ids)

        console.print("  Intervention run complete.")

        # -- Phase 4: Compare --
        clean_top_k = top_k_predictions(clean_logits, self.tokenizer)
        intervention_top_k = top_k_predictions(intervention_logits, self.tokenizer)

        kl = kl_divergence(clean_logits, intervention_logits)
        ranks = rank_changes(clean_logits, intervention_logits, self.tokenizer)

        duration = time.time() - start

        result = ExperimentResult(
            id=str(uuid.uuid4())[:8],
            config=config,
            config_hash=config.config_hash(),
            clean_top_k=clean_top_k,
            clean_output_token=clean_top_k[0].token,
            clean_output_prob=clean_top_k[0].prob,
            intervention_top_k=intervention_top_k,
            intervention_output_token=intervention_top_k[0].token,
            intervention_output_prob=intervention_top_k[0].prob,
            kl_divergence=kl,
            top_token_changed=clean_top_k[0].token_id != intervention_top_k[0].token_id,
            rank_changes=ranks,
            duration_seconds=round(duration, 3),
            device=device,
        )

        console.print(
            f"  [green]Done[/green] in {duration:.2f}s | "
            f"KL={kl:.4f} | "
            f"Top token: '{result.clean_output_token}' -> '{result.intervention_output_token}' | "
            f"Changed: {result.top_token_changed}"
        )

        return result

    def run_sweep(
        self,
        base_config: ExperimentConfig,
        sweep_layers: list[int] | None = None,
    ) -> list[ExperimentResult]:
        """Run the same intervention across multiple layers.

        Returns one result per layer, enabling per-layer causal attribution.
        """
        if sweep_layers is None:
            # Sweep all layers — get num_layers from model config
            num_layers = self.model.config.text_config.num_hidden_layers
            sweep_layers = list(range(num_layers))

        results = []
        for layer in sweep_layers:
            layer_config = base_config.model_copy(deep=True)
            layer_config.name = f"{base_config.name}_L{layer}"
            for spec in layer_config.interventions:
                spec.target_layer = layer

            try:
                result = self.run(layer_config)
                results.append(result)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "GPU" in str(e):
                    console.print(f"  [red]OOM at layer {layer}[/red] — stopping sweep early")
                    torch.cuda.empty_cache()
                    if not results:
                        raise
                    break
                raise

        return results

    def run_head_sweep(
        self,
        base_config: ExperimentConfig,
        layer: int,
        heads: list[int] | None = None,
    ) -> list[ExperimentResult]:
        """Run the same intervention across all attention heads in a single layer.

        Returns one result per head, enabling per-head causal attribution
        within the attention mechanism.
        """
        if heads is None:
            text_config = getattr(self.model.config, "text_config", self.model.config)
            num_heads = getattr(text_config, "num_attention_heads", 8)
            heads = list(range(num_heads))

        results = []
        for head in heads:
            head_config = base_config.model_copy(deep=True)
            head_config.name = f"{base_config.name}_L{layer}_H{head}"
            for spec in head_config.interventions:
                spec.target_layer = layer
                spec.target_head = head
                spec.target_component = "attn_output"

            try:
                result = self.run(head_config)
                results.append(result)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "GPU" in str(e):
                    console.print(f"  [red]OOM at head {head}[/red] — stopping sweep early")
                    torch.cuda.empty_cache()
                    if not results:
                        raise
                    break
                raise

        return results
