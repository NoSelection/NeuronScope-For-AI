from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ComponentType(Enum):
    """Types of hookable components within a transformer layer."""

    EMBEDDING = "embedding"
    RESIDUAL_PRE = "residual_pre"
    RESIDUAL_POST = "residual_post"
    ATTENTION_OUTPUT = "attn_output"
    ATTENTION_PATTERN = "attn_pattern"
    MLP_GATE = "mlp_gate"
    MLP_OUTPUT = "mlp_output"
    FINAL_LOGITS = "final_logits"


# Map from architecture prefix to module path templates.
# {L} is replaced with layer index.
_MODULE_TEMPLATES: dict[str, dict[ComponentType, str]] = {
    # Gemma 3 (conditional generation / multimodal)
    # Actual paths: model.language_model.layers.{L}, model.language_model.embed_tokens, lm_head
    "gemma3_conditional": {
        ComponentType.EMBEDDING: "model.language_model.embed_tokens",
        ComponentType.RESIDUAL_PRE: "model.language_model.layers.{L}",
        ComponentType.RESIDUAL_POST: "model.language_model.layers.{L}",
        ComponentType.ATTENTION_OUTPUT: "model.language_model.layers.{L}.self_attn.o_proj",
        ComponentType.MLP_GATE: "model.language_model.layers.{L}.mlp.gate_proj",
        ComponentType.MLP_OUTPUT: "model.language_model.layers.{L}.mlp.down_proj",
        ComponentType.FINAL_LOGITS: "lm_head",
    },
    # Standard causal LM (Llama, Gemma text-only, Pythia, etc.)
    "standard": {
        ComponentType.EMBEDDING: "model.embed_tokens",
        ComponentType.RESIDUAL_PRE: "model.layers.{L}",
        ComponentType.RESIDUAL_POST: "model.layers.{L}",
        ComponentType.ATTENTION_OUTPUT: "model.layers.{L}.self_attn.o_proj",
        ComponentType.MLP_GATE: "model.layers.{L}.mlp.gate_proj",
        ComponentType.MLP_OUTPUT: "model.layers.{L}.mlp.down_proj",
        ComponentType.FINAL_LOGITS: "lm_head",
    },
}


@dataclass(frozen=True)
class HookTarget:
    """Specifies exactly where to attach a hook in the model.

    Attributes:
        layer: Layer index (ignored for EMBEDDING and FINAL_LOGITS).
        component: Which component within the layer to hook.
        head: For attention, which head (None = all heads).
        token_position: Which sequence position (None = all).
        neuron_index: Which neuron in the hidden dim (None = all).
    """

    layer: int
    component: ComponentType
    head: int | None = None
    token_position: int | None = None
    neuron_index: int | None = None

    @property
    def is_pre_hook(self) -> bool:
        """Whether this target requires a pre-hook (input to module) vs output hook."""
        return self.component == ComponentType.RESIDUAL_PRE

    def to_module_name(self, module_map: dict[str, object]) -> str:
        """Resolve this target to an actual PyTorch module name.

        Tries architecture prefixes in order until one matches the module_map.
        """
        for prefix, templates in _MODULE_TEMPLATES.items():
            if self.component not in templates:
                continue
            template = templates[self.component]
            name = template.format(L=self.layer)
            if name in module_map:
                return name

        raise ValueError(
            f"Cannot resolve {self} to a module. "
            f"Available modules with 'layer': "
            f"{[n for n in module_map if 'layers' in n][:10]}..."
        )

    def to_key(self) -> str:
        """Unique string key for storing activations from this target."""
        parts = [f"L{self.layer}", self.component.value]
        if self.head is not None:
            parts.append(f"H{self.head}")
        if self.token_position is not None:
            parts.append(f"T{self.token_position}")
        if self.neuron_index is not None:
            parts.append(f"N{self.neuron_index}")
        return ".".join(parts)

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "component": self.component.value,
            "head": self.head,
            "token_position": self.token_position,
            "neuron_index": self.neuron_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> HookTarget:
        return cls(
            layer=d["layer"],
            component=ComponentType(d["component"]),
            head=d.get("head"),
            token_position=d.get("token_position"),
            neuron_index=d.get("neuron_index"),
        )
