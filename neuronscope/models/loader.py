from __future__ import annotations

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from rich.console import Console

from neuronscope.models.schema import ModelInfo

console = Console()


class ModelLoader:
    """Load a HuggingFace model with memory-conscious defaults for local GPU."""

    @staticmethod
    def load(
        model_path: str = "LLM",
        device: str = "cuda",
    ) -> tuple[torch.nn.Module, AutoTokenizer, dict[str, torch.nn.Module], ModelInfo]:
        """Load model, tokenizer, module map, and model info.

        Returns:
            model: The loaded model in eval mode.
            tokenizer: The tokenizer.
            module_map: Dict mapping module names to modules (for hook targeting).
            info: ModelInfo with architecture metadata.
        """
        path = Path(model_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / model_path

        console.print(f"[bold]Loading model from {path}...[/bold]")

        config = AutoConfig.from_pretrained(path)

        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device if device == "auto" else {"": device},
            low_cpu_mem_usage=True,
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(path)

        module_map = {name: mod for name, mod in model.named_modules()}

        text_config = getattr(config, "text_config", config)
        info = ModelInfo(
            name=path.name,
            path=str(path),
            architecture=config.architectures[0] if config.architectures else "unknown",
            num_layers=getattr(text_config, "num_hidden_layers", 0),
            hidden_size=getattr(text_config, "hidden_size", 0),
            intermediate_size=getattr(text_config, "intermediate_size", 0),
            num_attention_heads=getattr(text_config, "num_attention_heads", None),
            num_key_value_heads=getattr(text_config, "num_key_value_heads", None),
            vocab_size=getattr(config, "vocab_size", getattr(text_config, "vocab_size", 0)),
            dtype=str(config.torch_dtype) if hasattr(config, "torch_dtype") else "unknown",
            device=device,
            sliding_window=getattr(text_config, "sliding_window", None),
            has_vision=hasattr(config, "vision_config"),
            module_names=list(module_map.keys()),
        )

        console.print(
            f"[green]Model loaded:[/green] {info.architecture}, "
            f"{info.num_layers} layers, hidden_size={info.hidden_size}"
        )

        return model, tokenizer, module_map, info
