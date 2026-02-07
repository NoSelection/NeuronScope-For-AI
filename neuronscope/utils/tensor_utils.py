from __future__ import annotations

import torch


def get_device(model: torch.nn.Module) -> torch.device:
    """Get the device of a model's first parameter."""
    return next(model.parameters()).device


def to_device(tensor: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Move a tensor to the same device as a model."""
    return tensor.to(get_device(model))


def safe_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Detach and move to CPU. Safe to call on CPU tensors."""
    return tensor.detach().cpu()


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    """Quick summary statistics for a tensor."""
    t = tensor.float()
    return {
        "mean": t.mean().item(),
        "std": t.std().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "norm": t.norm().item(),
        "shape": list(tensor.shape),
    }
