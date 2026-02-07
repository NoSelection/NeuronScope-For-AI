from __future__ import annotations

import torch
from neuronscope.hooks.targets import HookTarget


def make_extraction_hook(
    target: HookTarget,
    storage: dict[str, torch.Tensor],
) -> callable:
    """Create a hook function that captures activations into storage.

    The captured tensor is detached and moved to CPU immediately
    to minimize GPU memory pressure.
    """
    key = target.to_key()

    def hook_fn(module, input, output):
        # For pre-hooks, output is actually the input tuple
        tensor = output
        if isinstance(tensor, tuple):
            tensor = tensor[0]

        # Slice to the specific dimensions requested
        tensor = _slice_tensor(tensor, target)

        storage[key] = tensor.detach().cpu()

    def pre_hook_fn(module, input):
        tensor = input
        if isinstance(tensor, tuple):
            tensor = tensor[0]

        tensor = _slice_tensor(tensor, target)
        storage[key] = tensor.detach().cpu()

    return pre_hook_fn if target.is_pre_hook else hook_fn


def _slice_tensor(tensor: torch.Tensor, target: HookTarget) -> torch.Tensor:
    """Slice a tensor according to the target's specificity.

    Handles token_position, head, and neuron_index slicing.
    Assumes tensor shape is (batch, seq_len, hidden) or (batch, heads, seq_len, seq_len).
    """
    if tensor.dim() < 2:
        return tensor

    if target.token_position is not None and tensor.dim() >= 2:
        tensor = tensor[:, target.token_position : target.token_position + 1]

    if target.neuron_index is not None and tensor.dim() >= 3:
        tensor = tensor[..., target.neuron_index : target.neuron_index + 1]

    return tensor
