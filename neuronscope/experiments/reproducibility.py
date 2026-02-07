from __future__ import annotations

import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Must be called before every forward pass to ensure identical
    results across experiment runs with the same config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_deterministic(enabled: bool = True) -> None:
    """Enable deterministic CUDA operations.

    Trades performance for bitwise reproducibility.
    """
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(enabled)
        except RuntimeError:
            # Some ops don't have deterministic implementations
            pass
