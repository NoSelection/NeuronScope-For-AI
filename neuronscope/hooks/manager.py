from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from neuronscope.hooks.targets import HookTarget
from neuronscope.hooks.extractors import make_extraction_hook
from neuronscope.hooks.interventions import Intervention, make_intervention_hook


class HookManager:
    """Manages PyTorch hooks with clean lifecycle guarantees.

    Key invariant: After a context manager exits or clear_all() is called,
    there are ZERO hooks attached to the model. Leaked hooks corrupt
    all subsequent runs — this is the single most dangerous failure mode.
    """

    def __init__(self, model: nn.Module, module_map: dict[str, nn.Module]):
        self._model = model
        self._module_map = module_map
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _get_module(self, target: HookTarget) -> nn.Module:
        """Resolve a HookTarget to the actual PyTorch module."""
        name = target.to_module_name(self._module_map)
        return self._module_map[name]

    def attach_extraction_hook(
        self,
        target: HookTarget,
        storage: dict[str, torch.Tensor],
    ) -> torch.utils.hooks.RemovableHandle:
        """Attach a read-only hook that captures activations into storage."""
        module = self._get_module(target)
        hook_fn = make_extraction_hook(target, storage)

        if target.is_pre_hook:
            handle = module.register_forward_pre_hook(hook_fn)
        else:
            handle = module.register_forward_hook(hook_fn)

        self._handles.append(handle)
        return handle

    def attach_intervention_hook(
        self,
        target: HookTarget,
        intervention: Intervention,
    ) -> torch.utils.hooks.RemovableHandle:
        """Attach a hook that modifies activations during forward pass."""
        module = self._get_module(target)
        hook_fn = make_intervention_hook(target, intervention)

        if target.is_pre_hook:
            handle = module.register_forward_pre_hook(hook_fn)
        else:
            handle = module.register_forward_hook(hook_fn)

        self._handles.append(handle)
        return handle

    def clear_all(self) -> None:
        """Remove all hooks. MUST be called after every experiment run."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def session(self):
        """Context manager that guarantees hook cleanup.

        Usage:
            with hook_manager.session() as hm:
                hm.attach_extraction_hook(target, storage)
                model(input_ids)
            # All hooks removed here, guaranteed.
        """
        try:
            yield self
        finally:
            self.clear_all()

    @property
    def active_hook_count(self) -> int:
        return len(self._handles)
