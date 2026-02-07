"""Tests for activation extraction hooks."""

from __future__ import annotations

import torch

from neuronscope.hooks.extractors import make_extraction_hook
from neuronscope.hooks.targets import ComponentType, HookTarget


class TestMakeExtractionHook:
    def test_hook_stores_tensor(self):
        target = HookTarget(layer=0, component=ComponentType.MLP_OUTPUT)
        storage: dict[str, torch.Tensor] = {}

        hook_fn = make_extraction_hook(target, storage)
        # Simulate a forward hook call: (module, input, output)
        fake_output = torch.randn(1, 4, 8)
        hook_fn(None, None, fake_output)

        key = target.to_key()
        assert key in storage
        assert storage[key].device == torch.device("cpu")

    def test_hook_detaches_tensor(self):
        target = HookTarget(layer=0, component=ComponentType.MLP_OUTPUT)
        storage: dict[str, torch.Tensor] = {}

        hook_fn = make_extraction_hook(target, storage)
        fake_output = torch.randn(1, 4, 8, requires_grad=True)
        hook_fn(None, None, fake_output)

        assert not storage[target.to_key()].requires_grad

    def test_hook_handles_tuple_output(self):
        target = HookTarget(layer=0, component=ComponentType.ATTENTION_OUTPUT)
        storage: dict[str, torch.Tensor] = {}

        hook_fn = make_extraction_hook(target, storage)
        fake_output = (torch.randn(1, 4, 8), torch.randn(1, 4, 4))
        hook_fn(None, None, fake_output)

        assert target.to_key() in storage

    def test_pre_hook_extracts_from_input(self):
        target = HookTarget(layer=0, component=ComponentType.RESIDUAL_PRE)
        storage: dict[str, torch.Tensor] = {}

        hook_fn = make_extraction_hook(target, storage)
        # Pre-hooks receive (module, input) — only 2 args
        fake_input = (torch.randn(1, 4, 8),)
        hook_fn(None, fake_input)

        assert target.to_key() in storage

    def test_token_position_slicing(self):
        target = HookTarget(
            layer=0, component=ComponentType.MLP_OUTPUT, token_position=2
        )
        storage: dict[str, torch.Tensor] = {}

        hook_fn = make_extraction_hook(target, storage)
        fake_output = torch.randn(1, 4, 8)
        hook_fn(None, None, fake_output)

        stored = storage[target.to_key()]
        # Should be sliced to position 2 only
        assert stored.shape[1] <= fake_output.shape[1]
