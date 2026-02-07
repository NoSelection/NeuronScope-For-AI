"""Tests for HookManager lifecycle and cleanup guarantees."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from neuronscope.hooks.manager import HookManager
from neuronscope.hooks.targets import ComponentType, HookTarget


class _Layers(nn.Module):
    """Wraps a ModuleList so named_modules produces model.layers.0.mlp.down_proj"""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mlp": nn.ModuleDict({
                    "down_proj": nn.Linear(8, 8, bias=False),
                }),
                "self_attn": nn.ModuleDict({
                    "o_proj": nn.Linear(8, 8, bias=False),
                }),
            })
        ])


class TinyModel(nn.Module):
    """Minimal model whose module paths match the 'standard' template.

    Named modules include: model.layers.0.mlp.down_proj, etc.
    """

    def __init__(self):
        super().__init__()
        self.model = _Layers()

    def forward(self, x):
        for layer in self.model.layers:
            x = layer["mlp"]["down_proj"](x)
        return x


@pytest.fixture
def tiny_model():
    return TinyModel()


@pytest.fixture
def module_map(tiny_model):
    return {name: mod for name, mod in tiny_model.named_modules()}


@pytest.fixture
def hook_manager(tiny_model, module_map):
    return HookManager(tiny_model, module_map)


class TestHookManagerBasics:
    def test_initial_count_zero(self, hook_manager):
        assert hook_manager.active_hook_count == 0

    def test_session_context_manager(self, hook_manager):
        with hook_manager.session() as hm:
            assert hm is hook_manager

    def test_session_clears_on_exit(self, hook_manager):
        target = HookTarget(layer=0, component=ComponentType.MLP_OUTPUT)
        with hook_manager.session() as hm:
            storage = {}
            hm.attach_extraction_hook(target, storage)
            assert hm.active_hook_count > 0

        # After exiting context, hooks should be cleared
        assert hook_manager.active_hook_count == 0

    def test_session_clears_on_exception(self, hook_manager):
        target = HookTarget(layer=0, component=ComponentType.MLP_OUTPUT)
        with pytest.raises(RuntimeError):
            with hook_manager.session() as hm:
                storage = {}
                hm.attach_extraction_hook(target, storage)
                raise RuntimeError("test error")

        # Even after exception, hooks should be cleared
        assert hook_manager.active_hook_count == 0


class TestHookAttachment:
    def test_attach_extraction_hook(self, hook_manager):
        target = HookTarget(layer=0, component=ComponentType.MLP_OUTPUT)
        storage = {}
        with hook_manager.session() as hm:
            hm.attach_extraction_hook(target, storage)
            assert hm.active_hook_count == 1

    def test_attach_multiple_hooks(self, hook_manager):
        targets = [
            HookTarget(layer=0, component=ComponentType.MLP_OUTPUT),
            HookTarget(layer=0, component=ComponentType.ATTENTION_OUTPUT),
        ]
        storage = {}
        with hook_manager.session() as hm:
            for t in targets:
                hm.attach_extraction_hook(t, storage)
            assert hm.active_hook_count == 2

    def test_clear_all(self, hook_manager):
        target = HookTarget(layer=0, component=ComponentType.MLP_OUTPUT)
        storage = {}
        hook_manager.attach_extraction_hook(target, storage)
        assert hook_manager.active_hook_count > 0
        hook_manager.clear_all()
        assert hook_manager.active_hook_count == 0


class TestHookExecution:
    def test_extraction_captures_activation(self, hook_manager, tiny_model):
        target = HookTarget(layer=0, component=ComponentType.MLP_OUTPUT)
        storage = {}

        with hook_manager.session() as hm:
            hm.attach_extraction_hook(target, storage)
            x = torch.randn(1, 4, 8)
            tiny_model(x)

        assert len(storage) > 0
        key = target.to_key()
        assert key in storage
        assert isinstance(storage[key], torch.Tensor)
