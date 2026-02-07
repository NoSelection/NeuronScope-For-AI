"""Tests for intervention types and their tensor operations."""

from __future__ import annotations

import pytest
import torch

from neuronscope.hooks.interventions import (
    ActivationPatching,
    AdditivePerturbation,
    MeanAblation,
    ZeroAblation,
)
from neuronscope.hooks.targets import ComponentType, HookTarget


@pytest.fixture
def activation():
    """3D activation tensor: [batch=1, seq=4, hidden=8]."""
    torch.manual_seed(42)
    return torch.randn(1, 4, 8)


@pytest.fixture
def target():
    return HookTarget(layer=0, component=ComponentType.MLP_OUTPUT)


@pytest.fixture
def target_with_neuron():
    return HookTarget(layer=0, component=ComponentType.MLP_OUTPUT, neuron_index=3)


@pytest.fixture
def target_with_position():
    return HookTarget(layer=0, component=ComponentType.MLP_OUTPUT, token_position=2)


class TestZeroAblation:
    def test_full_zero(self, activation, target):
        ablation = ZeroAblation()
        result = ablation.apply(activation, target)
        assert torch.all(result == 0)

    def test_does_not_modify_original(self, activation, target):
        original = activation.clone()
        ZeroAblation().apply(activation, target)
        assert torch.equal(activation, original)

    def test_neuron_specific(self, activation, target_with_neuron):
        result = ZeroAblation().apply(activation, target_with_neuron)
        # Neuron 3 should be zero
        assert torch.all(result[..., 3] == 0)
        # Other neurons should be unchanged
        assert torch.all(result[..., 0] == activation[..., 0])

    def test_position_specific(self, activation, target_with_position):
        result = ZeroAblation().apply(activation, target_with_position)
        # Position 2 should be zero
        assert torch.all(result[:, 2] == 0)
        # Other positions should be unchanged
        assert torch.all(result[:, 0] == activation[:, 0])

    def test_serialization_roundtrip(self):
        ablation = ZeroAblation()
        d = ablation.to_dict()
        assert d["type"] == "zero"


class TestMeanAblation:
    def test_full_replacement(self, activation, target):
        mean = torch.ones_like(activation) * 5.0
        result = MeanAblation(mean).apply(activation, target)
        assert torch.allclose(result, mean)

    def test_neuron_specific(self, activation, target_with_neuron):
        mean = torch.ones_like(activation) * 5.0
        result = MeanAblation(mean).apply(activation, target_with_neuron)
        # Neuron 3 should be 5.0
        assert torch.allclose(result[..., 3], torch.tensor(5.0))
        # Other neurons unchanged
        assert torch.allclose(result[..., 0], activation[..., 0])

    def test_does_not_modify_original(self, activation, target):
        original = activation.clone()
        MeanAblation(torch.zeros_like(activation)).apply(activation, target)
        assert torch.equal(activation, original)


class TestActivationPatching:
    def test_full_patch(self, activation, target):
        source = torch.ones_like(activation) * 3.0
        result = ActivationPatching(source).apply(activation, target)
        # Should match source (full replacement)
        assert torch.allclose(result, source, atol=1e-5)

    def test_position_specific_patch(self, activation, target_with_position):
        source = torch.ones_like(activation) * 3.0
        result = ActivationPatching(source).apply(activation, target_with_position)
        # Position 2 should be patched
        assert torch.allclose(result[:, 2], source[:, 2])
        # Position 0 should be unchanged
        assert torch.allclose(result[:, 0], activation[:, 0])

    def test_neuron_specific_patch(self, activation, target_with_neuron):
        source = torch.ones_like(activation) * 3.0
        result = ActivationPatching(source).apply(activation, target_with_neuron)
        # Neuron 3 should be patched
        assert torch.allclose(result[..., 3], source[..., 3])
        # Neuron 0 should be unchanged
        assert torch.allclose(result[..., 0], activation[..., 0])

    def test_does_not_modify_original(self, activation, target):
        original = activation.clone()
        source = torch.ones_like(activation)
        ActivationPatching(source).apply(activation, target)
        assert torch.equal(activation, original)

    def test_seq_length_mismatch(self, target_with_position):
        """Source shorter than activation — should handle gracefully."""
        activation = torch.randn(1, 4, 8)
        source = torch.randn(1, 2, 8)  # shorter
        # Position 2 is out of bounds for source (len 2)
        result = ActivationPatching(source).apply(activation, target_with_position)
        # Should still return a valid tensor
        assert result.shape == activation.shape


class TestAdditivePerturbation:
    def test_full_perturbation(self, activation, target):
        # Direction is a hidden-size vector [8] — broadcasts across batch & seq
        direction = torch.ones(8)
        result = AdditivePerturbation(direction, magnitude=2.0).apply(activation, target)
        expected = activation + direction * 2.0
        assert torch.allclose(result, expected)

    def test_position_specific(self, activation, target_with_position):
        # Direction is a hidden-size vector — only added at token_position
        direction = torch.ones(8)
        result = AdditivePerturbation(direction, magnitude=1.0).apply(activation, target_with_position)
        # Position 2 should be perturbed
        expected_pos2 = activation[:, 2] + direction * 1.0
        assert torch.allclose(result[:, 2], expected_pos2, atol=1e-5)
        # Position 0 should be unchanged
        assert torch.allclose(result[:, 0], activation[:, 0])

    def test_zero_magnitude(self, activation, target):
        direction = torch.ones(8)
        result = AdditivePerturbation(direction, magnitude=0.0).apply(activation, target)
        assert torch.allclose(result, activation)

    def test_does_not_modify_original(self, activation, target):
        original = activation.clone()
        AdditivePerturbation(torch.ones(8)).apply(activation, target)
        assert torch.equal(activation, original)
