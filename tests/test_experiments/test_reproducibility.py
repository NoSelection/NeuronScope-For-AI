"""Tests for reproducibility controls."""

from __future__ import annotations

import random

import numpy as np
import torch

from neuronscope.experiments.reproducibility import set_all_seeds, set_deterministic


class TestSetAllSeeds:
    def test_deterministic_random(self):
        set_all_seeds(42)
        a = random.random()
        set_all_seeds(42)
        b = random.random()
        assert a == b

    def test_deterministic_numpy(self):
        set_all_seeds(42)
        a = np.random.rand()
        set_all_seeds(42)
        b = np.random.rand()
        assert a == b

    def test_deterministic_torch(self):
        set_all_seeds(42)
        a = torch.randn(5)
        set_all_seeds(42)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_different_seeds_different_results(self):
        set_all_seeds(42)
        a = torch.randn(5)
        set_all_seeds(99)
        b = torch.randn(5)
        assert not torch.equal(a, b)


class TestSetDeterministic:
    def test_enable_deterministic(self):
        set_deterministic(True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_disable_deterministic(self):
        set_deterministic(False)
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True

    def test_does_not_raise(self):
        # Should not raise even without CUDA
        set_deterministic(True)
        set_deterministic(False)
