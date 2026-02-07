"""Tests for analysis metrics module."""

from __future__ import annotations

import pytest
import torch

from neuronscope.analysis.metrics import (
    kl_divergence,
    js_divergence,
    logit_diff,
    effect_size,
    top_k_overlap,
    probability_change,
    activation_norm_change,
)


class TestKLDivergence:
    def test_identical_is_zero(self, identical_logits):
        clean, interv = identical_logits
        assert abs(kl_divergence(clean, interv)) < 1e-5

    def test_different_is_positive(self, logits_pair):
        clean, interv = logits_pair
        assert kl_divergence(clean, interv) > 0


class TestJSDivergence:
    def test_identical_is_zero(self, identical_logits):
        clean, interv = identical_logits
        assert abs(js_divergence(clean, interv)) < 1e-5

    def test_symmetric(self, logits_pair):
        clean, interv = logits_pair
        js_ab = js_divergence(clean, interv)
        js_ba = js_divergence(interv, clean)
        assert abs(js_ab - js_ba) < 1e-5

    def test_different_is_positive(self, logits_pair):
        clean, interv = logits_pair
        assert js_divergence(clean, interv) > 0


class TestLogitDiff:
    def test_positive_when_target_higher(self):
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 3] = 8.0
        logits[0, 0, 5] = 2.0
        assert logit_diff(logits, 3, 5) > 0

    def test_negative_when_foil_higher(self):
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 3] = 2.0
        logits[0, 0, 5] = 8.0
        assert logit_diff(logits, 3, 5) < 0


class TestEffectSize:
    def test_basic(self):
        # clean=10, intervention=5, total range=10
        es = effect_size(10.0, 5.0, 10.0)
        assert abs(es - 0.5) < 1e-5

    def test_zero_total(self):
        # total near zero should return 0.0 safely
        es = effect_size(1.0, 0.5, 0.0)
        assert es == 0.0

    def test_no_change(self):
        es = effect_size(5.0, 5.0, 10.0)
        assert abs(es) < 1e-5


class TestTopKOverlap:
    def test_identical(self, identical_logits):
        clean, interv = identical_logits
        assert abs(top_k_overlap(clean, interv, k=5) - 1.0) < 1e-5

    def test_range(self, logits_pair):
        clean, interv = logits_pair
        overlap = top_k_overlap(clean, interv, k=5)
        assert 0.0 <= overlap <= 1.0


class TestProbabilityChange:
    def test_returns_dict(self, logits_pair):
        clean, interv = logits_pair
        result = probability_change(clean, interv, token_id=0)
        assert "absolute_change" in result
        assert "relative_change" in result

    def test_no_change_when_identical(self, identical_logits):
        clean, interv = identical_logits
        result = probability_change(clean, interv, token_id=0)
        assert abs(result["absolute_change"]) < 1e-5


class TestActivationNormChange:
    def test_returns_dict(self):
        a = torch.randn(1, 4, 8)
        b = torch.randn(1, 4, 8)
        result = activation_norm_change(a, b)
        assert "clean_norm" in result
        assert "diff_norm" in result
        assert "cosine_similarity" in result

    def test_identical_activations(self):
        a = torch.randn(1, 4, 8)
        result = activation_norm_change(a, a.clone())
        assert abs(result["diff_norm"]) < 1e-5
        assert abs(result["cosine_similarity"] - 1.0) < 1e-5
