"""Tests for output comparison and metrics computation."""

from __future__ import annotations

import pytest
import torch

from neuronscope.experiments.comparator import (
    kl_divergence,
    logit_diff,
    top_k_predictions,
    top_k_overlap,
    rank_changes,
)


class TestKLDivergence:
    def test_identical_distributions(self, identical_logits):
        clean, interv = identical_logits
        kl = kl_divergence(clean, interv)
        assert abs(kl) < 1e-5

    def test_different_distributions(self, logits_pair):
        clean, interv = logits_pair
        kl = kl_divergence(clean, interv)
        assert kl > 0

    def test_returns_float(self, logits_pair):
        clean, interv = logits_pair
        kl = kl_divergence(clean, interv)
        assert isinstance(kl, float)

    def test_larger_divergence_means_more_different(self):
        base = torch.zeros(1, 1, 10)
        base[0, 0, 0] = 10.0

        # Slightly different
        slight = base.clone()
        slight[0, 0, 0] = 9.0
        slight[0, 0, 1] = 1.0

        # Very different
        very = torch.zeros(1, 1, 10)
        very[0, 0, 5] = 10.0

        kl_slight = kl_divergence(base, slight)
        kl_very = kl_divergence(base, very)
        assert kl_very > kl_slight


class TestLogitDiff:
    def test_basic(self):
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 3] = 5.0  # target
        logits[0, 0, 7] = 2.0  # foil
        diff = logit_diff(logits, target_id=3, foil_id=7)
        assert abs(diff - 3.0) < 1e-5

    def test_negative_diff(self):
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 3] = 2.0
        logits[0, 0, 7] = 5.0
        diff = logit_diff(logits, target_id=3, foil_id=7)
        assert diff < 0


class TestTopKPredictions:
    def test_returns_k_items(self, mock_tokenizer):
        logits = torch.randn(1, 1, 12)
        preds = top_k_predictions(logits, mock_tokenizer, k=5)
        assert len(preds) == 5

    def test_sorted_by_probability(self, mock_tokenizer):
        logits = torch.randn(1, 1, 12)
        preds = top_k_predictions(logits, mock_tokenizer, k=5)
        probs = [p.prob for p in preds]
        assert probs == sorted(probs, reverse=True)

    def test_probabilities_sum_roughly_to_one(self, mock_tokenizer):
        logits = torch.randn(1, 1, 12)
        preds = top_k_predictions(logits, mock_tokenizer, k=12)
        total = sum(p.prob for p in preds)
        assert abs(total - 1.0) < 0.01


class TestTopKOverlap:
    def test_identical_perfect_overlap(self, identical_logits):
        clean, interv = identical_logits
        overlap = top_k_overlap(clean, interv, k=5)
        assert abs(overlap - 1.0) < 1e-5

    def test_different_lower_overlap(self, logits_pair):
        clean, interv = logits_pair
        overlap = top_k_overlap(clean, interv, k=5)
        assert 0.0 <= overlap <= 1.0

    def test_returns_float(self, logits_pair):
        clean, interv = logits_pair
        overlap = top_k_overlap(clean, interv, k=5)
        assert isinstance(overlap, float)


class TestRankChanges:
    def test_returns_dict(self, logits_pair, mock_tokenizer):
        clean, interv = logits_pair
        changes = rank_changes(clean, interv, mock_tokenizer, k=5)
        assert isinstance(changes, dict)

    def test_tracks_specified_tokens(self, logits_pair, mock_tokenizer):
        clean, interv = logits_pair
        changes = rank_changes(clean, interv, mock_tokenizer, track_token_ids=[0, 1], k=5)
        # Should have entries for tracked tokens
        assert len(changes) > 0

    def test_rank_delta_present(self, logits_pair, mock_tokenizer):
        clean, interv = logits_pair
        changes = rank_changes(clean, interv, mock_tokenizer, k=5)
        for token_str, info in changes.items():
            assert "rank_delta" in info
            assert "clean_rank" in info
            assert "intervention_rank" in info
