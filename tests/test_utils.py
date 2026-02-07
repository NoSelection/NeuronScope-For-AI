"""Tests for utility modules: tensor_utils, token_utils, reports/utils."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuronscope.utils.tensor_utils import get_device, safe_cpu, tensor_stats
from neuronscope.utils.token_utils import tokenize_with_positions, get_answer_token_id
from neuronscope.reports.utils import safe_text


# ── safe_text ──

class TestSafeText:
    def test_ascii_passthrough(self):
        assert safe_text("hello world") == "hello world"

    def test_latin1_preserved(self):
        assert safe_text("caf\u00e9") == "caf\u00e9"  # e-acute

    def test_emoji_replaced(self):
        result = safe_text("hello \U0001f600 world")
        assert "\U0001f600" not in result
        assert "hello" in result

    def test_replacement_char_handled(self):
        result = safe_text("test\ufffd")
        assert "\ufffd" not in result

    def test_cjk_replaced(self):
        result = safe_text("\u4e16\u754c")  # Chinese for "world"
        assert "\u4e16" not in result

    def test_empty_string(self):
        assert safe_text("") == ""

    def test_result_is_latin1_encodable(self):
        nasty = "Hello \U0001f600 \u2014 \ufffd \u4e16\u754c end"
        result = safe_text(nasty)
        # Should not raise
        result.encode("latin-1")

    def test_mixed_content(self):
        result = safe_text('Token: "<b>" logit=5.2')
        assert "Token" in result
        assert "logit" in result


# ── tensor_stats ──

class TestTensorStats:
    def test_returns_all_keys(self):
        t = torch.randn(3, 4)
        stats = tensor_stats(t)
        for key in ("mean", "std", "min", "max", "norm", "shape"):
            assert key in stats

    def test_known_values(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        stats = tensor_stats(t)
        assert abs(stats["mean"] - 2.5) < 1e-5
        assert abs(stats["min"] - 1.0) < 1e-5
        assert abs(stats["max"] - 4.0) < 1e-5

    def test_shape_is_list(self):
        t = torch.randn(2, 3, 4)
        stats = tensor_stats(t)
        assert stats["shape"] == [2, 3, 4]

    def test_single_element(self):
        t = torch.tensor([5.0])
        stats = tensor_stats(t)
        assert abs(stats["mean"] - 5.0) < 1e-5


# ── get_device ──

class TestGetDevice:
    def test_cpu_model(self):
        model = nn.Linear(4, 4)
        device = get_device(model)
        assert device == torch.device("cpu")


# ── safe_cpu ──

class TestSafeCpu:
    def test_moves_to_cpu(self):
        t = torch.randn(3, 4)
        result = safe_cpu(t)
        assert result.device == torch.device("cpu")

    def test_detaches(self):
        t = torch.randn(3, 4, requires_grad=True)
        result = safe_cpu(t)
        assert not result.requires_grad

    def test_idempotent(self):
        t = torch.randn(3, 4)
        r1 = safe_cpu(t)
        r2 = safe_cpu(r1)
        assert torch.equal(r1, r2)


# ── tokenize_with_positions ──

class TestTokenizeWithPositions:
    def test_returns_list(self, mock_tokenizer):
        result = tokenize_with_positions("hello world", mock_tokenizer)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_has_required_fields(self, mock_tokenizer):
        result = tokenize_with_positions("hello world", mock_tokenizer)
        for item in result:
            assert "token_id" in item
            assert "position" in item


# ── get_answer_token_id ──

class TestGetAnswerTokenId:
    def test_returns_int(self, mock_tokenizer):
        token_id = get_answer_token_id("hello", mock_tokenizer)
        assert isinstance(token_id, int)
