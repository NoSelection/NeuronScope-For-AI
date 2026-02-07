"""Tests for causal attribution module."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuronscope.analysis.attribution import attribute_by_layer, AttributionResult
from neuronscope.hooks.manager import HookManager


# ── Minimal mock model for attribution tests ──

class MockConfig:
    class text_config:
        num_hidden_layers = 4


class MockMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)))


class MockSelfAttn(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.o_proj(x)


class MockLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = MockSelfAttn(hidden_size)
        self.mlp = MockMLP(hidden_size)

    def forward(self, x):
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class MockInnerModel(nn.Module):
    """Inner model that matches the 'standard' template: model.layers.{L}, model.embed_tokens."""
    def __init__(self, num_layers: int = 4, hidden_size: int = 32, vocab_size: int = 12):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([MockLayer(hidden_size) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MockModel(nn.Module):
    """Mock model with module paths matching the 'standard' template in targets.py."""
    def __init__(self, num_layers: int = 4, hidden_size: int = 32, vocab_size: int = 12):
        super().__init__()
        self.model = MockInnerModel(num_layers, hidden_size, vocab_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = MockConfig()

    def forward(self, input_ids):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return type("Output", (), {"logits": logits})()


class MockTokenizer:
    def __init__(self):
        self._vocab = {
            "hello": 0, "world": 1, "the": 2, "cat": 3,
            "sat": 4, "on": 5, "mat": 6, "Paris": 7,
        }

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [self._vocab.get(c, 0) for c in text.split()]
        return type("Encoding", (), {"input_ids": torch.tensor([ids])})()


@pytest.fixture
def mock_model():
    model = MockModel(num_layers=4, hidden_size=32, vocab_size=12)
    model.eval()
    return model


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def module_map(mock_model):
    return dict(mock_model.named_modules())


@pytest.fixture
def hook_manager(mock_model, module_map):
    return HookManager(mock_model, module_map)


class TestAttributionResult:
    def test_to_dict(self):
        result = AttributionResult(
            layers=[0, 1, 2],
            kl_scores=[0.1, 0.5, 0.2],
            effect_sizes=[0.2, 1.0, 0.4],
            peak_layer=1,
            peak_kl=0.5,
            total_effect=0.8,
            component="mlp_output",
            intervention_type="zero",
        )
        d = result.to_dict()
        assert d["peak_layer"] == 1
        assert d["peak_kl"] == 0.5
        assert len(d["layers"]) == 3
        assert d["component"] == "mlp_output"

    def test_effect_sizes_normalized(self):
        result = AttributionResult(
            layers=[0, 1],
            kl_scores=[0.3, 0.6],
            effect_sizes=[0.5, 1.0],
            peak_layer=1,
            peak_kl=0.6,
            total_effect=0.9,
            component="attn_output",
            intervention_type="zero",
        )
        assert max(result.effect_sizes) == 1.0


class TestAttributeByLayer:
    def test_returns_attribution_result(self, mock_model, mock_tokenizer, hook_manager):
        """attribute_by_layer should return an AttributionResult with correct structure."""
        # Use standard module paths since our mock uses model.language_model.layers.*
        result = attribute_by_layer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            hook_manager=hook_manager,
            base_input="hello world",
            intervention_type="zero",
            component="mlp_output",
            layers=[0, 1, 2, 3],
        )
        assert isinstance(result, AttributionResult)
        assert len(result.layers) == 4
        assert len(result.kl_scores) == 4
        assert len(result.effect_sizes) == 4
        assert result.peak_layer in [0, 1, 2, 3]
        assert result.peak_kl >= 0.0

    def test_all_kl_scores_non_negative(self, mock_model, mock_tokenizer, hook_manager):
        result = attribute_by_layer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            hook_manager=hook_manager,
            base_input="the cat sat",
            intervention_type="zero",
            component="mlp_output",
            layers=[0, 1],
        )
        for kl in result.kl_scores:
            assert kl >= 0.0

    def test_effect_sizes_max_is_one(self, mock_model, mock_tokenizer, hook_manager):
        result = attribute_by_layer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            hook_manager=hook_manager,
            base_input="hello world",
            intervention_type="zero",
            component="mlp_output",
            layers=[0, 1, 2, 3],
        )
        # If any KL is nonzero, max effect size should be 1.0
        if result.peak_kl > 1e-8:
            assert abs(max(result.effect_sizes) - 1.0) < 1e-5

    def test_single_layer(self, mock_model, mock_tokenizer, hook_manager):
        result = attribute_by_layer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            hook_manager=hook_manager,
            base_input="hello",
            intervention_type="zero",
            component="mlp_output",
            layers=[2],
        )
        assert len(result.layers) == 1
        assert result.layers[0] == 2
        assert result.peak_layer == 2

    def test_hooks_cleaned_up(self, mock_model, mock_tokenizer, hook_manager):
        """After attribution, no hooks should remain on the model."""
        attribute_by_layer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            hook_manager=hook_manager,
            base_input="hello world",
            intervention_type="zero",
            component="mlp_output",
            layers=[0, 1],
        )
        assert hook_manager.active_hook_count == 0

    def test_invalid_intervention_type(self, mock_model, mock_tokenizer, hook_manager):
        with pytest.raises(ValueError, match="Unsupported intervention type"):
            attribute_by_layer(
                model=mock_model,
                tokenizer=mock_tokenizer,
                hook_manager=hook_manager,
                base_input="hello",
                intervention_type="patch",
                component="mlp_output",
                layers=[0],
            )

    def test_total_effect_is_sum(self, mock_model, mock_tokenizer, hook_manager):
        result = attribute_by_layer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            hook_manager=hook_manager,
            base_input="hello world",
            intervention_type="zero",
            component="mlp_output",
            layers=[0, 1, 2],
        )
        assert abs(result.total_effect - sum(result.kl_scores)) < 1e-5
