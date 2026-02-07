"""Shared fixtures for NeuronScope tests.

All fixtures are GPU-free — tests run on CPU with synthetic data.
"""

from __future__ import annotations

import pytest
import torch

from neuronscope.experiments.schema import (
    ExperimentConfig,
    ExperimentResult,
    InterventionSpec,
    TokenPrediction,
)
from neuronscope.hooks.targets import ComponentType, HookTarget


# ── Mock tokenizer ──

class MockTokenizer:
    """Minimal tokenizer mock for testing."""

    def __init__(self):
        self._vocab = {
            "hello": 0, "world": 1, "the": 2, "cat": 3,
            "sat": 4, "on": 5, "mat": 6, "Paris": 7,
            "France": 8, "Eiffel": 9, " ": 10, ".": 11,
        }
        self._id_to_token = {v: k for k, v in self._vocab.items()}

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [self._vocab.get(c, 10) for c in text.split()]
        result = type("Encoding", (), {
            "input_ids": torch.tensor([ids]),
        })()
        return result

    def decode(self, ids, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        return " ".join(self._id_to_token.get(i, "?") for i in ids)

    def encode(self, text, add_special_tokens=False, **kwargs):
        tokens = text.split()
        return [self._vocab.get(t, 10) for t in tokens]


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


# ── Tensor fixtures ──

@pytest.fixture
def logits_pair():
    """A pair of logit tensors with known distributions.

    clean:  token 0 is dominant (logit=10)
    interv: token 1 is dominant (logit=10)
    Shape: [1, 1, 12] (batch=1, seq=1, vocab=12)
    """
    clean = torch.zeros(1, 1, 12)
    clean[0, 0, 0] = 10.0  # token 0 dominant

    intervention = torch.zeros(1, 1, 12)
    intervention[0, 0, 1] = 10.0  # token 1 dominant

    return clean, intervention


@pytest.fixture
def identical_logits():
    """Two identical logit tensors (KL should be ~0)."""
    logits = torch.randn(1, 1, 12)
    return logits.clone(), logits.clone()


# ── HookTarget fixtures ──

@pytest.fixture
def mlp_target():
    return HookTarget(layer=17, component=ComponentType.MLP_OUTPUT)


@pytest.fixture
def attn_target():
    return HookTarget(
        layer=5,
        component=ComponentType.ATTENTION_OUTPUT,
        head=3,
        token_position=2,
    )


# ── ExperimentResult factory ──

def make_result(
    kl: float = 0.5,
    changed: bool = False,
    clean_token: str = "Paris",
    interv_token: str = "Paris",
    clean_prob: float = 0.8,
    interv_prob: float = 0.7,
    layer: int = 17,
    intervention_type: str = "zero",
    rank_changes: dict | None = None,
) -> ExperimentResult:
    """Create a minimal ExperimentResult for testing."""
    return ExperimentResult(
        id="test-001",
        config=ExperimentConfig(
            name="test",
            base_input="The Eiffel Tower is in",
            interventions=[
                InterventionSpec(
                    target_layer=layer,
                    target_component="mlp_output",
                    intervention_type=intervention_type,
                )
            ],
        ),
        config_hash="abc123",
        clean_top_k=[
            TokenPrediction(token=clean_token, token_id=0, logit=10.0, prob=clean_prob),
        ],
        clean_output_token=clean_token,
        clean_output_prob=clean_prob,
        intervention_top_k=[
            TokenPrediction(token=interv_token, token_id=1, logit=9.0, prob=interv_prob),
        ],
        intervention_output_token=interv_token,
        intervention_output_prob=interv_prob,
        kl_divergence=kl,
        top_token_changed=changed,
        rank_changes=rank_changes or {},
        duration_seconds=1.5,
        device="cpu",
    )


@pytest.fixture
def sample_result():
    return make_result()


@pytest.fixture
def critical_result():
    """Result with critical metrics — output flipped, high KL."""
    return make_result(
        kl=16.5,
        changed=True,
        clean_token="Paris",
        interv_token="France",
        clean_prob=0.85,
        interv_prob=0.20,
        rank_changes={
            "icosa": {"clean_rank": 17, "intervention_rank": 25629, "rank_delta": 25612},
        },
    )
