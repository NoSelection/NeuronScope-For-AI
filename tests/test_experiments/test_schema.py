"""Tests for experiment configuration and result schemas."""

from __future__ import annotations

import pytest

from neuronscope.experiments.schema import (
    ExperimentConfig,
    ExperimentResult,
    InterventionSpec,
    TokenPrediction,
)


class TestInterventionSpec:
    def test_creation(self):
        spec = InterventionSpec(
            target_layer=17,
            target_component="mlp_output",
            intervention_type="zero",
        )
        assert spec.target_layer == 17
        assert spec.target_head is None

    def test_optional_fields(self):
        spec = InterventionSpec(
            target_layer=5,
            target_component="attn_output",
            target_head=3,
            target_position=2,
            intervention_type="patch",
        )
        assert spec.target_head == 3
        assert spec.target_position == 2


class TestExperimentConfig:
    def test_defaults(self):
        config = ExperimentConfig(
            name="test",
            base_input="Hello",
            interventions=[],
        )
        assert config.seed == 42
        assert config.max_new_tokens == 1
        assert config.temperature == 0.0
        assert config.source_input is None

    def test_config_hash_deterministic(self):
        config = ExperimentConfig(
            name="test",
            base_input="The Eiffel Tower is in",
            interventions=[
                InterventionSpec(
                    target_layer=17,
                    target_component="mlp_output",
                    intervention_type="zero",
                )
            ],
        )
        h1 = config.config_hash()
        h2 = config.config_hash()
        assert h1 == h2
        assert len(h1) == 16  # sha256 first 16 hex chars

    def test_config_hash_changes_with_input(self):
        base = dict(
            name="test",
            interventions=[
                InterventionSpec(
                    target_layer=17,
                    target_component="mlp_output",
                    intervention_type="zero",
                )
            ],
        )
        c1 = ExperimentConfig(base_input="Hello", **base)
        c2 = ExperimentConfig(base_input="World", **base)
        assert c1.config_hash() != c2.config_hash()

    def test_config_hash_changes_with_layer(self):
        base = dict(name="test", base_input="Hello")
        c1 = ExperimentConfig(
            interventions=[InterventionSpec(target_layer=0, target_component="mlp_output", intervention_type="zero")],
            **base,
        )
        c2 = ExperimentConfig(
            interventions=[InterventionSpec(target_layer=1, target_component="mlp_output", intervention_type="zero")],
            **base,
        )
        assert c1.config_hash() != c2.config_hash()

    def test_config_hash_changes_with_seed(self):
        base = dict(
            name="test",
            base_input="Hello",
            interventions=[
                InterventionSpec(target_layer=0, target_component="mlp_output", intervention_type="zero")
            ],
        )
        c1 = ExperimentConfig(seed=42, **base)
        c2 = ExperimentConfig(seed=99, **base)
        assert c1.config_hash() != c2.config_hash()

    def test_serialization_roundtrip(self):
        config = ExperimentConfig(
            name="test",
            base_input="Hello",
            interventions=[
                InterventionSpec(
                    target_layer=17,
                    target_component="mlp_output",
                    intervention_type="zero",
                )
            ],
        )
        data = config.model_dump()
        restored = ExperimentConfig.model_validate(data)
        assert restored.config_hash() == config.config_hash()


class TestTokenPrediction:
    def test_creation(self):
        tp = TokenPrediction(token="Paris", token_id=7, logit=10.5, prob=0.85)
        assert tp.token == "Paris"
        assert tp.prob == 0.85


class TestExperimentResult:
    def test_creation(self, sample_result):
        assert sample_result.id == "test-001"
        assert sample_result.kl_divergence == 0.5

    def test_timestamp_auto_generated(self, sample_result):
        assert sample_result.timestamp  # non-empty

    def test_serialization_roundtrip(self, sample_result):
        data = sample_result.model_dump()
        restored = ExperimentResult.model_validate(data)
        assert restored.id == sample_result.id
        assert restored.kl_divergence == sample_result.kl_divergence
        assert restored.config.config_hash() == sample_result.config.config_hash()
