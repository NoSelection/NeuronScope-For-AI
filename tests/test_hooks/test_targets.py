"""Tests for hook target specification and module resolution."""

from __future__ import annotations

import pytest

from neuronscope.hooks.targets import ComponentType, HookTarget


class TestComponentType:
    def test_all_components_exist(self):
        expected = {
            "embedding", "residual_pre", "residual_post",
            "attn_output", "attn_pattern", "mlp_gate",
            "mlp_output", "final_logits",
        }
        actual = {c.value for c in ComponentType}
        assert actual == expected

    def test_component_from_string(self):
        assert ComponentType("mlp_output") == ComponentType.MLP_OUTPUT
        assert ComponentType("attn_output") == ComponentType.ATTENTION_OUTPUT


class TestHookTarget:
    def test_creation(self, mlp_target):
        assert mlp_target.layer == 17
        assert mlp_target.component == ComponentType.MLP_OUTPUT
        assert mlp_target.head is None
        assert mlp_target.token_position is None
        assert mlp_target.neuron_index is None

    def test_frozen(self, mlp_target):
        with pytest.raises(AttributeError):
            mlp_target.layer = 5

    def test_is_pre_hook(self):
        pre = HookTarget(layer=0, component=ComponentType.RESIDUAL_PRE)
        post = HookTarget(layer=0, component=ComponentType.MLP_OUTPUT)
        assert pre.is_pre_hook is True
        assert post.is_pre_hook is False

    def test_to_key_basic(self, mlp_target):
        key = mlp_target.to_key()
        assert "L17" in key
        assert "mlp_output" in key

    def test_to_key_with_extras(self, attn_target):
        key = attn_target.to_key()
        assert "L5" in key
        assert "attn_output" in key
        assert "H3" in key
        assert "T2" in key

    def test_to_key_deterministic(self, mlp_target):
        assert mlp_target.to_key() == mlp_target.to_key()

    def test_to_dict_roundtrip(self, attn_target):
        d = attn_target.to_dict()
        restored = HookTarget.from_dict(d)
        assert restored == attn_target

    def test_to_dict_minimal(self, mlp_target):
        d = mlp_target.to_dict()
        assert d["layer"] == 17
        assert d["component"] == "mlp_output"

    def test_from_dict_minimal(self):
        t = HookTarget.from_dict({"layer": 0, "component": "embedding"})
        assert t.component == ComponentType.EMBEDDING

    def test_to_module_name_standard(self):
        """Test module name resolution with a standard architecture mock."""
        target = HookTarget(layer=5, component=ComponentType.MLP_OUTPUT)
        module_map = {
            "model.layers.5.mlp.down_proj": "mock_module",
        }
        name = target.to_module_name(module_map)
        assert "5" in name
        assert "mlp" in name.lower() or "down_proj" in name

    def test_to_module_name_gemma3(self):
        """Test module name resolution with Gemma 3 architecture mock."""
        target = HookTarget(layer=10, component=ComponentType.MLP_OUTPUT)
        module_map = {
            "model.language_model.layers.10.mlp.down_proj": "mock_module",
        }
        name = target.to_module_name(module_map)
        assert "10" in name

    def test_to_module_name_attention_pattern_gemma3(self):
        """Test ATTENTION_PATTERN resolves to self_attn module."""
        target = HookTarget(layer=7, component=ComponentType.ATTENTION_PATTERN)
        module_map = {
            "model.language_model.layers.7.self_attn": "mock_module",
        }
        name = target.to_module_name(module_map)
        assert name == "model.language_model.layers.7.self_attn"

    def test_to_module_name_attention_pattern_standard(self):
        """Test ATTENTION_PATTERN resolves for standard architecture."""
        target = HookTarget(layer=3, component=ComponentType.ATTENTION_PATTERN)
        module_map = {
            "model.layers.3.self_attn": "mock_module",
        }
        name = target.to_module_name(module_map)
        assert name == "model.layers.3.self_attn"

    def test_to_module_name_missing_raises(self):
        target = HookTarget(layer=99, component=ComponentType.MLP_OUTPUT)
        with pytest.raises((KeyError, ValueError)):
            target.to_module_name({})

    def test_embedding_ignores_layer(self):
        t = HookTarget(layer=0, component=ComponentType.EMBEDDING)
        key = t.to_key()
        assert "embedding" in key
