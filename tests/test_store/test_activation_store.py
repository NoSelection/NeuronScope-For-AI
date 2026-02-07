"""Tests for activation persistence store."""

from __future__ import annotations

import pytest
import torch

from neuronscope.store.activation_store import ActivationStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Create an ActivationStore using a temporary database."""
    import neuronscope.store.database as db_mod
    monkeypatch.setattr(db_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "test.db")
    return ActivationStore()


@pytest.mark.asyncio
async def test_save_and_get(store):
    tensor = torch.randn(1, 4, 8)
    aid = await store.save("exp-001", "L5.mlp_output", tensor)
    assert isinstance(aid, str)
    assert len(aid) == 8

    record = await store.get(aid)
    assert record is not None
    assert record["id"] == aid
    assert record["experiment_id"] == "exp-001"
    assert record["target_key"] == "L5.mlp_output"
    assert record["shape"] == [1, 4, 8]
    assert "mean" in record["stats"]
    # Tensor should be reconstructed correctly
    assert torch.allclose(record["tensor"], tensor, atol=1e-5)


@pytest.mark.asyncio
async def test_get_nonexistent(store):
    result = await store.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_get_metadata(store):
    tensor = torch.randn(2, 3)
    aid = await store.save(None, "L0.attn_output", tensor)
    meta = await store.get_metadata(aid)
    assert meta is not None
    assert "tensor" not in meta
    assert meta["target_key"] == "L0.attn_output"


@pytest.mark.asyncio
async def test_list_by_experiment(store):
    t1 = torch.randn(1, 4)
    t2 = torch.randn(1, 8)
    t3 = torch.randn(1, 2)
    await store.save("exp-A", "L0.mlp_output", t1)
    await store.save("exp-A", "L1.mlp_output", t2)
    await store.save("exp-B", "L0.mlp_output", t3)

    results = await store.list_by_experiment("exp-A")
    assert len(results) == 2
    assert all(r["experiment_id"] == "exp-A" for r in results)


@pytest.mark.asyncio
async def test_list_all(store):
    await store.save("exp-1", "L0.mlp_output", torch.randn(4))
    await store.save("exp-2", "L1.attn_output", torch.randn(8))
    results = await store.list_all()
    assert len(results) == 2


@pytest.mark.asyncio
async def test_delete(store):
    aid = await store.save("exp-1", "L0.mlp_output", torch.randn(4))
    assert await store.delete(aid)
    assert await store.get(aid) is None


@pytest.mark.asyncio
async def test_delete_nonexistent(store):
    assert not await store.delete("nonexistent")


@pytest.mark.asyncio
async def test_tensor_compression_roundtrip(store):
    """Large tensor should survive compression/decompression."""
    tensor = torch.randn(1, 16, 64)
    aid = await store.save("exp-1", "L10.residual_post", tensor)
    record = await store.get(aid)
    assert torch.allclose(record["tensor"], tensor, atol=1e-5)


@pytest.mark.asyncio
async def test_save_with_null_experiment_id(store):
    aid = await store.save(None, "L0.mlp_output", torch.randn(4))
    record = await store.get(aid)
    assert record["experiment_id"] is None
