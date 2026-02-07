from __future__ import annotations

import asyncio
from functools import partial

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from neuronscope.hooks.manager import HookManager
from neuronscope.hooks.targets import HookTarget, ComponentType
from neuronscope.models.registry import ModelRegistry
from neuronscope.store.activation_store import ActivationStore
from neuronscope.utils.tensor_utils import tensor_stats

router = APIRouter()
_activation_store = ActivationStore()


class CaptureRequest(BaseModel):
    """Request to capture activations at specified points."""

    input_text: str
    targets: list[dict]  # List of HookTarget dicts
    save: bool = False  # Whether to persist to SQLite
    experiment_id: str | None = None


class CaptureResult(BaseModel):
    """Captured activation data."""

    target_key: str
    shape: list[int]
    stats: dict[str, float]
    values: list[float] | None = None  # Flattened values (only for small tensors)
    activation_id: str | None = None  # Set if saved to store


@router.post("/capture")
async def capture_activations(request: CaptureRequest) -> list[CaptureResult]:
    """Capture activations at specified hook points for a given input.

    Returns activation statistics and optionally raw values
    (for small tensors only, to avoid massive payloads).
    """
    registry = ModelRegistry()
    if not registry.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded")

    hook_manager = HookManager(registry.model, registry.module_map)
    targets = [HookTarget.from_dict(t) for t in request.targets]

    def _capture():
        storage: dict[str, torch.Tensor] = {}

        with hook_manager.session() as hm:
            for target in targets:
                hm.attach_extraction_hook(target, storage)

            input_ids = registry.tokenizer(
                request.input_text, return_tensors="pt"
            ).input_ids.to(next(registry.model.parameters()).device)

            with torch.no_grad():
                registry.model(input_ids)

        results = []
        for target in targets:
            key = target.to_key()
            tensor = storage.get(key)
            if tensor is None:
                continue

            result = CaptureResult(
                target_key=key,
                shape=list(tensor.shape),
                stats=tensor_stats(tensor),
            )
            # Include raw values only for small tensors (< 10k elements)
            if tensor.numel() < 10_000:
                result.values = tensor.float().flatten().tolist()

            results.append(result)

        return results, storage

    loop = asyncio.get_event_loop()
    results, storage = await loop.run_in_executor(None, _capture)

    # Optionally persist activations
    if request.save:
        for target in targets:
            key = target.to_key()
            tensor = storage.get(key)
            if tensor is None:
                continue
            activation_id = await _activation_store.save(
                request.experiment_id, key, tensor.cpu()
            )
            # Attach the ID to the matching result
            for r in results:
                if r.target_key == key:
                    r.activation_id = activation_id

    return results


@router.get("/stored")
async def list_stored_activations():
    """List all stored activations (metadata only)."""
    return await _activation_store.list_all()


@router.get("/stored/experiment/{experiment_id}")
async def list_activations_by_experiment(experiment_id: str):
    """List stored activations for a specific experiment."""
    return await _activation_store.list_by_experiment(experiment_id)


@router.get("/stored/{activation_id}")
async def get_stored_activation(activation_id: str):
    """Get a stored activation's metadata and values."""
    record = await _activation_store.get(activation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Activation not found")

    tensor = record.pop("tensor")
    # Include values for small tensors
    if tensor.numel() < 10_000:
        record["values"] = tensor.float().flatten().tolist()

    return record


@router.delete("/stored/{activation_id}")
async def delete_stored_activation(activation_id: str):
    """Delete a stored activation."""
    deleted = await _activation_store.delete(activation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Activation not found")
    return {"deleted": True}
