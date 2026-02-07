from __future__ import annotations

import asyncio
from functools import partial

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from neuronscope.hooks.manager import HookManager
from neuronscope.hooks.targets import HookTarget, ComponentType
from neuronscope.models.registry import ModelRegistry
from neuronscope.utils.tensor_utils import tensor_stats

router = APIRouter()


class CaptureRequest(BaseModel):
    """Request to capture activations at specified points."""

    input_text: str
    targets: list[dict]  # List of HookTarget dicts


class CaptureResult(BaseModel):
    """Captured activation data."""

    target_key: str
    shape: list[int]
    stats: dict[str, float]
    values: list[float] | None = None  # Flattened values (only for small tensors)


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

        return results

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _capture)
