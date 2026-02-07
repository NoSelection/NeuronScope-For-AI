from __future__ import annotations

import asyncio
import threading

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from neuronscope.analysis.attribution import attribute_by_layer
from neuronscope.hooks.manager import HookManager
from neuronscope.models.registry import ModelRegistry

router = APIRouter()

# Reuse the same GPU lock from the experiment runner
from neuronscope.experiments.runner import _gpu_lock


class AttributionRequest(BaseModel):
    """Request for per-layer causal attribution."""

    base_input: str
    intervention_type: str = "zero"
    component: str = "mlp_output"
    layers: list[int] | None = None


@router.post("/attribution")
async def run_attribution(request: AttributionRequest):
    """Run per-layer causal attribution and return scores.

    Iterates over layers, applying the specified intervention at each,
    and measures KL divergence from the clean baseline.
    """
    registry = ModelRegistry()
    if not registry.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded")

    hook_manager = HookManager(registry.model, registry.module_map)

    def _run():
        if not _gpu_lock.acquire(blocking=False):
            raise RuntimeError("GPU is busy with another operation")
        try:
            return attribute_by_layer(
                model=registry.model,
                tokenizer=registry.tokenizer,
                hook_manager=hook_manager,
                base_input=request.base_input,
                intervention_type=request.intervention_type,
                component=request.component,
                layers=request.layers,
            )
        finally:
            _gpu_lock.release()

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _run)
    except RuntimeError as e:
        if "GPU is busy" in str(e):
            raise HTTPException(status_code=409, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    return result.to_dict()
