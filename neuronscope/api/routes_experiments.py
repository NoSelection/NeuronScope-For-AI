from __future__ import annotations

import asyncio
from functools import partial

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from neuronscope.experiments.schema import ExperimentConfig, ExperimentResult
from neuronscope.experiments.runner import ExperimentRunner
from neuronscope.hooks.manager import HookManager
from neuronscope.models.registry import ModelRegistry
from neuronscope.store.experiment_store import ExperimentStore

router = APIRouter()
store = ExperimentStore()


class SweepRequest(BaseModel):
    config: ExperimentConfig
    layers: list[int] | None = None


def _get_runner() -> ExperimentRunner:
    """Get an ExperimentRunner with the current model."""
    registry = ModelRegistry()
    if not registry.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded")
    hook_manager = HookManager(registry.model, registry.module_map)
    return ExperimentRunner(registry.model, registry.tokenizer, hook_manager)


@router.post("/run")
async def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a complete experiment (clean + intervention + comparison)."""
    runner = _get_runner()

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, partial(runner.run, config))

    await store.save(result)
    return result


@router.post("/sweep")
async def run_sweep(request: SweepRequest) -> list[ExperimentResult]:
    """Run the same intervention across multiple layers."""
    runner = _get_runner()

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, partial(runner.run_sweep, request.config, request.layers)
    )

    await store.save_many(results)
    return results


@router.get("/")
async def list_experiments() -> list[dict]:
    """List all experiment results (persisted in SQLite)."""
    return await store.list_all()


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str) -> ExperimentResult:
    """Get a specific experiment result by ID."""
    result = await store.get(experiment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return result


@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: str) -> dict:
    """Delete an experiment result."""
    deleted = await store.delete(experiment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"status": "deleted"}
