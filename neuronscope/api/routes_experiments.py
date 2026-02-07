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
from neuronscope.store.sweep_store import SweepStore
from neuronscope.analysis.insights import generate_insights, generate_sweep_insights

router = APIRouter()
store = ExperimentStore()
sweep_store = SweepStore()


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
async def run_experiment(config: ExperimentConfig) -> dict:
    """Run a complete experiment (clean + intervention + comparison)."""
    runner = _get_runner()

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, partial(runner.run, config))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    await store.save(result)
    insights = generate_insights(result)
    return {"result": result, "insights": insights}


@router.post("/sweep")
async def run_sweep(request: SweepRequest) -> dict:
    """Run the same intervention across multiple layers."""
    runner = _get_runner()

    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None, partial(runner.run_sweep, request.config, request.layers)
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    await store.save_many(results)
    sweep_id = await sweep_store.save(
        request.config.name or "Untitled Sweep", request.config, results
    )
    insights = generate_sweep_insights(results)
    return {"results": results, "insights": insights, "sweep_id": sweep_id}


@router.get("/sweeps")
async def list_sweeps() -> list[dict]:
    """List all sweep summaries."""
    return await sweep_store.list_all()


@router.get("/sweeps/{sweep_id}")
async def get_sweep(sweep_id: str) -> dict:
    """Get a sweep with full results."""
    sweep = await sweep_store.get(sweep_id)
    if sweep is None:
        raise HTTPException(status_code=404, detail="Sweep not found")

    # Fetch full experiment results
    results = []
    for exp_id in sweep["experiment_ids"]:
        result = await store.get(exp_id)
        if result is not None:
            results.append(result)

    insights = generate_sweep_insights(results) if results else []
    return {**sweep, "results": results, "insights": insights}


@router.delete("/sweeps/{sweep_id}")
async def delete_sweep(sweep_id: str) -> dict:
    """Delete a sweep and its experiments."""
    deleted = await sweep_store.delete(sweep_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Sweep not found")
    return {"status": "deleted"}


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
