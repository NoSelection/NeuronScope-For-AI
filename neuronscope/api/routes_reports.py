"""API routes for PDF report generation."""

from __future__ import annotations

import asyncio
from functools import partial

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from neuronscope.experiments.schema import ExperimentConfig
from neuronscope.experiments.runner import ExperimentRunner
from neuronscope.hooks.manager import HookManager
from neuronscope.models.registry import ModelRegistry
from neuronscope.reports.sweep_report import generate_sweep_pdf
from neuronscope.reports.experiment_report import generate_experiment_pdf
from neuronscope.store.experiment_store import ExperimentStore
from neuronscope.store.sweep_store import SweepStore

router = APIRouter()
store = ExperimentStore()
sweep_store = SweepStore()


class SweepReportRequest(BaseModel):
    config: ExperimentConfig
    layers: list[int] | None = None


def _get_runner() -> ExperimentRunner:
    registry = ModelRegistry()
    if not registry.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded")
    hook_manager = HookManager(registry.model, registry.module_map)
    return ExperimentRunner(registry.model, registry.tokenizer, hook_manager)


@router.post("/sweep")
async def sweep_report(request: SweepReportRequest):
    """Run a sweep and return a comprehensive PDF report."""
    runner = _get_runner()

    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None, partial(runner.run_sweep, request.config, request.layers)
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    await store.save_many(results)
    pdf_bytes = bytes(generate_sweep_pdf(results))

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="neuronscope_sweep_report.pdf"',
        },
    )


@router.post("/experiment")
async def experiment_report_from_config(config: ExperimentConfig):
    """Run an experiment and return a PDF report."""
    runner = _get_runner()

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, partial(runner.run, config))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    await store.save(result)
    pdf_bytes = bytes(generate_experiment_pdf(result))

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="neuronscope_experiment_{result.id}.pdf"',
        },
    )


@router.get("/sweep/{sweep_id}")
async def sweep_report_from_id(sweep_id: str):
    """Generate a PDF report from a previously saved sweep (no re-running)."""
    sweep = await sweep_store.get(sweep_id)
    if sweep is None:
        raise HTTPException(status_code=404, detail="Sweep not found")

    results = []
    for exp_id in sweep["experiment_ids"]:
        result = await store.get(exp_id)
        if result is not None:
            results.append(result)

    if not results:
        raise HTTPException(status_code=404, detail="No experiment results found for this sweep")

    pdf_bytes = bytes(generate_sweep_pdf(results))

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="neuronscope_sweep_{sweep_id}.pdf"',
        },
    )


@router.get("/experiment/{experiment_id}")
async def experiment_report_from_id(experiment_id: str):
    """Generate a PDF report from a previously saved experiment."""
    result = await store.get(experiment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    pdf_bytes = bytes(generate_experiment_pdf(result))

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="neuronscope_experiment_{experiment_id}.pdf"',
        },
    )
