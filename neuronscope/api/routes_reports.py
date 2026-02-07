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

router = APIRouter()
store = ExperimentStore()


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
    results = await loop.run_in_executor(
        None, partial(runner.run_sweep, request.config, request.layers)
    )

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
    result = await loop.run_in_executor(None, partial(runner.run, config))

    await store.save(result)
    pdf_bytes = bytes(generate_experiment_pdf(result))

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="neuronscope_experiment_{result.id}.pdf"',
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
