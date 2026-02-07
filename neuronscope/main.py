"""NeuronScope API entry point.

Run with: uvicorn neuronscope.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neuronscope.api.routes_model import router as model_router
from neuronscope.api.routes_experiments import router as experiment_router
from neuronscope.api.routes_activations import router as activation_router
from neuronscope.api.routes_reports import router as report_router
from neuronscope.api.routes_analysis import router as analysis_router
from neuronscope.api.websocket import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    yield
    # Cleanup: unload model on shutdown
    from neuronscope.models.registry import ModelRegistry

    registry = ModelRegistry()
    if registry.is_loaded:
        registry.unload()


app = FastAPI(
    title="NeuronScope",
    description="Mechanistic interpretability through causal intervention",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_router, prefix="/api/model", tags=["model"])
app.include_router(experiment_router, prefix="/api/experiments", tags=["experiments"])
app.include_router(activation_router, prefix="/api/activations", tags=["activations"])
app.include_router(report_router, prefix="/api/reports", tags=["reports"])
app.include_router(analysis_router, prefix="/api/analysis", tags=["analysis"])
app.include_router(ws_router, prefix="/api", tags=["websocket"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
