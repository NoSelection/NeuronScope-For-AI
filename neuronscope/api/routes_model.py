from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from neuronscope.models.registry import ModelRegistry
from neuronscope.models.schema import ModelInfo

router = APIRouter()


class LoadRequest(BaseModel):
    model_path: str = "LLM"
    device: str = "cuda"


@router.get("/info")
async def model_info() -> ModelInfo | dict:
    """Get metadata about the currently loaded model."""
    registry = ModelRegistry()
    if not registry.is_loaded:
        return {"loaded": False}
    return registry.info


@router.post("/load")
async def load_model(request: LoadRequest) -> ModelInfo:
    """Load a model from disk."""
    registry = ModelRegistry()
    try:
        info = registry.load(request.model_path, request.device)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload")
async def unload_model() -> dict:
    """Unload the current model and free GPU memory."""
    registry = ModelRegistry()
    registry.unload()
    return {"status": "unloaded"}


@router.get("/modules")
async def list_modules() -> dict:
    """List all named modules in the model (for hook targeting)."""
    registry = ModelRegistry()
    if not registry.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded")

    modules = {}
    for name, mod in registry.model.named_modules():
        modules[name] = type(mod).__name__
    return modules


@router.post("/tokenize")
async def tokenize(text: str) -> list[dict]:
    """Tokenize input text and return tokens with positions."""
    registry = ModelRegistry()
    if not registry.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded")

    from neuronscope.utils.token_utils import tokenize_with_positions

    return tokenize_with_positions(text, registry.tokenizer)
