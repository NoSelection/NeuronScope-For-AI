from __future__ import annotations

import threading
import torch

from neuronscope.models.loader import ModelLoader
from neuronscope.models.schema import ModelInfo


class ModelRegistry:
    """Singleton registry that tracks loaded models and prevents double-loading."""

    _instance: ModelRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._model = None
                    cls._instance._tokenizer = None
                    cls._instance._module_map = None
                    cls._instance._info = None
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            raise RuntimeError("No model loaded. Call load() first.")
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("No model loaded. Call load() first.")
        return self._tokenizer

    @property
    def module_map(self) -> dict[str, torch.nn.Module]:
        if self._module_map is None:
            raise RuntimeError("No model loaded. Call load() first.")
        return self._module_map

    @property
    def info(self) -> ModelInfo:
        if self._info is None:
            raise RuntimeError("No model loaded. Call load() first.")
        return self._info

    def load(self, model_path: str = "LLM", device: str = "cuda") -> ModelInfo:
        """Load a model. Returns info. No-ops if already loaded from same path."""
        if self._info is not None and self._info.path == str(model_path):
            return self._info

        self.unload()
        self._model, self._tokenizer, self._module_map, self._info = ModelLoader.load(
            model_path, device
        )
        return self._info

    def unload(self) -> None:
        """Unload the current model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._tokenizer = None
            self._module_map = None
            self._info = None
            torch.cuda.empty_cache()
