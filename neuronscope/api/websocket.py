"""WebSocket endpoint for streaming activations during forward passes.

Client sends a JSON message with input_text and targets,
server streams back one JSON message per captured activation as they complete.
"""

from __future__ import annotations

import asyncio
import json

import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from neuronscope.hooks.manager import HookManager
from neuronscope.hooks.targets import HookTarget
from neuronscope.models.registry import ModelRegistry
from neuronscope.experiments.runner import _gpu_lock
from neuronscope.utils.tensor_utils import tensor_stats

router = APIRouter()


@router.websocket("/ws/stream")
async def stream_activations(ws: WebSocket):
    """Stream activations for each hook target as they are captured.

    Protocol:
    1. Client connects
    2. Server sends: {"type": "connected"}
    3. Client sends: {"input_text": "...", "targets": [...]}
    4. Server streams: {"type": "activation", "target_key": "...", "shape": [...], "stats": {...}, "values": [...]}
    5. Server sends: {"type": "done", "count": N}
    6. Connection stays open for more requests, or client disconnects
    """
    await ws.accept()
    await ws.send_json({"type": "connected"})

    registry = ModelRegistry()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            input_text = msg.get("input_text", "")
            target_dicts = msg.get("targets", [])

            if not input_text or not target_dicts:
                await ws.send_json({"type": "error", "detail": "input_text and targets required"})
                continue

            if not registry.is_loaded:
                await ws.send_json({"type": "error", "detail": "No model loaded"})
                continue

            # Try to acquire GPU lock
            if not _gpu_lock.acquire(blocking=False):
                await ws.send_json({"type": "error", "detail": "GPU is busy with another operation"})
                continue

            try:
                hook_manager = HookManager(registry.model, registry.module_map)
                targets = [HookTarget.from_dict(t) for t in target_dicts]

                # Use a queue to stream results from the executor thread
                queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_event_loop()

                def _capture_streaming():
                    """Run forward pass and put results on the queue as they're captured."""
                    storage: dict[str, torch.Tensor] = {}

                    with hook_manager.session() as hm:
                        for target in targets:
                            hm.attach_extraction_hook(target, storage)

                        input_ids = registry.tokenizer(
                            input_text, return_tensors="pt"
                        ).input_ids.to(next(registry.model.parameters()).device)

                        with torch.no_grad():
                            registry.model(input_ids)

                    # After forward pass, stream results one by one
                    for target in targets:
                        key = target.to_key()
                        tensor = storage.get(key)
                        if tensor is None:
                            continue

                        result = {
                            "type": "activation",
                            "target_key": key,
                            "shape": list(tensor.shape),
                            "stats": tensor_stats(tensor),
                        }
                        # Include values for small tensors
                        if tensor.numel() < 10_000:
                            result["values"] = tensor.float().flatten().tolist()

                        loop.call_soon_threadsafe(queue.put_nowait, result)

                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {"type": "done", "count": len(storage)},
                    )

                await loop.run_in_executor(None, _capture_streaming)

                # Drain the queue and send results
                while True:
                    result = await asyncio.wait_for(queue.get(), timeout=5.0)
                    await ws.send_json(result)
                    if result.get("type") == "done":
                        break

            finally:
                _gpu_lock.release()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
