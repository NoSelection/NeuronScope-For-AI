from __future__ import annotations

import io
import json
import uuid
import zlib
from datetime import datetime, timezone

import torch

from neuronscope.store.database import get_db
from neuronscope.utils.tensor_utils import tensor_stats


class ActivationStore:
    """Persistent storage for captured activations using SQLite."""

    async def save(
        self,
        experiment_id: str | None,
        target_key: str,
        tensor: torch.Tensor,
    ) -> str:
        """Save a captured activation tensor. Returns the activation ID."""
        activation_id = str(uuid.uuid4())[:8]
        shape = list(tensor.shape)
        stats = tensor_stats(tensor)

        # Serialize tensor: torch.save -> BytesIO -> zlib compress
        buf = io.BytesIO()
        torch.save(tensor.cpu(), buf)
        tensor_blob = zlib.compress(buf.getvalue())

        db = await get_db()
        try:
            await db.execute(
                "INSERT INTO activations (id, experiment_id, target_key, shape_json, stats_json, tensor_blob, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    activation_id,
                    experiment_id,
                    target_key,
                    json.dumps(shape),
                    json.dumps(stats),
                    tensor_blob,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
            return activation_id
        finally:
            await db.close()

    async def get(self, activation_id: str) -> dict | None:
        """Get activation metadata + tensor by ID."""
        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT id, experiment_id, target_key, shape_json, stats_json, tensor_blob, created_at "
                "FROM activations WHERE id = ?",
                (activation_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None

            tensor = self._decompress_tensor(row[5])
            return {
                "id": row[0],
                "experiment_id": row[1],
                "target_key": row[2],
                "shape": json.loads(row[3]),
                "stats": json.loads(row[4]),
                "tensor": tensor,
                "created_at": row[6],
            }
        finally:
            await db.close()

    async def get_metadata(self, activation_id: str) -> dict | None:
        """Get activation metadata without the tensor blob."""
        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT id, experiment_id, target_key, shape_json, stats_json, created_at "
                "FROM activations WHERE id = ?",
                (activation_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "experiment_id": row[1],
                "target_key": row[2],
                "shape": json.loads(row[3]),
                "stats": json.loads(row[4]),
                "created_at": row[5],
            }
        finally:
            await db.close()

    async def list_by_experiment(self, experiment_id: str) -> list[dict]:
        """List all activations for a given experiment (metadata only)."""
        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT id, experiment_id, target_key, shape_json, stats_json, created_at "
                "FROM activations WHERE experiment_id = ? ORDER BY created_at DESC",
                (experiment_id,),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "experiment_id": row[1],
                    "target_key": row[2],
                    "shape": json.loads(row[3]),
                    "stats": json.loads(row[4]),
                    "created_at": row[5],
                }
                for row in rows
            ]
        finally:
            await db.close()

    async def list_all(self) -> list[dict]:
        """List all stored activations (metadata only)."""
        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT id, experiment_id, target_key, shape_json, stats_json, created_at "
                "FROM activations ORDER BY created_at DESC"
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "experiment_id": row[1],
                    "target_key": row[2],
                    "shape": json.loads(row[3]),
                    "stats": json.loads(row[4]),
                    "created_at": row[5],
                }
                for row in rows
            ]
        finally:
            await db.close()

    async def delete(self, activation_id: str) -> bool:
        """Delete a stored activation. Returns True if found and deleted."""
        db = await get_db()
        try:
            cursor = await db.execute(
                "DELETE FROM activations WHERE id = ?", (activation_id,)
            )
            await db.commit()
            return cursor.rowcount > 0
        finally:
            await db.close()

    @staticmethod
    def _decompress_tensor(blob: bytes) -> torch.Tensor:
        """Decompress a zlib-compressed tensor blob."""
        raw = zlib.decompress(blob)
        buf = io.BytesIO(raw)
        return torch.load(buf, map_location="cpu", weights_only=True)
