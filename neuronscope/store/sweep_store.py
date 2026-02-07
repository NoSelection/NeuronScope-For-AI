from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from neuronscope.experiments.schema import ExperimentConfig, ExperimentResult
from neuronscope.store.database import get_db


class SweepStore:
    """Persistent storage for sweep records using SQLite."""

    async def save(
        self,
        name: str,
        config: ExperimentConfig,
        results: list[ExperimentResult],
    ) -> str:
        """Save a sweep record. Returns the sweep ID."""
        sweep_id = uuid.uuid4().hex[:12]
        experiment_ids = [r.id for r in results]
        peak_result = max(results, key=lambda r: r.kl_divergence)
        peak_layer = (
            peak_result.config.interventions[0].target_layer
            if peak_result.config.interventions
            else 0
        )

        db = await get_db()
        try:
            await db.execute(
                "INSERT OR REPLACE INTO sweeps "
                "(id, name, config_json, experiment_ids, num_layers, peak_kl, peak_layer, layers_changed, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sweep_id,
                    name,
                    config.model_dump_json(),
                    json.dumps(experiment_ids),
                    len(results),
                    peak_result.kl_divergence,
                    peak_layer,
                    sum(1 for r in results if r.top_token_changed),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
        finally:
            await db.close()

        return sweep_id

    async def list_all(self) -> list[dict]:
        """List all sweep summaries."""
        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT id, name, num_layers, peak_kl, peak_layer, layers_changed, created_at "
                "FROM sweeps ORDER BY created_at DESC"
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "num_layers": row[2],
                    "peak_kl": row[3],
                    "peak_layer": row[4],
                    "layers_changed": row[5],
                    "timestamp": row[6],
                }
                for row in rows
            ]
        finally:
            await db.close()

    async def get(self, sweep_id: str) -> dict | None:
        """Get a sweep record with its config and experiment IDs."""
        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT id, name, config_json, experiment_ids, num_layers, peak_kl, peak_layer, layers_changed, created_at "
                "FROM sweeps WHERE id = ?",
                (sweep_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "name": row[1],
                "config": json.loads(row[2]),
                "experiment_ids": json.loads(row[3]),
                "num_layers": row[4],
                "peak_kl": row[5],
                "peak_layer": row[6],
                "layers_changed": row[7],
                "timestamp": row[8],
            }
        finally:
            await db.close()

    async def delete(self, sweep_id: str) -> bool:
        """Delete a sweep record and its associated experiments."""
        db = await get_db()
        try:
            # Get experiment IDs first
            cursor = await db.execute(
                "SELECT experiment_ids FROM sweeps WHERE id = ?", (sweep_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return False

            experiment_ids = json.loads(row[0])

            # Delete experiments
            if experiment_ids:
                placeholders = ",".join("?" for _ in experiment_ids)
                await db.execute(
                    f"DELETE FROM experiments WHERE id IN ({placeholders})",
                    experiment_ids,
                )

            # Delete sweep
            await db.execute("DELETE FROM sweeps WHERE id = ?", (sweep_id,))
            await db.commit()
            return True
        finally:
            await db.close()
