from __future__ import annotations

import json

from neuronscope.experiments.schema import ExperimentResult
from neuronscope.store.database import get_db


class ExperimentStore:
    """Persistent storage for experiment results using SQLite."""

    async def save(self, result: ExperimentResult) -> None:
        """Save an experiment result."""
        db = await get_db()
        try:
            await db.execute(
                "INSERT OR REPLACE INTO experiments (id, config_hash, name, result_json) VALUES (?, ?, ?, ?)",
                (
                    result.id,
                    result.config_hash,
                    result.config.name,
                    result.model_dump_json(),
                ),
            )
            await db.commit()
        finally:
            await db.close()

    async def save_many(self, results: list[ExperimentResult]) -> None:
        """Save multiple experiment results in a single transaction."""
        db = await get_db()
        try:
            await db.executemany(
                "INSERT OR REPLACE INTO experiments (id, config_hash, name, result_json) VALUES (?, ?, ?, ?)",
                [
                    (r.id, r.config_hash, r.config.name, r.model_dump_json())
                    for r in results
                ],
            )
            await db.commit()
        finally:
            await db.close()

    async def get(self, experiment_id: str) -> ExperimentResult | None:
        """Get a single experiment result by ID."""
        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT result_json FROM experiments WHERE id = ?", (experiment_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return ExperimentResult.model_validate_json(row[0])
        finally:
            await db.close()

    async def get_by_hash(self, config_hash: str) -> ExperimentResult | None:
        """Get an experiment result by config hash (for reproducibility checks)."""
        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT result_json FROM experiments WHERE config_hash = ? LIMIT 1",
                (config_hash,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return ExperimentResult.model_validate_json(row[0])
        finally:
            await db.close()

    async def list_all(self) -> list[dict]:
        """List all experiment summaries (lightweight, no full results)."""
        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT id, config_hash, name, result_json, created_at FROM experiments ORDER BY created_at DESC"
            )
            rows = await cursor.fetchall()
            summaries = []
            for row in rows:
                result = json.loads(row[3])
                summaries.append({
                    "id": row[0],
                    "config_hash": row[1],
                    "name": row[2],
                    "kl_divergence": result.get("kl_divergence", 0),
                    "top_token_changed": result.get("top_token_changed", False),
                    "clean_token": result.get("clean_output_token", ""),
                    "intervention_token": result.get("intervention_output_token", ""),
                    "timestamp": result.get("timestamp", row[4]),
                })
            return summaries
        finally:
            await db.close()

    async def delete(self, experiment_id: str) -> bool:
        """Delete an experiment result. Returns True if found and deleted."""
        db = await get_db()
        try:
            cursor = await db.execute(
                "DELETE FROM experiments WHERE id = ?", (experiment_id,)
            )
            await db.commit()
            return cursor.rowcount > 0
        finally:
            await db.close()

    async def count(self) -> int:
        """Get total number of stored experiments."""
        db = await get_db()
        try:
            cursor = await db.execute("SELECT COUNT(*) FROM experiments")
            row = await cursor.fetchone()
            return row[0]
        finally:
            await db.close()
