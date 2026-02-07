from __future__ import annotations

import aiosqlite
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DB_PATH = DATA_DIR / "neuronscope.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    config_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_experiments_hash ON experiments(config_hash);
CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name);
"""


async def get_db() -> aiosqlite.Connection:
    """Get a connection to the SQLite database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.executescript(_CREATE_TABLE)
    return db
