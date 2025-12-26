from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime


class IdempotencyStore:
    """Simple SQLite-backed store to prevent duplicate order submissions per decision_id."""

    def __init__(self, db_path: str | Path = "state/idempotency.sqlite"):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency (
                    decision_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def exists(self, decision_id: str) -> bool:
        with sqlite3.connect(self.path) as con:
            cur = con.execute("SELECT 1 FROM idempotency WHERE decision_id=?", (decision_id,))
            return cur.fetchone() is not None

    def put(self, decision_id: str, idempotency_key: str) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute(
                "INSERT OR IGNORE INTO idempotency(decision_id, idempotency_key, created_at) VALUES(?,?,?)",
                (decision_id, idempotency_key, datetime.utcnow().isoformat()),
            )

    def get(self, decision_id: str) -> Optional[str]:
        with sqlite3.connect(self.path) as con:
            cur = con.execute("SELECT idempotency_key FROM idempotency WHERE decision_id=?", (decision_id,))
            row = cur.fetchone()
            return row[0] if row else None

