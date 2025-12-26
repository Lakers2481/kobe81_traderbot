from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime


class IdempotencyStore:
    """
    SQLite-backed store to prevent duplicate order submissions per decision_id.

    Uses WAL mode for better concurrency and connection pooling.
    """

    def __init__(self, db_path: str | Path = "state/idempotency.sqlite"):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection with WAL mode enabled."""
        con = sqlite3.connect(self.path, timeout=30.0)
        # Enable WAL mode for better concurrency
        con.execute("PRAGMA journal_mode=WAL")
        # Enable foreign keys
        con.execute("PRAGMA foreign_keys=ON")
        # Synchronous mode for safety
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    def _init_db(self):
        with self._get_connection() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency (
                    decision_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            # Create index for faster lookups
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_idempotency_created_at
                ON idempotency(created_at)
                """
            )

    def exists(self, decision_id: str) -> bool:
        with self._get_connection() as con:
            cur = con.execute("SELECT 1 FROM idempotency WHERE decision_id=?", (decision_id,))
            return cur.fetchone() is not None

    def put(self, decision_id: str, idempotency_key: str) -> None:
        with self._get_connection() as con:
            con.execute(
                "INSERT OR IGNORE INTO idempotency(decision_id, idempotency_key, created_at) VALUES(?,?,?)",
                (decision_id, idempotency_key, datetime.utcnow().isoformat()),
            )

    def get(self, decision_id: str) -> Optional[str]:
        with self._get_connection() as con:
            cur = con.execute("SELECT idempotency_key FROM idempotency WHERE decision_id=?", (decision_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def count(self) -> int:
        """Get total number of entries in the store."""
        with self._get_connection() as con:
            cur = con.execute("SELECT COUNT(*) FROM idempotency")
            return cur.fetchone()[0]

    def cleanup_older_than(self, days: int = 30) -> int:
        """
        Remove entries older than specified days.

        Args:
            days: Remove entries older than this many days

        Returns:
            Number of entries removed
        """
        cutoff = datetime.utcnow().isoformat()[:10]  # YYYY-MM-DD
        with self._get_connection() as con:
            cur = con.execute(
                "DELETE FROM idempotency WHERE created_at < date(?, ?)",
                (cutoff, f"-{days} days"),
            )
            return cur.rowcount

    def clear(self) -> None:
        """Clear all entries (for testing)."""
        with self._get_connection() as con:
            con.execute("DELETE FROM idempotency")
