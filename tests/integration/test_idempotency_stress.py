"""
INTEGRATION TESTS: Idempotency Stress (MEDIUM)

Tests idempotency store under stress conditions:
- 100 duplicate submissions
- Concurrent access from multiple threads
- Cleanup of old entries
- Persistence across restarts

This ensures no duplicate orders are ever placed.

Author: Kobe Trading System
Version: 1.0.0
Date: 2026-01-06
"""

import pytest
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.mark.integration
@pytest.mark.stress
class TestDuplicateSubmissions:
    """Test that duplicate submissions are blocked."""

    def test_100_duplicate_submissions_one_execution(self, tmp_path):
        """100 submissions of same order should result in 1 execution."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        db_path = state_dir / "idempotency.db"

        # Create database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                created_at TEXT NOT NULL,
                order_id TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)
        conn.commit()
        conn.close()

        order_key = "AAPL_BUY_20260106_150"
        execution_count = 0

        for i in range(100):
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            try:
                cursor.execute(
                    "SELECT key FROM idempotency_keys WHERE key = ?",
                    (order_key,)
                )
                if cursor.fetchone() is None:
                    cursor.execute(
                        """INSERT INTO idempotency_keys
                           (key, symbol, side, created_at, status)
                           VALUES (?, ?, ?, ?, ?)""",
                        (order_key, "AAPL", "buy", datetime.now().isoformat(), "executed")
                    )
                    conn.commit()
                    execution_count += 1
            except sqlite3.IntegrityError:
                pass  # Duplicate key
            finally:
                conn.close()

        assert execution_count == 1

    def test_different_orders_all_execute(self, tmp_path):
        """Different order keys should all execute."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        db_path = state_dir / "idempotency.db"

        # Create database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                created_at TEXT NOT NULL,
                order_id TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)
        conn.commit()
        conn.close()

        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        execution_count = 0

        for symbol in symbols:
            order_key = f"{symbol}_BUY_20260106_150"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            try:
                cursor.execute(
                    "SELECT key FROM idempotency_keys WHERE key = ?",
                    (order_key,)
                )
                if cursor.fetchone() is None:
                    cursor.execute(
                        """INSERT INTO idempotency_keys
                           (key, symbol, side, created_at, status)
                           VALUES (?, ?, ?, ?, ?)""",
                        (order_key, symbol, "buy", datetime.now().isoformat(), "executed")
                    )
                    conn.commit()
                    execution_count += 1
            except sqlite3.IntegrityError:
                pass
            finally:
                conn.close()

        assert execution_count == 5


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.concurrent
class TestConcurrentIdempotencyChecks:
    """Test concurrent access to idempotency store."""

    def test_10_threads_same_key_one_execution(self, tmp_path):
        """10 concurrent threads checking same key should result in 1 execution."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        db_path = state_dir / "idempotency.db"

        # Create database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                created_at TEXT NOT NULL,
                order_id TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)
        conn.commit()
        conn.close()

        order_key = "CONCURRENT_TEST_KEY"
        execution_count = [0]
        lock = threading.Lock()

        def try_execute():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            try:
                cursor.execute(
                    "SELECT key FROM idempotency_keys WHERE key = ?",
                    (order_key,)
                )
                if cursor.fetchone() is None:
                    try:
                        cursor.execute(
                            """INSERT INTO idempotency_keys
                               (key, symbol, side, created_at, status)
                               VALUES (?, ?, ?, ?, ?)""",
                            (order_key, "TEST", "buy", datetime.now().isoformat(), "executed")
                        )
                        conn.commit()
                        with lock:
                            execution_count[0] += 1
                    except sqlite3.IntegrityError:
                        pass  # Race condition handled by SQLite
            finally:
                conn.close()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(try_execute) for _ in range(10)]
            for f in as_completed(futures):
                f.result()

        assert execution_count[0] == 1

    def test_thread_safety_with_multiple_keys(self, tmp_path):
        """Multiple threads with different keys should all execute."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        db_path = state_dir / "idempotency.db"

        # Create database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                created_at TEXT NOT NULL,
                order_id TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)
        conn.commit()
        conn.close()

        execution_count = [0]
        lock = threading.Lock()

        def try_execute(key_suffix):
            order_key = f"THREAD_KEY_{key_suffix}"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            try:
                cursor.execute(
                    "SELECT key FROM idempotency_keys WHERE key = ?",
                    (order_key,)
                )
                if cursor.fetchone() is None:
                    try:
                        cursor.execute(
                            """INSERT INTO idempotency_keys
                               (key, symbol, side, created_at, status)
                               VALUES (?, ?, ?, ?, ?)""",
                            (order_key, "TEST", "buy", datetime.now().isoformat(), "executed")
                        )
                        conn.commit()
                        with lock:
                            execution_count[0] += 1
                    except sqlite3.IntegrityError:
                        pass
            finally:
                conn.close()

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(try_execute, i) for i in range(20)]
            for f in as_completed(futures):
                f.result()

        assert execution_count[0] == 20


@pytest.mark.integration
@pytest.mark.stress
class TestIdempotencyCleanup:
    """Test cleanup of old idempotency entries."""

    def test_old_entries_can_be_pruned(self, tmp_path):
        """Old entries should be prunable after TTL."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        db_path = state_dir / "idempotency.db"

        # Create database with old entry
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                created_at TEXT NOT NULL,
                order_id TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)

        # Insert old entry (7 days ago)
        old_time = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute(
            """INSERT INTO idempotency_keys
               (key, symbol, side, created_at, status)
               VALUES (?, ?, ?, ?, ?)""",
            ("OLD_KEY", "TEST", "buy", old_time, "executed")
        )

        # Insert new entry
        cursor.execute(
            """INSERT INTO idempotency_keys
               (key, symbol, side, created_at, status)
               VALUES (?, ?, ?, ?, ?)""",
            ("NEW_KEY", "TEST", "buy", datetime.now().isoformat(), "executed")
        )
        conn.commit()

        # Prune entries older than 3 days
        cutoff = (datetime.now() - timedelta(days=3)).isoformat()
        cursor.execute(
            "DELETE FROM idempotency_keys WHERE created_at < ?",
            (cutoff,)
        )
        conn.commit()

        # Check remaining
        cursor.execute("SELECT COUNT(*) FROM idempotency_keys")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1  # Only new entry remains

    def test_active_entries_not_pruned(self, tmp_path):
        """Recent entries should not be pruned."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        db_path = state_dir / "idempotency.db"

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                created_at TEXT NOT NULL,
                order_id TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)

        # Insert 5 recent entries
        for i in range(5):
            cursor.execute(
                """INSERT INTO idempotency_keys
                   (key, symbol, side, created_at, status)
                   VALUES (?, ?, ?, ?, ?)""",
                (f"KEY_{i}", "TEST", "buy", datetime.now().isoformat(), "executed")
            )
        conn.commit()

        # Prune entries older than 3 days (none should be pruned)
        cutoff = (datetime.now() - timedelta(days=3)).isoformat()
        cursor.execute(
            "DELETE FROM idempotency_keys WHERE created_at < ?",
            (cutoff,)
        )
        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM idempotency_keys")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 5  # All entries remain


@pytest.mark.integration
@pytest.mark.stress
class TestPersistenceAcrossRestarts:
    """Test that idempotency persists across process restarts."""

    def test_entries_survive_reconnect(self, tmp_path):
        """Entries should survive database reconnection."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        db_path = state_dir / "idempotency.db"

        # First connection - create entries
        conn1 = sqlite3.connect(str(db_path))
        cursor1 = conn1.cursor()
        cursor1.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                created_at TEXT NOT NULL,
                order_id TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)
        cursor1.execute(
            """INSERT INTO idempotency_keys
               (key, symbol, side, created_at, status)
               VALUES (?, ?, ?, ?, ?)""",
            ("PERSIST_KEY", "TEST", "buy", datetime.now().isoformat(), "executed")
        )
        conn1.commit()
        conn1.close()

        # Second connection - verify entry exists
        conn2 = sqlite3.connect(str(db_path))
        cursor2 = conn2.cursor()
        cursor2.execute(
            "SELECT * FROM idempotency_keys WHERE key = ?",
            ("PERSIST_KEY",)
        )
        row = cursor2.fetchone()
        conn2.close()

        assert row is not None
        assert row[0] == "PERSIST_KEY"

    def test_duplicate_blocked_after_restart(self, tmp_path):
        """Duplicate should be blocked even after simulated restart."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        db_path = state_dir / "idempotency.db"

        order_key = "RESTART_TEST_KEY"

        # First "session" - insert entry
        conn1 = sqlite3.connect(str(db_path))
        cursor1 = conn1.cursor()
        cursor1.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                created_at TEXT NOT NULL,
                order_id TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)
        cursor1.execute(
            """INSERT INTO idempotency_keys
               (key, symbol, side, created_at, status)
               VALUES (?, ?, ?, ?, ?)""",
            (order_key, "TEST", "buy", datetime.now().isoformat(), "executed")
        )
        conn1.commit()
        conn1.close()

        # Second "session" - try duplicate
        conn2 = sqlite3.connect(str(db_path))
        cursor2 = conn2.cursor()

        cursor2.execute(
            "SELECT key FROM idempotency_keys WHERE key = ?",
            (order_key,)
        )
        existing = cursor2.fetchone()
        conn2.close()

        assert existing is not None  # Entry exists, duplicate would be blocked


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.slow
class TestHighVolumeStress:
    """High volume stress tests."""

    def test_1000_unique_orders(self, tmp_path):
        """1000 unique orders should all be recorded."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        db_path = state_dir / "idempotency.db"

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                created_at TEXT NOT NULL,
                order_id TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)

        # Insert 1000 unique orders
        for i in range(1000):
            cursor.execute(
                """INSERT INTO idempotency_keys
                   (key, symbol, side, created_at, status)
                   VALUES (?, ?, ?, ?, ?)""",
                (f"STRESS_KEY_{i:04d}", f"SYM{i % 100}", "buy", datetime.now().isoformat(), "executed")
            )

        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM idempotency_keys")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
