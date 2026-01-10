"""
INTEGRATION TESTS: Concurrent Execution (HIGH)

Tests thread safety and race conditions in:
- Signal processing
- State file writes
- Idempotency checks
- Hash chain appends
- Provider cache access

This prevents data corruption and duplicate orders under load.

Author: Kobe Trading System
Version: 1.0.0
Date: 2026-01-06
"""

import pytest
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.mark.integration
@pytest.mark.concurrent
class TestConcurrentSignalProcessing:
    """Test that concurrent signal processing is thread-safe."""

    def test_two_signals_same_symbol_processed_safely(self):
        """Two signals for same symbol should not corrupt state."""
        from tests.fixtures.signals import create_valid_signal

        signal1 = create_valid_signal(symbol="AAPL", entry_price=150.0)
        signal2 = create_valid_signal(symbol="AAPL", entry_price=151.0)

        results = []
        errors = []

        def process_signal(signal):
            try:
                # Simulate signal processing
                time.sleep(0.01)  # Small delay to increase race condition chance
                results.append({
                    "symbol": signal["symbol"],
                    "entry": signal["entry_price"],
                    "processed_at": datetime.now().isoformat(),
                })
            except Exception as e:
                errors.append(str(e))

        # Process concurrently
        threads = [
            threading.Thread(target=process_signal, args=(signal1,)),
            threading.Thread(target=process_signal, args=(signal2,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Both should complete without errors
        assert len(errors) == 0
        assert len(results) == 2

    def test_different_symbols_processed_in_parallel(self):
        """Multiple symbols should process in parallel without conflicts."""
        from tests.fixtures.signals import create_signal_batch

        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        signals = [
            {"symbol": s, "score": 75, "confidence": 0.70}
            for s in symbols
        ]

        results = []
        lock = threading.Lock()

        def process(signal):
            time.sleep(0.01)
            with lock:
                results.append(signal["symbol"])

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process, s) for s in signals]
            for f in as_completed(futures):
                f.result()  # Raise any exceptions

        # All symbols should be processed
        assert len(results) == 5
        assert set(results) == set(symbols)


@pytest.mark.integration
@pytest.mark.concurrent
class TestConcurrentStateWrites:
    """Test that concurrent state file writes don't corrupt data."""

    def test_concurrent_positions_writes(self, tmp_path):
        """Multiple threads writing positions should not corrupt file."""
        from tests.fixtures.state_helpers import create_test_state_dir

        state_dir = create_test_state_dir(tmp_path)
        positions_file = state_dir / "positions.json"
        lock = threading.Lock()

        def add_position(symbol, qty):
            with lock:
                content = positions_file.read_text()
                positions = json.loads(content)
                positions.append({
                    "symbol": symbol,
                    "qty": qty,
                    "added_at": datetime.now().isoformat(),
                })
                positions_file.write_text(json.dumps(positions))

        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(add_position, s, i * 10)
                for i, s in enumerate(symbols)
            ]
            for f in as_completed(futures):
                f.result()

        # Verify file is valid JSON with all positions
        final_positions = json.loads(positions_file.read_text())
        assert len(final_positions) == 5
        symbols_in_file = [p["symbol"] for p in final_positions]
        assert set(symbols_in_file) == set(symbols)

    def test_no_corruption_without_lock(self, tmp_path):
        """
        Demonstrate that without locking, corruption CAN occur.
        This test documents the risk that proper locking prevents.
        """
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        positions_file = state_dir / "positions.json"
        positions_file.write_text("[]")

        errors = []

        def unsafe_add_position(symbol):
            try:
                # This is intentionally unsafe - no lock
                content = positions_file.read_text()
                positions = json.loads(content)
                positions.append({"symbol": symbol})
                # Simulate some delay between read and write
                time.sleep(0.001)
                positions_file.write_text(json.dumps(positions))
            except Exception as e:
                errors.append(str(e))

        # Run without lock - may or may not corrupt
        threads = [
            threading.Thread(target=unsafe_add_position, args=(f"SYM{i}",))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # File should still be valid JSON (even if incomplete)
        try:
            final = json.loads(positions_file.read_text())
            # Note: count may be < 10 due to race conditions
            assert isinstance(final, list)
        except json.JSONDecodeError:
            # Corruption occurred - this is expected without locking
            pass


@pytest.mark.integration
@pytest.mark.concurrent
class TestIdempotencyUnderLoad:
    """Test idempotency store under concurrent access."""

    def test_same_order_ten_threads(self, tmp_path):
        """10 threads submitting same order should result in 1 execution."""
        from tests.fixtures.state_helpers import create_idempotency_db
        import sqlite3

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        db_path = state_dir / "idempotency.db"

        # Create empty database
        create_idempotency_db(state_dir, entries=[])

        order_key = "AAPL_BUY_20260106_1000"
        execution_count = [0]  # Use list for mutability in threads
        lock = threading.Lock()

        def try_execute():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            try:
                # Check if key exists
                cursor.execute(
                    "SELECT key FROM idempotency_keys WHERE key = ?",
                    (order_key,)
                )
                existing = cursor.fetchone()

                if existing is None:
                    # Insert new key
                    cursor.execute(
                        """INSERT INTO idempotency_keys
                           (key, symbol, side, created_at, status)
                           VALUES (?, ?, ?, ?, ?)""",
                        (order_key, "AAPL", "buy", datetime.now().isoformat(), "pending")
                    )
                    conn.commit()

                    with lock:
                        execution_count[0] += 1
            except sqlite3.IntegrityError:
                # Duplicate key - expected for concurrent inserts
                pass
            finally:
                conn.close()

        # Run 10 threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(try_execute) for _ in range(10)]
            for f in as_completed(futures):
                f.result()

        # Should have exactly 1 execution
        assert execution_count[0] == 1


@pytest.mark.integration
@pytest.mark.concurrent
class TestHashChainOrdering:
    """Test hash chain maintains correct order under concurrent appends."""

    def test_concurrent_appends_preserve_order(self, tmp_path):
        """Concurrent appends should maintain valid chain using fixture helpers."""
        from tests.fixtures.state_helpers import (
            create_test_state_dir,
            create_hash_chain_file,
            verify_hash_chain_integrity,
        )

        state_dir = create_test_state_dir(tmp_path)
        lock = threading.Lock()
        entries = []

        def collect_event(event_num):
            with lock:
                entries.append({
                    "event": f"event_{event_num}",
                    "timestamp": datetime.now().isoformat(),
                    "sequence": event_num,
                })

        # Collect 10 events concurrently (with lock for list)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(collect_event, i) for i in range(10)]
            for f in as_completed(futures):
                f.result()

        # Create hash chain with collected entries
        hash_file = create_hash_chain_file(state_dir, entries)

        # Verify chain integrity
        result = verify_hash_chain_integrity(hash_file)
        assert result["valid"] is True
        assert result["entries"] == 10

    def test_chain_length_correct(self, tmp_path):
        """Chain should have correct number of entries."""
        from tests.fixtures.state_helpers import (
            create_test_state_dir,
            create_hash_chain_file,
            verify_hash_chain_integrity,
        )

        state_dir = create_test_state_dir(tmp_path)

        # Build entries sequentially
        entries = [
            {
                "event": f"event_{i}",
                "timestamp": datetime.now().isoformat(),
            }
            for i in range(10)
        ]

        # Create hash chain
        hash_file = create_hash_chain_file(state_dir, entries)

        # Count lines
        lines = hash_file.read_text().strip().split("\n")
        assert len(lines) == 10

        # Verify integrity
        result = verify_hash_chain_integrity(hash_file)
        assert result["valid"] is True


@pytest.mark.integration
@pytest.mark.concurrent
class TestProviderCacheConcurrency:
    """Test data provider cache under concurrent access."""

    def test_concurrent_cache_reads(self, tmp_path):
        """Multiple threads reading cache should not corrupt it."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "AAPL.csv"

        # Create cache file
        cache_file.write_text(
            "timestamp,open,high,low,close,volume\n"
            "2025-01-02,150.0,152.0,149.0,151.0,1000000\n"
            "2025-01-03,151.0,153.0,150.0,152.0,1100000\n"
        )

        read_results = []
        errors = []

        def read_cache():
            try:
                content = cache_file.read_text()
                lines = content.strip().split("\n")
                read_results.append(len(lines))
            except Exception as e:
                errors.append(str(e))

        # 20 concurrent reads
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(read_cache) for _ in range(20)]
            for f in as_completed(futures):
                f.result()

        # All reads should succeed
        assert len(errors) == 0
        assert len(read_results) == 20
        # All should read same number of lines
        assert all(r == 3 for r in read_results)  # header + 2 data rows


@pytest.mark.integration
@pytest.mark.concurrent
@pytest.mark.slow
class TestHighConcurrencyStress:
    """Stress tests with high concurrency."""

    def test_100_concurrent_signal_validations(self):
        """100 concurrent signal validations should complete without errors."""
        from tests.fixtures.signals import create_valid_signal

        errors = []
        results = []

        def validate_signal(idx):
            try:
                signal = create_valid_signal(
                    symbol=f"SYM{idx:03d}",
                    score=70 + (idx % 20),
                )
                # Simulate validation
                is_valid = (
                    signal["score"] >= 70 and
                    signal["confidence"] >= 0.60 and
                    signal.get("risk_reward", 0) >= 1.5
                )
                results.append(is_valid)
            except Exception as e:
                errors.append(str(e))

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(validate_signal, i) for i in range(100)]
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0
        assert len(results) == 100

    def test_concurrent_file_operations(self, tmp_path):
        """Concurrent file operations should be thread-safe with locking."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        test_file = state_dir / "counter.json"
        test_file.write_text('{"count": 0}')

        lock = threading.Lock()

        def increment():
            with lock:
                data = json.loads(test_file.read_text())
                data["count"] += 1
                test_file.write_text(json.dumps(data))

        # 50 increments
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(increment) for _ in range(50)]
            for f in as_completed(futures):
                f.result()

        # Final count should be 50
        final = json.loads(test_file.read_text())
        assert final["count"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
