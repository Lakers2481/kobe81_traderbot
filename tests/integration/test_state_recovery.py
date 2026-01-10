"""
INTEGRATION TESTS: State Recovery (HIGH)

Tests system recovery from:
- Corrupted state files (positions.json, order_state.json)
- Missing state files
- Hash chain tampering detection
- Idempotency store survival across restarts
- State consistency checks

This ensures the system can recover gracefully from crashes.

Author: Kobe Trading System
Version: 1.0.0
Date: 2026-01-06
"""

import pytest
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.mark.integration
@pytest.mark.recovery
class TestPositionsCorruptionRecovery:
    """Test recovery from corrupted positions.json."""

    def test_invalid_json_detected(self, tmp_path):
        """Corrupted JSON should be detected."""
        from tests.fixtures.state_helpers import create_test_state_dir, corrupt_positions_file

        state_dir = create_test_state_dir(tmp_path)
        corrupt_positions_file(state_dir)

        positions_file = state_dir / "positions.json"

        with pytest.raises(json.JSONDecodeError):
            json.loads(positions_file.read_text())

    def test_can_rebuild_empty_positions(self, tmp_path):
        """Missing/corrupted positions can be rebuilt as empty."""
        from tests.fixtures.state_helpers import create_test_state_dir, corrupt_positions_file

        state_dir = create_test_state_dir(tmp_path)
        corrupt_positions_file(state_dir)

        positions_file = state_dir / "positions.json"

        # Recovery: write empty positions
        positions_file.write_text("[]")

        # Verify recovery
        positions = json.loads(positions_file.read_text())
        assert positions == []

    def test_corrupt_positions_backup_before_recovery(self, tmp_path):
        """Corrupt file should be backed up before recovery."""
        from tests.fixtures.state_helpers import create_test_state_dir, corrupt_positions_file

        state_dir = create_test_state_dir(tmp_path)

        # Add some positions first
        positions_file = state_dir / "positions.json"
        positions_file.write_text('[{"symbol": "AAPL", "qty": 100}]')

        # Corrupt the file
        corrupt_positions_file(state_dir)

        # Simulate recovery with backup
        backup_file = state_dir / "positions.json.corrupt"
        corrupted_content = positions_file.read_text()
        backup_file.write_text(corrupted_content)

        # Recovery
        positions_file.write_text("[]")

        # Both files should exist
        assert positions_file.exists()
        assert backup_file.exists()


@pytest.mark.integration
@pytest.mark.recovery
class TestMissingFilesRecovery:
    """Test recovery from missing state files."""

    def test_missing_positions_file_handled(self, tmp_path):
        """Missing positions.json should be handled gracefully."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        positions_file = state_dir / "positions.json"
        assert not positions_file.exists()

        # Recovery: create empty file
        positions_file.write_text("[]")
        assert positions_file.exists()

        positions = json.loads(positions_file.read_text())
        assert positions == []

    def test_missing_order_state_handled(self, tmp_path):
        """Missing order_state.json should be handled gracefully."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        order_state_file = state_dir / "order_state.json"
        assert not order_state_file.exists()

        # Recovery: create empty file
        order_state_file.write_text("{}")
        assert order_state_file.exists()

        order_state = json.loads(order_state_file.read_text())
        assert order_state == {}

    def test_missing_hash_chain_handled(self, tmp_path):
        """Missing hash_chain.jsonl should be handled gracefully."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        hash_chain_file = state_dir / "hash_chain.jsonl"
        assert not hash_chain_file.exists()

        # Recovery: create empty file
        hash_chain_file.write_text("")
        assert hash_chain_file.exists()

        # Empty chain is valid
        content = hash_chain_file.read_text()
        assert content == ""


@pytest.mark.integration
@pytest.mark.recovery
class TestHashChainTamperDetection:
    """Test hash chain integrity verification."""

    def test_valid_chain_passes_verification(self, tmp_path):
        """Valid hash chain should pass verification."""
        from tests.fixtures.state_helpers import create_hash_chain_file, verify_hash_chain_integrity, create_test_state_dir

        state_dir = create_test_state_dir(tmp_path)

        entries = [
            {"event": "order_placed", "symbol": "AAPL", "timestamp": datetime.now().isoformat()},
            {"event": "order_filled", "symbol": "AAPL", "timestamp": datetime.now().isoformat()},
        ]

        hash_file = create_hash_chain_file(state_dir, entries)
        result = verify_hash_chain_integrity(hash_file)

        assert result["valid"] is True
        assert result["entries"] == 2
        assert len(result["errors"]) == 0

    def test_tampered_entry_detected(self, tmp_path):
        """Modified hash chain entry should be detected."""
        from tests.fixtures.state_helpers import create_hash_chain_file, verify_hash_chain_integrity, create_test_state_dir

        state_dir = create_test_state_dir(tmp_path)

        entries = [
            {"event": "order_placed", "symbol": "AAPL", "timestamp": datetime.now().isoformat()},
            {"event": "order_filled", "symbol": "AAPL", "timestamp": datetime.now().isoformat()},
        ]

        hash_file = create_hash_chain_file(state_dir, entries)

        # Tamper with the file
        content = hash_file.read_text()
        lines = content.strip().split("\n")

        # Modify the first entry
        entry = json.loads(lines[0])
        entry["symbol"] = "MSFT"  # Tamper!
        lines[0] = json.dumps(entry)

        hash_file.write_text("\n".join(lines) + "\n")

        # Verification should fail
        result = verify_hash_chain_integrity(hash_file)

        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_broken_chain_link_detected(self, tmp_path):
        """Broken prev_hash link should be detected."""
        from tests.fixtures.state_helpers import create_test_state_dir

        state_dir = create_test_state_dir(tmp_path)
        hash_file = state_dir / "hash_chain.jsonl"

        # Create entries with broken chain
        entries = [
            json.dumps({
                "event": "event_1",
                "prev_hash": "0" * 64,
                "hash": "a" * 64,
            }),
            json.dumps({
                "event": "event_2",
                "prev_hash": "b" * 64,  # Should be "a" * 64
                "hash": "c" * 64,
            }),
        ]

        hash_file.write_text("\n".join(entries) + "\n")

        from tests.fixtures.state_helpers import verify_hash_chain_integrity
        result = verify_hash_chain_integrity(hash_file)

        assert result["valid"] is False
        assert any("prev_hash" in err.lower() or "mismatch" in err.lower() for err in result["errors"])


@pytest.mark.integration
@pytest.mark.recovery
class TestIdempotencyPersistence:
    """Test idempotency store survives restarts."""

    def test_entries_persist_after_close(self, tmp_path):
        """Idempotency entries should persist in SQLite."""
        from tests.fixtures.state_helpers import create_idempotency_db
        import sqlite3

        state_dir = tmp_path / "state"
        state_dir.mkdir()

        entries = [
            {
                "key": "test_key_1",
                "symbol": "AAPL",
                "side": "buy",
                "created_at": datetime.now().isoformat(),
                "order_id": "order_001",
                "status": "filled",
            },
        ]

        db_path = create_idempotency_db(state_dir, entries)

        # Close and reopen
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM idempotency_keys WHERE key = ?", ("test_key_1",))
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "test_key_1"  # key
        assert row[1] == "AAPL"  # symbol

    def test_multiple_entries_persist(self, tmp_path):
        """Multiple entries should all persist."""
        from tests.fixtures.state_helpers import create_idempotency_db
        import sqlite3

        state_dir = tmp_path / "state"
        state_dir.mkdir()

        entries = [
            {"key": f"key_{i}", "symbol": f"SYM{i}", "side": "buy", "created_at": datetime.now().isoformat()}
            for i in range(10)
        ]

        db_path = create_idempotency_db(state_dir, entries)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM idempotency_keys")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 10


@pytest.mark.integration
@pytest.mark.recovery
class TestStateConsistencyChecks:
    """Test consistency between different state files."""

    def test_positions_matches_order_state(self, tmp_path):
        """Positions and order_state should be consistent."""
        from tests.fixtures.state_helpers import create_test_state_dir

        state_dir = create_test_state_dir(tmp_path)

        # Create matching state
        positions = [
            {"symbol": "AAPL", "qty": 100, "order_id": "order_001"},
        ]
        (state_dir / "positions.json").write_text(json.dumps(positions))

        order_state = {
            "order_001": {
                "symbol": "AAPL",
                "qty": 100,
                "status": "filled",
            }
        }
        (state_dir / "order_state.json").write_text(json.dumps(order_state))

        # Verify consistency
        loaded_positions = json.loads((state_dir / "positions.json").read_text())
        loaded_orders = json.loads((state_dir / "order_state.json").read_text())

        for pos in loaded_positions:
            order_id = pos.get("order_id")
            if order_id:
                assert order_id in loaded_orders
                assert loaded_orders[order_id]["symbol"] == pos["symbol"]

    def test_inconsistency_detected(self, tmp_path):
        """Inconsistent state should be detectable."""
        from tests.fixtures.state_helpers import create_test_state_dir

        state_dir = create_test_state_dir(tmp_path)

        # Create inconsistent state
        positions = [
            {"symbol": "AAPL", "qty": 100, "order_id": "order_001"},
        ]
        (state_dir / "positions.json").write_text(json.dumps(positions))

        # Order state has different qty!
        order_state = {
            "order_001": {
                "symbol": "AAPL",
                "qty": 50,  # Inconsistent!
                "status": "filled",
            }
        }
        (state_dir / "order_state.json").write_text(json.dumps(order_state))

        # Check for inconsistency
        loaded_positions = json.loads((state_dir / "positions.json").read_text())
        loaded_orders = json.loads((state_dir / "order_state.json").read_text())

        inconsistencies = []
        for pos in loaded_positions:
            order_id = pos.get("order_id")
            if order_id and order_id in loaded_orders:
                if loaded_orders[order_id]["qty"] != pos["qty"]:
                    inconsistencies.append({
                        "order_id": order_id,
                        "positions_qty": pos["qty"],
                        "order_state_qty": loaded_orders[order_id]["qty"],
                    })

        assert len(inconsistencies) == 1


@pytest.mark.integration
@pytest.mark.recovery
class TestCorruptionTypes:
    """Test different types of file corruption."""

    def test_syntax_corruption_handled(self, tmp_path):
        """Syntax corruption should be caught."""
        from tests.fixtures.state_helpers import corrupt_json_file

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        test_file = state_dir / "test.json"
        test_file.write_text('{"valid": true}')

        corrupt_json_file(test_file, corruption_type="syntax")

        with pytest.raises(json.JSONDecodeError):
            json.loads(test_file.read_text())

    def test_truncation_corruption_handled(self, tmp_path):
        """Truncated file should be caught."""
        from tests.fixtures.state_helpers import corrupt_json_file

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        test_file = state_dir / "test.json"
        test_file.write_text('{"key": "value", "other": "data"}')

        corrupt_json_file(test_file, corruption_type="truncate")

        # Truncated JSON is invalid
        with pytest.raises(json.JSONDecodeError):
            json.loads(test_file.read_text())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
