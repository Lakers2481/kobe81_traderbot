"""
Unit tests for core functionality.
"""

import json
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestHashChain:
    """Tests for hash chain audit trail."""

    def test_hash_chain_import(self):
        """Test that hash chain functions can be imported."""
        from core.hash_chain import append_block, verify_chain
        assert append_block is not None
        assert verify_chain is not None

    def test_hash_chain_append(self, tmp_path, monkeypatch):
        """Test appending to hash chain."""
        from core import hash_chain

        # Redirect CHAIN_FILE to temp path
        chain_file = tmp_path / "hash_chain.jsonl"
        monkeypatch.setattr(hash_chain, 'CHAIN_FILE', chain_file)

        # Append an event
        event = {"action": "test", "timestamp": datetime.now().isoformat()}
        block_hash = hash_chain.append_block(event)

        # File should exist and have content
        assert chain_file.exists()
        assert chain_file.stat().st_size > 0
        assert isinstance(block_hash, str)
        assert len(block_hash) == 64  # SHA256 hex length

    def test_hash_chain_verification(self, tmp_path, monkeypatch):
        """Test hash chain verification."""
        from core import hash_chain

        chain_file = tmp_path / "hash_chain.jsonl"
        monkeypatch.setattr(hash_chain, 'CHAIN_FILE', chain_file)

        # Add some events
        hash_chain.append_block({"action": "event1"})
        hash_chain.append_block({"action": "event2"})
        hash_chain.append_block({"action": "event3"})

        # Verification should pass
        is_valid = hash_chain.verify_chain()
        assert is_valid

    def test_empty_chain_is_valid(self, tmp_path, monkeypatch):
        """Test that empty/non-existent chain is valid."""
        from core import hash_chain

        chain_file = tmp_path / "nonexistent_chain.jsonl"
        monkeypatch.setattr(hash_chain, 'CHAIN_FILE', chain_file)

        # Non-existent file should return True
        assert hash_chain.verify_chain()

    def test_tamper_detection(self, tmp_path, monkeypatch):
        """Test that tampering is detected."""
        from core import hash_chain

        chain_file = tmp_path / "hash_chain.jsonl"
        monkeypatch.setattr(hash_chain, 'CHAIN_FILE', chain_file)

        # Add events
        hash_chain.append_block({"action": "event1"})
        hash_chain.append_block({"action": "event2"})

        # Tamper with file by appending a malformed block
        with open(chain_file, 'a') as f:
            f.write('{"prev_hash": "fake_hash", "payload": {"action": "tampered"}, "this_hash": "invalid"}\n')

        # Verification should fail
        is_valid = hash_chain.verify_chain()
        assert not is_valid


class TestStructuredLog:
    """Tests for structured logging."""

    def test_structured_log_import(self):
        """Test that structured log can be imported."""
        from core.structured_log import jlog
        assert jlog is not None
        assert callable(jlog)

    def test_log_event(self, tmp_path, monkeypatch):
        """Test logging an event."""
        from core import structured_log

        log_file = tmp_path / "events.jsonl"
        monkeypatch.setattr(structured_log, 'LOG_FILE', log_file)
        # Close existing handler before resetting
        if structured_log._file_handler is not None:
            structured_log._file_handler.close()
        monkeypatch.setattr(structured_log, '_file_handler', None)

        # Log an event
        structured_log.jlog("test_event", key="value")

        # Close handler to avoid ResourceWarning
        if structured_log._file_handler is not None:
            structured_log._file_handler.close()
            structured_log._file_handler = None

        # File should exist
        assert log_file.exists()

        # Should contain valid JSON
        with open(log_file) as f:
            line = f.readline()
            event = json.loads(line)
            assert event["event"] == "test_event"
            assert event["key"] == "value"
            assert "ts" in event
            assert event["level"] == "INFO"

    def test_log_levels(self, tmp_path, monkeypatch):
        """Test different log levels."""
        from core import structured_log

        log_file = tmp_path / "events.jsonl"
        monkeypatch.setattr(structured_log, 'LOG_FILE', log_file)
        # Close existing handler before resetting
        if structured_log._file_handler is not None:
            structured_log._file_handler.close()
        monkeypatch.setattr(structured_log, '_file_handler', None)

        # Log at different levels
        structured_log.jlog("debug_event", level="DEBUG")
        structured_log.jlog("info_event", level="INFO")
        structured_log.jlog("warning_event", level="WARNING")
        structured_log.jlog("error_event", level="ERROR")

        # Close handler to avoid ResourceWarning
        if structured_log._file_handler is not None:
            structured_log._file_handler.close()
            structured_log._file_handler = None

        # Count lines
        with open(log_file) as f:
            lines = f.readlines()

        assert len(lines) == 4

        # Verify levels
        for i, expected_level in enumerate(["DEBUG", "INFO", "WARNING", "ERROR"]):
            event = json.loads(lines[i])
            assert event["level"] == expected_level


class TestIdempotencyStore:
    """Tests for idempotency store."""

    def test_idempotency_store_import(self):
        """Test that idempotency store can be imported."""
        from oms.idempotency_store import IdempotencyStore
        assert IdempotencyStore is not None

    def test_store_and_check(self, tmp_path):
        """Test storing and checking idempotency keys."""
        from oms.idempotency_store import IdempotencyStore
        store_file = tmp_path / "idempotency.sqlite"
        store = IdempotencyStore(str(store_file))

        # First request should not exist
        decision_id = "order_123_AAPL_2024-01-15"
        assert not store.exists(decision_id)

        # Store the key
        store.put(decision_id, "idem_key_abc")

        # Now should exist
        assert store.exists(decision_id)

        # Should be able to retrieve the key
        retrieved = store.get(decision_id)
        assert retrieved == "idem_key_abc"

    def test_duplicate_prevention(self, tmp_path):
        """Test that duplicates are prevented."""
        from oms.idempotency_store import IdempotencyStore
        store_file = tmp_path / "idempotency.sqlite"
        store = IdempotencyStore(str(store_file))

        decision_id = "order_456_MSFT_2024-01-15"

        # First store should work
        assert not store.exists(decision_id)
        store.put(decision_id, "key_1")
        assert store.exists(decision_id)

        # Storing again with same decision_id is silently ignored (INSERT OR IGNORE)
        store.put(decision_id, "key_2")

        # Should still have original key
        assert store.get(decision_id) == "key_1"

    def test_nonexistent_key_returns_none(self, tmp_path):
        """Test that getting a nonexistent key returns None."""
        from oms.idempotency_store import IdempotencyStore
        store_file = tmp_path / "idempotency.sqlite"
        store = IdempotencyStore(str(store_file))

        result = store.get("nonexistent_key")
        assert result is None
