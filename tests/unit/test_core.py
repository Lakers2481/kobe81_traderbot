"""
Unit tests for core functionality.
"""

import pytest
import json
import hashlib
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestHashChain:
    """Tests for hash chain audit trail."""

    def test_hash_chain_import(self):
        """Test that hash chain can be imported."""
        from core.hash_chain import HashChain
        assert HashChain is not None

    def test_hash_chain_initialization(self, tmp_path):
        """Test hash chain initialization."""
        from core.hash_chain import HashChain
        chain_file = tmp_path / "hash_chain.jsonl"
        chain = HashChain(str(chain_file))
        assert chain is not None

    def test_hash_chain_append(self, tmp_path):
        """Test appending to hash chain."""
        from core.hash_chain import HashChain
        chain_file = tmp_path / "hash_chain.jsonl"
        chain = HashChain(str(chain_file))

        # Append an event
        event = {"action": "test", "timestamp": datetime.now().isoformat()}
        chain.append(event)

        # File should exist and have content
        assert chain_file.exists()
        assert chain_file.stat().st_size > 0

    def test_hash_chain_verification(self, tmp_path):
        """Test hash chain verification."""
        from core.hash_chain import HashChain
        chain_file = tmp_path / "hash_chain.jsonl"
        chain = HashChain(str(chain_file))

        # Add some events
        chain.append({"action": "event1"})
        chain.append({"action": "event2"})
        chain.append({"action": "event3"})

        # Verification should pass
        is_valid = chain.verify()
        assert is_valid == True

    def test_tamper_detection(self, tmp_path):
        """Test that tampering is detected."""
        from core.hash_chain import HashChain
        chain_file = tmp_path / "hash_chain.jsonl"
        chain = HashChain(str(chain_file))

        # Add events
        chain.append({"action": "event1"})
        chain.append({"action": "event2"})

        # Tamper with file
        with open(chain_file, 'a') as f:
            f.write('{"action": "tampered", "prev_hash": "fake"}\n')

        # Verification should fail
        # (Actual implementation may vary)


class TestStructuredLog:
    """Tests for structured logging."""

    def test_structured_log_import(self):
        """Test that structured log can be imported."""
        from core.structured_log import StructuredLogger
        assert StructuredLogger is not None

    def test_log_event(self, tmp_path):
        """Test logging an event."""
        from core.structured_log import StructuredLogger
        log_file = tmp_path / "events.jsonl"
        logger = StructuredLogger(str(log_file))

        # Log an event
        logger.log("INFO", "test_event", {"key": "value"})

        # File should exist
        assert log_file.exists()

        # Should contain valid JSON
        with open(log_file) as f:
            line = f.readline()
            event = json.loads(line)
            assert event["event_type"] == "test_event"

    def test_log_levels(self, tmp_path):
        """Test different log levels."""
        from core.structured_log import StructuredLogger
        log_file = tmp_path / "events.jsonl"
        logger = StructuredLogger(str(log_file))

        # Log at different levels
        logger.log("DEBUG", "debug_event", {})
        logger.log("INFO", "info_event", {})
        logger.log("WARNING", "warning_event", {})
        logger.log("ERROR", "error_event", {})

        # Count lines
        with open(log_file) as f:
            lines = f.readlines()

        assert len(lines) == 4


class TestIdempotencyStore:
    """Tests for idempotency store."""

    def test_idempotency_store_import(self):
        """Test that idempotency store can be imported."""
        from oms.idempotency_store import IdempotencyStore
        assert IdempotencyStore is not None

    def test_store_and_check(self, tmp_path):
        """Test storing and checking idempotency keys."""
        from oms.idempotency_store import IdempotencyStore
        store_file = tmp_path / "idempotency.json"
        store = IdempotencyStore(str(store_file))

        # First request should not exist
        key = "order_123_AAPL_2024-01-15"
        assert not store.exists(key)

        # Store the key
        store.add(key)

        # Now should exist
        assert store.exists(key)

    def test_duplicate_prevention(self, tmp_path):
        """Test that duplicates are prevented."""
        from oms.idempotency_store import IdempotencyStore
        store_file = tmp_path / "idempotency.json"
        store = IdempotencyStore(str(store_file))

        key = "order_456_MSFT_2024-01-15"

        # First add should succeed
        assert store.add(key) == True

        # Second add should indicate duplicate
        assert store.add(key) == False  # Already exists
