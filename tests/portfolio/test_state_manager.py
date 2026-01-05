"""
Tests for portfolio/state_manager.py - Central state management.
"""
from __future__ import annotations

import json
import pytest
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portfolio.state_manager import StateManager, get_state_manager, reset_state_manager


@pytest.fixture
def temp_state_dir(tmp_path):
    """Create a temporary state directory."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return state_dir


@pytest.fixture
def state_manager(temp_state_dir):
    """Create a StateManager with temp directory."""
    return StateManager(state_dir=temp_state_dir)


class TestStateManagerBasics:
    """Basic state manager functionality tests."""

    def test_creates_directories(self, temp_state_dir):
        """State manager creates required directories."""
        sm = StateManager(state_dir=temp_state_dir)

        assert (temp_state_dir / "autonomous").exists()
        assert (temp_state_dir / "reconciliation").exists()
        assert (temp_state_dir / "watchlist").exists()
        assert (temp_state_dir / "backups").exists()

    def test_get_positions_empty(self, state_manager):
        """Get positions returns empty dict when no state exists."""
        positions = state_manager.get_positions()
        assert positions == {}

    def test_set_and_get_positions(self, state_manager):
        """Set and get positions works correctly."""
        positions = {
            "AAPL": {"qty": 100, "entry": 150.0},
            "MSFT": {"qty": 50, "entry": 300.0},
        }

        result = state_manager.set_positions(positions)
        assert result is True

        retrieved = state_manager.get_positions()
        assert retrieved == positions

    def test_update_position(self, state_manager):
        """Update a single position works correctly."""
        # Set initial positions
        state_manager.set_positions({"AAPL": {"qty": 100}})

        # Update one
        state_manager.update_position("MSFT", {"qty": 50})

        positions = state_manager.get_positions()
        assert "AAPL" in positions
        assert "MSFT" in positions

    def test_remove_position(self, state_manager):
        """Remove position works correctly."""
        state_manager.set_positions({"AAPL": {"qty": 100}, "MSFT": {"qty": 50}})

        state_manager.remove_position("AAPL")

        positions = state_manager.get_positions()
        assert "AAPL" not in positions
        assert "MSFT" in positions

    def test_remove_nonexistent_position(self, state_manager):
        """Remove nonexistent position returns True without error."""
        result = state_manager.remove_position("UNKNOWN")
        assert result is True


class TestAtomicWrites:
    """Tests for atomic write behavior."""

    def test_creates_backup_on_write(self, state_manager, temp_state_dir):
        """Writing creates backup of previous state."""
        # Write initial state
        state_manager.set_positions({"AAPL": {"qty": 100}})

        # Write new state (should create backup)
        state_manager.set_positions({"MSFT": {"qty": 50}})

        # Check backup exists
        backups = list((temp_state_dir / "backups").glob("position_state_*.json"))
        assert len(backups) >= 1

    def test_write_survives_parent_dir_missing(self, state_manager, temp_state_dir):
        """Write creates parent directories if needed."""
        # Use a custom path that doesn't exist
        custom_path = temp_state_dir / "custom" / "nested" / "state.json"

        # Should not raise
        result = state_manager._write_json(custom_path, {"test": True})
        assert result is True
        assert custom_path.exists()


class TestWeeklyBudget:
    """Tests for weekly budget state."""

    def test_get_weekly_budget_defaults(self, state_manager):
        """Get weekly budget returns defaults when no state."""
        budget = state_manager.get_weekly_budget()
        assert "notional_used" in budget
        assert "max_notional" in budget

    def test_update_weekly_notional(self, state_manager):
        """Update weekly notional adds to usage."""
        state_manager.set_weekly_budget({
            "week_start": "2026-01-05",
            "notional_used": 1000.0,
            "max_notional": 10000.0,
        })

        state_manager.update_weekly_notional(500.0)

        budget = state_manager.get_weekly_budget()
        assert budget["notional_used"] == 1500.0


class TestBrainState:
    """Tests for brain state."""

    def test_get_brain_state_defaults(self, state_manager):
        """Get brain state returns defaults."""
        state = state_manager.get_brain_state()
        assert state["status"] == "idle"

    def test_set_brain_state(self, state_manager):
        """Set brain state works correctly."""
        state_manager.set_brain_state({
            "status": "active",
            "last_heartbeat": "2026-01-05T10:00:00",
            "current_task": "scanning",
        })

        retrieved = state_manager.get_brain_state()
        assert retrieved["status"] == "active"


class TestReconciliation:
    """Tests for reconciliation state."""

    def test_save_reconciliation_adds_timestamp(self, state_manager):
        """Save reconciliation adds timestamp."""
        result = {"positions_matched": True, "mismatches": []}
        state_manager.save_reconciliation(result)

        retrieved = state_manager.get_reconciliation()
        assert "timestamp" in retrieved


class TestGenericState:
    """Tests for generic state access."""

    def test_get_state_with_default(self, state_manager):
        """Get state returns default for missing state."""
        data = state_manager.get_state("nonexistent", default={"empty": True})
        assert data == {"empty": True}

    def test_set_and_get_generic_state(self, state_manager):
        """Set and get generic state works."""
        state_manager.set_state("custom_state", {"key": "value"})

        data = state_manager.get_state("custom_state")
        assert data == {"key": "value"}


class TestConcurrency:
    """Tests for concurrent access safety."""

    def test_concurrent_writes_are_safe(self, state_manager):
        """Multiple threads can write safely without corruption."""
        errors = []
        iterations = 50

        def writer(thread_id: int):
            try:
                for i in range(iterations):
                    state_manager.update_position(
                        f"SYM{thread_id}",
                        {"qty": i, "thread": thread_id}
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # All positions should be present
        positions = state_manager.get_positions()
        for i in range(5):
            assert f"SYM{i}" in positions

    def test_state_lock_context_manager(self, state_manager):
        """State lock context manager provides exclusive access."""
        lock_acquired = []

        def worker(thread_id: int):
            with state_manager.state_lock("positions"):
                lock_acquired.append(thread_id)
                time.sleep(0.01)  # Hold lock briefly
                lock_acquired.append(thread_id)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should see pairs (each thread acquires twice in a row)
        for i in range(0, len(lock_acquired), 2):
            assert lock_acquired[i] == lock_acquired[i + 1]


class TestGlobalInstance:
    """Tests for global state manager instance."""

    def test_get_state_manager_singleton(self):
        """get_state_manager returns same instance."""
        reset_state_manager()

        sm1 = get_state_manager()
        sm2 = get_state_manager()

        assert sm1 is sm2

        reset_state_manager()

    def test_reset_state_manager(self):
        """reset_state_manager creates new instance."""
        reset_state_manager()

        sm1 = get_state_manager()
        reset_state_manager()
        sm2 = get_state_manager()

        assert sm1 is not sm2

        reset_state_manager()


class TestErrorHandling:
    """Tests for error handling."""

    def test_handles_invalid_json(self, state_manager, temp_state_dir):
        """Handles invalid JSON gracefully."""
        # Write invalid JSON directly
        bad_path = temp_state_dir / "position_state.json"
        bad_path.write_text("not valid json {{{")

        # Should return default, not crash
        positions = state_manager.get_positions()
        assert positions == {}
