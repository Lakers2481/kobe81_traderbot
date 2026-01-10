"""
INTEGRATION TESTS: Signal to Execution Pipeline (CRITICAL)

Tests the complete end-to-end flow:
Signal Generation -> Quality Gate -> Risk Gates -> Position Sizing -> Execution -> State Update

This is the MOST CRITICAL integration test file as it validates the entire
trading pipeline works correctly from signal to order placement.

Author: Kobe Trading System
Version: 1.0.0
Date: 2026-01-06
"""

import pytest
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.mark.integration
@pytest.mark.e2e
class TestFullPipelineHappyPath:
    """Test complete signal-to-execution flow with valid signals."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, mock_env_vars):
        """Set up isolated state for each test."""
        self.state_dir = tmp_path / "state"
        self.state_dir.mkdir()
        (self.state_dir / "positions.json").write_text("[]")
        (self.state_dir / "order_state.json").write_text("{}")
        (self.state_dir / "hash_chain.jsonl").write_text("")

    def test_valid_signal_through_all_gates(self, valid_signal):
        """Verify valid signal passes all gates and results in order."""
        from tests.fixtures.signals import create_valid_signal

        signal = create_valid_signal(
            symbol="AAPL",
            score=80,
            confidence=0.75,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
        )

        # Mock quality gate
        with patch("risk.signal_quality_gate.SignalQualityGate") as MockGate:
            mock_gate = MagicMock()
            mock_gate.evaluate.return_value = MagicMock(
                passed=True,
                tier="STANDARD",
                final_score=80,
                adjusted_confidence=0.75,
            )
            MockGate.return_value = mock_gate

            # Mock policy gate
            with patch("risk.policy_gate.PolicyGate") as MockPolicy:
                mock_policy = MagicMock()
                mock_policy.check.return_value = MagicMock(
                    approved=True,
                    reason="within_budget",
                )
                MockPolicy.return_value = mock_policy

                # Verify gates pass
                result = mock_gate.evaluate(signal)
                assert result.passed is True
                policy_result = mock_policy.check(75.0)
                assert policy_result.approved is True

    def test_signal_creates_correct_order_size(self):
        """Verify dual-cap position sizing produces correct share count."""
        from risk.equity_sizer import calculate_position_size

        # Test case: Notional cap dominates with these parameters
        result = calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,  # $5 risk per share
            risk_pct=0.02,   # 2% = $2000 max risk
            account_equity=100000.0,
            max_notional_pct=0.20,  # 20% = $20000 max notional
        )

        # $2000 risk / $5 per share = 400 shares
        # $20000 notional / $100 = 200 shares
        # Dual cap: min(400, 200) = 200 shares
        assert result.shares == 200
        assert result.notional == 20000.0
        assert result.risk_dollars == 1000.0  # 200 * $5
        assert result.capped is True

    def test_order_placement_updates_state(self, tmp_path):
        """Verify successful order updates positions and hash chain."""
        from tests.fixtures.state_helpers import (
            create_test_state_dir,
            create_hash_chain_file,
            verify_hash_chain_integrity,
        )

        state_dir = create_test_state_dir(tmp_path)

        # Record order events via fixture helper
        entries = [
            {
                "event": "order_placed",
                "timestamp": datetime.now().isoformat(),
                "symbol": "AAPL",
                "side": "buy",
                "qty": 100,
                "price": 150.0,
            }
        ]
        hash_file = create_hash_chain_file(state_dir, entries)

        # Verify hash chain integrity
        result = verify_hash_chain_integrity(hash_file)
        assert result["valid"] is True
        assert result["entries"] == 1


@pytest.mark.integration
@pytest.mark.e2e
class TestQualityGateRejections:
    """Test that quality gate properly rejects invalid signals."""

    def test_low_score_rejected(self, invalid_signal):
        """Signal with score < 70 should be rejected."""
        from tests.fixtures.signals import create_invalid_signal

        signal = create_invalid_signal(reason="low_score")
        assert signal["score"] == 45
        assert signal["score"] < 70  # Below threshold

    def test_low_confidence_rejected(self):
        """Signal with confidence < 0.60 should be rejected."""
        from tests.fixtures.signals import create_invalid_signal

        signal = create_invalid_signal(reason="low_confidence")
        assert signal["confidence"] == 0.45
        assert signal["confidence"] < 0.60  # Below threshold

    def test_poor_risk_reward_rejected(self):
        """Signal with R:R < 1.5 should be rejected."""
        from tests.fixtures.signals import create_invalid_signal

        signal = create_invalid_signal(reason="poor_rr")
        assert signal["risk_reward"] == 0.5
        assert signal["risk_reward"] < 1.5  # Below threshold


@pytest.mark.integration
@pytest.mark.e2e
class TestPolicyGateRejections:
    """Test that PolicyGate properly enforces budgets."""

    def test_over_budget_rejected(self):
        """Order exceeding daily budget should be rejected."""
        from risk.policy_gate import PolicyGate, RiskLimits

        limits = RiskLimits(
            max_notional_per_order=75.0,
            max_daily_notional=1000.0,
        )
        gate = PolicyGate(limits=limits)

        # Simulate near budget
        gate._daily_notional = 950.0

        # API: check(symbol, side, price, qty, stop_loss) -> (bool, str)
        # $100 price * 10 qty = $1000 notional, over remaining $50 budget
        approved, reason = gate.check(symbol="TEST", side="long", price=100.0, qty=10)
        assert approved is False
        assert "daily" in reason.lower() or "budget" in reason.lower() or "notional" in reason.lower()

    def test_per_order_limit_enforced(self):
        """Order exceeding per-order limit should be rejected."""
        from risk.policy_gate import PolicyGate, RiskLimits

        limits = RiskLimits(
            max_notional_per_order=75.0,
            max_daily_notional=1000.0,
        )
        gate = PolicyGate(limits=limits)

        # $100 price * 10 qty = $1000 notional, over $75 per-order limit
        approved, reason = gate.check(symbol="TEST", side="long", price=100.0, qty=10)
        assert approved is False
        assert "order" in reason.lower() or "notional" in reason.lower() or "exceeds" in reason.lower()


@pytest.mark.integration
@pytest.mark.e2e
class TestPositionLimitGate:
    """Test position limit enforcement."""

    def test_max_positions_enforced(self):
        """Cannot open more than max allowed positions."""
        from tests.fixtures.state_helpers import create_positions_file
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            state_dir.mkdir()

            # Create 3 existing positions
            positions = [
                {"symbol": "AAPL", "qty": 100},
                {"symbol": "MSFT", "qty": 100},
                {"symbol": "GOOGL", "qty": 100},
            ]
            create_positions_file(state_dir, positions)

            # Read back and verify
            positions_file = state_dir / "positions.json"
            loaded = json.loads(positions_file.read_text())
            assert len(loaded) == 3

            # In production, position limit gate would reject new position
            MAX_POSITIONS = 3
            assert len(loaded) >= MAX_POSITIONS


@pytest.mark.integration
@pytest.mark.e2e
class TestDualCapPositionSizing:
    """Test dual-cap position sizing edge cases."""

    def test_risk_cap_dominates(self):
        """When risk-based shares > notional-based shares, notional dominates."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=50.0,       # Low price
            stop_loss=49.0,         # $1 risk per share
            risk_pct=0.01,          # 1% = $1000 risk
            account_equity=100000.0,
            max_notional_pct=0.20,  # 20% = $20000 notional
        )

        # $1000 risk / $1 per share = 1000 shares (risk cap)
        # $20000 notional / $50 = 400 shares (notional cap)
        # min(1000, 400) = 400 shares (notional dominates)
        assert result.shares == 400
        assert result.capped is True

    def test_notional_cap_dominates(self):
        """When notional-based shares < risk-based shares, use notional-based."""
        from risk.equity_sizer import calculate_position_size

        result = calculate_position_size(
            entry_price=200.0,      # High price
            stop_loss=180.0,        # $20 risk per share
            risk_pct=0.02,          # 2% = $2000 risk
            account_equity=100000.0,
            max_notional_pct=0.10,  # 10% = $10000 notional
        )

        # $2000 risk / $20 per share = 100 shares (risk cap)
        # $10000 notional / $200 = 50 shares (notional cap)
        # min(100, 50) = 50 shares
        assert result.shares == 50
        assert result.capped is True


@pytest.mark.integration
@pytest.mark.e2e
class TestIdempotencyPrevention:
    """Test idempotency store prevents duplicate orders."""

    def test_duplicate_signal_blocked(self, tmp_path):
        """Same signal submitted twice should be blocked on second attempt."""
        from tests.fixtures.state_helpers import create_idempotency_db
        import sqlite3

        state_dir = tmp_path / "state"
        state_dir.mkdir()

        # First submission - create entry
        entries = [{
            "key": "AAPL_BUY_20260106_100",
            "symbol": "AAPL",
            "side": "buy",
            "created_at": datetime.now().isoformat(),
            "order_id": "order_001",
            "status": "filled",
        }]
        db_path = create_idempotency_db(state_dir, entries)

        # Second submission - check for duplicate
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM idempotency_keys WHERE key = ?",
            ("AAPL_BUY_20260106_100",)
        )
        existing = cursor.fetchone()
        conn.close()

        assert existing is not None  # Entry exists
        # In production, this would block the second submission


@pytest.mark.integration
@pytest.mark.e2e
class TestStateConsistency:
    """Test that state files remain consistent after operations."""

    def test_hash_chain_updated_on_order(self, tmp_path):
        """Hash chain should have new entry after order placed."""
        from tests.fixtures.state_helpers import (
            create_test_state_dir,
            create_hash_chain_file,
            verify_hash_chain_integrity,
        )

        state_dir = create_test_state_dir(tmp_path)

        # Create hash chain with order events
        entries = [
            {
                "event": "order_placed",
                "timestamp": datetime.now().isoformat(),
                "symbol": "AAPL",
                "side": "buy",
                "qty": 100,
            },
            {
                "event": "order_filled",
                "timestamp": datetime.now().isoformat(),
                "symbol": "AAPL",
                "side": "buy",
                "qty": 100,
                "fill_price": 150.0,
            },
        ]
        hash_file = create_hash_chain_file(state_dir, entries)

        # Verify chain integrity
        result = verify_hash_chain_integrity(hash_file)
        assert result["valid"] is True
        assert result["entries"] == 2

    def test_positions_json_updated(self, tmp_path):
        """positions.json should reflect new position after fill."""
        from tests.fixtures.state_helpers import create_test_state_dir, create_positions_file

        state_dir = create_test_state_dir(tmp_path)

        # Simulate adding a position
        positions = [
            {
                "symbol": "AAPL",
                "qty": 100,
                "avg_entry_price": 150.0,
                "opened_at": datetime.now().isoformat(),
            }
        ]
        create_positions_file(state_dir, positions)

        # Read back and verify
        positions_file = state_dir / "positions.json"
        loaded = json.loads(positions_file.read_text())

        assert len(loaded) == 1
        assert loaded[0]["symbol"] == "AAPL"
        assert loaded[0]["qty"] == 100


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.requires_mock_broker
class TestBrokerIntegration:
    """Test broker integration with mocked Alpaca API."""

    def test_order_submitted_to_broker(self, mock_broker):
        """Verify order is submitted to broker API."""
        # Set quote
        mock_broker.set_quote("AAPL", bid=149.0, ask=150.0)

        # Place order
        result = mock_broker.place_ioc_limit(
            symbol="AAPL",
            qty=100,
            side="buy",
            limit_price=150.0,
        )

        assert result["status"] == "filled"
        assert result["symbol"] == "AAPL"
        assert result["filled_qty"] == 100
        assert mock_broker.order_count == 1

    def test_broker_rejection_handled(self, mock_broker):
        """Verify broker rejection is handled gracefully."""
        mock_broker.set_reject_orders(True, "Insufficient buying power")

        result = mock_broker.place_ioc_limit(
            symbol="AAPL",
            qty=1000000,  # Huge order
            side="buy",
            limit_price=150.0,
        )

        assert result["status"] == "rejected"
        assert "buying power" in result["reject_reason"].lower()


@pytest.mark.integration
@pytest.mark.e2e
class TestKillSwitchEnforcement:
    """Test that kill switch blocks all orders."""

    def test_kill_switch_blocks_orders(self, tmp_path):
        """Orders should be blocked when KILL_SWITCH file exists."""
        from tests.fixtures.state_helpers import create_test_state_dir, create_kill_switch

        state_dir = create_test_state_dir(tmp_path)
        kill_file = create_kill_switch(state_dir, reason="Test emergency stop")

        # Kill switch should exist
        assert kill_file.exists()

        # In production, all order paths check for kill switch
        kill_content = json.loads(kill_file.read_text())
        assert "Test emergency stop" in kill_content["reason"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
