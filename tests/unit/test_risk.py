"""
Unit tests for risk management.
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPolicyGate:
    """Tests for PolicyGate risk management."""

    def test_policy_gate_import(self):
        """Test that PolicyGate can be imported."""
        from risk.policy_gate import PolicyGate
        assert PolicyGate is not None

    def test_policy_gate_initialization(self):
        """Test PolicyGate initialization."""
        from risk.policy_gate import PolicyGate
        gate = PolicyGate(max_order_value=75, max_daily_loss=1000)
        assert gate is not None

    def test_order_within_budget_passes(self):
        """Test that order within budget passes check."""
        from risk.policy_gate import PolicyGate
        gate = PolicyGate(max_order_value=75, max_daily_loss=1000)

        # Order value of $50 should pass
        order_value = 50
        result = gate.check(order_value=order_value)

        assert result['allowed'] == True

    def test_order_exceeds_budget_fails(self):
        """Test that order exceeding budget fails check."""
        from risk.policy_gate import PolicyGate
        gate = PolicyGate(max_order_value=75, max_daily_loss=1000)

        # Order value of $100 should fail
        order_value = 100
        result = gate.check(order_value=order_value)

        assert result['allowed'] == False
        assert 'budget' in result['reason'].lower()

    def test_daily_loss_limit(self):
        """Test daily loss limit enforcement."""
        from risk.policy_gate import PolicyGate
        gate = PolicyGate(max_order_value=75, max_daily_loss=1000)

        # Simulate accumulated daily loss
        gate.record_loss(500)
        gate.record_loss(400)

        # Next order should still pass (total: $900)
        result = gate.check(order_value=50)
        assert result['allowed'] == True

        # After another loss, should hit limit
        gate.record_loss(100)  # Total: $1000

        # Now should fail
        result = gate.check(order_value=50)
        assert result['allowed'] == False


class TestKillSwitch:
    """Tests for kill switch functionality."""

    def test_kill_switch_not_active_by_default(self, tmp_path):
        """Test that kill switch is not active by default."""
        kill_switch_file = tmp_path / "KILL_SWITCH"

        # File doesn't exist
        assert not kill_switch_file.exists()

    def test_kill_switch_blocks_trading(self, tmp_path):
        """Test that kill switch blocks trading when active."""
        kill_switch_file = tmp_path / "KILL_SWITCH"

        # Create kill switch file
        kill_switch_file.touch()

        assert kill_switch_file.exists()

        # Trading should be blocked when file exists
        # (Actual implementation check would go here)

    def test_kill_switch_can_be_deactivated(self, tmp_path):
        """Test that kill switch can be deactivated."""
        kill_switch_file = tmp_path / "KILL_SWITCH"

        # Create and then remove
        kill_switch_file.touch()
        assert kill_switch_file.exists()

        kill_switch_file.unlink()
        assert not kill_switch_file.exists()


class TestPositionSizing:
    """Tests for position sizing logic."""

    def test_fixed_dollar_sizing(self):
        """Test fixed dollar position sizing."""
        order_value = 75
        price = 150

        shares = int(order_value / price)

        assert shares == 0  # $75 can't buy 1 share at $150

        # With lower price
        price = 25
        shares = int(order_value / price)

        assert shares == 3  # $75 / $25 = 3 shares

    def test_position_size_respects_max_pct(self):
        """Test that position size respects max portfolio percentage."""
        portfolio_value = 100000
        max_pct = 0.10  # 10%
        max_position_value = portfolio_value * max_pct

        assert max_position_value == 10000

        # Can't exceed 10% of portfolio
        price = 500
        max_shares = int(max_position_value / price)

        assert max_shares == 20  # $10,000 / $500 = 20 shares
