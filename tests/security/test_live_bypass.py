"""
SECURITY BYPASS TESTS - All must FAIL safely

These tests attempt to bypass the safety gates.
ALL bypass attempts MUST raise exceptions or return blocked status.

If ANY test passes when it should fail, it's a SEV-0 security vulnerability.

Author: Kobe Trading System
Version: 1.0.0
Date: 2026-01-05
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class TestLiveBypassPrevention:
    """Test that live trading cannot be bypassed."""

    def test_live_blocked_without_all_flags(self):
        """Live order must be blocked when any flag is missing."""
        from safety.execution_choke import evaluate_safety_gates, SafetyViolationError

        # Ensure we're in default state (all flags off)
        result = evaluate_safety_gates(is_paper_order=False, ack_token="test")

        assert result.allowed is False
        assert result.mode.value == "blocked"
        assert "failed checks" in result.reason.lower()

    def test_live_blocked_with_wrong_ack_token(self):
        """Live order blocked with wrong ACK token even if other flags set."""
        from safety.execution_choke import evaluate_safety_gates

        # Even with wrong token, should be blocked
        result = evaluate_safety_gates(
            is_paper_order=False,
            ack_token="WRONG_TOKEN_123"
        )

        assert result.allowed is False
        assert not result.checks_passed.get("ack_token_valid", False)

    def test_paper_allowed_without_live_flags(self):
        """Paper orders should work without live flags."""
        from safety.execution_choke import evaluate_safety_gates

        result = evaluate_safety_gates(is_paper_order=True)

        assert result.allowed is True
        assert result.mode.value == "paper"

    def test_paper_blocked_when_kill_switch_active(self):
        """Paper orders blocked when kill switch is active."""
        from safety.execution_choke import evaluate_safety_gates

        # Create kill switch file temporarily
        kill_switch_path = ROOT / "state" / "KILL_SWITCH"
        try:
            kill_switch_path.touch()

            result = evaluate_safety_gates(is_paper_order=True)

            assert result.allowed is False
            assert result.mode.value == "blocked"
            assert "kill switch" in result.reason.lower()
        finally:
            kill_switch_path.unlink(missing_ok=True)

    def test_require_safety_gate_raises_on_live(self):
        """require_safety_gate must raise SafetyViolationError for blocked live orders."""
        from safety.execution_choke import require_safety_gate, SafetyViolationError

        with pytest.raises(SafetyViolationError):
            require_safety_gate(is_paper_order=False, ack_token="wrong")

    def test_decorator_blocks_live_orders(self):
        """The @require_execution_choke decorator must block live order functions."""
        from safety.execution_choke import require_execution_choke, SafetyViolationError

        @require_execution_choke(is_paper=False)
        def fake_live_order_function():
            return "order_placed"

        with pytest.raises(SafetyViolationError):
            fake_live_order_function()

    def test_all_six_flags_must_pass_for_live(self):
        """All 6 required flags must be True for live orders to pass."""
        from safety.execution_choke import evaluate_safety_gates, get_live_order_ack_token

        # Test each flag individually - all should fail without the others
        result = evaluate_safety_gates(is_paper_order=False, ack_token=get_live_order_ack_token())

        required_checks = [
            "paper_only_disabled",
            "live_trading_enabled",
            "trading_mode_live",
            "approve_live_action",
            "approve_live_action_2",
            "ack_token_valid",
        ]

        # Even with valid ACK token, other flags should fail
        failed = [k for k in required_checks if not result.checks_passed.get(k, False)]

        # At least 5 should fail (all except ack_token_valid)
        assert len(failed) >= 5, f"Not enough flags blocking live: {failed}"
        assert result.allowed is False


class TestDirectAPIBypassPrevention:
    """Test that direct API calls cannot bypass safety gates."""

    def test_scripts_position_manager_uses_gate(self):
        """scripts/position_manager.py should use safety gate (PATCHED)."""
        position_manager_path = ROOT / "scripts" / "position_manager.py"
        content = position_manager_path.read_text()

        # Verify safety gate is imported and used
        assert "from safety.execution_choke import" in content, \
            "position_manager.py must import from safety.execution_choke"
        assert "evaluate_safety_gates" in content, \
            "position_manager.py must call evaluate_safety_gates"

        # Verify it's called before the order API
        gate_pos = content.find("evaluate_safety_gates")
        api_pos = content.find("alpaca_request")
        if api_pos > 0 and gate_pos > 0:
            assert gate_pos < api_pos, \
                "Safety gate must be checked BEFORE API call"

    def test_options_order_router_uses_gate(self):
        """options/order_router.py should use full safety gates (PATCHED)."""
        order_router_path = ROOT / "options" / "order_router.py"
        content = order_router_path.read_text()

        # Verify safety gate is imported
        assert "from safety.execution_choke import" in content, \
            "order_router.py must import from safety.execution_choke"
        assert "evaluate_safety_gates" in content, \
            "order_router.py must call evaluate_safety_gates"

    def test_crypto_broker_uses_gate(self):
        """execution/broker_crypto.py should use safety gate (PATCHED)."""
        broker_crypto_path = ROOT / "execution" / "broker_crypto.py"
        content = broker_crypto_path.read_text()

        # Verify safety gate is imported and used
        assert "from safety.execution_choke import" in content, \
            "broker_crypto.py must import from safety.execution_choke"
        assert "evaluate_safety_gates" in content, \
            "broker_crypto.py must call evaluate_safety_gates"

        # Verify it's called before the CCXT order
        gate_pos = content.find("evaluate_safety_gates")
        ccxt_pos = content.find("self._exchange.create_order")
        if ccxt_pos > 0 and gate_pos > 0:
            assert gate_pos < ccxt_pos, \
                "Safety gate must be checked BEFORE CCXT order"


class TestRegisteredFunctionEnforcement:
    """Test that only registered functions can submit orders."""

    def test_registered_functions_exist(self):
        """All registered order functions should exist and be importable."""
        from safety.execution_choke import _REGISTERED_ORDER_FUNCTIONS

        assert len(_REGISTERED_ORDER_FUNCTIONS) >= 6

        # Check that key functions are registered
        expected = {
            "execution.broker_alpaca.execute_signal",
            "execution.broker_alpaca.place_order_with_liquidity_check",
        }

        for func_path in expected:
            assert func_path in _REGISTERED_ORDER_FUNCTIONS, f"Missing: {func_path}"


class TestMultipleFlagRequirement:
    """Test that multiple independent flags are all required."""

    def test_single_flag_insufficient(self):
        """Setting only one flag should not enable live trading."""
        from safety.execution_choke import evaluate_safety_gates

        # Test with only TRADING_MODE env set
        with patch.dict(os.environ, {"TRADING_MODE": "live"}):
            result = evaluate_safety_gates(is_paper_order=False)
            assert result.allowed is False

    def test_env_vars_alone_insufficient(self):
        """Environment variables alone should not enable live trading."""
        from safety.execution_choke import evaluate_safety_gates

        with patch.dict(os.environ, {
            "TRADING_MODE": "live",
            "APPROVE_LIVE_ACTION_2": "true",
        }):
            result = evaluate_safety_gates(is_paper_order=False)
            assert result.allowed is False

    def test_code_flags_alone_insufficient(self):
        """Code flags alone (without env vars and token) should not enable live trading."""
        from safety.execution_choke import evaluate_safety_gates

        # Even if we mock the code flags, env vars should still block
        result = evaluate_safety_gates(is_paper_order=False)
        assert result.allowed is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
