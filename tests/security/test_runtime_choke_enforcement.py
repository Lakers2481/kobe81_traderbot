"""
RUNTIME TESTS: Prove choke point is called at runtime, not via string matching.
Uses monkeypatching to intercept low-level order functions and verify call chain.

These tests PROVE that:
1. Safety gate is called BEFORE any order API call
2. If gate blocks, the API is NEVER called
3. Kill switch blocks ALL paths
4. Logging occurs even on blocked orders

Author: Kobe Trading System
Version: 1.0.0
Date: 2026-01-06
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class TestRuntimeChokeEnforcement:
    """Prove choke is called at RUNTIME, not just present in source."""

    def test_alpaca_broker_calls_gate_before_any_api(self):
        """AlpacaBroker.place_order() must call gate BEFORE any network call."""
        with patch("execution.broker_alpaca.evaluate_safety_gates") as mock_gate:
            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.reason = "blocked"
            mock_gate.return_value = mock_result

            # Patch the actual API call to track if it's reached
            with patch("execution.broker_alpaca.place_ioc_limit") as mock_api:
                from execution.broker_alpaca import AlpacaBroker
                from execution.broker_base import Order, OrderSide, OrderType

                broker = AlpacaBroker(api_key="test", api_secret="test", paper=True)
                order = Order(symbol="AAPL", side=OrderSide.BUY, qty=10, order_type=OrderType.MARKET)

                result = broker.place_order(order)

                # PROOF: Gate was called
                assert mock_gate.called, "Safety gate NOT called - BYPASS DETECTED"

                # PROOF: API was NOT called (blocked)
                assert not mock_api.called, "API called despite gate blocking - BYPASS DETECTED"

                # PROOF: Result indicates blocked
                assert result.success is False
                assert "safety_gate_blocked" in result.error_message

    def test_alpaca_crypto_broker_calls_gate_before_api(self):
        """AlpacaCryptoBroker.place_order() must call gate BEFORE any API call."""
        try:
            from alpaca.trading.client import TradingClient
        except ImportError:
            pytest.skip("alpaca-py not installed")

        with patch("execution.broker_alpaca_crypto.evaluate_safety_gates") as mock_gate:
            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.reason = "blocked"
            mock_result.mode = MagicMock(value="blocked")
            mock_gate.return_value = mock_result

            from execution.broker_alpaca_crypto import AlpacaCryptoBroker
            from execution.broker_base import Order, OrderSide, OrderType

            with patch("execution.broker_alpaca_crypto.TradingClient"):
                with patch("execution.broker_alpaca_crypto.CryptoHistoricalDataClient"):
                    broker = AlpacaCryptoBroker(
                        api_key="test_key",
                        api_secret="test_secret",
                        paper=True
                    )
                    broker._connected = True

                    order = Order(symbol="BTC/USD", side=OrderSide.BUY, qty=0.01, order_type=OrderType.MARKET)

                    # Patch the submit method to ensure it's not called
                    with patch.object(broker, "_trading_client") as mock_client:
                        mock_client.submit_order = MagicMock()

                        result = broker.place_order(order)

                        # PROOF: Gate called
                        assert mock_gate.called, "Safety gate NOT called - BYPASS DETECTED"

                        # PROOF: API NOT called
                        assert not mock_client.submit_order.called, "API called despite blocking"

                        # PROOF: Result indicates blocked
                        assert result.success is False
                        assert "safety_gate_blocked" in result.error_message

    def test_ccxt_crypto_broker_calls_gate_before_ccxt(self):
        """CryptoBroker.place_order() must call gate BEFORE CCXT create_order."""
        try:
            import ccxt
        except ImportError:
            pytest.skip("ccxt not installed")

        with patch("execution.broker_crypto.evaluate_safety_gates") as mock_gate:
            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.reason = "blocked"
            mock_result.mode = MagicMock(value="blocked")
            mock_gate.return_value = mock_result

            with patch("execution.broker_crypto.ccxt") as mock_ccxt:
                mock_exchange = MagicMock()
                mock_ccxt.binance.return_value = mock_exchange

                from execution.broker_crypto import CryptoBroker
                from execution.broker_base import Order, OrderSide, OrderType

                broker = CryptoBroker(exchange="binance", sandbox=True)
                broker._connected = True
                broker._markets_loaded = True

                order = Order(symbol="BTC/USDT", side=OrderSide.BUY, qty=0.01, order_type=OrderType.MARKET)
                result = broker.place_order(order)

                # PROOF: Gate called
                assert mock_gate.called, "Safety gate NOT called - BYPASS DETECTED"

                # PROOF: CCXT NOT called
                assert not mock_exchange.create_order.called, "CCXT called despite blocking"

    def test_position_manager_calls_gate_before_alpaca_request(self):
        """close_position() must call gate BEFORE alpaca_request."""
        with patch("scripts.position_manager.evaluate_safety_gates") as mock_gate:
            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.reason = "blocked"
            mock_gate.return_value = mock_result

            with patch("scripts.position_manager.alpaca_request") as mock_api:
                from scripts.position_manager import close_position

                result = close_position("AAPL", 100, "sell")

                # PROOF: Gate called
                assert mock_gate.called, "Safety gate NOT called - BYPASS DETECTED"

                # PROOF: API NOT called
                assert not mock_api.called, "API called despite blocking"

                # PROOF: Returns False when blocked
                assert result is False

    def test_options_router_calls_gate_before_execution(self):
        """OptionsOrderRouter.submit_order() must call gate BEFORE any execution."""
        with patch("options.order_router.evaluate_safety_gates") as mock_gate:
            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.reason = "blocked"
            mock_gate.return_value = mock_result

            from options.order_router import (
                OptionsOrderRouter, OptionsOrder, OptionsOrderType,
                OptionsOrderSide, OptionsOrderLeg
            )

            router = OptionsOrderRouter(paper_mode=True)

            with patch.object(router, "_simulate_order") as mock_sim:
                with patch.object(router, "_execute_alpaca_order") as mock_exec:
                    leg = OptionsOrderLeg(
                        contract_symbol="AAPL260117C00155000",
                        side=OptionsOrderSide.BUY_TO_OPEN,
                        quantity=1,
                    )
                    order = OptionsOrder(symbol="AAPL", legs=[leg], order_type=OptionsOrderType.MARKET)

                    result = router.submit_order(order)

                    # PROOF: Gate called
                    assert mock_gate.called, "Safety gate NOT called"

                    # PROOF: Neither execution path called
                    assert not mock_sim.called, "Simulation called despite blocking"
                    assert not mock_exec.called, "Live execution called despite blocking"

    def test_paper_broker_calls_gate_even_for_simulation(self):
        """PaperBroker.place_order() must call gate even for simulated orders."""
        with patch("execution.broker_paper.evaluate_safety_gates") as mock_gate:
            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.reason = "blocked"
            mock_result.mode = MagicMock(value="blocked")
            mock_gate.return_value = mock_result

            from execution.broker_paper import PaperBroker
            from execution.broker_base import Order, OrderSide, OrderType

            broker = PaperBroker(initial_equity=100000)
            broker.connect()

            order = Order(symbol="AAPL", side=OrderSide.BUY, qty=10, order_type=OrderType.MARKET)
            result = broker.place_order(order)

            # PROOF: Gate called even for paper
            assert mock_gate.called, "Safety gate NOT called for paper - BYPASS DETECTED"
            assert result.success is False


class TestKillSwitchBlocking:
    """Verify kill switch blocks ALL order paths."""

    def test_kill_switch_blocks_paper_orders(self):
        """Kill switch must block paper orders."""
        with patch("safety.execution_choke._check_kill_switch", return_value=True):
            from safety.execution_choke import evaluate_safety_gates

            result = evaluate_safety_gates(is_paper_order=True)
            assert result.allowed is False, "Paper order allowed with kill switch!"
            assert "kill switch" in result.reason.lower()

    def test_kill_switch_blocks_live_orders(self):
        """Kill switch must block live orders."""
        with patch("safety.execution_choke._check_kill_switch", return_value=True):
            from safety.execution_choke import evaluate_safety_gates

            result = evaluate_safety_gates(is_paper_order=False)
            assert result.allowed is False, "Live order allowed with kill switch!"


class TestIdempotencyAndLogging:
    """Verify logging is invoked for all gate evaluations."""

    def test_gate_logs_on_allowed(self):
        """Allowed orders must be logged."""
        with patch("safety.execution_choke.jlog") as mock_log:
            with patch("safety.execution_choke._check_kill_switch", return_value=False):
                from safety.execution_choke import require_safety_gate

                result = require_safety_gate(is_paper_order=True)

                # PROOF: Logging was called
                assert mock_log.called, "Logging NOT invoked - audit trail broken"

                # Verify allowed
                assert result.allowed is True

    def test_gate_logs_on_blocked(self):
        """Blocked orders must still be logged."""
        with patch("safety.execution_choke.jlog") as mock_log:
            with patch("safety.execution_choke._check_kill_switch", return_value=True):
                from safety.execution_choke import require_safety_gate, SafetyViolationError

                with pytest.raises(SafetyViolationError):
                    require_safety_gate(is_paper_order=True)

                # PROOF: Logging was called even on block
                assert mock_log.called, "Logging NOT invoked - audit trail broken"


class TestCallOrderVerification:
    """Verify the exact order of operations in order execution."""

    def test_gate_called_before_order_creation(self):
        """Gate MUST be called before any order is created."""
        call_sequence = []

        def track_gate(*args, **kwargs):
            call_sequence.append("GATE")
            result = MagicMock()
            result.allowed = False
            result.reason = "test"
            return result

        def track_order(*args, **kwargs):
            call_sequence.append("ORDER")
            return MagicMock()

        with patch("execution.broker_alpaca.evaluate_safety_gates", side_effect=track_gate):
            with patch("execution.broker_alpaca.place_ioc_limit", side_effect=track_order):
                from execution.broker_alpaca import AlpacaBroker
                from execution.broker_base import Order, OrderSide, OrderType

                broker = AlpacaBroker(api_key="test", api_secret="test", paper=True)
                order = Order(symbol="AAPL", side=OrderSide.BUY, qty=10, order_type=OrderType.MARKET)

                broker.place_order(order)

                # PROOF: Gate was called first (or only GATE if blocked)
                assert len(call_sequence) >= 1, "No calls recorded"
                assert call_sequence[0] == "GATE", f"Gate not called first! Sequence: {call_sequence}"

                # PROOF: ORDER was NOT called after GATE blocked
                assert "ORDER" not in call_sequence, f"ORDER called after GATE blocked! Sequence: {call_sequence}"


class TestAllBrokersHaveGate:
    """Verify ALL broker types have safety gate integration."""

    def test_all_brokers_import_safety_gate(self):
        """All broker modules must import evaluate_safety_gates."""
        broker_files = [
            ROOT / "execution" / "broker_alpaca.py",
            ROOT / "execution" / "broker_paper.py",
            ROOT / "execution" / "broker_crypto.py",
            ROOT / "execution" / "broker_alpaca_crypto.py",
        ]

        for broker_file in broker_files:
            if broker_file.exists():
                content = broker_file.read_text()
                assert "evaluate_safety_gates" in content, f"{broker_file.name} missing evaluate_safety_gates import"
                assert "safety.execution_choke" in content, f"{broker_file.name} missing safety.execution_choke import"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
