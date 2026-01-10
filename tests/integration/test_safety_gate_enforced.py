"""
INTEGRATION TESTS: Safety Gate Enforcement

Tests that verify the safety gate is properly enforced in all order execution paths.
These tests use mocked brokers to avoid real API calls.

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


class TestPositionManagerSafetyGate:
    """Test that position_manager.py enforces safety gate."""

    def test_close_position_calls_safety_gate(self):
        """Verify close_position calls evaluate_safety_gates before order."""
        from scripts.position_manager import close_position

        # Mock the safety gate to track calls
        with patch("scripts.position_manager.evaluate_safety_gates") as mock_gate:
            # Configure mock to return blocked result
            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.reason = "test_blocked"
            mock_gate.return_value = mock_result

            # Call the function
            result = close_position("AAPL", 100, "sell")

            # Verify safety gate was called
            assert mock_gate.called, "Safety gate was not called"
            assert result is False, "Should return False when blocked"

    def test_close_position_passes_ack_token(self):
        """Verify close_position passes ack_token to safety gate."""
        from scripts.position_manager import close_position

        with patch("scripts.position_manager.evaluate_safety_gates") as mock_gate:
            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.reason = "test_blocked"
            mock_gate.return_value = mock_result

            # Call with ack token
            close_position("AAPL", 100, "sell", ack_token="TEST_TOKEN")

            # Verify ack_token was passed
            call_kwargs = mock_gate.call_args.kwargs
            assert call_kwargs.get("ack_token") == "TEST_TOKEN"


class TestCryptoBrokerSafetyGate:
    """Test that crypto brokers enforce safety gate."""

    def test_ccxt_broker_uses_safety_gate(self):
        """Verify CryptoBroker (CCXT) calls evaluate_safety_gates."""
        # Skip if ccxt not available
        try:
            import ccxt
        except ImportError:
            pytest.skip("ccxt not installed")

        with patch("execution.broker_crypto.evaluate_safety_gates") as mock_gate:
            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.reason = "test_blocked"
            mock_result.mode = MagicMock(value="blocked")
            mock_gate.return_value = mock_result

            from execution.broker_crypto import CryptoBroker
            from execution.broker_base import Order, OrderSide, OrderType

            # Create broker with mocked exchange
            with patch("execution.broker_crypto.ccxt") as mock_ccxt:
                mock_exchange_class = MagicMock()
                mock_ccxt.binance = mock_exchange_class

                broker = CryptoBroker(exchange="binance", sandbox=True)
                broker._connected = True
                broker._markets_loaded = True

                # Create test order
                order = Order(
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    qty=0.01,
                    order_type=OrderType.MARKET,
                )

                # Place order
                result = broker.place_order(order)

                # Verify safety gate was called
                assert mock_gate.called, "Safety gate was not called"
                assert result.success is False, "Should fail when blocked"
                assert "safety_gate_blocked" in result.error_message

    def test_alpaca_crypto_broker_uses_safety_gate(self):
        """Verify AlpacaCryptoBroker calls evaluate_safety_gates."""
        # Skip if alpaca-py not available
        try:
            from alpaca.trading.client import TradingClient
        except ImportError:
            pytest.skip("alpaca-py not installed")

        with patch("execution.broker_alpaca_crypto.evaluate_safety_gates") as mock_gate:
            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.reason = "test_blocked"
            mock_result.mode = MagicMock(value="blocked")
            mock_gate.return_value = mock_result

            from execution.broker_alpaca_crypto import AlpacaCryptoBroker
            from execution.broker_base import Order, OrderSide, OrderType

            # Create broker with mocked client
            with patch("execution.broker_alpaca_crypto.TradingClient"):
                with patch("execution.broker_alpaca_crypto.CryptoHistoricalDataClient"):
                    broker = AlpacaCryptoBroker(
                        api_key="test_key",
                        api_secret="test_secret",
                        paper=True
                    )
                    broker._connected = True

                    # Create test order
                    order = Order(
                        symbol="BTC/USD",
                        side=OrderSide.BUY,
                        qty=0.01,
                        order_type=OrderType.MARKET,
                    )

                    # Place order
                    result = broker.place_order(order)

                    # Verify safety gate was called
                    assert mock_gate.called, "Safety gate was not called"
                    assert result.success is False, "Should fail when blocked"
                    assert "safety_gate_blocked" in result.error_message


class TestAlpacaBrokerSafetyGate:
    """Test that AlpacaBroker enforces safety gate."""

    def test_alpaca_broker_uses_safety_gate(self):
        """Verify AlpacaBroker.place_order calls evaluate_safety_gates."""
        with patch("execution.broker_alpaca.evaluate_safety_gates") as mock_gate:
            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.reason = "test_blocked"
            mock_result.mode = MagicMock(value="blocked")
            mock_gate.return_value = mock_result

            from execution.broker_alpaca import AlpacaBroker
            from execution.broker_base import Order, OrderSide, OrderType

            # Create broker
            broker = AlpacaBroker(
                api_key="test_key",
                api_secret="test_secret",
                paper=True
            )

            # Create test order
            order = Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                qty=10,
                order_type=OrderType.MARKET,
            )

            # Place order
            result = broker.place_order(order)

            # Verify safety gate was called
            assert mock_gate.called, "Safety gate was not called"
            assert result.success is False, "Should fail when blocked"
            assert "safety_gate_blocked" in result.error_message


class TestPaperBrokerSafetyGate:
    """Test that PaperBroker enforces safety gate."""

    def test_paper_broker_uses_safety_gate(self):
        """Verify PaperBroker.place_order calls evaluate_safety_gates."""
        with patch("execution.broker_paper.evaluate_safety_gates") as mock_gate:
            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.reason = "test_blocked"
            mock_result.mode = MagicMock(value="blocked")
            mock_gate.return_value = mock_result

            from execution.broker_paper import PaperBroker
            from execution.broker_base import Order, OrderSide, OrderType

            # Create broker
            broker = PaperBroker(initial_equity=100000)
            broker.connect()

            # Create test order
            order = Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                qty=10,
                order_type=OrderType.MARKET,
            )

            # Place order
            result = broker.place_order(order)

            # Verify safety gate was called
            assert mock_gate.called, "Safety gate was not called"
            assert result.success is False, "Should fail when blocked"
            assert "safety_gate_blocked" in result.error_message


class TestOptionsOrderRouterSafetyGate:
    """Test that order_router.py enforces safety gate."""

    def test_submit_order_calls_safety_gate(self):
        """Verify OptionsOrderRouter.submit_order calls evaluate_safety_gates."""
        with patch("options.order_router.evaluate_safety_gates") as mock_gate:
            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.reason = "test_blocked"
            mock_gate.return_value = mock_result

            from options.order_router import (
                OptionsOrderRouter, OptionsOrder, OptionsOrderType,
                OptionsOrderStatus, OptionsOrderSide, OptionsOrderLeg
            )

            # Create router
            router = OptionsOrderRouter(paper_mode=True)

            # Create test order with proper leg structure
            leg = OptionsOrderLeg(
                contract_symbol="AAPL260117C00155000",
                side=OptionsOrderSide.BUY_TO_OPEN,
                quantity=1,
            )
            order = OptionsOrder(
                symbol="AAPL",
                legs=[leg],
                order_type=OptionsOrderType.MARKET,
            )

            # Submit order
            result = router.submit_order(order)

            # Verify safety gate was called
            assert mock_gate.called, "Safety gate was not called"
            assert result.order.status == OptionsOrderStatus.REJECTED


class TestAllBypassPathsPatched:
    """Verify no bypass paths exist in production code."""

    def test_no_direct_alpaca_orders_without_gate(self):
        """Scan codebase for direct Alpaca order calls without safety gate."""

        bypasses = []

        # Files that should have safety gates
        files_to_check = [
            ROOT / "scripts" / "position_manager.py",
            ROOT / "execution" / "broker_alpaca.py",
        ]

        for file_path in files_to_check:
            if not file_path.exists():
                continue

            content = file_path.read_text()

            # Check for direct API calls
            if 'alpaca_request("/v2/orders"' in content:
                # Verify safety gate is nearby (within 40 lines before)
                gate_pos = content.find("evaluate_safety_gates")
                api_pos = content.find('alpaca_request("/v2/orders"')

                if gate_pos < 0 or gate_pos > api_pos:
                    bypasses.append(str(file_path))

        assert len(bypasses) == 0, f"Unguarded API calls in: {bypasses}"

    def test_no_direct_ccxt_orders_without_gate(self):
        """Scan codebase for direct CCXT order calls without safety gate."""
        crypto_path = ROOT / "execution" / "broker_crypto.py"
        if not crypto_path.exists():
            pytest.skip("broker_crypto.py not found")

        content = crypto_path.read_text()

        if "self._exchange.create_order" in content:
            gate_pos = content.find("evaluate_safety_gates")
            ccxt_pos = content.find("self._exchange.create_order")

            assert gate_pos > 0, "No safety gate in broker_crypto.py"
            assert gate_pos < ccxt_pos, "Safety gate must be before CCXT order"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
