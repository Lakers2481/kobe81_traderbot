"""
Tests for Broker-Liquidity Gate Integration.

Tests that the liquidity gate is properly wired into the broker execution flow.
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from execution.broker_alpaca import (
    OrderResult,
    BrokerExecutionResult,
    get_liquidity_gate,
    set_liquidity_gate,
    enable_liquidity_gate,
    is_liquidity_gate_enabled,
    check_liquidity_for_order,
    place_order_with_liquidity_check,
    execute_signal,
    get_quote_with_sizes,
    get_avg_volume,
    _liquidity_gate,
)
from oms.order_state import OrderRecord, OrderStatus
from risk.liquidity_gate import LiquidityGate, LiquidityCheck, LiquidityIssue


class TestLiquidityGateIntegration:
    """Tests for liquidity gate integration in broker."""

    def test_get_liquidity_gate_creates_default(self):
        """Should create default gate if none set."""
        # Reset global
        import execution.broker_alpaca as broker
        broker._liquidity_gate = None

        gate = get_liquidity_gate()

        assert gate is not None
        assert isinstance(gate, LiquidityGate)
        assert gate.min_adv_usd == 100_000
        assert gate.max_spread_pct == 0.50

    def test_set_liquidity_gate(self):
        """Should allow setting custom gate."""
        custom_gate = LiquidityGate(
            min_adv_usd=500_000,
            max_spread_pct=0.25,
        )

        set_liquidity_gate(custom_gate)
        gate = get_liquidity_gate()

        assert gate.min_adv_usd == 500_000
        assert gate.max_spread_pct == 0.25

        # Reset for other tests
        import execution.broker_alpaca as broker
        broker._liquidity_gate = None

    def test_enable_disable_liquidity_gate(self):
        """Should toggle liquidity gate on/off."""
        enable_liquidity_gate(False)
        assert is_liquidity_gate_enabled() == False

        enable_liquidity_gate(True)
        assert is_liquidity_gate_enabled() == True


class TestCheckLiquidityForOrder:
    """Tests for check_liquidity_for_order function."""

    @patch('execution.broker_alpaca.get_quote_with_sizes')
    @patch('execution.broker_alpaca.get_avg_volume')
    def test_check_passes_good_stock(self, mock_volume, mock_quotes):
        """Should pass for stocks with good liquidity."""
        mock_quotes.return_value = (149.98, 150.02, 1000, 1000)
        mock_volume.return_value = 50_000_000

        check = check_liquidity_for_order(
            symbol='AAPL',
            qty=100,
            price=150.0,
        )

        assert check.passed
        assert check.adv_usd > 0
        assert check.spread_pct < 0.5

    @patch('execution.broker_alpaca.get_quote_with_sizes')
    @patch('execution.broker_alpaca.get_avg_volume')
    def test_check_fails_low_volume(self, mock_volume, mock_quotes):
        """Should fail for stocks with low ADV."""
        mock_quotes.return_value = (9.98, 10.02, 100, 100)
        mock_volume.return_value = 1_000  # Only $10k ADV

        check = check_liquidity_for_order(
            symbol='LOWVOL',
            qty=100,
            price=10.0,
        )

        assert not check.passed
        assert LiquidityIssue.INSUFFICIENT_ADV in check.issues

    @patch('execution.broker_alpaca.get_quote_with_sizes')
    @patch('execution.broker_alpaca.get_avg_volume')
    def test_check_fails_wide_spread(self, mock_volume, mock_quotes):
        """Should fail for stocks with wide spread."""
        mock_quotes.return_value = (49.00, 51.00, 1000, 1000)  # 4% spread
        mock_volume.return_value = 1_000_000

        check = check_liquidity_for_order(
            symbol='WIDESPREAD',
            qty=100,
            price=50.0,
        )

        assert not check.passed
        assert LiquidityIssue.WIDE_SPREAD in check.issues

    @patch('execution.broker_alpaca.get_quote_with_sizes')
    def test_check_fails_no_quote(self, mock_quotes):
        """Should fail when no quote available."""
        mock_quotes.return_value = (None, None, None, None)

        check = check_liquidity_for_order(
            symbol='NOQUOTE',
            qty=100,
        )

        assert not check.passed
        assert "Unable to fetch" in check.reason


class TestPlaceOrderWithLiquidityCheck:
    """Tests for place_order_with_liquidity_check function."""

    def _create_test_order(self, symbol='AAPL', qty=100, price=150.0) -> OrderRecord:
        """Create a test order record."""
        return OrderRecord(
            decision_id=f"DEC_TEST_{symbol}",
            signal_id=f"SIG_TEST_{symbol}",
            symbol=symbol,
            side="BUY",
            qty=qty,
            limit_price=price,
            tif="IOC",
            order_type="IOC_LIMIT",
            idempotency_key=f"IDK_TEST_{symbol}",
            created_at=datetime.utcnow(),
        )

    @patch('execution.broker_alpaca.check_liquidity_for_order')
    @patch('execution.broker_alpaca.place_ioc_limit')
    def test_places_order_when_liquidity_passes(self, mock_place, mock_check):
        """Should place order when liquidity check passes."""
        enable_liquidity_gate(True)

        mock_check.return_value = LiquidityCheck(
            symbol='AAPL',
            passed=True,
            adv_usd=5_000_000,
            spread_pct=0.02,
        )

        order = self._create_test_order()
        order.status = OrderStatus.SUBMITTED
        mock_place.return_value = BrokerExecutionResult(
            order=order,
            market_bid_at_execution=149.98,
            market_ask_at_execution=150.02,
        )

        result = place_order_with_liquidity_check(order)

        assert result.success
        assert not result.blocked_by_liquidity
        assert result.liquidity_check.passed
        mock_place.assert_called_once()

    @patch('execution.broker_alpaca.check_liquidity_for_order')
    @patch('execution.broker_alpaca.place_ioc_limit')
    def test_blocks_order_when_liquidity_fails(self, mock_place, mock_check):
        """Should block order when liquidity check fails."""
        enable_liquidity_gate(True)

        mock_check.return_value = LiquidityCheck(
            symbol='LOWVOL',
            passed=False,
            reason="ADV $10,000 < min $100,000",
            issues=[LiquidityIssue.INSUFFICIENT_ADV],
        )

        order = self._create_test_order(symbol='LOWVOL')

        result = place_order_with_liquidity_check(order)

        assert not result.success
        assert result.blocked_by_liquidity
        assert result.order.status == OrderStatus.REJECTED
        assert "liquidity_gate" in result.order.notes
        mock_place.assert_not_called()

    @patch('execution.broker_alpaca.place_ioc_limit')
    def test_bypasses_check_when_disabled(self, mock_place):
        """Should bypass check when gate is disabled."""
        enable_liquidity_gate(False)

        order = self._create_test_order()
        order.status = OrderStatus.SUBMITTED
        mock_place.return_value = BrokerExecutionResult(
            order=order,
            market_bid_at_execution=149.98,
            market_ask_at_execution=150.02,
        )

        result = place_order_with_liquidity_check(order)

        assert result.liquidity_check is None
        assert not result.blocked_by_liquidity
        mock_place.assert_called_once()

        # Re-enable for other tests
        enable_liquidity_gate(True)


class TestExecuteSignal:
    """Tests for execute_signal function."""

    @patch('execution.broker_alpaca.get_best_bid')
    @patch('execution.broker_alpaca.get_best_ask')
    @patch('execution.broker_alpaca.check_liquidity_for_order')
    @patch('execution.broker_alpaca.place_ioc_limit')
    def test_full_execution_flow(self, mock_place, mock_check, mock_ask, mock_bid):
        """Should execute full flow: quote -> construct -> check -> place."""
        enable_liquidity_gate(True)

        mock_ask.return_value = 150.0
        mock_bid.return_value = 149.98
        mock_check.return_value = LiquidityCheck(
            symbol='AAPL',
            passed=True,
            adv_usd=5_000_000,
            spread_pct=0.02,
        )

        def set_submitted(order):
            order.status = OrderStatus.SUBMITTED
            order.broker_order_id = "BROKER123"
            return BrokerExecutionResult(
                order=order,
                market_bid_at_execution=149.98,
                market_ask_at_execution=150.02,
            )

        mock_place.side_effect = set_submitted

        result = execute_signal(
            symbol='AAPL',
            side='BUY',
            qty=100,
        )

        assert result.success
        assert result.order.symbol == 'AAPL'
        assert result.order.qty == 100
        assert result.order.status == OrderStatus.SUBMITTED

    @patch('execution.broker_alpaca.get_best_ask')
    def test_rejects_when_no_quote(self, mock_ask):
        """Should reject when no quote available."""
        mock_ask.return_value = None

        result = execute_signal(
            symbol='NOQUOTE',
            side='BUY',
            qty=100,
        )

        assert not result.success
        assert result.order.status == OrderStatus.REJECTED
        assert "no_quote" in result.order.notes

    @patch('execution.broker_alpaca.get_best_ask')
    @patch('execution.broker_alpaca.check_liquidity_for_order')
    @patch('execution.broker_alpaca.place_ioc_limit')
    def test_skips_liquidity_when_disabled(self, mock_place, mock_check, mock_ask):
        """Should skip liquidity check when disabled."""
        enable_liquidity_gate(False)

        mock_ask.return_value = 150.0

        def set_submitted(order):
            order.status = OrderStatus.SUBMITTED
            return BrokerExecutionResult(
                order=order,
                market_bid_at_execution=149.0,
                market_ask_at_execution=150.0,
            )

        mock_place.side_effect = set_submitted

        result = execute_signal(
            symbol='AAPL',
            side='BUY',
            qty=100,
            check_liquidity=True,  # Would check, but gate disabled
        )

        assert result.liquidity_check is None
        mock_check.assert_not_called()

        # Re-enable
        enable_liquidity_gate(True)

    @patch('execution.broker_alpaca.get_best_ask')
    @patch('execution.broker_alpaca.place_ioc_limit')
    def test_skips_liquidity_when_param_false(self, mock_place, mock_ask):
        """Should skip liquidity check when check_liquidity=False."""
        enable_liquidity_gate(True)

        mock_ask.return_value = 150.0

        def set_submitted(order):
            order.status = OrderStatus.SUBMITTED
            return BrokerExecutionResult(
                order=order,
                market_bid_at_execution=149.0,
                market_ask_at_execution=150.0,
            )

        mock_place.side_effect = set_submitted

        result = execute_signal(
            symbol='AAPL',
            side='BUY',
            qty=100,
            check_liquidity=False,
        )

        assert result.liquidity_check is None


class TestOrderResult:
    """Tests for OrderResult dataclass."""

    def test_success_when_submitted(self):
        """Should be success when submitted and not blocked."""
        order = OrderRecord(
            decision_id="TEST",
            signal_id="TEST",
            symbol="AAPL",
            side="BUY",
            qty=100,
            limit_price=150.0,
            tif="IOC",
            order_type="IOC_LIMIT",
            idempotency_key="TEST",
            created_at=datetime.utcnow(),
            status=OrderStatus.SUBMITTED,
        )

        result = OrderResult(order=order, blocked_by_liquidity=False)
        assert result.success

    def test_not_success_when_rejected(self):
        """Should not be success when rejected."""
        order = OrderRecord(
            decision_id="TEST",
            signal_id="TEST",
            symbol="AAPL",
            side="BUY",
            qty=100,
            limit_price=150.0,
            tif="IOC",
            order_type="IOC_LIMIT",
            idempotency_key="TEST",
            created_at=datetime.utcnow(),
            status=OrderStatus.REJECTED,
        )

        result = OrderResult(order=order, blocked_by_liquidity=False)
        assert not result.success

    def test_not_success_when_blocked(self):
        """Should not be success when blocked by liquidity."""
        order = OrderRecord(
            decision_id="TEST",
            signal_id="TEST",
            symbol="AAPL",
            side="BUY",
            qty=100,
            limit_price=150.0,
            tif="IOC",
            order_type="IOC_LIMIT",
            idempotency_key="TEST",
            created_at=datetime.utcnow(),
            status=OrderStatus.REJECTED,
        )

        result = OrderResult(order=order, blocked_by_liquidity=True)
        assert not result.success


# Run with: pytest tests/test_broker_liquidity_integration.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
