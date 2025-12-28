import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import requests
import os

from execution.broker_alpaca import (
    place_ioc_limit, get_best_ask, get_best_bid, log_trade_event,
    _alpaca_cfg, _auth_headers, check_liquidity_for_order,
    place_order_with_liquidity_check, execute_signal,
    BrokerExecutionResult
)
from oms.order_state import OrderRecord, OrderStatus
from oms.idempotency_store import IdempotencyStore
from core.kill_switch import activate_kill_switch, deactivate_kill_switch, require_no_kill_switch
from risk.liquidity_gate import LiquidityCheck, LiquidityGate


# Fixture for a mock OrderRecord
@pytest.fixture
def sample_order_record():
    return OrderRecord(
        decision_id="DEC-TEST-123",
        signal_id="SIG-TEST-456",
        symbol="AAPL",
        side="BUY",
        qty=10,
        limit_price=150.00,
        tif="IOC",
        order_type="IOC_LIMIT",
        idempotency_key="IDEMP-TEST-789",
        created_at=datetime.utcnow(),
        entry_price_decision=149.95,
        strategy_used="test_strategy"
    )

# Fixture to mock Alpaca API requests using requests_mock
@pytest.fixture
def alpaca_requests_mock(requests_mock):
    # Mock for quotes
    requests_mock.get(
        "https://data.alpaca.markets/v2/stocks/quotes",
        json={"quotes": [{"bp": 149.90, "ap": 150.10, "bs": 100, "as": 150}]},
        status_code=200
    )
    # Mock for orders submission (success)
    requests_mock.post(
        "https://paper-api.alpaca.markets/v2/orders",
        json={"id": "broker-order-id-123", "status": "submitted"},
        status_code=200
    )
    # Mock for order status resolution (filled)
    requests_mock.get(
        "https://paper-api.alpaca.markets/v2/orders/broker-order-id-123",
        json={"id": "broker-order-id-123", "status": "filled", "filled_avg_price": 150.05, "filled_qty": "10"},
        status_code=200
    )
    yield requests_mock

# Fixture to mock IdempotencyStore - patch where it's imported, not where it's defined
@pytest.fixture
def mock_idempotency_store():
    with patch('execution.broker_alpaca.IdempotencyStore') as MockIdempotencyStore:
        mock_store = MockIdempotencyStore.return_value
        mock_store.exists.return_value = False
        yield mock_store

# Fixture for mock liquidity gate
@pytest.fixture
def mock_liquidity_gate():
    with patch('execution.broker_alpaca.get_liquidity_gate') as mock_get_liquidity_gate:
        mock_gate = MagicMock(spec=LiquidityGate)
        mock_check = MagicMock(spec=LiquidityCheck)
        mock_check.passed = True
        mock_check.symbol = "AAPL"
        mock_check.reason = "OK"
        mock_check.adv_usd = 10000000.0
        mock_check.spread_pct = 0.05
        mock_check.order_pct_of_adv = 0.1
        mock_check.min_adv_usd = 100000.0
        mock_check.max_spread_pct = 0.50
        mock_gate.check_liquidity.return_value = mock_check
        mock_get_liquidity_gate.return_value = mock_gate
        yield mock_gate

# Fixture to mock kill switch
@pytest.fixture(autouse=True)
def mock_kill_switch():
    with patch('core.kill_switch.is_kill_switch_active', return_value=False):
        yield

# Fixture for mock update_trade_event
@pytest.fixture(autouse=True)
def mock_update_trade_event():
    with patch('monitor.health_endpoints.update_trade_event') as mock_update:
        yield mock_update

# Mock os.getenv for Alpaca keys (using APCA_ prefix which the broker code uses)
@pytest.fixture(autouse=True)
def mock_alpaca_env():
    with patch.dict(os.environ, {
        "APCA_API_KEY_ID": "test_key",
        "APCA_API_SECRET_KEY": "test_secret",
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets"
    }):
        yield


class TestAlpacaBrokerFunctions:
    """Tests for core Alpaca interaction functions."""

    def test_get_best_ask_success(self, alpaca_requests_mock):
        ask = get_best_ask("AAPL")
        assert ask == 150.10

    def test_get_best_bid_success(self, alpaca_requests_mock):
        bid = get_best_bid("AAPL")
        assert bid == 149.90

    def test_get_best_ask_no_quote(self, alpaca_requests_mock):
        alpaca_requests_mock.get("https://data.alpaca.markets/v2/stocks/quotes", json={}, status_code=200)
        ask = get_best_ask("AAPL")
        assert ask is None

    def test_log_trade_event(self, sample_order_record, tmp_path):
        log_file = tmp_path / "trades.jsonl"
        with patch('execution.broker_alpaca.TRADES_LOG_PATH', log_file):
            log_trade_event(
                sample_order_record,
                market_bid=149.90,
                market_ask=150.10
            )
            assert log_file.exists()
            with open(log_file, 'r') as f:
                line = f.readline()
                event = json.loads(line)
                assert event['symbol'] == "AAPL"
                assert event['market_bid_at_execution'] == 149.90
                assert event['market_ask_at_execution'] == 150.10
                assert event['entry_price_decision'] == 149.95
                assert event['strategy_used'] == "test_strategy"


class TestPlaceIOCLimit:
    """Tests for place_ioc_limit function, including TCA integration."""

    def test_place_ioc_limit_success(self, sample_order_record, alpaca_requests_mock, mock_idempotency_store):
        broker_result = place_ioc_limit(sample_order_record)

        assert isinstance(broker_result, BrokerExecutionResult)
        assert broker_result.order.status == OrderStatus.FILLED
        assert broker_result.order.broker_order_id == "broker-order-id-123"
        assert broker_result.order.fill_price == 150.05
        assert broker_result.order.filled_qty == 10
        assert broker_result.market_bid_at_execution == 149.90
        assert broker_result.market_ask_at_execution == 150.10

        # Verify idempotency store and trade event logging
        mock_idempotency_store.put.assert_called_once()

    def test_place_ioc_limit_no_quotes(self, sample_order_record, alpaca_requests_mock, mock_idempotency_store):
        alpaca_requests_mock.get(
            "https://data.alpaca.markets/v2/stocks/quotes",
            json={}, status_code=200
        )
        broker_result = place_ioc_limit(sample_order_record)

        assert isinstance(broker_result, BrokerExecutionResult)
        assert broker_result.order.status == OrderStatus.REJECTED
        assert "no_quotes_available" in broker_result.order.notes
        assert broker_result.market_bid_at_execution is None
        assert broker_result.market_ask_at_execution is None
        mock_idempotency_store.put.assert_not_called()

    def test_place_ioc_limit_alpaca_reject(self, sample_order_record, alpaca_requests_mock, mock_idempotency_store):
        alpaca_requests_mock.post(
            "https://paper-api.alpaca.markets/v2/orders",
            json={"code": 403, "message": "forbidden"},
            status_code=403
        )
        broker_result = place_ioc_limit(sample_order_record)

        assert isinstance(broker_result, BrokerExecutionResult)
        assert broker_result.order.status == OrderStatus.REJECTED
        assert "alpaca_http_403" in broker_result.order.notes
        mock_idempotency_store.put.assert_not_called()


class TestPlaceOrderWithLiquidityCheck:
    """Tests for place_order_with_liquidity_check."""

    def test_liquidity_check_passes(self, sample_order_record, alpaca_requests_mock, mock_liquidity_gate, mock_idempotency_store):
        from execution.broker_alpaca import is_liquidity_gate_enabled, enable_liquidity_gate

        enable_liquidity_gate(True) # Ensure gate is enabled
        result = place_order_with_liquidity_check(sample_order_record)

        assert result.order.status == OrderStatus.FILLED
        assert result.blocked_by_liquidity is False
        assert result.liquidity_check is not None
        mock_liquidity_gate.check_liquidity.assert_called_once()
        assert result.market_bid_at_execution == 149.90
        assert result.market_ask_at_execution == 150.10


    def test_liquidity_check_fails(self, sample_order_record, alpaca_requests_mock, mock_liquidity_gate):
        from execution.broker_alpaca import is_liquidity_gate_enabled, enable_liquidity_gate

        enable_liquidity_gate(True) # Ensure gate is enabled
        mock_check = MagicMock(spec=LiquidityCheck)
        mock_check.passed = False
        mock_check.symbol = "AAPL"
        mock_check.reason = "Low liquidity"
        mock_check.adv_usd = 50000.0
        mock_check.spread_pct = 0.8
        mock_check.order_pct_of_adv = 5.0
        mock_check.min_adv_usd = 100000.0
        mock_check.max_spread_pct = 0.50
        mock_liquidity_gate.check_liquidity.return_value = mock_check

        result = place_order_with_liquidity_check(sample_order_record)

        assert result.order.status == OrderStatus.REJECTED
        assert result.blocked_by_liquidity is True
        assert "Low liquidity" in result.order.notes
        assert result.market_bid_at_execution is None # Should be None as order not placed
        assert result.market_ask_at_execution is None


class TestExecuteSignal:
    """Tests for execute_signal high-level function."""

    def test_execute_signal_success(self, sample_order_record, alpaca_requests_mock, mock_liquidity_gate, mock_idempotency_store):
        from execution.broker_alpaca import enable_liquidity_gate
        enable_liquidity_gate(True)

        result = execute_signal(
            symbol=sample_order_record.symbol,
            side=sample_order_record.side,
            qty=sample_order_record.qty,
            atr_value=1.0,
        )

        assert result.success is True
        assert result.order.status == OrderStatus.FILLED
        assert result.order.broker_order_id == "broker-order-id-123"
        assert result.market_bid_at_execution == 149.90
        assert result.market_ask_at_execution == 150.10


    def test_execute_signal_no_quotes(self, alpaca_requests_mock):
        alpaca_requests_mock.get(
            "https://data.alpaca.markets/v2/stocks/quotes",
            json={}, status_code=200
        )
        result = execute_signal(symbol="NOQUOTE", side="BUY", qty=10)

        assert result.success is False
        assert result.order.status == OrderStatus.REJECTED
        assert "no_quote_available" in result.order.notes
