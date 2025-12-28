import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import time

from execution.order_manager import OrderManager, get_order_manager
from oms.order_state import OrderRecord, OrderStatus
from execution.broker_alpaca import BrokerExecutionResult, get_best_ask, get_best_bid
from execution.tca.transaction_cost_analyzer import TransactionCostAnalyzer

# Fixture for a mock OrderRecord
@pytest.fixture
def sample_order_record():
    return OrderRecord(
        decision_id="DEC-TEST-123",
        signal_id="SIG-TEST-456",
        symbol="AAPL",
        side="BUY",
        qty=100,
        limit_price=150.00,
        tif="IOC",
        order_type="LIMIT",
        idempotency_key="IDEMP-TEST-789",
        created_at=datetime.utcnow(),
        entry_price_decision=149.95,
        strategy_used="test_strategy"
    )

# Fixture to mock broker_alpaca functions
@pytest.fixture
def mock_broker_alpaca():
    with patch('execution.order_manager.place_ioc_limit') as mock_place_ioc_limit, \
         patch('execution.order_manager.get_best_ask') as mock_get_best_ask, \
         patch('execution.order_manager.get_best_bid') as mock_get_best_bid:
        
        # Default mock return for get_best_ask/bid
        mock_get_best_ask.return_value = 150.10
        mock_get_best_bid.return_value = 149.90

        # Configure mock_place_ioc_limit to simulate a successful fill
        mock_order_filled = MagicMock(spec=OrderRecord)
        mock_order_filled.status = OrderStatus.FILLED
        mock_order_filled.broker_order_id = "ALPACA-ORDER-123"
        mock_order_filled.fill_price = 150.05
        mock_order_filled.filled_qty = 100
        mock_order_filled.notes = None

        mock_broker_execution_result = BrokerExecutionResult(
            order=mock_order_filled,
            market_bid_at_execution=149.90,
            market_ask_at_execution=150.10
        )
        mock_place_ioc_limit.return_value = mock_broker_execution_result
        
        yield mock_place_ioc_limit, mock_get_best_ask, mock_get_best_bid

# Fixture to mock TCA analyzer
@pytest.fixture
def mock_tca_analyzer():
    with patch('execution.order_manager.get_tca_analyzer') as mock_get_tca_analyzer:
        mock_tca = MagicMock(spec=TransactionCostAnalyzer)
        mock_get_tca_analyzer.return_value = mock_tca
        yield mock_tca

# Mock time.sleep for TWAP/VWAP tests
@pytest.fixture(autouse=True)
def mock_time_sleep():
    with patch('time.sleep', return_value=None):
        yield

class TestOrderManagerInitialization:
    def test_default_initialization(self, mock_tca_analyzer):
        manager = OrderManager()
        assert manager.default_execution_strategy == "LIMIT"
        assert manager.tca_analyzer is mock_tca_analyzer


class TestSubmitOrder:
    def test_submit_order_limit_success(self, sample_order_record, mock_broker_alpaca, mock_tca_analyzer):
        mock_place_ioc_limit, mock_get_best_ask, mock_get_best_bid = mock_broker_alpaca
        manager = OrderManager()

        execution_id = manager.submit_order(sample_order_record, execution_strategy="LIMIT")
        
        assert execution_id.startswith("EXEC-")
        assert sample_order_record.status == OrderStatus.FILLED
        assert sample_order_record.broker_order_id == "ALPACA-ORDER-123"
        assert sample_order_record.fill_price == 150.05
        assert sample_order_record.filled_qty == 100
        
        mock_place_ioc_limit.assert_called_once()
        mock_tca_analyzer.record_execution.assert_called_once()

    def test_submit_order_twap_success(self, sample_order_record, mock_broker_alpaca, mock_tca_analyzer):
        mock_place_ioc_limit, _, _ = mock_broker_alpaca
        mock_place_ioc_limit.return_value.order.filled_qty = 10 # Simulate partial fills for TWAP
        mock_place_ioc_limit.return_value.order.fill_price = 150.05 # Simulate fill price for TWAP
        
        manager = OrderManager()
        sample_order_record.qty = 100 # Large order for TWAP

        # Simulate 10 slices, each filling 10 shares
        mock_place_ioc_limit.return_value.order.filled_qty = 10
        mock_place_ioc_limit.return_value.order.fill_price = 150.05
        
        execution_id = manager.submit_order(sample_order_record, execution_strategy="TWAP")
        
        assert execution_id.startswith("EXEC-")
        assert sample_order_record.status == OrderStatus.FILLED
        assert sample_order_record.filled_qty == 100 # Total filled
        assert mock_place_ioc_limit.call_count == 10 # Called 10 times for 10 slices
        assert mock_tca_analyzer.record_execution.call_count == 10 # TCA recorded for each slice

    def test_submit_order_vwap_calls_twap(self, sample_order_record, mock_broker_alpaca, mock_tca_analyzer):
        mock_place_ioc_limit, _, _ = mock_broker_alpaca
        mock_place_ioc_limit.return_value.order.filled_qty = 10 # Simulate partial fills for TWAP
        mock_place_ioc_limit.return_value.order.fill_price = 150.05 # Simulate fill price for TWAP
        
        manager = OrderManager()
        sample_order_record.qty = 100 # Large order for TWAP

        # Simulate 10 slices, each filling 10 shares
        mock_place_ioc_limit.return_value.order.filled_qty = 10
        mock_place_ioc_limit.return_value.order.fill_price = 150.05
        
        execution_id = manager.submit_order(sample_order_record, execution_strategy="TWAP")
        
        assert execution_id.startswith("EXEC-")
        assert sample_order_record.status == OrderStatus.FILLED
        assert sample_order_record.filled_qty == 100 # Total filled
        assert mock_place_ioc_limit.call_count == 10 # Called 10 times for 10 slices
        assert mock_tca_analyzer.record_execution.call_count == 10 # TCA recorded for each slice

    def test_submit_order_unknown_strategy_defaults_to_limit(self, sample_order_record, mock_broker_alpaca, mock_tca_analyzer):
        mock_place_ioc_limit, _, _ = mock_broker_alpaca
        manager = OrderManager()
        execution_id = manager.submit_order(sample_order_record, execution_strategy="UNKNOWN")
        
        assert "UNKNOWN" in execution_id
        assert sample_order_record.status == OrderStatus.FILLED
        mock_place_ioc_limit.assert_called_once()
        mock_tca_analyzer.record_execution.assert_called_once()

    def test_submit_order_error_updates_status(self, sample_order_record, mock_broker_alpaca, mock_tca_analyzer):
        mock_place_ioc_limit, mock_get_best_ask, mock_get_best_bid = mock_broker_alpaca
        mock_get_best_ask.return_value = None # Simulate no quote, causing ValueError
        manager = OrderManager()

        execution_id = manager.submit_order(sample_order_record)
        
        assert execution_id.startswith("EXEC-")
        assert sample_order_record.status == OrderStatus.FAILED
        assert "Could not get current market quotes" in sample_order_record.notes
        mock_place_ioc_limit.assert_not_called()
        mock_tca_analyzer.record_execution.assert_not_called()

class TestExecuteSimpleIOCLimit:
    def test_execute_simple_ioc_limit_success(self, sample_order_record, mock_broker_alpaca, mock_tca_analyzer):
        mock_place_ioc_limit, _, _ = mock_broker_alpaca
        manager = OrderManager()
        
        manager._execute_simple_ioc_limit(sample_order_record, market_bid_at_submission=149.90, market_ask_at_submission=150.10)
        
        mock_place_ioc_limit.assert_called_once_with(sample_order_record)
        mock_tca_analyzer.record_execution.assert_called_once_with(
            order=mock_place_ioc_limit.return_value.order,
            fill_price=mock_place_ioc_limit.return_value.order.fill_price,
            market_bid_at_execution=mock_place_ioc_limit.return_value.market_bid_at_execution,
            market_ask_at_execution=mock_place_ioc_limit.return_value.market_ask_at_execution,
            entry_price_decision=sample_order_record.entry_price_decision
        )
        assert sample_order_record.status == OrderStatus.FILLED


class TestExecuteTWAP:
    def test_execute_twap_fills_order(self, sample_order_record, mock_broker_alpaca, mock_tca_analyzer):
        mock_place_ioc_limit, _, _ = mock_broker_alpaca
        manager = OrderManager()
        sample_order_record.qty = 100 # 10 slices of 10 shares
        
        # Configure mock_place_ioc_limit to simulate successful partial fills for TWAP
        mock_slice_order_filled = MagicMock(spec=OrderRecord)
        mock_slice_order_filled.status = OrderStatus.FILLED
        mock_slice_order_filled.filled_qty = 10
        mock_slice_order_filled.fill_price = 150.05
        
        mock_broker_execution_result_slice = BrokerExecutionResult(
            order=mock_slice_order_filled,
            market_bid_at_execution=149.90,
            market_ask_at_execution=150.10
        )
        mock_place_ioc_limit.return_value = mock_broker_execution_result_slice
        
        manager._execute_twap(sample_order_record, market_bid_at_submission=149.90, market_ask_at_submission=150.10, duration_minutes=1, slice_count=10)
        
        assert sample_order_record.status == OrderStatus.FILLED
        assert sample_order_record.filled_qty == 100
        assert mock_place_ioc_limit.call_count == 10
        assert mock_tca_analyzer.record_execution.call_count == 10

class TestExecuteVWAP:
    def test_execute_vwap_calls_twap(self, sample_order_record, mock_tca_analyzer):
        manager = OrderManager()
        with patch.object(manager, '_execute_twap') as mock_execute_twap:
            manager._execute_vwap(sample_order_record, market_bid_at_submission=149.90, market_ask_at_submission=150.10)
            mock_execute_twap.assert_called_once()

class TestSingletonFactory:
    def test_get_order_manager_singleton(self, mock_tca_analyzer):
        manager1 = get_order_manager()
        manager2 = get_order_manager()
        assert manager1 is manager2
        assert isinstance(manager1, OrderManager)
