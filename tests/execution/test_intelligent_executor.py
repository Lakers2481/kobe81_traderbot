import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from execution.intelligent_executor import IntelligentExecutor, get_intelligent_executor
from oms.order_state import OrderRecord, OrderStatus
from portfolio.risk_manager import TradeDecision

# Fixtures for mock dependencies
@pytest.fixture
def mock_strategy_selector():
    with patch('execution.intelligent_executor.AdaptiveStrategySelector') as MockSelector:
        mock_instance = MockSelector.return_value
        mock_config = MagicMock()
        mock_config.strategy_name = "mock_strategy"
        mock_config.skip_trading = False
        mock_instance.get_strategy_for_regime.return_value = (Mock(), mock_config) # (strategy_obj, config_obj)
        mock_instance._current_regime.value = "BULL"
        yield mock_instance

@pytest.fixture
def mock_confidence_integrator():
    with patch('execution.intelligent_executor.get_confidence_integrator') as MockIntegrator:
        mock_instance = MockIntegrator.return_value
        mock_instance.get_simple_confidence.return_value = 0.8
        yield mock_instance

@pytest.fixture
def mock_risk_manager():
    with patch('execution.intelligent_executor.get_risk_manager') as MockManager:
        mock_instance = MockManager.return_value
        mock_decision = MagicMock(spec=TradeDecision)
        mock_decision.approved = True
        mock_decision.shares = 10
        mock_decision.position_size = 1500.0
        mock_decision.rejection_reason = None
        mock_decision.warnings = []
        mock_instance.evaluate_trade.return_value = mock_decision
        yield mock_instance

@pytest.fixture
def mock_policy_gate():
    with patch('execution.intelligent_executor.PolicyGate') as MockGate:
        mock_instance = MockGate.return_value
        mock_instance.check.return_value = (True, None) # (allowed, reason)
        yield mock_instance

@pytest.fixture
def mock_trailing_stop_manager():
    with patch('execution.intelligent_executor.get_trailing_stop_manager') as MockManager:
        mock_instance = MockManager.return_value
        mock_instance.update_all_stops.return_value = []
        yield mock_instance

@pytest.fixture
def mock_order_manager():
    with patch('execution.intelligent_executor.get_order_manager') as MockManager:
        mock_instance = MockManager.return_value
        # Configure submit_order to update the passed OrderRecord
        def mock_submit_order(order_record: OrderRecord, *args, **kwargs):
            order_record.status = OrderStatus.FILLED
            order_record.broker_order_id = "BROKER-ID-123"
            order_record.fill_price = order_record.limit_price # Assume filled at limit
            order_record.filled_qty = order_record.qty
            return "EXEC-ID-456" # Return a mock execution ID
        mock_instance.submit_order.side_effect = mock_submit_order
        yield mock_instance

# Fixture for a sample signal
@pytest.fixture
def sample_signal():
    return {
        'symbol': 'AAPL',
        'side': 'BUY',
        'strategy': 'mock_strategy',
        'entry_price': 150.0,
        'stop_loss': 145.0,
        'take_profit': 160.0,
        'signal_id': 'SIG-TEST-456',
    }

# Fixture for sample price data
@pytest.fixture
def sample_price_data():
    return pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'close': [150.0, 151.0]
    })

# Fixture for sample SPY data
@pytest.fixture
def sample_spy_data():
    return pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'close': [400.0, 401.0]
    })


class TestIntelligentExecutorInitialization:
    def test_default_initialization(self):
        executor = IntelligentExecutor()
        assert executor.equity == 100000.0
        assert executor.paper_mode is True
        assert executor.min_confidence == 0.5
        assert executor._order_manager is None # Lazy loaded

    def test_lazy_loading_components(self):
        executor = IntelligentExecutor()
        _ = executor.strategy_selector
        _ = executor.confidence_integrator
        _ = executor.risk_manager
        _ = executor.policy_gate
        _ = executor.order_manager # Access the new order manager
        assert executor._strategy_selector is not None
        assert executor._order_manager is not None


class TestExecuteSignalIntelligently:
    """Tests for execute_signal_intelligently method."""

    def test_execute_signal_success(
        self,
        sample_signal,
        sample_price_data,
        mock_confidence_integrator,
        mock_risk_manager,
        mock_policy_gate,
        mock_order_manager,
    ):
        executor = IntelligentExecutor()
        result = executor.execute_signal_intelligently(
            signal=sample_signal,
            price_data=sample_price_data,
            dry_run=False,
        )

        assert result.approved is True
        assert result.executed is True
        assert result.shares == 10 # From mock_risk_manager
        assert result.position_size == 1500.0 # From mock_risk_manager
        assert result.ml_confidence == 0.8
        assert result.broker_order_id == "BROKER-ID-123" # Updated by mock_order_manager
        
        mock_confidence_integrator.get_simple_confidence.assert_called_once()
        mock_risk_manager.evaluate_trade.assert_called_once()
        mock_policy_gate.check.assert_called_once()
        mock_order_manager.submit_order.assert_called_once()
        args, kwargs = mock_order_manager.submit_order.call_args
        submitted_order = args[0]
        assert submitted_order.symbol == "AAPL"
        assert submitted_order.qty == 10
        assert submitted_order.entry_price_decision == 150.0 # From signal
        assert submitted_order.strategy_used == "mock_strategy"
        assert submitted_order.status == OrderStatus.FILLED # Updated by mock_submit_order

    def test_execute_signal_rejected_by_confidence(
        self,
        sample_signal,
        sample_price_data,
        mock_confidence_integrator,
        mock_risk_manager,
        mock_policy_gate,
        mock_order_manager,
    ):
        executor = IntelligentExecutor(min_confidence=0.9) # Set high min confidence
        result = executor.execute_signal_intelligently(
            signal=sample_signal,
            price_data=sample_price_data,
        )

        assert result.approved is False
        assert result.executed is False
        assert "below threshold" in result.rejection_reason
        mock_confidence_integrator.get_simple_confidence.assert_called_once()
        mock_risk_manager.evaluate_trade.assert_not_called()
        mock_policy_gate.check.assert_not_called()
        mock_order_manager.submit_order.assert_not_called()

    def test_execute_signal_rejected_by_risk_manager(
        self,
        sample_signal,
        sample_price_data,
        mock_confidence_integrator,
        mock_risk_manager,
        mock_policy_gate,
        mock_order_manager,
    ):
        mock_risk_manager.evaluate_trade.return_value.approved = False
        mock_risk_manager.evaluate_trade.return_value.rejection_reason = "Risk too high"
        executor = IntelligentExecutor()
        result = executor.execute_signal_intelligently(
            signal=sample_signal,
            price_data=sample_price_data,
        )

        assert result.approved is False
        assert result.executed is False
        assert "Risk too high" in result.rejection_reason
        mock_risk_manager.evaluate_trade.assert_called_once()
        mock_policy_gate.check.assert_not_called()
        mock_order_manager.submit_order.assert_not_called()

    def test_execute_signal_rejected_by_policy_gate(
        self,
        sample_signal,
        sample_price_data,
        mock_confidence_integrator,
        mock_risk_manager,
        mock_policy_gate,
        mock_order_manager,
    ):
        mock_policy_gate.check.return_value = (False, "Prohibited symbol") # (allowed, reason)
        executor = IntelligentExecutor()
        result = executor.execute_signal_intelligently(
            signal=sample_signal,
            price_data=sample_price_data,
        )

        assert result.approved is False
        assert result.executed is False
        assert "PolicyGate blocked" in result.rejection_reason
        mock_policy_gate.check.assert_called_once()
        mock_order_manager.submit_order.assert_not_called()

    def test_execute_signal_dry_run(
        self,
        sample_signal,
        sample_price_data,
        mock_confidence_integrator,
        mock_risk_manager,
        mock_policy_gate,
        mock_order_manager,
    ):
        executor = IntelligentExecutor()
        result = executor.execute_signal_intelligently(
            signal=sample_signal,
            price_data=sample_price_data,
            dry_run=True, # Should not call order_manager
        )

        assert result.approved is True
        assert result.executed is False # Not actually executed in dry run
        mock_order_manager.submit_order.assert_not_called()

    def test_execute_signal_order_manager_failure(
        self,
        sample_signal,
        sample_price_data,
        mock_confidence_integrator,
        mock_risk_manager,
        mock_policy_gate,
        mock_order_manager,
    ):
        # Configure mock_order_manager to fail submission
        def mock_submit_order_fail(order_record: OrderRecord, *args, **kwargs):
            order_record.status = OrderStatus.FAILED
            order_record.notes = "Broker API error"
            raise Exception("Broker API connection failed") # Simulate exception during submission
        mock_order_manager.submit_order.side_effect = mock_submit_order_fail

        executor = IntelligentExecutor()
        result = executor.execute_signal_intelligently(
            signal=sample_signal,
            price_data=sample_price_data,
            dry_run=False,
        )

        assert result.approved is True # Approved by internal logic
        assert result.executed is False # But not successfully executed
        assert result.rejection_reason == "Order submission failed with status: FAILED"
        mock_order_manager.submit_order.assert_called_once()


class TestExecutePipeline:
    """Tests for execute_pipeline method."""

    @pytest.fixture
    def mock_universe_data(self, sample_price_data):
        return {
            "AAPL": sample_price_data,
            "MSFT": sample_price_data.copy(),
        }

    def test_execute_pipeline_success(
        self,
        mock_strategy_selector,
        mock_confidence_integrator,
        mock_risk_manager,
        mock_policy_gate,
        mock_trailing_stop_manager,
        mock_order_manager,
        mock_universe_data,
        sample_spy_data,
    ):
        # Mock strategy.generate_signals to return a signal
        mock_strategy_selector.get_strategy_for_regime.return_value[0].generate_signals.return_value = pd.DataFrame([
            {
                'entry_price': 150.0, 'stop_loss': 145.0, 'take_profit': 160.0,
                'signal_id': 'SIG-TEST-AAPL', 'conf_score': 0.8
            },
            {
                'entry_price': 300.0, 'stop_loss': 290.0, 'take_profit': 310.0,
                'signal_id': 'SIG-TEST-MSFT', 'conf_score': 0.7
            },
        ])
        
        executor = IntelligentExecutor()
        result = executor.execute_pipeline(
            universe_data=mock_universe_data,
            spy_data=sample_spy_data,
            dry_run=False,
        )

        # Mock generates 2 signals per symbol * 2 symbols = 4 signals
        assert result.signals_generated == 4
        assert result.signals_approved == 4
        assert result.signals_executed == 4
        assert result.total_capital_deployed == 6000.0 # 4 signals * 1500.0 position size
        assert len(result.execution_results) == 4
        assert result.execution_results[0].symbol == "AAPL"
        assert result.execution_results[2].symbol == "MSFT"
        mock_order_manager.submit_order.call_count == 4
        # Trailing stop manager only called when current_positions is passed
        # mock_trailing_stop_manager.update_all_stops.assert_called_once()

    def test_execute_pipeline_skip_trading_regime(
        self,
        mock_strategy_selector,
        mock_universe_data,
        sample_spy_data,
    ):
        mock_config = MagicMock()
        mock_config.strategy_name = "mock_strategy"
        mock_config.skip_trading = True
        mock_config.notes = "Market closed"
        mock_strategy_selector.get_strategy_for_regime.return_value = (Mock(), mock_config)
        
        executor = IntelligentExecutor()
        result = executor.execute_pipeline(
            universe_data=mock_universe_data,
            spy_data=sample_spy_data,
        )

        assert result.signals_generated == 0
        assert "BULL" in result.regime # From mock strategy selector
        assert "Market closed" in mock_config.notes


class TestSingletonFactory:
    def test_get_intelligent_executor_singleton(self):
        executor1 = get_intelligent_executor()
        executor2 = get_intelligent_executor()
        assert executor1 is executor2
        assert isinstance(executor1, IntelligentExecutor)
