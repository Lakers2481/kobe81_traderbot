"""
Integration Pipeline Tests
===========================

Tests for the new advanced integration components:
- PortfolioRiskManager
- TrailingStopManager
- AdaptiveStrategySelector
- ConfidenceIntegrator
- IntelligentExecutor
- CognitiveSignalProcessor (with News & LLM integration)
- OrderManager (with TCA integration)

Run: python -m pytest tests/test_integration_pipeline.py -v
"""

import pytest
import numpy as np
import pandas as pd
from contextlib import ExitStack
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from oms.order_state import OrderRecord, OrderStatus
from execution.broker_alpaca import BrokerExecutionResult

# Mock all lazy-loaded singletons and external API interactions
@pytest.fixture(autouse=True)
def mock_all_dependencies():
    with ExitStack() as stack:
        # Intelligent executor patches
        mock_get_signal_processor = stack.enter_context(patch('execution.intelligent_executor.get_signal_processor'))
        mock_get_risk_manager = stack.enter_context(patch('execution.intelligent_executor.get_risk_manager'))
        MockPolicyGate = stack.enter_context(patch('execution.intelligent_executor.PolicyGate'))
        mock_get_order_manager = stack.enter_context(patch('execution.intelligent_executor.get_order_manager'))
        mock_get_trailing_stop_manager = stack.enter_context(patch('execution.intelligent_executor.get_trailing_stop_manager'))
        MockStrategySelector = stack.enter_context(patch('execution.intelligent_executor.AdaptiveStrategySelector'))
        mock_get_confidence_integrator = stack.enter_context(patch('execution.intelligent_executor.get_confidence_integrator'))

        # Cognitive signal processor patches
        mock_get_cognitive_brain = stack.enter_context(patch('cognitive.signal_processor.get_cognitive_brain'))
        mock_get_news_processor = stack.enter_context(patch('cognitive.signal_processor.get_news_processor'))

        # Reflection engine patches
        mock_get_episodic_memory = stack.enter_context(patch('cognitive.reflection_engine.get_episodic_memory'))
        mock_get_semantic_memory = stack.enter_context(patch('cognitive.reflection_engine.get_semantic_memory'))
        mock_get_self_model = stack.enter_context(patch('cognitive.reflection_engine.get_self_model'))
        mock_get_workspace = stack.enter_context(patch('cognitive.reflection_engine.get_workspace'))
        mock_get_llm_analyzer = stack.enter_context(patch('cognitive.reflection_engine.get_llm_analyzer'))

        # Order manager patches
        mock_place_ioc_limit = stack.enter_context(patch('execution.order_manager.place_ioc_limit'))
        mock_get_best_ask = stack.enter_context(patch('execution.order_manager.get_best_ask'))
        mock_get_best_bid = stack.enter_context(patch('execution.order_manager.get_best_bid'))
        mock_get_tca_analyzer = stack.enter_context(patch('execution.order_manager.get_tca_analyzer'))

        # Broker alpaca patches
        mock_get_quote_with_sizes = stack.enter_context(patch('execution.broker_alpaca.get_quote_with_sizes'))
        MockIdempotencyStore = stack.enter_context(patch('execution.broker_alpaca.IdempotencyStore'))
        mock_update_trade_event = stack.enter_context(patch('execution.broker_alpaca.update_trade_event'))
        mock_requests = stack.enter_context(patch('execution.broker_alpaca.requests'))
        stack.enter_context(patch('execution.broker_alpaca.is_kill_switch_active', return_value=False))
        stack.enter_context(patch('execution.broker_alpaca.is_clamp_enabled', return_value=False))
        stack.enter_context(patch('execution.broker_alpaca.is_liquidity_gate_enabled', return_value=False))
        
        # --- Configure Mocks ---
        # Mock CognitiveBrain.deliberate
        mock_decision = MagicMock()
        mock_decision.should_act = True
        mock_decision.confidence = 0.85
        mock_decision.reasoning_trace = ["Cognitive approved"]
        mock_decision.concerns = []
        mock_decision.knowledge_gaps = []
        mock_decision.invalidators = []
        mock_decision.episode_id = "mock_cognitive_episode_123"
        mock_decision.decision_mode = "slow"
        mock_decision.action = {'size_multiplier': 1.0, 'type': 'trade'}
        mock_get_cognitive_brain.return_value.deliberate.return_value = mock_decision

        # Mock OrderManager.submit_order
        def mock_submit_order(order_record: OrderRecord, *args, **kwargs):
            order_record.status = OrderStatus.FILLED
            order_record.broker_order_id = "BROKER-ID-MOCK"
            order_record.fill_price = order_record.limit_price * 1.0001 # Slight slippage
            order_record.filled_qty = order_record.qty
            return "EXEC-ID-MOCK"
        mock_get_order_manager.return_value.submit_order.side_effect = mock_submit_order
        
        # Mock broker_alpaca quotes
        mock_get_best_ask.return_value = 150.10
        mock_get_best_bid.return_value = 149.90
        mock_get_quote_with_sizes.return_value = (149.90, 150.10, 100, 100) # bid, ask, bid_size, ask_size

        # Mock broker_alpaca place_ioc_limit (for TCA tests)
        mock_broker_order_filled = MagicMock(spec=OrderRecord)
        mock_broker_order_filled.status = OrderStatus.FILLED
        mock_broker_order_filled.broker_order_id = "ALPACA-ORDER-MOCK"
        mock_broker_order_filled.fill_price = 150.05
        mock_broker_order_filled.filled_qty = 100
        mock_broker_order_filled.notes = None
        mock_place_ioc_limit.return_value = BrokerExecutionResult(
            order=mock_broker_order_filled,
            market_bid_at_execution=149.90,
            market_ask_at_execution=150.10
        )

        # Mock NewsProcessor
        mock_get_news_processor.return_value.get_aggregated_sentiment.side_effect = [
            {'compound': 0.7, 'positive': 0.8}, # Market sentiment
            {'compound': 0.6, 'positive': 0.7}, # AAPL sentiment
        ]

        # Mock other components that are called
        mock_get_self_model.return_value.record_trade_outcome.return_value = None
        mock_get_semantic_memory.return_value.add_rule.return_value = None
        mock_get_episodic_memory.return_value.add_postmortem.return_value = None
        mock_get_episodic_memory.return_value.get_recent_episodes.return_value = []
        
        # Mock SignalProcessor's brain property (as it's used directly in web.main)
        mock_get_signal_processor.return_value.brain = mock_get_cognitive_brain.return_value
        mock_get_signal_processor.return_value._active_episodes = {"AAPL|mock_strategy": "mock_cognitive_episode_123"}
        mock_get_signal_processor.return_value.get_cognitive_status.return_value = {"brain_init": True}
        mock_get_signal_processor.return_value.build_market_context.return_value = {"regime": "BULL", "market_sentiment": {"compound": 0.5}}

        yield

# Generate mock price data
def generate_mock_price_data(days: int = 300, start_price: float = 100.0) -> pd.DataFrame:
    """Generate realistic mock price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    returns = np.random.normal(0.0005, 0.02, days)
    prices = start_price * np.cumprod(1 + returns)
    return pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days),
    }, index=dates)


@pytest.mark.skip(reason="Integration tests need refactoring - mock targets non-existent modules")
class TestFullIntegrationPipeline:
    """
    Comprehensive integration tests for the entire trading pipeline,
    including cognitive and execution enhancements.
    """

    @pytest.fixture
    def mock_strategy_signal_generator(self):
        # Mock for strategy.generate_signals
        mock_strategy = MagicMock()
        mock_strategy.generate_signals.return_value = pd.DataFrame([
            {'entry_price': 150.0, 'stop_loss': 145.0, 'take_profit': 160.0, 'signal_id': 'SIG-TEST-AAPL', 'conf_score': 0.8},
        ])
        return mock_strategy

    def test_full_pipeline_execution_and_learning_loop(
        self,
        mock_all_dependencies,
        mock_strategy_selector,
        mock_confidence_integrator,
        mock_risk_manager,
        mock_policy_gate,
        mock_order_manager,
        mock_tca_analyzer,
        mock_get_cognitive_brain,
        mock_get_news_processor,
        mock_get_llm_analyzer,
        mock_get_episodic_memory,
        mock_get_semantic_memory,
        mock_get_self_model,
        mock_update_trade_event, # for broker_alpaca.py log
        sample_signal, # for direct intelligent_executor test
        sample_price_data, # for direct intelligent_executor test
        sample_spy_data # for direct intelligent_executor test
    ):
        # --- Setup Mocks for Pipeline ---
        # Mock strategy selector to return our mock signal generator
        mock_strategy_selector.return_value.get_strategy_for_regime.return_value = (
            MagicMock(), MagicMock(strategy_name="mock_strategy", skip_trading=False)
        )
        # Configure the mock strategy to generate a signal
        mock_strategy_selector.return_value.get_strategy_for_regime.return_value[0].generate_signals.return_value = pd.DataFrame([
            {'entry_price': 150.0, 'stop_loss': 145.0, 'take_profit': 160.0, 'signal_id': 'SIG-TEST-AAPL', 'conf_score': 0.8},
        ])

        # Mock NewsProcessor for symbol-specific call
        mock_get_news_processor.return_value.get_aggregated_sentiment.side_effect = [
            {'compound': 0.7, 'positive': 0.8}, # Market sentiment
            {'compound': 0.6, 'positive': 0.7}, # AAPL sentiment
        ]

        from execution.intelligent_executor import IntelligentExecutor
        from cognitive.signal_processor import CognitiveSignalProcessor

        executor = IntelligentExecutor(equity=100000, paper_mode=True)
        processor = CognitiveSignalProcessor()

        # --- Test: Full pipeline execution ---
        universe_data = {"AAPL": generate_mock_price_data()}
        pipeline_result = executor.execute_pipeline(
            universe_data=universe_data,
            spy_data=generate_mock_price_data(days=100, start_price=400),
            dry_run=False
        )

        assert pipeline_result.signals_executed == 1
        assert pipeline_result.execution_results[0].symbol == "AAPL"
        assert pipeline_result.execution_results[0].executed is True

        # --- Verification of Cognitive Components Call Chain ---
        # CognitiveSignalProcessor.build_market_context should have been called
        mock_get_news_processor.return_value.get_aggregated_sentiment.assert_any_call() # For market sentiment

        # CognitiveBrain.deliberate should have been called
        mock_get_cognitive_brain.return_value.deliberate.assert_called_once()
        deliberate_kwargs = mock_get_cognitive_brain.return_value.deliberate.call_args.kwargs
        assert 'symbol_sentiment' in deliberate_kwargs['context']
        assert deliberate_kwargs['context']['symbol_sentiment']['compound'] == 0.6 # Symbol-specific sentiment

        # OrderManager.submit_order should have been called
        mock_get_order_manager.return_value.submit_order.assert_called_once()
        submitted_order: OrderRecord = mock_get_order_manager.return_value.submit_order.call_args.args[0]
        assert submitted_order.symbol == "AAPL"
        assert submitted_order.status == OrderStatus.FILLED # Updated by mock_order_manager

        # --- Verification of Learning Loop ---
        # CognitiveBrain.learn_from_outcome should have been called
        mock_get_cognitive_brain.return_value.learn_from_outcome.assert_called_once()
        learn_kwargs = mock_get_cognitive_brain.return_value.learn_from_outcome.call_args.kwargs
        assert learn_kwargs['episode_id'] == "mock_cognitive_episode_123" # From mock_decision
        assert learn_kwargs['outcome']['won'] is True # Assuming success for this test

        # ReflectionEngine should have been triggered (via learn_from_outcome internally)
        # and it should have called the LLM analyzer
        mock_get_llm_analyzer.return_value.analyze_reflection.assert_called_once()
        # Ensure that the Reflection object passed to LLM has the basic summary
        reflection_arg = mock_get_llm_analyzer.return_value.analyze_reflection.call_args.args[0]
        assert "Cognitive approved" in reflection_arg.summary # From mock_decision.reasoning_trace

        # SelfModel and SemanticMemory should have been updated by _apply_learnings
        mock_get_self_model.return_value.record_trade_outcome.assert_called_once()
        mock_get_semantic_memory.return_value.add_rule.assert_called_once() # For the winning trade

        # TCA Analyzer should have recorded execution
        mock_get_tca_analyzer.return_value.record_execution.assert_called_once()
        tca_kwargs = mock_get_tca_analyzer.return_value.record_execution.call_args.kwargs
        assert tca_kwargs['order'].symbol == "AAPL"
        assert tca_kwargs['fill_price'] == 150.05
        assert tca_kwargs['entry_price_decision'] == 150.0


    def test_full_pipeline_rejection_by_cognitive_brain(
        self,
        mock_all_dependencies,
        mock_get_cognitive_brain,
        mock_order_manager,
    ):
        from execution.intelligent_executor import IntelligentExecutor
        
        # Configure cognitive brain to reject the signal
        mock_get_cognitive_brain.return_value.deliberate.return_value.should_act = False
        mock_get_cognitive_brain.return_value.deliberate.return_value.confidence = 0.3
        mock_get_cognitive_brain.return_value.deliberate.return_value.rejection_reason = "Low confidence from brain"

        executor = IntelligentExecutor(equity=100000, paper_mode=True)
        universe_data = {"AAPL": generate_mock_price_data()}

        pipeline_result = executor.execute_pipeline(
            universe_data=universe_data,
            spy_data=generate_mock_price_data(days=100, start_price=400),
            dry_run=False
        )
        
        assert pipeline_result.signals_executed == 0
        assert pipeline_result.signals_rejected == 1
        assert "Low confidence from brain" in pipeline_result.execution_results[0].rejection_reason
        mock_get_order_manager.return_value.submit_order.assert_not_called()
        mock_get_cognitive_brain.return_value.learn_from_outcome.assert_called_once() # Brain still learns from its decision not to act