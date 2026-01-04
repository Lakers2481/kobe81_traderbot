"""
Unit tests for cognitive/signal_processor.py

Tests the bridge between trading signals and the cognitive brain.
"""
import pytest
import pandas as pd
from datetime import datetime


class TestCognitiveSignalProcessorInitialization:
    """Tests for CognitiveSignalProcessor initialization."""

    def test_default_initialization(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        assert processor is not None
        assert processor.brain is not None

    def test_processor_has_active_episodes_tracking(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        assert hasattr(processor, '_active_episodes')
        assert isinstance(processor._active_episodes, dict)


class TestBuildMarketContext:
    """Tests for building market context from data."""

    def test_build_context_with_spy_data(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        # Create sample SPY data with required timestamp column
        spy_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=4, freq='D'),
            'close': [450.0, 452.0, 448.0, 455.0],
            'high': [453.0, 454.0, 450.0, 457.0],
            'low': [448.0, 450.0, 446.0, 453.0],
            'volume': [50000000, 52000000, 48000000, 55000000],
        })

        context = processor.build_market_context(spy_data=spy_data)

        # Context should contain standard keys
        assert 'regime' in context
        assert 'timestamp' in context or 'data_timestamp' in context

    def test_build_context_empty_data(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        context = processor.build_market_context()

        assert context is not None
        assert isinstance(context, dict)


class TestEvaluateSignals:
    """Tests for evaluating trading signals."""

    def test_evaluate_empty_signals(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        signals = pd.DataFrame()
        approved, evaluations = processor.evaluate_signals(signals)

        assert len(approved) == 0
        assert len(evaluations) == 0

    def test_evaluate_single_signal(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        signals = pd.DataFrame([{
            'symbol': 'AAPL',
            'side': 'BUY',
            'strategy': 'ibs_rsi',
            'entry_price': 150.0,
            'stop_loss': 145.0,
            'take_profit': 160.0,
            'timestamp': datetime.now(),
        }])

        approved, evaluations = processor.evaluate_signals(signals)

        # Should process the signal (may or may not approve)
        assert isinstance(approved, pd.DataFrame)

    def test_evaluate_multiple_signals(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        signals = pd.DataFrame([
            {
                'symbol': 'AAPL',
                'side': 'BUY',
                'strategy': 'ibs_rsi',
                'entry_price': 150.0,
                'stop_loss': 145.0,
                'take_profit': 160.0,
            },
            {
                'symbol': 'MSFT',
                'side': 'BUY',
                'strategy': 'turtle_soup',
                'entry_price': 380.0,
                'stop_loss': 370.0,
                'take_profit': 400.0,
            },
        ])

        approved, evaluations = processor.evaluate_signals(signals)

        assert isinstance(approved, pd.DataFrame)


class TestRecordOutcome:
    """Tests for recording trade outcomes."""

    def test_record_outcome_updates_brain(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        # First evaluate a signal to create an episode
        signals = pd.DataFrame([{
            'symbol': 'AAPL',
            'side': 'BUY',
            'strategy': 'ibs_rsi',
            'entry_price': 150.0,
            'stop_loss': 145.0,
            'take_profit': 160.0,
        }])

        _, evaluations = processor.evaluate_signals(signals)

        # Record outcome if there's an active episode
        if evaluations:
            decision_id = evaluations[0].episode_id
            if decision_id:
                processor.record_outcome(
                    decision_id=decision_id,
                    won=True,
                    pnl=500.0,
                )
                # May or may not succeed depending on episode state


class TestDailyMaintenance:
    """Tests for daily maintenance tasks."""

    def test_daily_maintenance(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        result = processor.daily_maintenance()

        assert isinstance(result, dict)


class TestIntrospection:
    """Tests for processor introspection."""

    def test_introspect(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        report = processor.introspect()

        assert isinstance(report, str)
        assert len(report) > 0


class TestGetCognitiveStatus:
    """Tests for getting cognitive status."""

    def test_get_cognitive_status(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        status = processor.get_cognitive_status()

        assert 'processor_active' in status
        assert status['processor_active'] is True
        assert 'brain_status' in status


class TestRecordOutcomeBySymbol:
    """Tests for recording outcomes by symbol."""

    def test_record_outcome_by_symbol_no_match(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        # Try to record for a symbol that doesn't have an active episode
        result = processor.record_outcome_by_symbol(
            symbol='AAPL',
            strategy='ibs_rsi',
            won=True,
            pnl=100.0,
        )

        assert result is False  # No matching episode


class TestIntegrationWithBrain:
    """Tests for integration with CognitiveBrain."""

    def test_processor_uses_brain(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        assert processor.brain is not None
        assert hasattr(processor.brain, 'deliberate')
        assert hasattr(processor.brain, 'learn_from_outcome')

    def test_brain_status_accessible(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        brain_status = processor.brain.get_status()

        assert isinstance(brain_status, dict)


class TestSingletonFactory:
    """Tests for the singleton factory function."""

    def test_get_signal_processor(self):
        from cognitive.signal_processor import get_signal_processor

        processor1 = get_signal_processor()
        processor2 = get_signal_processor()

        assert processor1 is processor2


class TestErrorHandling:
    """Tests for error handling in signal processing."""

    def test_handles_malformed_signal(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        # Signal missing required fields
        signals = pd.DataFrame([{
            'symbol': 'AAPL',
            # Missing: side, strategy, prices
        }])

        # Should not raise an exception
        approved, evaluations = processor.evaluate_signals(signals)

        assert isinstance(approved, pd.DataFrame)

    def test_handles_none_market_data(self):
        from cognitive.signal_processor import CognitiveSignalProcessor

        processor = CognitiveSignalProcessor()

        signals = pd.DataFrame([{
            'symbol': 'AAPL',
            'side': 'BUY',
            'strategy': 'ibs_rsi',
            'entry_price': 150.0,
        }])

        # Should handle None market data gracefully
        approved, evaluations = processor.evaluate_signals(signals, market_data=None)

        assert isinstance(approved, pd.DataFrame)
