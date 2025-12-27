"""
Tests for Explainability Module.

Tests trade explainer, narrative generator, and decision tracker.
"""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from explainability import (
    # Trade Explainer
    TradeExplainer,
    TradeExplanation,
    ExplanationFactor,
    explain_trade,
    get_explainer,
    # Narrative Generator
    NarrativeGenerator,
    Narrative,
    NarrativeStyle,
    generate_narrative,
    generate_daily_summary,
    # Decision Tracker
    DecisionTracker,
    DecisionRecord,
    DecisionContext,
    record_decision,
    get_decision_history,
)

from explainability.trade_explainer import ExplanationLevel, FactorType
from explainability.decision_tracker import DecisionType, DecisionReason


class TestTradeExplainer:
    """Tests for TradeExplainer."""

    def test_initialization(self):
        """Should initialize with defaults."""
        explainer = TradeExplainer()
        assert explainer.include_confidence == True
        assert explainer.default_level == ExplanationLevel.STANDARD

    def test_explain_basic_signal(self):
        """Should explain a basic trade signal."""
        explainer = TradeExplainer()

        signal = {
            'symbol': 'AAPL',
            'side': 'long',
            'entry_price': 150.0,
            'reason': 'RSI oversold',
        }

        explanation = explainer.explain(signal)

        assert explanation.symbol == 'AAPL'
        assert explanation.side == 'long'
        assert explanation.entry_price == 150.0
        assert len(explanation.primary_factors) > 0

    def test_explain_with_indicators(self):
        """Should include indicator analysis."""
        explainer = TradeExplainer()

        signal = {
            'symbol': 'MSFT',
            'side': 'long',
            'entry_price': 300.0,
        }

        indicators = {
            'rsi': 25.0,
            'ibs': 0.15,
            'sma_200': 290.0,
        }

        explanation = explainer.explain(signal, indicators)

        # Should have extracted indicator factors
        all_factors = (
            explanation.primary_factors +
            explanation.secondary_factors +
            explanation.risk_factors
        )
        factor_names = [f.name for f in all_factors]

        assert 'rsi' in factor_names
        assert 'ibs' in factor_names

    def test_explanation_to_brief(self):
        """Should generate brief explanation."""
        explainer = TradeExplainer()

        signal = {
            'symbol': 'TSLA',
            'side': 'short',
            'entry_price': 200.0,
            'reason': 'Overbought',
        }

        explanation = explainer.explain(signal)
        brief = explanation.to_brief()

        assert 'TSLA' in brief
        assert len(brief) < 200  # Should be short

    def test_explanation_to_detailed(self):
        """Should generate detailed explanation."""
        explainer = TradeExplainer()

        signal = {
            'symbol': 'GOOG',
            'side': 'long',
            'entry_price': 140.0,
            'strategy': 'RSI2',
        }

        indicators = {'rsi_2': 8.0, 'atr': 2.5}

        explanation = explainer.explain(signal, indicators)
        detailed = explanation.to_detailed()

        assert 'GOOG' in detailed
        assert 'LONG' in detailed
        assert 'PRIMARY FACTORS' in detailed

    def test_explanation_with_confidence(self):
        """Should include confidence score."""
        explainer = TradeExplainer(include_confidence=True)

        signal = {
            'symbol': 'AMD',
            'side': 'long',
            'entry_price': 100.0,
            'confidence': 0.75,
        }

        explanation = explainer.explain(signal)

        assert explanation.confidence_score == 0.75

    def test_explanation_to_dict(self):
        """Should convert to dictionary."""
        explanation = TradeExplanation(
            symbol='NVDA',
            side='long',
            entry_price=500.0,
            headline='NVDA: RSI triggered long signal',
        )

        d = explanation.to_dict()

        assert d['symbol'] == 'NVDA'
        assert d['side'] == 'long'
        assert d['entry_price'] == 500.0


class TestExplanationFactor:
    """Tests for ExplanationFactor."""

    def test_to_sentence(self):
        """Should generate human-readable sentence."""
        factor = ExplanationFactor(
            factor_type=FactorType.INDICATOR,
            name='RSI',
            value=25.0,
            threshold=30,
        )

        sentence = factor.to_sentence()

        assert 'RSI' in sentence
        assert '25' in sentence

    def test_to_dict(self):
        """Should convert to dictionary."""
        factor = ExplanationFactor(
            factor_type=FactorType.PATTERN,
            name='Bullish Engulfing',
            value=True,
            contribution=0.8,
        )

        d = factor.to_dict()

        assert d['type'] == 'pattern'
        assert d['name'] == 'Bullish Engulfing'


class TestNarrativeGenerator:
    """Tests for NarrativeGenerator."""

    def test_initialization(self):
        """Should initialize with default style."""
        gen = NarrativeGenerator()
        assert gen.default_style == NarrativeStyle.TECHNICAL

    def test_daily_summary_no_trades(self):
        """Should handle days with no trades."""
        gen = NarrativeGenerator()

        narrative = gen.generate_daily_summary(
            date_str='2024-12-26',
            trades=[],
            pnl=0,
        )

        assert 'No' in narrative.content or 'no' in narrative.content
        assert narrative.metadata['trades'] == 0

    def test_daily_summary_with_trades(self):
        """Should summarize trading day."""
        gen = NarrativeGenerator()

        trades = [
            {'symbol': 'AAPL', 'side': 'long', 'pnl': 100},
            {'symbol': 'MSFT', 'side': 'long', 'pnl': -50},
        ]

        narrative = gen.generate_daily_summary(
            date_str='2024-12-26',
            trades=trades,
            pnl=50,
            win_rate=0.5,
        )

        assert '2024-12-26' in narrative.content or '2 trade' in narrative.content
        assert narrative.metadata['pnl'] == 50

    def test_daily_summary_styles(self):
        """Should generate different styles."""
        gen = NarrativeGenerator()

        trades = [{'symbol': 'SPY', 'pnl': 200}]

        technical = gen.generate_daily_summary('2024-12-26', trades, 200, style=NarrativeStyle.TECHNICAL)
        casual = gen.generate_daily_summary('2024-12-26', trades, 200, style=NarrativeStyle.CASUAL)
        executive = gen.generate_daily_summary('2024-12-26', trades, 200, style=NarrativeStyle.EXECUTIVE)

        # Different styles should produce different content
        assert technical.style == NarrativeStyle.TECHNICAL
        assert casual.style == NarrativeStyle.CASUAL
        assert executive.style == NarrativeStyle.EXECUTIVE

    def test_trade_narrative(self):
        """Should generate narrative for single trade."""
        gen = NarrativeGenerator()

        trade = {
            'symbol': 'AMZN',
            'side': 'long',
            'entry_price': 150.0,
            'exit_price': 155.0,
            'pnl': 50.0,
            'reason': 'RSI oversold',
        }

        narrative = gen.generate_trade_narrative(trade)

        assert 'AMZN' in narrative.content
        assert narrative.metadata['symbol'] == 'AMZN'

    def test_performance_recap(self):
        """Should generate performance recap."""
        gen = NarrativeGenerator()

        narrative = gen.generate_performance_recap(
            period='Week of Dec 23',
            total_pnl=500.0,
            total_trades=10,
            win_rate=0.6,
            sharpe=1.5,
        )

        assert 'Dec 23' in narrative.content or 'Week' in narrative.content
        assert narrative.metadata['total_pnl'] == 500.0

    def test_narrative_to_dict(self):
        """Should convert narrative to dictionary."""
        narrative = Narrative(
            title='Test Narrative',
            content='This is a test.',
            style=NarrativeStyle.TECHNICAL,
        )

        d = narrative.to_dict()

        assert d['title'] == 'Test Narrative'
        assert d['content'] == 'This is a test.'
        assert d['style'] == 'technical'


class TestDecisionTracker:
    """Tests for DecisionTracker."""

    @pytest.fixture
    def tracker(self):
        """Create tracker with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = DecisionTracker(
                log_dir=Path(tmpdir),
                auto_persist=False,
            )
            yield tracker

    def test_initialization(self, tracker):
        """Should initialize with defaults."""
        assert len(tracker._records) == 0
        assert tracker.max_records == 10000

    def test_record_decision(self, tracker):
        """Should record a decision."""
        context = DecisionContext(
            symbol='AAPL',
            price=150.0,
            indicators={'rsi': 25.0},
        )

        record = tracker.record(
            decision_type=DecisionType.ENTRY,
            reason=DecisionReason.SIGNAL,
            context=context,
            action_taken='LONG AAPL @ 150.0',
            rationale='RSI oversold',
        )

        assert record.decision_id.startswith('D')
        assert record.decision_type == DecisionType.ENTRY
        assert record.context.symbol == 'AAPL'
        assert len(tracker._records) == 1

    def test_record_entry(self, tracker):
        """Should record entry with convenience method."""
        record = tracker.record_entry(
            symbol='MSFT',
            price=300.0,
            indicators={'rsi_2': 8.0},
            reason='Extreme oversold',
            confidence=0.8,
        )

        assert record.decision_type == DecisionType.ENTRY
        assert record.context.symbol == 'MSFT'
        assert record.context.model_confidence == 0.8

    def test_record_exit(self, tracker):
        """Should record exit."""
        record = tracker.record_exit(
            symbol='GOOGL',
            price=140.0,
            reason=DecisionReason.STOP_LOSS,
            pnl=-100.0,
            rationale='Hit stop at 140',
        )

        assert record.decision_type == DecisionType.EXIT
        assert record.outcome_pnl == -100.0

    def test_record_skip(self, tracker):
        """Should record skipped signal."""
        record = tracker.record_skip(
            symbol='NVDA',
            price=500.0,
            reason='Failed regime filter',
            filters_failed=['regime_filter', 'volume_filter'],
        )

        assert record.decision_type == DecisionType.SKIP
        assert 'regime_filter' in record.filters_failed

    def test_record_outcome(self, tracker):
        """Should update record with outcome."""
        record = tracker.record_entry(
            symbol='AMD',
            price=100.0,
            indicators={},
        )

        success = tracker.record_outcome(
            decision_id=record.decision_id,
            pnl=50.0,
            duration=3,
        )

        assert success == True
        assert record.outcome_pnl == 50.0
        assert record.outcome_duration == 3

    def test_get_history(self, tracker):
        """Should retrieve history with filters."""
        tracker.record_entry('AAPL', 150, {})
        tracker.record_entry('MSFT', 300, {})
        tracker.record_skip('GOOG', 140, 'filtered')

        # Get all
        history = tracker.get_history()
        assert len(history) == 3

        # Filter by symbol
        aapl_history = tracker.get_history(symbol='AAPL')
        assert len(aapl_history) == 1
        assert aapl_history[0].context.symbol == 'AAPL'

        # Filter by type
        entries = tracker.get_history(decision_type=DecisionType.ENTRY)
        assert len(entries) == 2

    def test_get_stats(self, tracker):
        """Should calculate statistics."""
        record1 = tracker.record_entry('AAPL', 150, {})
        record2 = tracker.record_entry('MSFT', 300, {})
        tracker.record_skip('GOOG', 140, 'filtered')

        # Record outcomes
        tracker.record_outcome(record1.decision_id, 100)
        tracker.record_outcome(record2.decision_id, -50)

        stats = tracker.get_stats()

        assert stats['total_decisions'] == 3
        assert stats['entries'] == 2
        assert stats['skips'] == 1
        assert stats['completed_trades'] == 2
        assert stats['win_rate'] == 0.5

    def test_max_records_limit(self, tracker):
        """Should limit records in memory."""
        tracker.max_records = 5

        for i in range(10):
            tracker.record_entry(f'SYM{i}', 100.0 + i, {})

        assert len(tracker._records) == 5
        # Should keep most recent
        assert tracker._records[0].context.symbol == 'SYM5'

    def test_decision_was_successful(self):
        """Should determine success from outcome."""
        context = DecisionContext(symbol='TEST', price=100)
        record = DecisionRecord(
            decision_id='TEST001',
            decision_type=DecisionType.ENTRY,
            reason=DecisionReason.SIGNAL,
            context=context,
        )

        # No outcome yet
        assert record.was_successful is None

        # Profitable outcome
        record.outcome_pnl = 50.0
        assert record.was_successful == True

        # Unprofitable outcome
        record.outcome_pnl = -25.0
        assert record.was_successful == False

    def test_record_to_dict(self, tracker):
        """Should convert record to dictionary."""
        record = tracker.record_entry(
            symbol='TSLA',
            price=200.0,
            indicators={'rsi': 30},
        )

        d = record.to_dict()

        assert d['decision_id'] == record.decision_id
        assert d['decision_type'] == 'entry'
        assert d['context']['symbol'] == 'TSLA'


class TestDecisionContext:
    """Tests for DecisionContext."""

    def test_to_dict(self):
        """Should convert context to dictionary."""
        context = DecisionContext(
            symbol='SPY',
            price=450.0,
            indicators={'rsi': 50, 'macd': 0.5},
            model_confidence=0.7,
            regime='bull',
        )

        d = context.to_dict()

        assert d['symbol'] == 'SPY'
        assert d['price'] == 450.0
        assert d['indicators'] == {'rsi': 50, 'macd': 0.5}
        assert d['model_confidence'] == 0.7
        assert d['regime'] == 'bull'


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_explain_trade(self):
        """Should explain trade with convenience function."""
        signal = {
            'symbol': 'META',
            'side': 'long',
            'entry_price': 350.0,
        }

        explanation = explain_trade(signal)

        assert isinstance(explanation, TradeExplanation)
        assert explanation.symbol == 'META'

    def test_generate_narrative(self):
        """Should generate narrative with convenience function."""
        data = {
            'date': '2024-12-26',
            'trades': [{'symbol': 'AAPL', 'pnl': 100}],
            'pnl': 100,
        }

        narrative = generate_narrative(data, narrative_type='summary')

        assert isinstance(narrative, Narrative)
        assert narrative.metadata['pnl'] == 100

    def test_generate_daily_summary(self):
        """Should generate daily summary with convenience function."""
        narrative = generate_daily_summary(
            date_str='2024-12-26',
            trades=[],
            pnl=0,
        )

        assert isinstance(narrative, Narrative)

    def test_get_explainer(self):
        """Should return singleton explainer."""
        e1 = get_explainer()
        e2 = get_explainer()
        assert e1 is e2


# Run with: pytest tests/test_explainability.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
