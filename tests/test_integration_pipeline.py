"""
Integration Pipeline Tests
===========================

Tests for the new advanced integration components:
- PortfolioRiskManager
- TrailingStopManager
- AdaptiveStrategySelector
- ConfidenceIntegrator
- IntelligentExecutor

Run: python -m pytest tests/test_integration_pipeline.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_mock_price_data(days: int = 300, start_price: float = 100.0) -> pd.DataFrame:
    """Generate realistic mock price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Generate random walk with drift
    returns = np.random.normal(0.0005, 0.02, days)
    prices = start_price * np.cumprod(1 + returns)

    high = prices * (1 + np.abs(np.random.normal(0, 0.01, days)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, days)))
    volume = np.random.randint(1000000, 10000000, days)

    return pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, days)),
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume,
    }, index=dates)


class TestPortfolioRiskManager:
    """Tests for PortfolioRiskManager."""

    def test_import(self):
        """Test that PortfolioRiskManager can be imported."""
        from portfolio.risk_manager import PortfolioRiskManager, get_risk_manager
        assert PortfolioRiskManager is not None
        assert get_risk_manager is not None

    def test_initialization(self):
        """Test PortfolioRiskManager initialization."""
        from portfolio.risk_manager import PortfolioRiskManager

        prm = PortfolioRiskManager(equity=100000)
        assert prm.equity == 100000
        assert prm.max_position_pct == 0.05

    def test_evaluate_trade_approval(self):
        """Test trade evaluation with valid signal."""
        from portfolio.risk_manager import PortfolioRiskManager

        prm = PortfolioRiskManager(equity=100000, use_kelly=False, use_ml_confidence=False)

        signal = {
            'symbol': 'AAPL',
            'entry_price': 150.0,
            'stop_loss': 145.0,
            'side': 'long',
        }

        decision = prm.evaluate_trade(signal, current_positions=[])
        assert decision.approved == True
        assert decision.shares > 0
        assert decision.position_size > 0

    def test_evaluate_trade_rejection_low_confidence(self):
        """Test trade rejection due to low ML confidence."""
        from portfolio.risk_manager import PortfolioRiskManager

        prm = PortfolioRiskManager(
            equity=100000,
            use_ml_confidence=True,
            min_confidence_threshold=0.6
        )

        signal = {
            'symbol': 'AAPL',
            'entry_price': 150.0,
            'stop_loss': 145.0,
            'side': 'long',
        }

        decision = prm.evaluate_trade(signal, current_positions=[], ml_confidence=0.4)
        assert decision.approved == False
        assert 'confidence' in decision.rejection_reason.lower()


class TestTrailingStopManager:
    """Tests for TrailingStopManager."""

    def test_import(self):
        """Test that TrailingStopManager can be imported."""
        from risk.trailing_stops import TrailingStopManager, get_trailing_stop_manager
        assert TrailingStopManager is not None
        assert get_trailing_stop_manager is not None

    def test_initialization(self):
        """Test TrailingStopManager initialization."""
        from risk.trailing_stops import TrailingStopManager

        tsm = TrailingStopManager()
        assert tsm.breakeven_threshold == 1.0
        assert tsm.trail_1r_threshold == 2.0

    def test_calculate_r_multiple(self):
        """Test R-multiple calculation."""
        from risk.trailing_stops import TrailingStopManager

        tsm = TrailingStopManager()

        # Long: Entry 100, Stop 95 (risk = 5), Price 110 (profit = 10)
        r = tsm.calculate_r_multiple(
            entry_price=100,
            current_price=110,
            initial_stop=95,
            side='long'
        )
        assert r == 2.0  # 10/5 = 2R profit

    def test_update_stop_breakeven(self):
        """Test stop moves to breakeven at 1R profit."""
        from risk.trailing_stops import TrailingStopManager, StopState

        tsm = TrailingStopManager()

        position = {
            'symbol': 'AAPL',
            'entry_price': 100.0,
            'stop_loss': 95.0,
            'initial_stop': 95.0,
            'side': 'long',
        }

        # Price at 106 = 1.2R profit (should trigger breakeven)
        update = tsm.update_stop(position, current_price=106.0)

        assert update.state == StopState.BREAKEVEN
        assert update.new_stop >= 99.0  # Should be at/near breakeven
        assert update.should_update == True

    def test_update_stop_trailing(self):
        """Test stop trails at 1R behind after 2R profit."""
        from risk.trailing_stops import TrailingStopManager, StopState

        tsm = TrailingStopManager()

        position = {
            'symbol': 'AAPL',
            'entry_price': 100.0,
            'stop_loss': 95.0,
            'initial_stop': 95.0,
            'side': 'long',
        }

        # Price at 112 = 2.4R profit (should trigger trailing)
        update = tsm.update_stop(position, current_price=112.0)

        assert update.state == StopState.TRAILING_1R
        assert update.new_stop > 95.0
        assert update.should_update == True


class TestAdaptiveStrategySelector:
    """Tests for AdaptiveStrategySelector."""

    def test_import(self):
        """Test that AdaptiveStrategySelector can be imported."""
        from strategies.adaptive_selector import AdaptiveStrategySelector, MarketRegime
        assert AdaptiveStrategySelector is not None
        assert MarketRegime is not None

    def test_initialization(self):
        """Test AdaptiveStrategySelector initialization."""
        from strategies.adaptive_selector import AdaptiveStrategySelector

        selector = AdaptiveStrategySelector()
        assert selector.regime_lookback == 60
        assert selector.confidence_threshold == 0.6

    def test_detect_regime_simple_bull(self):
        """Test simple regime detection identifies bull market."""
        from strategies.adaptive_selector import AdaptiveStrategySelector, MarketRegime

        selector = AdaptiveStrategySelector(use_hmm=False)

        # Create uptrending data (price above SMA50 above SMA200)
        price_data = generate_mock_price_data(days=250, start_price=100)
        # Force uptrend
        price_data['close'] = np.linspace(80, 150, len(price_data))
        price_data['close'] = price_data['close'] + np.random.normal(0, 1, len(price_data))

        regime, confidence = selector.detect_regime_simple(price_data)

        assert regime == MarketRegime.BULL
        assert confidence > 0.5

    def test_detect_regime_simple_bear(self):
        """Test simple regime detection identifies bear market."""
        from strategies.adaptive_selector import AdaptiveStrategySelector, MarketRegime

        selector = AdaptiveStrategySelector(use_hmm=False)

        # Create downtrending data
        price_data = generate_mock_price_data(days=250, start_price=150)
        price_data['close'] = np.linspace(150, 80, len(price_data))
        price_data['close'] = price_data['close'] + np.random.normal(0, 1, len(price_data))

        regime, confidence = selector.detect_regime_simple(price_data)

        assert regime == MarketRegime.BEAR
        assert confidence > 0.5


class TestConfidenceIntegrator:
    """Tests for ConfidenceIntegrator."""

    def test_import(self):
        """Test that ConfidenceIntegrator can be imported."""
        from ml_features.confidence_integrator import ConfidenceIntegrator, get_ml_confidence
        assert ConfidenceIntegrator is not None
        assert get_ml_confidence is not None

    def test_initialization(self):
        """Test ConfidenceIntegrator initialization."""
        from ml_features.confidence_integrator import ConfidenceIntegrator

        ci = ConfidenceIntegrator()
        assert ci.conviction_weight == 0.40
        assert ci.ensemble_weight == 0.35
        assert ci.lstm_weight == 0.25

    def test_get_simple_confidence(self):
        """Test simple confidence calculation."""
        from ml_features.confidence_integrator import ConfidenceIntegrator

        ci = ConfidenceIntegrator()
        price_data = generate_mock_price_data(days=250)

        signal = {
            'symbol': 'TEST',
            'entry_price': 100.0,
            'stop_loss': 95.0,
            'take_profit': 110.0,
        }

        confidence = ci.get_simple_confidence(signal, price_data)

        assert 0.0 <= confidence <= 1.0
        assert confidence >= ci.min_confidence_floor


class TestIntelligentExecutor:
    """Tests for IntelligentExecutor."""

    def test_import(self):
        """Test that IntelligentExecutor can be imported."""
        from execution.intelligent_executor import IntelligentExecutor, get_intelligent_executor
        assert IntelligentExecutor is not None
        assert get_intelligent_executor is not None

    def test_initialization(self):
        """Test IntelligentExecutor initialization."""
        from execution.intelligent_executor import IntelligentExecutor

        executor = IntelligentExecutor(equity=100000, paper_mode=True)
        assert executor.equity == 100000
        assert executor.paper_mode == True
        assert executor.min_confidence == 0.5

    def test_get_status(self):
        """Test getting executor status."""
        from execution.intelligent_executor import IntelligentExecutor

        executor = IntelligentExecutor(equity=50000)
        status = executor.get_status()

        assert status['equity'] == 50000
        assert 'components' in status

    def test_execute_signal_dry_run(self):
        """Test signal execution in dry run mode."""
        from execution.intelligent_executor import IntelligentExecutor

        executor = IntelligentExecutor(equity=100000, paper_mode=True)
        price_data = generate_mock_price_data(days=250)

        signal = {
            'symbol': 'TEST',
            'entry_price': 100.0,
            'stop_loss': 95.0,
            'take_profit': 110.0,
            'side': 'long',
        }

        result = executor.execute_signal_intelligently(
            signal=signal,
            price_data=price_data,
            dry_run=True
        )

        # Should process but not execute in dry run
        assert result.symbol == 'TEST'
        assert result.executed == False  # dry_run=True


class TestComponentIntegration:
    """Integration tests across all components."""

    def test_full_pipeline_dry_run(self):
        """Test the full pipeline end-to-end in dry run mode."""
        from execution.intelligent_executor import IntelligentExecutor

        executor = IntelligentExecutor(equity=100000, paper_mode=True)

        # Create mock universe
        universe = {
            'AAPL': generate_mock_price_data(days=300, start_price=150),
            'MSFT': generate_mock_price_data(days=300, start_price=350),
            'GOOGL': generate_mock_price_data(days=300, start_price=140),
        }

        spy_data = generate_mock_price_data(days=300, start_price=450)

        result = executor.execute_pipeline(
            universe_data=universe,
            spy_data=spy_data,
            vix_level=18.5,
            current_positions=[],
            dry_run=True
        )

        assert result is not None
        assert hasattr(result, 'signals_generated')
        assert hasattr(result, 'regime')
        assert hasattr(result, 'strategy_used')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
