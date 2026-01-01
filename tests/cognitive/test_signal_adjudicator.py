"""
Unit tests for the SignalAdjudicator module.

Tests:
1. Signal strength scoring (IBS/RSI tiering)
2. Pattern confluence detection
3. Volatility contraction scoring
4. Sector strength scoring
5. Full adjudication pipeline
6. Config validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from cognitive.signal_adjudicator import (
    SignalAdjudicator,
    AdjudicatorConfig,
    AdjudicationResult,
    get_adjudicator,
    adjudicate_signals,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    np.random.seed(42)

    # Create realistic price data
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.02, 100)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'AAPL',
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
        'high': prices * (1 + np.random.uniform(0, 0.02, 100)),
        'low': prices * (1 - np.random.uniform(0, 0.02, 100)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100),
    })

    return df


@pytest.fixture
def sample_spy_data():
    """Generate sample SPY data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    np.random.seed(43)

    base_price = 450.0
    returns = np.random.normal(0.0003, 0.015, 100)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'SPY',
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
        'high': prices * (1 + np.random.uniform(0, 0.02, 100)),
        'low': prices * (1 - np.random.uniform(0, 0.02, 100)),
        'close': prices,
        'volume': np.random.randint(10000000, 100000000, 100),
    })

    return df


@pytest.fixture
def sample_signals():
    """Generate sample signals for testing."""
    return pd.DataFrame([
        {
            'timestamp': datetime.now(),
            'symbol': 'AAPL',
            'strategy': 'ibs_rsi',
            'side': 'buy',
            'entry_price': 185.50,
            'stop_loss': 183.10,
            'take_profit': 188.20,
            'score': 15.0,
            'ibs': 0.03,
            'rsi2': 3.5,
        },
        {
            'timestamp': datetime.now(),
            'symbol': 'MSFT',
            'strategy': 'turtle_soup',
            'side': 'buy',
            'entry_price': 378.00,
            'stop_loss': 374.50,
            'take_profit': 385.00,
            'score': 120.0,
            'ibs': 0.12,
            'rsi2': 15.0,
        },
    ])


@pytest.fixture
def adjudicator():
    """Create a fresh SignalAdjudicator for each test."""
    return SignalAdjudicator(AdjudicatorConfig())


# ============================================================================
# Config Tests
# ============================================================================

class TestAdjudicatorConfig:
    """Tests for AdjudicatorConfig."""

    def test_default_weights_sum_to_100(self):
        """Verify default weights sum to 100."""
        config = AdjudicatorConfig()
        total = (
            config.signal_strength_weight +
            config.pattern_confluence_weight +
            config.volatility_contraction_weight +
            config.sector_strength_weight
        )
        assert abs(total - 100.0) < 0.01

    def test_validate_returns_true_for_valid_config(self):
        """Test validation passes for valid config."""
        config = AdjudicatorConfig()
        assert config.validate() is True

    def test_validate_returns_false_for_invalid_config(self):
        """Test validation fails for invalid weights."""
        config = AdjudicatorConfig(
            signal_strength_weight=50.0,
            pattern_confluence_weight=50.0,
            volatility_contraction_weight=50.0,  # Too high
            sector_strength_weight=10.0,
        )
        assert config.validate() is False


# ============================================================================
# Signal Strength Tests
# ============================================================================

class TestSignalStrength:
    """Tests for signal strength scoring."""

    def test_extreme_ibs_gets_max_score(self, adjudicator, sample_price_data):
        """IBS < 0.05 should get max IBS score."""
        signal = {'symbol': 'AAPL', 'strategy': 'ibs_rsi', 'ibs': 0.02, 'rsi2': 3.0}
        score, ibs, rsi, tier = adjudicator._score_signal_strength(signal, sample_price_data)

        assert ibs == 0.02
        assert tier == "extreme"
        assert score >= 90  # Both IBS and RSI are extreme

    def test_near_extreme_rsi_gets_partial_score(self, adjudicator, sample_price_data):
        """RSI between 5-10 should get partial score (60 points)."""
        signal = {'symbol': 'AAPL', 'strategy': 'ibs_rsi', 'ibs': 0.03, 'rsi2': 7.5}
        score, ibs, rsi, tier = adjudicator._score_signal_strength(signal, sample_price_data)

        assert tier == "near_extreme"
        assert 70 <= score <= 90  # IBS extreme, RSI near-extreme

    def test_normal_rsi_gets_no_bonus(self, adjudicator, sample_price_data):
        """RSI >= 10 should get no RSI bonus."""
        signal = {'symbol': 'AAPL', 'strategy': 'ibs_rsi', 'ibs': 0.03, 'rsi2': 25.0}
        score, ibs, rsi, tier = adjudicator._score_signal_strength(signal, sample_price_data)

        assert tier == "normal"
        assert score <= 60  # Only IBS score, no RSI bonus


# ============================================================================
# Pattern Confluence Tests
# ============================================================================

class TestPatternConfluence:
    """Tests for pattern confluence detection."""

    def test_detects_ibs_oversold_pattern(self, adjudicator):
        """Should detect IBS < 0.10 as oversold pattern."""
        # Create price data with IBS < 0.10
        df = pd.DataFrame({
            'high': [100.0],
            'low': [90.0],
            'close': [90.5],  # IBS = 0.05
            'volume': [1000000],
        })
        signal = {'symbol': 'AAPL', 'strategy': 'ibs_rsi'}

        score, patterns = adjudicator._score_pattern_confluence(signal, df)

        assert "IBS_OVERSOLD" in patterns

    def test_multiple_patterns_increase_score(self, adjudicator):
        """More patterns should result in higher score."""
        # Create data with multiple oversold patterns
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        closes = [100 - i * 0.5 for i in range(30)]  # Declining prices

        df = pd.DataFrame({
            'timestamp': dates,
            'high': [c * 1.01 for c in closes],
            'low': [c * 0.99 for c in closes],
            'close': closes,
            'volume': [1000000] * 30,
        })
        signal = {'symbol': 'AAPL', 'strategy': 'ibs_rsi'}

        score, patterns = adjudicator._score_pattern_confluence(signal, df)

        # Should detect at least DOWN_STREAK and possibly others
        assert len(patterns) >= 1
        assert score > 0


# ============================================================================
# Volatility Contraction Tests
# ============================================================================

class TestVolatilityContraction:
    """Tests for volatility contraction scoring."""

    def test_low_percentile_gets_high_score(self, adjudicator):
        """BB width at low percentile should get high score."""
        # Create price data with current low BB width
        dates = pd.date_range(end=datetime.now(), periods=80, freq='D')
        np.random.seed(44)

        # High volatility historically, low currently
        closes = [100 + np.random.normal(0, 5) for _ in range(60)]
        # Last 20 days: very tight range
        closes.extend([100 + np.random.normal(0, 0.5) for _ in range(20)])

        df = pd.DataFrame({
            'timestamp': dates,
            'close': closes,
        })

        score, percentile = adjudicator._score_volatility_contraction(df)

        # Should have low percentile (contracted volatility)
        assert percentile < 30
        assert score >= 60

    def test_insufficient_data_returns_neutral(self, adjudicator):
        """Insufficient data should return neutral score."""
        df = pd.DataFrame({'close': [100.0] * 5})  # Too little data

        score, percentile = adjudicator._score_volatility_contraction(df)

        assert score == 0.0
        assert percentile == 50.0


# ============================================================================
# Sector Strength Tests
# ============================================================================

class TestSectorStrength:
    """Tests for sector strength scoring."""

    def test_outperforming_sector_gets_bonus(self, adjudicator):
        """Outperforming sector should get high score."""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')

        # Symbol outperforming SPY by 5%
        symbol_prices = [100 + i * 0.3 for i in range(30)]  # +9% over period
        spy_prices = [450 + i * 0.1 for i in range(30)]     # +2% over period

        symbol_data = pd.DataFrame({
            'timestamp': dates,
            'close': symbol_prices,
        })
        spy_data = pd.DataFrame({
            'timestamp': dates,
            'close': spy_prices,
        })

        score, sector, sym_ret, spy_ret = adjudicator._score_sector_strength(
            'AAPL', symbol_data, spy_data
        )

        assert score >= 80  # Outperforming
        assert sym_ret > spy_ret

    def test_underperforming_sector_gets_no_bonus(self, adjudicator):
        """Underperforming sector should get no bonus."""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')

        # Symbol underperforming SPY
        symbol_prices = [100 - i * 0.1 for i in range(30)]  # -3% over period
        spy_prices = [450 + i * 0.2 for i in range(30)]     # +4% over period

        symbol_data = pd.DataFrame({
            'timestamp': dates,
            'close': symbol_prices,
        })
        spy_data = pd.DataFrame({
            'timestamp': dates,
            'close': spy_prices,
        })

        score, sector, sym_ret, spy_ret = adjudicator._score_sector_strength(
            'AAPL', symbol_data, spy_data
        )

        assert score == 0.0  # Underperforming
        assert sym_ret < spy_ret


# ============================================================================
# Integration Tests
# ============================================================================

class TestAdjudicatorIntegration:
    """Integration tests for full adjudication pipeline."""

    def test_adjudicate_returns_sorted_dataframe(
        self, adjudicator, sample_signals, sample_price_data, sample_spy_data
    ):
        """Adjudicate should return signals sorted by score."""
        # Add symbol column to price data
        price_data = sample_price_data.copy()
        price_data['symbol'] = 'AAPL'

        # Add second symbol data
        msft_data = price_data.copy()
        msft_data['symbol'] = 'MSFT'

        combined = pd.concat([price_data, msft_data], ignore_index=True)

        result = adjudicator.adjudicate(
            signals=sample_signals,
            price_data=combined,
            spy_data=sample_spy_data,
        )

        assert not result.empty
        assert 'adjudication_score' in result.columns

        # Check sorted descending
        if len(result) > 1:
            scores = result['adjudication_score'].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_get_top_n_returns_correct_count(
        self, adjudicator, sample_signals, sample_price_data
    ):
        """get_top_n should return at most N signals."""
        price_data = sample_price_data.copy()
        price_data['symbol'] = 'AAPL'

        result = adjudicator.get_top_n(
            signals=sample_signals,
            price_data=price_data,
            n=1,
        )

        assert len(result) <= 1

    def test_empty_signals_returns_empty(self, adjudicator, sample_price_data):
        """Empty signals should return empty DataFrame."""
        result = adjudicator.adjudicate(
            signals=pd.DataFrame(),
            price_data=sample_price_data,
        )

        assert result.empty


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_adjudicator_returns_singleton(self):
        """get_adjudicator should return singleton instance."""
        adj1 = get_adjudicator()
        adj2 = get_adjudicator()

        assert adj1 is adj2

    def test_get_adjudicator_with_config_creates_new(self):
        """get_adjudicator with config should create new instance."""
        config = AdjudicatorConfig(signal_strength_weight=50.0)
        adj = get_adjudicator(config)

        assert adj.config.signal_strength_weight == 50.0

    def test_adjudicate_signals_convenience(
        self, sample_signals, sample_price_data
    ):
        """adjudicate_signals should work as convenience function."""
        price_data = sample_price_data.copy()
        price_data['symbol'] = 'AAPL'

        result = adjudicate_signals(
            signals=sample_signals,
            price_data=price_data,
            max_signals=3,
        )

        assert isinstance(result, pd.DataFrame)


# ============================================================================
# RSI Tiering Tests
# ============================================================================

class TestRSITiering:
    """Specific tests for RSI tiering logic."""

    def test_rsi_under_5_is_extreme(self, adjudicator, sample_price_data):
        """RSI < 5 should be classified as extreme."""
        signal = {'symbol': 'AAPL', 'strategy': 'ibs_rsi', 'ibs': 0.50, 'rsi2': 4.9}
        _, _, _, tier = adjudicator._score_signal_strength(signal, sample_price_data)
        assert tier == "extreme"

    def test_rsi_5_to_10_is_near_extreme(self, adjudicator, sample_price_data):
        """RSI 5-10 should be classified as near_extreme."""
        signal = {'symbol': 'AAPL', 'strategy': 'ibs_rsi', 'ibs': 0.50, 'rsi2': 7.0}
        _, _, _, tier = adjudicator._score_signal_strength(signal, sample_price_data)
        assert tier == "near_extreme"

    def test_rsi_over_10_is_normal(self, adjudicator, sample_price_data):
        """RSI >= 10 should be classified as normal."""
        signal = {'symbol': 'AAPL', 'strategy': 'ibs_rsi', 'ibs': 0.50, 'rsi2': 10.0}
        _, _, _, tier = adjudicator._score_signal_strength(signal, sample_price_data)
        assert tier == "normal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
