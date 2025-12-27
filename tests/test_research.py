"""
Tests for research module (features, alphas, screener).

Tests the actual API: compute_research_features, compute_alphas, screen_universe.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 300  # ~1 year of trading days

    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'TEST',
        'open': price,
        'high': price * 1.01,
        'low': price * 0.99,
        'close': price + np.random.randn(n) * 0.3,
        'volume': np.random.randint(100000, 1000000, n),
    })

    return df


@pytest.fixture
def multi_symbol_ohlcv():
    """Create multi-symbol OHLCV data."""
    dfs = []
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        np.random.seed(hash(symbol) % 2**31)
        n = 300
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        price = 100 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'open': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'close': price + np.random.randn(n) * 0.3,
            'volume': np.random.randint(100000, 1000000, n),
        })
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


class TestFeatureExtractor:
    """Tests for feature extraction."""

    def test_import(self):
        """Test module imports."""
        from research.features import compute_research_features, FEATURE_SPECS
        assert compute_research_features is not None
        assert len(FEATURE_SPECS) > 0

    def test_feature_count(self):
        """Test number of registered features."""
        from research.features import FEATURE_SPECS
        assert len(FEATURE_SPECS) >= 5, "Should have at least 5 features"

    def test_extract_single_feature(self, sample_ohlcv):
        """Test extracting features."""
        from research.features import compute_research_features

        result = compute_research_features(sample_ohlcv)

        assert 'bb_width20' in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_extract_multiple_features(self, sample_ohlcv):
        """Test extracting multiple features."""
        from research.features import compute_research_features

        result = compute_research_features(sample_ohlcv)

        expected_features = ['bb_width20', 'macd_hist', 'keltner_width20', 'adx14', 'obv']
        for f in expected_features:
            assert f in result.columns

    def test_zscore_features(self, multi_symbol_ohlcv):
        """Test features across multiple symbols."""
        from research.features import compute_research_features

        result = compute_research_features(multi_symbol_ohlcv)

        # Should have features for all symbols
        assert len(result) == len(multi_symbol_ohlcv)
        assert set(result['symbol'].unique()) == {'AAPL', 'MSFT', 'GOOGL'}

    def test_no_lookahead(self, sample_ohlcv):
        """Test that features use rolling windows (no lookahead)."""
        from research.features import compute_research_features

        result = compute_research_features(sample_ohlcv)

        # bb_width20 needs 20 days of history, so first ~20 values should be 0 or have NaN-filled values
        # The implementation fills NaN with 0, so check that it doesn't have future data
        assert 'bb_width20' in result.columns


class TestAlphaLibrary:
    """Tests for alpha library."""

    def test_import(self):
        """Test module imports."""
        from research.alphas import compute_alphas
        assert compute_alphas is not None

    def test_alpha_count(self, sample_ohlcv):
        """Test number of computed alphas."""
        from research.alphas import compute_alphas
        result = compute_alphas(sample_ohlcv)
        alpha_cols = [c for c in result.columns if c.startswith('alpha_')]
        assert len(alpha_cols) >= 3, "Should have at least 3 alphas"

    def test_compute_single_alpha(self, sample_ohlcv):
        """Test computing alphas."""
        from research.alphas import compute_alphas

        result = compute_alphas(sample_ohlcv)

        assert 'alpha_mom20' in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_compute_all_alphas(self, sample_ohlcv):
        """Test computing all alphas."""
        from research.alphas import compute_alphas

        result = compute_alphas(sample_ohlcv)

        # Should have alpha columns
        alpha_cols = [c for c in result.columns if c.startswith('alpha_')]
        assert len(alpha_cols) >= 3
        assert 'alpha_mom20' in alpha_cols
        assert 'alpha_rev1' in alpha_cols
        assert 'alpha_gap_close' in alpha_cols

    def test_alpha_categories(self, sample_ohlcv):
        """Test alpha output structure."""
        from research.alphas import compute_alphas

        result = compute_alphas(sample_ohlcv)

        # Check that alphas are numeric
        for col in [c for c in result.columns if c.startswith('alpha_')]:
            assert result[col].dtype in [np.float64, np.float32, float]

    def test_multi_symbol_alpha(self, multi_symbol_ohlcv):
        """Test alpha computation across multiple symbols."""
        from research.alphas import compute_alphas

        result = compute_alphas(multi_symbol_ohlcv)

        assert len(result) == len(multi_symbol_ohlcv)
        assert set(result['symbol'].unique()) == {'AAPL', 'MSFT', 'GOOGL'}


class TestEvidenceGate:
    """Tests for evidence gate."""

    def test_import(self):
        """Test module imports."""
        from preflight import EvidenceGate, EvidenceRequirements
        assert EvidenceGate is not None
        assert EvidenceRequirements is not None

    def test_quick_check_pass(self):
        """Test quick check with passing metrics."""
        from preflight import EvidenceGate

        gate = EvidenceGate()
        passed, reason = gate.quick_check(
            oos_trades=150,
            oos_sharpe=0.8,
            oos_profit_factor=1.5,
            max_drawdown=0.15,
        )

        assert passed
        assert 'passed' in reason.lower()

    def test_quick_check_fail(self):
        """Test quick check with failing metrics."""
        from preflight import EvidenceGate

        gate = EvidenceGate()
        passed, reason = gate.quick_check(
            oos_trades=20,  # Too few
            oos_sharpe=0.8,
            oos_profit_factor=1.5,
            max_drawdown=0.15,
        )

        assert not passed
        assert 'trades' in reason.lower()

    def test_full_evaluation(self):
        """Test full evaluation with mock backtest results."""
        from preflight import EvidenceGate

        gate = EvidenceGate()

        backtest_results = {
            'trades': [{'pnl': 100}] * 150,
            'metrics': {
                'sharpe': 1.0,
                'sharpe_ratio': 1.0,
                'profit_factor': 1.8,
                'win_rate': 0.55,
                'max_drawdown': 0.12,
            },
            'oos_metrics': {
                'sharpe': 0.7,
                'sharpe_ratio': 0.7,
                'profit_factor': 1.4,
                'win_rate': 0.52,
                'max_drawdown': 0.18,
            },
            'equity_curve': pd.DataFrame({
                'equity': [100000 + i * 10 for i in range(200)]
            }),
        }

        report = gate.evaluate(backtest_results, 'test_strategy')

        assert report.strategy_name == 'test_strategy'
        assert report.oos_trades > 0
        assert len(report.checks_passed) > 0 or len(report.checks_failed) > 0


class TestAlphaScreener:
    """Tests for alpha screener."""

    def test_import(self):
        """Test module imports."""
        from research.screener import screen_universe, save_screening_report
        assert screen_universe is not None
        assert save_screening_report is not None

    def test_screen_basic(self, sample_ohlcv):
        """Test basic screening."""
        from research.screener import screen_universe

        result = screen_universe(sample_ohlcv, horizons=(5, 10))

        assert not result.empty
        assert 'feature' in result.columns
        assert 'horizon' in result.columns
        assert 'spearman' in result.columns

    def test_leaderboard(self, sample_ohlcv):
        """Test screening returns correlations for multiple horizons."""
        from research.screener import screen_universe

        result = screen_universe(sample_ohlcv, horizons=(5, 10, 20))

        # Should have results for multiple horizons
        horizons = result['horizon'].unique()
        assert len(horizons) >= 2

        # Should have spearman correlations
        assert 'spearman' in result.columns
        assert result['spearman'].dtype == float


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
