"""
Tests for research module (features, alphas, screener).
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
        from research.features import FeatureExtractor, FEATURE_REGISTRY
        assert FeatureExtractor is not None
        assert len(FEATURE_REGISTRY) > 0

    def test_feature_count(self):
        """Test number of registered features."""
        from research.features import FEATURE_REGISTRY
        assert len(FEATURE_REGISTRY) >= 20, "Should have at least 20 features"

    def test_extract_single_feature(self, sample_ohlcv):
        """Test extracting a single feature."""
        from research.features import FeatureExtractor

        extractor = FeatureExtractor(features=['return_5d'], zscore=False)
        result = extractor.extract(sample_ohlcv)

        assert 'return_5d' in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_extract_multiple_features(self, sample_ohlcv):
        """Test extracting multiple features."""
        from research.features import FeatureExtractor

        features = ['return_5d', 'rsi_14', 'volatility_20d']
        extractor = FeatureExtractor(features=features, zscore=False)
        result = extractor.extract(sample_ohlcv)

        for f in features:
            assert f in result.columns

    def test_zscore_features(self, multi_symbol_ohlcv):
        """Test z-scoring across symbols."""
        from research.features import FeatureExtractor

        extractor = FeatureExtractor(features=['return_5d'], zscore=True)
        result = extractor.extract(multi_symbol_ohlcv)

        assert 'return_5d_zscore' in result.columns

    def test_no_lookahead(self, sample_ohlcv):
        """Test that features don't use future data."""
        from research.features import FeatureExtractor

        extractor = FeatureExtractor(features=['return_5d'], zscore=False)
        result = extractor.extract(sample_ohlcv)

        # First 5 values should be NaN (no history)
        assert result['return_5d'].iloc[:5].isna().sum() >= 4


class TestAlphaLibrary:
    """Tests for alpha library."""

    def test_import(self):
        """Test module imports."""
        from research.alphas import AlphaLibrary, ALPHA_REGISTRY
        assert AlphaLibrary is not None
        assert len(ALPHA_REGISTRY) > 0

    def test_alpha_count(self):
        """Test number of registered alphas."""
        from research.alphas import ALPHA_REGISTRY
        assert len(ALPHA_REGISTRY) >= 15, "Should have at least 15 alphas"

    def test_compute_single_alpha(self, sample_ohlcv):
        """Test computing a single alpha."""
        from research.alphas import get_alpha_library

        library = get_alpha_library()
        signal = library.compute_alpha('rsi2_oversold', sample_ohlcv)

        assert len(signal) == len(sample_ohlcv)
        # RSI2 oversold should be binary (0 or 1)
        assert signal.dropna().isin([0, 1]).all()

    def test_compute_all_alphas(self, sample_ohlcv):
        """Test computing all alphas."""
        from research.alphas import get_alpha_library

        library = get_alpha_library()
        result = library.compute_all(sample_ohlcv)

        # Should have alpha columns
        alpha_cols = [c for c in result.columns if c.startswith('alpha_')]
        assert len(alpha_cols) > 0

    def test_alpha_categories(self):
        """Test alpha categorization."""
        from research.alphas import ALPHA_REGISTRY, get_alphas_by_category

        categories = set(a.category for a in ALPHA_REGISTRY.values())
        assert 'momentum' in categories
        assert 'mean_reversion' in categories

        momentum_alphas = get_alphas_by_category('momentum')
        assert len(momentum_alphas) >= 3

    def test_multi_symbol_alpha(self, multi_symbol_ohlcv):
        """Test alpha computation across multiple symbols."""
        from research.alphas import get_alpha_library

        library = get_alpha_library()
        signal = library.compute_alpha('momentum_3m', multi_symbol_ohlcv)

        assert len(signal) == len(multi_symbol_ohlcv)


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
        from research.screener import AlphaScreener
        assert AlphaScreener is not None

    def test_screen_basic(self, sample_ohlcv):
        """Test basic screening."""
        from research.screener import AlphaScreener

        screener = AlphaScreener(train_days=50, test_days=20)
        result = screener.screen(
            sample_ohlcv,
            alphas=['rsi2_oversold', 'momentum_breakout'],
            dataset_id='test',
        )

        assert result.alphas_tested == 2
        assert len(result.results) == 2

    def test_leaderboard(self, sample_ohlcv):
        """Test leaderboard generation."""
        from research.screener import AlphaScreener

        screener = AlphaScreener(train_days=50, test_days=20)
        result = screener.screen(
            sample_ohlcv,
            alphas=['rsi2_oversold', 'momentum_breakout', 'bollinger_lower'],
            dataset_id='test',
        )

        assert not result.leaderboard.empty
        assert 'Rank' in result.leaderboard.columns
        assert 'OOS Sharpe' in result.leaderboard.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
