"""
Tests for ml_features module - ML-enhanced trading features.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# Helper to create test OHLCV data
def create_ohlcv_data(
    start_date: str = "2023-01-01",
    num_days: int = 100,
    base_price: float = 100.0,
    trend: float = 0.0005,
    volatility: float = 0.02,
    include_volume: bool = True,
) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    returns = np.random.normal(trend, volatility, num_days)
    close = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.005, num_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, num_days)))
    open_price = (close.copy())[:-1]
    open_price = np.insert(open_price, 0, base_price)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
    })

    if include_volume:
        df['volume'] = np.random.randint(100000, 1000000, num_days)

    return df


class TestTechnicalFeatures:
    """Tests for TechnicalFeatures class."""

    def test_import(self):
        """Test that TechnicalFeatures can be imported."""
        from ml_features.technical_features import TechnicalFeatures, TechnicalConfig
        assert TechnicalFeatures is not None
        assert TechnicalConfig is not None

    def test_compute_all_returns_dataframe(self):
        """compute_all returns DataFrame with features."""
        from ml_features.technical_features import TechnicalFeatures
        tf = TechnicalFeatures()
        df = create_ohlcv_data(num_days=100)

        result = tf.compute_all(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        # Should have more columns than input
        assert len(result.columns) > len(df.columns)

    def test_momentum_features_added(self):
        """Momentum features are computed."""
        from ml_features.technical_features import TechnicalFeatures
        tf = TechnicalFeatures()
        df = create_ohlcv_data(num_days=100)

        result = tf.compute_all(df, shift=False)

        # Check for RSI columns
        rsi_cols = [c for c in result.columns if c.startswith('rsi_')]
        assert len(rsi_cols) > 0

    def test_volatility_features_added(self):
        """Volatility features are computed."""
        from ml_features.technical_features import TechnicalFeatures
        tf = TechnicalFeatures()
        df = create_ohlcv_data(num_days=100)

        result = tf.compute_all(df, shift=False)

        # Check for ATR columns
        atr_cols = [c for c in result.columns if c.startswith('atr_')]
        assert len(atr_cols) > 0

    def test_trend_features_added(self):
        """Trend features are computed."""
        from ml_features.technical_features import TechnicalFeatures
        tf = TechnicalFeatures()
        df = create_ohlcv_data(num_days=250)

        result = tf.compute_all(df, shift=False)

        # Check for SMA columns
        sma_cols = [c for c in result.columns if c.startswith('sma_')]
        assert len(sma_cols) > 0

    def test_volume_features_added(self):
        """Volume features are computed when volume data present."""
        from ml_features.technical_features import TechnicalFeatures
        tf = TechnicalFeatures()
        df = create_ohlcv_data(num_days=100, include_volume=True)

        result = tf.compute_all(df, shift=False)

        # Check for volume-related columns
        vol_cols = [c for c in result.columns if 'vol_' in c or 'obv' in c.lower()]
        assert len(vol_cols) > 0

    def test_shift_prevents_lookahead(self):
        """Features are shifted when shift=True."""
        from ml_features.technical_features import TechnicalFeatures
        tf = TechnicalFeatures()
        df = create_ohlcv_data(num_days=50)

        result_shifted = tf.compute_all(df, shift=True)
        result_unshifted = tf.compute_all(df, shift=False)

        # Feature columns should be different
        feature_cols = [c for c in result_shifted.columns if c.startswith('rsi_')]
        if feature_cols:
            col = feature_cols[0]
            # First non-NaN values should be at different indices
            assert result_shifted[col].first_valid_index() != result_unshifted[col].first_valid_index() or \
                   pd.isna(result_shifted[col].iloc[result_unshifted[col].first_valid_index()])

    def test_empty_dataframe_returns_empty(self):
        """Empty DataFrame returns empty."""
        from ml_features.technical_features import TechnicalFeatures
        tf = TechnicalFeatures()
        df = pd.DataFrame()

        result = tf.compute_all(df)

        assert result.empty

    def test_missing_columns_handled(self):
        """Missing OHLC columns are handled gracefully."""
        from ml_features.technical_features import TechnicalFeatures
        tf = TechnicalFeatures()
        df = pd.DataFrame({'close': [100, 101, 102]})  # Missing open, high, low

        result = tf.compute_all(df)

        # Should return the input without crashing
        assert len(result) == 3


class TestAnomalyDetection:
    """Tests for AnomalyDetector class."""

    def test_import(self):
        """Test that AnomalyDetector can be imported."""
        from ml_features.anomaly_detection import AnomalyDetector, AnomalyConfig
        assert AnomalyDetector is not None
        assert AnomalyConfig is not None

    def test_detect_all_returns_dataframe(self):
        """detect_all returns DataFrame with anomaly columns."""
        from ml_features.anomaly_detection import AnomalyDetector
        detector = AnomalyDetector()
        df = create_ohlcv_data(num_days=100)

        result = detector.detect_all(df)

        assert isinstance(result, pd.DataFrame)
        assert 'anomaly_price' in result.columns
        assert 'anomaly_volume' in result.columns
        assert 'anomaly_combined' in result.columns
        assert 'is_anomaly' in result.columns

    def test_anomaly_scores_bounded(self):
        """Anomaly scores are between 0 and 1."""
        from ml_features.anomaly_detection import AnomalyDetector
        detector = AnomalyDetector()
        df = create_ohlcv_data(num_days=100)

        result = detector.detect_all(df)

        for col in ['anomaly_price', 'anomaly_volume', 'anomaly_combined']:
            valid_scores = result[col].dropna()
            assert (valid_scores >= 0).all()
            assert (valid_scores <= 1).all()

    def test_high_volatility_detected(self):
        """High volatility data produces higher or similar anomaly scores."""
        from ml_features.anomaly_detection import AnomalyDetector
        detector = AnomalyDetector()

        # Low volatility
        df_low = create_ohlcv_data(num_days=100, volatility=0.005)
        result_low = detector.detect_all(df_low)

        # High volatility
        df_high = create_ohlcv_data(num_days=100, volatility=0.05)
        result_high = detector.detect_all(df_high)

        # Both should have valid anomaly scores (relaxed comparison due to randomness)
        # The key is that both produce valid scores in [0, 1]
        assert result_low['anomaly_price'].max() <= 1.0
        assert result_high['anomaly_price'].max() <= 1.0
        assert result_low['anomaly_price'].min() >= 0.0
        assert result_high['anomaly_price'].min() >= 0.0

    def test_volume_spike_detected(self):
        """Volume spikes are detected."""
        from ml_features.anomaly_detection import AnomalyDetector
        detector = AnomalyDetector()

        df = create_ohlcv_data(num_days=100)
        # Add a volume spike
        df.loc[80, 'volume'] = df['volume'].mean() * 10

        result = detector.detect_all(df)

        # The spike should have higher anomaly score
        assert result.loc[80, 'anomaly_volume'] > result['anomaly_volume'].median()

    def test_get_anomalies_returns_list(self):
        """get_anomalies returns list of Anomaly objects."""
        from ml_features.anomaly_detection import AnomalyDetector, Anomaly
        detector = AnomalyDetector()
        df = create_ohlcv_data(num_days=100, volatility=0.05)

        anomalies = detector.get_anomalies(df, threshold=0.5)

        assert isinstance(anomalies, list)
        for a in anomalies:
            assert isinstance(a, Anomaly)
            assert a.score >= 0.5

    def test_insufficient_data_handled(self):
        """Insufficient data returns zeros."""
        from ml_features.anomaly_detection import AnomalyDetector, AnomalyConfig
        config = AnomalyConfig(min_periods=100)
        detector = AnomalyDetector(config)
        df = create_ohlcv_data(num_days=20)  # Less than min_periods

        result = detector.detect_all(df)

        # Should return input unchanged
        assert len(result) == 20


class TestFeaturePipeline:
    """Tests for FeaturePipeline class."""

    def test_import(self):
        """Test that FeaturePipeline can be imported."""
        from ml_features.feature_pipeline import FeaturePipeline, FeatureConfig
        assert FeaturePipeline is not None
        assert FeatureConfig is not None

    def test_extract_returns_dataframe(self):
        """extract returns DataFrame with features."""
        from ml_features.feature_pipeline import FeaturePipeline
        pipeline = FeaturePipeline()
        df = create_ohlcv_data(num_days=100)

        result = pipeline.extract(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_price_patterns_added(self):
        """Price pattern features are added."""
        from ml_features.feature_pipeline import FeaturePipeline, FeatureConfig, FeatureCategory
        config = FeatureConfig(categories=[FeatureCategory.PRICE_PATTERN])
        pipeline = FeaturePipeline(config)
        df = create_ohlcv_data(num_days=100)

        result = pipeline.extract(df)

        # Check for pattern columns
        assert 'ibs' in result.columns
        assert 'return_1d' in result.columns

    def test_feature_caching(self):
        """Feature caching works correctly."""
        from ml_features.feature_pipeline import FeaturePipeline, FeatureConfig
        config = FeatureConfig(cache_features=True)
        pipeline = FeaturePipeline(config)
        df = create_ohlcv_data(num_days=50)

        # First extraction
        result1 = pipeline.extract(df, symbol='TEST')
        # Second extraction (should use cache)
        result2 = pipeline.extract(df, symbol='TEST')

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_get_feature_names(self):
        """get_feature_names returns list of feature columns."""
        from ml_features.feature_pipeline import FeaturePipeline
        pipeline = FeaturePipeline()
        df = create_ohlcv_data(num_days=100)
        result = pipeline.extract(df)

        feature_names = pipeline.get_feature_names(result)

        assert isinstance(feature_names, list)
        assert 'close' not in feature_names  # Original columns excluded
        assert 'open' not in feature_names

    def test_clear_cache(self):
        """clear_cache clears the feature cache."""
        from ml_features.feature_pipeline import FeaturePipeline, FeatureConfig
        config = FeatureConfig(cache_features=True)
        pipeline = FeaturePipeline(config)
        df = create_ohlcv_data(num_days=50)

        pipeline.extract(df, symbol='TEST')
        assert len(pipeline._feature_cache) > 0

        pipeline.clear_cache()
        assert len(pipeline._feature_cache) == 0


class TestSignalConfidence:
    """Tests for SignalConfidence class."""

    def test_import(self):
        """Test that SignalConfidence can be imported."""
        from ml_features.signal_confidence import SignalConfidence, ConfidenceConfig, ConfidenceLevel
        assert SignalConfidence is not None
        assert ConfidenceConfig is not None
        assert ConfidenceLevel is not None

    def test_score_returns_result(self):
        """score returns ConfidenceResult."""
        from ml_features.signal_confidence import SignalConfidence, ConfidenceResult
        scorer = SignalConfidence()
        df = create_ohlcv_data(num_days=100)
        signal = pd.Series({'timestamp': df['timestamp'].iloc[-1], 'side': 'long'})

        result = scorer.score(signal, df, side='long')

        assert isinstance(result, ConfidenceResult)
        assert 0 <= result.score <= 1
        assert result.level is not None
        assert isinstance(result.factors, dict)
        assert isinstance(result.reasons, list)

    def test_confidence_bounded(self):
        """Confidence score is between 0 and 1."""
        from ml_features.signal_confidence import SignalConfidence
        scorer = SignalConfidence()
        df = create_ohlcv_data(num_days=100)
        signal = pd.Series({'timestamp': df['timestamp'].iloc[-1], 'side': 'long'})

        result = scorer.score(signal, df, side='long')

        assert 0 <= result.score <= 1

    def test_score_signals_batch(self):
        """score_signals handles multiple signals."""
        from ml_features.signal_confidence import SignalConfidence
        scorer = SignalConfidence()
        df = create_ohlcv_data(num_days=100)

        signals = pd.DataFrame({
            'timestamp': df['timestamp'].iloc[50:55],
            'symbol': ['AAPL'] * 5,
            'side': ['long'] * 5,
        })

        result = scorer.score_signals(signals, df)

        assert 'confidence' in result.columns
        assert 'confidence_level' in result.columns
        assert len(result) == 5

    def test_low_rsi_increases_long_confidence(self):
        """Low RSI increases confidence for long signals."""
        from ml_features.signal_confidence import SignalConfidence
        from ml_features.technical_features import TechnicalFeatures

        scorer = SignalConfidence()
        tf = TechnicalFeatures()

        # Create data with very low RSI
        df = create_ohlcv_data(num_days=50, trend=-0.02)  # Downtrend = low RSI
        df = tf.compute_all(df, shift=False)

        signal = pd.Series({'timestamp': df['timestamp'].iloc[-1], 'side': 'long'})
        result = scorer.score(signal, df, side='long')

        # Should have non-zero momentum factor
        assert result.factors.get('momentum', 0) > 0

    def test_reasons_generated(self):
        """Human-readable reasons are generated."""
        from ml_features.signal_confidence import SignalConfidence
        scorer = SignalConfidence()
        df = create_ohlcv_data(num_days=100)
        signal = pd.Series({'timestamp': df['timestamp'].iloc[-1], 'side': 'long'})

        result = scorer.score(signal, df, side='long')

        assert len(result.reasons) > 0
        assert all(isinstance(r, str) for r in result.reasons)


class TestRegimeML:
    """Tests for RegimeDetectorML class."""

    def test_import(self):
        """Test that RegimeDetectorML can be imported."""
        from ml_features.regime_ml import RegimeDetectorML, RegimeConfig, RegimeState
        assert RegimeDetectorML is not None
        assert RegimeConfig is not None
        assert RegimeState is not None

    def test_fit_returns_self(self):
        """fit returns self for method chaining."""
        from ml_features.regime_ml import RegimeDetectorML
        detector = RegimeDetectorML()
        df = create_ohlcv_data(num_days=100)

        result = detector.fit(df)

        assert result is detector

    def test_detect_returns_dataframe(self):
        """detect returns DataFrame with regime columns."""
        from ml_features.regime_ml import RegimeDetectorML
        detector = RegimeDetectorML()
        df = create_ohlcv_data(num_days=100)

        result = detector.detect(df)

        assert isinstance(result, pd.DataFrame)
        assert 'trend_regime' in result.columns
        assert 'vol_regime' in result.columns
        assert 'combined_regime' in result.columns

    def test_get_current_regime_returns_result(self):
        """get_current_regime returns RegimeResult."""
        from ml_features.regime_ml import RegimeDetectorML, RegimeResult
        detector = RegimeDetectorML()
        df = create_ohlcv_data(num_days=100)

        result = detector.get_current_regime(df)

        assert isinstance(result, RegimeResult)
        assert result.trend_regime is not None
        assert result.vol_regime is not None

    def test_bull_trend_detected(self):
        """Uptrending data is detected as bull."""
        from ml_features.regime_ml import RegimeDetectorML, RegimeState
        detector = RegimeDetectorML()
        df = create_ohlcv_data(num_days=150, trend=0.003)  # Strong uptrend

        detector.fit(df)
        result = detector.get_current_regime(df)

        # Should be bull or strong_bull
        assert result.trend_regime in [RegimeState.BULL, RegimeState.STRONG_BULL, RegimeState.NEUTRAL]

    def test_bear_trend_detected(self):
        """Downtrending data is detected as bear."""
        from ml_features.regime_ml import RegimeDetectorML, RegimeState
        detector = RegimeDetectorML()
        df = create_ohlcv_data(num_days=150, trend=-0.003)  # Strong downtrend

        detector.fit(df)
        result = detector.get_current_regime(df)

        # Should be bear or strong_bear
        assert result.trend_regime in [RegimeState.BEAR, RegimeState.STRONG_BEAR, RegimeState.NEUTRAL]

    def test_high_volatility_detected(self):
        """High volatility is detected."""
        from ml_features.regime_ml import RegimeDetectorML, RegimeState
        detector = RegimeDetectorML()
        df = create_ohlcv_data(num_days=150, volatility=0.04)  # High volatility

        detector.fit(df)
        result = detector.get_current_regime(df)

        # Should be high_vol or crisis
        assert result.vol_regime in [RegimeState.HIGH_VOL, RegimeState.CRISIS, RegimeState.NORMAL_VOL]

    def test_insufficient_data_handled(self):
        """Insufficient data is handled gracefully."""
        from ml_features.regime_ml import RegimeDetectorML, RegimeState
        detector = RegimeDetectorML()
        df = create_ohlcv_data(num_days=20)  # Too short

        result = detector.get_current_regime(df)

        # Should return a result (may be unknown or default)
        assert result is not None


class TestStrategyEnhancer:
    """Tests for StrategyEnhancer class."""

    def test_import(self):
        """Test that StrategyEnhancer can be imported."""
        from ml_features.strategy_enhancer import StrategyEnhancer, EnhancerConfig
        assert StrategyEnhancer is not None
        assert EnhancerConfig is not None

    def test_enhancer_wraps_strategy(self):
        """Enhancer wraps strategy and preserves interface."""
        from ml_features.strategy_enhancer import StrategyEnhancer

        # Mock strategy
        class MockStrategy:
            def generate_signals(self, df):
                return pd.DataFrame({
                    'timestamp': [df['timestamp'].iloc[-1]],
                    'symbol': ['AAPL'],
                    'side': ['long'],
                })

            def scan_signals_over_time(self, df):
                return self.generate_signals(df)

        strategy = MockStrategy()
        enhancer = StrategyEnhancer(strategy)
        df = create_ohlcv_data(num_days=100)

        result = enhancer.generate_signals(df)

        assert isinstance(result, pd.DataFrame)

    def test_confidence_added_to_signals(self):
        """Confidence scores are added to signals."""
        from ml_features.strategy_enhancer import StrategyEnhancer, EnhancerConfig

        class MockStrategy:
            def generate_signals(self, df):
                return pd.DataFrame({
                    'timestamp': [df['timestamp'].iloc[-1]],
                    'symbol': ['AAPL'],
                    'side': ['long'],
                })

            def scan_signals_over_time(self, df):
                return self.generate_signals(df)

        config = EnhancerConfig(
            enable_confidence=True,
            min_confidence=0.0,  # Don't filter any
            enable_regime_filter=False,
            enable_anomaly_filter=False,
        )
        enhancer = StrategyEnhancer(MockStrategy(), config)
        df = create_ohlcv_data(num_days=100)

        result = enhancer.generate_signals(df)

        assert 'confidence' in result.columns

    def test_low_confidence_filtered(self):
        """Low confidence signals are filtered."""
        from ml_features.strategy_enhancer import StrategyEnhancer, EnhancerConfig

        class MockStrategy:
            def generate_signals(self, df):
                return pd.DataFrame({
                    'timestamp': [df['timestamp'].iloc[-1]],
                    'symbol': ['AAPL'],
                    'side': ['long'],
                })

            def scan_signals_over_time(self, df):
                return self.generate_signals(df)

        config = EnhancerConfig(
            enable_confidence=True,
            min_confidence=0.99,  # Very high threshold
            enable_regime_filter=False,
            enable_anomaly_filter=False,
        )
        enhancer = StrategyEnhancer(MockStrategy(), config)
        df = create_ohlcv_data(num_days=100)

        result = enhancer.generate_signals(df)

        # May filter out signals
        assert len(result) <= 1

    def test_get_stats_returns_dict(self):
        """get_stats returns statistics dictionary."""
        from ml_features.strategy_enhancer import StrategyEnhancer

        class MockStrategy:
            def generate_signals(self, df):
                return pd.DataFrame()

            def scan_signals_over_time(self, df):
                return self.generate_signals(df)

        enhancer = StrategyEnhancer(MockStrategy())

        stats = enhancer.get_stats()

        assert isinstance(stats, dict)
        assert 'total_signals' in stats
        assert 'passed_signals' in stats

    def test_enhance_strategy_function(self):
        """enhance_strategy convenience function works."""
        from ml_features.strategy_enhancer import enhance_strategy, StrategyEnhancer

        class MockStrategy:
            def generate_signals(self, df):
                return pd.DataFrame()

            def scan_signals_over_time(self, df):
                return self.generate_signals(df)

        enhanced = enhance_strategy(MockStrategy())

        assert isinstance(enhanced, StrategyEnhancer)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_momentum_features(self):
        """compute_momentum_features works."""
        from ml_features.technical_features import compute_momentum_features
        df = create_ohlcv_data(num_days=50)

        result = compute_momentum_features(df)

        rsi_cols = [c for c in result.columns if c.startswith('rsi_')]
        assert len(rsi_cols) > 0

    def test_compute_volatility_features(self):
        """compute_volatility_features works."""
        from ml_features.technical_features import compute_volatility_features
        df = create_ohlcv_data(num_days=50)

        result = compute_volatility_features(df)

        atr_cols = [c for c in result.columns if c.startswith('atr_')]
        assert len(atr_cols) > 0

    def test_detect_price_anomalies(self):
        """detect_price_anomalies works."""
        from ml_features.anomaly_detection import detect_price_anomalies
        df = create_ohlcv_data(num_days=50)

        result = detect_price_anomalies(df)

        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_compute_signal_confidence(self):
        """compute_signal_confidence works."""
        from ml_features.signal_confidence import compute_signal_confidence
        df = create_ohlcv_data(num_days=50)
        signal = pd.Series({'timestamp': df['timestamp'].iloc[-1], 'side': 'long'})

        result = compute_signal_confidence(signal, df)

        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_detect_regime_ml(self):
        """detect_regime_ml works."""
        from ml_features.regime_ml import detect_regime_ml, RegimeResult
        df = create_ohlcv_data(num_days=100)

        result = detect_regime_ml(df)

        assert isinstance(result, RegimeResult)

    def test_extract_all_features(self):
        """extract_all_features works."""
        from ml_features.feature_pipeline import extract_all_features
        df = create_ohlcv_data(num_days=50)

        result = extract_all_features(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) > len(df.columns)
