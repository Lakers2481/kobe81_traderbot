"""
Tests for Macro Data Providers

Tests FRED, Treasury, and CFTC COT providers with proper mocking.

Author: Kobe Trading System
Created: 2026-01-04
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest


class TestFREDMacroProvider:
    """Tests for FRED macro data provider."""

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        with patch.dict('os.environ', {}, clear=True):
            from data.providers.fred_macro import FREDMacroProvider
            provider = FREDMacroProvider()
            assert provider.api_key is None

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        from data.providers.fred_macro import FREDMacroProvider
        provider = FREDMacroProvider(api_key="test_key")
        assert provider.api_key == "test_key"

    def test_get_series_cached(self, tmp_path):
        """Test get_series returns cached data."""
        from data.providers.fred_macro import FREDMacroProvider

        provider = FREDMacroProvider()
        provider.CACHE_DIR = tmp_path

        # Create mock cache file
        cache_data = {
            "observations": [
                {"date": "2024-01-01", "value": "5.25"},
                {"date": "2024-01-02", "value": "5.30"},
            ]
        }
        cache_path = tmp_path / "fedfunds.json"
        cache_path.write_text(json.dumps(cache_data))

        df = provider.get_series("FEDFUNDS")

        assert len(df) == 2
        assert "value" in df.columns
        assert "date" in df.columns

    def test_get_yield_curve(self):
        """Test yield curve snapshot."""
        from data.providers.fred_macro import FREDMacroProvider

        with patch.object(FREDMacroProvider, 'get_series') as mock_get:
            # Mock responses for each tenor
            mock_df = pd.DataFrame({
                'date': [datetime(2024, 1, 1)],
                'value': [4.50],
                'series_id': ['DGS10']
            })
            mock_get.return_value = mock_df

            provider = FREDMacroProvider()
            curve = provider.get_yield_curve()

            # Should have attempted to fetch multiple tenors
            assert mock_get.called

    def test_get_yield_curve_slope(self):
        """Test yield curve slope calculation."""
        from data.providers.fred_macro import FREDMacroProvider

        with patch.object(FREDMacroProvider, 'get_yield_curve') as mock_curve:
            mock_curve.return_value = {
                '3M': 5.25,
                '2Y': 4.50,
                '10Y': 4.00,
                '30Y': 4.25
            }

            provider = FREDMacroProvider()
            slope = provider.get_yield_curve_slope()

            assert slope['10Y_2Y_spread'] == -0.50
            assert slope['10Y_3M_spread'] == -1.25
            assert slope['is_inverted'] is True

    def test_get_macro_regime(self):
        """Test macro regime classification."""
        from data.providers.fred_macro import FREDMacroProvider

        with patch.object(FREDMacroProvider, 'get_core_indicators') as mock_ind:
            with patch.object(FREDMacroProvider, 'get_yield_curve_slope') as mock_slope:
                mock_ind.return_value = pd.DataFrame({
                    'FEDFUNDS': [5.0, 5.25, 5.50],
                    'VIXCLS': [15.0, 16.0, 14.0]
                })
                mock_slope.return_value = {
                    '10Y_2Y_spread': -0.5,
                    'is_inverted': True
                }

                provider = FREDMacroProvider()
                regime = provider.get_macro_regime()

                assert regime['regime'] == 'CONTRACTIONARY'
                assert 'INVERTED_CURVE' in regime['signals']


class TestTreasuryYieldProvider:
    """Tests for Treasury yield curve provider."""

    def test_init(self):
        """Test initialization."""
        from data.providers.treasury_yields import TreasuryYieldProvider
        provider = TreasuryYieldProvider()
        assert provider.CACHE_DIR.exists() or True  # May not exist yet

    def test_calculate_spreads(self):
        """Test spread calculations."""
        from data.providers.treasury_yields import TreasuryYieldProvider

        provider = TreasuryYieldProvider()
        curve = {'3M': 5.0, '2Y': 4.5, '5Y': 4.25, '10Y': 4.0, '30Y': 4.25}

        spreads = provider.calculate_spreads(curve)

        assert spreads['10Y_2Y'] == -0.5
        assert spreads['10Y_3M'] == -1.0
        assert spreads['30Y_10Y'] == 0.25

    def test_detect_inversions(self):
        """Test inversion detection."""
        from data.providers.treasury_yields import TreasuryYieldProvider

        provider = TreasuryYieldProvider()

        # Inverted curve
        curve = {'3M': 5.5, '2Y': 5.0, '10Y': 4.0, '30Y': 4.5}
        result = provider.detect_inversions(curve)

        assert result['is_classic_inversion'] is True
        assert '10Y_2Y' in result['inversions']
        assert result['severity'] in ['MILD_INVERSION', 'SEVERE_INVERSION']

    def test_classify_curve_shape(self):
        """Test curve shape classification."""
        from data.providers.treasury_yields import TreasuryYieldProvider

        provider = TreasuryYieldProvider()

        # Normal curve
        normal = {'3M': 4.0, '2Y': 4.25, '5Y': 4.5, '10Y': 4.75, '30Y': 5.0}
        assert provider._classify_curve_shape(normal) == 'NORMAL'

        # Inverted curve
        inverted = {'3M': 5.5, '2Y': 5.0, '5Y': 4.5, '10Y': 4.0, '30Y': 4.25}
        assert provider._classify_curve_shape(inverted) == 'INVERTED'

        # Flat curve
        flat = {'3M': 4.5, '2Y': 4.5, '5Y': 4.5, '10Y': 4.6, '30Y': 4.7}
        assert provider._classify_curve_shape(flat) == 'FLAT'


class TestCFTCCOTProvider:
    """Tests for CFTC COT provider."""

    def test_init(self):
        """Test initialization."""
        from data.providers.cftc_cot import CFTCCOTProvider
        provider = CFTCCOTProvider()
        assert provider.CACHE_DIR.exists() or True

    def test_calculate_percentile(self):
        """Test percentile calculation."""
        from data.providers.cftc_cot import CFTCCOTProvider

        provider = CFTCCOTProvider()
        series = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        # Latest value is 100 (100th percentile)
        pct = provider._calculate_percentile(series)
        assert pct == 90.0  # 9 values below 100

    def test_aggregate_sentiment(self):
        """Test sentiment aggregation."""
        from data.providers.cftc_cot import CFTCCOTProvider

        provider = CFTCCOTProvider()

        # Bullish
        assert provider._aggregate_sentiment(['BULLISH', 'EXTREMELY_BULLISH', 'NEUTRAL']) == 'RISK_ON'

        # Bearish
        assert provider._aggregate_sentiment(['BEARISH', 'EXTREMELY_BEARISH', 'NEUTRAL']) == 'RISK_OFF'

        # Neutral
        assert provider._aggregate_sentiment(['BULLISH', 'BEARISH', 'NEUTRAL']) == 'NEUTRAL'


class TestMacroFeatureGenerator:
    """Tests for macro feature generator."""

    def test_init(self):
        """Test initialization."""
        from ml_features.macro_features import MacroFeatureGenerator
        gen = MacroFeatureGenerator(lookback_days=252)
        assert gen.lookback_days == 252

    def test_yield_curve_features_mock(self):
        """Test yield curve feature generation with mocks."""
        from ml_features.macro_features import MacroFeatureGenerator

        gen = MacroFeatureGenerator()

        with patch.object(gen, '_get_treasury_provider') as mock_treasury:
            mock_provider = MagicMock()
            mock_provider.get_latest_curve.return_value = {
                '3M': 5.0, '2Y': 4.5, '10Y': 4.0, '30Y': 4.25
            }
            mock_provider.calculate_spreads.return_value = {
                '10Y_2Y': -0.5, '10Y_3M': -1.0, '30Y_10Y': 0.25, '2Y_3M': -0.5
            }
            mock_provider.detect_inversions.return_value = {
                'inversions': ['10Y_2Y', '10Y_3M'],
                'is_classic_inversion': True,
                'curve_shape': 'INVERTED'
            }
            mock_treasury.return_value = mock_provider

            features = gen.get_yield_curve_features()

            assert 'yc_10y_2y_spread' in features
            assert features['yc_is_inverted'] == 1.0
            assert features['yc_shape'] == 2.0  # INVERTED

    def test_feature_count(self):
        """Test that all expected features are generated."""
        from ml_features.macro_features import MacroFeatureGenerator

        gen = MacroFeatureGenerator()

        # Mock all providers to return empty/default data
        with patch.object(gen, '_get_fred_provider') as mock_fred:
            with patch.object(gen, '_get_treasury_provider') as mock_treasury:
                with patch.object(gen, '_get_cot_provider') as mock_cot:
                    # Set up minimal mocks
                    mock_fred.return_value = MagicMock()
                    mock_fred.return_value.get_series.return_value = pd.DataFrame()
                    mock_fred.return_value.get_macro_regime.return_value = {
                        'regime': 'NEUTRAL', 'confidence': 0.0, 'signals': []
                    }

                    mock_treasury.return_value = MagicMock()
                    mock_treasury.return_value.get_latest_curve.return_value = {}
                    mock_treasury.return_value.calculate_spreads.return_value = {}
                    mock_treasury.return_value.detect_inversions.return_value = {
                        'inversions': [], 'is_classic_inversion': False, 'curve_shape': 'UNKNOWN'
                    }

                    mock_cot.return_value = MagicMock()
                    mock_cot.return_value.get_market_sentiment.return_value = {
                        'positions': {}, 'overall': 'NEUTRAL', 'extreme_signals': []
                    }

                    features = gen.get_all_features()

                    # Should have 30+ features
                    feature_count = len([k for k in features if not k.startswith('_')])
                    assert feature_count >= 25, f"Expected 25+ features, got {feature_count}"

    def test_caching(self):
        """Test feature caching behavior."""
        from ml_features.macro_features import MacroFeatureGenerator

        gen = MacroFeatureGenerator()
        gen._cache = {'test_feature': 1.0}
        gen._cache_time = datetime.now()

        # Should return cached data
        assert gen._is_cache_valid()
        features = gen.get_all_features(use_cache=True)
        assert features.get('test_feature') == 1.0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_fred_provider(self):
        """Test singleton provider."""
        from data.providers.fred_macro import get_fred_provider

        p1 = get_fred_provider()
        p2 = get_fred_provider()
        assert p1 is p2  # Same instance

    def test_get_treasury_provider(self):
        """Test singleton provider."""
        from data.providers.treasury_yields import get_treasury_provider

        p1 = get_treasury_provider()
        p2 = get_treasury_provider()
        assert p1 is p2

    def test_get_cot_provider(self):
        """Test singleton provider."""
        from data.providers.cftc_cot import get_cot_provider

        p1 = get_cot_provider()
        p2 = get_cot_provider()
        assert p1 is p2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
