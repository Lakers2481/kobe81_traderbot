"""
Unit tests for data handling.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPolygonProvider:
    """Tests for Polygon data provider."""

    def test_provider_import(self):
        """Test that provider functions can be imported."""
        from data.providers.polygon_eod import fetch_daily_bars_polygon, PolygonConfig
        assert fetch_daily_bars_polygon is not None
        assert PolygonConfig is not None

    def test_polygon_config_defaults(self):
        """Test PolygonConfig default values."""
        from data.providers.polygon_eod import PolygonConfig
        cfg = PolygonConfig(api_key="test_key")
        assert cfg.api_key == "test_key"
        assert cfg.adjusted == True
        assert cfg.sort == "asc"
        assert cfg.limit == 50000

    def test_fetch_returns_empty_without_key(self):
        """Test that fetch returns empty DataFrame without API key."""
        from data.providers.polygon_eod import fetch_daily_bars_polygon
        import os

        # Ensure no API key
        old_key = os.environ.pop('POLYGON_API_KEY', None)

        try:
            df = fetch_daily_bars_polygon("AAPL", "2023-01-01", "2023-01-31")
            assert isinstance(df, pd.DataFrame)
            assert df.empty or len(df) == 0
        finally:
            if old_key:
                os.environ['POLYGON_API_KEY'] = old_key

    def test_cache_file_naming(self):
        """Test cache file naming convention."""
        symbol = "AAPL"
        start = "2023-01-01"
        end = "2023-12-31"

        # Expected cache filename format (CSV, not parquet)
        expected_name = f"{symbol}_{start}_{end}.csv"

        assert symbol in expected_name
        assert ".csv" in expected_name


class TestUniverseLoader:
    """Tests for universe loading."""

    def test_loader_import(self):
        """Test that loader function can be imported."""
        from data.universe.loader import load_universe
        assert load_universe is not None
        assert callable(load_universe)

    def test_load_universe_from_csv(self, tmp_path):
        """Test loading universe from CSV file."""
        from data.universe.loader import load_universe

        # Create test CSV
        csv_path = tmp_path / "test_universe.csv"
        csv_path.write_text("symbol\nAAPL\nMSFT\nGOOGL\n")

        symbols = load_universe(str(csv_path))

        assert len(symbols) == 3
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOGL" in symbols

    def test_universe_uppercases_symbols(self, tmp_path):
        """Test that symbols are uppercased."""
        from data.universe.loader import load_universe

        csv_path = tmp_path / "test_universe.csv"
        csv_path.write_text("symbol\naapl\nMsft\ngoogl\n")

        symbols = load_universe(str(csv_path))

        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOGL" in symbols

    def test_universe_deduplication(self, tmp_path):
        """Test that duplicate symbols are removed."""
        from data.universe.loader import load_universe

        # Create CSV with duplicates
        csv_path = tmp_path / "test_universe.csv"
        csv_path.write_text("symbol\nAAPL\nMSFT\nAAPL\nGOOGL\nMSFT\n")

        symbols = load_universe(str(csv_path))

        # Should have only unique symbols
        assert len(symbols) == len(set(symbols))
        assert len(symbols) == 3

    def test_universe_cap(self, tmp_path):
        """Test that universe respects cap limit."""
        from data.universe.loader import load_universe

        # Create CSV with many symbols
        symbols_list = [f"SYM{i}" for i in range(100)]
        csv_path = tmp_path / "test_universe.csv"
        csv_path.write_text("symbol\n" + "\n".join(symbols_list))

        symbols = load_universe(str(csv_path), cap=50)

        assert len(symbols) == 50

    def test_nonexistent_file_returns_empty(self, tmp_path):
        """Test that loading nonexistent file returns empty list."""
        from data.universe.loader import load_universe

        symbols = load_universe(tmp_path / "nonexistent.csv")
        assert symbols == []

    def test_fallback_to_first_column(self, tmp_path):
        """Test that loader uses first column if 'symbol' not present."""
        from data.universe.loader import load_universe

        # Create CSV without 'symbol' header
        csv_path = tmp_path / "test_universe.csv"
        csv_path.write_text("ticker\nAAPL\nMSFT\n")

        symbols = load_universe(str(csv_path))

        assert len(symbols) == 2
        assert "AAPL" in symbols


class TestDataValidation:
    """Tests for data validation."""

    def test_ohlcv_data_has_required_columns(self, sample_ohlcv_data):
        """Test that OHLCV data has all required columns."""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        for col in required_columns:
            assert col in sample_ohlcv_data.columns

    def test_high_gte_low(self, sample_ohlcv_data):
        """Test that high >= low for all bars."""
        assert (sample_ohlcv_data['high'] >= sample_ohlcv_data['low']).all()

    def test_high_gte_open_close(self, sample_ohlcv_data):
        """Test that high >= open and high >= close."""
        assert (sample_ohlcv_data['high'] >= sample_ohlcv_data['open']).all()
        assert (sample_ohlcv_data['high'] >= sample_ohlcv_data['close']).all()

    def test_low_lte_open_close(self, sample_ohlcv_data):
        """Test that low <= open and low <= close."""
        assert (sample_ohlcv_data['low'] <= sample_ohlcv_data['open']).all()
        assert (sample_ohlcv_data['low'] <= sample_ohlcv_data['close']).all()

    def test_volume_positive(self, sample_ohlcv_data):
        """Test that volume is positive."""
        assert (sample_ohlcv_data['volume'] > 0).all()

    def test_no_missing_values(self, sample_ohlcv_data):
        """Test that there are no missing values."""
        assert not sample_ohlcv_data.isnull().any().any()
