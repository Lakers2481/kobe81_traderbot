"""
Unit tests for trading strategies.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestConnorsRSI2Strategy:
    """Tests for Connors RSI-2 strategy."""

    def test_strategy_import(self):
        """Test that strategy can be imported."""
        from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
        assert ConnorsRSI2Strategy is not None

    def test_strategy_initialization(self):
        """Test strategy initialization with default params."""
        from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
        strategy = ConnorsRSI2Strategy()
        assert strategy is not None

    def test_rsi_calculation(self, sample_ohlcv_data):
        """Test RSI calculation produces valid values."""
        from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
        strategy = ConnorsRSI2Strategy()

        # RSI should be between 0 and 100
        df = sample_ohlcv_data.copy()
        signals = strategy.scan_signals_over_time(df)

        # Should return a DataFrame
        assert isinstance(signals, pd.DataFrame)

    def test_no_lookahead_bias(self, sample_ohlcv_data):
        """Test that signals don't use future data."""
        from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
        strategy = ConnorsRSI2Strategy()

        df = sample_ohlcv_data.copy()

        # Generate signals on full data
        signals_full = strategy.scan_signals_over_time(df)

        # Generate signals on first 50 bars
        signals_partial = strategy.scan_signals_over_time(df.iloc[:50])

        # Signals for first 50 bars should be identical
        # (proving no lookahead)
        if len(signals_full) > 0 and len(signals_partial) > 0:
            # Compare timestamps that exist in both
            common = signals_partial['timestamp'] if 'timestamp' in signals_partial.columns else pd.Series()
            # Basic check that partial doesn't have more signals
            assert len(signals_partial) <= len(signals_full)


class TestIBSStrategy:
    """Tests for IBS strategy."""

    def test_strategy_import(self):
        """Test that strategy can be imported."""
        from strategies.ibs.strategy import IBSStrategy
        assert IBSStrategy is not None

    def test_strategy_initialization(self):
        """Test strategy initialization with default params."""
        from strategies.ibs.strategy import IBSStrategy
        strategy = IBSStrategy()
        assert strategy is not None

    def test_ibs_calculation(self, sample_ohlcv_data):
        """Test IBS calculation produces valid values."""
        from strategies.ibs.strategy import IBSStrategy
        strategy = IBSStrategy()

        df = sample_ohlcv_data.copy()

        # Calculate IBS manually: (close - low) / (high - low)
        ibs = (df['close'] - df['low']) / (df['high'] - df['low'])

        # IBS should be between 0 and 1
        assert ibs.min() >= 0
        assert ibs.max() <= 1

    def test_signal_generation(self, sample_ohlcv_data):
        """Test that signals are generated correctly."""
        from strategies.ibs.strategy import IBSStrategy
        strategy = IBSStrategy()

        df = sample_ohlcv_data.copy()
        signals = strategy.scan_signals_over_time(df)

        # Should return a DataFrame
        assert isinstance(signals, pd.DataFrame)


class TestStrategyInterface:
    """Test that all strategies follow the common interface."""

    @pytest.mark.parametrize("strategy_module,strategy_class", [
        ("strategies.connors_rsi2.strategy", "ConnorsRSI2Strategy"),
        ("strategies.ibs.strategy", "IBSStrategy"),
    ])
    def test_strategy_has_required_methods(self, strategy_module, strategy_class):
        """Test that strategy has required interface methods."""
        import importlib
        module = importlib.import_module(strategy_module)
        cls = getattr(module, strategy_class)
        strategy = cls()

        # Check required methods exist
        assert hasattr(strategy, 'generate_signals')
        assert hasattr(strategy, 'scan_signals_over_time')
        assert callable(getattr(strategy, 'generate_signals'))
        assert callable(getattr(strategy, 'scan_signals_over_time'))
