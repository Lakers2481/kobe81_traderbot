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


class TestDonchianStrategy:
    """Tests for Donchian breakout strategy."""

    def test_strategy_import(self):
        from strategies.donchian.strategy import DonchianBreakoutStrategy
        assert DonchianBreakoutStrategy is not None

    def test_signal_generation(self, sample_ohlcv_data):
        from strategies.donchian.strategy import DonchianBreakoutStrategy
        strategy = DonchianBreakoutStrategy()
        signals = strategy.scan_signals_over_time(sample_ohlcv_data.copy())
        assert isinstance(signals, pd.DataFrame)


class TestICTTurtleSoupStrategy:
    """Tests for ICT Turtle Soup strategy."""

    def test_strategy_import(self):
        from strategies.ict.turtle_soup import TurtleSoupStrategy
        assert TurtleSoupStrategy is not None

    def test_signal_generation(self, sample_ohlcv_data):
        from strategies.ict.turtle_soup import TurtleSoupStrategy
        strategy = TurtleSoupStrategy()
        signals = strategy.scan_signals_over_time(sample_ohlcv_data.copy())
        assert isinstance(signals, pd.DataFrame)


class TestStrategyInterface:
    """Test that selected strategies follow the common interface."""

    @pytest.mark.parametrize("strategy_module,strategy_class", [
        ("strategies.donchian.strategy", "DonchianBreakoutStrategy"),
        ("strategies.ict.turtle_soup", "TurtleSoupStrategy"),
    ])
    def test_strategy_has_required_methods(self, strategy_module, strategy_class):
        import importlib
        module = importlib.import_module(strategy_module)
        cls = getattr(module, strategy_class)
        strategy = cls()
        assert hasattr(strategy, 'generate_signals')
        assert hasattr(strategy, 'scan_signals_over_time')
        assert callable(getattr(strategy, 'generate_signals'))
        assert callable(getattr(strategy, 'scan_signals_over_time'))
