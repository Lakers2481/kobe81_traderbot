"""
Unit tests for trading strategies.
"""

import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestDualStrategyScanner:
    """Tests for canonical DualStrategyScanner (IBS+RSI + Turtle Soup combined)."""

    def test_scanner_import(self):
        from strategies.registry import get_production_scanner, DualStrategyScanner
        assert DualStrategyScanner is not None
        scanner = get_production_scanner()
        assert scanner is not None

    def test_signal_generation(self, sample_ohlcv_data):
        from strategies.registry import get_production_scanner
        scanner = get_production_scanner()
        signals = scanner.scan_signals_over_time(sample_ohlcv_data.copy())
        assert isinstance(signals, pd.DataFrame)

    def test_scanner_has_required_methods(self):
        from strategies.registry import get_production_scanner
        scanner = get_production_scanner()
        assert hasattr(scanner, 'generate_signals')
        assert hasattr(scanner, 'scan_signals_over_time')
        assert callable(getattr(scanner, 'generate_signals'))
        assert callable(getattr(scanner, 'scan_signals_over_time'))


class TestStrategyRegistry:
    """Test that strategy registry works correctly."""

    def test_get_production_scanner(self):
        from strategies.registry import get_production_scanner
        scanner = get_production_scanner()
        assert scanner is not None

    def test_verified_performance_metadata(self):
        from strategies.registry import VERIFIED_PERFORMANCE
        assert VERIFIED_PERFORMANCE['version'] == 'v2.2'
        assert VERIFIED_PERFORMANCE['combined']['win_rate'] > 0.60
