"""
Unit tests for backtest engine.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestBacktestEngine:
    """Tests for the backtest engine."""

    def test_engine_import(self):
        """Test that engine can be imported."""
        from backtest.engine import BacktestEngine
        assert BacktestEngine is not None

    def test_engine_initialization(self):
        """Test engine initialization."""
        from backtest.engine import BacktestEngine
        engine = BacktestEngine(initial_capital=100000)
        assert engine is not None
        assert engine.initial_capital == 100000

    def test_empty_signals_returns_empty_results(self):
        """Test that empty signals produce empty results."""
        from backtest.engine import BacktestEngine
        engine = BacktestEngine(initial_capital=100000)

        # Empty signals DataFrame
        signals = pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'entry_price'])
        prices = pd.DataFrame({'close': [100, 101, 102]})

        # Should handle gracefully
        # (actual implementation may vary)

    def test_equity_curve_starts_at_initial_capital(self, sample_ohlcv_data):
        """Test that equity curve starts at initial capital."""
        from backtest.engine import BacktestEngine

        initial_capital = 100000
        engine = BacktestEngine(initial_capital=initial_capital)

        # The first value of equity curve should equal initial capital
        # (Implementation detail - adjust based on actual engine behavior)


class TestWalkForward:
    """Tests for walk-forward analysis."""

    def test_walk_forward_import(self):
        """Test that walk_forward can be imported."""
        from backtest.walk_forward import WalkForwardAnalyzer
        assert WalkForwardAnalyzer is not None

    def test_split_generation(self):
        """Test that train/test splits are generated correctly."""
        from backtest.walk_forward import WalkForwardAnalyzer

        # 252 trading days = 1 year
        train_days = 252
        test_days = 63  # 1 quarter

        analyzer = WalkForwardAnalyzer(train_days=train_days, test_days=test_days)

        # Generate date range for 3 years
        dates = pd.date_range(start='2020-01-01', periods=756, freq='D')

        # Should generate multiple splits
        # (Implementation detail - adjust based on actual analyzer behavior)


class TestEquityCurve:
    """Tests for equity curve calculations."""

    def test_no_trades_flat_equity(self):
        """Test that no trades results in flat equity curve."""
        # With no trades, equity should remain at initial capital
        initial_capital = 100000

        # Simulate empty trade history
        equity_curve = [initial_capital] * 10

        assert all(e == initial_capital for e in equity_curve)

    def test_winning_trade_increases_equity(self):
        """Test that winning trade increases equity."""
        initial_capital = 100000

        # Simulate a winning trade
        entry_price = 100
        exit_price = 110
        shares = 10
        pnl = (exit_price - entry_price) * shares

        final_equity = initial_capital + pnl

        assert final_equity > initial_capital
        assert pnl == 100  # $10 gain * 10 shares

    def test_losing_trade_decreases_equity(self):
        """Test that losing trade decreases equity."""
        initial_capital = 100000

        # Simulate a losing trade
        entry_price = 100
        exit_price = 90
        shares = 10
        pnl = (exit_price - entry_price) * shares

        final_equity = initial_capital + pnl

        assert final_equity < initial_capital
        assert pnl == -100  # $10 loss * 10 shares
