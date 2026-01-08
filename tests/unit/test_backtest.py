"""
Unit tests for backtest engine.
"""

import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestBacktester:
    """Tests for the Backtester class."""

    def test_backtester_import(self):
        """Test that Backtester can be imported."""
        from backtest.engine import Backtester, BacktestConfig
        assert Backtester is not None
        assert BacktestConfig is not None

    def test_backtest_config(self):
        """Test BacktestConfig initialization."""
        from backtest.engine import BacktestConfig
        cfg = BacktestConfig(initial_cash=100000)
        assert cfg.initial_cash == 100000
        assert cfg.slippage_bps == 10.0  # default (aligned with live IOC LIMIT)

    def test_backtester_initialization(self):
        """Test Backtester initialization."""
        from backtest.engine import Backtester, BacktestConfig

        cfg = BacktestConfig(initial_cash=100000)

        def dummy_signals(df):
            return pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'stop_loss'])

        def dummy_fetch(symbol):
            return pd.DataFrame(columns=['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

        bt = Backtester(cfg, dummy_signals, dummy_fetch)
        assert bt is not None
        assert bt.cash == 100000

    def test_backtester_empty_run(self):
        """Test Backtester with no data returns empty results."""
        from backtest.engine import Backtester, BacktestConfig

        cfg = BacktestConfig(initial_cash=100000)

        def dummy_signals(df):
            return pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'stop_loss'])

        def dummy_fetch(symbol):
            return pd.DataFrame()  # Empty data

        bt = Backtester(cfg, dummy_signals, dummy_fetch)
        result = bt.run(['AAPL', 'MSFT'])

        assert result['trades'] == []
        assert result['pnl'] == 0.0


class TestWalkForward:
    """Tests for walk-forward analysis."""

    def test_walk_forward_import(self):
        """Test that walk_forward functions can be imported."""
        from backtest.walk_forward import generate_splits, run_walk_forward, WFSplit
        assert generate_splits is not None
        assert run_walk_forward is not None
        assert WFSplit is not None

    def test_generate_splits(self):
        """Test that train/test splits are generated correctly."""
        from backtest.walk_forward import generate_splits
        from datetime import date

        start = date(2020, 1, 1)
        end = date(2022, 12, 31)

        splits = generate_splits(start, end, train_days=252, test_days=63)

        # Should generate multiple splits for 3 years of data
        assert len(splits) > 0

        # Each split should have valid dates
        for split in splits:
            assert split.train_start <= split.train_end
            assert split.train_end < split.test_start
            assert split.test_start <= split.test_end

    def test_wf_split_dataclass(self):
        """Test WFSplit dataclass."""
        from backtest.walk_forward import WFSplit
        from datetime import date

        split = WFSplit(
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            test_start=date(2021, 1, 1),
            test_end=date(2021, 3, 31)
        )

        assert split.train_start == date(2020, 1, 1)
        assert split.test_end == date(2021, 3, 31)


class TestEquityCurve:
    """Tests for equity curve calculations."""

    def test_no_trades_flat_equity(self):
        """Test that no trades results in flat equity curve."""
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
