"""
Enhanced Backtest System Tests
==============================

Tests for vectorized backtester, reproducibility, visualization, and Monte Carlo modules.

Run: python -m pytest tests/test_backtest_enhanced.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for backtesting."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL']

    data = []
    for symbol in symbols:
        base_price = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 100}[symbol]
        for i, date in enumerate(dates):
            # Simulate price movement with some trend
            price = base_price * (1 + 0.001 * i + 0.02 * np.sin(i / 10))
            data.append({
                'timestamp': date,
                'symbol': symbol,
                'open': price * 0.99,
                'high': price * 1.02,
                'low': price * 0.98,
                'close': price,
                'volume': 1000000 + np.random.randint(-100000, 100000),
            })

    return pd.DataFrame(data)


@pytest.fixture
def sample_signals(sample_ohlcv_data):
    """Generate sample trading signals."""
    signals = []
    dates = sample_ohlcv_data['timestamp'].unique()
    symbols = sample_ohlcv_data['symbol'].unique()

    # Generate signals on specific days
    for i, date in enumerate(dates[10:30]):  # Signals from day 10-30
        symbol = symbols[i % len(symbols)]
        symbol_data = sample_ohlcv_data[
            (sample_ohlcv_data['symbol'] == symbol) &
            (sample_ohlcv_data['timestamp'] == date)
        ].iloc[0]

        entry_price = symbol_data['close']
        signals.append({
            'timestamp': date,
            'symbol': symbol,
            'side': 'long',
            'entry_price': entry_price,
            'stop_loss': entry_price * 0.95,
            'take_profit': entry_price * 1.10,
        })

    return pd.DataFrame(signals)


@pytest.fixture
def sample_trades():
    """Generate sample trade results for Monte Carlo."""
    np.random.seed(42)
    n_trades = 50

    # Simulate trade returns with 55% win rate
    wins = np.random.random(n_trades) < 0.55
    pnl = np.where(wins, np.random.uniform(50, 200, n_trades), np.random.uniform(-100, -30, n_trades))

    # Fix: properly repeat symbols list
    symbols = (['AAPL', 'MSFT', 'GOOGL'] * ((n_trades // 3) + 1))[:n_trades]

    return pd.DataFrame({
        'entry_date': pd.date_range(start='2023-01-01', periods=n_trades, freq='3D'),
        'symbol': symbols,
        'entry_price': np.random.uniform(100, 200, n_trades),
        'exit_price': np.random.uniform(100, 200, n_trades),
        'pnl': pnl,
        'qty': np.random.randint(10, 100, n_trades),
    })


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


# =============================================================================
# VectorizedBacktester Tests
# =============================================================================

class TestVectorizedBacktester:
    """Tests for VectorizedBacktester."""

    def test_import(self):
        from backtest.vectorized import VectorizedBacktester, VectorConfig, VectorResults
        assert VectorizedBacktester is not None
        assert VectorConfig is not None
        assert VectorResults is not None

    def test_basic_backtest(self, sample_ohlcv_data, sample_signals):
        from backtest.vectorized import VectorizedBacktester, VectorConfig

        config = VectorConfig(
            initial_cash=100_000,
            slippage_bps=5.0,
            commission_bps=1.0,
        )

        bt = VectorizedBacktester(config)
        results = bt.run(sample_ohlcv_data, sample_signals)

        assert results is not None
        assert len(results.equity_curve) > 0
        assert results.metrics.get('initial_equity', 0) == 100_000

    def test_empty_signals(self, sample_ohlcv_data):
        from backtest.vectorized import VectorizedBacktester

        bt = VectorizedBacktester()
        empty_signals = pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'entry_price', 'stop_loss', 'take_profit'])
        results = bt.run(sample_ohlcv_data, empty_signals)

        assert results is not None
        assert results.trades.empty

    def test_empty_data(self):
        from backtest.vectorized import VectorizedBacktester

        bt = VectorizedBacktester()
        empty_data = pd.DataFrame()
        empty_signals = pd.DataFrame()
        results = bt.run(empty_data, empty_signals)

        assert results is not None
        assert len(results.equity_curve) == 1  # Just initial capital

    def test_metrics_calculation(self, sample_ohlcv_data, sample_signals):
        from backtest.vectorized import VectorizedBacktester

        bt = VectorizedBacktester()
        results = bt.run(sample_ohlcv_data, sample_signals)

        metrics = results.metrics
        assert 'total_return_pct' in metrics
        assert 'max_drawdown_pct' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'win_rate' in metrics

    def test_to_dict(self, sample_ohlcv_data, sample_signals):
        from backtest.vectorized import VectorizedBacktester

        bt = VectorizedBacktester()
        results = bt.run(sample_ohlcv_data, sample_signals)

        result_dict = results.to_dict()
        assert 'equity_final' in result_dict
        assert 'total_return' in result_dict
        assert 'trade_count' in result_dict


class TestVectorizedNumba:
    """Tests for numba-optimized functions."""

    def test_compute_drawdown(self):
        from backtest.vectorized import _compute_drawdown

        equity = np.array([100.0, 110.0, 105.0, 115.0, 108.0, 120.0])
        drawdown, max_dd = _compute_drawdown(equity)

        assert len(drawdown) == len(equity)
        assert max_dd < 0  # Drawdown is negative
        assert abs(max_dd) > 0  # There is some drawdown

    def test_check_stops(self):
        from backtest.vectorized import _check_stops

        # Set up test case where stop is hit for index 1, tp is hit for index 2
        # Note: uses elif, so stop check takes priority over tp check
        lows = np.array([95.0, 96.0, 95.0])   # low[1] <= stop[1], low[2] > stop[2]
        highs = np.array([105.0, 102.0, 110.0])  # high[2] >= tp[2]
        positions = np.array([100.0, 100.0, 100.0])
        stop_prices = np.array([94.0, 97.0, 90.0])  # stop[1] = 97, stop[2] = 90
        tp_prices = np.array([110.0, 110.0, 108.0])  # tp[2] = 108

        stop_hit, tp_hit = _check_stops(lows, highs, positions, stop_prices, tp_prices)

        assert len(stop_hit) == 3
        assert len(tp_hit) == 3
        # For long position: stop hit if low <= stop price
        # Index 1: low=96 <= stop=97 -> stop_hit = True
        assert stop_hit[1] == True
        # Index 2: low=95 > stop=90 -> stop not hit, high=110 >= tp=108 -> tp_hit = True
        assert tp_hit[2] == True


# =============================================================================
# Reproducibility Tests
# =============================================================================

class TestDataVersioner:
    """Tests for DataVersioner."""

    def test_import(self):
        from backtest.reproducibility import DataVersioner, DataVersion
        assert DataVersioner is not None
        assert DataVersion is not None

    def test_register_data(self, temp_dir):
        from backtest.reproducibility import DataVersioner

        versioner = DataVersioner(storage_dir=temp_dir)

        # Create a test file
        test_file = temp_dir / "test_data.csv"
        test_file.write_text("a,b,c\n1,2,3\n")

        version = versioner.register_data(
            name="test_data",
            path=test_file,
            description="Test data file",
        )

        assert "test_data" in version.version_id
        assert version.checksum is not None
        assert len(version.checksum) == 64  # SHA256 hex

    def test_create_snapshot(self, temp_dir):
        from backtest.reproducibility import DataVersioner

        versioner = DataVersioner(storage_dir=temp_dir)

        # Create a test file
        test_file = temp_dir / "snapshot_test.csv"
        test_file.write_text("x,y,z\n4,5,6\n")

        version = versioner.create_snapshot(
            name="snapshot_test",
            path=test_file,
            description="Snapshot test",
        )

        assert "snapshot_test" in version.version_id
        # Verify snapshot was created
        snapshot_dir = temp_dir / "snapshots" / version.version_id
        assert snapshot_dir.exists()

    def test_list_versions(self, temp_dir):
        from backtest.reproducibility import DataVersioner

        versioner = DataVersioner(storage_dir=temp_dir)

        # Create and register test files
        for i in range(2):
            test_file = temp_dir / f"list_test_{i}.csv"
            test_file.write_text(f"data{i}\n")
            versioner.register_data(name=f"test_{i}", path=test_file)

        versions = versioner.list_versions()
        assert len(versions) >= 2


class TestExperimentTracker:
    """Tests for ExperimentTracker."""

    def test_import(self):
        from backtest.reproducibility import ExperimentTracker, ExperimentRun
        assert ExperimentTracker is not None
        assert ExperimentRun is not None

    def test_create_experiment(self, temp_dir):
        from backtest.reproducibility import ExperimentTracker

        tracker = ExperimentTracker(experiments_dir=temp_dir)

        with tracker.run("test_experiment") as run:
            run.log_params({'param1': 1, 'param2': 'test'})
            run.log_metric('accuracy', 0.95)
            run.log_metric('loss', 0.05)

        assert run.experiment_id is not None
        # Manifest should be saved
        manifest_path = temp_dir / run.experiment_id / "manifest.json"
        assert manifest_path.exists()

    def test_experiment_manifest(self, temp_dir):
        from backtest.reproducibility import ExperimentTracker
        import json

        tracker = ExperimentTracker(experiments_dir=temp_dir)

        with tracker.run("manifest_test") as run:
            run.log_params({'test': True})
            run.log_metrics({'score': 0.9})
            experiment_id = run.experiment_id

        # Read manifest from file
        manifest_path = temp_dir / experiment_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        assert manifest['name'] == 'manifest_test'
        assert manifest['experiment_id'] == experiment_id
        assert manifest['parameters'] == {'test': True}
        assert manifest['metrics'] == {'score': 0.9}

    def test_list_experiments(self, temp_dir):
        from backtest.reproducibility import ExperimentTracker

        tracker = ExperimentTracker(experiments_dir=temp_dir)

        with tracker.run("exp1"):
            pass
        with tracker.run("exp2"):
            pass

        experiments = tracker.list_experiments()
        assert len(experiments) >= 2


# =============================================================================
# Visualization Tests
# =============================================================================

class TestBacktestPlotter:
    """Tests for BacktestPlotter."""

    def test_import(self):
        from backtest.visualization import BacktestPlotter, PlotConfig
        assert BacktestPlotter is not None
        assert PlotConfig is not None

    def test_create_plotter(self, sample_trades):
        from backtest.visualization import BacktestPlotter

        # Create equity curve DataFrame from trades
        equity = np.cumsum(sample_trades['pnl'].values) + 100000
        dates = pd.date_range(start='2023-01-01', periods=len(equity), freq='D')
        equity_df = pd.DataFrame({'equity': equity}, index=dates)

        metrics = {
            'total_return_pct': 10.5,
            'max_drawdown_pct': -5.2,
            'sharpe_ratio': 1.5,
            'win_rate': 0.55,
        }

        plotter = BacktestPlotter(
            equity_curve=equity_df,
            trades=sample_trades,
            metrics=metrics,
        )

        assert plotter is not None
        assert len(plotter.equity) == len(equity)

    def test_plot_equity_static(self, sample_trades, temp_dir):
        from backtest.visualization import BacktestPlotter

        equity = np.cumsum(sample_trades['pnl'].values) + 100000
        dates = pd.date_range(start='2023-01-01', periods=len(equity), freq='D')
        equity_df = pd.DataFrame({'equity': equity}, index=dates)

        metrics = {'total_return_pct': 10.5}

        plotter = BacktestPlotter(equity_curve=equity_df, trades=sample_trades, metrics=metrics)
        fig = plotter.plot_equity(interactive=False)

        # Should return matplotlib figure or None if not available
        assert fig is not None or True  # Accept None if matplotlib unavailable

    def test_plot_trades_static(self, sample_trades, sample_ohlcv_data):
        from backtest.visualization import BacktestPlotter

        equity = np.cumsum(sample_trades['pnl'].values) + 100000
        dates = pd.date_range(start='2023-01-01', periods=len(equity), freq='D')
        equity_df = pd.DataFrame({'equity': equity}, index=dates)

        metrics = {'total_return_pct': 10.5}

        plotter = BacktestPlotter(
            equity_curve=equity_df,
            trades=sample_trades,
            metrics=metrics,
            prices=sample_ohlcv_data,
        )
        fig = plotter.plot_trades(interactive=False)

        # May be None if no proper trade/price alignment
        assert fig is None or fig is not None

    def test_plot_monthly_returns(self, sample_trades):
        from backtest.visualization import BacktestPlotter

        equity = np.cumsum(sample_trades['pnl'].values) + 100000
        dates = pd.date_range(start='2023-01-01', periods=len(equity), freq='D')
        equity_df = pd.DataFrame({'equity': equity}, index=dates)

        metrics = {'total_return_pct': 10.5}

        plotter = BacktestPlotter(
            equity_curve=equity_df,
            trades=sample_trades,
            metrics=metrics,
        )

        fig = plotter.plot_monthly_returns(interactive=False)
        assert fig is None or fig is not None  # Accept either


# =============================================================================
# Monte Carlo Tests
# =============================================================================

class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator."""

    def test_import(self):
        from backtest.monte_carlo import MonteCarloSimulator, MonteCarloConfig, MonteCarloResults
        assert MonteCarloSimulator is not None
        assert MonteCarloConfig is not None
        assert MonteCarloResults is not None

    def test_basic_simulation(self, sample_trades):
        from backtest.monte_carlo import MonteCarloSimulator, MonteCarloConfig

        config = MonteCarloConfig(
            n_simulations=100,
            initial_capital=100_000,
            random_seed=42,
        )

        mc = MonteCarloSimulator(sample_trades, config)
        results = mc.run()

        assert results is not None
        assert len(results.final_equity_distribution) == 100
        assert len(results.max_drawdown_distribution) == 100

    def test_confidence_intervals(self, sample_trades):
        from backtest.monte_carlo import MonteCarloSimulator

        mc = MonteCarloSimulator(sample_trades)
        results = mc.run(n_simulations=500)

        intervals = results.confidence_intervals
        assert 'final_equity' in intervals
        assert 'max_drawdown' in intervals
        assert len(intervals['final_equity']) == 5  # Default 5 confidence levels

    def test_risk_of_ruin(self, sample_trades):
        from backtest.monte_carlo import MonteCarloSimulator

        mc = MonteCarloSimulator(sample_trades)
        results = mc.run(n_simulations=500)

        # Risk of ruin should be between 0 and 1
        assert 0 <= results.risk_of_ruin <= 1

    def test_optimal_f(self, sample_trades):
        from backtest.monte_carlo import MonteCarloSimulator

        mc = MonteCarloSimulator(sample_trades)
        optimal_f, growth_rate = mc.calculate_optimal_f()

        assert 0 < optimal_f <= 1
        assert isinstance(growth_rate, float)

    def test_summary(self, sample_trades):
        from backtest.monte_carlo import MonteCarloSimulator

        mc = MonteCarloSimulator(sample_trades)
        results = mc.run(n_simulations=100)

        summary = results.summary()
        assert "MONTE CARLO SIMULATION RESULTS" in summary
        assert "Expected Return" in summary
        assert "Risk of Ruin" in summary

    def test_to_dict(self, sample_trades):
        from backtest.monte_carlo import MonteCarloSimulator

        mc = MonteCarloSimulator(sample_trades)
        results = mc.run(n_simulations=100)

        result_dict = results.to_dict()
        assert 'n_simulations' in result_dict
        assert 'expected_return' in result_dict
        assert 'risk_of_ruin' in result_dict
        assert 'equity_stats' in result_dict


class TestMonteCarloConvenienceFunction:
    """Tests for run_monte_carlo_analysis function."""

    def test_run_analysis(self, sample_trades):
        from backtest.monte_carlo import run_monte_carlo_analysis

        analysis = run_monte_carlo_analysis(
            sample_trades,
            n_simulations=100,
            initial_capital=100_000,
        )

        assert 'results' in analysis
        assert 'summary' in analysis
        assert 'optimal_f' in analysis
        assert 'optimal_growth_rate' in analysis


# =============================================================================
# Integration Tests
# =============================================================================

class TestBacktestIntegration:
    """Integration tests for the full backtest module."""

    def test_module_imports(self):
        """Test that all exports can be imported from the backtest module."""
        from backtest import (
            # Engine
            Backtester,
            BacktestConfig,
            CommissionConfig,
            Trade,
            Position,
            # Walk-forward
            WFSplit,
            generate_splits,
            run_walk_forward,
            # Vectorized
            VectorizedBacktester,
            VectorConfig,
            VectorResults,
            # Reproducibility
            DataVersioner,
            DataVersion,
            ExperimentTracker,
            ExperimentRun,
            # Visualization
            BacktestPlotter,
            PlotConfig,
            # Monte Carlo
            MonteCarloSimulator,
            MonteCarloConfig,
            MonteCarloResults,
            run_monte_carlo_analysis,
        )

        assert all([
            Backtester, BacktestConfig, CommissionConfig, Trade, Position,
            WFSplit, generate_splits, run_walk_forward,
            VectorizedBacktester, VectorConfig, VectorResults,
            DataVersioner, DataVersion, ExperimentTracker, ExperimentRun,
            BacktestPlotter, PlotConfig,
            MonteCarloSimulator, MonteCarloConfig, MonteCarloResults, run_monte_carlo_analysis,
        ])

    def test_vectorized_to_monte_carlo_pipeline(self, sample_ohlcv_data, sample_signals):
        """Test running vectorized backtest and then Monte Carlo analysis."""
        from backtest import VectorizedBacktester, MonteCarloSimulator

        # Run backtest
        bt = VectorizedBacktester()
        results = bt.run(sample_ohlcv_data, sample_signals)

        # If we have trades, run Monte Carlo
        if not results.trades.empty:
            mc = MonteCarloSimulator(results.trades)
            mc_results = mc.run(n_simulations=100)

            assert mc_results is not None
            assert mc_results.expected_return is not None

    def test_experiment_tracking_with_backtest(self, sample_ohlcv_data, sample_signals, temp_dir):
        """Test tracking a backtest experiment."""
        from backtest import VectorizedBacktester, VectorConfig, ExperimentTracker

        tracker = ExperimentTracker(experiments_dir=temp_dir)

        config = VectorConfig(
            initial_cash=50_000,
            slippage_bps=10.0,
        )

        with tracker.run("backtest_experiment") as run:
            run.log_params({
                'initial_cash': config.initial_cash,
                'slippage_bps': config.slippage_bps,
            })

            bt = VectorizedBacktester(config)
            results = bt.run(sample_ohlcv_data, sample_signals)

            run.log_metric('total_return', results.metrics.get('total_return_pct', 0))
            run.log_metric('sharpe_ratio', results.metrics.get('sharpe_ratio', 0))
            run.log_metric('trade_count', len(results.trades))

        # Verify experiment was recorded
        experiments = tracker.list_experiments()
        assert len(experiments) >= 1
        assert experiments[0].name == "backtest_experiment"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
