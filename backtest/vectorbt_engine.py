"""
VectorBT Fast Backtesting Engine.

Provides 10-100x faster backtesting using vectorized operations.
VectorBT is optimized for portfolio-level backtesting with:
- Fully vectorized calculations (no loops)
- Multi-asset support
- Advanced order types
- Built-in performance metrics
- Memory-efficient operations

Source: https://vectorbt.dev/

Usage:
    from backtest.vectorbt_engine import VectorBTBacktester

    backtester = VectorBTBacktester()
    results = backtester.run(df, signals)
    stats = backtester.get_stats()
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Dict, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.structured_log import jlog

# Check for vectorbt availability
try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    jlog("vectorbt_not_available", level="INFO",
         message="Install vectorbt: pip install vectorbt")


@dataclass
class VectorBTConfig:
    """Configuration for VectorBT backtesting."""

    # Portfolio settings
    init_cash: float = 100_000.0
    fees: float = 0.001  # 0.1% per trade
    slippage: float = 0.001  # 0.1% slippage
    size_type: str = "percent"  # "percent", "amount", "value"
    size: float = 0.02  # 2% of portfolio per trade

    # Risk management
    sl_stop: Optional[float] = None  # Stop loss (e.g., 0.02 = 2%)
    tp_stop: Optional[float] = None  # Take profit
    time_stop: Optional[int] = None  # Time stop in bars

    # Execution
    accumulate: bool = False  # Allow position accumulation
    short_enable: bool = False  # Enable short selling
    freq: str = "D"  # Data frequency ("D", "H", "M", etc.)

    # Performance
    use_numba: bool = True  # Use Numba acceleration
    seed: int = 42


class VectorBTBacktester:
    """
    Fast vectorized backtester using VectorBT.

    Provides 10-100x speedup over event-driven backtesting
    through fully vectorized operations.

    Features:
    - Multi-asset portfolio backtesting
    - Built-in slippage and commission modeling
    - Stop loss and take profit orders
    - Comprehensive performance metrics
    - Trade-level analysis
    """

    def __init__(self, config: Optional[VectorBTConfig] = None):
        self.config = config or VectorBTConfig()
        self.portfolio = None
        self._results: Dict[str, Any] = {}

    def run(
        self,
        close: pd.DataFrame,
        entries: pd.DataFrame,
        exits: Optional[pd.DataFrame] = None,
        size: Optional[pd.DataFrame] = None
    ) -> 'VectorBTBacktester':
        """
        Run vectorized backtest.

        Args:
            close: Close prices (DatetimeIndex, columns = symbols)
            entries: Entry signals (True/False or 1/0)
            exits: Exit signals (optional, uses stops if not provided)
            size: Position size per trade (optional)

        Returns:
            Self with results populated
        """
        if not VBT_AVAILABLE:
            jlog("vbt_run_skipped", level="WARNING",
                 message="vectorbt not installed")
            return self

        jlog("vbt_backtest_starting", level="INFO",
             symbols=close.shape[1] if close.ndim > 1 else 1,
             bars=len(close))

        # Convert to proper format
        close = self._ensure_dataframe(close)
        entries = self._ensure_dataframe(entries).astype(bool)

        if exits is not None:
            exits = self._ensure_dataframe(exits).astype(bool)

        # Build portfolio
        if self.config.sl_stop is not None or self.config.tp_stop is not None:
            # Use stops
            self.portfolio = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                size=size if size is not None else self.config.size,
                size_type=self.config.size_type,
                init_cash=self.config.init_cash,
                fees=self.config.fees,
                slippage=self.config.slippage,
                sl_stop=self.config.sl_stop,
                tp_stop=self.config.tp_stop,
                accumulate=self.config.accumulate,
                freq=self.config.freq
            )
        else:
            # Simple signals
            self.portfolio = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                size=size if size is not None else self.config.size,
                size_type=self.config.size_type,
                init_cash=self.config.init_cash,
                fees=self.config.fees,
                slippage=self.config.slippage,
                accumulate=self.config.accumulate,
                freq=self.config.freq
            )

        # Calculate key metrics
        self._calculate_results()

        jlog("vbt_backtest_complete", level="INFO",
             total_return=self._results.get('total_return', 0),
             sharpe=self._results.get('sharpe_ratio', 0),
             n_trades=self._results.get('n_trades', 0))

        return self

    def _ensure_dataframe(self, data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Ensure data is a DataFrame."""
        if isinstance(data, pd.Series):
            return data.to_frame()
        return data

    def _calculate_results(self) -> None:
        """Calculate and cache performance metrics."""
        if self.portfolio is None:
            return

        try:
            # Basic stats
            stats = self.portfolio.stats()

            self._results = {
                'total_return': self.portfolio.total_return(),
                'total_profit': self.portfolio.total_profit(),
                'sharpe_ratio': self.portfolio.sharpe_ratio(),
                'sortino_ratio': self.portfolio.sortino_ratio(),
                'calmar_ratio': self.portfolio.calmar_ratio(),
                'max_drawdown': self.portfolio.max_drawdown(),
                'win_rate': self.portfolio.trades.win_rate() if hasattr(self.portfolio, 'trades') else 0,
                'profit_factor': self.portfolio.trades.profit_factor() if hasattr(self.portfolio, 'trades') else 0,
                'n_trades': self.portfolio.trades.count() if hasattr(self.portfolio, 'trades') else 0,
                'exposure_time': self.portfolio.trades.total_duration() if hasattr(self.portfolio, 'trades') else 0,
                'equity_curve': self.portfolio.value(),
                'returns': self.portfolio.returns(),
                'stats': stats
            }

        except Exception as e:
            jlog("vbt_results_error", level="WARNING", error=str(e))

    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dict with performance metrics
        """
        if not self._results:
            return {}

        return {
            'total_return': self._results.get('total_return'),
            'total_return_pct': self._results.get('total_return', 0) * 100,
            'sharpe_ratio': self._results.get('sharpe_ratio'),
            'sortino_ratio': self._results.get('sortino_ratio'),
            'calmar_ratio': self._results.get('calmar_ratio'),
            'max_drawdown': self._results.get('max_drawdown'),
            'max_drawdown_pct': self._results.get('max_drawdown', 0) * 100,
            'win_rate': self._results.get('win_rate'),
            'win_rate_pct': self._results.get('win_rate', 0) * 100,
            'profit_factor': self._results.get('profit_factor'),
            'n_trades': self._results.get('n_trades'),
            'avg_trade_return': (
                self._results.get('total_return', 0) / max(self._results.get('n_trades', 1), 1)
            )
        }

    def get_equity_curve(self) -> pd.Series:
        """
        Get equity curve.

        Returns:
            Series with portfolio value over time
        """
        return self._results.get('equity_curve', pd.Series())

    def get_returns(self) -> pd.Series:
        """
        Get return series.

        Returns:
            Series with daily returns
        """
        return self._results.get('returns', pd.Series())

    def get_trades(self) -> pd.DataFrame:
        """
        Get trade list.

        Returns:
            DataFrame with trade details
        """
        if self.portfolio is None:
            return pd.DataFrame()

        try:
            trades = self.portfolio.trades.records_readable
            return trades
        except Exception:
            return pd.DataFrame()

    def get_drawdowns(self) -> pd.DataFrame:
        """
        Get drawdown analysis.

        Returns:
            DataFrame with drawdown details
        """
        if self.portfolio is None:
            return pd.DataFrame()

        try:
            drawdowns = self.portfolio.drawdowns.records_readable
            return drawdowns
        except Exception:
            return pd.DataFrame()

    def plot_equity(self, **kwargs) -> Any:
        """
        Plot equity curve.

        Returns:
            Plotly figure
        """
        if self.portfolio is None:
            return None

        return self.portfolio.plot(**kwargs)

    def plot_drawdowns(self, **kwargs) -> Any:
        """
        Plot drawdowns.

        Returns:
            Plotly figure
        """
        if self.portfolio is None:
            return None

        return self.portfolio.drawdowns.plot(**kwargs)


class VectorBTOptimizer:
    """
    Parameter optimization using VectorBT.

    Leverages vectorized operations to test thousands
    of parameter combinations efficiently.
    """

    def __init__(self, config: Optional[VectorBTConfig] = None):
        self.config = config or VectorBTConfig()
        self.results = None
        self.best_params = None

    def optimize(
        self,
        close: pd.DataFrame,
        signal_func: callable,
        param_grid: Dict[str, List[Any]],
        metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters.

        Args:
            close: Close prices
            signal_func: Function(close, **params) -> entries
            param_grid: Parameter grid to search
            metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)

        Returns:
            Dict with best parameters and results
        """
        if not VBT_AVAILABLE:
            jlog("vbt_optimize_skipped", level="WARNING")
            return {}

        jlog("vbt_optimization_starting", level="INFO",
             n_combinations=np.prod([len(v) for v in param_grid.values()]))

        # Generate all parameter combinations
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        best_metric = -np.inf
        best_params = {}
        all_results = []

        for combo in product(*param_values):
            params = dict(zip(param_names, combo))

            try:
                # Generate signals with these parameters
                entries = signal_func(close, **params)

                # Run backtest
                backtester = VectorBTBacktester(self.config)
                backtester.run(close, entries)
                stats = backtester.get_stats()

                metric_value = stats.get(metric, 0)
                all_results.append({**params, metric: metric_value})

                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params.copy()

            except Exception as e:
                jlog("vbt_optimize_combo_error", level="DEBUG",
                     params=params, error=str(e))
                continue

        self.results = pd.DataFrame(all_results)
        self.best_params = best_params

        jlog("vbt_optimization_complete", level="INFO",
             best_params=best_params,
             best_metric=best_metric)

        return {
            'best_params': best_params,
            'best_metric': best_metric,
            'all_results': self.results
        }


class VectorBTMultiAsset:
    """
    Multi-asset portfolio backtesting.

    Handles portfolio-level concerns:
    - Position sizing across assets
    - Correlation-based allocation
    - Rebalancing
    - Risk budgeting
    """

    def __init__(self, config: Optional[VectorBTConfig] = None):
        self.config = config or VectorBTConfig()
        self.portfolio = None

    def run_equal_weight(
        self,
        close: pd.DataFrame,
        entries: pd.DataFrame,
        exits: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run equal-weight portfolio backtest.

        Args:
            close: Close prices (columns = symbols)
            entries: Entry signals
            exits: Exit signals

        Returns:
            Dict with portfolio results
        """
        if not VBT_AVAILABLE:
            return {}

        n_assets = close.shape[1]
        size = 1.0 / n_assets  # Equal allocation

        self.portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            size=size,
            size_type="percent",
            init_cash=self.config.init_cash,
            fees=self.config.fees,
            slippage=self.config.slippage,
            group_by=True,  # Group as single portfolio
            cash_sharing=True,  # Share cash across assets
            freq=self.config.freq
        )

        return self._get_portfolio_stats()

    def run_risk_parity(
        self,
        close: pd.DataFrame,
        entries: pd.DataFrame,
        lookback: int = 60,
        exits: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run risk parity portfolio backtest.

        Allocates inversely proportional to volatility.

        Args:
            close: Close prices
            entries: Entry signals
            lookback: Volatility lookback period
            exits: Exit signals

        Returns:
            Dict with portfolio results
        """
        if not VBT_AVAILABLE:
            return {}

        # Calculate rolling volatility
        returns = close.pct_change()
        volatility = returns.rolling(lookback).std()

        # Inverse volatility weights
        inv_vol = 1 / volatility
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        weights = weights.fillna(1 / close.shape[1])

        self.portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            size=weights,
            size_type="percent",
            init_cash=self.config.init_cash,
            fees=self.config.fees,
            slippage=self.config.slippage,
            group_by=True,
            cash_sharing=True,
            freq=self.config.freq
        )

        return self._get_portfolio_stats()

    def _get_portfolio_stats(self) -> Dict[str, Any]:
        """Get portfolio statistics."""
        if self.portfolio is None:
            return {}

        try:
            return {
                'total_return': self.portfolio.total_return(),
                'sharpe_ratio': self.portfolio.sharpe_ratio(),
                'sortino_ratio': self.portfolio.sortino_ratio(),
                'max_drawdown': self.portfolio.max_drawdown(),
                'calmar_ratio': self.portfolio.calmar_ratio(),
                'equity_curve': self.portfolio.value(),
                'returns': self.portfolio.returns()
            }
        except Exception as e:
            jlog("vbt_portfolio_stats_error", level="WARNING", error=str(e))
            return {}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_backtest(
    close: pd.DataFrame,
    entries: pd.DataFrame,
    exits: Optional[pd.DataFrame] = None,
    init_cash: float = 100_000,
    fees: float = 0.001
) -> Dict[str, Any]:
    """
    Run quick backtest with default settings.

    Args:
        close: Close prices
        entries: Entry signals
        exits: Exit signals
        init_cash: Initial capital
        fees: Transaction fees

    Returns:
        Dict with backtest results
    """
    config = VectorBTConfig(init_cash=init_cash, fees=fees)
    backtester = VectorBTBacktester(config)
    backtester.run(close, entries, exits)
    return backtester.get_stats()


def compare_strategies(
    close: pd.DataFrame,
    strategies: Dict[str, pd.DataFrame],
    init_cash: float = 100_000
) -> pd.DataFrame:
    """
    Compare multiple strategies.

    Args:
        close: Close prices
        strategies: Dict of {strategy_name: entry_signals}
        init_cash: Initial capital

    Returns:
        DataFrame comparing strategy performance
    """
    if not VBT_AVAILABLE:
        return pd.DataFrame()

    results = []
    for name, entries in strategies.items():
        config = VectorBTConfig(init_cash=init_cash)
        backtester = VectorBTBacktester(config)
        backtester.run(close, entries)
        stats = backtester.get_stats()
        stats['strategy'] = name
        results.append(stats)

    return pd.DataFrame(results).set_index('strategy')


def backtest_with_stops(
    close: pd.DataFrame,
    entries: pd.DataFrame,
    sl_pct: float = 0.02,
    tp_pct: float = 0.04,
    init_cash: float = 100_000
) -> Dict[str, Any]:
    """
    Backtest with stop loss and take profit.

    Args:
        close: Close prices
        entries: Entry signals
        sl_pct: Stop loss percentage (e.g., 0.02 = 2%)
        tp_pct: Take profit percentage
        init_cash: Initial capital

    Returns:
        Dict with backtest results
    """
    config = VectorBTConfig(
        init_cash=init_cash,
        sl_stop=sl_pct,
        tp_stop=tp_pct
    )
    backtester = VectorBTBacktester(config)
    backtester.run(close, entries)
    return backtester.get_stats()
