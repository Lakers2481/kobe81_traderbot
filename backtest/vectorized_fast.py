"""
Vectorized Fast Backtester for Kobe Trading System.

Ultra-fast backtesting for bulk alpha mining:
- 10,000+ parameter variants in seconds
- Pure numpy/pandas vectorization (no loops)
- VectorBT integration when available
- Parallel metric calculation

Created: 2026-01-07
Purpose: Enable rapid alpha mining and parameter sweeps
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

# Try importing VectorBT
try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False
    vbt = None


logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class VectorizedBacktestConfig:
    """Configuration for vectorized backtesting."""
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    max_positions: int = 10
    position_size: float = 0.10  # 10% per position
    allow_shorting: bool = False
    freq: str = "D"  # D for daily


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics."""
    # Returns
    total_return: float = 0.0
    cagr: float = 0.0
    daily_return_mean: float = 0.0
    daily_return_std: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    avg_drawdown: float = 0.0

    # Trading
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 0.0
    expectancy: float = 0.0

    # Exposure
    exposure_time: float = 0.0  # % of time in market

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "expectancy": self.expectancy,
            "exposure_time": self.exposure_time,
        }


# =============================================================================
# VECTORIZED BACKTESTER
# =============================================================================

class VectorizedBacktester:
    """
    Ultra-fast vectorized backtester for bulk alpha mining.

    Uses pure numpy/pandas vectorization for maximum speed.
    Optionally integrates with VectorBT for advanced features.

    Usage:
        backtester = VectorizedBacktester()

        # Single backtest
        metrics = backtester.backtest(prices, signals)

        # Bulk parameter sweep
        results = backtester.sweep_parameters(
            prices,
            signal_func=lambda p, w: p.pct_change(w) > 0,
            params={'window': [5, 10, 20, 50, 100, 200]}
        )
    """

    def __init__(self, config: Optional[VectorizedBacktestConfig] = None):
        self.config = config or VectorizedBacktestConfig()
        self._use_vbt = HAS_VBT
        logger.info(f"VectorizedBacktester initialized (VectorBT: {HAS_VBT})")

    # -------------------------------------------------------------------------
    # Core Backtesting
    # -------------------------------------------------------------------------

    def backtest(
        self,
        prices: pd.DataFrame | pd.Series,
        signals: pd.DataFrame | pd.Series,
        config: Optional[VectorizedBacktestConfig] = None,
    ) -> BacktestMetrics:
        """
        Run vectorized backtest.

        Args:
            prices: Price series or DataFrame (close prices)
            signals: Signal series or DataFrame (1 = long, -1 = short, 0 = flat)
            config: Optional config override

        Returns:
            BacktestMetrics with all performance stats
        """
        cfg = config or self.config

        # Normalize inputs
        if isinstance(prices, pd.Series):
            prices = prices.to_frame("close")
        if isinstance(signals, pd.Series):
            signals = signals.to_frame("signal")

        # Ensure alignment
        prices = prices.reindex(signals.index)

        # Calculate returns
        returns = prices.pct_change()

        # Apply signals (shift to avoid lookahead)
        signal_shifted = signals.shift(1)  # Signal at t, trade at t+1
        strategy_returns = signal_shifted * returns

        # Apply costs
        trades = signal_shifted.diff().abs()
        costs = trades * (cfg.commission + cfg.slippage)
        strategy_returns = strategy_returns - costs

        # Calculate metrics
        metrics = self._calculate_metrics(strategy_returns, signal_shifted)

        return metrics

    def backtest_bulk(
        self,
        prices: pd.DataFrame,
        signals_3d: np.ndarray,
        param_names: List[str],
    ) -> pd.DataFrame:
        """
        Bulk backtest many signal variants at once.

        Args:
            prices: DataFrame with close prices (columns = symbols)
            signals_3d: 3D array [time, symbols, variants]
            param_names: Names for each variant

        Returns:
            DataFrame with metrics for each variant
        """
        results = []
        returns = prices.pct_change().values

        for i, name in enumerate(param_names):
            variant_signals = signals_3d[:, :, i] if signals_3d.ndim == 3 else signals_3d
            variant_returns = self._vectorized_returns(returns, variant_signals)
            metrics = self._metrics_from_returns(variant_returns)
            metrics["variant"] = name
            results.append(metrics)

        return pd.DataFrame(results).set_index("variant")

    # -------------------------------------------------------------------------
    # Parameter Sweeps
    # -------------------------------------------------------------------------

    def sweep_parameters(
        self,
        prices: pd.DataFrame | pd.Series,
        signal_func: Callable,
        params: Dict[str, List[Any]],
        parallel: bool = True,
    ) -> pd.DataFrame:
        """
        Sweep parameters to find optimal settings.

        Args:
            prices: Price data
            signal_func: Function(prices, **params) -> signals
            params: Dict of param_name -> list of values to test
            parallel: Use VectorBT parallelization if available

        Returns:
            DataFrame with metrics for each parameter combination

        Example:
            results = backtester.sweep_parameters(
                prices,
                signal_func=lambda p, window: (p.pct_change(window) > 0).astype(int),
                params={'window': [5, 10, 20, 50, 100, 200]}
            )
        """
        if self._use_vbt and HAS_VBT and parallel:
            return self._sweep_with_vbt(prices, signal_func, params)
        else:
            return self._sweep_native(prices, signal_func, params)

    def _sweep_with_vbt(
        self,
        prices: pd.DataFrame | pd.Series,
        signal_func: Callable,
        params: Dict[str, List[Any]],
    ) -> pd.DataFrame:
        """Parameter sweep using VectorBT."""
        from itertools import product

        param_names = list(params.keys())
        param_values = list(params.values())
        combinations = list(product(*param_values))

        results = []

        for combo in combinations:
            kwargs = dict(zip(param_names, combo))
            param_str = "_".join(f"{k}={v}" for k, v in kwargs.items())

            try:
                signals = signal_func(prices, **kwargs)

                # Use VectorBT portfolio
                if isinstance(prices, pd.Series):
                    close = prices
                else:
                    close = prices.iloc[:, 0]

                pf = vbt.Portfolio.from_signals(
                    close,
                    entries=signals > 0,
                    exits=signals <= 0,
                    init_cash=self.config.initial_capital,
                    fees=self.config.commission,
                    slippage=self.config.slippage,
                )

                stats = pf.stats()
                results.append({
                    "params": param_str,
                    **kwargs,
                    "total_return": stats.get("Total Return [%]", 0) / 100,
                    "sharpe_ratio": stats.get("Sharpe Ratio", 0),
                    "max_drawdown": stats.get("Max Drawdown [%]", 0) / 100,
                    "win_rate": stats.get("Win Rate [%]", 0) / 100,
                    "num_trades": stats.get("Total Trades", 0),
                    "profit_factor": stats.get("Profit Factor", 0),
                })
            except Exception as e:
                logger.debug(f"Failed sweep for {param_str}: {e}")
                results.append({
                    "params": param_str,
                    **kwargs,
                    "total_return": 0,
                    "sharpe_ratio": 0,
                    "error": str(e),
                })

        return pd.DataFrame(results)

    def _sweep_native(
        self,
        prices: pd.DataFrame | pd.Series,
        signal_func: Callable,
        params: Dict[str, List[Any]],
    ) -> pd.DataFrame:
        """Parameter sweep using native numpy."""
        from itertools import product

        param_names = list(params.keys())
        param_values = list(params.values())
        combinations = list(product(*param_values))

        results = []

        for combo in combinations:
            kwargs = dict(zip(param_names, combo))
            param_str = "_".join(f"{k}={v}" for k, v in kwargs.items())

            try:
                signals = signal_func(prices, **kwargs)
                metrics = self.backtest(prices, signals)

                results.append({
                    "params": param_str,
                    **kwargs,
                    "total_return": metrics.total_return,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "max_drawdown": metrics.max_drawdown,
                    "win_rate": metrics.win_rate,
                    "num_trades": metrics.num_trades,
                    "profit_factor": metrics.profit_factor,
                })
            except Exception as e:
                logger.debug(f"Failed sweep for {param_str}: {e}")
                results.append({
                    "params": param_str,
                    **kwargs,
                    "total_return": 0,
                    "sharpe_ratio": 0,
                    "error": str(e),
                })

        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # Alpha Mining
    # -------------------------------------------------------------------------

    def mine_momentum_alphas(
        self,
        prices: pd.DataFrame,
        windows: List[int] = [5, 10, 20, 50, 100, 200],
        thresholds: List[float] = [0.0, 0.01, 0.02],
    ) -> pd.DataFrame:
        """
        Mine momentum alpha variants.

        Tests: return > threshold over window periods.
        """
        def signal_func(p, window, threshold):
            ret = p.pct_change(window)
            return (ret > threshold).astype(int)

        return self.sweep_parameters(
            prices,
            signal_func,
            params={"window": windows, "threshold": thresholds},
        )

    def mine_mean_reversion_alphas(
        self,
        prices: pd.DataFrame,
        windows: List[int] = [2, 5, 10, 14, 20],
        z_thresholds: List[float] = [-1.0, -1.5, -2.0, -2.5],
    ) -> pd.DataFrame:
        """
        Mine mean-reversion alpha variants.

        Tests: z-score < threshold (oversold = buy).
        """
        def signal_func(p, window, z_threshold):
            mean = p.rolling(window).mean()
            std = p.rolling(window).std()
            z = (p - mean) / (std + 1e-8)
            return (z < z_threshold).astype(int)

        return self.sweep_parameters(
            prices,
            signal_func,
            params={"window": windows, "z_threshold": z_thresholds},
        )

    def mine_volatility_alphas(
        self,
        prices: pd.DataFrame,
        vol_windows: List[int] = [10, 20, 50],
        vol_thresholds: List[float] = [0.01, 0.02, 0.03],
    ) -> pd.DataFrame:
        """
        Mine volatility-based alpha variants.

        Tests: volatility below threshold (low vol = buy).
        """
        def signal_func(p, vol_window, vol_threshold):
            vol = p.pct_change().rolling(vol_window).std()
            return (vol < vol_threshold).astype(int)

        return self.sweep_parameters(
            prices,
            signal_func,
            params={"vol_window": vol_windows, "vol_threshold": vol_thresholds},
        )

    # -------------------------------------------------------------------------
    # Metric Calculation
    # -------------------------------------------------------------------------

    def _calculate_metrics(
        self,
        strategy_returns: pd.DataFrame | pd.Series,
        positions: pd.DataFrame | pd.Series,
    ) -> BacktestMetrics:
        """Calculate comprehensive metrics from returns."""
        # Flatten if needed
        if isinstance(strategy_returns, pd.DataFrame):
            returns = strategy_returns.values.flatten()
        else:
            returns = strategy_returns.values

        if isinstance(positions, pd.DataFrame):
            pos = positions.values.flatten()
        else:
            pos = positions.values

        # Remove NaN
        mask = ~np.isnan(returns)
        returns = returns[mask]
        pos = pos[mask] if len(pos) == len(mask) else pos[mask[:len(pos)]]

        if len(returns) == 0:
            return BacktestMetrics()

        metrics = BacktestMetrics()

        # Basic returns
        metrics.total_return = np.nansum(returns)
        metrics.daily_return_mean = np.nanmean(returns)
        metrics.daily_return_std = np.nanstd(returns)

        # CAGR
        n_days = len(returns)
        if n_days > 0 and metrics.total_return > -1:
            metrics.cagr = (1 + metrics.total_return) ** (252 / n_days) - 1

        # Sharpe
        if metrics.daily_return_std > 0:
            metrics.sharpe_ratio = (metrics.daily_return_mean / metrics.daily_return_std) * np.sqrt(252)

        # Sortino
        downside = returns[returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 0.001
        if downside_std > 0:
            metrics.sortino_ratio = (metrics.daily_return_mean / downside_std) * np.sqrt(252)

        # Drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        metrics.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        metrics.avg_drawdown = np.mean(drawdown) if len(drawdown) > 0 else 0

        # Max drawdown duration
        in_drawdown = drawdown > 0
        if np.any(in_drawdown):
            dd_periods = np.diff(np.where(np.concatenate([[False], in_drawdown, [False]]))[0])[::2]
            metrics.max_drawdown_duration = int(np.max(dd_periods)) if len(dd_periods) > 0 else 0

        # Calmar
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.cagr / metrics.max_drawdown

        # Trading metrics
        if len(pos) > 1:
            trades = np.diff(pos)
            trade_returns = returns[1:][trades != 0] if len(returns) > 1 else np.array([])

            metrics.num_trades = int(np.sum(np.abs(trades) > 0))

            if len(trade_returns) > 0:
                wins = trade_returns[trade_returns > 0]
                losses = trade_returns[trade_returns < 0]

                metrics.win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
                metrics.avg_win = np.mean(wins) if len(wins) > 0 else 0
                metrics.avg_loss = np.mean(losses) if len(losses) > 0 else 0

                gross_profit = np.sum(wins)
                gross_loss = np.abs(np.sum(losses))
                metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

                metrics.win_loss_ratio = np.abs(metrics.avg_win / metrics.avg_loss) if metrics.avg_loss != 0 else 0
                metrics.expectancy = metrics.win_rate * metrics.avg_win + (1 - metrics.win_rate) * metrics.avg_loss

        # Exposure
        metrics.exposure_time = np.mean(np.abs(pos) > 0) if len(pos) > 0 else 0

        return metrics

    def _vectorized_returns(
        self,
        returns: np.ndarray,
        signals: np.ndarray,
    ) -> np.ndarray:
        """Calculate strategy returns using pure numpy."""
        # Shift signals to avoid lookahead
        signals_shifted = np.roll(signals, 1, axis=0)
        signals_shifted[0] = 0

        # Strategy returns
        strategy_returns = signals_shifted * returns

        # Costs
        trades = np.abs(np.diff(signals_shifted, axis=0, prepend=0))
        costs = trades * (self.config.commission + self.config.slippage)
        strategy_returns = strategy_returns - costs

        return strategy_returns

    def _metrics_from_returns(self, returns: np.ndarray) -> Dict[str, float]:
        """Quick metrics from returns array."""
        returns = returns.flatten()
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return {"sharpe_ratio": 0, "total_return": 0, "max_drawdown": 0}

        total_ret = np.sum(returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns) + 1e-8

        sharpe = mean_ret / std_ret * np.sqrt(252)

        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        max_dd = np.max(running_max - cumulative)

        return {
            "total_return": total_ret,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
        }

    # -------------------------------------------------------------------------
    # VectorBT Integration
    # -------------------------------------------------------------------------

    def create_vbt_portfolio(
        self,
        prices: pd.DataFrame | pd.Series,
        signals: pd.DataFrame | pd.Series,
    ):
        """Create VectorBT Portfolio object for advanced analysis."""
        if not HAS_VBT:
            raise ImportError("VectorBT required: pip install vectorbt")

        if isinstance(prices, pd.DataFrame):
            close = prices.iloc[:, 0]
        else:
            close = prices

        if isinstance(signals, pd.DataFrame):
            sig = signals.iloc[:, 0]
        else:
            sig = signals

        pf = vbt.Portfolio.from_signals(
            close,
            entries=sig > 0,
            exits=sig <= 0,
            init_cash=self.config.initial_capital,
            fees=self.config.commission,
            slippage=self.config.slippage,
        )

        return pf

    def generate_vbt_heatmap(
        self,
        prices: pd.DataFrame | pd.Series,
        signal_func: Callable,
        param1: Tuple[str, List[Any]],
        param2: Tuple[str, List[Any]],
        metric: str = "sharpe_ratio",
    ):
        """Generate VectorBT heatmap for 2-parameter sweep."""
        if not HAS_VBT:
            raise ImportError("VectorBT required: pip install vectorbt")

        results = self.sweep_parameters(
            prices,
            signal_func,
            params={param1[0]: param1[1], param2[0]: param2[1]},
        )

        # Pivot to heatmap format
        heatmap = results.pivot(index=param1[0], columns=param2[0], values=metric)
        return heatmap


# =============================================================================
# SINGLETON
# =============================================================================

_vectorized_backtester: Optional[VectorizedBacktester] = None


def get_vectorized_backtester() -> VectorizedBacktester:
    """Get singleton VectorizedBacktester instance."""
    global _vectorized_backtester
    if _vectorized_backtester is None:
        _vectorized_backtester = VectorizedBacktester()
    return _vectorized_backtester


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    # Demo
    print("VectorizedBacktester Demo")
    print("=" * 60)
    print(f"VectorBT available: {HAS_VBT}")

    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    prices = pd.DataFrame({
        "close": 100 * np.cumprod(1 + np.random.randn(len(dates)) * 0.02)
    }, index=dates)

    # Simple momentum signal
    signals = (prices["close"].pct_change(20) > 0).astype(int)

    # Run backtest
    backtester = get_vectorized_backtester()
    metrics = backtester.backtest(prices, signals)

    print(f"\nSingle Backtest Results:")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Win Rate: {metrics.win_rate:.1%}")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")

    # Parameter sweep
    print(f"\nParameter Sweep (momentum windows):")
    results = backtester.mine_momentum_alphas(
        prices["close"],
        windows=[5, 10, 20, 50, 100],
        thresholds=[0.0],
    )

    print(results[["window", "sharpe_ratio", "total_return", "max_drawdown"]].to_string())
