"""
Gate 1: Baseline Beat
=====================

Strategy must outperform simple baselines:
- Buy and Hold (same universe, same period)
- 50/200 MA Crossover
- Naive Mean Reversion

If you can't beat these, you don't have an edge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result from baseline comparison."""
    passed: bool
    beats_buy_hold: bool
    beats_ma_cross: bool
    beats_naive_mr: bool
    strategy_metrics: Dict[str, float]
    baseline_metrics: Dict[str, Dict[str, float]]
    margin_vs_best: float  # How much better than best baseline
    details: Dict[str, Any]


class Gate1Baseline:
    """
    Gate 1: Must beat simple baselines.

    A strategy that can't beat buy-and-hold has no edge.
    A strategy that can't beat MA crossover is too complex.
    A strategy that can't beat naive MR isn't finding real mean reversion.

    FAIL = ARCHIVE
    """

    # Required margin over baselines (in Sharpe)
    MIN_MARGIN_SHARPE = 0.1

    def __init__(self):
        pass

    def validate(
        self,
        strategy_equity: pd.Series,
        benchmark_prices: pd.Series,
        universe_prices: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.02,
    ) -> BaselineResult:
        """
        Compare strategy to baselines.

        Args:
            strategy_equity: Strategy equity curve
            benchmark_prices: Benchmark prices (e.g., SPY)
            universe_prices: Optional universe for MA/MR baselines
            risk_free_rate: Annual risk-free rate

        Returns:
            BaselineResult with comparison
        """
        details = {}

        # Calculate strategy metrics
        strategy_metrics = self._calculate_metrics(strategy_equity, risk_free_rate)
        details["strategy"] = strategy_metrics

        # Calculate buy-and-hold
        bh_equity = benchmark_prices / benchmark_prices.iloc[0] * strategy_equity.iloc[0]
        bh_metrics = self._calculate_metrics(bh_equity, risk_free_rate)
        details["buy_hold"] = bh_metrics

        # Calculate MA crossover baseline
        if universe_prices is not None:
            ma_equity = self._run_ma_crossover(universe_prices, strategy_equity.iloc[0])
            ma_metrics = self._calculate_metrics(ma_equity, risk_free_rate)
        else:
            ma_metrics = {"sharpe": 0.3, "cagr": 0.05}  # Conservative defaults
        details["ma_cross"] = ma_metrics

        # Calculate naive mean reversion baseline
        if universe_prices is not None:
            mr_equity = self._run_naive_mr(universe_prices, strategy_equity.iloc[0])
            mr_metrics = self._calculate_metrics(mr_equity, risk_free_rate)
        else:
            mr_metrics = {"sharpe": 0.25, "cagr": 0.04}  # Conservative defaults
        details["naive_mr"] = mr_metrics

        # Compare
        beats_bh = strategy_metrics["sharpe"] > bh_metrics["sharpe"]
        beats_ma = strategy_metrics["sharpe"] > ma_metrics["sharpe"]
        beats_mr = strategy_metrics["sharpe"] > mr_metrics["sharpe"]

        # Margin over best baseline
        best_baseline = max(bh_metrics["sharpe"], ma_metrics["sharpe"], mr_metrics["sharpe"])
        margin = strategy_metrics["sharpe"] - best_baseline

        passed = beats_bh and beats_ma and beats_mr and margin >= self.MIN_MARGIN_SHARPE

        return BaselineResult(
            passed=passed,
            beats_buy_hold=beats_bh,
            beats_ma_cross=beats_ma,
            beats_naive_mr=beats_mr,
            strategy_metrics=strategy_metrics,
            baseline_metrics={
                "buy_hold": bh_metrics,
                "ma_cross": ma_metrics,
                "naive_mr": mr_metrics,
            },
            margin_vs_best=margin,
            details=details,
        )

    def _calculate_metrics(self, equity: pd.Series, rf_rate: float) -> Dict[str, float]:
        """Calculate performance metrics from equity curve."""
        if len(equity) < 2:
            return {"sharpe": 0.0, "cagr": 0.0, "max_dd": 0.0}

        returns = equity.pct_change().dropna()

        if len(returns) < 20:
            return {"sharpe": 0.0, "cagr": 0.0, "max_dd": 0.0}

        # Sharpe ratio (annualized)
        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)
        sharpe = (mean_return - rf_rate) / std_return if std_return > 0 else 0

        # CAGR
        n_years = len(returns) / 252
        total_return = equity.iloc[-1] / equity.iloc[0]
        cagr = (total_return ** (1 / n_years)) - 1 if n_years > 0 else 0

        # Max drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())

        return {
            "sharpe": float(sharpe),
            "cagr": float(cagr),
            "max_dd": float(max_dd),
        }

    def _run_ma_crossover(self, prices: pd.DataFrame, initial_capital: float) -> pd.Series:
        """
        Run simple 50/200 MA crossover strategy.
        """
        if prices.empty:
            return pd.Series([initial_capital])

        # Use first column or mean if multiple symbols
        if len(prices.columns) == 1:
            price = prices.iloc[:, 0]
        else:
            price = prices.mean(axis=1)

        ma50 = price.rolling(50).mean()
        ma200 = price.rolling(200).mean()

        # Signal: 1 when MA50 > MA200
        signal = (ma50 > ma200).astype(int).shift(1).fillna(0)

        # Returns
        returns = price.pct_change() * signal

        # Equity curve
        equity = (1 + returns).cumprod() * initial_capital

        return equity

    def _run_naive_mr(self, prices: pd.DataFrame, initial_capital: float) -> pd.Series:
        """
        Run naive mean reversion strategy.
        Buy when price < 20-day low, sell when price > 20-day high.
        """
        if prices.empty:
            return pd.Series([initial_capital])

        # Use first column or mean
        if len(prices.columns) == 1:
            price = prices.iloc[:, 0]
        else:
            price = prices.mean(axis=1)

        low_20 = price.rolling(20).min()
        high_20 = price.rolling(20).max()

        # Signal: 1 when near 20-day low
        signal = ((price - low_20) / (high_20 - low_20) < 0.2).astype(int).shift(1).fillna(0)

        # Returns
        returns = price.pct_change() * signal

        # Equity curve
        equity = (1 + returns).cumprod() * initial_capital

        return equity


def compare_to_baselines(
    strategy_equity: pd.Series,
    benchmark_prices: pd.Series,
) -> BaselineResult:
    """
    Convenience function to compare strategy to baselines.

    Args:
        strategy_equity: Strategy equity curve
        benchmark_prices: Benchmark prices

    Returns:
        BaselineResult
    """
    gate = Gate1Baseline()
    return gate.validate(strategy_equity, benchmark_prices)
