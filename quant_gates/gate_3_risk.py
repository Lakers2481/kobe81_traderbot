"""
Gate 3: Risk Realism
====================

Strategy must meet realistic risk constraints:
- Max drawdown: 25%
- Min trades: 100 (statistical significance)
- Min symbols: 30 (diversification)
- Max ticker P&L: 20% (no single stock dependency)
- Max turnover: 100x annual
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskResult:
    """Result from risk realism checks."""
    passed: bool
    max_drawdown_pct: float
    total_trades: int
    unique_symbols: int
    max_ticker_pnl_pct: float
    annual_turnover: float
    violations: List[str]
    details: Dict[str, Any]


class Gate3RiskRealism:
    """
    Gate 3: Risk realism validation.

    Requirements:
    - Max drawdown <= 25%
    - Min trades >= 100
    - Min unique symbols >= 30
    - Max single ticker P&L <= 20% of total
    - Annual turnover <= 100x

    FAIL = ARCHIVE
    """

    # Thresholds
    MAX_DRAWDOWN = 0.25
    MIN_TRADES = 100
    MIN_SYMBOLS = 30
    MAX_TICKER_PNL_PCT = 0.20
    MAX_ANNUAL_TURNOVER = 100.0

    def __init__(self):
        self.violations: List[str] = []

    def validate(
        self,
        equity_curve: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> RiskResult:
        """
        Run risk realism validation.

        Args:
            equity_curve: Equity curve series
            trades: Trades DataFrame with columns: symbol, pnl, date
            metrics: Pre-calculated metrics dict

        Returns:
            RiskResult
        """
        self.violations = []
        details = {}

        # Calculate max drawdown
        max_dd = 0.0
        if equity_curve is not None and len(equity_curve) > 0:
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_dd = abs(drawdown.min())
        elif metrics and "max_drawdown" in metrics:
            max_dd = metrics["max_drawdown"]
        details["max_drawdown"] = max_dd

        if max_dd > self.MAX_DRAWDOWN:
            self.violations.append(f"Max drawdown {max_dd:.1%} > {self.MAX_DRAWDOWN:.0%}")

        # Count trades
        total_trades = 0
        if trades is not None:
            total_trades = len(trades)
        elif metrics and "total_trades" in metrics:
            total_trades = metrics["total_trades"]
        details["total_trades"] = total_trades

        if total_trades < self.MIN_TRADES:
            self.violations.append(f"Total trades {total_trades} < {self.MIN_TRADES}")

        # Count unique symbols
        unique_symbols = 0
        if trades is not None and "symbol" in trades.columns:
            unique_symbols = trades["symbol"].nunique()
        elif metrics and "unique_symbols" in metrics:
            unique_symbols = metrics["unique_symbols"]
        details["unique_symbols"] = unique_symbols

        if unique_symbols < self.MIN_SYMBOLS:
            self.violations.append(f"Unique symbols {unique_symbols} < {self.MIN_SYMBOLS}")

        # Calculate max ticker P&L concentration
        max_ticker_pnl = 0.0
        if trades is not None and "pnl" in trades.columns and "symbol" in trades.columns:
            total_pnl = trades["pnl"].sum()
            if total_pnl != 0:
                ticker_pnl = trades.groupby("symbol")["pnl"].sum().abs()
                max_ticker_pnl = ticker_pnl.max() / abs(total_pnl)
        elif metrics and "max_ticker_pnl_pct" in metrics:
            max_ticker_pnl = metrics["max_ticker_pnl_pct"]
        details["max_ticker_pnl"] = max_ticker_pnl

        if max_ticker_pnl > self.MAX_TICKER_PNL_PCT:
            self.violations.append(f"Max ticker P&L {max_ticker_pnl:.1%} > {self.MAX_TICKER_PNL_PCT:.0%}")

        # Calculate annual turnover
        annual_turnover = 0.0
        if trades is not None and equity_curve is not None:
            # Turnover = total traded value / average equity
            if "value" in trades.columns:
                total_traded = trades["value"].sum()
                avg_equity = equity_curve.mean()
                if avg_equity > 0:
                    years = len(equity_curve) / 252
                    annual_turnover = (total_traded / avg_equity) / years if years > 0 else 0
        elif metrics and "annual_turnover" in metrics:
            annual_turnover = metrics["annual_turnover"]
        details["annual_turnover"] = annual_turnover

        if annual_turnover > self.MAX_ANNUAL_TURNOVER:
            self.violations.append(f"Annual turnover {annual_turnover:.0f}x > {self.MAX_ANNUAL_TURNOVER:.0f}x")

        passed = len(self.violations) == 0

        return RiskResult(
            passed=passed,
            max_drawdown_pct=max_dd,
            total_trades=total_trades,
            unique_symbols=unique_symbols,
            max_ticker_pnl_pct=max_ticker_pnl,
            annual_turnover=annual_turnover,
            violations=self.violations,
            details=details,
        )


def check_risk_realism(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
) -> RiskResult:
    """
    Convenience function for risk realism check.

    Args:
        equity_curve: Equity curve
        trades: Trades DataFrame

    Returns:
        RiskResult
    """
    gate = Gate3RiskRealism()
    return gate.validate(equity_curve, trades)
