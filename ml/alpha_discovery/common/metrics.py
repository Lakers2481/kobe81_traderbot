"""
Performance metrics for ML Alpha Discovery.
"""
from __future__ import annotations

from typing import List, Union
import numpy as np
import pandas as pd


def calculate_win_rate(
    pnl: Union[List[float], np.ndarray, pd.Series],
) -> float:
    """
    Calculate win rate from P&L values.

    Args:
        pnl: List of P&L values per trade

    Returns:
        Win rate as decimal (0.0 to 1.0)
    """
    if len(pnl) == 0:
        return 0.0
    arr = np.asarray(pnl)
    wins = np.sum(arr > 0)
    return float(wins / len(arr))


def calculate_profit_factor(
    pnl: Union[List[float], np.ndarray, pd.Series],
) -> float:
    """
    Calculate profit factor from P&L values.

    Args:
        pnl: List of P&L values per trade

    Returns:
        Profit factor (gross wins / gross losses)
    """
    if len(pnl) == 0:
        return 0.0
    arr = np.asarray(pnl)
    gross_wins = np.sum(arr[arr > 0])
    gross_losses = np.abs(np.sum(arr[arr < 0]))

    if gross_losses < 0.0001:
        return float(gross_wins) if gross_wins > 0 else 0.0
    return float(gross_wins / gross_losses)


def calculate_sharpe(
    returns: Union[List[float], np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Daily/periodic returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns)
    excess = arr - (risk_free_rate / periods_per_year)
    mean_excess = np.mean(excess)
    std_excess = np.std(excess, ddof=1)

    if std_excess < 1e-10:
        return 0.0
    return float(mean_excess / std_excess * np.sqrt(periods_per_year))


def calculate_max_drawdown(
    equity: Union[List[float], np.ndarray, pd.Series],
) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity: Equity values over time

    Returns:
        Maximum drawdown as decimal (0.0 to 1.0)
    """
    if len(equity) < 2:
        return 0.0
    arr = np.asarray(equity)
    peak = np.maximum.accumulate(arr)
    drawdown = (peak - arr) / peak
    return float(np.max(drawdown))


def calculate_r_multiple(
    entry_price: float,
    exit_price: float,
    stop_price: float,
    side: str = "long",
) -> float:
    """
    Calculate R-multiple (return as multiple of risk).

    Args:
        entry_price: Entry price
        exit_price: Exit price
        stop_price: Stop loss price
        side: "long" or "short"

    Returns:
        R-multiple (positive = win, negative = loss)
    """
    if side.lower() == "long":
        risk = entry_price - stop_price
        reward = exit_price - entry_price
    else:
        risk = stop_price - entry_price
        reward = entry_price - exit_price

    if abs(risk) < 0.0001:
        return 0.0
    return float(reward / risk)


def calculate_expectancy(
    pnl: Union[List[float], np.ndarray, pd.Series],
) -> float:
    """
    Calculate trading expectancy (average P&L per trade).

    Args:
        pnl: List of P&L values per trade

    Returns:
        Average P&L per trade
    """
    if len(pnl) == 0:
        return 0.0
    return float(np.mean(pnl))


def calculate_sortino(
    returns: Union[List[float], np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).

    Args:
        returns: Daily/periodic returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns)
    excess = arr - (risk_free_rate / periods_per_year)
    mean_excess = np.mean(excess)

    # Downside deviation (only negative returns)
    downside = arr[arr < 0]
    if len(downside) < 2:
        return float(mean_excess * np.sqrt(periods_per_year)) if mean_excess > 0 else 0.0

    downside_std = np.std(downside, ddof=1)
    if downside_std < 1e-10:
        return 0.0

    return float(mean_excess / downside_std * np.sqrt(periods_per_year))


def calculate_calmar(
    total_return: float,
    max_drawdown: float,
) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).

    Args:
        total_return: Total return as decimal
        max_drawdown: Maximum drawdown as decimal

    Returns:
        Calmar ratio
    """
    if max_drawdown < 0.0001:
        return float(total_return) if total_return > 0 else 0.0
    return float(total_return / max_drawdown)


def calculate_cluster_stats(
    trades_df: pd.DataFrame,
    pnl_col: str = "pnl",
    won_col: str = "won",
) -> dict:
    """
    Calculate comprehensive statistics for a cluster of trades.

    Args:
        trades_df: DataFrame with trade data
        pnl_col: Column name for P&L
        won_col: Column name for win indicator

    Returns:
        Dict with win_rate, profit_factor, avg_pnl, sharpe, etc.
    """
    if trades_df.empty:
        return {
            'n_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_pnl': 0.0,
            'total_pnl': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
        }

    pnl = trades_df[pnl_col].values if pnl_col in trades_df.columns else []

    return {
        'n_trades': len(trades_df),
        'win_rate': calculate_win_rate(pnl),
        'profit_factor': calculate_profit_factor(pnl),
        'avg_pnl': calculate_expectancy(pnl),
        'total_pnl': float(np.sum(pnl)) if len(pnl) > 0 else 0.0,
        'sharpe': calculate_sharpe(pnl) if len(pnl) > 10 else 0.0,
        'max_drawdown': 0.0,  # Would need equity curve
    }
