#!/usr/bin/env python3
"""
Performance Metrics Calculator for Kobe Trading System.

Calculates win rate, profit factor, Sharpe ratio, returns analysis,
and max drawdown from trade history.

Usage:
    python metrics.py --period 30d --strategy donchian
    python metrics.py --wfdir wf_outputs/donchian --dotenv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TradeMetrics:
    """Container for computed trade metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade: float = 0.0
    expectancy: float = 0.0


@dataclass
class EquityMetrics:
    """Container for equity curve metrics."""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # days
    cagr: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    total_return: float = 0.0
    daily_returns_mean: float = 0.0
    daily_returns_std: float = 0.0


@dataclass
class PeriodReturns:
    """Container for period returns."""
    daily: pd.Series
    weekly: pd.Series
    monthly: pd.Series


def load_env(dotenv_path: Optional[str] = None) -> None:
    """Load environment variables from .env file."""
    if dotenv_path is None:
        dotenv_path = Path(__file__).parent.parent / '.env'
    else:
        dotenv_path = Path(dotenv_path)

    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ[key] = val


def load_trades_from_csv(path: Path) -> pd.DataFrame:
    """Load trades from CSV file."""
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df


def load_trades_from_jsonl(path: Path) -> pd.DataFrame:
    """Load trades from JSONL file."""
    if not path.exists():
        return pd.DataFrame()

    records = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.columns = df.columns.str.lower().str.strip()

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df


def load_equity_curve(path: Path) -> pd.DataFrame:
    """Load equity curve from CSV file."""
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    df.columns = df.columns.str.lower().str.strip()

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    return df


def discover_trade_files(base_dir: Path, strategy: Optional[str] = None) -> List[Path]:
    """Discover trade list files in a directory structure."""
    files = []

    if strategy:
        strategy_dir = base_dir / strategy
        if strategy_dir.exists():
            files.extend(strategy_dir.glob('**/trade_list.csv'))
    else:
        files.extend(base_dir.glob('**/trade_list.csv'))

    # Also check state directory
    state_dir = base_dir.parent / 'state'
    if state_dir.exists():
        files.extend(state_dir.glob('trades.jsonl'))
        files.extend(state_dir.glob('**/trades.jsonl'))

    return sorted(files)


def discover_equity_files(base_dir: Path, strategy: Optional[str] = None) -> List[Path]:
    """Discover equity curve files in a directory structure."""
    files = []

    if strategy:
        strategy_dir = base_dir / strategy
        if strategy_dir.exists():
            files.extend(strategy_dir.glob('**/equity_curve.csv'))
    else:
        files.extend(base_dir.glob('**/equity_curve.csv'))

    return sorted(files)


def compute_trade_metrics(trades_df: pd.DataFrame) -> TradeMetrics:
    """Compute trade-level metrics from a trades DataFrame."""
    metrics = TradeMetrics()

    if trades_df.empty:
        return metrics

    # Ensure required columns
    required_cols = {'symbol', 'side', 'qty', 'price'}
    if not required_cols.issubset(set(trades_df.columns)):
        print(f"Warning: Missing required columns. Have: {list(trades_df.columns)}")
        return metrics

    # FIFO matching of BUY/SELL trades per symbol
    buys: Dict[str, deque] = defaultdict(deque)
    realized_pnl: List[float] = []

    for _, row in trades_df.iterrows():
        symbol = str(row['symbol'])
        side = str(row['side']).upper()
        qty = int(row['qty'])
        price = float(row['price'])

        if side == 'BUY':
            buys[symbol].append((qty, price))
        elif side == 'SELL':
            remaining = qty
            while remaining > 0 and buys[symbol]:
                buy_qty, buy_price = buys[symbol][0]
                matched = min(remaining, buy_qty)
                pnl = (price - buy_price) * matched
                realized_pnl.append(pnl)
                remaining -= matched
                buy_qty -= matched
                if buy_qty == 0:
                    buys[symbol].popleft()
                else:
                    buys[symbol][0] = (buy_qty, buy_price)

    if not realized_pnl:
        return metrics

    # Calculate metrics
    pnl_array = np.array(realized_pnl)
    wins = pnl_array[pnl_array > 0]
    losses = pnl_array[pnl_array < 0]

    metrics.total_trades = len(realized_pnl)
    metrics.winning_trades = len(wins)
    metrics.losing_trades = len(losses)
    metrics.gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    metrics.gross_loss = float(losses.sum()) if len(losses) > 0 else 0.0
    metrics.net_profit = float(pnl_array.sum())

    if metrics.total_trades > 0:
        metrics.win_rate = metrics.winning_trades / metrics.total_trades
        metrics.avg_trade = metrics.net_profit / metrics.total_trades

    if len(wins) > 0:
        metrics.avg_win = float(wins.mean())
        metrics.largest_win = float(wins.max())

    if len(losses) > 0:
        metrics.avg_loss = float(losses.mean())
        metrics.largest_loss = float(losses.min())

    if metrics.gross_loss < 0:
        metrics.profit_factor = abs(metrics.gross_profit / metrics.gross_loss)
    elif metrics.gross_profit > 0:
        metrics.profit_factor = float('inf')

    # Expectancy = (Win Rate * Avg Win) + (Loss Rate * Avg Loss)
    if metrics.total_trades > 0:
        loss_rate = metrics.losing_trades / metrics.total_trades
        metrics.expectancy = (metrics.win_rate * metrics.avg_win) + (loss_rate * metrics.avg_loss)

    return metrics


def compute_equity_metrics(
    equity_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    trading_days: int = 252
) -> EquityMetrics:
    """Compute equity curve metrics."""
    metrics = EquityMetrics()

    if equity_df.empty or 'equity' not in equity_df.columns:
        return metrics

    equity = equity_df['equity'].dropna()
    if len(equity) < 2:
        return metrics

    # Calculate returns
    if 'returns' in equity_df.columns:
        returns = equity_df['returns'].dropna()
    else:
        returns = equity.pct_change().dropna()

    if len(returns) < 2:
        return metrics

    # Basic statistics
    metrics.daily_returns_mean = float(returns.mean())
    metrics.daily_returns_std = float(returns.std(ddof=1))
    metrics.volatility = metrics.daily_returns_std * np.sqrt(trading_days)

    # Total return
    initial_equity = equity.iloc[0]
    final_equity = equity.iloc[-1]
    metrics.total_return = (final_equity - initial_equity) / initial_equity

    # CAGR
    n_days = (equity.index[-1] - equity.index[0]).days if hasattr(equity.index[-1], 'days') else len(equity)
    if isinstance(n_days, int) and n_days > 0:
        years = n_days / 365.25
    else:
        # Estimate from number of trading days
        years = len(equity) / trading_days

    if years > 0 and initial_equity > 0 and final_equity > 0:
        metrics.cagr = (final_equity / initial_equity) ** (1 / years) - 1

    # Sharpe Ratio
    excess_returns = returns - (risk_free_rate / trading_days)
    if metrics.daily_returns_std > 0:
        metrics.sharpe_ratio = float(excess_returns.mean() / returns.std(ddof=1) * np.sqrt(trading_days))

    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = float(downside_returns.std(ddof=1))
        if downside_std > 0:
            metrics.sortino_ratio = float(excess_returns.mean() / downside_std * np.sqrt(trading_days))

    # Max Drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    metrics.max_drawdown = float(drawdown.min())

    # Max Drawdown Duration
    in_drawdown = drawdown < 0
    dd_groups = (in_drawdown != in_drawdown.shift()).cumsum()
    dd_durations = in_drawdown.groupby(dd_groups).sum()
    if len(dd_durations) > 0:
        metrics.max_drawdown_duration = int(dd_durations.max())

    # Calmar Ratio
    if metrics.max_drawdown < 0:
        metrics.calmar_ratio = metrics.cagr / abs(metrics.max_drawdown)

    return metrics


def compute_period_returns(equity_df: pd.DataFrame) -> PeriodReturns:
    """Compute daily, weekly, and monthly returns."""
    if equity_df.empty or 'equity' not in equity_df.columns:
        return PeriodReturns(
            daily=pd.Series(dtype=float),
            weekly=pd.Series(dtype=float),
            monthly=pd.Series(dtype=float)
        )

    equity = equity_df['equity'].copy()

    # Daily returns
    daily = equity.pct_change().dropna()

    # Weekly returns (resample to week end)
    try:
        weekly_equity = equity.resample('W').last().dropna()
        weekly = weekly_equity.pct_change().dropna()
    except Exception:
        weekly = pd.Series(dtype=float)

    # Monthly returns
    try:
        monthly_equity = equity.resample('ME').last().dropna()
        monthly = monthly_equity.pct_change().dropna()
    except Exception:
        monthly = pd.Series(dtype=float)

    return PeriodReturns(daily=daily, weekly=weekly, monthly=monthly)


def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Filter DataFrame by period string (e.g., '30d', '1y', 'ytd')."""
    if df.empty:
        return df

    # Find the timestamp column
    ts_col = None
    if 'timestamp' in df.columns:
        ts_col = 'timestamp'
    elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df['timestamp'] = df.index
        ts_col = 'timestamp'

    if ts_col is None:
        return df

    now = datetime.now()
    period = period.lower().strip()

    if period == 'ytd':
        start_date = datetime(now.year, 1, 1)
    elif period.endswith('d'):
        days = int(period[:-1])
        start_date = now - timedelta(days=days)
    elif period.endswith('w'):
        weeks = int(period[:-1])
        start_date = now - timedelta(weeks=weeks)
    elif period.endswith('m'):
        months = int(period[:-1])
        start_date = now - timedelta(days=months * 30)
    elif period.endswith('y'):
        years = int(period[:-1])
        start_date = now - timedelta(days=years * 365)
    else:
        return df

    return df[df[ts_col] >= start_date]


def format_metrics_table(
    trade_metrics: TradeMetrics,
    equity_metrics: EquityMetrics,
    period_returns: PeriodReturns
) -> str:
    """Format metrics as a text table."""
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append("PERFORMANCE METRICS REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Trade Statistics
    lines.append("-" * 40)
    lines.append("TRADE STATISTICS")
    lines.append("-" * 40)
    lines.append(f"{'Total Trades:':<25} {trade_metrics.total_trades:>12}")
    lines.append(f"{'Winning Trades:':<25} {trade_metrics.winning_trades:>12}")
    lines.append(f"{'Losing Trades:':<25} {trade_metrics.losing_trades:>12}")
    lines.append(f"{'Win Rate:':<25} {trade_metrics.win_rate:>11.2%}")
    lines.append(f"{'Profit Factor:':<25} {trade_metrics.profit_factor:>12.2f}")
    lines.append("")

    # P&L Statistics
    lines.append("-" * 40)
    lines.append("P&L STATISTICS")
    lines.append("-" * 40)
    lines.append(f"{'Gross Profit:':<25} ${trade_metrics.gross_profit:>10,.2f}")
    lines.append(f"{'Gross Loss:':<25} ${trade_metrics.gross_loss:>10,.2f}")
    lines.append(f"{'Net Profit:':<25} ${trade_metrics.net_profit:>10,.2f}")
    lines.append(f"{'Avg Win:':<25} ${trade_metrics.avg_win:>10,.2f}")
    lines.append(f"{'Avg Loss:':<25} ${trade_metrics.avg_loss:>10,.2f}")
    lines.append(f"{'Largest Win:':<25} ${trade_metrics.largest_win:>10,.2f}")
    lines.append(f"{'Largest Loss:':<25} ${trade_metrics.largest_loss:>10,.2f}")
    lines.append(f"{'Avg Trade:':<25} ${trade_metrics.avg_trade:>10,.2f}")
    lines.append(f"{'Expectancy:':<25} ${trade_metrics.expectancy:>10,.2f}")
    lines.append("")

    # Risk Metrics
    lines.append("-" * 40)
    lines.append("RISK METRICS")
    lines.append("-" * 40)
    lines.append(f"{'Sharpe Ratio:':<25} {equity_metrics.sharpe_ratio:>12.3f}")
    lines.append(f"{'Sortino Ratio:':<25} {equity_metrics.sortino_ratio:>12.3f}")
    lines.append(f"{'Calmar Ratio:':<25} {equity_metrics.calmar_ratio:>12.3f}")
    lines.append(f"{'Max Drawdown:':<25} {equity_metrics.max_drawdown:>11.2%}")
    lines.append(f"{'Max DD Duration (days):':<25} {equity_metrics.max_drawdown_duration:>12}")
    lines.append(f"{'Volatility (annual):':<25} {equity_metrics.volatility:>11.2%}")
    lines.append("")

    # Return Metrics
    lines.append("-" * 40)
    lines.append("RETURN METRICS")
    lines.append("-" * 40)
    lines.append(f"{'Total Return:':<25} {equity_metrics.total_return:>11.2%}")
    lines.append(f"{'CAGR:':<25} {equity_metrics.cagr:>11.2%}")
    lines.append(f"{'Daily Return (mean):':<25} {equity_metrics.daily_returns_mean:>11.4%}")
    lines.append(f"{'Daily Return (std):':<25} {equity_metrics.daily_returns_std:>11.4%}")
    lines.append("")

    # Period Returns Summary
    if not period_returns.daily.empty:
        lines.append("-" * 40)
        lines.append("PERIOD RETURNS SUMMARY")
        lines.append("-" * 40)

        # Daily
        if len(period_returns.daily) > 0:
            lines.append(f"Daily   - Mean: {period_returns.daily.mean():>7.3%}  "
                        f"Std: {period_returns.daily.std():>7.3%}  "
                        f"Min: {period_returns.daily.min():>7.3%}  "
                        f"Max: {period_returns.daily.max():>7.3%}")

        # Weekly
        if len(period_returns.weekly) > 0:
            lines.append(f"Weekly  - Mean: {period_returns.weekly.mean():>7.3%}  "
                        f"Std: {period_returns.weekly.std():>7.3%}  "
                        f"Min: {period_returns.weekly.min():>7.3%}  "
                        f"Max: {period_returns.weekly.max():>7.3%}")

        # Monthly
        if len(period_returns.monthly) > 0:
            lines.append(f"Monthly - Mean: {period_returns.monthly.mean():>7.3%}  "
                        f"Std: {period_returns.monthly.std():>7.3%}  "
                        f"Min: {period_returns.monthly.min():>7.3%}  "
                        f"Max: {period_returns.monthly.max():>7.3%}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate performance metrics from trade history'
    )
    parser.add_argument(
        '--wfdir',
        type=str,
        default='wf_outputs',
        help='Directory containing WF outputs (default: wf_outputs)'
    )
    parser.add_argument(
        '--state',
        type=str,
        help='Path to state directory containing trades.jsonl'
    )
    parser.add_argument(
        '--period',
        type=str,
        help='Filter period: 30d, 90d, 1y, ytd, etc.'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        help='Filter by strategy name (e.g., donchian, TURTLE_SOUP)'
    )
    parser.add_argument(
        '--dotenv',
        action='store_true',
        help='Load environment variables from .env file'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output metrics as JSON'
    )
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.0,
        help='Annual risk-free rate for Sharpe calculation (default: 0.0)'
    )

    args = parser.parse_args()

    if args.dotenv:
        load_env()

    # Determine base directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    wf_dir = Path(args.wfdir)
    if not wf_dir.is_absolute():
        wf_dir = project_dir / wf_dir

    # Load trades
    all_trades = []
    trade_files = discover_trade_files(wf_dir, args.strategy)

    # Also check state directory
    if args.state:
        state_dir = Path(args.state)
        if not state_dir.is_absolute():
            state_dir = project_dir / state_dir
        if state_dir.exists():
            jsonl_files = list(state_dir.glob('trades.jsonl'))
            for f in jsonl_files:
                trade_files.append(f)

    for tf in trade_files:
        if tf.suffix == '.jsonl':
            df = load_trades_from_jsonl(tf)
        else:
            df = load_trades_from_csv(tf)
        if not df.empty:
            # Add source info
            df['source_file'] = str(tf)
            all_trades.append(df)

    if not all_trades:
        print("No trade data found.")
        sys.exit(1)

    trades_df = pd.concat(all_trades, ignore_index=True)

    # Load equity curves
    all_equity = []
    equity_files = discover_equity_files(wf_dir, args.strategy)

    for ef in equity_files:
        df = load_equity_curve(ef)
        if not df.empty:
            all_equity.append(df)

    # Merge equity curves if multiple (take mean or just use first)
    if all_equity:
        # For simplicity, use the first one or concatenate
        equity_df = all_equity[0]
    else:
        equity_df = pd.DataFrame()

    # Apply period filter
    if args.period:
        trades_df = filter_by_period(trades_df, args.period)
        if not equity_df.empty:
            equity_df = filter_by_period(equity_df, args.period)

    # Compute metrics
    trade_metrics = compute_trade_metrics(trades_df)
    equity_metrics = compute_equity_metrics(equity_df, args.risk_free_rate)
    period_returns = compute_period_returns(equity_df)

    if args.json:
        output = {
            'trade_metrics': {
                'total_trades': trade_metrics.total_trades,
                'winning_trades': trade_metrics.winning_trades,
                'losing_trades': trade_metrics.losing_trades,
                'win_rate': trade_metrics.win_rate,
                'profit_factor': trade_metrics.profit_factor,
                'gross_profit': trade_metrics.gross_profit,
                'gross_loss': trade_metrics.gross_loss,
                'net_profit': trade_metrics.net_profit,
                'avg_win': trade_metrics.avg_win,
                'avg_loss': trade_metrics.avg_loss,
                'largest_win': trade_metrics.largest_win,
                'largest_loss': trade_metrics.largest_loss,
                'avg_trade': trade_metrics.avg_trade,
                'expectancy': trade_metrics.expectancy,
            },
            'equity_metrics': {
                'sharpe_ratio': equity_metrics.sharpe_ratio,
                'sortino_ratio': equity_metrics.sortino_ratio,
                'calmar_ratio': equity_metrics.calmar_ratio,
                'max_drawdown': equity_metrics.max_drawdown,
                'max_drawdown_duration': equity_metrics.max_drawdown_duration,
                'cagr': equity_metrics.cagr,
                'volatility': equity_metrics.volatility,
                'total_return': equity_metrics.total_return,
            },
            'period_returns': {
                'daily_count': len(period_returns.daily),
                'weekly_count': len(period_returns.weekly),
                'monthly_count': len(period_returns.monthly),
            }
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_metrics_table(trade_metrics, equity_metrics, period_returns))


if __name__ == '__main__':
    main()

