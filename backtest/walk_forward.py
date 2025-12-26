from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Iterable, List, Dict, Any

import pandas as pd

from .engine import Backtester, BacktestConfig


@dataclass
class WFSplit:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def generate_splits(
    start: date,
    end: date,
    train_days: int = 252,
    test_days: int = 63,
    anchored: bool = False,
) -> List[WFSplit]:
    """Generate rolling or anchored walk-forward splits in trading days (approx calendar days)."""
    splits: List[WFSplit] = []
    cur_train_start = start
    while True:
        if anchored:
            cur_train_start = start
        train_end = cur_train_start + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days - 1)
        if test_start > end:
            break
        if test_end > end:
            test_end = end
        splits.append(WFSplit(
            train_start=train_start_to_date(cur_train_start),
            train_end=train_start_to_date(train_end),
            test_start=train_start_to_date(test_start),
            test_end=train_start_to_date(test_end),
        ))
        # Advance by test_days
        cur_train_start = cur_train_start + timedelta(days=test_days)
        if cur_train_start + timedelta(days=train_days) > end:
            break
    return splits


def train_start_to_date(d: date | datetime) -> date:
    return d if isinstance(d, date) and not isinstance(d, datetime) else d.date()


def run_walk_forward(
    symbols: List[str],
    fetch_bars_for_window: Callable[[str, str, str], pd.DataFrame],
    get_signals: Callable[[pd.DataFrame], pd.DataFrame],
    splits: List[WFSplit],
    outdir: str,
    initial_cash: float = 100_000.0,
) -> List[Dict[str, Any]]:
    """
    Run walk-forward backtests over splits.

    fetch_bars_for_window: (symbol, start_str, end_str) -> DataFrame
    get_signals: strategy function mapping merged OHLCV -> signal DataFrame
    """
    results: List[Dict[str, Any]] = []
    for i, sp in enumerate(splits, start=1):
        # Fetch from train_start to ensure indicators (e.g., SMA200) have sufficient lookback
        start_s = sp.train_start.isoformat()
        end_s = sp.test_end.isoformat()

        def fetcher(sym: str) -> pd.DataFrame:
            return fetch_bars_for_window(sym, start_s, end_s)

        cfg = BacktestConfig(initial_cash=initial_cash)
        bt = Backtester(cfg, get_signals, fetcher)
        res = bt.run(symbols, outdir=f"{outdir}/split_{i:02d}")
        m = res.get("metrics", {})
        results.append({
            "split": i,
            "test_start": start_s,
            "test_end": end_s,
            **m,
        })
    return results


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"splits": 0}
    df = pd.DataFrame(results)
    summary = {
        "splits": len(results),
        "trades_total": int(df["trades"].sum()),
        "win_rate_avg": float(df["win_rate"].mean()),
        "profit_factor_avg": float(df["profit_factor"].replace([float('inf')], pd.NA).dropna().mean() or 0.0),
        "sharpe_avg": float(df["sharpe"].mean()),
        "max_drawdown_avg": float(df["max_drawdown"].mean()),
    }
    return summary
