#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
from strategies.ibs.strategy import IBSStrategy
from backtest.engine import Backtester, BacktestConfig


def synthetic_bars(symbol: str, days: int = 260) -> pd.DataFrame:
    np.random.seed(abs(hash(symbol)) % 2**32)
    dates = pd.date_range(end=datetime.utcnow().date(), periods=days, freq='B')
    rets = np.random.normal(0.0004, 0.01, days)
    close = 100 * np.cumprod(1 + rets)
    high = close * (1 + np.random.uniform(0, 0.01, days))
    low = close * (1 - np.random.uniform(0, 0.01, days))
    openp = close * (1 + np.random.uniform(-0.002, 0.002, days))
    vol = np.random.randint(1_000_000, 5_000_000, days)
    return pd.DataFrame({
        'timestamp': dates,
        'symbol': symbol,
        'open': openp,
        'high': high,
        'low': low,
        'close': close,
        'volume': vol,
    })


def main():
    symbols = ['AAPL','MSFT','NVDA']
    frames = [synthetic_bars(s) for s in symbols]
    data = pd.concat(frames, ignore_index=True)

    for name, strat in (
        ('rsi2', ConnorsRSI2Strategy()),
        ('ibs', IBSStrategy()),
    ):
        def get_signals(df: pd.DataFrame) -> pd.DataFrame:
            return strat.scan_signals_over_time(df)
        def fetcher(sym: str) -> pd.DataFrame:
            return data[data['symbol']==sym]
        bt = Backtester(BacktestConfig(initial_cash=100_000.0), get_signals, fetcher)
        res = bt.run(symbols, outdir=f'smoke_outputs/{name}')
        m = res.get('metrics', {})
        print(f"{name}: trades={len(res['trades'])} pnl={res.get('pnl',0.0):.2f} WR={m.get('win_rate',0.0):.2f} PF={m.get('profit_factor',0.0):.2f}")


if __name__ == '__main__':
    main()

