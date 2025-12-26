#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
from strategies.ibs.strategy import IBSStrategy
from backtest.engine import Backtester, BacktestConfig

# Synthetic fetcher (deterministic)
def fetch_bars(symbol: str) -> pd.DataFrame:
    np.random.seed(abs(hash(symbol)) % 2**32)
    days = 260
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
    # Choose one strategy at a time
    strat = ConnorsRSI2Strategy()
    # strat = IBSStrategy()

    def get_signals(df: pd.DataFrame) -> pd.DataFrame:
        # Use backtest-friendly multi-bar scan
        return strat.scan_signals_over_time(df)

    cfg = BacktestConfig(initial_cash=100_000.0)
    bt = Backtester(cfg, get_signals, fetch_bars)
    symbols = ['AAPL','MSFT','NVDA','AMZN']
    res = bt.run(symbols)
    m = res.get('metrics', {})
    print(f"Trades: {len(res['trades'])} | PnL: {res.get('pnl', 0.0):.2f} | WR: {m.get('win_rate',0.0):.2f} | PF: {m.get('profit_factor',0.0):.2f} | Sharpe: {m.get('sharpe',0.0):.2f} | MaxDD: {m.get('max_drawdown',0.0):.2f}")

if __name__ == '__main__':
    main()
