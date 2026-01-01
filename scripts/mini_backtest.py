#!/usr/bin/env python3
"""
Quick mini-backtest on a single stock - continuous validation.

NOTE: Uses production-like criteria (IBS<0.2, RSI2<10, SMA200 filter).
This is a SIMPLIFIED version - for accurate results use backtest_dual_strategy.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from data.providers.polygon_eod import fetch_daily_bars_polygon
from core.clock.tz_utils import now_et
import pandas as pd
import numpy as np
import json


def mini_backtest(symbol: str, days: int, dotenv: str) -> dict:
    """Run quick backtest on single stock."""
    load_env(Path(dotenv))

    end_date = now_et().date()
    start_date = end_date - timedelta(days=days + 250)  # Extra for SMA200 warmup

    try:
        df = fetch_daily_bars_polygon(symbol, start_date.isoformat(), end_date.isoformat())
        if df is None or len(df) < 50:
            return {'symbol': symbol, 'status': 'insufficient_data'}

        # Production-like IBS mean reversion strategy
        df['ibs'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)

        # RSI(2)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi2'] = 100 - (100 / (1 + rs))

        # SMA(200) filter - only trade uptrends
        df['sma200'] = df['close'].rolling(200).mean()

        # Signals: IBS < 0.2, RSI2 < 10, above SMA200 (relaxed from production for more signals)
        # Production uses IBS<0.08, RSI2<5 - we use looser thresholds for continuous learning
        ibs_cond = df['ibs'].shift(1) < 0.2
        rsi_cond = df['rsi2'].shift(1) < 10
        sma_cond = df['close'].shift(1) > df['sma200'].shift(1)

        df['signal'] = (ibs_cond & rsi_cond & sma_cond).astype(int)
        df['next_open'] = df['open'].shift(-1)
        df['exit_close'] = df['close'].shift(-3)

        # Calculate returns
        trades = df[df['signal'] == 1].copy()
        if len(trades) > 0:
            trades['return'] = (trades['exit_close'] - trades['next_open']) / trades['next_open']
            trades = trades.dropna(subset=['return'])

            wins = (trades['return'] > 0).sum()
            total = len(trades)
            win_rate = wins / total if total > 0 else 0
            avg_return = trades['return'].mean()
            total_return = (1 + trades['return']).prod() - 1

            result = {
                'symbol': symbol,
                'strategy': 'IBS_RSI_SMA200',  # Production-like
                'days': days,
                'trades': total,
                'wins': wins,
                'win_rate': float(win_rate),
                'avg_return': float(avg_return),
                'total_return': float(total_return),
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }

            print(f"[BACKTEST] {symbol}: {total} trades, {win_rate:.1%} WR, {total_return:.1%} return")
        else:
            result = {
                'symbol': symbol,
                'strategy': 'IBS_RSI_SMA200',  # Production-like
                'trades': 0,
                'status': 'no_signals'
            }
            print(f"[BACKTEST] {symbol}: No signals (needs IBS<0.2, RSI2<10, >SMA200)")

        return result

    except Exception as e:
        print(f"[BACKTEST] {symbol}: ERROR - {e}")
        return {'symbol': symbol, 'status': 'error', 'error': str(e)}


def main():
    ap = argparse.ArgumentParser(description='Mini backtest on single stock')
    ap.add_argument('--symbol', required=True, help='Stock symbol')
    ap.add_argument('--days', type=int, default=90, help='Days to backtest')
    ap.add_argument('--dotenv', default='.env', help='Path to .env file')
    args = ap.parse_args()

    result = mini_backtest(args.symbol, args.days, args.dotenv)
    return 0 if result.get('status') in ('success', 'no_signals') else 1


if __name__ == '__main__':
    sys.exit(main())
