#!/usr/bin/env python3
"""Study volatility patterns - continuous research."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from data.providers.polygon_eod import fetch_daily_bars_polygon
from core.clock.tz_utils import now_et
import numpy as np

def study_volatility(symbol: str, dotenv: str) -> int:
    load_env(Path(dotenv))
    end = now_et().date()
    start = end - timedelta(days=252)

    try:
        df = fetch_daily_bars_polygon(symbol, start.isoformat(), end.isoformat())
        if df is None or len(df) < 60:
            print(f"[VOLATILITY] {symbol}: insufficient data")
            return 1

        returns = df['close'].pct_change()

        # Various volatility measures
        vol_20 = returns.tail(20).std() * np.sqrt(252)
        vol_60 = returns.tail(60).std() * np.sqrt(252)
        vol_252 = returns.std() * np.sqrt(252)

        # Volatility ratio (short vs long)
        vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1

        # Parkinson volatility (uses high-low)
        log_hl = np.log(df['high'] / df['low'])
        parkinson = np.sqrt((log_hl ** 2).tail(20).mean() / (4 * np.log(2))) * np.sqrt(252)

        # ATR-based volatility
        atr = ((df['high'] - df['low']).tail(14).mean() / df['close'].iloc[-1])

        print(f"[VOLATILITY] {symbol}: 20d={vol_20:.1%}, 60d={vol_60:.1%}, ratio={vol_ratio:.2f}, ATR%={atr:.1%}")

        # Volatility regime
        if vol_ratio > 1.3:
            print(f"[VOLATILITY] {symbol}: EXPANDING volatility")
        elif vol_ratio < 0.7:
            print(f"[VOLATILITY] {symbol}: CONTRACTING volatility")
        else:
            print(f"[VOLATILITY] {symbol}: STABLE volatility")

        return 0

    except Exception as e:
        print(f"[VOLATILITY] {symbol}: ERROR - {e}")
        return 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--dotenv', default='.env')
    args = ap.parse_args()
    return study_volatility(args.symbol, args.dotenv)

if __name__ == '__main__':
    sys.exit(main())
