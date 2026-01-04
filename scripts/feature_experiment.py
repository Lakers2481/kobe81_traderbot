#!/usr/bin/env python3
"""Feature engineering experiments - continuous learning."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from data.providers.polygon_eod import fetch_daily_bars_polygon
from core.clock.tz_utils import now_et
import numpy as np

def experiment(symbol: str, dotenv: str) -> int:
    load_env(Path(dotenv))
    end = now_et().date()
    start = end - timedelta(days=365)

    try:
        df = fetch_daily_bars_polygon(symbol, start.isoformat(), end.isoformat())
        if df is None or len(df) < 60:
            print(f"[FEATURE] {symbol}: insufficient data")
            return 1

        # Test various feature ideas
        features_tested = []

        # 1. Gap feature
        df['gap'] = df['open'] / df['close'].shift(1) - 1
        if df['gap'].std() > 0:
            features_tested.append('gap')

        # 2. Range expansion
        df['range_exp'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1)).replace(0, np.nan)
        features_tested.append('range_expansion')

        # 3. Close position in range
        df['close_pos'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
        features_tested.append('close_position')

        # 4. Volume delta
        df['vol_delta'] = df['volume'] / df['volume'].shift(1) - 1
        features_tested.append('volume_delta')

        print(f"[FEATURE] {symbol}: Tested {len(features_tested)} features: {', '.join(features_tested)}")
        return 0

    except Exception as e:
        print(f"[FEATURE] {symbol}: ERROR - {e}")
        return 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--dotenv', default='.env')
    args = ap.parse_args()
    return experiment(args.symbol, args.dotenv)

if __name__ == '__main__':
    sys.exit(main())
