#!/usr/bin/env python3
"""Check correlations between stocks - continuous analysis."""
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
import pandas as pd

def check_correlation(symbols: str, dotenv: str) -> int:
    load_env(Path(dotenv))
    end = now_et().date()
    start = end - timedelta(days=60)

    symbol_list = [s.strip() for s in symbols.split(',')]
    returns_dict = {}

    for symbol in symbol_list:
        try:
            df = fetch_daily_bars_polygon(symbol, start.isoformat(), end.isoformat())
            if df is not None and len(df) > 20:
                returns_dict[symbol] = df['close'].pct_change()
        except:
            pass

    if len(returns_dict) < 2:
        print(f"[CORRELATION] Not enough data for correlation analysis")
        return 1

    returns_df = pd.DataFrame(returns_dict).dropna()
    corr_matrix = returns_df.corr()

    # Find highly correlated pairs
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            c = corr_matrix.iloc[i, j]
            if abs(c) > 0.7:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], c))

    if high_corr:
        print(f"[CORRELATION] Highly correlated pairs found:")
        for s1, s2, c in sorted(high_corr, key=lambda x: -abs(x[2]))[:5]:
            print(f"  {s1} <-> {s2}: {c:.2f}")
    else:
        print(f"[CORRELATION] No highly correlated pairs (>{0.7}) found in sample")

    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', required=True, help='Comma-separated symbols')
    ap.add_argument('--dotenv', default='.env')
    args = ap.parse_args()
    return check_correlation(args.symbols, args.dotenv)

if __name__ == '__main__':
    sys.exit(main())
