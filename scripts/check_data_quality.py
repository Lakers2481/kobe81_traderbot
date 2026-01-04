#!/usr/bin/env python3
"""Check data quality for stocks - continuous validation."""
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

def check_quality(symbols: str, dotenv: str) -> int:
    load_env(Path(dotenv))
    end = now_et().date()
    start = end - timedelta(days=30)

    symbol_list = [s.strip() for s in symbols.split(',')]
    issues = []

    for symbol in symbol_list:
        try:
            df = fetch_daily_bars_polygon(symbol, start.isoformat(), end.isoformat())
            if df is None or df.empty:
                issues.append(f"{symbol}: no data")
            elif len(df) < 15:
                issues.append(f"{symbol}: only {len(df)} bars (expected ~20)")
            else:
                # Check for OHLC violations
                violations = ((df['high'] < df['low']) | (df['close'] > df['high']) | (df['close'] < df['low'])).sum()
                if violations > 0:
                    issues.append(f"{symbol}: {violations} OHLC violations")
                # Check for zero volume
                zero_vol = (df['volume'] == 0).sum()
                if zero_vol > 0:
                    issues.append(f"{symbol}: {zero_vol} zero-volume bars")
        except Exception as e:
            issues.append(f"{symbol}: error - {e}")

    if issues:
        print("[DATA_QUALITY] Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    else:
        print(f"[DATA_QUALITY] All {len(symbol_list)} symbols OK")
        return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', required=True, help='Comma-separated symbols')
    ap.add_argument('--dotenv', default='.env')
    args = ap.parse_args()
    return check_quality(args.symbols, args.dotenv)

if __name__ == '__main__':
    sys.exit(main())
