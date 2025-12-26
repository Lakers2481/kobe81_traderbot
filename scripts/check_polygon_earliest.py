#!/usr/bin/env python3
from __future__ import annotations

import sys, os
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_env(dotenv_path: _P):
    if not dotenv_path.exists():
        return {}
    loaded = {}
    for line in dotenv_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        key = key.strip(); val = val.strip().strip('"').strip("'")
        os.environ[key] = val
        loaded[key] = val
    return loaded
from data.providers.polygon_eod import fetch_daily_bars_polygon


def main():
    ap = argparse.ArgumentParser(description='Check earliest Polygon daily bar available for a symbol')
    ap.add_argument('--symbol', type=str, default='AAPL')
    ap.add_argument('--start', type=str, default='1970-01-01')
    ap.add_argument('--end', type=str, default=datetime.utcnow().date().isoformat())
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
    ap.add_argument('--cache', type=str, default='data/cache')
    args = ap.parse_args()

    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)
    df = fetch_daily_bars_polygon(args.symbol, args.start, args.end, cache_dir=Path(args.cache))
    if df.empty:
        print(f"No data returned for {args.symbol} in [{args.start},{args.end}]")
        return
    first = df.iloc[0]['timestamp']
    last = df.iloc[-1]['timestamp']
    print(f"Symbol: {args.symbol} | earliest: {first.date()} | latest: {last.date()} | rows: {len(df)}")

if __name__ == '__main__':
    main()
