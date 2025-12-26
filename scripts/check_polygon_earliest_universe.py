#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.env_loader import load_env
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon


def main():
    ap = argparse.ArgumentParser(description='Check earliest/latest Polygon daily bars for an entire universe')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--start', type=str, default='1970-01-01')
    ap.add_argument('--end', type=str, default=datetime.utcnow().date().isoformat())
    ap.add_argument('--cache', type=str, default='data/cache')
    ap.add_argument('--concurrency', type=int, default=3)
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
    ap.add_argument('--out', type=str, default='data/universe/earliest_latest_universe.csv')
    args = ap.parse_args()

    dotenv = Path(args.dotenv)
    if dotenv.exists():
        _loaded = load_env(dotenv)
        print('Loaded %d env vars from %s' % (len(_loaded), dotenv))

    symbols = load_universe(Path(args.universe))
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def task(sym: str) -> Dict[str, Any]:
        df = fetch_daily_bars_polygon(sym, args.start, args.end, cache_dir=cache_dir)
        if df.empty:
            return {"symbol": sym, "earliest": None, "latest": None, "rows": 0}
        df = df.sort_values('timestamp')
        return {"symbol": sym, "earliest": str(df.iloc[0]['timestamp']), "latest": str(df.iloc[-1]['timestamp']), "rows": int(len(df))}

    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = {ex.submit(task, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futs), start=1):
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({"symbol": futs[fut], "earliest": None, "latest": None, "rows": 0, "error": str(e)})
            if i % 50 == 0:
                print(f"Processed {i}/{len(symbols)} symbols")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote earliest/latest per symbol: {out}")


if __name__ == '__main__':
    main()

