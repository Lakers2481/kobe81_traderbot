#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.multi_source import fetch_daily_bars_multi
from config.env_loader import load_env


def main():
    ap = argparse.ArgumentParser(description='Prefetch and cache Polygon EOD bars for a universe')
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--start', type=str, required=True)
    ap.add_argument('--end', type=str, default=datetime.utcnow().date().isoformat())
    ap.add_argument('--cache', type=str, default='data/cache')
    ap.add_argument('--fallback-free', action='store_true', default=False)
    ap.add_argument('--concurrency', type=int, default=3)
    ap.add_argument('--dotenv', type=str, default='./.env')
    args = ap.parse_args()

    dotenv = Path(args.dotenv)
    if dotenv.exists():
        _loaded = load_env(dotenv)
        print('Loaded %d env vars from %s' % (len(_loaded), dotenv))

    symbols = load_universe(Path(args.universe))
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def task(sym: str) -> str:
        df = (fetch_daily_bars_multi if args.fallback_free else fetch_daily_bars_polygon)(sym, args.start, args.end, cache_dir=cache_dir)
        return f"{sym}:{len(df)}"

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = {ex.submit(task, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futs), start=1):
            try:
                msg = fut.result()
            except Exception as e:
                msg = f"{futs[fut]}:error:{e}"
            if i % 25 == 0:
                print(f"Prefetched {i}/{len(symbols)}... last: {msg}")
    print('Prefetch complete.')


if __name__ == '__main__':
    main()
