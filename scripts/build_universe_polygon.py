#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

import sys
import os
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.providers.polygon_eod import fetch_daily_bars_polygon, has_options_polygon
from data.providers.multi_source import fetch_daily_bars_multi
from data.universe.loader import load_universe
from config.env_loader import load_env


def analyze_symbol(sym: str, start: str, end: str, cache_dir: Path, use_fallback: bool = False) -> Dict[str, Any]:
    df = (fetch_daily_bars_multi if use_fallback else fetch_daily_bars_polygon)(sym, start, end, cache_dir=cache_dir)
    if df.empty:
        return {"symbol": sym, "ok": False, "reason": "no_data"}
    df = df.sort_values('timestamp')
    first = pd.to_datetime(df.iloc[0]['timestamp']).date()
    last = pd.to_datetime(df.iloc[-1]['timestamp']).date()
    years = (last - max(first, pd.to_datetime(start).date())).days / 365.25
    # ADV over last 60 days
    tail = df.tail(60)
    adv_shares = float(tail['volume'].mean()) if not tail.empty else 0.0
    last_close = float(df.iloc[-1]['close'])
    adv_dollar = adv_shares * last_close
    has_opts = has_options_polygon(sym)
    return {
        "symbol": sym,
        "ok": True,
        "earliest": str(first),
        "latest": str(last),
        "years": years,
        "adv_shares": adv_shares,
        "adv_dollar": adv_dollar,
        "has_options": has_opts,
    }


def main():
    ap = argparse.ArgumentParser(description='Build a 900-stock optionable, liquid universe using Polygon EOD')
    ap.add_argument('--candidates', type=str, required=True, help='CSV with candidate symbols (column: symbol)')
    ap.add_argument('--start', type=str, default='2015-01-01')
    ap.add_argument('--end', type=str, default=datetime.utcnow().date().isoformat())
    ap.add_argument('--min-years', type=int, default=10)
    ap.add_argument('--cap', type=int, default=900)
    ap.add_argument('--concurrency', type=int, default=3)
    ap.add_argument('--cache', type=str, default='data/cache')
    ap.add_argument('--out', type=str, default='data/universe/optionable_liquid_900.csv')
    ap.add_argument('--dotenv', type=str, default='./.env')
    ap.add_argument('--fallback-free', action='store_true', default=False, help='Backfill pre-Polygon coverage with Yahoo Finance')
    args = ap.parse_args()

    dotenv = Path(args.dotenv)
    if dotenv.exists():
        _loaded = load_env(dotenv)
        print('Loaded %d env vars from %s' % (len(_loaded), dotenv))
    candidates = load_universe(Path(args.candidates), cap=None)
    cache_dir = Path(args.cache)

    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = {ex.submit(analyze_symbol, sym, args.start, args.end, cache_dir, args.fallback_free): sym for sym in candidates}
        for fut in as_completed(futs):
            rec = fut.result()
            if rec.get('ok'):
                rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        print('No symbols qualified.')
        return
    # Filter by years and options availability
    df = df[(df['years'] >= args.min_years) & (df['has_options'])]
    if df.empty:
        print('No symbols met min-years and options filters.')
        return
    # Sort by ADV dollar and select top cap
    df = df.sort_values('adv_dollar', ascending=False)
    if len(df) < args.cap:
        print(f"Warning: only {len(df)} symbols met the criteria (< {args.cap}).")
    final_df = df.head(args.cap)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    final_df[['symbol']].to_csv(args.out, index=False)
    # Also write full details
    final_df.to_csv(str(Path(args.out).with_suffix('.full.csv')), index=False)
    print(f"Universe written: {args.out} ({len(final_df)} symbols)")


if __name__ == '__main__':
    main()
