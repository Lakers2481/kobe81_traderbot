#!/usr/bin/env python3
from __future__ import annotations

"""
Fetch Polygon news for a universe and compute daily sentiment cache.

Writes per-day sentiment CSV to data/sentiment/sentiment_YYYY-MM-DD.csv
"""

import argparse
import os
from datetime import date, timedelta
from pathlib import Path
from typing import List

import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from data.universe.loader import load_universe
from altdata.sentiment import fetch_polygon_news, compute_daily_sentiment, write_daily_cache


def main() -> None:
    ap = argparse.ArgumentParser(description='Update daily sentiment cache from Polygon news')
    ap.add_argument('--universe', type=str, default='data/universe/optionable_liquid_800.csv')
    ap.add_argument('--date', type=str, default=None, help='YYYY-MM-DD (default: today)')
    ap.add_argument('--lookback', type=int, default=1, help='Days of news to include (default 1)')
    ap.add_argument('--dotenv', type=str, default='./.env')
    args = ap.parse_args()

    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)
    api_key = os.getenv('POLYGON_API_KEY', '')
    if not api_key:
        print('POLYGON_API_KEY not set; cannot fetch news.')
        return

    d = date.fromisoformat(args.date) if args.date else date.today()
    start = (d - timedelta(days=int(args.lookback))).isoformat()
    end = d.isoformat()

    symbols = load_universe(Path(args.universe), cap=None)
    all_frames: List[pd.DataFrame] = []
    for sym in symbols:
        items = fetch_polygon_news(sym, start, end, api_key=api_key, limit=50)
        agg = compute_daily_sentiment(items)
        if not agg.empty:
            all_frames.append(agg)
    out = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(columns=['date','symbol','sent_mean','sent_count'])
    p = write_daily_cache(out, end)
    print('Wrote sentiment cache:', p)


if __name__ == '__main__':
    main()
