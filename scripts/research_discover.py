#!/usr/bin/env python3
from __future__ import annotations

"""
Research: Quick discovery run for features/alphas.

Loads a small universe slice, computes research features/alphas, screens them
against forward returns, and writes a CSV summary to outputs/research/.

This is optional and not part of the production two‑strategy pipeline.
"""

import argparse
from pathlib import Path
from typing import List
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load environment variables for API keys
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from data.providers.multi_source import fetch_daily_bars_multi
from data.universe.loader import load_universe
from research.screener import screen_universe, save_screening_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Research discovery run (features/alphas screening)")
    ap.add_argument("--universe", type=str, default=str(ROOT / "data" / "universe" / "optionable_liquid_800.csv"))
    ap.add_argument("--start", type=str, default="2022-01-01")
    ap.add_argument("--end", type=str, default="2024-12-31")
    ap.add_argument("--cap", type=int, default=50, help="Limit symbols for faster screening")
    args = ap.parse_args()

    symbols: List[str] = load_universe(Path(args.universe), cap=args.cap)
    frames: List[pd.DataFrame] = []
    for s in symbols:
        df = fetch_daily_bars_multi(s, args.start, args.end, cache_dir=ROOT / 'data' / 'cache')
        if not df.empty:
            if 'symbol' not in df:
                df = df.copy(); df['symbol'] = s
            frames.append(df)
    if not frames:
        print("No data; abort.")
        return
    data = pd.concat(frames, ignore_index=True).sort_values(["symbol","timestamp"]).reset_index(drop=True)
    summary = screen_universe(data, horizons=(5, 10))
    p = save_screening_report(summary)
    print("Research screening summary:", p)


if __name__ == "__main__":
    main()

