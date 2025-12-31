#!/usr/bin/env python3
"""
Alpha Screener CLI
==================

Run walk-forward alpha screening on the universe to discover profitable alphas.

Usage:
    # Screen all alphas on universe
    python scripts/run_alpha_screener.py --universe data/universe/optionable_liquid_900.csv

    # Limit symbols for faster screening
    python scripts/run_alpha_screener.py --cap 50

    # Show top results
    python scripts/run_alpha_screener.py --top 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load environment variables for API keys
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from research.alphas import ALPHA_REGISTRY, compute_alphas
from research.screener import screen_universe, save_screening_report
from data.providers.multi_source import fetch_daily_bars_multi
from data.universe.loader import load_universe


def main():
    parser = argparse.ArgumentParser(
        description="Run alpha screening to discover profitable alphas",
    )
    parser.add_argument(
        '--universe',
        type=str,
        default=str(ROOT / 'data' / 'universe' / 'optionable_liquid_900.csv'),
        help='Path to universe CSV file',
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2022-01-01',
        help='Start date for screening',
    )
    parser.add_argument(
        '--end',
        type=str,
        default='2024-12-31',
        help='End date for screening',
    )
    parser.add_argument(
        '--cap',
        type=int,
        default=50,
        help='Limit symbols for faster screening',
    )
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top alphas to display',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save results to CSV file',
    )
    parser.add_argument(
        '--list-alphas',
        action='store_true',
        help='List all available alphas and exit',
    )

    args = parser.parse_args()

    # List alphas mode
    if args.list_alphas:
        print("Available alphas:")
        for name in ALPHA_REGISTRY.keys():
            print(f"  - {name}")
        return

    print(f"Alpha Screener - Scanning {args.cap} symbols")
    print("=" * 50)

    # Load universe
    symbols: List[str] = load_universe(Path(args.universe), cap=args.cap)
    print(f"Loaded {len(symbols)} symbols from universe")

    # Fetch data
    frames: List[pd.DataFrame] = []
    for i, s in enumerate(symbols):
        df = fetch_daily_bars_multi(s, args.start, args.end, cache_dir=ROOT / 'data' / 'cache')
        if not df.empty:
            if 'symbol' not in df:
                df = df.copy()
                df['symbol'] = s
            frames.append(df)
        if (i + 1) % 10 == 0:
            print(f"  Fetched {i + 1}/{len(symbols)} symbols...")

    if not frames:
        print("No data fetched; aborting.")
        sys.exit(1)

    data = pd.concat(frames, ignore_index=True).sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    print(f"Total bars: {len(data):,}")

    # Run screening
    print("\nRunning alpha screening...")
    summary = screen_universe(data, horizons=(5, 10, 20))

    # Save results
    if args.output:
        outpath = Path(args.output)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(outpath, index=False)
        print(f"\nResults saved to: {outpath}")
    else:
        outpath = save_screening_report(summary)
        print(f"\nResults saved to: {outpath}")

    # Display top results
    print(f"\n{'='*50}")
    print(f"TOP {args.top} ALPHAS BY FORWARD CORRELATION")
    print(f"{'='*50}")

    for horizon in sorted(summary['horizon'].unique()):
        print(f"\n{horizon}-Day Forward Returns:")
        print("-" * 40)
        horizon_df = summary[summary['horizon'] == horizon].head(args.top)
        for _, row in horizon_df.iterrows():
            corr = row['spearman']
            sign = '+' if corr > 0 else ''
            print(f"  {row['feature']:20s}  {sign}{corr:.4f}")

    print(f"\n{'='*50}")
    print("Screening complete!")


if __name__ == '__main__':
    main()
