#!/usr/bin/env python3
"""
Build All Bounce Profiles

Generate PLTR-style bounce profiles for ALL tickers in parallel.

Usage:
    python tools/build_bounce_profiles_all.py --years 10
    python tools/build_bounce_profiles_all.py --years 5 --max_workers 8

Options:
    --years: Window years (10 or 5)
    --max_workers: Parallel workers (default: 8)
"""

import argparse
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from bounce.profile_generator import generate_all_profiles


def main():
    parser = argparse.ArgumentParser(
        description="Generate bounce profiles for all tickers"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        choices=[5, 10],
        help="Window years (10 or 5)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Parallel workers",
    )

    args = parser.parse_args()

    # Load data
    reports_dir = PROJECT_ROOT / "reports" / "bounce"
    per_stock_path = reports_dir / f"week_down_then_bounce_per_stock_{args.years}y.csv"
    events_path = reports_dir / f"week_down_then_bounce_events_{args.years}y.parquet"

    if not per_stock_path.exists():
        print(f"ERROR: Per-stock summary not found: {per_stock_path}")
        print(f"Run: python tools/build_bounce_db.py --years {args.years}")
        sys.exit(1)

    if not events_path.exists():
        print(f"ERROR: Events file not found: {events_path}")
        print(f"Run: python tools/build_bounce_db.py --years {args.years}")
        sys.exit(1)

    print(f"Loading data for {args.years}Y profiles...")
    per_stock_df = pd.read_csv(per_stock_path)
    events_df = pd.read_parquet(events_path)

    output_dir = reports_dir / f"profiles_{args.years}y"

    print(f"Generating profiles to: {output_dir}")
    start_time = time.time()

    generated_files = generate_all_profiles(
        per_stock_df=per_stock_df,
        events_df=events_df,
        years=args.years,
        output_dir=output_dir,
        max_workers=args.max_workers,
        verbose=True,
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("PROFILE GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Profiles generated: {len(generated_files)}")
    print(f"  Elapsed time: {elapsed:.1f}s")
    print(f"  Output directory: {output_dir}")
    print(f"  Index: {output_dir / 'index.md'}")


if __name__ == "__main__":
    main()
