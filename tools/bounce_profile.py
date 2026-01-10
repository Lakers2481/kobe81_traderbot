#!/usr/bin/env python3
"""
Bounce Profile Generator (Single Ticker)

Generate PLTR-style bounce profile for a single ticker.

Usage:
    python tools/bounce_profile.py --ticker PLTR --years 10
    python tools/bounce_profile.py --ticker TSLA --years 5

Options:
    --ticker: Stock ticker symbol
    --years: Window years (10 or 5)
    --output: Output file path (default: stdout)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from bounce.profile_generator import generate_ticker_profile


def main():
    parser = argparse.ArgumentParser(
        description="Generate bounce profile for a single ticker"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        choices=[5, 10],
        help="Window years (10 or 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: reports/bounce/profiles_{years}y/{ticker}_bounce_{years}y.md)",
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

    per_stock_df = pd.read_csv(per_stock_path)
    events_df = pd.read_parquet(events_path)

    # Check if ticker exists
    ticker = args.ticker.upper()
    if ticker not in per_stock_df['ticker'].values:
        print(f"ERROR: Ticker '{ticker}' not found in database")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_dir = Path(args.output).parent
        Path(args.output)
    else:
        output_dir = reports_dir / f"profiles_{args.years}y"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate profile
    output_path = generate_ticker_profile(
        ticker=ticker,
        per_stock_df=per_stock_df,
        events_df=events_df,
        years=args.years,
        output_dir=output_dir,
    )

    print(f"Generated: {output_path}")

    # Also print to stdout
    print()
    print("=" * 60)
    print(output_path.read_text())


if __name__ == "__main__":
    main()
