#!/usr/bin/env python3
"""
Freeze Equities EOD Data
=========================

Downloads daily OHLCV data and freezes it into the data lake.

Once frozen, data NEVER changes - providing reproducible backtesting.

Providers:
- stooq (primary, free, recommended)
- yfinance (fallback, unofficial)

Usage:
    # Windows:
    python scripts\\freeze_equities_eod.py ^
        --universe data\\universe\\optionable_liquid_900.csv ^
        --start 2015-01-01 --end 2025-12-31 ^
        --provider stooq ^
        --out data\\lake

    # Linux/macOS:
    python scripts/freeze_equities_eod.py \\
        --universe data/universe/optionable_liquid_900.csv \\
        --start 2015-01-01 --end 2025-12-31 \\
        --provider stooq \\
        --out data/lake

    # With symbol limit (for testing):
    python scripts/freeze_equities_eod.py --universe data/universe/sp500.csv --limit 10

    # Use Yahoo Finance fallback:
    python scripts/freeze_equities_eod.py --provider yfinance ...
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def progress_bar(current: int, total: int, symbol: str = ""):
    """Simple progress display."""
    pct = current / total * 100
    bar_len = 40
    filled = int(bar_len * current / total)
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\r[{bar}] {pct:5.1f}% ({current}/{total}) {symbol:12}", end="", flush=True)
    if current == total:
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Freeze equities EOD data into the data lake",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Freeze 900+ stock universe with Stooq (free)
    python scripts/freeze_equities_eod.py \\
        --universe data/universe/optionable_liquid_900.csv \\
        --start 2015-01-01 --end 2025-12-31 \\
        --provider stooq

    # Quick test with 10 symbols
    python scripts/freeze_equities_eod.py \\
        --universe data/universe/optionable_liquid_900.csv \\
        --limit 10

    # Use Yahoo Finance fallback
    python scripts/freeze_equities_eod.py --provider yfinance ...

Output:
    Creates a frozen dataset in data/lake/<dataset_id>/
    Generates manifest in data/manifests/<dataset_id>.json
    Dataset ID is deterministic based on inputs (same inputs = same ID)
"""
    )

    parser.add_argument(
        '--universe',
        required=True,
        help='Path to universe CSV file with symbol column',
    )
    parser.add_argument(
        '--start',
        default='2015-01-01',
        help='Start date (YYYY-MM-DD, default: 2015-01-01)',
    )
    parser.add_argument(
        '--end',
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD, default: today)',
    )
    parser.add_argument(
        '--provider',
        choices=['stooq', 'yfinance'],
        default='stooq',
        help='Data provider (default: stooq)',
    )
    parser.add_argument(
        '--out',
        default='data/lake',
        help='Output directory for frozen data (default: data/lake)',
    )
    parser.add_argument(
        '--manifest-dir',
        default='data/manifests',
        help='Manifest directory (default: data/manifests)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of symbols (for testing)',
    )
    parser.add_argument(
        '--symbol-column',
        default='symbol',
        help='Column name containing symbols (default: symbol)',
    )
    parser.add_argument(
        '--partition-by',
        choices=['symbol', 'year', 'none'],
        default='none',
        help='Partition strategy (default: none = single file)',
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=0.5,
        help='Seconds between requests (default: 0.5)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually fetching',
    )

    args = parser.parse_args()

    universe_path = Path(args.universe)
    lake_dir = Path(args.out)
    manifest_dir = Path(args.manifest_dir)

    # Validate universe file
    if not universe_path.exists():
        print(f"ERROR: Universe file not found: {universe_path}")
        sys.exit(1)

    # Read universe to show summary
    universe_df = pd.read_csv(universe_path)

    # Find symbol column
    symbol_col = args.symbol_column
    if symbol_col not in universe_df.columns:
        for alt in ['Symbol', 'ticker', 'Ticker', 'SYMBOL']:
            if alt in universe_df.columns:
                symbol_col = alt
                break
        else:
            print(f"ERROR: Symbol column '{args.symbol_column}' not found")
            print(f"Available columns: {list(universe_df.columns)}")
            sys.exit(1)

    symbols = universe_df[symbol_col].unique().tolist()
    if args.limit:
        symbols = symbols[:args.limit]

    print("=" * 60)
    print("FREEZE EQUITIES EOD DATA")
    print("=" * 60)
    print(f"Universe:   {universe_path}")
    print(f"Symbols:    {len(symbols)}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Provider:   {args.provider}")
    print(f"Output:     {lake_dir}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would fetch data for these symbols:")
        for i, sym in enumerate(symbols[:20]):
            print(f"  {sym}")
        if len(symbols) > 20:
            print(f"  ... and {len(symbols) - 20} more")
        print("\nRun without --dry-run to actually fetch and freeze data.")
        sys.exit(0)

    # Check if dataset already exists
    from data.lake import compute_dataset_id, dataset_exists

    dataset_id = compute_dataset_id(
        provider=args.provider,
        timeframe='1d',
        start_date=args.start,
        end_date=args.end,
        universe_path=universe_path,
    )

    if dataset_exists(dataset_id, manifest_dir):
        print(f"\nDataset already exists: {dataset_id}")
        print("Immutable datasets cannot be overwritten.")
        print("To create a new dataset, change inputs (dates, universe, etc.)")
        sys.exit(0)

    print(f"\nDataset ID will be: {dataset_id}")
    print(f"\nFetching data from {args.provider}...")

    # Create provider
    if args.provider == 'stooq':
        from data.providers.stooq_eod import StooqEODProvider
        provider = StooqEODProvider(rate_limit_delay=args.rate_limit)
    else:
        from data.providers.yfinance_eod import YFinanceEODProvider
        provider = YFinanceEODProvider(rate_limit_delay=args.rate_limit, warn_unofficial=True)

    # Fetch data
    df = provider.fetch_universe(
        symbols=symbols,
        start=args.start,
        end=args.end,
        progress_callback=progress_bar,
    )

    if df.empty:
        print("\nERROR: No data fetched")
        sys.exit(1)

    print(f"\nFetched {len(df):,} rows for {df['symbol'].nunique()} symbols")

    # Summary stats
    print("\nData Summary:")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Symbols:    {df['symbol'].nunique()}")
    print(f"  Rows:       {len(df):,}")

    # Freeze to lake
    print(f"\nFreezing to data lake...")

    from data.lake import LakeWriter

    writer = LakeWriter(
        lake_dir=lake_dir,
        manifest_dir=manifest_dir,
    )

    partition = None if args.partition_by == 'none' else args.partition_by

    manifest = writer.freeze_dataframe(
        df=df,
        provider=args.provider,
        timeframe='1d',
        universe_path=universe_path,
        start_date=args.start,
        end_date=args.end,
        partition_by=partition,
    )

    print("\n" + "=" * 60)
    print("FREEZE COMPLETE")
    print("=" * 60)
    print(f"Dataset ID:  {manifest.dataset_id}")
    print(f"Total rows:  {manifest.total_rows:,}")
    print(f"Symbols:     {manifest.total_symbols}")
    print(f"Files:       {len(manifest.files)}")
    print(f"Lake dir:    {lake_dir / manifest.dataset_id}")
    print(f"Manifest:    {manifest_dir / (manifest.dataset_id + '.json')}")
    print("=" * 60)

    print("\nTo use this dataset in backtests:")
    print(f"  --dataset-id {manifest.dataset_id}")

    print("\nTo validate the frozen data:")
    print(f"  python scripts/validate_lake.py --dataset-id {manifest.dataset_id}")


if __name__ == '__main__':
    main()
