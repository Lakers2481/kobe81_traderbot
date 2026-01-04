#!/usr/bin/env python3
"""
Freeze Crypto OHLCV Data
=========================

Downloads crypto OHLCV data (hourly) and freezes it into the data lake.

Stores hourly data for maximum resolution; can resample to daily for swing trading.

Usage:
    # Windows:
    python scripts\\freeze_crypto_ohlcv.py ^
        --symbols BTCUSDT,ETHUSDT ^
        --start 2020-01-01 --end 2025-12-31 ^
        --timeframe 1h ^
        --out data\\lake

    # Linux/macOS:
    python scripts/freeze_crypto_ohlcv.py \\
        --symbols BTCUSDT,ETHUSDT,SOLUSDT \\
        --start 2020-01-01 --end 2025-12-31 \\
        --timeframe 1h \\
        --out data/lake

    # Top 20 by market cap:
    python scripts/freeze_crypto_ohlcv.py --symbols auto --top 20

    # Daily timeframe:
    python scripts/freeze_crypto_ohlcv.py --timeframe 1d ...
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default popular crypto pairs
DEFAULT_PAIRS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
    'LINKUSDT', 'SHIBUSDT', 'LTCUSDT', 'ATOMUSDT', 'UNIUSDT',
    'XLMUSDT', 'FILUSDT', 'NEARUSDT', 'APTUSDT', 'ARBUSDT',
]


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
        description="Freeze crypto OHLCV data into the data lake",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Freeze BTC and ETH hourly data
    python scripts/freeze_crypto_ohlcv.py \\
        --symbols BTCUSDT,ETHUSDT \\
        --start 2020-01-01 --end 2025-12-31

    # Freeze top 20 crypto pairs
    python scripts/freeze_crypto_ohlcv.py --symbols auto --top 20

    # Freeze daily data instead of hourly
    python scripts/freeze_crypto_ohlcv.py --timeframe 1d ...

    # Partition by symbol for large datasets
    python scripts/freeze_crypto_ohlcv.py --partition-by symbol ...

Notes:
    - Data is from Binance public API (free, no API key)
    - Hourly data recommended for swing trading (resample to daily)
    - USDT pairs are most liquid
"""
    )

    parser.add_argument(
        '--symbols',
        default='BTCUSDT,ETHUSDT',
        help='Comma-separated symbols or "auto" for defaults (default: BTCUSDT,ETHUSDT)',
    )
    parser.add_argument(
        '--top',
        type=int,
        default=20,
        help='Number of top symbols when using --symbols auto (default: 20)',
    )
    parser.add_argument(
        '--start',
        default='2020-01-01',
        help='Start date (YYYY-MM-DD, default: 2020-01-01)',
    )
    parser.add_argument(
        '--end',
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD, default: today)',
    )
    parser.add_argument(
        '--timeframe',
        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
        default='1h',
        help='Timeframe (default: 1h)',
    )
    parser.add_argument(
        '--out',
        default='data/lake',
        help='Output directory (default: data/lake)',
    )
    parser.add_argument(
        '--manifest-dir',
        default='data/manifests',
        help='Manifest directory (default: data/manifests)',
    )
    parser.add_argument(
        '--partition-by',
        choices=['symbol', 'year', 'none'],
        default='symbol',
        help='Partition strategy (default: symbol)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without fetching',
    )

    args = parser.parse_args()

    lake_dir = Path(args.out)
    manifest_dir = Path(args.manifest_dir)

    # Determine symbols
    if args.symbols.lower() == 'auto':
        symbols = DEFAULT_PAIRS[:args.top]
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]

    print("=" * 60)
    print("FREEZE CRYPTO OHLCV DATA")
    print("=" * 60)
    print(f"Symbols:    {len(symbols)}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Timeframe:  {args.timeframe}")
    print(f"Output:     {lake_dir}")
    print(f"Pairs:      {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would fetch data for:")
        for sym in symbols:
            print(f"  {sym}")
        print("\nRun without --dry-run to actually fetch and freeze data.")
        sys.exit(0)

    # Create a pseudo-universe file for dataset_id computation
    # (crypto doesn't use a file-based universe, so we create a temp one)
    universe_dir = Path("data/universe")
    universe_dir.mkdir(parents=True, exist_ok=True)
    universe_path = universe_dir / f"crypto_{len(symbols)}_pairs.csv"

    # Write universe file
    pd.DataFrame({'symbol': symbols}).to_csv(universe_path, index=False)

    # Check if dataset already exists
    from data.lake import compute_dataset_id, dataset_exists

    dataset_id = compute_dataset_id(
        provider='binance',
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        universe_path=universe_path,
    )

    if dataset_exists(dataset_id, manifest_dir):
        print(f"\nDataset already exists: {dataset_id}")
        print("Immutable datasets cannot be overwritten.")
        sys.exit(0)

    print(f"\nDataset ID will be: {dataset_id}")
    print("\nFetching data from Binance...")

    # Fetch data
    from data.providers.binance_klines import BinanceKlinesProvider

    provider = BinanceKlinesProvider()
    df = provider.fetch_universe(
        symbols=symbols,
        start=args.start,
        end=args.end,
        timeframe=args.timeframe,
        progress_callback=progress_bar,
    )

    if df.empty:
        print("\nERROR: No data fetched")
        sys.exit(1)

    print(f"\nFetched {len(df):,} rows for {df['symbol'].nunique()} pairs")

    # Summary
    print("\nData Summary:")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Pairs:      {df['symbol'].nunique()}")
    print(f"  Rows:       {len(df):,}")

    # Freeze to lake
    print("\nFreezing to data lake...")

    from data.lake import LakeWriter

    writer = LakeWriter(
        lake_dir=lake_dir,
        manifest_dir=manifest_dir,
    )

    partition = None if args.partition_by == 'none' else args.partition_by

    manifest = writer.freeze_dataframe(
        df=df,
        provider='binance',
        timeframe=args.timeframe,
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
    print(f"Pairs:       {manifest.total_symbols}")
    print(f"Files:       {len(manifest.files)}")
    print(f"Lake dir:    {lake_dir / manifest.dataset_id}")
    print("=" * 60)

    print("\nTo use this dataset:")
    print(f"  --dataset-id {manifest.dataset_id}")

    print("\nTo resample hourly to daily in code:")
    print("  df = df.set_index('timestamp').resample('D').agg({...})")


if __name__ == '__main__':
    main()
