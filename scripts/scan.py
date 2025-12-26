#!/usr/bin/env python3
"""
Kobe Daily Stock Scanner

Scans the universe for trading signals using RSI-2 and IBS strategies.

Features:
- Loads universe from data/universe/optionable_liquid_final.csv
- Fetches latest EOD data via Polygon
- Runs both RSI-2 and IBS strategies (or filter by --strategy)
- Outputs signals to stdout and logs/signals.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.env_loader import load_env
from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.universe.loader import load_universe
from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
from strategies.ibs.strategy import IBSStrategy


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "optionable_liquid_final.csv"
SIGNALS_LOG = ROOT / "logs" / "signals.jsonl"
CACHE_DIR = ROOT / "data" / "cache"
LOOKBACK_DAYS = 300  # Need 200+ days for SMA(200) + buffer


# -----------------------------------------------------------------------------
# Scanner functions
# -----------------------------------------------------------------------------
def fetch_symbol_data(
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch EOD data for a single symbol."""
    try:
        df = fetch_daily_bars_polygon(
            symbol=symbol,
            start=start_date,
            end=end_date,
            cache_dir=cache_dir,
        )
        return df
    except Exception as e:
        print(f"  [WARN] Failed to fetch {symbol}: {e}", file=sys.stderr)
        return pd.DataFrame()


def run_strategies(
    data: pd.DataFrame,
    strategies: List[str],
) -> pd.DataFrame:
    """Run specified strategies and return combined signals."""
    all_signals: List[pd.DataFrame] = []

    if "rsi2" in strategies or "all" in strategies:
        try:
            rsi2_strat = ConnorsRSI2Strategy()
            rsi2_signals = rsi2_strat.generate_signals(data)
            if not rsi2_signals.empty:
                rsi2_signals["strategy"] = "rsi2"
                all_signals.append(rsi2_signals)
        except Exception as e:
            print(f"  [WARN] RSI-2 strategy error: {e}", file=sys.stderr)

    if "ibs" in strategies or "all" in strategies:
        try:
            ibs_strat = IBSStrategy()
            ibs_signals = ibs_strat.generate_signals(data)
            if not ibs_signals.empty:
                ibs_signals["strategy"] = "ibs"
                all_signals.append(ibs_signals)
        except Exception as e:
            print(f"  [WARN] IBS strategy error: {e}", file=sys.stderr)

    if all_signals:
        return pd.concat(all_signals, ignore_index=True)
    return pd.DataFrame()


def log_signals(signals: pd.DataFrame, scan_id: str) -> None:
    """Append signals to JSONL log file."""
    SIGNALS_LOG.parent.mkdir(parents=True, exist_ok=True)

    with SIGNALS_LOG.open("a", encoding="utf-8") as f:
        for _, row in signals.iterrows():
            record = {
                "ts": datetime.utcnow().isoformat(),
                "scan_id": scan_id,
                "event": "signal",
                **{k: v for k, v in row.items() if pd.notna(v)},
            }
            # Convert Timestamp to string
            for k, v in record.items():
                if isinstance(v, pd.Timestamp):
                    record[k] = v.isoformat()
            f.write(json.dumps(record, default=str) + "\n")


def format_signal_row(row: pd.Series) -> str:
    """Format a single signal for display."""
    parts = [
        f"{row.get('strategy', '?'):>5}",
        f"{row.get('symbol', '?'):<6}",
        f"{row.get('side', '?'):<6}",
        f"@ ${row.get('entry_price', 0):>8.2f}",
        f"stop ${row.get('stop_loss', 0):>8.2f}",
    ]
    reason = row.get("reason", "")
    if reason:
        parts.append(f"| {reason}")
    return " ".join(parts)


def print_signals_table(signals: pd.DataFrame) -> None:
    """Print signals in a formatted table."""
    if signals.empty:
        print("\n  No signals generated.")
        return

    print("\n  SIGNALS")
    print("  " + "-" * 76)
    print(f"  {'STRAT':>5} {'SYMBOL':<6} {'SIDE':<6} {'ENTRY':>12} {'STOP':>12} | REASON")
    print("  " + "-" * 76)

    for _, row in signals.iterrows():
        print("  " + format_signal_row(row))

    print("  " + "-" * 76)
    print(f"  Total: {len(signals)} signal(s)")

    # Summary by strategy
    if "strategy" in signals.columns:
        by_strat = signals.groupby("strategy").size()
        print(f"  By strategy: {dict(by_strat)}")

    # Summary by side
    if "side" in signals.columns:
        by_side = signals.groupby("side").size()
        print(f"  By side: {dict(by_side)}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Kobe Daily Stock Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/scan.py                        # Scan all strategies
  python scripts/scan.py --strategy rsi2        # Only RSI-2 signals
  python scripts/scan.py --strategy ibs         # Only IBS signals
  python scripts/scan.py --cap 50               # Scan first 50 symbols
  python scripts/scan.py --json                 # Output as JSON
        """,
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file",
    )
    ap.add_argument(
        "--universe",
        type=str,
        default=str(DEFAULT_UNIVERSE),
        help="Path to universe CSV file",
    )
    ap.add_argument(
        "--strategy",
        type=str,
        choices=["rsi2", "ibs", "all"],
        default="all",
        help="Strategy to run (default: all)",
    )
    ap.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Limit number of symbols to scan",
    )
    ap.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD), default: lookback from today",
    )
    ap.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), default: today",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output signals as JSON",
    )
    ap.add_argument(
        "--no-log",
        action="store_true",
        help="Skip writing to signals.jsonl",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    args = ap.parse_args()

    # Load environment
    dotenv_path = Path(args.dotenv)
    if dotenv_path.exists():
        loaded = load_env(dotenv_path)
        if args.verbose:
            print(f"Loaded {len(loaded)} env vars from {dotenv_path}")
    else:
        if args.verbose:
            print(f"Warning: dotenv file not found: {dotenv_path}", file=sys.stderr)

    # Check Polygon API key
    if not os.getenv("POLYGON_API_KEY"):
        print("Error: POLYGON_API_KEY not set. Please provide via --dotenv.", file=sys.stderr)
        return 1

    # Load universe
    universe_path = Path(args.universe)
    if not universe_path.exists():
        print(f"Error: Universe file not found: {universe_path}", file=sys.stderr)
        return 1

    symbols = load_universe(universe_path, cap=args.cap)
    if not symbols:
        print(f"Error: No symbols loaded from {universe_path}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Loaded {len(symbols)} symbols from {universe_path}")

    # Determine date range
    end_date = args.end or datetime.utcnow().date().isoformat()
    if args.start:
        start_date = args.start
    else:
        end_dt = datetime.fromisoformat(end_date)
        start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
        start_date = start_dt.date().isoformat()

    if args.verbose:
        print(f"Date range: {start_date} to {end_date}")

    # Determine strategies to run
    strategies = [args.strategy] if args.strategy != "all" else ["rsi2", "ibs"]
    if args.verbose:
        print(f"Strategies: {strategies}")

    # Scan ID for logging
    scan_id = f"SCAN_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # Fetch data and run strategies
    print(f"\nKobe Scanner - {scan_id}")
    print(f"Scanning {len(symbols)} symbols for {', '.join(strategies)} signals...")
    print("-" * 60)

    all_data: List[pd.DataFrame] = []
    success_count = 0
    fail_count = 0

    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for i, symbol in enumerate(symbols, 1):
        if args.verbose:
            print(f"  [{i}/{len(symbols)}] Fetching {symbol}...", end=" ")

        df = fetch_symbol_data(symbol, start_date, end_date, CACHE_DIR)
        if not df.empty and len(df) > 0:
            all_data.append(df)
            success_count += 1
            if args.verbose:
                print(f"{len(df)} bars")
        else:
            fail_count += 1
            if args.verbose:
                print("SKIP (no data)")

    print(f"\nFetched: {success_count} symbols, skipped: {fail_count}")

    if not all_data:
        print("Error: No data fetched for any symbols.", file=sys.stderr)
        return 1

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Total bars: {len(combined):,}")

    # Run strategies
    print("\nRunning strategies...")
    signals = run_strategies(combined, strategies)

    # Output results
    if args.json:
        if not signals.empty:
            # Convert to JSON-serializable format
            output = []
            for _, row in signals.iterrows():
                rec = {k: v for k, v in row.items() if pd.notna(v)}
                for k, v in rec.items():
                    if isinstance(v, pd.Timestamp):
                        rec[k] = v.isoformat()
                output.append(rec)
            print(json.dumps(output, indent=2, default=str))
        else:
            print("[]")
    else:
        print_signals_table(signals)

    # Log signals
    if not args.no_log and not signals.empty:
        log_signals(signals, scan_id)
        print(f"\nSignals logged to: {SIGNALS_LOG}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Scan complete: {len(signals)} signal(s) generated")

    return 0


if __name__ == "__main__":
    sys.exit(main())
