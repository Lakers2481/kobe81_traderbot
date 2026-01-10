#!/usr/bin/env python3
"""
Build Bounce Database

Main CLI tool to build the bounce analysis database for all 900 tickers.

Usage:
    python tools/build_bounce_db.py --years 10 --window 7 --max_streak 7
    python tools/build_bounce_db.py --years 5 --window 7 --max_streak 7  # Derives from 10Y

Options:
    --years: Number of years of history (10 or 5)
    --window: Forward recovery window in days (default: 7)
    --max_streak: Maximum streak level to analyze (default: 7)
    --max_workers: Parallel workers for data loading (default: 4)
    --validate_sample: Number of tickers to cross-validate (default: 25)
    --force: Force rebuild even if output exists
    --cap: Limit universe size (for testing)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from bounce.data_loader import load_universe_data, get_data_health_summary
from bounce.streak_analyzer import build_events_table
from bounce.event_table import (
    compute_overall_summary,
    compute_per_stock_summary,
    derive_5y_from_10y,
    generate_data_health_block,
)
from bounce.validation import run_all_validations
from bounce.profile_generator import generate_summary_report
from data.universe.loader import load_universe


def build_bounce_db(
    years: int = 10,
    window: int = 7,
    max_streak: int = 7,
    max_workers: int = 4,
    validate_sample: int = 25,
    force: bool = False,
    cap: int = None,
    output_dir: Path = None,
):
    """
    Build the bounce analysis database.

    Args:
        years: Number of years of history
        window: Forward recovery window
        max_streak: Maximum streak level
        max_workers: Parallel workers
        validate_sample: Number to validate
        force: Force rebuild
        cap: Limit universe size
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "reports" / "bounce"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file paths
    events_path = output_dir / f"week_down_then_bounce_events_{years}y.parquet"
    overall_path = output_dir / f"week_down_then_bounce_overall_{years}y.csv"
    per_stock_path = output_dir / f"week_down_then_bounce_per_stock_{years}y.csv"
    summary_path = output_dir / f"week_down_then_bounce_summary_{years}y.md"
    runlog_path = output_dir / f"week_down_then_bounce_runlog_{years}y.json"
    quality_path = output_dir / "data_quality_flags.csv"

    # Check if already exists
    if events_path.exists() and not force:
        print(f"Output already exists: {events_path}")
        print("Use --force to rebuild")
        return

    start_time = time.time()
    runlog = {
        "start_time": datetime.now().isoformat(),
        "years": years,
        "window": window,
        "max_streak": max_streak,
        "max_workers": max_workers,
        "validate_sample": validate_sample,
    }

    print("=" * 60)
    print(f"BUILDING BOUNCE DATABASE ({years}Y)")
    print("=" * 60)
    print()

    # Load universe
    print("Loading universe...")
    symbols = load_universe(cap=cap or 900)
    print(f"  Universe size: {len(symbols)}")
    runlog["universe_size"] = len(symbols)

    # Check if this is 5Y (derive from 10Y)
    if years == 5:
        # Load 10Y events and derive 5Y
        events_10y_path = output_dir / "week_down_then_bounce_events_10y.parquet"

        if not events_10y_path.exists():
            print("ERROR: 10Y events file not found. Run --years 10 first.")
            return

        print("Deriving 5Y from 10Y events...")
        events_10y = pd.read_parquet(events_10y_path)

        # Load quality report
        quality_10y_path = output_dir / "data_quality_flags.csv"
        quality_report = pd.read_csv(quality_10y_path) if quality_10y_path.exists() else None

        # Derive 5Y
        events_5y = derive_5y_from_10y(events_10y)

        print(f"  10Y events: {len(events_10y):,}")
        print(f"  5Y events: {len(events_5y):,}")

        # Compute summaries
        print("Computing summaries...")
        overall_summary = compute_overall_summary(events_5y)
        per_stock_summary = compute_per_stock_summary(
            events_5y,
            all_symbols=symbols,
            streak_levels=list(range(1, max_streak + 1)),
        )

        # Run validation
        print("Running validation checks...")
        validation_results = run_all_validations(events_5y, quality_report)

        # Save outputs
        print("Saving outputs...")
        events_5y.to_parquet(events_path, index=False)
        overall_summary.to_csv(overall_path, index=False)
        per_stock_summary.to_csv(per_stock_path, index=False)

        # Generate summary report
        generate_summary_report(
            overall_df=overall_summary,
            per_stock_df=per_stock_summary,
            quality_report=quality_report,
            years=years,
            output_path=summary_path,
        )

        elapsed = time.time() - start_time
        runlog["elapsed_seconds"] = elapsed
        runlog["events_count"] = len(events_5y)
        runlog["validation"] = validation_results
        runlog["end_time"] = datetime.now().isoformat()

        with open(runlog_path, 'w') as f:
            json.dump(runlog, f, indent=2, default=str)

        print()
        print("=" * 60)
        print("COMPLETE")
        print("=" * 60)
        print(f"  Events: {len(events_5y):,}")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Validation: {'PASSED' if validation_results['all_passed'] else 'FAILED'}")
        print(f"  Output: {output_dir}")
        return

    # Full 10Y build
    print()
    print("Step 1: Loading data for all tickers...")
    ticker_data, quality_report = load_universe_data(
        symbols=symbols,
        years=years,
        max_workers=max_workers,
        validate_sample=validate_sample,
        verbose=True,
    )

    data_health = get_data_health_summary(quality_report)
    runlog["data_health"] = data_health

    # Save quality report
    quality_report.to_csv(quality_path, index=False)

    print()
    print("Step 2: Building events table...")
    streak_levels = list(range(1, max_streak + 1))

    # Build ticker metadata for source tracking
    ticker_metadata = {}
    for _, row in quality_report.iterrows():
        ticker_metadata[row['symbol']] = {
            "source_used": row.get('source_used', 'polygon'),
            "adjustment_basis": "split_adjusted",
        }

    events_df = build_events_table(
        ticker_data=ticker_data,
        ticker_metadata=ticker_metadata,
        streak_levels=streak_levels,
        window=window,
        verbose=True,
    )

    runlog["events_count"] = len(events_df)

    print()
    print("Step 3: Computing summaries...")
    overall_summary = compute_overall_summary(events_df)
    per_stock_summary = compute_per_stock_summary(
        events_df,
        all_symbols=symbols,
        streak_levels=streak_levels,
    )

    print(f"  Overall summary: {len(overall_summary)} streak levels")
    print(f"  Per-stock summary: {len(per_stock_summary)} rows")

    print()
    print("Step 4: Running validation checks...")
    validation_results = run_all_validations(events_df, quality_report)
    runlog["validation"] = validation_results

    if validation_results["all_passed"]:
        print("  Lookahead bias check: PASSED")
        print("  Unit test: PASSED")
    else:
        print("  WARNING: Validation checks FAILED")
        print(f"  Details: {validation_results}")

    print()
    print("Step 5: Saving outputs...")

    # Save events parquet
    events_df.to_parquet(events_path, index=False)
    print(f"  Saved: {events_path}")

    # Save overall summary
    overall_summary.to_csv(overall_path, index=False)
    print(f"  Saved: {overall_path}")

    # Save per-stock summary
    per_stock_summary.to_csv(per_stock_path, index=False)
    print(f"  Saved: {per_stock_path}")

    # Generate summary report
    generate_summary_report(
        overall_df=overall_summary,
        per_stock_df=per_stock_summary,
        quality_report=quality_report,
        years=years,
        output_path=summary_path,
    )
    print(f"  Saved: {summary_path}")

    # Save runlog
    elapsed = time.time() - start_time
    runlog["elapsed_seconds"] = elapsed
    runlog["end_time"] = datetime.now().isoformat()

    with open(runlog_path, 'w') as f:
        json.dump(runlog, f, indent=2, default=str)
    print(f"  Saved: {runlog_path}")

    # Print summary
    print()
    print("=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print()
    print(f"Tickers attempted:  {len(symbols)}")
    print(f"Tickers loaded:     {len(ticker_data)}")
    print(f"Events generated:   {len(events_df):,}")
    print(f"Elapsed time:       {elapsed:.1f}s")
    print(f"Validation:         {'PASSED' if validation_results['all_passed'] else 'FAILED'}")
    print()
    print("Output files:")
    print(f"  {events_path}")
    print(f"  {overall_path}")
    print(f"  {per_stock_path}")
    print(f"  {summary_path}")
    print(f"  {runlog_path}")
    print(f"  {quality_path}")

    # Print DATA HEALTH block
    print()
    print(generate_data_health_block(quality_report, events_df, years))


def main():
    parser = argparse.ArgumentParser(
        description="Build bounce analysis database for all tickers"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="Number of years of history (10 or 5)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="Forward recovery window in days",
    )
    parser.add_argument(
        "--max_streak",
        type=int,
        default=7,
        help="Maximum streak level to analyze",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Parallel workers for data loading",
    )
    parser.add_argument(
        "--validate_sample",
        type=int,
        default=25,
        help="Number of tickers to cross-validate",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if output exists",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Limit universe size (for testing)",
    )

    args = parser.parse_args()

    build_bounce_db(
        years=args.years,
        window=args.window,
        max_streak=args.max_streak,
        max_workers=args.max_workers,
        validate_sample=args.validate_sample,
        force=args.force,
        cap=args.cap,
    )


if __name__ == "__main__":
    main()
