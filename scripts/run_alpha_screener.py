#!/usr/bin/env python3
"""
Alpha Screener CLI
==================

Run walk-forward alpha screening on the universe to discover profitable alphas.

Usage:
    # Screen all registered alphas
    python scripts/run_alpha_screener.py --universe data/universe/optionable_liquid_900.csv

    # Screen specific alphas
    python scripts/run_alpha_screener.py --alphas rsi2_oversold,momentum_breakout

    # Custom date range and splits
    python scripts/run_alpha_screener.py --start 2015-01-01 --end 2024-12-31 --train-days 252 --test-days 63

    # Show detailed leaderboard
    python scripts/run_alpha_screener.py --top 20 --verbose

    # Save results to CSV
    python scripts/run_alpha_screener.py --output reports/alpha_screener_results.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.screener import (
    AlphaScreener,
    ScreenerConfig,
    list_available_alphas,
)
from research.alphas import ALPHA_REGISTRY
from data.universe.loader import load_universe


def main():
    parser = argparse.ArgumentParser(
        description="Run walk-forward alpha screening to discover profitable alphas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Screen all alphas on 900-stock universe
    python scripts/run_alpha_screener.py --universe data/universe/optionable_liquid_900.csv

    # Screen specific alphas with custom settings
    python scripts/run_alpha_screener.py --alphas rsi2_oversold,ibs_rsi_oversold --train-days 504 --test-days 126

    # Output detailed results
    python scripts/run_alpha_screener.py --top 25 --verbose --output reports/alpha_results.csv
"""
    )

    # Data settings
    parser.add_argument(
        '--universe',
        type=str,
        default='data/universe/optionable_liquid_900.csv',
        help='Path to universe CSV file (default: data/universe/optionable_liquid_900.csv)',
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2015-01-01',
        help='Start date for screening (default: 2015-01-01)',
    )
    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date for screening (default: today)',
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='data/cache',
        help='Directory for cached price data (default: data/cache)',
    )

    # Screening settings
    parser.add_argument(
        '--alphas',
        type=str,
        default=None,
        help='Comma-separated list of alphas to screen (default: all registered alphas)',
    )
    parser.add_argument(
        '--train-days',
        type=int,
        default=252,
        help='Training window in trading days (default: 252 = 1 year)',
    )
    parser.add_argument(
        '--test-days',
        type=int,
        default=63,
        help='Test window in trading days (default: 63 = 1 quarter)',
    )
    parser.add_argument(
        '--min-symbols',
        type=int,
        default=10,
        help='Minimum symbols for cross-sectional analysis (default: 10)',
    )

    # Output settings
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top alphas to display (default: 10)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save results to CSV file',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output',
    )
    parser.add_argument(
        '--list-alphas',
        action='store_true',
        help='List all available alphas and exit',
    )

    args = parser.parse_args()

    # List alphas mode
    if args.list_alphas:
        print("\n" + "=" * 60)
        print("AVAILABLE ALPHAS")
        print("=" * 60)

        available = list_available_alphas()
        for alpha in available:
            meta = ALPHA_REGISTRY.get(alpha, {})
            category = meta.get('category', 'unknown')
            desc = meta.get('description', 'No description')
            hypothesis = meta.get('hypothesis', '')

            print(f"\n{alpha} ({category})")
            print(f"  {desc}")
            if hypothesis:
                print(f"  Hypothesis: {hypothesis}")

        print(f"\n{'=' * 60}")
        print(f"Total: {len(available)} alphas")
        return

    # Validate universe
    universe_path = Path(args.universe)
    if not universe_path.exists():
        print(f"ERROR: Universe file not found: {universe_path}")
        print("Run: python scripts/build_universe_polygon.py first")
        sys.exit(1)

    # Load universe
    symbols = load_universe(universe_path)
    print(f"\nLoaded {len(symbols)} symbols from {universe_path}")

    # Set end date
    end_date = args.end or datetime.now().strftime('%Y-%m-%d')

    # Parse alpha list
    alpha_names: Optional[List[str]] = None
    if args.alphas:
        alpha_names = [a.strip() for a in args.alphas.split(',')]

        # Validate alpha names
        available = list_available_alphas()
        invalid = [a for a in alpha_names if a not in available]
        if invalid:
            print(f"ERROR: Unknown alphas: {invalid}")
            print(f"Available: {available}")
            sys.exit(1)

    # Create screener config
    config = ScreenerConfig(
        train_window_days=args.train_days,
        test_window_days=args.test_days,
        min_symbols=args.min_symbols,
        min_samples=50,
        forward_days=5,
    )

    print(f"\n{'=' * 60}")
    print("ALPHA SCREENER")
    print("=" * 60)
    print(f"Universe: {len(symbols)} symbols")
    print(f"Date range: {args.start} to {end_date}")
    print(f"Train window: {args.train_days} days")
    print(f"Test window: {args.test_days} days")
    print(f"Alphas: {alpha_names or 'ALL'}")
    print("=" * 60)

    # Create screener
    screener = AlphaScreener(
        symbols=symbols,
        start_date=args.start,
        end_date=end_date,
        cache_dir=Path(args.cache_dir),
        config=config,
    )

    # Run screening
    print("\nRunning alpha screening (this may take a while)...")
    results = screener.screen_alphas(alpha_names=alpha_names)

    # Generate leaderboard
    leaderboard = screener.get_leaderboard(top_n=args.top)

    if leaderboard.empty:
        print("\nNo valid alpha results found.")
        print("This could be due to insufficient data or overly strict filters.")
        sys.exit(0)

    # Display results
    print(f"\n{'=' * 60}")
    print(f"TOP {min(args.top, len(leaderboard))} ALPHAS")
    print("=" * 60)

    for idx, row in leaderboard.iterrows():
        rank = idx + 1
        alpha_name = row['alpha']
        mean_ret = row['mean_return'] * 100  # Convert to percentage
        sharpe = row['sharpe']
        win_rate = row['win_rate'] * 100
        trades = row['n_trades']

        print(f"\n{rank}. {alpha_name}")
        print(f"   Sharpe: {sharpe:.2f} | Win Rate: {win_rate:.1f}% | Mean Return: {mean_ret:.2f}%")
        print(f"   Trades: {trades}")

        if args.verbose:
            meta = ALPHA_REGISTRY.get(alpha_name, {})
            if meta:
                print(f"   Category: {meta.get('category', 'unknown')}")
                print(f"   Hypothesis: {meta.get('hypothesis', 'N/A')}")

    print(f"\n{'=' * 60}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    # Save detailed results
    results_dir = ROOT / 'reports' / 'alpha_screening'
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f'screening_{timestamp}.json'

    # Convert results to JSON-serializable format
    json_results = {
        'screened_at': datetime.now().isoformat(),
        'config': {
            'universe': str(args.universe),
            'start': args.start,
            'end': end_date,
            'train_days': args.train_days,
            'test_days': args.test_days,
            'symbols_count': len(symbols),
        },
        'leaderboard': leaderboard.to_dict(orient='records'),
        'results_count': len(results),
    }

    results_path.write_text(json.dumps(json_results, indent=2, default=str))
    print(f"Detailed results: {results_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Alphas screened: {len(results)}")
    print(f"Profitable alphas (Sharpe > 0.5): {len(leaderboard[leaderboard['sharpe'] > 0.5])}")
    print(f"High-confidence (Sharpe > 1.0): {len(leaderboard[leaderboard['sharpe'] > 1.0])}")

    if len(leaderboard) > 0:
        best = leaderboard.iloc[0]
        print(f"\nBest alpha: {best['alpha']}")
        print(f"  Sharpe: {best['sharpe']:.2f}")
        print(f"  Win Rate: {best['win_rate']*100:.1f}%")


if __name__ == '__main__':
    main()
