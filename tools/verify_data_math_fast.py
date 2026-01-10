"""
FAST DATA & MATH VERIFIER
==========================

Verifies claims on a SUBSET of symbols for quick validation.
Full verification runs on all 800 symbols.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.providers.yfinance_eod import YFinanceEODProvider


def verify_markov_fast(symbols: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict]:
    """Fast Markov 5-down pattern verification on subset."""
    print(f"\n{'='*80}")
    print("FAST MARKOV 5-DOWN PATTERN VERIFICATION")
    print(f"{'='*80}")
    print(f"Symbols: {len(symbols)}")
    print(f"Period: {start_date} to {end_date}\n")

    provider = YFinanceEODProvider(warn_unofficial=False)
    all_instances = []

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Fetching {symbol}...", end=' ')

        try:
            df = provider.fetch_symbol(symbol, start_date, end_date)
            if df is None or len(df) == 0:
                print("SKIP (no data)")
                continue

            df = df.set_index('timestamp')
            print(f"OK ({len(df)} bars)")

            # Calculate returns
            df['return'] = df['close'].pct_change()
            df['is_down'] = (df['return'] < 0).astype(int)

            # Count consecutive down days
            df['down_streak'] = 0
            streak = 0
            for idx in range(len(df)):
                if df['is_down'].iloc[idx] == 1:
                    streak += 1
                    df.iloc[idx, df.columns.get_loc('down_streak')] = streak
                else:
                    streak = 0

            # Find 5-down patterns
            five_down_mask = (df['down_streak'] == 5)
            five_down_dates = df[five_down_mask].index.tolist()

            if len(five_down_dates) == 0:
                continue

            # Check next day return for each pattern
            for date in five_down_dates:
                date_loc = df.index.get_loc(date)
                if date_loc + 1 < len(df):
                    next_day_return = df['return'].iloc[date_loc + 1]
                    next_day_up = 1 if next_day_return > 0 else 0

                    all_instances.append({
                        'symbol': symbol,
                        'date': date,
                        'next_day_return': next_day_return,
                        'next_day_up': next_day_up,
                    })

        except Exception as e:
            print(f"FAIL ({e})")

    # Create DataFrame
    instances_df = pd.DataFrame(all_instances)

    if len(instances_df) > 0:
        total_instances = len(instances_df)
        total_up = instances_df['next_day_up'].sum()
        up_probability = total_up / total_instances

        # 95% CI
        z = 1.96
        se = np.sqrt(up_probability * (1 - up_probability) / total_instances)
        ci_lower = up_probability - z * se
        ci_upper = up_probability + z * se

        summary = {
            'total_instances': total_instances,
            'next_day_up': total_up,
            'next_day_down': total_instances - total_up,
            'up_probability': up_probability,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'symbols_with_pattern': instances_df['symbol'].nunique(),
        }
    else:
        summary = {
            'total_instances': 0,
            'next_day_up': 0,
            'next_day_down': 0,
            'up_probability': None,
            'ci_95_lower': None,
            'ci_95_upper': None,
            'symbols_with_pattern': 0,
        }

    print(f"\n{'-'*80}")
    print("MARKOV 5-DOWN RESULTS")
    print(f"{'-'*80}")
    print(f"Total instances: {summary['total_instances']}")
    print(f"Next day up: {summary['next_day_up']}")
    print(f"Next day down: {summary['next_day_down']}")
    if summary['up_probability']:
        print(f"Up probability: {summary['up_probability']:.1%}")
        print(f"95% CI: [{summary['ci_95_lower']:.1%}, {summary['ci_95_upper']:.1%}]")
    print(f"Symbols with pattern: {summary['symbols_with_pattern']}")
    print(f"{'-'*80}\n")

    return instances_df, summary


def main():
    """Run fast verification on top 50 symbols."""
    print(f"{'='*80}")
    print("FAST DATA & MATH VERIFIER")
    print(f"{'='*80}\n")

    # Load universe
    universe_path = project_root / "data" / "universe" / "optionable_liquid_800.csv"
    df = pd.read_csv(universe_path)
    symbols = df['symbol'].tolist()[:50]  # Top 50 only

    print(f"Testing on first {len(symbols)} symbols from universe")
    print(f"Full verification runs on all 800 symbols\n")

    # Verify Markov pattern
    instances_df, summary = verify_markov_fast(
        symbols=symbols,
        start_date="2015-01-01",
        end_date="2024-12-31"
    )

    # Save results
    output_dir = project_root / "data" / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "fast_markov_instances.csv"
    instances_df.to_csv(csv_path, index=False)
    print(f"[SAVED] Results saved to {csv_path}\n")

    # Compare to claim
    print(f"{'='*80}")
    print("CLAIM VERIFICATION")
    print(f"{'='*80}")
    print(f"Claimed: 64.0% up probability")
    if summary['up_probability']:
        print(f"Actual (50 symbols): {summary['up_probability']:.1%}")
        diff = abs(0.64 - summary['up_probability'])
        print(f"Difference: {diff:.1%}")
        if diff < 0.05:
            print("Result: VERIFIED (within 5%)")
        elif diff < 0.10:
            print("Result: PARTIALLY VERIFIED (within 10%)")
        else:
            print("Result: NOT VERIFIED (>10% difference)")
    else:
        print("Result: NO DATA")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
