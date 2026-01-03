#!/usr/bin/env python3
"""
SCAN FOR STOCKS IN CONSECUTIVE DOWN STREAKS

Finds stocks that are currently in 3+ consecutive down days,
cross-referenced with historical bounce rate data.

These are potential mean-reversion candidates based on the
quant pattern analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Load historical bounce rate data
def load_bounce_rates() -> Dict[str, Dict]:
    """Load historical bounce rate analysis."""
    report_dir = Path("reports")

    # Find most recent analysis file
    json_files = sorted(report_dir.glob("quant_pattern_analysis_*.json"), reverse=True)
    if json_files:
        with open(json_files[0]) as f:
            data = json.load(f)
            # Build lookup by symbol
            bounce_lookup = {}
            for pattern in data.get("top_50_patterns", []):
                symbol = pattern["symbol"]
                if symbol not in bounce_lookup or pattern["bounce_5d"] > bounce_lookup[symbol]["bounce_5d"]:
                    bounce_lookup[symbol] = pattern
            return bounce_lookup
    return {}


def get_recent_data(symbol: str) -> Optional[pd.DataFrame]:
    """Get recent price data for a symbol."""
    # Check polygon_cache first (simple SYMBOL.csv format)
    polygon_cache = Path("data/polygon_cache")
    cached_file = polygon_cache / f"{symbol}.csv"

    if cached_file.exists():
        try:
            df = pd.read_csv(cached_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.tail(20)  # Last 20 days
        except Exception:
            pass

    # Check data/cache/polygon/ for date-ranged files
    polygon_api_cache = Path("data/cache/polygon")
    if polygon_api_cache.exists():
        # Look for most recent file
        for f in sorted(polygon_api_cache.glob(f"{symbol}_*.csv"), reverse=True):
            try:
                df = pd.read_csv(f, parse_dates=['timestamp'])
                return df.tail(20)
            except Exception:
                continue

    return None


def calculate_streak(df: pd.DataFrame) -> int:
    """Calculate current consecutive down day streak."""
    if df is None or len(df) < 2:
        return 0

    df = df.sort_values('timestamp').reset_index(drop=True)
    df['return'] = df['close'].pct_change()

    # Count consecutive down days from most recent
    streak = 0
    for i in range(len(df) - 1, 0, -1):
        if df.loc[i, 'return'] < 0:
            streak += 1
        else:
            break

    return streak


def main():
    print("=" * 80)
    print("SCAN: STOCKS CURRENTLY IN DOWN STREAKS")
    print("=" * 80)
    print(f"Time: {datetime.now()}")
    print()

    # Load universe
    universe_file = Path("data/universe/optionable_liquid_900.csv")
    if not universe_file.exists():
        print("ERROR: Universe file not found")
        return

    universe = pd.read_csv(universe_file)
    symbols = universe['symbol'].tolist()
    print(f"Scanning {len(symbols)} stocks...")
    print()

    # Load historical bounce rates
    bounce_rates = load_bounce_rates()
    print(f"Loaded bounce rates for {len(bounce_rates)} top performers")
    print()

    # Scan all stocks
    results = []

    for i, symbol in enumerate(symbols):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{len(symbols)}...")

        df = get_recent_data(symbol)
        if df is None or len(df) < 5:
            continue

        streak = calculate_streak(df)
        if streak >= 3:
            # Get latest price info
            latest = df.iloc[-1]
            prev_close = df.iloc[-2]['close'] if len(df) > 1 else latest['close']

            result = {
                'symbol': symbol,
                'streak': streak,
                'close': latest['close'],
                'change_pct': ((latest['close'] - prev_close) / prev_close) * 100,
                'date': latest['timestamp'].strftime('%Y-%m-%d') if pd.notna(latest['timestamp']) else 'N/A',
            }

            # Add historical bounce rate if available
            if symbol in bounce_rates:
                br = bounce_rates[symbol]
                result['hist_bounce_5d'] = br.get('bounce_5d', 0)
                result['hist_avg_move'] = br.get('avg_5d', 0)
                result['hist_samples'] = br.get('count', 0)
                result['tier'] = 1 if br.get('bounce_5d', 0) >= 85 else (2 if br.get('bounce_5d', 0) >= 75 else 3)
            else:
                result['hist_bounce_5d'] = None
                result['hist_avg_move'] = None
                result['hist_samples'] = None
                result['tier'] = None

            results.append(result)

    print()
    print("=" * 80)
    print(f"FOUND {len(results)} STOCKS IN 3+ DAY DOWN STREAKS")
    print("=" * 80)
    print()

    if not results:
        print("No stocks currently in 3+ consecutive down days.")
        return

    # Sort by streak length (descending), then by historical bounce rate
    results.sort(key=lambda x: (-x['streak'], -(x['hist_bounce_5d'] or 0)))

    # Separate by streak length
    streaks_7plus = [r for r in results if r['streak'] >= 7]
    streaks_6 = [r for r in results if r['streak'] == 6]
    streaks_5 = [r for r in results if r['streak'] == 5]
    streaks_4 = [r for r in results if r['streak'] == 4]
    streaks_3 = [r for r in results if r['streak'] == 3]

    # Print results
    def print_table(stocks: List[Dict], header: str):
        if not stocks:
            return
        print(f"\n{'='*60}")
        print(header)
        print(f"{'='*60}")
        print(f"| {'Symbol':6s} | {'Streak':6s} | {'Close':>8s} | {'Change':>7s} | {'Hist Bounce':>11s} | {'Tier':4s} |")
        print(f"|{'-'*8}|{'-'*8}|{'-'*10}|{'-'*9}|{'-'*13}|{'-'*6}|")
        for s in stocks[:20]:  # Top 20 per category
            bounce = f"{s['hist_bounce_5d']:.0f}%" if s['hist_bounce_5d'] else "N/A"
            tier = f"T{s['tier']}" if s['tier'] else "-"
            print(f"| {s['symbol']:6s} | {s['streak']:6d} | ${s['close']:>7.2f} | {s['change_pct']:>+6.1f}% | {bounce:>11s} | {tier:4s} |")

    print_table(streaks_7plus, "7+ CONSECUTIVE DOWN DAYS (EXTREME OVERSOLD)")
    print_table(streaks_6, "6 CONSECUTIVE DOWN DAYS (SEVERELY OVERSOLD)")
    print_table(streaks_5, "5 CONSECUTIVE DOWN DAYS (OVERSOLD)")
    print_table(streaks_4, "4 CONSECUTIVE DOWN DAYS")
    print_table(streaks_3[:30], "3 CONSECUTIVE DOWN DAYS (Top 30)")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"7+ down days: {len(streaks_7plus)} stocks")
    print(f"6 down days:  {len(streaks_6)} stocks")
    print(f"5 down days:  {len(streaks_5)} stocks")
    print(f"4 down days:  {len(streaks_4)} stocks")
    print(f"3 down days:  {len(streaks_3)} stocks")
    print(f"TOTAL:        {len(results)} stocks in down streaks")
    print()

    # High-confidence picks (Tier 1 & 2 with 5+ day streaks)
    high_confidence = [r for r in results if r['streak'] >= 5 and r.get('tier') in [1, 2]]
    if high_confidence:
        print("=" * 80)
        print("HIGH-CONFIDENCE BOUNCE CANDIDATES (5+ days + Tier 1/2 History)")
        print("=" * 80)
        for s in high_confidence[:10]:
            print(f"  {s['symbol']:6s}: {s['streak']} down days, {s['hist_bounce_5d']:.0f}% historical bounce, avg +{s['hist_avg_move']:.1f}%")

    # Save to CSV
    output_file = Path("logs/down_streak_scan.csv")
    output_file.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(output_file, index=False)
    print()
    print(f"Full results saved to: {output_file}")


if __name__ == "__main__":
    main()
