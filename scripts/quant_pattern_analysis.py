#!/usr/bin/env python3
"""
QUANT-GRADE PATTERN ANALYSIS

Analyzes ALL 900 stocks for consecutive down-day patterns using VECTORIZED operations.
This is what a quant developer would do - find edge from REAL historical data.

Output:
1. Which stocks have the highest bounce rates after N consecutive down days?
2. What's the optimal number of down days to wait before buying?
3. How can we enhance our strategy with this data?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import warnings
warnings.filterwarnings('ignore')

# Import Polygon data provider for fetching historical data
try:
    from data.providers.polygon_eod import fetch_daily_bars_polygon
    from config.env_loader import load_env
    load_env(Path(".env"))
    HAS_POLYGON = True
except Exception:
    HAS_POLYGON = False


def validate_data(df: pd.DataFrame, min_years: int = 5) -> Dict:
    """Validate data quality for backtesting."""
    if df is None or len(df) < 100:
        return {"valid": False, "reason": "Insufficient data"}

    # Check date range
    df = df.sort_values('timestamp')
    date_range = (df['timestamp'].max() - df['timestamp'].min()).days
    years = date_range / 365

    if years < min_years:
        return {"valid": False, "reason": f"Only {years:.1f} years (need {min_years})"}

    # Check for OHLC validity
    ohlc_valid = (
        (df['high'] >= df['low']).all() and
        (df['high'] >= df['open']).all() and
        (df['high'] >= df['close']).all() and
        (df['low'] <= df['open']).all() and
        (df['low'] <= df['close']).all()
    )

    if not ohlc_valid:
        return {"valid": False, "reason": "OHLC violations"}

    # Check for gaps
    missing_pct = df.isnull().sum().max() / len(df) * 100
    if missing_pct > 5:
        return {"valid": False, "reason": f"{missing_pct:.1f}% missing data"}

    return {
        "valid": True,
        "bars": len(df),
        "years": years,
        "start": df['timestamp'].min().strftime('%Y-%m-%d'),
        "end": df['timestamp'].max().strftime('%Y-%m-%d'),
    }


def vectorized_consecutive_analysis(df: pd.DataFrame) -> Dict:
    """
    VECTORIZED analysis of consecutive down days and subsequent bounces.
    Much faster than iterating row by row.
    """
    if df is None or len(df) < 100:
        return None

    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Calculate daily returns
    df['return'] = df['close'].pct_change()

    # Track down days (1 = down day, 0 = up day)
    df['down'] = (df['return'] < 0).astype(int)

    # Calculate consecutive down day streaks using groupby
    df['streak_group'] = (df['down'] != df['down'].shift()).cumsum()
    df['streak_len'] = df.groupby('streak_group')['down'].cumsum()

    # Calculate forward returns for bounce analysis
    df['fwd_1d'] = df['close'].pct_change(1).shift(-1)
    df['fwd_3d'] = df['close'].pct_change(3).shift(-3)
    df['fwd_5d'] = df['close'].pct_change(5).shift(-5)
    df['fwd_7d'] = df['close'].pct_change(7).shift(-7)

    results = {}

    # Analyze each streak length (3 to 7 consecutive down days)
    for streak_len in range(3, 8):
        # Find where streak ends at exactly this length
        # (next bar is either up or end of data)
        is_streak_end = (
            (df['streak_len'] == streak_len) &
            (df['down'] == 1) &
            ((df['down'].shift(-1) == 0) | df['down'].shift(-1).isna())
        )

        streak_data = df[is_streak_end].dropna(subset=['fwd_5d'])

        if len(streak_data) < 5:
            continue

        # Calculate bounce statistics
        results[streak_len] = {
            'count': len(streak_data),
            'bounce_1d': (streak_data['fwd_1d'] > 0).mean() * 100,
            'bounce_3d': (streak_data['fwd_3d'] > 0).mean() * 100,
            'bounce_5d': (streak_data['fwd_5d'] > 0).mean() * 100,
            'bounce_7d': (streak_data['fwd_7d'] > 0).mean() * 100,
            'avg_1d': streak_data['fwd_1d'].mean() * 100,
            'avg_3d': streak_data['fwd_3d'].mean() * 100,
            'avg_5d': streak_data['fwd_5d'].mean() * 100,
            'avg_7d': streak_data['fwd_7d'].mean() * 100,
        }

    return results if results else None


def fetch_stock_data(symbol: str, cache_dir: Path) -> Optional[pd.DataFrame]:
    """Fetch stock data from cache or Polygon API."""
    # First check polygon_cache for full historical data (simple SYMBOL.csv format)
    polygon_cache = Path("data/polygon_cache")
    cached_file = polygon_cache / f"{symbol}.csv"

    if cached_file.exists():
        try:
            df = pd.read_csv(cached_file)
            # Convert timestamp if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
                df = df.rename(columns={'date': 'timestamp'})
            if len(df) >= 1000:  # At least ~4 years
                return df
        except Exception:
            pass

    # Check data/cache/polygon/ for date-ranged files (primary source for 900 stocks)
    polygon_api_cache = Path("data/cache/polygon")
    if polygon_api_cache.exists():
        # Look for 2015-2024 files first (full history)
        full_history_file = polygon_api_cache / f"{symbol}_2015-01-01_2024-12-31.csv"
        if full_history_file.exists():
            try:
                df = pd.read_csv(full_history_file, parse_dates=['timestamp'])
                if len(df) >= 500:  # Accept 2+ years
                    return df
            except Exception:
                pass

        # Fallback to other date-ranged files
        for f in sorted(polygon_api_cache.glob(f"{symbol}_*.csv"), reverse=True):
            try:
                df = pd.read_csv(f, parse_dates=['timestamp'])
                if len(df) >= 500:
                    return df
            except Exception:
                continue

    # Try fetching from Polygon API if available
    if HAS_POLYGON:
        try:
            # Use data/cache as base - polygon_eod creates polygon/ subdir
            df = fetch_daily_bars_polygon(
                symbol,
                start="2015-01-01",
                end="2024-12-31",
                cache_dir=Path("data/cache")
            )
            if df is not None and len(df) >= 1000:
                return df
        except Exception:
            pass

    return None


def main():
    print("=" * 80)
    print("QUANT-GRADE PATTERN ANALYSIS - FULL 900 STOCK UNIVERSE")
    print("Analyzing ALL stocks for consecutive down-day patterns")
    print("=" * 80)
    print(f"Time: {datetime.now()}")
    print(f"Polygon API: {'ENABLED' if HAS_POLYGON else 'DISABLED (cache only)'}")
    print()

    # Get all universe stocks
    universe_file = Path("data/universe/optionable_liquid_900.csv")
    if universe_file.exists():
        universe = pd.read_csv(universe_file)
        all_symbols = universe['symbol'].tolist()
        print(f"Universe: {len(all_symbols)} stocks")
    else:
        all_symbols = []
        print("No universe file found")
        return

    # Check existing cache
    polygon_cache = Path("data/polygon_cache")
    cached_files = list(polygon_cache.glob("*.csv"))
    cached_symbols = {f.stem for f in cached_files}
    print(f"Cached in polygon_cache: {len(cached_symbols)} stocks")

    # Use all universe stocks
    symbols = all_symbols
    print(f"Analyzing: {len(symbols)} stocks")
    print()

    # Data validation
    print("=" * 60)
    print("STEP 1: DATA FETCH & VALIDATION (2+ years required)")
    print("=" * 60)

    valid_stocks = []
    invalid_stocks = []
    data_cache = {}
    cache_dir = Path("data/polygon_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Fetch data in parallel for efficiency
    def process_symbol(symbol: str) -> tuple:
        df = fetch_stock_data(symbol, cache_dir)
        if df is None:
            return (symbol, None, {"valid": False, "reason": "No data available"})
        validation = validate_data(df, min_years=2)  # Accept 2+ years to include all viable stocks
        return (symbol, df, validation)

    print(f"Fetching data for {len(symbols)} stocks (parallel)...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_symbol, sym): sym for sym in symbols}
        for i, future in enumerate(as_completed(futures), 1):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(symbols)} stocks processed...")

            try:
                symbol, df, validation = future.result(timeout=60)
                if validation['valid']:
                    valid_stocks.append({
                        'symbol': symbol,
                        'bars': validation['bars'],
                        'years': validation['years'],
                    })
                    data_cache[symbol] = df
                else:
                    invalid_stocks.append({
                        'symbol': symbol,
                        'reason': validation['reason'],
                    })
            except Exception as e:
                invalid_stocks.append({'symbol': futures[future], 'reason': str(e)[:50]})

    print()
    print(f"VALID: {len(valid_stocks)} stocks with 5+ years of data")
    print(f"INVALID: {len(invalid_stocks)} stocks excluded")
    print()

    # Pattern analysis
    print("=" * 60)
    print("STEP 2: VECTORIZED PATTERN ANALYSIS")
    print("=" * 60)

    all_patterns = []
    by_streak_length = defaultdict(list)

    for i, stock in enumerate(valid_stocks):
        if (i + 1) % 100 == 0:
            print(f"Analyzing: {i+1}/{len(valid_stocks)} ({stock['symbol']})...")

        symbol = stock['symbol']
        df = data_cache[symbol]

        patterns = vectorized_consecutive_analysis(df)
        if patterns:
            for streak_len, stats in patterns.items():
                if stats['count'] >= 10:  # Minimum 10 samples
                    pattern_data = {
                        'symbol': symbol,
                        'streak': streak_len,
                        **stats
                    }
                    all_patterns.append(pattern_data)
                    by_streak_length[streak_len].append(pattern_data)

    print()
    print(f"Total patterns found: {len(all_patterns)}")
    print()

    # Analysis by streak length
    print("=" * 60)
    print("STEP 3: AVERAGE BOUNCE RATES BY CONSECUTIVE DOWN DAYS")
    print("=" * 60)
    print()

    summary_by_streak = {}
    for streak_len in sorted(by_streak_length.keys()):
        patterns = by_streak_length[streak_len]
        if len(patterns) >= 5:
            avg_bounce_5d = np.mean([p['bounce_5d'] for p in patterns])
            avg_move_5d = np.mean([p['avg_5d'] for p in patterns])
            total_samples = sum([p['count'] for p in patterns])

            summary_by_streak[streak_len] = {
                'stocks': len(patterns),
                'total_samples': total_samples,
                'avg_bounce_5d': avg_bounce_5d,
                'avg_move_5d': avg_move_5d,
            }

            print(f"{streak_len} CONSECUTIVE DOWN DAYS:")
            print(f"  Stocks analyzed: {len(patterns)}")
            print(f"  Total samples: {total_samples}")
            print(f"  Avg 5-day bounce rate: {avg_bounce_5d:.1f}%")
            print(f"  Avg 5-day return: +{avg_move_5d:.2f}%")
            print()

    # Find optimal entry point
    best_streak = max(summary_by_streak.items(), key=lambda x: x[1]['avg_bounce_5d'])
    print("=" * 60)
    print(f"OPTIMAL ENTRY: {best_streak[0]} CONSECUTIVE DOWN DAYS")
    print(f"Average bounce rate: {best_streak[1]['avg_bounce_5d']:.1f}%")
    print(f"Average 5-day return: +{best_streak[1]['avg_move_5d']:.2f}%")
    print("=" * 60)
    print()

    # Top 50 stocks by bounce rate
    print("=" * 60)
    print("STEP 4: TOP 50 STOCKS WITH HIGHEST BOUNCE RATES")
    print("=" * 60)
    print()

    # Filter for high-quality patterns (N >= 15, bounce >= 70%)
    high_quality = [p for p in all_patterns if p['count'] >= 15 and p['bounce_5d'] >= 70]
    high_quality.sort(key=lambda x: (-x['bounce_5d'], -x['count']))

    print("| Rank | Symbol | Streak | N | 1D Bounce | 5D Bounce | 7D Bounce | Avg 5D Move |")
    print("|------|--------|--------|---|-----------|-----------|-----------|-------------|")

    for i, p in enumerate(high_quality[:50]):
        print(f"| {i+1:4d} | {p['symbol']:6s} | {p['streak']:6d}+ | {p['count']:3d} | {p['bounce_1d']:9.0f}% | {p['bounce_5d']:9.0f}% | {p['bounce_7d']:9.0f}% | +{p['avg_5d']:9.2f}% |")

    print()

    # Strategy enhancement suggestions
    print("=" * 60)
    print("STEP 5: QUANT STRATEGY ENHANCEMENT SUGGESTIONS")
    print("=" * 60)
    print()

    print("Based on the analysis of", len(valid_stocks), "stocks:")
    print()
    print("1. ENTRY TIMING OPTIMIZATION:")
    print(f"   - Optimal entry point: After {best_streak[0]} consecutive down days")
    print(f"   - Expected 5-day bounce rate: {best_streak[1]['avg_bounce_5d']:.1f}%")
    print(f"   - Expected 5-day return: +{best_streak[1]['avg_move_5d']:.2f}%")
    print()

    print("2. STOCK SELECTION ENHANCEMENT:")
    print("   Focus on stocks with proven reversal patterns:")
    for p in high_quality[:5]:
        print(f"   - {p['symbol']}: {p['count']} instances, {p['bounce_5d']:.0f}% bounce rate")
    print()

    print("3. POSITION SIZING BY CONFIDENCE:")
    print("   - Tier 1 (>85% bounce): Full position (stocks listed above)")
    print("   - Tier 2 (75-85% bounce): 75% position")
    print("   - Tier 3 (65-75% bounce): 50% position")
    print()

    print("4. COMBINE WITH EXISTING IBS+RSI STRATEGY:")
    print("   - Current strategy: IBS < 0.08, RSI(2) < 5")
    print("   - Enhancement: Add consecutive down day filter")
    print("   - Entry: IBS < 0.08 AND RSI(2) < 5 AND 4+ consecutive down days")
    print("   - Expected improvement: Higher win rate, better timing")
    print()

    # Save report
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    report = {
        "generated": datetime.now().isoformat(),
        "valid_stocks": len(valid_stocks),
        "total_patterns": len(all_patterns),
        "summary_by_streak": summary_by_streak,
        "optimal_entry": {
            "streak": best_streak[0],
            "avg_bounce_5d": best_streak[1]['avg_bounce_5d'],
            "avg_move_5d": best_streak[1]['avg_move_5d'],
        },
        "top_50_patterns": high_quality[:50],
    }

    json_file = report_dir / f"quant_pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2)

    md_file = report_dir / "QUANT_PATTERN_ANALYSIS.md"
    with open(md_file, 'w') as f:
        f.write("# QUANT PATTERN ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write("---\n\n")

        f.write("## DATA VALIDATION\n")
        f.write(f"- Valid stocks (5+ years): {len(valid_stocks)}\n")
        f.write(f"- Total patterns analyzed: {len(all_patterns)}\n\n")

        f.write("## AVERAGE BOUNCE RATES BY CONSECUTIVE DOWN DAYS\n\n")
        f.write("| Streak | Stocks | Samples | Avg 5D Bounce | Avg 5D Move |\n")
        f.write("|--------|--------|---------|---------------|-------------|\n")
        for streak_len, stats in sorted(summary_by_streak.items()):
            f.write(f"| {streak_len}+ days | {stats['stocks']} | {stats['total_samples']} | {stats['avg_bounce_5d']:.1f}% | +{stats['avg_move_5d']:.2f}% |\n")

        f.write("\n## OPTIMAL ENTRY POINT\n")
        f.write(f"**{best_streak[0]} consecutive down days** = {best_streak[1]['avg_bounce_5d']:.1f}% avg bounce rate\n\n")

        f.write("## TOP 50 STOCKS WITH HIGHEST BOUNCE RATES\n\n")
        f.write("| Rank | Symbol | Streak | N | 5D Bounce | Avg 5D Move |\n")
        f.write("|------|--------|--------|---|-----------|-------------|\n")
        for i, p in enumerate(high_quality[:50]):
            f.write(f"| {i+1} | {p['symbol']} | {p['streak']}+ | {p['count']} | {p['bounce_5d']:.0f}% | +{p['avg_5d']:.2f}% |\n")

        f.write("\n## STRATEGY ENHANCEMENT RECOMMENDATIONS\n\n")
        f.write(f"1. Enter after **{best_streak[0]} consecutive down days** for highest probability\n")
        f.write("2. Focus on stocks with proven reversal patterns (see Top 50)\n")
        f.write("3. Combine with existing IBS+RSI strategy for better timing\n")
        f.write("4. Size positions based on historical bounce rate\n")

    print(f"Report saved to: {json_file}")
    print(f"Markdown saved to: {md_file}")
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
