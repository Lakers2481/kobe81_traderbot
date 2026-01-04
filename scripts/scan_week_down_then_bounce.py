#!/usr/bin/env python3
"""
QUANT SCAN: Consecutive Down-Day Streaks and Bounce Analysis
=============================================================

MISSION: Analyze how stocks bounce after consecutive down days.

CRITICAL RULES (NO VIOLATIONS):
1. NO LOOKAHEAD - Events defined using ONLY data up to day t
2. EXACT 900 UNIVERSE - Hard fail if not exactly 900 stocks
3. REAL DATA ONLY - From cached Polygon data, no synthetic
4. NO TAUTOLOGY - Never define event as "streak ended" (fake 100%)

Author: Kobe Trading System
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Universe
    "universe_file": "data/universe/optionable_liquid_900.csv",
    "expected_count": 900,
    "magnificent_7": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],

    # Data
    "cache_dirs": [
        Path("data/polygon_cache"),
        Path("data/cache/polygon"),
    ],
    "min_bars": 252,  # 1 year minimum (more inclusive)
    "use_close_field": "close",  # Document: using 'close' not 'adj_close'

    # Analysis
    "streak_levels": [1, 2, 3, 4, 5, 6, 7],
    "bounce_horizon": 14,  # 14 trading days = ~2 weeks
    "target_pct": 0.02,  # 2% target for hit analysis

    # Output
    "output_dir": Path("reports"),
}


# =============================================================================
# UNIVERSE LOADING WITH STRICT VALIDATION
# =============================================================================

def load_universe_strict() -> List[str]:
    """
    Load EXACTLY 900 stocks. HARD FAIL if not 900.
    Also verify Magnificent 7 are present.
    """
    universe_path = Path(CONFIG["universe_file"])

    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")

    df = pd.read_csv(universe_path)

    # Find symbol column
    if 'symbol' in df.columns:
        symbols = df['symbol'].astype(str).str.strip().str.upper().tolist()
    else:
        symbols = df.iloc[:, 0].astype(str).str.strip().str.upper().tolist()

    # Remove empty and dedupe while preserving order
    seen = set()
    unique_symbols = []
    for s in symbols:
        if s and s not in seen and s != 'SYMBOL':
            seen.add(s)
            unique_symbols.append(s)

    # HARD VALIDATION: Must be exactly 900
    if len(unique_symbols) != CONFIG["expected_count"]:
        raise ValueError(
            f"UNIVERSE VALIDATION FAILED: Expected {CONFIG['expected_count']} stocks, "
            f"got {len(unique_symbols)}. HARD FAIL."
        )

    # HARD VALIDATION: Magnificent 7 must be present
    missing_mag7 = [s for s in CONFIG["magnificent_7"] if s not in unique_symbols]
    if missing_mag7:
        raise ValueError(
            f"UNIVERSE VALIDATION FAILED: Missing Magnificent 7: {missing_mag7}. HARD FAIL."
        )

    return unique_symbols


# =============================================================================
# DATA LOADING WITH HYGIENE
# =============================================================================

def find_cache_file(symbol: str) -> Optional[Path]:
    """Find cached data file for symbol."""
    for cache_dir in CONFIG["cache_dirs"]:
        if not cache_dir.exists():
            continue

        # Try simple format first
        simple_path = cache_dir / f"{symbol}.csv"
        if simple_path.exists():
            return simple_path

        # Try date-ranged format
        for f in cache_dir.glob(f"{symbol}_*.csv"):
            return f

    return None


def load_stock_data(symbol: str) -> Optional[pd.DataFrame]:
    """
    Load and clean stock data with strict hygiene.

    Returns None if data is insufficient or invalid.
    """
    cache_file = find_cache_file(symbol)
    if cache_file is None:
        return None

    try:
        df = pd.read_csv(cache_file)

        # Identify columns
        date_col = None
        for col in ['timestamp', 'date', 'Date', 'Timestamp']:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            return None

        # Parse dates
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])

        # Rename to standard
        df = df.rename(columns={date_col: 'date'})

        # Ensure required columns
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        # Handle case sensitivity
        df.columns = df.columns.str.lower()

        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            return None

        # Select and clean
        df = df[required].copy()

        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN in critical columns
        df = df.dropna(subset=['close', 'high', 'low'])

        # Validate no negative/zero prices
        df = df[(df['close'] > 0) & (df['high'] > 0) & (df['low'] > 0)]

        # Sort by date and remove duplicates
        df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        df = df.reset_index(drop=True)

        # Check minimum bars
        if len(df) < CONFIG["min_bars"]:
            return None

        # Use last 5 years only
        df = df.tail(252 * 5).reset_index(drop=True)

        return df

    except Exception as e:
        return None


# =============================================================================
# STREAK CALCULATION (NO LOOKAHEAD)
# =============================================================================

def calculate_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate consecutive down-day streaks.

    CRITICAL: Uses ONLY data up to day t. No lookahead.

    down[t] = close[t] < close[t-1]
    streak_len[t] = consecutive down days ending at t
    """
    df = df.copy()

    # Down day: close < previous close
    df['down'] = (df['close'] < df['close'].shift(1)).astype(int)

    # Calculate streak length using cumsum trick
    # Reset counter when not a down day
    df['streak_group'] = (~df['down'].astype(bool)).cumsum()
    df['streak_len'] = df.groupby('streak_group')['down'].cumsum()

    # First row has no previous, so streak_len = 0
    df.loc[df.index[0], 'streak_len'] = 0

    return df


# =============================================================================
# EVENT DETECTION (FIRST-HIT ONLY, NO TAUTOLOGY)
# =============================================================================

def detect_events(df: pd.DataFrame, streak_levels: List[int]) -> pd.DataFrame:
    """
    Detect events when streak FIRST hits each level N.

    CRITICAL RULES:
    1. Event triggers when streak_len[t] == N for FIRST time in this streak
    2. Same streak can generate events at N=1,2,3,4,5,6,7 as it grows
    3. NEVER defined as "streak ended" (that would be lookahead/tautology)

    Example: A 5-day streak generates events at:
    - Day 1 of streak: event_1
    - Day 2 of streak: event_2
    - Day 3 of streak: event_3
    - Day 4 of streak: event_4
    - Day 5 of streak: event_5
    """
    events = []

    for N in streak_levels:
        # Event at first time streak hits N
        # This means: streak_len[t] == N AND streak_len[t-1] == N-1
        # Which simplifies to: streak_len[t] == N (since we're counting up)
        mask = df['streak_len'] == N

        event_indices = df[mask].index.tolist()

        for idx in event_indices:
            # Ensure we have enough future bars for bounce analysis
            if idx + CONFIG["bounce_horizon"] >= len(df):
                continue

            events.append({
                'idx': idx,
                'date': df.loc[idx, 'date'],
                'streak_level': N,
                'close_at_event': df.loc[idx, 'close'],
            })

    return pd.DataFrame(events)


# =============================================================================
# BOUNCE ANALYSIS (MULTIPLE DEFINITIONS)
# =============================================================================

def analyze_bounce(df: pd.DataFrame, event_idx: int, close_at_event: float) -> Dict:
    """
    Analyze bounce after event using multiple definitions.

    All analysis uses future bars t+1 to t+H only (NO lookahead).

    BOUNCE DEFINITIONS:
    1. RECOVERY: close[t+j] > close_at_event (price recovers above event close)
       - This is NOT guaranteed - requires actual recovery
    2. POSITIVE CLOSE: 14-day return > 0 (ended higher than event close)
    3. TARGET HIT: high reaches +2% above event close
    """
    H = CONFIG["bounce_horizon"]
    target_pct = CONFIG["target_pct"]

    # Future window: t+1 to t+H
    future_start = event_idx + 1
    future_end = event_idx + H + 1  # +1 because slice is exclusive

    if future_end > len(df):
        future_end = len(df)

    future_df = df.iloc[future_start:future_end].copy()

    if len(future_df) == 0:
        return {
            'recovered': False,
            'days_to_recovery': np.nan,
            'return_at_recovery': np.nan,
            'positive_14d': False,
            'return_14d': np.nan,
            'best_return_14d': np.nan,
            'worst_drawdown_14d': np.nan,
            'hit_target': False,
            'days_to_target': np.nan,
            'recovered_within_7': False,
            'hit_target_within_7': False,
        }

    results = {}

    # =========================================================================
    # 1. RECOVERY BOUNCE: First day where close > close_at_event
    # =========================================================================
    # This is the TRUE bounce - price recovers above where the selloff ended
    future_df['recovered'] = future_df['close'] > close_at_event

    recovery_days = future_df[future_df['recovered']]

    if len(recovery_days) > 0:
        first_recovery_idx = recovery_days.index[0]
        days_to_recovery = first_recovery_idx - event_idx
        close_at_recovery = future_df.loc[first_recovery_idx, 'close']
        return_at_recovery = (close_at_recovery / close_at_event) - 1

        results['recovered'] = True
        results['days_to_recovery'] = days_to_recovery
        results['return_at_recovery'] = return_at_recovery
        results['recovered_within_7'] = days_to_recovery <= 7
    else:
        results['recovered'] = False
        results['days_to_recovery'] = np.nan
        results['return_at_recovery'] = np.nan
        results['recovered_within_7'] = False

    # =========================================================================
    # 2. POSITIVE 14-DAY RETURN
    # =========================================================================
    # Did the stock close higher than event close after 14 days?
    final_close = future_df.iloc[-1]['close'] if len(future_df) > 0 else close_at_event
    return_14d = (final_close / close_at_event) - 1

    results['positive_14d'] = return_14d > 0
    results['return_14d'] = return_14d

    # =========================================================================
    # 3. BEST MOVE (max favorable excursion) within H
    # =========================================================================
    max_high = future_df['high'].max()
    min_low = future_df['low'].min()

    results['best_return_14d'] = (max_high / close_at_event) - 1
    results['worst_drawdown_14d'] = (min_low / close_at_event) - 1

    # =========================================================================
    # 4. TARGET HIT timing (2% target)
    # =========================================================================
    target_price = close_at_event * (1 + target_pct)

    # Find first day where high >= target
    target_hits = future_df[future_df['high'] >= target_price]

    if len(target_hits) > 0:
        first_hit_idx = target_hits.index[0]
        days_to_target = first_hit_idx - event_idx

        results['hit_target'] = True
        results['days_to_target'] = days_to_target
        results['hit_target_within_7'] = days_to_target <= 7
    else:
        results['hit_target'] = False
        results['days_to_target'] = np.nan
        results['hit_target_within_7'] = False

    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_symbol(symbol: str) -> Optional[Dict]:
    """Analyze single symbol for all streak levels."""
    df = load_stock_data(symbol)
    if df is None:
        return None

    # Calculate streaks
    df = calculate_streaks(df)

    # Detect events
    events_df = detect_events(df, CONFIG["streak_levels"])

    if len(events_df) == 0:
        return None

    # Analyze each event
    all_events = []

    for _, event in events_df.iterrows():
        bounce_results = analyze_bounce(df, event['idx'], event['close_at_event'])

        event_data = {
            'symbol': symbol,
            'date': event['date'],
            'streak_level': event['streak_level'],
            'close_at_event': event['close_at_event'],
            **bounce_results
        }
        all_events.append(event_data)

    return {
        'symbol': symbol,
        'bars': len(df),
        'date_range': (df['date'].min(), df['date'].max()),
        'events': all_events,
    }


def run_full_analysis() -> Dict:
    """Run full analysis across all 900 stocks."""

    print("=" * 70)
    print("QUANT SCAN: CONSECUTIVE DOWN-DAY BOUNCE ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print()

    # =========================================================================
    # STEP 1: Load and validate universe
    # =========================================================================
    print("STEP 1: Loading universe...")
    symbols = load_universe_strict()
    print(f"  Universe loaded: {len(symbols)} stocks")
    print(f"  Magnificent 7 verified: {CONFIG['magnificent_7']}")
    print()

    # =========================================================================
    # STEP 2: Process all symbols
    # =========================================================================
    print("STEP 2: Processing all symbols...")

    all_events = []
    per_stock_stats = []
    excluded_symbols = []

    for i, symbol in enumerate(symbols):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(symbols)} symbols...")

        result = analyze_symbol(symbol)

        if result is None:
            excluded_symbols.append({'symbol': symbol, 'reason': 'insufficient data'})
            continue

        # Collect events
        for event in result['events']:
            all_events.append(event)

        # Calculate per-stock stats
        events_df = pd.DataFrame(result['events'])

        stock_stats = {'symbol': symbol, 'total_bars': result['bars']}

        for N in CONFIG["streak_levels"]:
            n_events = events_df[events_df['streak_level'] == N]

            if len(n_events) == 0:
                stock_stats[f'events_N{N}'] = 0
                stock_stats[f'recovery_rate_14_N{N}'] = np.nan
                stock_stats[f'recovery_rate_7_N{N}'] = np.nan
                stock_stats[f'avg_days_recovery_N{N}'] = np.nan
                stock_stats[f'avg_return_recovery_N{N}'] = np.nan
                stock_stats[f'positive_14d_rate_N{N}'] = np.nan
                stock_stats[f'avg_return_14d_N{N}'] = np.nan
                stock_stats[f'hit_target_rate_14_N{N}'] = np.nan
                stock_stats[f'avg_days_target_N{N}'] = np.nan
                stock_stats[f'avg_best_return_N{N}'] = np.nan
                stock_stats[f'avg_worst_dd_N{N}'] = np.nan
            else:
                stock_stats[f'events_N{N}'] = len(n_events)
                stock_stats[f'recovery_rate_14_N{N}'] = n_events['recovered'].mean()
                stock_stats[f'recovery_rate_7_N{N}'] = n_events['recovered_within_7'].mean()

                recovered = n_events[n_events['recovered']]
                stock_stats[f'avg_days_recovery_N{N}'] = recovered['days_to_recovery'].mean() if len(recovered) > 0 else np.nan
                stock_stats[f'avg_return_recovery_N{N}'] = recovered['return_at_recovery'].mean() if len(recovered) > 0 else np.nan

                stock_stats[f'positive_14d_rate_N{N}'] = n_events['positive_14d'].mean()
                stock_stats[f'avg_return_14d_N{N}'] = n_events['return_14d'].mean()

                stock_stats[f'hit_target_rate_14_N{N}'] = n_events['hit_target'].mean()
                hits = n_events[n_events['hit_target']]
                stock_stats[f'avg_days_target_N{N}'] = hits['days_to_target'].mean() if len(hits) > 0 else np.nan

                stock_stats[f'avg_best_return_N{N}'] = n_events['best_return_14d'].mean()
                stock_stats[f'avg_worst_dd_N{N}'] = n_events['worst_drawdown_14d'].mean()

        per_stock_stats.append(stock_stats)

    print(f"  Completed: {len(symbols)} symbols")
    print(f"  Excluded: {len(excluded_symbols)} (insufficient data)")
    print(f"  Total events: {len(all_events)}")
    print()

    # =========================================================================
    # STEP 3: Calculate overall statistics
    # =========================================================================
    print("STEP 3: Calculating overall statistics...")

    events_df = pd.DataFrame(all_events)
    overall_stats = {}

    for N in CONFIG["streak_levels"]:
        n_events = events_df[events_df['streak_level'] == N]

        if len(n_events) == 0:
            continue

        stats = {
            'streak_level': N,
            'total_events': len(n_events),
            'unique_symbols': n_events['symbol'].nunique(),

            # Recovery bounce (close > event close)
            'pct_recovered_14': n_events['recovered'].mean() * 100,
            'pct_recovered_7': n_events['recovered_within_7'].mean() * 100,

            # Positive 14-day return
            'pct_positive_14d': n_events['positive_14d'].mean() * 100,
            'avg_return_14d': n_events['return_14d'].mean() * 100,

            # Timing distribution for recovery
            'pct_day_1': (n_events['days_to_recovery'] == 1).mean() * 100,
            'pct_day_2': (n_events['days_to_recovery'] == 2).mean() * 100,
            'pct_day_3': (n_events['days_to_recovery'] == 3).mean() * 100,
            'pct_day_4': (n_events['days_to_recovery'] == 4).mean() * 100,
            'pct_day_5': (n_events['days_to_recovery'] == 5).mean() * 100,
            'pct_day_6': (n_events['days_to_recovery'] == 6).mean() * 100,
            'pct_day_7': (n_events['days_to_recovery'] == 7).mean() * 100,
            'pct_day_8_14': ((n_events['days_to_recovery'] >= 8) & (n_events['days_to_recovery'] <= 14)).mean() * 100,
            'pct_no_recovery': (~n_events['recovered']).mean() * 100,

            # Averages (recovered only)
            'avg_days_to_recovery': n_events[n_events['recovered']]['days_to_recovery'].mean(),
            'avg_return_at_recovery': n_events[n_events['recovered']]['return_at_recovery'].mean() * 100,

            # Best/worst (all events)
            'avg_best_return_14d': n_events['best_return_14d'].mean() * 100,
            'avg_worst_drawdown_14d': n_events['worst_drawdown_14d'].mean() * 100,

            # Target hit (2%)
            'pct_hit_target_2pct_14': n_events['hit_target'].mean() * 100,
            'pct_hit_target_2pct_7': n_events['hit_target_within_7'].mean() * 100,
            'avg_days_to_target': n_events[n_events['hit_target']]['days_to_target'].mean(),
        }

        overall_stats[N] = stats

    # =========================================================================
    # STEP 4: Sanity checks
    # =========================================================================
    print("STEP 4: Running sanity checks...")

    warnings_list = []

    for N, stats in overall_stats.items():
        # Check for suspicious 100% recovery rates
        if stats['pct_recovered_14'] >= 98 and stats['total_events'] > 100:
            warnings_list.append(
                f"WARNING: N={N} has {stats['pct_recovered_14']:.1f}% recovery rate "
                f"with {stats['total_events']} events. Check for tautology!"
            )

        # Check recovery rate is reasonable (not too low either)
        if stats['pct_recovered_14'] < 40:
            print(f"  Note: N={N} has only {stats['pct_recovered_14']:.1f}% recovery rate (< 40%)")

    if warnings_list:
        print("  WARNINGS DETECTED:")
        for w in warnings_list:
            print(f"    {w}")
    else:
        print("  All sanity checks passed.")
    print()

    return {
        'symbols': symbols,
        'events_df': events_df,
        'per_stock_stats': pd.DataFrame(per_stock_stats),
        'overall_stats': overall_stats,
        'excluded_symbols': excluded_symbols,
        'warnings': warnings_list,
    }


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_outputs(results: Dict):
    """Generate all output files."""

    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(exist_ok=True)

    events_df = results['events_df']
    per_stock_df = results['per_stock_stats']
    overall_stats = results['overall_stats']

    # =========================================================================
    # 1. Overall CSV
    # =========================================================================
    overall_rows = []
    for N, stats in overall_stats.items():
        overall_rows.append(stats)
    overall_df = pd.DataFrame(overall_rows)
    overall_csv_path = output_dir / "week_down_then_bounce_overall.csv"
    overall_df.to_csv(overall_csv_path, index=False)

    # =========================================================================
    # 2. Per-stock CSV
    # =========================================================================
    per_stock_csv_path = output_dir / "week_down_then_bounce_per_stock.csv"
    per_stock_df.to_csv(per_stock_csv_path, index=False)

    # =========================================================================
    # 3. Events CSV (parquet if available)
    # =========================================================================
    events_csv_path = output_dir / "week_down_then_bounce_events.csv"
    events_df.to_csv(events_csv_path, index=False)

    try:
        events_parquet_path = output_dir / "week_down_then_bounce_events.parquet"
        events_df.to_parquet(events_parquet_path, index=False)
    except (ImportError, OSError):
        pass  # Parquet not available

    # =========================================================================
    # 4. Run log JSON
    # =========================================================================
    runlog = {
        'generated_at': datetime.now().isoformat(),
        'parameters': {
            'universe_file': str(CONFIG['universe_file']),
            'expected_count': CONFIG['expected_count'],
            'min_bars': CONFIG['min_bars'],
            'bounce_horizon': CONFIG['bounce_horizon'],
            'target_pct': CONFIG['target_pct'],
            'close_field_used': CONFIG['use_close_field'],
        },
        'results': {
            'total_symbols': len(results['symbols']),
            'symbols_with_data': len(results['symbols']) - len(results['excluded_symbols']),
            'excluded_count': len(results['excluded_symbols']),
            'total_events': len(events_df),
        },
        'excluded_symbols': results['excluded_symbols'][:20],  # First 20 only
        'warnings': results['warnings'],
    }

    runlog_path = output_dir / "week_down_then_bounce_runlog.json"
    with open(runlog_path, 'w') as f:
        json.dump(runlog, f, indent=2)

    # =========================================================================
    # 5. Markdown Summary
    # =========================================================================
    md_lines = [
        "# CONSECUTIVE DOWN-DAY BOUNCE ANALYSIS",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## METHODOLOGY (NO LOOKAHEAD)",
        "",
        "**Event Definition (CRITICAL):**",
        "- DOWN DAY: close[t] < close[t-1]",
        "- STREAK: Consecutive down days ending at day t",
        "- EVENT: First time streak reaches level N (1-7)",
        "- **NOT** defined as 'streak ended' (that would be tautological)",
        "",
        "**Bounce Definitions:**",
        "1. RECOVERY: First day where close > close_at_event (price recovers above selloff close)",
        "2. POSITIVE 14D: Final close after 14 days > event close",
        "3. BEST RETURN: Max(high[t+1..t+14]) / close[t] - 1",
        "4. TARGET HIT: First day where high >= close * 1.02 (2% target)",
        "",
        "---",
        "",
        "## UNIVERSE VERIFICATION",
        f"- **Total Stocks: {len(results['symbols'])} (EXACT)**",
        f"- **Magnificent 7: ALL PRESENT** ({', '.join(CONFIG['magnificent_7'])})",
        f"- Excluded (insufficient data): {len(results['excluded_symbols'])}",
        f"- Total Events Analyzed: {len(events_df):,}",
        "",
        "---",
        "",
        "## OVERALL SUMMARY BY STREAK LENGTH",
        "",
        "| Streak | Events | Stocks | Recover 14D | Recover 7D | Positive 14D | Avg Days | Best 14D | Hit 2% |",
        "|--------|--------|--------|-------------|------------|--------------|----------|----------|--------|",
    ]

    for N in CONFIG["streak_levels"]:
        if N not in overall_stats:
            continue
        s = overall_stats[N]
        avg_days = s['avg_days_to_recovery']
        avg_days_str = f"{avg_days:.1f}" if not np.isnan(avg_days) else "-"
        md_lines.append(
            f"| {N} down | {s['total_events']:,} | {s['unique_symbols']} | "
            f"{s['pct_recovered_14']:.1f}% | {s['pct_recovered_7']:.1f}% | "
            f"{s['pct_positive_14d']:.1f}% | {avg_days_str} | "
            f"{s['avg_best_return_14d']:+.2f}% | {s['pct_hit_target_2pct_14']:.1f}% |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## RECOVERY TIMING DISTRIBUTION",
        "",
        "| Streak | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | Day 6 | Day 7 | Day 8-14 | None |",
        "|--------|-------|-------|-------|-------|-------|-------|-------|----------|------|",
    ])

    for N in CONFIG["streak_levels"]:
        if N not in overall_stats:
            continue
        s = overall_stats[N]
        md_lines.append(
            f"| {N} down | {s['pct_day_1']:.1f}% | {s['pct_day_2']:.1f}% | "
            f"{s['pct_day_3']:.1f}% | {s['pct_day_4']:.1f}% | {s['pct_day_5']:.1f}% | "
            f"{s['pct_day_6']:.1f}% | {s['pct_day_7']:.1f}% | {s['pct_day_8_14']:.1f}% | "
            f"{s['pct_no_recovery']:.1f}% |"
        )

    # Magnificent 7 section
    md_lines.extend([
        "",
        "---",
        "",
        "## MAGNIFICENT 7 ANALYSIS",
        "",
    ])

    mag7_df = per_stock_df[per_stock_df['symbol'].isin(CONFIG['magnificent_7'])]

    for symbol in CONFIG['magnificent_7']:
        stock_row = mag7_df[mag7_df['symbol'] == symbol]
        if len(stock_row) == 0:
            md_lines.append(f"### {symbol}: No data available")
            continue

        stock = stock_row.iloc[0]
        md_lines.append(f"### {symbol}")
        md_lines.append("")
        md_lines.append("| Streak | Events | Recover 14D | Recover 7D | Positive 14D | Hit 2% |")
        md_lines.append("|--------|--------|-------------|------------|--------------|--------|")

        for N in CONFIG["streak_levels"]:
            events_n = stock.get(f'events_N{N}', 0)
            if events_n == 0:
                md_lines.append(f"| {N} down | 0 | - | - | - | - |")
            else:
                recover_14 = stock.get(f'recovery_rate_14_N{N}', 0) or 0
                recover_7 = stock.get(f'recovery_rate_7_N{N}', 0) or 0
                positive_14d = stock.get(f'positive_14d_rate_N{N}', 0) or 0
                hit_rate = stock.get(f'hit_target_rate_14_N{N}', 0) or 0

                md_lines.append(
                    f"| {N} down | {int(events_n)} | {recover_14*100:.1f}% | {recover_7*100:.1f}% | "
                    f"{positive_14d*100:.1f}% | {hit_rate*100:.1f}% |"
                )

        md_lines.append("")

    # Top stocks by bounce rate for N=5-7
    md_lines.extend([
        "---",
        "",
        "## TOP 20 STOCKS BY BOUNCE RATE (N=5+ down days)",
        "",
        "Minimum 10 events required.",
        "",
        "| Rank | Symbol | Events | Bounce 14D | Bounce 7D | Avg Days | Best Return |",
        "|------|--------|--------|------------|-----------|----------|-------------|",
    ])

    # Filter for N>=5 with enough events
    top_stocks = per_stock_df[per_stock_df['events_N5'] >= 10].copy()
    top_stocks['recovery_rate_5'] = top_stocks['recovery_rate_14_N5'].fillna(0)
    top_stocks = top_stocks.sort_values('recovery_rate_5', ascending=False).head(20)

    for rank, (_, row) in enumerate(top_stocks.iterrows(), 1):
        md_lines.append(
            f"| {rank} | {row['symbol']} | {int(row['events_N5'])} | "
            f"{row['recovery_rate_14_N5']*100:.1f}% | "
            f"{row['recovery_rate_7_N5']*100:.1f}% | "
            f"{row['avg_days_recovery_N5']:.1f} | "
            f"{row['avg_best_return_N5']*100:+.2f}% |"
        )

    # How many stocks went down for how long
    md_lines.extend([
        "",
        "---",
        "",
        "## HOW MANY STOCKS EXPERIENCED EACH STREAK LEVEL",
        "",
        "| Streak | Unique Stocks | Total Events | Avg Events/Stock |",
        "|--------|---------------|--------------|------------------|",
    ])

    for N in CONFIG["streak_levels"]:
        stocks_with_events = per_stock_df[per_stock_df[f'events_N{N}'] > 0]
        total_events = stocks_with_events[f'events_N{N}'].sum()
        avg_per_stock = total_events / len(stocks_with_events) if len(stocks_with_events) > 0 else 0
        md_lines.append(
            f"| {N}+ down | {len(stocks_with_events)} | {int(total_events):,} | {avg_per_stock:.1f} |"
        )

    # Stocks with most deep selloffs (N=7)
    md_lines.extend([
        "",
        "### Stocks with Most 7+ Day Selloffs",
        "",
        "| Rank | Symbol | N=7 Events | Bounce Rate |",
        "|------|--------|------------|-------------|",
    ])

    deep_selloffs = per_stock_df[per_stock_df['events_N7'] > 0].copy()
    deep_selloffs = deep_selloffs.sort_values('events_N7', ascending=False).head(15)

    for rank, (_, row) in enumerate(deep_selloffs.iterrows(), 1):
        recovery_rate = row.get('recovery_rate_14_N7', 0) or 0
        md_lines.append(
            f"| {rank} | {row['symbol']} | {int(row['events_N7'])} | {recovery_rate*100:.1f}% |"
        )

    # Warnings
    if results['warnings']:
        md_lines.extend([
            "",
            "---",
            "",
            "## WARNINGS",
            "",
        ])
        for w in results['warnings']:
            md_lines.append(f"- {w}")

    md_lines.extend([
        "",
        "---",
        "",
        "## FILES GENERATED",
        "",
        f"- `{overall_csv_path.name}` - Overall statistics by streak level",
        f"- `{per_stock_csv_path.name}` - Per-stock statistics",
        f"- `{events_csv_path.name}` - All individual events",
        f"- `{runlog_path.name}` - Run parameters and metadata",
        "",
        "---",
        "",
        "*Analysis complete. NO lookahead bias. REAL data only.*",
    ])

    md_path = output_dir / "week_down_then_bounce_summary.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))

    return {
        'md_path': md_path,
        'overall_csv_path': overall_csv_path,
        'per_stock_csv_path': per_stock_csv_path,
        'events_csv_path': events_csv_path,
        'runlog_path': runlog_path,
    }


def print_console_summary(results: Dict, output_paths: Dict):
    """Print human-friendly console summary."""

    overall_stats = results['overall_stats']
    per_stock_df = results['per_stock_stats']

    print("=" * 70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 70)
    print()

    print("UNIVERSE VERIFICATION:")
    print(f"  Total Stocks: {len(results['symbols'])} (EXACT 900)")
    print(f"  Magnificent 7: ALL PRESENT")
    print(f"  Excluded: {len(results['excluded_symbols'])} (insufficient data)")
    print()

    print("OVERALL RESULTS BY STREAK LENGTH:")
    print("-" * 70)
    print(f"{'Streak':<8} {'Events':>8} {'Recover14D':>11} {'Recover7D':>10} {'AvgDays':>8} {'Best14D':>10}")
    print("-" * 70)

    for N in CONFIG["streak_levels"]:
        if N not in overall_stats:
            continue
        s = overall_stats[N]
        avg_days = s['avg_days_to_recovery']
        avg_days_str = f"{avg_days:>7.1f}" if not np.isnan(avg_days) else "    -  "
        print(
            f"{N} down   {s['total_events']:>8,} {s['pct_recovered_14']:>10.1f}% "
            f"{s['pct_recovered_7']:>9.1f}% {avg_days_str} "
            f"{s['avg_best_return_14d']:>+9.2f}%"
        )

    print()
    print("TOP 10 STOCKS BY RECOVERY RATE (N=5+, min 10 events):")
    print("-" * 70)

    top_stocks = per_stock_df[per_stock_df['events_N5'] >= 10].copy()
    top_stocks['recovery_rate_5'] = top_stocks['recovery_rate_14_N5'].fillna(0)
    top_stocks = top_stocks.sort_values('recovery_rate_5', ascending=False).head(10)

    print(f"{'Rank':<6} {'Symbol':<8} {'Events':>8} {'Recover14D':>11} {'Best14D':>10}")
    print("-" * 70)

    for rank, (_, row) in enumerate(top_stocks.iterrows(), 1):
        recovery_rate = row.get('recovery_rate_14_N5', 0) or 0
        best_return = row.get('avg_best_return_N5', 0) or 0
        print(
            f"{rank:<6} {row['symbol']:<8} {int(row['events_N5']):>8} "
            f"{recovery_rate*100:>10.1f}% "
            f"{best_return*100:>+9.2f}%"
        )

    print()
    print("OUTPUT FILES:")
    for name, path in output_paths.items():
        print(f"  {path}")

    print()
    print("=" * 70)
    print(f"Completed: {datetime.now()}")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    try:
        # Run analysis
        results = run_full_analysis()

        # Generate outputs
        output_paths = generate_outputs(results)

        # Print summary
        print_console_summary(results, output_paths)

        return 0

    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
