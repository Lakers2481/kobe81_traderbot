"""
Event Table Module for Bounce Analysis

Generates:
- Overall summary statistics by streak level
- Per-stock summary statistics
- 5Y derived from 10Y filtering
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compute_overall_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overall statistics by streak level.

    Args:
        events_df: Events DataFrame from build_events_table

    Returns:
        DataFrame with overall stats per streak_n (1-7)
    """
    if events_df is None or len(events_df) == 0:
        return pd.DataFrame()

    # Filter out edge cases for stats (but keep for counts)
    valid_events = events_df[events_df['edge_case_flag'].isna()].copy()

    summary_rows = []

    for streak_n in sorted(events_df['streak_n'].unique()):
        all_streak = events_df[events_df['streak_n'] == streak_n]
        streak_events = valid_events[valid_events['streak_n'] == streak_n]

        if len(streak_events) == 0:
            continue

        # Basic counts
        total_events = len(all_streak)
        valid_events_count = len(streak_events)

        # Recovery rates
        recovery_close_rate = streak_events['recovered_7d_close'].mean()
        recovery_high_rate = streak_events['recovered_7d_high'].mean()

        # Days to recover (recovered only)
        recovered_close = streak_events[streak_events['recovered_7d_close']]
        streak_events[streak_events['recovered_7d_high']]

        avg_days_close = recovered_close['days_to_recover_close'].mean() if len(recovered_close) > 0 else np.nan
        median_days_close = recovered_close['days_to_recover_close'].median() if len(recovered_close) > 0 else np.nan

        # Day distribution (for close recovery)
        day_dist = {}
        for day in range(1, 8):
            day_pct = (streak_events['days_to_recover_close'] == day).sum() / len(streak_events) if len(streak_events) > 0 else 0
            day_dist[f'day{day}_pct'] = day_pct * 100

        not_recovered_pct = (~streak_events['recovered_7d_close']).sum() / len(streak_events) * 100 if len(streak_events) > 0 else 0

        # Return metrics
        avg_best_return = streak_events['forward_7d_best_close_return_pct'].mean()
        median_best_return = streak_events['forward_7d_best_close_return_pct'].median()
        p95_best_return = streak_events['forward_7d_best_close_return_pct'].quantile(0.95)

        # Drawdown metrics
        avg_drawdown = streak_events['max_drawdown_7d_pct'].mean()
        median_drawdown = streak_events['max_drawdown_7d_pct'].median()

        row = {
            'streak_n': streak_n,
            'events': total_events,
            'valid_events': valid_events_count,
            'recovery_7d_close_rate': recovery_close_rate,
            'recovery_7d_high_rate': recovery_high_rate,
            'avg_days_to_recover_7d': avg_days_close,
            'median_days_to_recover_7d': median_days_close,
            **day_dist,
            'not_recovered_pct': not_recovered_pct,
            'avg_best_7d_return': avg_best_return,
            'median_best_7d_return': median_best_return,
            'p95_best_7d_return': p95_best_return,
            'avg_max_drawdown_7d_pct': avg_drawdown,
            'median_max_drawdown_7d_pct': median_drawdown,
        }

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def compute_per_stock_summary(
    events_df: pd.DataFrame,
    all_symbols: List[str] = None,
    streak_levels: List[int] = None,
    min_events_for_good: int = 20,
) -> pd.DataFrame:
    """
    Compute per-ticker statistics for ALL tickers x 7 streaks.

    Args:
        events_df: Events DataFrame
        all_symbols: List of all symbols (to emit NO_EVENTS rows)
        streak_levels: Streak levels (default 1-7)
        min_events_for_good: Min events for GOOD quality flag

    Returns:
        DataFrame with per-ticker stats (ticker x streak_n rows)
    """
    if streak_levels is None:
        streak_levels = [1, 2, 3, 4, 5, 6, 7]

    if all_symbols is None:
        all_symbols = events_df['ticker'].unique().tolist() if len(events_df) > 0 else []

    # Get symbols with events
    symbols_with_events = set(events_df['ticker'].unique()) if len(events_df) > 0 else set()

    # Filter valid events
    valid_events = events_df[events_df['edge_case_flag'].isna()].copy() if len(events_df) > 0 else pd.DataFrame()

    summary_rows = []

    for symbol in all_symbols:
        for streak_n in streak_levels:
            # Default row (NO_EVENTS)
            row = {
                'ticker': symbol,
                'streak_n': streak_n,
                'events': 0,
                'recovery_7d_close_rate': np.nan,
                'recovery_7d_high_rate': np.nan,
                'avg_days_to_recover_7d': np.nan,
                'median_days_to_recover_7d': np.nan,
                'avg_best_7d_return': np.nan,
                'median_best_7d_return': np.nan,
                'p95_best_7d_return': np.nan,
                'avg_max_drawdown_7d_pct': np.nan,
                'median_max_drawdown_7d_pct': np.nan,
                'last_event_date': None,
                'sample_quality_flag': 'NO_EVENTS',
            }

            if symbol not in symbols_with_events:
                row['sample_quality_flag'] = 'INSUFFICIENT_HISTORY'
                summary_rows.append(row)
                continue

            # Get events for this ticker + streak
            ticker_events = valid_events[
                (valid_events['ticker'] == symbol) &
                (valid_events['streak_n'] == streak_n)
            ]

            if len(ticker_events) == 0:
                summary_rows.append(row)
                continue

            # Calculate stats
            row['events'] = len(ticker_events)

            # Recovery rates
            row['recovery_7d_close_rate'] = ticker_events['recovered_7d_close'].mean()
            row['recovery_7d_high_rate'] = ticker_events['recovered_7d_high'].mean()

            # Days to recover (recovered only)
            recovered = ticker_events[ticker_events['recovered_7d_close']]
            if len(recovered) > 0:
                row['avg_days_to_recover_7d'] = recovered['days_to_recover_close'].mean()
                row['median_days_to_recover_7d'] = recovered['days_to_recover_close'].median()

            # Return metrics
            row['avg_best_7d_return'] = ticker_events['forward_7d_best_close_return_pct'].mean()
            row['median_best_7d_return'] = ticker_events['forward_7d_best_close_return_pct'].median()
            row['p95_best_7d_return'] = ticker_events['forward_7d_best_close_return_pct'].quantile(0.95)

            # Drawdown metrics
            row['avg_max_drawdown_7d_pct'] = ticker_events['max_drawdown_7d_pct'].mean()
            row['median_max_drawdown_7d_pct'] = ticker_events['max_drawdown_7d_pct'].median()

            # Last event date
            row['last_event_date'] = ticker_events['event_date'].max()

            # Sample quality flag
            if len(ticker_events) >= min_events_for_good:
                row['sample_quality_flag'] = 'GOOD'
            elif len(ticker_events) >= 10:
                row['sample_quality_flag'] = 'GOOD'  # Still usable
            else:
                row['sample_quality_flag'] = 'LOW_SAMPLE'

            summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def derive_5y_from_10y(
    events_10y: pd.DataFrame,
    ticker_max_dates: Dict[str, datetime] = None,
) -> pd.DataFrame:
    """
    Filter 10Y events to get 5Y events (do NOT recompute).

    Args:
        events_10y: 10-year events DataFrame
        ticker_max_dates: Dict of ticker -> max_date (if None, uses global max)

    Returns:
        5-year events DataFrame (filtered from 10Y)
    """
    if events_10y is None or len(events_10y) == 0:
        return pd.DataFrame()

    # Convert event_date to datetime for filtering
    events = events_10y.copy()
    events['event_date_dt'] = pd.to_datetime(events['event_date'])

    # Calculate 5Y cutoff per ticker or globally
    if ticker_max_dates is None:
        # Use global max date
        global_max = events['event_date_dt'].max()
        cutoff = global_max - timedelta(days=5 * 365)
        events_5y = events[events['event_date_dt'] >= cutoff].copy()
    else:
        # Per-ticker cutoff
        def get_cutoff(row):
            max_date = ticker_max_dates.get(row['ticker'])
            if max_date is None:
                return row['event_date_dt']  # Keep all if no max date
            cutoff = pd.to_datetime(max_date) - timedelta(days=5 * 365)
            return cutoff

        events['cutoff'] = events.apply(get_cutoff, axis=1)
        events_5y = events[events['event_date_dt'] >= events['cutoff']].copy()
        events_5y = events_5y.drop(columns=['cutoff'])

    # Clean up
    events_5y = events_5y.drop(columns=['event_date_dt'])

    return events_5y


def get_ticker_max_dates(ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, datetime]:
    """
    Get max date per ticker from data.

    Args:
        ticker_data: Dict[symbol, DataFrame]

    Returns:
        Dict[symbol, max_date]
    """
    max_dates = {}

    for symbol, df in ticker_data.items():
        if df is not None and len(df) > 0:
            if 'date' in df.columns:
                max_dates[symbol] = pd.to_datetime(df['date']).max()
            elif 'timestamp' in df.columns:
                max_dates[symbol] = pd.to_datetime(df['timestamp']).max()

    return max_dates


def generate_data_health_block(
    quality_report: pd.DataFrame,
    events_df: pd.DataFrame,
    years: int,
) -> str:
    """
    Generate DATA HEALTH markdown block.

    Args:
        quality_report: Quality report from data loading
        events_df: Events DataFrame
        years: Window years (10 or 5)

    Returns:
        Markdown string
    """
    total = len(quality_report) if quality_report is not None else 0
    loaded = quality_report['bars_count'].gt(0).sum() if quality_report is not None else 0
    failed = total - loaded

    # Source breakdown
    polygon_only = 0
    fallback_used = 0

    if quality_report is not None and 'source_used' in quality_report.columns:
        polygon_only = (quality_report['source_used'] == 'polygon').sum()
        fallback_used = quality_report['source_used'].isin(['yfinance', 'stooq']).sum()

    # Validation
    validated = quality_report['validated'].sum() if quality_report is not None and 'validated' in quality_report.columns else 0
    flagged = quality_report['mismatch_flagged'].sum() if quality_report is not None and 'mismatch_flagged' in quality_report.columns else 0
    rejected = quality_report['rejected'].sum() if quality_report is not None and 'rejected' in quality_report.columns else 0

    # Coverage
    if quality_report is not None and 'bars_count' in quality_report.columns:
        bars = quality_report['bars_count']
        years_coverage = bars / 252
        min_years = years_coverage.min()
        median_years = years_coverage.median()
        max_years = years_coverage.max()
    else:
        min_years = median_years = max_years = 0

    # Events
    total_events = len(events_df) if events_df is not None else 0
    edge_cases = events_df['edge_case_flag'].notna().sum() if events_df is not None and 'edge_case_flag' in events_df.columns else 0

    md = f"""## DATA HEALTH ({years}Y Window)

| Metric | Value |
|--------|-------|
| Tickers Attempted | {total} |
| Tickers Loaded | {loaded} |
| Tickers Failed | {failed} |
| Polygon Only | {polygon_only} |
| Fallback Used | {fallback_used} |
| Validated Sample | {validated} |
| Mismatch Flagged | {flagged} |
| Rejected (Bad Data) | {rejected} |
| Coverage (Years) | {min_years:.1f} / {median_years:.1f} / {max_years:.1f} (min/med/max) |
| Total Events | {total_events:,} |
| Edge Cases (Insufficient Forward) | {edge_cases:,} |
"""

    return md
