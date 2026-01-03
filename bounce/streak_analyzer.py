"""
Streak Analyzer Module for Bounce Analysis

Vectorized calculation of:
- Consecutive down-day streaks
- Forward recovery metrics (7D window)
- Event detection (first-hit only)

NO LOOKAHEAD BIAS - All metrics use ONLY future data from event_date+1 onwards.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def calculate_streaks_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate consecutive down-day streaks (vectorized, NO loops).

    A down day is defined as: Close[t] < Close[t-1]
    Streak_n[t] = number of consecutive down days ending at day t

    Args:
        df: DataFrame with 'close' column (and 'date' column)

    Returns:
        DataFrame with added columns: down, streak_group, streak_len
    """
    df = df.copy()

    # Ensure sorted by date
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)

    # Down day: close < previous close
    df['down'] = (df['close'] < df['close'].shift(1)).astype(int)

    # First row has no previous, so down = 0
    df.loc[df.index[0], 'down'] = 0

    # Calculate streak length using cumsum trick
    # Reset counter when not a down day
    df['streak_group'] = (~df['down'].astype(bool)).cumsum()
    df['streak_len'] = df.groupby('streak_group')['down'].cumsum()

    return df


def calculate_forward_metrics(
    df: pd.DataFrame,
    event_idx: int,
    event_close: float,
    window: int = 7,
) -> Dict:
    """
    Calculate forward-looking metrics for a single event.

    CRITICAL: Uses ONLY bars from event_idx+1 to event_idx+window.
    NO LOOKAHEAD - event_idx bar is NOT included in forward metrics.

    Args:
        df: Full DataFrame for the ticker
        event_idx: Index of the event row
        event_close: Close price at event
        window: Forward window size (default 7 trading days)

    Returns:
        Dict with:
            - recovered_7d_close: bool (close >= event_close within window)
            - recovered_7d_high: bool (high >= event_close within window)
            - days_to_recover_close: int or None (1-7)
            - days_to_recover_high: int or None (1-7)
            - best_close_return_pct: float (max close / event_close - 1)
            - max_high_return_pct: float (max high / event_close - 1)
            - max_drawdown_pct: float (min low / event_close - 1, negative)
            - forward_days_available: int
            - edge_case_flag: str or None
    """
    result = {
        "recovered_7d_close": False,
        "recovered_7d_high": False,
        "days_to_recover_close": None,
        "days_to_recover_high": None,
        "best_close_return_pct": None,
        "max_high_return_pct": None,
        "max_drawdown_pct": None,
        "forward_days_available": 0,
        "edge_case_flag": None,
    }

    # Get forward slice (event_idx+1 to event_idx+window inclusive)
    start_idx = event_idx + 1
    end_idx = min(event_idx + window + 1, len(df))

    forward_df = df.iloc[start_idx:end_idx].copy()
    result["forward_days_available"] = len(forward_df)

    # Edge case: insufficient forward data
    if len(forward_df) < window:
        result["edge_case_flag"] = "insufficient_forward_data"

    if len(forward_df) == 0:
        return result

    # Recovery by CLOSE: first day where close >= event_close
    close_recovery = forward_df[forward_df['close'] >= event_close]
    if len(close_recovery) > 0:
        result["recovered_7d_close"] = True
        # Days to recover = position in forward window (1-based)
        first_recovery_idx = close_recovery.index[0]
        result["days_to_recover_close"] = list(forward_df.index).index(first_recovery_idx) + 1

    # Recovery by HIGH: first day where high >= event_close
    high_recovery = forward_df[forward_df['high'] >= event_close]
    if len(high_recovery) > 0:
        result["recovered_7d_high"] = True
        first_recovery_idx = high_recovery.index[0]
        result["days_to_recover_high"] = list(forward_df.index).index(first_recovery_idx) + 1

    # Best close return in window
    max_close = forward_df['close'].max()
    result["best_close_return_pct"] = (max_close / event_close - 1) * 100

    # Best high return in window
    max_high = forward_df['high'].max()
    result["max_high_return_pct"] = (max_high / event_close - 1) * 100

    # Max drawdown (worst pain) in window
    min_low = forward_df['low'].min()
    result["max_drawdown_pct"] = (min_low / event_close - 1) * 100

    return result


def detect_events(
    df: pd.DataFrame,
    streak_levels: List[int] = None,
) -> pd.DataFrame:
    """
    Detect events when streak FIRST hits each level N.

    CRITICAL: Events are detected at the FIRST time the streak reaches level N.
    A 5-day streak generates 5 separate events (one for each level 1-5).

    Args:
        df: DataFrame with streak_len column
        streak_levels: List of streak levels to detect (default 1-7)

    Returns:
        DataFrame with event rows
    """
    if streak_levels is None:
        streak_levels = [1, 2, 3, 4, 5, 6, 7]

    events = []

    for N in streak_levels:
        # Event occurs when streak_len == N (first time reaching this level)
        event_mask = df['streak_len'] == N
        event_rows = df[event_mask].copy()

        if len(event_rows) > 0:
            event_rows['streak_n'] = N
            events.append(event_rows)

    if events:
        return pd.concat(events, ignore_index=True)
    else:
        return pd.DataFrame()


def analyze_ticker(
    df: pd.DataFrame,
    symbol: str,
    streak_levels: List[int] = None,
    window: int = 7,
    source_used: str = "polygon",
    adjustment_basis: str = "split_adjusted",
) -> List[Dict]:
    """
    Analyze a single ticker for bounce events.

    Args:
        df: OHLCV DataFrame for ticker
        symbol: Ticker symbol
        streak_levels: Streak levels to analyze (default 1-7)
        window: Forward window for recovery (default 7)
        source_used: Data source
        adjustment_basis: Price adjustment basis

    Returns:
        List of event dicts
    """
    if streak_levels is None:
        streak_levels = [1, 2, 3, 4, 5, 6, 7]

    if df is None or len(df) < 2:
        return []

    # Calculate streaks
    df = calculate_streaks_vectorized(df)

    # Detect events
    events_df = detect_events(df, streak_levels)

    if len(events_df) == 0:
        return []

    # Calculate forward metrics for each event
    all_events = []

    for _, event_row in events_df.iterrows():
        event_idx = event_row.name if isinstance(event_row.name, int) else df.index.get_loc(event_row.name)

        # Get original index in main df
        original_idx = df.index.get_loc(event_row.name) if event_row.name in df.index else event_idx

        event_close = event_row['close']
        prior_close = df.iloc[original_idx - 1]['close'] if original_idx > 0 else None

        # Calculate forward metrics
        forward_metrics = calculate_forward_metrics(
            df, original_idx, event_close, window
        )

        event_data = {
            "ticker": symbol,
            "event_date": event_row.get('date', event_row.get('timestamp')),
            "streak_n": int(event_row['streak_n']),
            "event_close": event_close,
            "prior_close": prior_close,
            "event_return_pct": ((event_close / prior_close - 1) * 100) if prior_close else None,
            "forward_7d_best_close_return_pct": forward_metrics["best_close_return_pct"],
            "forward_7d_max_high_return_pct": forward_metrics["max_high_return_pct"],
            "recovered_7d_close": forward_metrics["recovered_7d_close"],
            "days_to_recover_close": forward_metrics["days_to_recover_close"],
            "recovered_7d_high": forward_metrics["recovered_7d_high"],
            "days_to_recover_high": forward_metrics["days_to_recover_high"],
            "max_drawdown_7d_pct": forward_metrics["max_drawdown_pct"],
            "source_used": source_used,
            "adjustment_basis": adjustment_basis,
            "edge_case_flag": forward_metrics["edge_case_flag"],
        }

        all_events.append(event_data)

    return all_events


def build_events_table(
    ticker_data: Dict[str, pd.DataFrame],
    ticker_metadata: Dict[str, Dict] = None,
    streak_levels: List[int] = None,
    window: int = 7,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build complete events table for all tickers.

    Args:
        ticker_data: Dict[symbol, DataFrame]
        ticker_metadata: Dict[symbol, metadata dict] (optional)
        streak_levels: Streak levels to analyze (default 1-7)
        window: Forward window for recovery
        verbose: Print progress

    Returns:
        DataFrame with all events across all tickers
    """
    if streak_levels is None:
        streak_levels = [1, 2, 3, 4, 5, 6, 7]

    if ticker_metadata is None:
        ticker_metadata = {}

    all_events = []
    processed = 0
    total = len(ticker_data)

    for symbol, df in ticker_data.items():
        metadata = ticker_metadata.get(symbol, {})
        source_used = metadata.get("source_used", "polygon")
        adjustment_basis = metadata.get("adjustment_basis", "split_adjusted")

        events = analyze_ticker(
            df=df,
            symbol=symbol,
            streak_levels=streak_levels,
            window=window,
            source_used=source_used,
            adjustment_basis=adjustment_basis,
        )

        all_events.extend(events)
        processed += 1

        if verbose and processed % 100 == 0:
            print(f"  Analyzed {processed}/{total} tickers ({len(all_events):,} events so far)...")

    if verbose:
        print(f"  Total: {len(all_events):,} events from {processed} tickers")

    if all_events:
        events_df = pd.DataFrame(all_events)

        # Ensure proper date type
        if 'event_date' in events_df.columns:
            events_df['event_date'] = pd.to_datetime(events_df['event_date']).dt.date

        return events_df

    return pd.DataFrame()


def get_current_streaks(
    ticker_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Get current streak status for all tickers.

    Useful for generating today's bounce watchlist.

    Args:
        ticker_data: Dict[symbol, DataFrame]

    Returns:
        DataFrame with: ticker, current_streak, last_close, last_date
    """
    records = []

    for symbol, df in ticker_data.items():
        if df is None or len(df) < 2:
            continue

        df = calculate_streaks_vectorized(df)
        last_row = df.iloc[-1]

        records.append({
            "ticker": symbol,
            "current_streak": int(last_row['streak_len']),
            "last_close": last_row['close'],
            "last_date": last_row.get('date', last_row.get('timestamp')),
            "last_high_20d": df['high'].tail(20).max() if len(df) >= 20 else df['high'].max(),
        })

    result = pd.DataFrame(records)

    if len(result) > 0 and 'last_close' in result.columns:
        result['pct_off_high_20d'] = (result['last_close'] / result['last_high_20d'] - 1) * 100

    return result
