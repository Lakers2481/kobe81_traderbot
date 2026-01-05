"""
Corporate Actions Canary for Price Discontinuity Detection.

FIX (2026-01-05): Added to detect >50% day-over-day price changes that
may indicate splits, dividends, or data quality issues.

This canary runs periodically to:
1. Detect large price discontinuities in historical data
2. Identify potential unadjusted splits or dividends
3. Alert when data quality may be compromised

Usage:
    from data.quality.corporate_actions_canary import (
        run_price_discontinuity_check,
        check_symbol_for_splits,
    )

    # Check specific symbol
    result = check_symbol_for_splits("AAPL", df)

    # Check all symbols in universe
    results = run_price_discontinuity_check(universe_df)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Counter (lazy import)
# =============================================================================

def _get_discontinuity_counter():
    """Lazy import of Prometheus counter to avoid circular imports."""
    try:
        from trade_logging.prometheus_metrics import PROMETHEUS_AVAILABLE
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import Counter
            # Create counter if not exists (singleton pattern)
            if not hasattr(_get_discontinuity_counter, "_counter"):
                from trade_logging.prometheus_metrics import REGISTRY
                _get_discontinuity_counter._counter = Counter(
                    'kobe_price_discontinuity_detected_total',
                    'Price discontinuities detected (potential splits/divs)',
                    ['symbol', 'type'],
                    registry=REGISTRY,
                )
            return _get_discontinuity_counter._counter
    except ImportError:
        pass
    return None


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class DiscontinuityEvent:
    """A detected price discontinuity event."""

    symbol: str
    date: datetime
    prev_close: float
    next_open: float
    pct_change: float
    likely_cause: str  # "split", "reverse_split", "dividend", "data_error", "unknown"


@dataclass
class CanaryResult:
    """Result of canary check for a symbol."""

    symbol: str
    passed: bool
    events: List[DiscontinuityEvent]
    total_bars: int
    check_date: datetime


# =============================================================================
# Detection Functions
# =============================================================================

def calculate_overnight_returns(df: pd.DataFrame) -> pd.Series:
    """
    Calculate overnight returns (prev close to current open).

    Args:
        df: OHLCV DataFrame with 'open' and 'close' columns

    Returns:
        Series of overnight percentage changes
    """
    if "close" not in df.columns or "open" not in df.columns:
        return pd.Series(dtype=float)

    # Overnight return: (today's open - yesterday's close) / yesterday's close
    overnight_pct = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100
    return overnight_pct


def classify_discontinuity(pct_change: float) -> str:
    """
    Classify the likely cause of a price discontinuity.

    Args:
        pct_change: Percentage change (e.g., -50 for 50% drop)

    Returns:
        Classification: "split", "reverse_split", "dividend", "data_error", "unknown"
    """
    abs_change = abs(pct_change)

    # Common split ratios
    if -55 <= pct_change <= -45:  # ~50% drop = 2:1 split
        return "split"
    if -68 <= pct_change <= -62:  # ~66% drop = 3:1 split
        return "split"
    if -77 <= pct_change <= -73:  # ~75% drop = 4:1 split
        return "split"

    # Common reverse split ratios
    if 90 <= pct_change <= 110:  # ~100% gain = 1:2 reverse split
        return "reverse_split"
    if 180 <= pct_change <= 220:  # ~200% gain = 1:3 reverse split
        return "reverse_split"

    # Large dividends (typically 5-20%)
    if -20 <= pct_change <= -5:
        return "dividend"

    # Extreme changes likely data errors
    if abs_change > 80:
        return "data_error"

    return "unknown"


def check_symbol_for_splits(
    symbol: str,
    df: pd.DataFrame,
    threshold_pct: float = 50.0,
) -> CanaryResult:
    """
    Check a single symbol's data for price discontinuities.

    Args:
        symbol: Stock ticker
        df: OHLCV DataFrame for the symbol
        threshold_pct: Minimum % change to flag (default: 50%)

    Returns:
        CanaryResult with detected events
    """
    events = []

    if df.empty or len(df) < 2:
        return CanaryResult(
            symbol=symbol,
            passed=True,
            events=[],
            total_bars=len(df),
            check_date=datetime.now(),
        )

    overnight = calculate_overnight_returns(df)

    # Find discontinuities above threshold
    mask = overnight.abs() >= threshold_pct
    discontinuities = df[mask].copy()

    for idx in discontinuities.index:
        try:
            prev_idx = df.index.get_loc(idx) - 1
            if prev_idx < 0:
                continue

            prev_close = df.iloc[prev_idx]["close"]
            next_open = df.loc[idx, "open"]
            pct_change = overnight.loc[idx]
            likely_cause = classify_discontinuity(pct_change)

            event = DiscontinuityEvent(
                symbol=symbol,
                date=idx if isinstance(idx, datetime) else pd.to_datetime(idx),
                prev_close=prev_close,
                next_open=next_open,
                pct_change=pct_change,
                likely_cause=likely_cause,
            )
            events.append(event)

            # Increment Prometheus counter
            counter = _get_discontinuity_counter()
            if counter:
                counter.labels(symbol=symbol, type=likely_cause).inc()

            logger.warning(
                f"Price discontinuity detected for {symbol} on {event.date}: "
                f"{event.pct_change:.1f}% ({event.likely_cause})"
            )

        except Exception as e:
            logger.debug(f"Error processing discontinuity for {symbol}: {e}")
            continue

    return CanaryResult(
        symbol=symbol,
        passed=len(events) == 0,
        events=events,
        total_bars=len(df),
        check_date=datetime.now(),
    )


def run_price_discontinuity_check(
    data: Dict[str, pd.DataFrame],
    threshold_pct: float = 50.0,
) -> Dict[str, CanaryResult]:
    """
    Run price discontinuity checks across multiple symbols.

    Args:
        data: Dict mapping symbol -> OHLCV DataFrame
        threshold_pct: Minimum % change to flag (default: 50%)

    Returns:
        Dict mapping symbol -> CanaryResult
    """
    results = {}

    for symbol, df in data.items():
        try:
            results[symbol] = check_symbol_for_splits(symbol, df, threshold_pct)
        except Exception as e:
            logger.error(f"Canary check failed for {symbol}: {e}")
            results[symbol] = CanaryResult(
                symbol=symbol,
                passed=False,
                events=[],
                total_bars=0,
                check_date=datetime.now(),
            )

    # Summary log
    failed = [s for s, r in results.items() if not r.passed]
    if failed:
        logger.warning(
            f"Price discontinuity canary: {len(failed)} symbols flagged: {failed[:10]}"
        )
    else:
        logger.info(
            f"Price discontinuity canary: all {len(results)} symbols passed"
        )

    return results


def check_recent_data(
    symbol: str,
    days: int = 30,
    threshold_pct: float = 50.0,
) -> CanaryResult:
    """
    Check recent data for a symbol (fetches from provider).

    Convenience function that fetches recent data and checks for discontinuities.

    Args:
        symbol: Stock ticker
        days: Number of days to check
        threshold_pct: Minimum % change to flag

    Returns:
        CanaryResult
    """
    try:
        from data.providers.polygon_eod import PolygonEODProvider
        from datetime import timedelta

        provider = PolygonEODProvider()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = provider.get_daily_bars(
            symbol,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        return check_symbol_for_splits(symbol, df, threshold_pct)

    except Exception as e:
        logger.error(f"Failed to check recent data for {symbol}: {e}")
        return CanaryResult(
            symbol=symbol,
            passed=False,
            events=[],
            total_bars=0,
            check_date=datetime.now(),
        )


def get_canary_summary(results: Dict[str, CanaryResult]) -> Dict[str, any]:
    """
    Generate summary statistics from canary results.

    Args:
        results: Dict of symbol -> CanaryResult

    Returns:
        Summary dict with statistics
    """
    total = len(results)
    passed = sum(1 for r in results.values() if r.passed)
    failed = total - passed

    all_events = []
    for r in results.values():
        all_events.extend(r.events)

    by_cause = {}
    for event in all_events:
        cause = event.likely_cause
        by_cause[cause] = by_cause.get(cause, 0) + 1

    return {
        "total_symbols": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total > 0 else 0,
        "total_events": len(all_events),
        "events_by_cause": by_cause,
        "symbols_flagged": [s for s, r in results.items() if not r.passed],
    }
