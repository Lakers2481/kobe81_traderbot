"""
BounceScore Module for Bounce Analysis

Calculates bounce scores (0-100) based on:
- Recovery rate
- Speed of recovery
- Opportunity size
- Sample size
- Pain tolerance

Implements gates for strategy integration.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def calculate_bounce_score(
    recovery_rate: float,
    avg_days: float,
    avg_return: float,
    events: int,
    avg_drawdown: float,
) -> float:
    """
    Calculate BounceScore (0-100) for a bounce profile.

    Formula:
        BounceScore = (
            recovery_rate * 40 +                              # 0-40 (recovery component)
            (1 - min(avg_days / 7.0, 1.0)) * 20 +            # 0-20 (speed component)
            min(avg_return / 0.10, 1.0) * 20 +               # 0-20 (opportunity component)
            min(events / 50.0, 1.0) * 10 +                   # 0-10 (sample component)
            (1 - min(abs(avg_drawdown) / 0.10, 1.0)) * 10   # 0-10 (pain component)
        )

    Args:
        recovery_rate: Recovery rate (0-1)
        avg_days: Average days to recover (1-7)
        avg_return: Average best 7D return (as decimal, e.g., 0.05 = 5%)
        events: Number of historical events
        avg_drawdown: Average max drawdown (negative number, e.g., -0.03 = -3%)

    Returns:
        BounceScore (0-100)
    """
    # Validate inputs
    if pd.isna(recovery_rate) or pd.isna(avg_days) or pd.isna(avg_return):
        return 0.0

    # Recovery component (40 points max)
    recovery_component = recovery_rate * 40

    # Speed component (20 points max) - faster recovery = higher score
    # If avg_days = 1, full points. If avg_days = 7, zero points.
    if pd.isna(avg_days) or avg_days <= 0:
        speed_component = 0
    else:
        speed_component = (1 - min(avg_days / 7.0, 1.0)) * 20

    # Opportunity component (20 points max) - bigger returns = higher score
    # 10% return gives full points
    if pd.isna(avg_return) or avg_return < 0:
        opportunity_component = 0
    else:
        opportunity_component = min(avg_return / 0.10, 1.0) * 20

    # Sample component (10 points max) - more events = higher confidence
    # 50 events gives full points
    sample_component = min(events / 50.0, 1.0) * 10

    # Pain component (10 points max) - less drawdown = higher score
    # 10% drawdown gives zero points
    if pd.isna(avg_drawdown):
        pain_component = 10  # No data means assume neutral
    else:
        pain_component = (1 - min(abs(avg_drawdown) / 0.10, 1.0)) * 10

    total_score = (
        recovery_component +
        speed_component +
        opportunity_component +
        sample_component +
        pain_component
    )

    return min(max(total_score, 0), 100)


def apply_bounce_gates(
    events: int,
    recovery_rate: float,
    avg_days: float,
) -> Tuple[bool, Optional[str]]:
    """
    Apply FIRM gates for bounce profile.

    Gates:
    - events >= 20
    - recovery_7d_close_rate >= 0.75
    - avg_days_to_recover_7d <= 3.2

    Args:
        events: Number of historical events
        recovery_rate: Recovery rate (0-1)
        avg_days: Average days to recover

    Returns:
        (passed: bool, reject_reason: str or None)
    """
    # Gate 1: Minimum sample size
    if events < 20:
        return False, f"LOW_SAMPLE: {events} events < 20 required"

    # Gate 2: Minimum recovery rate
    if pd.isna(recovery_rate) or recovery_rate < 0.75:
        rate_pct = f"{recovery_rate:.0%}" if pd.notna(recovery_rate) else "N/A"
        return False, f"LOW_RECOVERY: {rate_pct} < 75% required"

    # Gate 3: Maximum recovery time
    if pd.isna(avg_days) or avg_days > 3.2:
        days_str = f"{avg_days:.1f}" if pd.notna(avg_days) else "N/A"
        return False, f"SLOW_RECOVERY: {days_str} days > 3.2 max"

    return True, None


def get_bounce_profile_for_signal(
    ticker: str,
    current_streak: int,
    per_stock_5y: pd.DataFrame,
    per_stock_10y: pd.DataFrame,
    min_events: int = 20,
) -> Dict:
    """
    Get bounce profile for a signal candidate.

    Window selection:
    1. Use 5Y if events_5y >= min_events AND sample_quality == GOOD
    2. Else use 10Y if events_10y >= min_events AND sample_quality == GOOD
    3. Else return bounce_window_used="NONE_LOW_SAMPLE"

    Args:
        ticker: Stock ticker
        current_streak: Current streak level (1-7)
        per_stock_5y: 5Y per-stock summary DataFrame
        per_stock_10y: 10Y per-stock summary DataFrame
        min_events: Minimum events required

    Returns:
        Dict with:
            bounce_window_used: "5Y" / "10Y" / "NONE_LOW_SAMPLE"
            events, recovery_rate, avg_days, avg_return, avg_drawdown
            bounce_score, gate_passed, reject_reason
    """
    result = {
        "ticker": ticker,
        "streak_n": current_streak,
        "bounce_window_used": "NONE_LOW_SAMPLE",
        "events": 0,
        "recovery_rate": None,
        "avg_days": None,
        "avg_return": None,
        "avg_drawdown": None,
        "bounce_score": 0,
        "gate_passed": False,
        "reject_reason": "NO_DATA",
    }

    # Helper to extract profile from summary
    def get_profile(summary_df: pd.DataFrame) -> Optional[Dict]:
        if summary_df is None or len(summary_df) == 0:
            return None

        # Filter for this ticker and streak
        mask = (summary_df['ticker'] == ticker) & (summary_df['streak_n'] == current_streak)
        row = summary_df[mask]

        if len(row) == 0:
            return None

        row = row.iloc[0]

        # Check sample quality
        quality = row.get('sample_quality_flag', 'NO_EVENTS')
        if quality in ['NO_EVENTS', 'INSUFFICIENT_HISTORY']:
            return None

        events = row.get('events', 0)
        if events < min_events:
            return None

        return {
            "events": events,
            "recovery_rate": row.get('recovery_7d_close_rate'),
            "avg_days": row.get('avg_days_to_recover_7d'),
            "avg_return": row.get('avg_best_7d_return'),
            "avg_drawdown": row.get('avg_max_drawdown_7d_pct'),
            "sample_quality": quality,
        }

    # Try 5Y first
    profile_5y = get_profile(per_stock_5y)
    if profile_5y is not None:
        result.update(profile_5y)
        result["bounce_window_used"] = "5Y"
    else:
        # Fall back to 10Y
        profile_10y = get_profile(per_stock_10y)
        if profile_10y is not None:
            result.update(profile_10y)
            result["bounce_window_used"] = "10Y"
        else:
            # No valid data
            result["reject_reason"] = "NO_HISTORICAL_DATA"
            return result

    # Calculate BounceScore
    # Convert percentages to decimals if needed
    avg_return = result["avg_return"]
    avg_drawdown = result["avg_drawdown"]

    if avg_return is not None and abs(avg_return) > 1:
        # Assume percentage, convert to decimal
        avg_return = avg_return / 100
    if avg_drawdown is not None and abs(avg_drawdown) > 1:
        avg_drawdown = avg_drawdown / 100

    result["bounce_score"] = calculate_bounce_score(
        recovery_rate=result["recovery_rate"],
        avg_days=result["avg_days"],
        avg_return=avg_return if avg_return else 0,
        events=result["events"],
        avg_drawdown=avg_drawdown if avg_drawdown else 0,
    )

    # Apply gates
    gate_passed, reject_reason = apply_bounce_gates(
        events=result["events"],
        recovery_rate=result["recovery_rate"],
        avg_days=result["avg_days"],
    )
    result["gate_passed"] = gate_passed
    result["reject_reason"] = reject_reason

    return result


def adjust_signal_for_bounce(
    signal: Dict,
    profile: Dict,
) -> Dict:
    """
    Adjust signal parameters based on bounce profile.

    Adjustments:
    - confidence *= (bounce_score / 100)
    - size: if bounce_score < 60: size *= 0.5; >= 80: full size
    - target: consider p95_best_return * 0.8
    - stop: consider abs(median_drawdown) * 1.2
    - time_stop: consider median_days + 2

    Args:
        signal: Original signal dict
        profile: Bounce profile dict

    Returns:
        Adjusted signal dict
    """
    adjusted = signal.copy()

    bounce_score = profile.get("bounce_score", 0)

    # Adjust confidence
    if "confidence" in adjusted:
        adjusted["original_confidence"] = adjusted["confidence"]
        adjusted["confidence"] = adjusted["confidence"] * (bounce_score / 100)

    # Adjust position size multiplier
    if bounce_score >= 80:
        adjusted["bounce_size_factor"] = 1.0
    elif bounce_score >= 60:
        adjusted["bounce_size_factor"] = 0.75
    else:
        adjusted["bounce_size_factor"] = 0.5

    # Add bounce profile info
    adjusted["bounce_score"] = bounce_score
    adjusted["bounce_window_used"] = profile.get("bounce_window_used", "NONE")
    adjusted["bounce_gate_passed"] = profile.get("gate_passed", False)
    adjusted["bounce_reject_reason"] = profile.get("reject_reason")
    adjusted["bounce_events"] = profile.get("events", 0)
    adjusted["bounce_recovery_rate"] = profile.get("recovery_rate")
    adjusted["bounce_avg_days"] = profile.get("avg_days")

    return adjusted


def rank_signals_by_bounce(
    signals: list,
    per_stock_5y: pd.DataFrame,
    per_stock_10y: pd.DataFrame,
    require_gate_pass: bool = True,
) -> list:
    """
    Rank signals by BounceScore.

    Args:
        signals: List of signal dicts (must have 'ticker' and 'streak' keys)
        per_stock_5y: 5Y per-stock summary DataFrame
        per_stock_10y: 10Y per-stock summary DataFrame
        require_gate_pass: If True, filter out signals that fail gates

    Returns:
        List of signals sorted by BounceScore (descending)
    """
    enriched_signals = []

    for signal in signals:
        ticker = signal.get('ticker') or signal.get('symbol')
        streak = signal.get('streak') or signal.get('current_streak', 1)

        profile = get_bounce_profile_for_signal(
            ticker=ticker,
            current_streak=streak,
            per_stock_5y=per_stock_5y,
            per_stock_10y=per_stock_10y,
        )

        # Filter if required
        if require_gate_pass and not profile.get("gate_passed", False):
            continue

        adjusted = adjust_signal_for_bounce(signal, profile)
        enriched_signals.append(adjusted)

    # Sort by bounce_score descending
    enriched_signals.sort(key=lambda x: x.get("bounce_score", 0), reverse=True)

    return enriched_signals


def get_bounce_score_breakdown(
    recovery_rate: float,
    avg_days: float,
    avg_return: float,
    events: int,
    avg_drawdown: float,
) -> Dict:
    """
    Get detailed breakdown of BounceScore components.

    Useful for debugging and reporting.

    Args:
        Same as calculate_bounce_score

    Returns:
        Dict with component scores and total
    """
    if pd.isna(recovery_rate) or pd.isna(avg_days) or pd.isna(avg_return):
        return {
            "recovery_component": 0,
            "speed_component": 0,
            "opportunity_component": 0,
            "sample_component": 0,
            "pain_component": 0,
            "total_score": 0,
            "error": "Missing required inputs",
        }

    recovery_component = recovery_rate * 40

    if pd.isna(avg_days) or avg_days <= 0:
        speed_component = 0
    else:
        speed_component = (1 - min(avg_days / 7.0, 1.0)) * 20

    if pd.isna(avg_return) or avg_return < 0:
        opportunity_component = 0
    else:
        opportunity_component = min(avg_return / 0.10, 1.0) * 20

    sample_component = min(events / 50.0, 1.0) * 10

    if pd.isna(avg_drawdown):
        pain_component = 10
    else:
        pain_component = (1 - min(abs(avg_drawdown) / 0.10, 1.0)) * 10

    total_score = (
        recovery_component +
        speed_component +
        opportunity_component +
        sample_component +
        pain_component
    )

    return {
        "recovery_component": round(recovery_component, 2),
        "speed_component": round(speed_component, 2),
        "opportunity_component": round(opportunity_component, 2),
        "sample_component": round(sample_component, 2),
        "pain_component": round(pain_component, 2),
        "total_score": round(min(max(total_score, 0), 100), 2),
        "inputs": {
            "recovery_rate": recovery_rate,
            "avg_days": avg_days,
            "avg_return": avg_return,
            "events": events,
            "avg_drawdown": avg_drawdown,
        },
    }
