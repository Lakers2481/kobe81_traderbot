"""
Verify Order Rejection Rate (Jim Simons Standard)
==================================================

Analyzes order rejection reasons and rates to ensure quality gates are working.

Key Metrics:
    - Fill rate: % of signals that result in fills
    - Rejection reasons breakdown
    - Quality gate effectiveness
    - Time-based rejection patterns

Jim Simons Standard:
    - Fill rate 40-60% = GOOD (quality gates working)
    - Fill rate > 80% = TOO PERMISSIVE (weak filters)
    - Fill rate < 20% = TOO RESTRICTIVE (missing opportunities)
    - Top rejection reason should be quality-related, not technical errors

Usage:
    python scripts/verify_rejection_rate.py
    python scripts/verify_rejection_rate.py --lookback-days 30
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RejectionAnalysis:
    """Analysis of order rejections."""
    total_signals: int
    total_fills: int
    total_rejections: int
    fill_rate: float
    rejection_rate: float

    rejection_reasons: Dict[str, int]
    rejection_reasons_pct: Dict[str, float]

    hourly_fill_rates: Dict[int, float]
    passed: bool
    failure_reasons: List[str]

    # Thresholds
    min_fill_rate: float = 0.20
    max_fill_rate: float = 0.80
    target_fill_rate: float = 0.50


# ============================================================================
# Analysis Functions
# ============================================================================

def load_trades(lookback_days: int = 30) -> List[Dict]:
    """Load trades from logs/trades.jsonl."""
    trades_file = Path("logs/trades.jsonl")
    if not trades_file.exists():
        print(f"\n[WARNING] No trades file found at {trades_file}")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    trades = []

    with open(trades_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                trade = json.loads(line)
                ts = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                if ts >= cutoff:
                    trades.append(trade)
            except Exception:
                pass  # Skip unparseable trades

    return trades


def categorize_rejection_reason(notes: str) -> str:
    """Categorize rejection reason into buckets."""
    if not notes:
        return "unknown"

    notes_lower = notes.lower()

    # Quality gates (GOOD - these should be most common)
    if "quality_gate" in notes_lower or "score" in notes_lower or "confidence" in notes_lower:
        return "quality_gate"
    if "liquidity_gate" in notes_lower or "liquidity" in notes_lower or "adv" in notes_lower:
        return "liquidity_gate"
    if "kill_zone" in notes_lower or "time" in notes_lower:
        return "kill_zone"

    # Technical errors (BAD - should be rare)
    if "http" in notes_lower or "403" in notes_lower or "401" in notes_lower:
        return "api_error"
    if "quotes" in notes_lower or "no_quotes" in notes_lower:
        return "no_quotes"
    if "broker" in notes_lower or "alpaca" in notes_lower:
        return "broker_error"

    # Other
    return "other"


def analyze_rejections(trades: List[Dict]) -> RejectionAnalysis:
    """Analyze rejection patterns."""
    total_signals = len(trades)
    fills = [t for t in trades if t['status'] == 'FILLED']
    rejections = [t for t in trades if t['status'] == 'REJECTED']

    total_fills = len(fills)
    total_rejections = len(rejections)
    fill_rate = total_fills / total_signals if total_signals > 0 else 0.0
    rejection_rate = total_rejections / total_signals if total_signals > 0 else 0.0

    # Categorize rejection reasons
    rejection_reasons_raw = Counter()
    for rej in rejections:
        reason = categorize_rejection_reason(rej.get('notes', ''))
        rejection_reasons_raw[reason] += 1

    # Convert to percentages
    rejection_reasons_pct = {
        reason: (count / total_rejections * 100) if total_rejections > 0 else 0.0
        for reason, count in rejection_reasons_raw.items()
    }

    # Hourly fill rates
    hourly_trades = {}
    for trade in trades:
        try:
            ts = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
            hour = ts.hour
            if hour not in hourly_trades:
                hourly_trades[hour] = {'fills': 0, 'total': 0}
            hourly_trades[hour]['total'] += 1
            if trade['status'] == 'FILLED':
                hourly_trades[hour]['fills'] += 1
        except Exception:
            pass

    hourly_fill_rates = {
        hour: (stats['fills'] / stats['total']) if stats['total'] > 0 else 0.0
        for hour, stats in hourly_trades.items()
    }

    # Check if passed
    failure_reasons = []

    if fill_rate < 0.20:
        failure_reasons.append(f"Fill rate {fill_rate:.1%} < 20% - TOO RESTRICTIVE")
    elif fill_rate > 0.80:
        failure_reasons.append(f"Fill rate {fill_rate:.1%} > 80% - TOO PERMISSIVE")

    # Check top rejection reason
    if rejection_reasons_raw:
        top_reason = max(rejection_reasons_raw, key=rejection_reasons_raw.get)
        if top_reason in ['api_error', 'broker_error']:
            failure_reasons.append(
                f"Top rejection reason is '{top_reason}' - should be quality_gate"
            )

    passed = len(failure_reasons) == 0

    return RejectionAnalysis(
        total_signals=total_signals,
        total_fills=total_fills,
        total_rejections=total_rejections,
        fill_rate=fill_rate,
        rejection_rate=rejection_rate,
        rejection_reasons=dict(rejection_reasons_raw),
        rejection_reasons_pct=rejection_reasons_pct,
        hourly_fill_rates=hourly_fill_rates,
        passed=passed,
        failure_reasons=failure_reasons,
    )


def print_report(analysis: RejectionAnalysis):
    """Pretty print rejection analysis report."""
    print("\n" + "=" * 80)
    print("ORDER REJECTION RATE ANALYSIS")
    print("Jim Simons / Renaissance Technologies Standard")
    print("=" * 80)

    print(f"\nOverall Statistics:")
    print(f"  Total Signals: {analysis.total_signals}")
    print(f"  Total Fills: {analysis.total_fills}")
    print(f"  Total Rejections: {analysis.total_rejections}")
    print(f"  Fill Rate: {analysis.fill_rate:.1%}")
    print(f"  Rejection Rate: {analysis.rejection_rate:.1%}")

    print(f"\nRejection Reasons:")
    print(f"  {'Reason':<20} {'Count':<10} {'Percentage'}")
    print(f"  {'-'*50}")
    for reason, count in sorted(analysis.rejection_reasons.items(), key=lambda x: x[1], reverse=True):
        pct = analysis.rejection_reasons_pct[reason]
        print(f"  {reason:<20} {count:<10} {pct:>6.1f}%")

    print(f"\nHourly Fill Rate Pattern:")
    if analysis.hourly_fill_rates:
        print(f"  {'Hour (ET)':<12} {'Fill Rate'}")
        print(f"  {'-'*30}")
        for hour in sorted(analysis.hourly_fill_rates.keys()):
            fill_rate = analysis.hourly_fill_rates[hour]
            print(f"  {hour:02d}:00        {fill_rate:>6.1%}")

    print(f"\nQuality Assessment:")
    if analysis.fill_rate < 0.20:
        print(f"  [WARN]  Fill rate {analysis.fill_rate:.1%} is TOO RESTRICTIVE")
        print(f"  Action: Loosen quality gates or check for technical issues")
    elif analysis.fill_rate > 0.80:
        print(f"  [WARN]  Fill rate {analysis.fill_rate:.1%} is TOO PERMISSIVE")
        print(f"  Action: Tighten quality gates to filter weak signals")
    else:
        print(f"  [OK] Fill rate {analysis.fill_rate:.1%} is in acceptable range (20-80%)")

    # Check top rejection reason
    if analysis.rejection_reasons:
        top_reason = max(analysis.rejection_reasons, key=analysis.rejection_reasons.get)
        top_count = analysis.rejection_reasons[top_reason]
        top_pct = analysis.rejection_reasons_pct[top_reason]

        print(f"\n  Top Rejection Reason: {top_reason} ({top_count} signals, {top_pct:.1f}%)")

        if top_reason in ['quality_gate', 'liquidity_gate', 'kill_zone']:
            print(f"  [OK] Quality gates are working as intended")
        else:
            print(f"  [WARN]  Top rejection is '{top_reason}' - investigate technical issues")

    # Verdict
    print("\n" + "=" * 80)
    if analysis.passed:
        print("[OK] PASSED - Rejection rate indicates healthy quality gates")
    else:
        print("[FAIL] NEEDS ATTENTION:")
        for reason in analysis.failure_reasons:
            print(f"  - {reason}")
    print("=" * 80)


def save_report(analysis: RejectionAnalysis):
    """Save report to file."""
    output_file = Path("reports/REJECTION_RATE_VERIFICATION.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("# Order Rejection Rate Verification\n")
        f.write("**Jim Simons / Renaissance Technologies Standard**\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Status:** {'PASSED' if analysis.passed else 'NEEDS ATTENTION'}\n\n")
        f.write("---\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Signals:** {analysis.total_signals}\n")
        f.write(f"- **Fill Rate:** {analysis.fill_rate:.1%}\n")
        f.write(f"- **Rejection Rate:** {analysis.rejection_rate:.1%}\n")

        if analysis.rejection_reasons:
            top_reason = max(analysis.rejection_reasons, key=analysis.rejection_reasons.get)
            f.write(f"- **Top Rejection Reason:** {top_reason}\n")

        f.write("\n---\n\n")

        f.write("## Rejection Breakdown\n\n")
        f.write("| Reason | Count | Percentage |\n")
        f.write("|--------|-------|------------|\n")
        for reason, count in sorted(analysis.rejection_reasons.items(), key=lambda x: x[1], reverse=True):
            pct = analysis.rejection_reasons_pct[reason]
            f.write(f"| {reason} | {count} | {pct:.1f}% |\n")

        f.write("\n---\n\n")

        f.write("## Hourly Fill Rate\n\n")
        if analysis.hourly_fill_rates:
            f.write("| Hour (ET) | Fill Rate |\n")
            f.write("|-----------|------------|\n")
            for hour in sorted(analysis.hourly_fill_rates.keys()):
                fill_rate = analysis.hourly_fill_rates[hour]
                f.write(f"| {hour:02d}:00 | {fill_rate:.1%} |\n")

        f.write("\n---\n\n")

        if not analysis.passed:
            f.write("## Issues Detected\n\n")
            for reason in analysis.failure_reasons:
                f.write(f"- {reason}\n")
            f.write("\n---\n\n")

        f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Verification Standard:** Jim Simons / Renaissance Technologies\n")

    print(f"\n[OK] Report saved to {output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify order rejection rate (Jim Simons standard)"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Number of days to analyze (default: 30)",
    )

    args = parser.parse_args()

    # Load trades
    print(f"\nLoading trades from last {args.lookback_days} days...")
    trades = load_trades(lookback_days=args.lookback_days)

    if not trades:
        print("[ERROR] No trades found in logs/trades.jsonl")
        print("Run trading system to collect trade data.")
        sys.exit(1)

    print(f"[OK] Loaded {len(trades)} trades")

    # Analyze rejections
    analysis = analyze_rejections(trades)

    # Print results
    print_report(analysis)

    # Save to file
    save_report(analysis)

    # Exit code
    sys.exit(0 if analysis.passed else 1)


if __name__ == "__main__":
    main()
