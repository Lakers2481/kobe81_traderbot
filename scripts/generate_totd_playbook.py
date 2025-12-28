#!/usr/bin/env python3
"""
Generate Trade of the Day (TOTD) Playbook.

Builds a decision packet from the latest TOTD signal and generates
a human-readable playbook using Claude API or deterministic fallback.

Usage:
    python scripts/generate_totd_playbook.py --date 2025-12-27 --output reports/totd/
    python scripts/generate_totd_playbook.py --symbol AAPL --output reports/totd/
"""

import argparse
import json
import sys
from datetime import datetime, date
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from explainability.decision_packet import (
    DecisionPacket,
    build_decision_packet,
    ExecutionPlan,
)
from explainability.playbook_generator import PlaybookGenerator


def load_totd_signal(signal_date: date) -> dict:
    """Load TOTD signal from daily output files."""
    # Check various locations for TOTD data
    paths_to_check = [
        project_root / f"reports/totd/totd_{signal_date.isoformat()}.json",
        project_root / f"outputs/daily/totd_{signal_date.isoformat()}.json",
        project_root / f"logs/daily_picks.csv",
    ]

    for path in paths_to_check:
        if path.exists():
            if path.suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            elif path.suffix == ".csv":
                # Parse CSV for the date
                import csv
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_date = row.get("date", "")
                        if row_date == signal_date.isoformat():
                            return row

    return {}


def build_packet_from_signal(signal: dict, signal_date: date) -> DecisionPacket:
    """Build decision packet from signal data."""
    # Extract fields from signal
    symbol = signal.get("symbol", signal.get("ticker", "UNKNOWN"))
    side = signal.get("side", "buy")
    strategy = signal.get("strategy", signal.get("strategy_name", "unknown"))

    # Build execution plan if we have price data
    exec_plan = None
    if "entry_price" in signal or "entry" in signal:
        entry = float(signal.get("entry_price", signal.get("entry", 0)))
        stop = float(signal.get("stop_loss", signal.get("stop", entry * 0.95)))
        target = float(signal.get("take_profit", signal.get("target", entry * 1.10)))
        size = int(signal.get("position_size", signal.get("shares", 10)))

        risk = abs(entry - stop) * size
        reward = abs(target - entry) * size
        rr = reward / risk if risk > 0 else 0

        exec_plan = {
            "entry_price": entry,
            "stop_loss": stop,
            "take_profit": target,
            "position_size": size,
            "notional": entry * size,
            "risk_amount": risk,
            "reward_amount": reward,
            "reward_risk_ratio": rr,
        }

    # Extract feature values if available
    feature_values = {}
    for key, val in signal.items():
        if key.startswith("feat_") or key in ["rsi", "atr", "volume_ratio", "sma_dist"]:
            try:
                feature_values[key] = float(val)
            except (ValueError, TypeError):
                pass

    # Extract ML outputs if available
    ml_outputs = {}
    for key in ["probability", "proba", "confidence", "score"]:
        if key in signal:
            try:
                ml_outputs[key] = float(signal[key])
            except (ValueError, TypeError):
                pass

    # Build reasons
    reasons = []
    if "reason" in signal:
        reasons.append(signal["reason"])
    if "reasons" in signal:
        reasons.extend(signal["reasons"])

    # Build packet
    packet = build_decision_packet(
        symbol=symbol,
        side=side,
        strategy_name=strategy,
        signal={"reasons": reasons, "description": signal.get("description", "")},
        ml_result=ml_outputs if ml_outputs else None,
        execution_plan=exec_plan,
        feature_values=feature_values if feature_values else None,
        sentiment_score=signal.get("sentiment_score"),
        sentiment_source=signal.get("sentiment_source"),
        metadata={"source_date": signal_date.isoformat()},
    )

    return packet


def main():
    parser = argparse.ArgumentParser(description="Generate TOTD playbook")
    parser.add_argument(
        "--date",
        type=str,
        default=date.today().isoformat(),
        help="Date for TOTD (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Specific symbol to generate playbook for",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/totd",
        help="Output directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["md", "html", "both"],
        default="both",
        help="Output format",
    )
    parser.add_argument(
        "--no-claude",
        action="store_true",
        help="Use deterministic generation only (no Claude API)",
    )
    parser.add_argument(
        "--packet-only",
        action="store_true",
        help="Only generate decision packet JSON, no playbook",
    )

    args = parser.parse_args()

    # Parse date
    try:
        signal_date = date.fromisoformat(args.date)
    except ValueError:
        print(f"Invalid date format: {args.date}")
        sys.exit(1)

    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating TOTD playbook for {signal_date}")

    # Load signal
    signal = load_totd_signal(signal_date)

    if not signal:
        print(f"No TOTD signal found for {signal_date}")
        print("Creating sample packet for demonstration...")

        # Create sample packet
        packet = build_decision_packet(
            symbol=args.symbol or "SAMPLE",
            side="buy",
            strategy_name="IBS_RSI",
            signal={
                "reasons": ["20-day high breakout", "Volume confirmation"],
                "description": "Price broke above 20-day high with above-average volume",
            },
            execution_plan={
                "entry_price": 150.00,
                "stop_loss": 145.00,
                "take_profit": 160.00,
                "position_size": 10,
                "notional": 1500.00,
                "risk_amount": 50.00,
                "reward_amount": 100.00,
                "reward_risk_ratio": 2.0,
            },
            feature_values={
                "rsi_14": 55.0,
                "atr_14": 3.50,
                "volume_ratio": 1.5,
                "sma_200_dist": 0.05,
            },
            metadata={"note": "Sample packet - no actual signal found"},
        )
    else:
        # Build packet from signal
        if args.symbol:
            # Filter for specific symbol if requested
            if isinstance(signal, list):
                signal = next((s for s in signal if s.get("symbol") == args.symbol), {})
            elif signal.get("symbol") != args.symbol:
                print(f"Symbol {args.symbol} not found in TOTD signal")
                sys.exit(1)

        packet = build_packet_from_signal(signal, signal_date)

    # Save decision packet
    packet_path = output_dir / f"{signal_date}_{packet.symbol}_packet.json"
    packet.save(packet_path)
    print(f"Decision packet saved: {packet_path}")

    if args.packet_only:
        print("Packet-only mode, skipping playbook generation")
        return

    # Generate playbook
    generator = PlaybookGenerator(use_claude=not args.no_claude)
    playbook = generator.generate_from_packet(packet)

    # Save playbook
    saved_paths = generator.save_playbook(playbook, output_dir, format=args.format)

    print(f"Playbook generated using: {playbook.generation_method}")
    for path in saved_paths:
        print(f"  Saved: {path}")

    # Print summary
    print("\n" + "=" * 60)
    print(playbook.executive_summary)
    print("=" * 60)


if __name__ == "__main__":
    main()

