#!/usr/bin/env python3
"""
Kobe Trading System - Signals Viewer

View latest generated signals and signal history from logs.
Filter by strategy and symbol.

Usage:
    python signals.py                  # Show latest signals
    python signals.py --latest         # Show most recent signals
    python signals.py --history        # Show signal history from logs
    python signals.py --strategy rsi2  # Filter by strategy
    python signals.py --symbol AAPL    # Filter by symbol
    python signals.py --tail 50        # Show last 50 signals
    python signals.py --dotenv /path/to/.env
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.env_loader import load_env

# Default paths
STATE_DIR = ROOT / "state"
SIGNALS_FILE = STATE_DIR / "signals.json"
LOGS_DIR = ROOT / "logs"
DEFAULT_DOTENV = "C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env"


def load_latest_signals() -> List[Dict[str, Any]]:
    """Load latest signals from state file."""
    signals: List[Dict[str, Any]] = []

    if SIGNALS_FILE.exists():
        try:
            data = json.loads(SIGNALS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                signals.extend(data)
            elif isinstance(data, dict):
                signals.extend(data.get("signals", []))
        except Exception as e:
            print(f"[WARN] Failed to load signals.json: {e}")

    return signals


def load_signal_history() -> List[Dict[str, Any]]:
    """Load signal history from log files."""
    signals: List[Dict[str, Any]] = []

    # Parse events.jsonl for signal events
    events_file = LOGS_DIR / "events.jsonl"
    if events_file.exists():
        try:
            for line in events_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if event.get("event") in (
                        "signal_generated",
                        "signal",
                        "entry_signal",
                        "exit_signal",
                        "strategy_signal",
                    ):
                        signals.append(event)
                except Exception:
                    continue
        except Exception as e:
            print(f"[WARN] Failed to parse events.jsonl: {e}")

    # Also check for dedicated signals log
    signals_log = LOGS_DIR / "signals.jsonl"
    if signals_log.exists():
        try:
            for line in signals_log.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    sig = json.loads(line)
                    signals.append(sig)
                except Exception:
                    continue
        except Exception:
            pass

    return signals


def parse_timestamp(ts: Any) -> Optional[datetime]:
    """Parse various timestamp formats."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        ts_str = str(ts)
        if "T" in ts_str:
            ts_str = ts_str.replace("Z", "+00:00")
            if "+" in ts_str:
                ts_str = ts_str.split("+")[0]
            return datetime.fromisoformat(ts_str)
        if len(ts_str) >= 10:
            return datetime.strptime(ts_str[:10], "%Y-%m-%d")
    except Exception:
        pass
    return None


def filter_signals(
    signals: List[Dict[str, Any]],
    strategy: Optional[str] = None,
    symbol: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter signals by strategy and/or symbol."""
    filtered = signals

    if strategy:
        strategy_lower = strategy.lower()
        filtered = [
            s for s in filtered
            if strategy_lower in (
                s.get("strategy", "")
                or s.get("signal_source", "")
                or s.get("reason", "")
                or s.get("event", "")
            ).lower()
            or (strategy_lower == "rsi2" and "rsi" in str(s).lower())
            or (strategy_lower == "ibs" and "ibs" in str(s).lower())
        ]

    if symbol:
        symbol_upper = symbol.upper()
        filtered = [
            s for s in filtered
            if s.get("symbol", "").upper() == symbol_upper
        ]

    return filtered


def sort_signals(signals: List[Dict[str, Any]], reverse: bool = True) -> List[Dict[str, Any]]:
    """Sort signals by timestamp (newest first by default)."""
    def get_ts(s: Dict[str, Any]) -> datetime:
        ts = parse_timestamp(s.get("timestamp") or s.get("ts"))
        return ts or datetime.min

    return sorted(signals, key=get_ts, reverse=reverse)


def get_strategy_from_signal(signal: Dict[str, Any]) -> str:
    """Extract strategy name from signal."""
    strat = signal.get("strategy") or signal.get("signal_source") or ""
    if not strat:
        reason = signal.get("reason", "").lower()
        if "rsi" in reason:
            strat = "RSI-2"
        elif "ibs" in reason:
            strat = "IBS"
        else:
            strat = "Unknown"
    return strat


def format_side(side: str) -> str:
    """Format side with visual indicator."""
    side_upper = side.upper()
    if side_upper in ("LONG", "BUY"):
        return "[+] LONG"
    elif side_upper in ("SHORT", "SELL"):
        return "[-] SHORT"
    elif side_upper == "EXIT":
        return "[X] EXIT"
    return f"[ ] {side_upper}"


def print_signals_table(signals: List[Dict[str, Any]], detailed: bool = False):
    """Print signals in table format."""
    if not signals:
        print("\n[INFO] No signals to display.")
        return

    print()
    if detailed:
        print(
            f"{'Timestamp':<20} {'Symbol':<8} {'Side':<12} {'Strategy':<10} "
            f"{'Entry':>10} {'Stop':>10} {'RSI/IBS':>8} {'Reason':<30}"
        )
        print("-" * 120)
    else:
        print(
            f"{'Timestamp':<20} {'Symbol':<8} {'Side':<12} {'Strategy':<10} "
            f"{'Entry':>10} {'Stop':>10}"
        )
        print("-" * 80)

    for sig in signals:
        # Timestamp
        ts = parse_timestamp(sig.get("timestamp") or sig.get("ts"))
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "-"

        symbol = sig.get("symbol", "???")
        side = sig.get("side", "-")
        strategy = get_strategy_from_signal(sig)

        entry_price = sig.get("entry_price")
        entry_str = f"${float(entry_price):.2f}" if entry_price else "-"

        stop_loss = sig.get("stop_loss")
        stop_str = f"${float(stop_loss):.2f}" if stop_loss else "-"

        if detailed:
            # Get indicator value
            indicator = sig.get("rsi2") or sig.get("ibs") or sig.get("rsi")
            ind_str = f"{float(indicator):.2f}" if indicator else "-"

            reason = sig.get("reason", "-")
            if len(reason) > 30:
                reason = reason[:27] + "..."

            print(
                f"{ts_str:<20} {symbol:<8} {format_side(side):<12} {strategy:<10} "
                f"{entry_str:>10} {stop_str:>10} {ind_str:>8} {reason:<30}"
            )
        else:
            print(
                f"{ts_str:<20} {symbol:<8} {format_side(side):<12} {strategy:<10} "
                f"{entry_str:>10} {stop_str:>10}"
            )


def print_signal_summary(signals: List[Dict[str, Any]]):
    """Print summary statistics for signals."""
    if not signals:
        return

    # Count by strategy
    by_strategy: Dict[str, int] = {}
    by_side: Dict[str, int] = {"LONG": 0, "SHORT": 0, "EXIT": 0, "OTHER": 0}
    by_symbol: Dict[str, int] = {}

    for sig in signals:
        # Strategy
        strat = get_strategy_from_signal(sig)
        by_strategy[strat] = by_strategy.get(strat, 0) + 1

        # Side
        side = sig.get("side", "").upper()
        if side in ("LONG", "BUY"):
            by_side["LONG"] += 1
        elif side in ("SHORT", "SELL"):
            by_side["SHORT"] += 1
        elif side == "EXIT":
            by_side["EXIT"] += 1
        else:
            by_side["OTHER"] += 1

        # Symbol
        symbol = sig.get("symbol", "???")
        by_symbol[symbol] = by_symbol.get(symbol, 0) + 1

    print(f"\n{'='*60}")
    print(" Signal Summary")
    print(f"{'='*60}")
    print()
    print(f"  {'Total Signals:':<25} {len(signals)}")
    print()

    # By strategy
    print("  By Strategy:")
    for strat, count in sorted(by_strategy.items()):
        print(f"    {strat:<20} {count:>6}")
    print()

    # By side
    print("  By Side:")
    for side, count in by_side.items():
        if count > 0:
            print(f"    {side:<20} {count:>6}")
    print()

    # Top symbols
    print("  Top Symbols:")
    for symbol, count in sorted(by_symbol.items(), key=lambda x: -x[1])[:10]:
        print(f"    {symbol:<20} {count:>6}")


def show_signals(
    latest: bool = False,
    history: bool = False,
    strategy: Optional[str] = None,
    symbol: Optional[str] = None,
    tail: Optional[int] = None,
    detailed: bool = False,
):
    """Main signal display function."""
    signals: List[Dict[str, Any]] = []

    if history:
        print("[INFO] Loading signal history from logs...")
        signals = load_signal_history()
        source = "Log History"
    elif latest:
        print("[INFO] Loading latest signals...")
        signals = load_latest_signals()
        source = "Latest State"
    else:
        # Default: combine both
        print("[INFO] Loading signals from all sources...")
        signals = load_latest_signals() + load_signal_history()
        source = "All Sources"

    if not signals:
        print(f"\n[{source}] No signals found.")
        print("       Checked: state/signals.json, logs/events.jsonl, logs/signals.jsonl")
        return

    print(f"[INFO] Loaded {len(signals)} signals from {source}")

    # Apply filters
    signals = filter_signals(signals, strategy=strategy, symbol=symbol)

    if not signals:
        print("\n[INFO] No signals match the filter criteria.")
        return

    # Sort by time (newest first)
    signals = sort_signals(signals, reverse=True)

    # Apply tail limit
    if tail:
        signals = signals[:tail]

    print(f"\n{'#'*80}")
    print(f"#  KOBE TRADING SYSTEM - SIGNALS VIEWER")
    print(f"#  Source: {source}")
    print(f"#  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if strategy:
        print(f"#  Filter - Strategy: {strategy}")
    if symbol:
        print(f"#  Filter - Symbol: {symbol.upper()}")
    if tail:
        print(f"#  Showing: Last {tail} signals")
    print(f"{'#'*80}")

    print_signals_table(signals, detailed=detailed)
    print_signal_summary(signals)
    print()


def main():
    ap = argparse.ArgumentParser(description="Kobe Trading System - View Signals")
    ap.add_argument(
        "--dotenv",
        type=str,
        default=DEFAULT_DOTENV,
        help="Path to .env file",
    )
    ap.add_argument(
        "--latest",
        action="store_true",
        help="Show only latest signals from state file",
    )
    ap.add_argument(
        "--history",
        action="store_true",
        help="Show signal history from logs",
    )
    ap.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Filter by strategy (e.g., rsi2, ibs)",
    )
    ap.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Filter by symbol (e.g., AAPL)",
    )
    ap.add_argument(
        "--tail",
        type=int,
        default=None,
        help="Show only the last N signals",
    )
    ap.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show detailed output with indicator values and reasons",
    )
    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
        print(f"[INFO] Loaded {len(loaded)} env vars from {dotenv}")

    show_signals(
        latest=args.latest,
        history=args.history,
        strategy=args.strategy,
        symbol=args.symbol,
        tail=args.tail,
        detailed=args.detailed,
    )


if __name__ == "__main__":
    main()
