#!/usr/bin/env python3
"""
Kobe Watchlist Manager - Create and manage custom symbol watchlists.

Usage:
    python watchlist.py --create momentum --symbols AAPL,MSFT,NVDA
    python watchlist.py --add momentum --symbols AMD,TSLA
    python watchlist.py --remove momentum --symbols TSLA
    python watchlist.py --list
    python watchlist.py --list --name momentum
    python watchlist.py --check momentum  # Check for signals on watchlist symbols
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.env_loader import load_env

STATE_DIR = ROOT / "state"
WATCHLISTS_FILE = STATE_DIR / "watchlists.json"


def ensure_state_dir() -> None:
    """Ensure state directory exists."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def load_watchlists() -> Dict[str, Dict[str, Any]]:
    """Load all watchlists from JSON file."""
    if not WATCHLISTS_FILE.exists():
        return {}
    try:
        return json.loads(WATCHLISTS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, Exception):
        return {}


def save_watchlists(watchlists: Dict[str, Dict[str, Any]]) -> None:
    """Save all watchlists to JSON file."""
    ensure_state_dir()
    WATCHLISTS_FILE.write_text(json.dumps(watchlists, indent=2), encoding="utf-8")


def create_watchlist(name: str, symbols: List[str], description: str = "") -> Dict[str, Any]:
    """Create a new watchlist."""
    watchlists = load_watchlists()
    if name in watchlists:
        raise ValueError(f"Watchlist '{name}' already exists. Use --add to update.")

    symbols = [s.upper().strip() for s in symbols if s.strip()]
    watchlist = {
        "name": name,
        "symbols": symbols,
        "description": description,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    watchlists[name] = watchlist
    save_watchlists(watchlists)
    return watchlist


def add_to_watchlist(name: str, symbols: List[str]) -> Dict[str, Any]:
    """Add symbols to an existing watchlist."""
    watchlists = load_watchlists()
    if name not in watchlists:
        raise ValueError(f"Watchlist '{name}' does not exist. Use --create first.")

    new_symbols = [s.upper().strip() for s in symbols if s.strip()]
    existing = set(watchlists[name]["symbols"])
    for s in new_symbols:
        existing.add(s)
    watchlists[name]["symbols"] = sorted(list(existing))
    watchlists[name]["updated_at"] = datetime.utcnow().isoformat()
    save_watchlists(watchlists)
    return watchlists[name]


def remove_from_watchlist(name: str, symbols: List[str]) -> Dict[str, Any]:
    """Remove symbols from a watchlist."""
    watchlists = load_watchlists()
    if name not in watchlists:
        raise ValueError(f"Watchlist '{name}' does not exist.")

    remove_symbols = set(s.upper().strip() for s in symbols if s.strip())
    existing = watchlists[name]["symbols"]
    watchlists[name]["symbols"] = [s for s in existing if s not in remove_symbols]
    watchlists[name]["updated_at"] = datetime.utcnow().isoformat()
    save_watchlists(watchlists)
    return watchlists[name]


def delete_watchlist(name: str) -> bool:
    """Delete a watchlist entirely."""
    watchlists = load_watchlists()
    if name not in watchlists:
        return False
    del watchlists[name]
    save_watchlists(watchlists)
    return True


def list_watchlists(name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """List all watchlists or a specific one."""
    watchlists = load_watchlists()
    if name:
        if name in watchlists:
            return {name: watchlists[name]}
        return {}
    return watchlists


def check_watchlist_signals(name: str) -> Dict[str, Any]:
    """
    Check for trading signals on symbols in a watchlist.
    Returns signal status for each symbol.
    """
    watchlists = load_watchlists()
    if name not in watchlists:
        raise ValueError(f"Watchlist '{name}' does not exist.")

    symbols = watchlists[name]["symbols"]
    results = {
        "watchlist": name,
        "checked_at": datetime.utcnow().isoformat(),
        "symbols": {},
    }

    # Try to import strategies and check for signals
    try:
        from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
        from strategies.ibs.strategy import IBSStrategy
        from data.providers.polygon_eod import fetch_daily_bars_polygon

        rsi2_strat = ConnorsRSI2Strategy()
        ibs_strat = IBSStrategy()

        end_date = datetime.utcnow().date().isoformat()
        start_date = (datetime.utcnow().date() - timedelta(days=90)).isoformat()

        for symbol in symbols:
            try:
                df = fetch_daily_bars_polygon(symbol, start_date, end_date)
                if df.empty:
                    results["symbols"][symbol] = {"status": "no_data", "signals": []}
                    continue

                signals = []
                # Check RSI2 signals
                try:
                    rsi2_signals = rsi2_strat.scan_signals_over_time(df)
                    if not rsi2_signals.empty:
                        latest = rsi2_signals.iloc[-1] if len(rsi2_signals) > 0 else None
                        if latest is not None:
                            signals.append({
                                "strategy": "connors_rsi2",
                                "direction": str(latest.get("direction", "N/A")),
                                "date": str(latest.get("timestamp", "N/A")),
                            })
                except Exception:
                    pass

                # Check IBS signals
                try:
                    ibs_signals = ibs_strat.scan_signals_over_time(df)
                    if not ibs_signals.empty:
                        latest = ibs_signals.iloc[-1] if len(ibs_signals) > 0 else None
                        if latest is not None:
                            signals.append({
                                "strategy": "ibs",
                                "direction": str(latest.get("direction", "N/A")),
                                "date": str(latest.get("timestamp", "N/A")),
                            })
                except Exception:
                    pass

                results["symbols"][symbol] = {
                    "status": "checked",
                    "signals": signals,
                    "has_signal": len(signals) > 0,
                }

            except Exception as e:
                results["symbols"][symbol] = {"status": f"error: {e}", "signals": []}

    except ImportError as e:
        # Strategies not available, return basic check
        for symbol in symbols:
            results["symbols"][symbol] = {
                "status": "unchecked",
                "signals": [],
                "note": f"Strategy import failed: {e}",
            }

    return results


def print_watchlist(name: str, watchlist: Dict[str, Any]) -> None:
    """Pretty-print a watchlist."""
    symbols = watchlist.get("symbols", [])
    created = watchlist.get("created_at", "N/A")
    updated = watchlist.get("updated_at", "N/A")
    desc = watchlist.get("description", "")

    print(f"Watchlist: {name}")
    print(f"  Description: {desc if desc else 'none'}")
    print(f"  Symbols ({len(symbols)}): {', '.join(symbols) if symbols else 'empty'}")
    print(f"  Created: {created}")
    print(f"  Updated: {updated}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Kobe Watchlist Manager")
    ap.add_argument("--dotenv", type=str, default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
                    help="Path to .env file")

    # Actions (mutually exclusive)
    action_group = ap.add_mutually_exclusive_group()
    action_group.add_argument("--create", type=str, metavar="NAME",
                              help="Create a new watchlist")
    action_group.add_argument("--add", type=str, metavar="NAME",
                              help="Add symbols to existing watchlist")
    action_group.add_argument("--remove", type=str, metavar="NAME",
                              help="Remove symbols from watchlist")
    action_group.add_argument("--delete", type=str, metavar="NAME",
                              help="Delete a watchlist entirely")
    action_group.add_argument("--list", action="store_true",
                              help="List all watchlists")
    action_group.add_argument("--check", type=str, metavar="NAME",
                              help="Check watchlist for signals")

    # Options
    ap.add_argument("--name", type=str, help="Watchlist name (for --list filter)")
    ap.add_argument("--symbols", type=str, help="Comma-separated symbols")
    ap.add_argument("--description", type=str, default="", help="Watchlist description")

    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    try:
        if args.create:
            if not args.symbols:
                print("Error: --symbols is required for --create")
                sys.exit(1)
            symbols = [s.strip() for s in args.symbols.split(",")]
            watchlist = create_watchlist(args.create, symbols, args.description)
            print(f"Created watchlist '{args.create}':")
            print_watchlist(args.create, watchlist)

        elif args.add:
            if not args.symbols:
                print("Error: --symbols is required for --add")
                sys.exit(1)
            symbols = [s.strip() for s in args.symbols.split(",")]
            watchlist = add_to_watchlist(args.add, symbols)
            print(f"Updated watchlist '{args.add}':")
            print_watchlist(args.add, watchlist)

        elif args.remove:
            if not args.symbols:
                print("Error: --symbols is required for --remove")
                sys.exit(1)
            symbols = [s.strip() for s in args.symbols.split(",")]
            watchlist = remove_from_watchlist(args.remove, symbols)
            print(f"Updated watchlist '{args.remove}':")
            print_watchlist(args.remove, watchlist)

        elif args.delete:
            if delete_watchlist(args.delete):
                print(f"Deleted watchlist '{args.delete}'")
            else:
                print(f"Watchlist '{args.delete}' not found")

        elif args.list:
            watchlists = list_watchlists(name=args.name)
            if not watchlists:
                if args.name:
                    print(f"Watchlist '{args.name}' not found")
                else:
                    print("No watchlists found. Use --create to make one.")
            else:
                print(f"Found {len(watchlists)} watchlist(s):")
                print("=" * 60)
                for name, wl in watchlists.items():
                    print_watchlist(name, wl)

        elif args.check:
            print(f"Checking signals for watchlist '{args.check}'...")
            results = check_watchlist_signals(args.check)
            print(f"\nWatchlist: {results['watchlist']}")
            print(f"Checked at: {results['checked_at']}")
            print("=" * 60)
            for symbol, data in results["symbols"].items():
                status = data.get("status", "unknown")
                signals = data.get("signals", [])
                has_signal = data.get("has_signal", False)
                signal_icon = "[SIGNAL]" if has_signal else "[-]"
                print(f"{signal_icon} {symbol}: {status}")
                for sig in signals:
                    print(f"    - {sig['strategy']}: {sig['direction']} on {sig['date']}")

        else:
            print("Specify an action: --create, --add, --remove, --delete, --list, or --check")
            print("Use --help for usage information")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
