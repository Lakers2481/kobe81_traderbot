#!/usr/bin/env python3
"""
Kobe Earnings Calendar - Track upcoming earnings for positions and universe.

Usage:
    python earnings.py --upcoming
    python earnings.py --upcoming --days 14
    python earnings.py --positions
    python earnings.py --symbol AAPL
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.env_loader import load_env

STATE_DIR = ROOT / "state"
RECONCILE_DIR = STATE_DIR / "reconcile"
UNIVERSES_DIR = ROOT / "universes"

# Polygon earnings endpoint
POLYGON_EARNINGS_URL = "https://api.polygon.io/vX/reference/financials"
# Alternative: Use ticker details for next earnings date
POLYGON_TICKER_DETAILS = "https://api.polygon.io/v3/reference/tickers/{ticker}"


def fetch_earnings_date(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch next earnings date for a symbol from Polygon.
    Note: This uses the ticker details endpoint which may include earnings info.
    Full earnings data requires higher tier subscription.
    """
    import requests

    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return None

    url = POLYGON_TICKER_DETAILS.format(ticker=symbol.upper())
    params = {"apiKey": api_key}

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()
        results = data.get("results", {})

        # Extract any earnings-related fields
        return {
            "symbol": symbol.upper(),
            "name": results.get("name", ""),
            "market_cap": results.get("market_cap"),
            "sic_description": results.get("sic_description", ""),
            # Note: Polygon basic tier may not include next_earnings_date
            "note": "Detailed earnings data requires Polygon financials API or external source",
        }
    except Exception as e:
        return {"symbol": symbol.upper(), "error": str(e)}


def fetch_earnings_calendar_external() -> List[Dict[str, Any]]:
    """
    Placeholder for external earnings calendar fetch.
    Could integrate with:
    - Polygon Financials API (paid tier)
    - Alpha Vantage earnings calendar
    - Yahoo Finance earnings scrape
    - Earnings Whispers API
    """
    print("Note: External earnings calendar integration not configured.")
    print("Returning sample/placeholder data.")

    # Return empty - in production, integrate with actual earnings API
    return []


def get_universe_symbols(universe_file: Optional[str] = None) -> List[str]:
    """Load symbols from universe file."""
    if universe_file:
        path = Path(universe_file)
    else:
        # Try to find a universe file
        for name in ["universe.csv", "sp500.csv", "nasdaq100.csv"]:
            path = UNIVERSES_DIR / name
            if path.exists():
                break
        else:
            return []

    if not path.exists():
        return []

    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower() == "symbol":
            continue
        # Handle CSV format
        symbol = line.split(",")[0].strip().upper()
        if symbol:
            symbols.append(symbol)
    return symbols


def get_current_positions() -> List[str]:
    """Get symbols of current positions."""
    import requests

    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    key_id = os.getenv("ALPACA_API_KEY_ID", "")
    secret = os.getenv("ALPACA_API_SECRET_KEY", "")

    symbols = []

    if key_id and secret:
        headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret,
        }
        try:
            resp = requests.get(f"{base_url}/v2/positions", headers=headers, timeout=10)
            if resp.status_code == 200:
                for pos in resp.json():
                    symbols.append(pos.get("symbol", "").upper())
        except Exception:
            pass

    # Fall back to local state
    if not symbols:
        positions_file = RECONCILE_DIR / "positions.json"
        if positions_file.exists():
            try:
                data = json.loads(positions_file.read_text(encoding="utf-8"))
                for pos in data:
                    symbols.append(pos.get("symbol", "").upper())
            except Exception:
                pass

    return symbols


def check_earnings_soon(symbol: str, days_ahead: int = 7) -> Dict[str, Any]:
    """
    Check if a symbol has earnings within specified days.
    Returns earnings info if available.
    """
    # This is a placeholder - real implementation needs earnings data source
    info = fetch_earnings_date(symbol)
    if not info:
        return {
            "symbol": symbol,
            "earnings_soon": False,
            "days_until": None,
            "note": "Could not fetch earnings data",
        }

    return {
        "symbol": symbol,
        "earnings_soon": False,  # Would be True if we had real earnings dates
        "days_until": None,
        "note": info.get("note", "Earnings date not available in basic API tier"),
    }


def print_earnings_report(
    symbols: List[str],
    days_ahead: int = 7,
    show_all: bool = False,
) -> None:
    """Print earnings report for symbols."""
    if not symbols:
        print("No symbols to check.")
        return

    print(f"\nEarnings Check ({len(symbols)} symbols, {days_ahead}-day window)")
    print("=" * 70)

    earnings_soon = []
    checked = 0
    errors = 0

    for symbol in symbols:
        result = check_earnings_soon(symbol, days_ahead)
        checked += 1

        if result.get("error"):
            errors += 1
            if show_all:
                print(f"[ERROR] {symbol}: {result.get('error')}")
        elif result.get("earnings_soon"):
            earnings_soon.append(result)
            days = result.get("days_until", "?")
            print(f"[EARNINGS] {symbol}: {days} days until earnings")
        elif show_all:
            print(f"[-] {symbol}: No imminent earnings")

    print("\n" + "-" * 70)
    print(f"Checked: {checked} | Earnings Soon: {len(earnings_soon)} | Errors: {errors}")

    if not earnings_soon:
        print("\nNo symbols have earnings within the specified window.")

    print("\n" + "=" * 70)
    print("NOTE: Accurate earnings dates require integration with:")
    print("  - Polygon Financials API (higher tier)")
    print("  - Alpha Vantage Earnings Calendar")
    print("  - Yahoo Finance")
    print("  - Earnings Whispers or similar service")
    print("\nCurrent implementation provides placeholder functionality.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kobe Earnings Calendar")
    ap.add_argument("--dotenv", type=str, default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
                    help="Path to .env file")

    # Actions
    ap.add_argument("--upcoming", action="store_true",
                    help="Show upcoming earnings for universe symbols")
    ap.add_argument("--positions", action="store_true",
                    help="Check earnings for current positions")
    ap.add_argument("--symbol", type=str,
                    help="Check earnings for a specific symbol")

    # Options
    ap.add_argument("--days", type=int, default=7,
                    help="Days ahead to look for earnings (default: 7)")
    ap.add_argument("--universe", type=str,
                    help="Path to universe file")
    ap.add_argument("--all", action="store_true",
                    help="Show all symbols, not just those with earnings")

    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    if not args.upcoming and not args.positions and not args.symbol:
        print("Specify --upcoming, --positions, or --symbol")
        print("Use --help for usage information")
        sys.exit(1)

    if args.symbol:
        print(f"Checking earnings for: {args.symbol.upper()}")
        result = check_earnings_soon(args.symbol.upper(), args.days)
        print(f"\nSymbol: {result['symbol']}")
        print(f"Earnings Soon: {result.get('earnings_soon', False)}")
        if result.get("days_until"):
            print(f"Days Until: {result['days_until']}")
        if result.get("note"):
            print(f"Note: {result['note']}")
        if result.get("error"):
            print(f"Error: {result['error']}")

    elif args.positions:
        print("Checking earnings for current positions...")
        symbols = get_current_positions()
        if not symbols:
            print("No current positions found.")
            print("Ensure Alpaca credentials are set or run reconcile_alpaca.py")
        else:
            print_earnings_report(symbols, args.days, show_all=args.all)

    elif args.upcoming:
        print("Checking earnings for universe symbols...")
        symbols = get_universe_symbols(args.universe)
        if not symbols:
            print("No universe file found.")
            print(f"Looked in: {UNIVERSES_DIR}")
        else:
            print_earnings_report(symbols, args.days, show_all=args.all)


if __name__ == "__main__":
    main()
