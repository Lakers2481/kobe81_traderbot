#!/usr/bin/env python3
"""
Kobe Options Lookup - Display options chain and basic Greeks.

Usage:
    python options.py --symbol AAPL
    python options.py --symbol AAPL --expiry 2024-03-15
    python options.py --symbol AAPL --type call --strikes 5

Note: Requires Polygon API with options tier or another options data provider.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.env_loader import load_env

# Polygon options endpoints
POLYGON_OPTIONS_CONTRACTS = "https://api.polygon.io/v3/reference/options/contracts"
POLYGON_OPTIONS_CHAIN = "https://api.polygon.io/v3/snapshot/options/{underlying}"


def fetch_options_contracts(
    symbol: str,
    expiry_gte: Optional[str] = None,
    expiry_lte: Optional[str] = None,
    contract_type: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """
    Fetch options contracts for a symbol from Polygon.
    Returns contract metadata (not live quotes).
    """
    import requests

    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return {"error": "POLYGON_API_KEY not set", "contracts": []}

    params: Dict[str, Any] = {
        "underlying_ticker": symbol.upper(),
        "limit": limit,
        "apiKey": api_key,
        "order": "asc",
        "sort": "expiration_date",
    }

    if expiry_gte:
        params["expiration_date.gte"] = expiry_gte
    if expiry_lte:
        params["expiration_date.lte"] = expiry_lte
    if contract_type:
        params["contract_type"] = contract_type.lower()

    try:
        resp = requests.get(POLYGON_OPTIONS_CONTRACTS, params=params, timeout=15)
        if resp.status_code == 403:
            return {
                "error": "Options data not available with current Polygon subscription",
                "contracts": [],
                "note": "Options data requires Polygon Options tier or higher",
            }
        if resp.status_code != 200:
            return {"error": f"API error: HTTP {resp.status_code}", "contracts": []}

        data = resp.json()
        return {
            "symbol": symbol.upper(),
            "contracts": data.get("results", []),
            "count": len(data.get("results", [])),
        }
    except Exception as e:
        return {"error": str(e), "contracts": []}


def fetch_options_snapshot(symbol: str) -> Dict[str, Any]:
    """
    Fetch options chain snapshot with IV and Greeks from Polygon.
    Requires Options tier subscription.
    """
    import requests

    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return {"error": "POLYGON_API_KEY not set", "data": None}

    url = POLYGON_OPTIONS_CHAIN.format(underlying=symbol.upper())
    params = {"apiKey": api_key}

    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code == 403:
            return {
                "error": "Options snapshot not available with current Polygon subscription",
                "data": None,
                "note": "Options chain snapshot requires Polygon Options tier",
            }
        if resp.status_code != 200:
            return {"error": f"API error: HTTP {resp.status_code}", "data": None}

        data = resp.json()
        return {
            "symbol": symbol.upper(),
            "data": data.get("results", []),
        }
    except Exception as e:
        return {"error": str(e), "data": None}


def display_contracts(contracts: List[Dict[str, Any]], max_display: int = 20) -> None:
    """Display options contracts in a readable format."""
    if not contracts:
        print("No contracts found.")
        return

    print(f"\n{'Symbol':<25} {'Type':<6} {'Strike':>10} {'Expiry':<12}")
    print("-" * 60)

    for i, c in enumerate(contracts[:max_display]):
        ticker = c.get("ticker", "N/A")
        ctype = c.get("contract_type", "N/A").upper()[:4]
        strike = c.get("strike_price", 0)
        expiry = c.get("expiration_date", "N/A")
        print(f"{ticker:<25} {ctype:<6} ${strike:>9.2f} {expiry:<12}")

    if len(contracts) > max_display:
        print(f"\n... and {len(contracts) - max_display} more contracts")


def display_snapshot(data: List[Dict[str, Any]], max_display: int = 10) -> None:
    """Display options snapshot with IV and Greeks."""
    if not data:
        print("No snapshot data available.")
        return

    print(f"\n{'Contract':<25} {'Last':>8} {'IV':>8} {'Delta':>8} {'Theta':>8} {'Gamma':>8}")
    print("-" * 75)

    count = 0
    for item in data:
        if count >= max_display:
            break

        details = item.get("details", {})
        day = item.get("day", {})
        greeks = item.get("greeks", {})
        iv = item.get("implied_volatility")

        ticker = details.get("ticker", item.get("ticker", "N/A"))
        last_price = day.get("close", day.get("last_price", 0)) or 0
        iv_val = iv if iv else 0
        delta = greeks.get("delta", 0) or 0
        theta = greeks.get("theta", 0) or 0
        gamma = greeks.get("gamma", 0) or 0

        print(f"{ticker:<25} ${last_price:>7.2f} {iv_val:>7.1%} {delta:>8.4f} {theta:>8.4f} {gamma:>8.4f}")
        count += 1

    if len(data) > max_display:
        print(f"\n... and {len(data) - max_display} more options")


def check_options_availability(symbol: str) -> bool:
    """Quick check if options data is available for a symbol."""
    try:
        from data.providers.polygon_eod import has_options_polygon
        return has_options_polygon(symbol)
    except ImportError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Kobe Options Lookup")
    ap.add_argument("--dotenv", type=str, default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
                    help="Path to .env file")

    # Required
    ap.add_argument("--symbol", type=str, required=True, help="Underlying symbol (e.g., AAPL)")

    # Filters
    ap.add_argument("--expiry", type=str, help="Filter by expiration date (YYYY-MM-DD)")
    ap.add_argument("--type", type=str, choices=["call", "put"], help="Filter by option type")
    ap.add_argument("--strikes", type=int, default=10, help="Number of strikes to show")

    # Mode
    ap.add_argument("--snapshot", action="store_true", help="Fetch live snapshot with Greeks (requires higher tier)")

    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    symbol = args.symbol.upper()
    print(f"Options Lookup for: {symbol}")
    print("=" * 60)

    # Check API key
    if not os.getenv("POLYGON_API_KEY"):
        print("\nError: POLYGON_API_KEY environment variable not set.")
        print("Options data requires a valid Polygon API key.")
        sys.exit(1)

    # Quick availability check
    has_options = check_options_availability(symbol)
    if not has_options:
        print(f"\nNote: {symbol} may not have options available, or options data")
        print("may require a higher Polygon subscription tier.")

    # Set date range
    expiry_gte = args.expiry if args.expiry else datetime.utcnow().date().isoformat()
    expiry_lte = args.expiry if args.expiry else (datetime.utcnow().date() + timedelta(days=90)).isoformat()

    if args.snapshot:
        # Try to fetch live snapshot with Greeks
        print("\nFetching options snapshot (requires Options tier)...")
        result = fetch_options_snapshot(symbol)

        if "error" in result and result["error"]:
            print(f"\nError: {result['error']}")
            if result.get("note"):
                print(f"Note: {result['note']}")
            print("\n--- Alternative: Fetching contract list instead ---")
            # Fall back to contract list
            args.snapshot = False
        else:
            display_snapshot(result.get("data", []), max_display=args.strikes * 2)
            return

    if not args.snapshot:
        # Fetch contract metadata
        print(f"\nFetching options contracts...")
        print(f"  Expiry range: {expiry_gte} to {expiry_lte}")
        if args.type:
            print(f"  Type filter: {args.type}")

        result = fetch_options_contracts(
            symbol,
            expiry_gte=expiry_gte,
            expiry_lte=expiry_lte,
            contract_type=args.type,
            limit=args.strikes * 4,  # Get enough for calls and puts
        )

        if "error" in result and result["error"]:
            print(f"\nError: {result['error']}")
            if result.get("note"):
                print(f"Note: {result['note']}")
            print("\n" + "=" * 60)
            print("OPTIONS DATA NOT AVAILABLE")
            print("=" * 60)
            print("\nTo access options data, you need:")
            print("  1. Polygon.io Options tier subscription")
            print("  2. Or integrate with another options data provider:")
            print("     - TD Ameritrade API")
            print("     - IBKR API")
            print("     - CBOE Data")
            print("\nCurrent implementation is a placeholder for future integration.")
            sys.exit(0)

        print(f"\nFound {result.get('count', 0)} contracts")
        display_contracts(result.get("contracts", []), max_display=args.strikes * 2)

    print("\n" + "-" * 60)
    print("Note: For live IV and Greeks, use --snapshot (requires Options tier)")


if __name__ == "__main__":
    main()
