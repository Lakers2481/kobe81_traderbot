#!/usr/bin/env python3
"""
Kobe Trading System - Positions Viewer

Show open positions from local state and/or broker (Alpaca).
Displays symbol, entry price, current price, unrealized P&L,
days held, stop distance, and total exposure.

Usage:
    python positions.py             # Default: show local positions
    python positions.py --live      # Fetch live positions from broker
    python positions.py --local     # Show local positions only
    python positions.py --dotenv /path/to/.env
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.env_loader import load_env

# Default paths
STATE_DIR = ROOT / "state"
POSITIONS_FILE = STATE_DIR / "positions.json"
DEFAULT_DOTENV = "C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env"


def _alpaca_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY_ID", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET_KEY", ""),
    }


def _alpaca_base() -> str:
    return os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")


def fetch_live_positions() -> List[Dict[str, Any]]:
    """Fetch positions from Alpaca API."""
    url = f"{_alpaca_base()}/v2/positions"
    try:
        r = requests.get(url, headers=_alpaca_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch live positions: {e}")
        return []


def fetch_current_price(symbol: str) -> Optional[float]:
    """Fetch current market price from Alpaca."""
    base = _alpaca_base()
    # Try latest quote endpoint
    url = f"{base}/v2/stocks/{symbol.upper()}/quotes/latest"
    try:
        r = requests.get(url, headers=_alpaca_headers(), timeout=5)
        if r.status_code == 200:
            data = r.json()
            quote = data.get("quote", {})
            ask = quote.get("ap") or quote.get("ask_price")
            bid = quote.get("bp") or quote.get("bid_price")
            if ask and bid:
                return (float(ask) + float(bid)) / 2
            return float(ask) if ask else float(bid) if bid else None
    except Exception:
        pass
    return None


def load_local_positions() -> List[Dict[str, Any]]:
    """Load positions from local state file."""
    if not POSITIONS_FILE.exists():
        return []
    try:
        data = json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Handle dict format: {symbol: position_data}
            return [{"symbol": k, **v} for k, v in data.items()]
    except Exception as e:
        print(f"[WARN] Failed to load local positions: {e}")
    return []


def format_currency(value: float) -> str:
    """Format value as currency with color indicator."""
    if value >= 0:
        return f"+${value:,.2f}"
    return f"-${abs(value):,.2f}"


def format_pct(value: float) -> str:
    """Format value as percentage."""
    if value >= 0:
        return f"+{value:.2f}%"
    return f"{value:.2f}%"


def calc_days_held(entry_time: Optional[str]) -> Optional[int]:
    """Calculate days held from entry timestamp."""
    if not entry_time:
        return None
    try:
        if "T" in entry_time:
            entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
        else:
            entry_dt = datetime.strptime(entry_time[:10], "%Y-%m-%d")
        return (datetime.now() - entry_dt.replace(tzinfo=None)).days
    except Exception:
        return None


def print_table_header():
    """Print table header."""
    print("-" * 115)
    print(
        f"{'Symbol':<10} {'Side':<6} {'Qty':>8} {'Entry':>10} {'Current':>10} "
        f"{'Unrealized P&L':>16} {'P&L %':>8} {'Days':>6} {'Stop Dist':>10}"
    )
    print("-" * 115)


def print_position_row(pos: Dict[str, Any], current_price: Optional[float] = None):
    """Print a single position row."""
    symbol = pos.get("symbol", "???")
    side = pos.get("side", "long").upper()
    qty = int(pos.get("qty", 0))

    # Entry price - handle different field names
    entry_price = pos.get("avg_entry_price") or pos.get("entry_price") or pos.get("avg_cost", 0)
    entry_price = float(entry_price) if entry_price else 0.0

    # Current price - from pos data or fetched
    curr_price = current_price
    if curr_price is None:
        curr_price = pos.get("current_price") or pos.get("market_value")
        if pos.get("market_value") and qty:
            curr_price = float(pos.get("market_value", 0)) / qty
        else:
            curr_price = float(curr_price) if curr_price else entry_price

    # Unrealized P&L
    unrealized_pl = pos.get("unrealized_pl")
    if unrealized_pl is not None:
        unrealized_pl = float(unrealized_pl)
    elif entry_price and curr_price:
        if side in ("LONG", "BUY"):
            unrealized_pl = (curr_price - entry_price) * qty
        else:
            unrealized_pl = (entry_price - curr_price) * qty
    else:
        unrealized_pl = 0.0

    # P&L percentage
    pnl_pct = 0.0
    if entry_price > 0:
        pnl_pct = ((curr_price - entry_price) / entry_price) * 100
        if side in ("SHORT", "SELL"):
            pnl_pct = -pnl_pct

    # Days held
    days = calc_days_held(pos.get("entry_time") or pos.get("created_at"))
    days_str = str(days) if days is not None else "-"

    # Stop distance
    stop_loss = pos.get("stop_loss")
    stop_dist_str = "-"
    if stop_loss and curr_price:
        stop_dist = abs(curr_price - float(stop_loss))
        stop_pct = (stop_dist / curr_price) * 100 if curr_price else 0
        stop_dist_str = f"${stop_dist:.2f} ({stop_pct:.1f}%)"

    print(
        f"{symbol:<10} {side:<6} {qty:>8} ${entry_price:>9.2f} ${curr_price:>9.2f} "
        f"{format_currency(unrealized_pl):>16} {format_pct(pnl_pct):>8} {days_str:>6} {stop_dist_str:>10}"
    )


def show_positions(live: bool = False, local: bool = False, fetch_prices: bool = False):
    """Display positions table."""
    positions: List[Dict[str, Any]] = []

    if live:
        print("[INFO] Fetching live positions from broker...")
        positions = fetch_live_positions()
        source = "Broker (Alpaca)"
    else:
        print("[INFO] Loading local positions...")
        positions = load_local_positions()
        source = "Local State"

        # Optionally fetch current prices for local positions
        if fetch_prices and positions:
            print("[INFO] Fetching current prices...")
            for pos in positions:
                symbol = pos.get("symbol")
                if symbol:
                    price = fetch_current_price(symbol)
                    if price:
                        pos["current_price"] = price

    if not positions:
        print(f"\n[{source}] No open positions found.")
        return

    print(f"\n=== Open Positions ({source}) ===\n")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print_table_header()

    total_exposure = 0.0
    total_unrealized = 0.0

    for pos in positions:
        # For live positions, calculate exposure
        qty = int(pos.get("qty", 0))
        market_value = pos.get("market_value")
        if market_value:
            exposure = abs(float(market_value))
        else:
            price = pos.get("current_price") or pos.get("avg_entry_price") or 0
            exposure = abs(qty * float(price))
        total_exposure += exposure

        unrealized = pos.get("unrealized_pl")
        if unrealized:
            total_unrealized += float(unrealized)

        current_price = None
        if live:
            # For live positions, current_price is calculated from market_value
            if market_value and qty:
                current_price = float(market_value) / abs(qty)

        print_position_row(pos, current_price)

    print("-" * 115)
    print(f"\n{'Total Positions:':<25} {len(positions)}")
    print(f"{'Total Exposure:':<25} ${total_exposure:,.2f}")
    print(f"{'Total Unrealized P&L:':<25} {format_currency(total_unrealized)}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Kobe Trading System - View Open Positions")
    ap.add_argument(
        "--dotenv",
        type=str,
        default=DEFAULT_DOTENV,
        help="Path to .env file for API credentials",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="Fetch positions from broker (Alpaca)",
    )
    ap.add_argument(
        "--local",
        action="store_true",
        help="Show local positions only (from state/positions.json)",
    )
    ap.add_argument(
        "--fetch-prices",
        action="store_true",
        help="Fetch current prices for local positions",
    )
    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
        print(f"[INFO] Loaded {len(loaded)} env vars from {dotenv}")
    elif args.live:
        print(f"[WARN] .env file not found at {dotenv}, broker calls may fail")

    # Determine mode
    if args.live and args.local:
        print("[ERROR] Cannot specify both --live and --local")
        sys.exit(1)

    show_positions(
        live=args.live,
        local=args.local or not args.live,
        fetch_prices=args.fetch_prices,
    )


if __name__ == "__main__":
    main()
