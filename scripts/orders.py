#!/usr/bin/env python3
"""
Kobe Trading System - Order History Viewer

List recent orders from state/orders.json with fill details and slippage analysis.
Supports filtering by status, symbol, and date.

Usage:
    python orders.py                     # Show all orders
    python orders.py --status FILLED     # Filter by status
    python orders.py --symbol AAPL       # Filter by symbol
    python orders.py --tail 20           # Show last 20 orders
    python orders.py --live              # Fetch from broker
    python orders.py --dotenv /path/to/.env
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.env_loader import load_env
from oms.order_state import OrderStatus

# Default paths
STATE_DIR = ROOT / "state"
ORDERS_FILE = STATE_DIR / "orders.json"
RECONCILE_DIR = STATE_DIR / "reconcile"
LOGS_DIR = ROOT / "logs"
DEFAULT_DOTENV = "C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env"

# Valid order statuses
VALID_STATUSES = {s.value for s in OrderStatus}


def _alpaca_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY_ID", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET_KEY", ""),
    }


def _alpaca_base() -> str:
    return os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")


def fetch_live_orders(status: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """Fetch orders from Alpaca API."""
    status_param = status.lower() if status else "all"
    url = f"{_alpaca_base()}/v2/orders?status={status_param}&limit={limit}"
    try:
        r = requests.get(url, headers=_alpaca_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch live orders: {e}")
        return []


def load_local_orders() -> List[Dict[str, Any]]:
    """Load orders from local state files."""
    orders: List[Dict[str, Any]] = []

    # Main orders file
    if ORDERS_FILE.exists():
        try:
            data = json.loads(ORDERS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                orders.extend(data)
            elif isinstance(data, dict):
                orders.extend(data.get("orders", []))
        except Exception as e:
            print(f"[WARN] Failed to load orders.json: {e}")

    # Reconcile snapshots
    for fname in ["orders_all.json", "orders_open.json"]:
        fpath = RECONCILE_DIR / fname
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    # Deduplicate by order ID
                    existing_ids = {o.get("id") or o.get("broker_order_id") for o in orders}
                    for o in data:
                        oid = o.get("id") or o.get("broker_order_id")
                        if oid and oid not in existing_ids:
                            orders.append(o)
                            existing_ids.add(oid)
            except Exception:
                pass

    # Parse log files for order events
    events_file = LOGS_DIR / "events.jsonl"
    if events_file.exists():
        try:
            existing_ids = {o.get("id") or o.get("broker_order_id") or o.get("decision_id") for o in orders}
            for line in events_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if event.get("event") in ("order_submitted", "order_filled", "order_rejected", "order_cancelled"):
                        oid = event.get("broker_order_id") or event.get("decision_id")
                        if oid and oid not in existing_ids:
                            orders.append(event)
                            existing_ids.add(oid)
                except Exception:
                    continue
        except Exception:
            pass

    return orders


def parse_timestamp(ts: Any) -> Optional[datetime]:
    """Parse various timestamp formats."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        ts_str = str(ts)
        if "T" in ts_str:
            # Handle ISO format with timezone
            ts_str = ts_str.replace("Z", "+00:00")
            if "+" in ts_str:
                ts_str = ts_str.split("+")[0]
            return datetime.fromisoformat(ts_str)
        if len(ts_str) >= 10:
            return datetime.strptime(ts_str[:10], "%Y-%m-%d")
    except Exception:
        pass
    return None


def filter_orders(
    orders: List[Dict[str, Any]],
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter orders by criteria."""
    filtered = orders

    if status:
        status_upper = status.upper()
        filtered = [
            o for o in filtered
            if (o.get("status", "").upper() == status_upper)
        ]

    if symbol:
        symbol_upper = symbol.upper()
        filtered = [
            o for o in filtered
            if o.get("symbol", "").upper() == symbol_upper
        ]

    if date_from:
        try:
            from_dt = datetime.strptime(date_from, "%Y-%m-%d")
            filtered = [
                o for o in filtered
                if parse_timestamp(o.get("created_at") or o.get("submitted_at") or o.get("ts"))
                and parse_timestamp(o.get("created_at") or o.get("submitted_at") or o.get("ts")) >= from_dt
            ]
        except ValueError:
            print(f"[WARN] Invalid date format for --from: {date_from}")

    if date_to:
        try:
            to_dt = datetime.strptime(date_to, "%Y-%m-%d")
            filtered = [
                o for o in filtered
                if parse_timestamp(o.get("created_at") or o.get("submitted_at") or o.get("ts"))
                and parse_timestamp(o.get("created_at") or o.get("submitted_at") or o.get("ts")) <= to_dt
            ]
        except ValueError:
            print(f"[WARN] Invalid date format for --to: {date_to}")

    return filtered


def sort_orders(orders: List[Dict[str, Any]], reverse: bool = True) -> List[Dict[str, Any]]:
    """Sort orders by timestamp (newest first by default)."""
    def get_ts(o: Dict[str, Any]) -> datetime:
        ts = parse_timestamp(
            o.get("created_at")
            or o.get("submitted_at")
            or o.get("filled_at")
            or o.get("ts")
        )
        return ts or datetime.min

    return sorted(orders, key=get_ts, reverse=reverse)


def calculate_slippage(order: Dict[str, Any]) -> Optional[float]:
    """Calculate slippage in basis points."""
    limit_price = order.get("limit_price")
    filled_avg_price = order.get("filled_avg_price") or order.get("avg_fill_price")

    if limit_price and filled_avg_price:
        limit_price = float(limit_price)
        filled_avg_price = float(filled_avg_price)
        if limit_price > 0:
            slippage_pct = ((filled_avg_price - limit_price) / limit_price) * 10000  # bps
            side = order.get("side", "").lower()
            # For buys, positive slippage is bad; for sells, negative is bad
            if side in ("sell", "short"):
                slippage_pct = -slippage_pct
            return slippage_pct
    return None


def format_status(status: str) -> str:
    """Format status with visual indicator."""
    status_upper = status.upper()
    indicators = {
        "FILLED": "[OK]",
        "SUBMITTED": "[..]",
        "PENDING": "[..]",
        "APPROVED": "[OK]",
        "CANCELLED": "[X]",
        "REJECTED": "[!]",
        "VETOED": "[!]",
        "CLOSED": "[OK]",
    }
    return f"{indicators.get(status_upper, '[ ]')} {status_upper}"


def print_orders_table(orders: List[Dict[str, Any]], detailed: bool = False):
    """Print orders in table format."""
    if not orders:
        print("\n[INFO] No orders to display.")
        return

    print()
    if detailed:
        print(
            f"{'ID':<12} {'Symbol':<8} {'Side':<6} {'Type':<10} {'Qty':>8} "
            f"{'Limit':>10} {'Fill Avg':>10} {'Slip(bps)':>10} {'Status':<15} {'Time':<20}"
        )
        print("-" * 120)
    else:
        print(
            f"{'Symbol':<8} {'Side':<6} {'Qty':>8} {'Limit':>10} {'Fill':>10} "
            f"{'Status':<15} {'Created':<20}"
        )
        print("-" * 90)

    for order in orders:
        symbol = order.get("symbol", "???")
        side = order.get("side", "-").upper()
        qty = order.get("qty") or order.get("filled_qty") or "-"
        limit_price = order.get("limit_price")
        limit_str = f"${float(limit_price):.2f}" if limit_price else "-"

        filled_price = order.get("filled_avg_price") or order.get("avg_fill_price")
        fill_str = f"${float(filled_price):.2f}" if filled_price else "-"

        status = order.get("status", "UNKNOWN")

        # Timestamp
        ts = parse_timestamp(
            order.get("created_at")
            or order.get("submitted_at")
            or order.get("ts")
        )
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "-"

        if detailed:
            order_id = (order.get("id") or order.get("broker_order_id") or order.get("decision_id") or "-")[:12]
            order_type = order.get("type") or order.get("order_type") or "-"

            slippage = calculate_slippage(order)
            slip_str = f"{slippage:+.1f}" if slippage is not None else "-"

            print(
                f"{order_id:<12} {symbol:<8} {side:<6} {order_type:<10} {str(qty):>8} "
                f"{limit_str:>10} {fill_str:>10} {slip_str:>10} {format_status(status):<15} {ts_str:<20}"
            )
        else:
            print(
                f"{symbol:<8} {side:<6} {str(qty):>8} {limit_str:>10} {fill_str:>10} "
                f"{format_status(status):<15} {ts_str:<20}"
            )


def print_fill_summary(orders: List[Dict[str, Any]]):
    """Print fill statistics summary."""
    filled_orders = [o for o in orders if o.get("status", "").upper() == "FILLED"]

    if not filled_orders:
        return

    slippages = []
    for order in filled_orders:
        slip = calculate_slippage(order)
        if slip is not None:
            slippages.append(slip)

    print(f"\n{'='*50}")
    print(" Fill Statistics")
    print(f"{'='*50}")
    print()
    print(f"  {'Total Orders:':<25} {len(orders)}")
    print(f"  {'Filled Orders:':<25} {len(filled_orders)}")
    print(f"  {'Fill Rate:':<25} {len(filled_orders)/len(orders)*100:.1f}%" if orders else "  Fill Rate: N/A")

    if slippages:
        avg_slip = sum(slippages) / len(slippages)
        max_slip = max(slippages)
        min_slip = min(slippages)
        print()
        print(f"  {'Avg Slippage (bps):':<25} {avg_slip:+.2f}")
        print(f"  {'Max Slippage (bps):':<25} {max_slip:+.2f}")
        print(f"  {'Min Slippage (bps):':<25} {min_slip:+.2f}")


def show_orders(
    live: bool = False,
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    tail: Optional[int] = None,
    detailed: bool = False,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    """Main order display function."""
    if live:
        print("[INFO] Fetching orders from broker...")
        orders = fetch_live_orders(status=status)
        source = "Broker (Alpaca)"
    else:
        print("[INFO] Loading local orders...")
        orders = load_local_orders()
        source = "Local State"

    if not orders:
        print(f"\n[{source}] No orders found.")
        return

    print(f"[INFO] Loaded {len(orders)} orders from {source}")

    # Apply filters
    orders = filter_orders(orders, status=status, symbol=symbol, date_from=date_from, date_to=date_to)

    if not orders:
        print("\n[INFO] No orders match the filter criteria.")
        return

    # Sort by time (newest first)
    orders = sort_orders(orders, reverse=True)

    # Apply tail limit
    if tail:
        orders = orders[:tail]

    print(f"\n{'#'*80}")
    print(f"#  KOBE TRADING SYSTEM - ORDER HISTORY")
    print(f"#  Source: {source}")
    print(f"#  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if status:
        print(f"#  Filter - Status: {status.upper()}")
    if symbol:
        print(f"#  Filter - Symbol: {symbol.upper()}")
    if tail:
        print(f"#  Showing: Last {tail} orders")
    print(f"{'#'*80}")

    print_orders_table(orders, detailed=detailed)
    print_fill_summary(orders)
    print()


def main():
    ap = argparse.ArgumentParser(description="Kobe Trading System - Order History")
    ap.add_argument(
        "--dotenv",
        type=str,
        default=DEFAULT_DOTENV,
        help="Path to .env file",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="Fetch orders from broker instead of local state",
    )
    ap.add_argument(
        "--status",
        type=str,
        default=None,
        help=f"Filter by status ({', '.join(VALID_STATUSES)})",
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
        help="Show only the last N orders",
    )
    ap.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show detailed output with order IDs and slippage",
    )
    ap.add_argument(
        "--from",
        dest="date_from",
        type=str,
        default=None,
        help="Filter orders from date (YYYY-MM-DD)",
    )
    ap.add_argument(
        "--to",
        dest="date_to",
        type=str,
        default=None,
        help="Filter orders to date (YYYY-MM-DD)",
    )
    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
        print(f"[INFO] Loaded {len(loaded)} env vars from {dotenv}")
    elif args.live:
        print(f"[WARN] .env file not found at {dotenv}, broker calls may fail")

    show_orders(
        live=args.live,
        status=args.status,
        symbol=args.symbol,
        tail=args.tail,
        detailed=args.detailed,
        date_from=args.date_from,
        date_to=args.date_to,
    )


if __name__ == "__main__":
    main()
