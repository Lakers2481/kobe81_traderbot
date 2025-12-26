#!/usr/bin/env python3
"""
state.py - View all state files for the Kobe trading system.

Usage:
    python scripts/state.py --all
    python scripts/state.py --positions
    python scripts/state.py --orders
    python scripts/state.py --kill-switch
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.env_loader import load_env

STATE_DIR = ROOT / "state"
POSITIONS_FILE = STATE_DIR / "positions.json"
ORDERS_FILE = STATE_DIR / "orders.json"
KILL_SWITCH_FILE = STATE_DIR / "KILL_SWITCH"


def format_json(data: Any, indent: int = 2) -> str:
    """Format JSON data for display."""
    return json.dumps(data, indent=indent, default=str)


def list_state_files() -> List[Dict[str, Any]]:
    """List all files in the state directory with metadata."""
    files = []
    if not STATE_DIR.exists():
        return files

    for f in sorted(STATE_DIR.iterdir()):
        if f.is_file():
            stat = f.stat()
            files.append({
                "name": f.name,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    return files


def read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    """Read and parse a JSON file."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}"}
    except Exception as e:
        return {"error": str(e)}


def show_positions() -> None:
    """Display positions.json contents."""
    print("\n=== POSITIONS ===")
    print(f"File: {POSITIONS_FILE}")

    if not POSITIONS_FILE.exists():
        print("Status: File does not exist (no open positions)")
        return

    data = read_json_file(POSITIONS_FILE)
    if data is None:
        print("Status: Empty file")
        return

    if isinstance(data, dict) and "error" in data:
        print(f"Error: {data['error']}")
        return

    if isinstance(data, list):
        print(f"Open positions: {len(data)}")
        for i, pos in enumerate(data, 1):
            symbol = pos.get("symbol", "?")
            qty = pos.get("qty", 0)
            side = pos.get("side", "?")
            entry_price = pos.get("entry_price", 0)
            entry_time = pos.get("entry_time", "?")
            print(f"  {i}. {symbol}: {side} {qty} @ ${entry_price:.2f} (entry: {entry_time})")
    else:
        print(format_json(data))


def show_orders() -> None:
    """Display orders.json contents."""
    print("\n=== ORDERS ===")
    print(f"File: {ORDERS_FILE}")

    if not ORDERS_FILE.exists():
        print("Status: File does not exist (no orders)")
        return

    data = read_json_file(ORDERS_FILE)
    if data is None:
        print("Status: Empty file")
        return

    if isinstance(data, dict) and "error" in data:
        print(f"Error: {data['error']}")
        return

    if isinstance(data, list):
        print(f"Total orders: {len(data)}")
        # Show last 10 orders
        recent = data[-10:] if len(data) > 10 else data
        print(f"Showing last {len(recent)} orders:")
        for order in recent:
            order_id = order.get("decision_id", order.get("order_id", "?"))[:12]
            symbol = order.get("symbol", "?")
            side = order.get("side", "?")
            qty = order.get("qty", 0)
            status = order.get("status", "?")
            limit_price = order.get("limit_price", 0)
            print(f"  [{status}] {order_id}... {symbol} {side} {qty} @ ${limit_price:.2f}")
    elif isinstance(data, dict):
        print(f"Total orders: {len(data)}")
        # Show last 10 orders from dict
        items = list(data.items())[-10:]
        print(f"Showing last {len(items)} orders:")
        for order_id, order in items:
            symbol = order.get("symbol", "?")
            side = order.get("side", "?")
            qty = order.get("qty", 0)
            status = order.get("status", "?")
            print(f"  [{status}] {order_id[:12]}... {symbol} {side} {qty}")
    else:
        print(format_json(data))


def show_kill_switch() -> None:
    """Display kill switch status."""
    print("\n=== KILL SWITCH ===")
    print(f"File: {KILL_SWITCH_FILE}")

    if KILL_SWITCH_FILE.exists():
        print("Status: ACTIVE - Trading is HALTED")
        try:
            content = KILL_SWITCH_FILE.read_text(encoding="utf-8").strip()
            if content:
                print(f"Reason: {content}")
            stat = KILL_SWITCH_FILE.stat()
            activated = datetime.fromtimestamp(stat.st_mtime).isoformat()
            print(f"Activated: {activated}")
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print("Status: INACTIVE - Trading is allowed")


def show_all_files() -> None:
    """List all files in state directory."""
    print("\n=== STATE DIRECTORY ===")
    print(f"Path: {STATE_DIR}")

    if not STATE_DIR.exists():
        print("Status: Directory does not exist")
        return

    files = list_state_files()
    if not files:
        print("Status: Empty directory")
        return

    print(f"Files: {len(files)}")
    for f in files:
        size = f["size_bytes"]
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        print(f"  {f['name']:30} {size_str:>10}  {f['modified']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="View Kobe trading system state files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/state.py --all
    python scripts/state.py --positions
    python scripts/state.py --orders
    python scripts/state.py --kill-switch
    python scripts/state.py --positions --orders
        """
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file"
    )
    ap.add_argument(
        "--positions",
        action="store_true",
        help="Show positions.json contents"
    )
    ap.add_argument(
        "--orders",
        action="store_true",
        help="Show orders.json contents"
    )
    ap.add_argument(
        "--kill-switch",
        action="store_true",
        help="Show kill switch status"
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Show all state information"
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format"
    )

    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # If no flags, default to --all
    show_any = args.positions or args.orders or args.kill_switch or args.all
    if not show_any:
        args.all = True

    print("=" * 60)
    print("KOBE TRADING SYSTEM - STATE VIEWER")
    print(f"Time: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    if args.json:
        # JSON output mode
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "state_dir": str(STATE_DIR),
        }
        if args.all or args.positions:
            output["positions"] = read_json_file(POSITIONS_FILE)
        if args.all or args.orders:
            output["orders"] = read_json_file(ORDERS_FILE)
        if args.all or args.kill_switch:
            output["kill_switch"] = {
                "active": KILL_SWITCH_FILE.exists(),
                "file": str(KILL_SWITCH_FILE),
            }
        if args.all:
            output["files"] = list_state_files()
        print(format_json(output))
        return

    # Human-readable output
    if args.all:
        show_all_files()
        show_positions()
        show_orders()
        show_kill_switch()
    else:
        if args.positions:
            show_positions()
        if args.orders:
            show_orders()
        if args.kill_switch:
            show_kill_switch()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
