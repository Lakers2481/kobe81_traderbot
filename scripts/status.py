#!/usr/bin/env python3
"""
Kobe System Health Dashboard

Displays:
- Trading mode (PAPER/LIVE/HALTED)
- Kill switch state
- Last scan time
- Open positions count
- Today's P&L
- Broker connection status
- Data freshness
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
STATE_DIR = ROOT / "state"
LOGS_DIR = ROOT / "logs"
KILL_SWITCH_FILE = STATE_DIR / "KILL_SWITCH"
RUNNER_STATE_FILE = STATE_DIR / "runner_last.json"
EVENTS_LOG = LOGS_DIR / "events.jsonl"
SIGNALS_LOG = LOGS_DIR / "signals.jsonl"


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def get_alpaca_config() -> Dict[str, str]:
    """Get Alpaca configuration from environment."""
    return {
        "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/"),
        "key_id": os.getenv("ALPACA_API_KEY_ID", ""),
        "secret": os.getenv("ALPACA_API_SECRET_KEY", ""),
    }


def alpaca_headers(cfg: Dict[str, str]) -> Dict[str, str]:
    """Build authentication headers for Alpaca API."""
    return {
        "APCA-API-KEY-ID": cfg["key_id"],
        "APCA-API-SECRET-KEY": cfg["secret"],
    }


def check_kill_switch() -> bool:
    """Return True if kill switch is active."""
    return KILL_SWITCH_FILE.exists()


def get_trading_mode() -> str:
    """Determine current trading mode."""
    if check_kill_switch():
        return "HALTED"
    base_url = os.getenv("ALPACA_BASE_URL", "")
    if "paper" in base_url.lower():
        return "PAPER"
    elif base_url and "paper" not in base_url.lower():
        return "LIVE"
    return "UNKNOWN"


def get_last_scan_time() -> Optional[datetime]:
    """Get the last scan time from runner state or signals log."""
    # Try runner state file first
    if RUNNER_STATE_FILE.exists():
        try:
            data = json.loads(RUNNER_STATE_FILE.read_text(encoding="utf-8"))
            # Find most recent run timestamp
            for key in sorted(data.keys(), reverse=True):
                val = data[key]
                if isinstance(val, str) and "-" in val:
                    try:
                        return datetime.fromisoformat(val)
                    except ValueError:
                        continue
        except Exception:
            pass

    # Try signals log
    if SIGNALS_LOG.exists():
        try:
            lines = SIGNALS_LOG.read_text(encoding="utf-8").strip().split("\n")
            if lines and lines[-1]:
                last = json.loads(lines[-1])
                ts = last.get("ts") or last.get("timestamp")
                if ts:
                    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            pass

    # Try events log
    if EVENTS_LOG.exists():
        try:
            lines = EVENTS_LOG.read_text(encoding="utf-8").strip().split("\n")
            for line in reversed(lines[-100:]):
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("event") in ("scan_complete", "runner_done", "signals_generated"):
                    ts = rec.get("ts")
                    if ts:
                        return datetime.fromisoformat(ts)
        except Exception:
            pass

    return None


def fetch_alpaca_account() -> Optional[Dict[str, Any]]:
    """Fetch account info from Alpaca."""
    cfg = get_alpaca_config()
    if not cfg["key_id"] or not cfg["secret"]:
        return None
    try:
        url = f"{cfg['base_url']}/v2/account"
        r = requests.get(url, headers=alpaca_headers(cfg), timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_alpaca_positions() -> list:
    """Fetch open positions from Alpaca."""
    cfg = get_alpaca_config()
    if not cfg["key_id"] or not cfg["secret"]:
        return []
    try:
        url = f"{cfg['base_url']}/v2/positions"
        r = requests.get(url, headers=alpaca_headers(cfg), timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def check_broker_connection() -> tuple[bool, str]:
    """Check if broker connection is healthy."""
    cfg = get_alpaca_config()
    if not cfg["key_id"] or not cfg["secret"]:
        return False, "Missing API credentials"
    try:
        url = f"{cfg['base_url']}/v2/account"
        r = requests.get(url, headers=alpaca_headers(cfg), timeout=10)
        if r.status_code == 200:
            data = r.json()
            status = data.get("status", "unknown")
            if status == "ACTIVE":
                return True, "Connected (ACTIVE)"
            return True, f"Connected ({status})"
        elif r.status_code == 401:
            return False, "Authentication failed"
        elif r.status_code == 403:
            return False, "Access forbidden"
        else:
            return False, f"HTTP {r.status_code}"
    except requests.exceptions.Timeout:
        return False, "Connection timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection error"
    except Exception as e:
        return False, f"Error: {e}"


def check_polygon_connection() -> tuple[bool, str]:
    """Check if Polygon data connection is healthy."""
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return False, "Missing POLYGON_API_KEY"
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/SPY/prev?apiKey={api_key}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            if results:
                last_date = results[0].get("t")
                if last_date:
                    from datetime import datetime
                    dt = datetime.utcfromtimestamp(last_date / 1000)
                    return True, f"Connected (last: {dt.strftime('%Y-%m-%d')})"
            return True, "Connected"
        elif r.status_code == 401:
            return False, "Authentication failed"
        else:
            return False, f"HTTP {r.status_code}"
    except requests.exceptions.Timeout:
        return False, "Connection timeout"
    except Exception as e:
        return False, f"Error: {e}"


def get_data_freshness() -> Dict[str, Any]:
    """Check data cache freshness."""
    cache_dir = ROOT / "data" / "cache"
    result = {
        "cache_exists": cache_dir.exists(),
        "files": 0,
        "newest_file": None,
        "oldest_file": None,
        "age_hours": None,
    }
    if not cache_dir.exists():
        return result

    csv_files = list(cache_dir.glob("*.csv"))
    result["files"] = len(csv_files)
    if not csv_files:
        return result

    # Find newest and oldest
    newest = max(csv_files, key=lambda f: f.stat().st_mtime)
    oldest = min(csv_files, key=lambda f: f.stat().st_mtime)
    newest_time = datetime.fromtimestamp(newest.stat().st_mtime)
    datetime.fromtimestamp(oldest.stat().st_mtime)

    result["newest_file"] = newest.name
    result["oldest_file"] = oldest.name
    result["age_hours"] = round((datetime.now() - newest_time).total_seconds() / 3600, 1)

    return result


def calculate_todays_pnl(positions: list) -> float:
    """Calculate today's unrealized P&L from positions."""
    total = 0.0
    for pos in positions:
        try:
            unrealized = float(pos.get("unrealized_intraday_pl", 0) or 0)
            total += unrealized
        except (ValueError, TypeError):
            continue
    return total


# -----------------------------------------------------------------------------
# Display functions
# -----------------------------------------------------------------------------
def print_header(title: str, width: int = 60) -> None:
    """Print a formatted header."""
    print("=" * width)
    print(f" {title}".center(width))
    print("=" * width)


def print_section(title: str, width: int = 60) -> None:
    """Print a section header."""
    print()
    print(f"--- {title} ".ljust(width, "-"))


def print_status_row(label: str, value: str, status: Optional[str] = None, width: int = 40) -> None:
    """Print a status row with optional status indicator."""
    if status:
        indicator = "[OK]" if status == "ok" else "[!!]" if status == "error" else "[--]"
        print(f"  {label:<{width}} {value} {indicator}")
    else:
        print(f"  {label:<{width}} {value}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Kobe System Health Dashboard")
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text",
    )
    args = ap.parse_args()

    # Load environment
    dotenv_path = Path(args.dotenv)
    if dotenv_path.exists():
        load_env(dotenv_path)

    # Collect status data
    status_data: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "trading_mode": get_trading_mode(),
        "kill_switch_active": check_kill_switch(),
        "last_scan_time": None,
        "open_positions": 0,
        "todays_pnl": 0.0,
        "broker_connected": False,
        "broker_status": "",
        "polygon_connected": False,
        "polygon_status": "",
        "data_freshness": {},
        "account": {},
    }

    # Get last scan time
    last_scan = get_last_scan_time()
    if last_scan:
        status_data["last_scan_time"] = last_scan.isoformat()

    # Check broker connection
    broker_ok, broker_msg = check_broker_connection()
    status_data["broker_connected"] = broker_ok
    status_data["broker_status"] = broker_msg

    # Get account info
    account = fetch_alpaca_account()
    if account:
        status_data["account"] = {
            "equity": float(account.get("equity", 0)),
            "buying_power": float(account.get("buying_power", 0)),
            "cash": float(account.get("cash", 0)),
            "status": account.get("status", "unknown"),
        }

    # Get positions
    positions = fetch_alpaca_positions()
    status_data["open_positions"] = len(positions)
    status_data["todays_pnl"] = calculate_todays_pnl(positions)

    # Check Polygon connection
    polygon_ok, polygon_msg = check_polygon_connection()
    status_data["polygon_connected"] = polygon_ok
    status_data["polygon_status"] = polygon_msg

    # Get data freshness
    status_data["data_freshness"] = get_data_freshness()

    # Output
    if args.json:
        print(json.dumps(status_data, indent=2, default=str))
        return 0

    # Formatted output
    print_header("KOBE SYSTEM STATUS")
    print(f"  Timestamp: {status_data['timestamp']}")

    print_section("Trading Status")
    mode = status_data["trading_mode"]
    mode_status = "ok" if mode in ("PAPER", "LIVE") else "error" if mode == "HALTED" else "warn"
    print_status_row("Trading Mode:", mode, mode_status)

    kill_active = status_data["kill_switch_active"]
    print_status_row(
        "Kill Switch:",
        "ACTIVE - Trading halted" if kill_active else "Inactive",
        "error" if kill_active else "ok",
    )

    if status_data["last_scan_time"]:
        scan_time = datetime.fromisoformat(status_data["last_scan_time"])
        age = datetime.utcnow() - scan_time
        age_str = f"{scan_time.strftime('%Y-%m-%d %H:%M:%S')} ({age.seconds // 3600}h {(age.seconds // 60) % 60}m ago)"
        print_status_row("Last Scan:", age_str)
    else:
        print_status_row("Last Scan:", "No recent scans", "warn")

    print_section("Positions & P&L")
    print_status_row("Open Positions:", str(status_data["open_positions"]))
    pnl = status_data["todays_pnl"]
    pnl_str = f"${pnl:+,.2f}"
    pnl_status = "ok" if pnl >= 0 else "warn"
    print_status_row("Today's P&L:", pnl_str, pnl_status)

    if status_data["account"]:
        acc = status_data["account"]
        print_status_row("Account Equity:", f"${acc['equity']:,.2f}")
        print_status_row("Buying Power:", f"${acc['buying_power']:,.2f}")
        print_status_row("Cash:", f"${acc['cash']:,.2f}")

    print_section("Connections")
    print_status_row(
        "Broker (Alpaca):",
        status_data["broker_status"],
        "ok" if status_data["broker_connected"] else "error",
    )
    print_status_row(
        "Data (Polygon):",
        status_data["polygon_status"],
        "ok" if status_data["polygon_connected"] else "error",
    )

    print_section("Data Freshness")
    df = status_data["data_freshness"]
    if df.get("cache_exists"):
        print_status_row("Cache Files:", str(df.get("files", 0)))
        if df.get("newest_file"):
            age = df.get("age_hours", 0)
            freshness_status = "ok" if age < 24 else "warn" if age < 72 else "error"
            print_status_row("Newest Cache:", f"{df['newest_file']} ({age}h old)", freshness_status)
    else:
        print_status_row("Cache:", "No cache directory", "warn")

    print()
    print("=" * 60)

    # Return exit code based on critical status
    if status_data["kill_switch_active"]:
        return 1
    if not status_data["broker_connected"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
