#!/usr/bin/env python3
"""
Broker Connection Status for Kobe Trading System

Checks Alpaca API connectivity, shows account balance/buying power, and open orders.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.env_loader import load_env


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_DOTENV = "C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env"
ALPACA_TIMEOUT = 10


# -----------------------------------------------------------------------------
# Alpaca API Helpers
# -----------------------------------------------------------------------------
def get_alpaca_config() -> Dict[str, str]:
    """Get Alpaca API configuration from environment."""
    return {
        "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/"),
        "key_id": os.getenv("ALPACA_API_KEY_ID", ""),
        "secret": os.getenv("ALPACA_API_SECRET_KEY", "")
    }


def get_auth_headers(config: Dict[str, str]) -> Dict[str, str]:
    """Get authentication headers for Alpaca API."""
    return {
        "APCA-API-KEY-ID": config["key_id"],
        "APCA-API-SECRET-KEY": config["secret"],
        "Content-Type": "application/json"
    }


def alpaca_request(endpoint: str, method: str = "GET") -> Dict[str, Any]:
    """Make a request to the Alpaca API."""
    config = get_alpaca_config()
    url = f"{config['base_url']}{endpoint}"
    headers = get_auth_headers(config)

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=ALPACA_TIMEOUT)
        else:
            response = requests.request(method, url, headers=headers, timeout=ALPACA_TIMEOUT)

        return {
            "success": response.status_code in (200, 201),
            "status_code": response.status_code,
            "data": response.json() if response.content else None,
            "error": None
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": None,
            "data": None,
            "error": "Request timed out"
        }
    except requests.exceptions.ConnectionError as e:
        return {
            "success": False,
            "status_code": None,
            "data": None,
            "error": f"Connection error: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": None,
            "data": None,
            "error": str(e)
        }


# -----------------------------------------------------------------------------
# Status Checks
# -----------------------------------------------------------------------------
def check_connectivity() -> Dict[str, Any]:
    """Check basic API connectivity."""
    config = get_alpaca_config()

    # Verify credentials are set
    if not config["key_id"] or not config["secret"]:
        return {
            "check": "connectivity",
            "status": "FAIL",
            "details": {
                "base_url": config["base_url"],
                "error": "API credentials not configured",
                "key_id_set": bool(config["key_id"]),
                "secret_set": bool(config["secret"])
            }
        }

    # Test account endpoint
    result = alpaca_request("/v2/account")

    if result["success"]:
        return {
            "check": "connectivity",
            "status": "PASS",
            "details": {
                "base_url": config["base_url"],
                "status_code": result["status_code"],
                "api_accessible": True,
                "key_id_prefix": config["key_id"][:4] + "****" if len(config["key_id"]) > 4 else "****"
            }
        }
    else:
        return {
            "check": "connectivity",
            "status": "FAIL",
            "details": {
                "base_url": config["base_url"],
                "status_code": result["status_code"],
                "error": result["error"] or f"HTTP {result['status_code']}"
            }
        }


def check_credentials() -> Dict[str, Any]:
    """Verify API credentials are valid."""
    config = get_alpaca_config()

    if not config["key_id"] or not config["secret"]:
        return {
            "check": "credentials",
            "status": "FAIL",
            "details": {
                "error": "Missing API credentials",
                "ALPACA_API_KEY_ID": "set" if config["key_id"] else "MISSING",
                "ALPACA_API_SECRET_KEY": "set" if config["secret"] else "MISSING"
            }
        }

    result = alpaca_request("/v2/account")

    if result["success"]:
        account = result["data"] or {}
        return {
            "check": "credentials",
            "status": "PASS",
            "details": {
                "valid": True,
                "account_id": account.get("id", "N/A"),
                "account_number": account.get("account_number", "N/A"),
                "status": account.get("status", "N/A"),
                "crypto_status": account.get("crypto_status", "N/A")
            }
        }
    elif result["status_code"] == 401:
        return {
            "check": "credentials",
            "status": "FAIL",
            "details": {
                "valid": False,
                "error": "Invalid API credentials (401 Unauthorized)"
            }
        }
    elif result["status_code"] == 403:
        return {
            "check": "credentials",
            "status": "FAIL",
            "details": {
                "valid": False,
                "error": "Access forbidden (403) - check API key permissions"
            }
        }
    else:
        return {
            "check": "credentials",
            "status": "FAIL",
            "details": {
                "valid": False,
                "error": result["error"] or f"HTTP {result['status_code']}"
            }
        }


def get_account_info() -> Dict[str, Any]:
    """Get account balance and buying power."""
    result = alpaca_request("/v2/account")

    if not result["success"]:
        return {
            "check": "account",
            "status": "FAIL",
            "details": {
                "error": result["error"] or f"HTTP {result['status_code']}"
            }
        }

    account = result["data"] or {}

    # Parse financial values
    cash = float(account.get("cash", 0))
    buying_power = float(account.get("buying_power", 0))
    portfolio_value = float(account.get("portfolio_value", 0))
    equity = float(account.get("equity", 0))
    last_equity = float(account.get("last_equity", 0))

    # Calculate day P&L
    day_pnl = equity - last_equity
    day_pnl_pct = (day_pnl / last_equity * 100) if last_equity > 0 else 0

    return {
        "check": "account",
        "status": "PASS",
        "details": {
            "account_status": account.get("status", "N/A"),
            "trading_blocked": account.get("trading_blocked", False),
            "account_blocked": account.get("account_blocked", False),
            "pattern_day_trader": account.get("pattern_day_trader", False),
            "financials": {
                "cash": f"${cash:,.2f}",
                "buying_power": f"${buying_power:,.2f}",
                "portfolio_value": f"${portfolio_value:,.2f}",
                "equity": f"${equity:,.2f}",
                "day_pnl": f"${day_pnl:+,.2f} ({day_pnl_pct:+.2f}%)"
            },
            "margins": {
                "initial_margin": f"${float(account.get('initial_margin', 0)):,.2f}",
                "maintenance_margin": f"${float(account.get('maintenance_margin', 0)):,.2f}",
                "daytrade_count": account.get("daytrade_count", 0),
                "multiplier": account.get("multiplier", "N/A")
            }
        }
    }


def get_open_orders() -> Dict[str, Any]:
    """Get list of open orders."""
    result = alpaca_request("/v2/orders?status=open")

    if not result["success"]:
        return {
            "check": "open_orders",
            "status": "FAIL",
            "details": {
                "error": result["error"] or f"HTTP {result['status_code']}"
            }
        }

    orders = result["data"] or []

    order_list = []
    for order in orders:
        order_list.append({
            "id": order.get("id", "N/A")[:8] + "...",
            "client_order_id": order.get("client_order_id", "N/A"),
            "symbol": order.get("symbol", "N/A"),
            "side": order.get("side", "N/A"),
            "qty": order.get("qty", "N/A"),
            "type": order.get("type", "N/A"),
            "status": order.get("status", "N/A"),
            "limit_price": order.get("limit_price", "N/A"),
            "created_at": order.get("created_at", "N/A")
        })

    return {
        "check": "open_orders",
        "status": "PASS",
        "details": {
            "count": len(orders),
            "orders": order_list
        }
    }


def get_positions() -> Dict[str, Any]:
    """Get current positions."""
    result = alpaca_request("/v2/positions")

    if not result["success"]:
        return {
            "check": "positions",
            "status": "FAIL",
            "details": {
                "error": result["error"] or f"HTTP {result['status_code']}"
            }
        }

    positions = result["data"] or []

    position_list = []
    total_market_value = 0.0
    total_unrealized_pnl = 0.0

    for pos in positions:
        market_value = float(pos.get("market_value", 0))
        unrealized_pnl = float(pos.get("unrealized_pl", 0))
        unrealized_pnl_pct = float(pos.get("unrealized_plpc", 0)) * 100

        total_market_value += market_value
        total_unrealized_pnl += unrealized_pnl

        position_list.append({
            "symbol": pos.get("symbol", "N/A"),
            "qty": pos.get("qty", "N/A"),
            "side": pos.get("side", "N/A"),
            "avg_entry_price": f"${float(pos.get('avg_entry_price', 0)):.2f}",
            "current_price": f"${float(pos.get('current_price', 0)):.2f}",
            "market_value": f"${market_value:,.2f}",
            "unrealized_pnl": f"${unrealized_pnl:+,.2f} ({unrealized_pnl_pct:+.2f}%)"
        })

    return {
        "check": "positions",
        "status": "PASS",
        "details": {
            "count": len(positions),
            "total_market_value": f"${total_market_value:,.2f}",
            "total_unrealized_pnl": f"${total_unrealized_pnl:+,.2f}",
            "positions": position_list
        }
    }


def run_status_checks() -> List[Dict[str, Any]]:
    """Run all status checks."""
    return [
        check_connectivity(),
        check_credentials(),
        get_account_info(),
        get_positions()
    ]


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def print_results(results: List[Dict[str, Any]], verbose: bool = False) -> int:
    """Print results and return exit code."""
    config = get_alpaca_config()

    print("=" * 70)
    print("KOBE BROKER CONNECTION STATUS")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print(f"Broker: Alpaca ({config['base_url']})")
    print("=" * 70)

    all_pass = True

    for result in results:
        check_name = result.get("check", "unknown")
        status = result.get("status", "UNKNOWN")

        if status == "FAIL":
            all_pass = False
            status_str = "[FAIL]"
        elif status == "WARN":
            status_str = "[WARN]"
        else:
            status_str = "[PASS]"

        print(f"\n{status_str} {check_name}")
        print("-" * 50)

        details = result.get("details", {})
        for key, value in details.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            elif isinstance(value, list):
                print(f"  {key}: ({len(value)} items)")
                if verbose or len(value) <= 5:
                    for item in value:
                        if isinstance(item, dict):
                            parts = [f"{k}={v}" for k, v in item.items()]
                            print(f"    - {', '.join(parts)}")
                        else:
                            print(f"    - {item}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    if all_pass:
        print("OVERALL: PASS - Broker connection healthy")
        return 0
    else:
        print("OVERALL: FAIL - Broker connection issues detected")
        return 2


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Kobe Broker Connection Status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python broker.py --status            # Full status check
  python broker.py --account           # Show account info only
  python broker.py --orders            # Show open orders only
  python broker.py --status --verbose  # Detailed output
        """
    )
    parser.add_argument("--dotenv", type=str, default=DEFAULT_DOTENV,
                        help="Path to .env file")
    parser.add_argument("--status", action="store_true",
                        help="Full status check (connectivity, credentials, account)")
    parser.add_argument("--account", action="store_true",
                        help="Show account balance and buying power")
    parser.add_argument("--orders", action="store_true",
                        help="Show open orders")
    parser.add_argument("--positions", action="store_true",
                        help="Show current positions")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
        if args.verbose:
            print(f"Loaded {len(loaded)} env vars from {dotenv}")

    # Default to --status if no flags
    if not args.status and not args.account and not args.orders and not args.positions:
        args.status = True

    try:
        results = []

        if args.status:
            results = run_status_checks()
        else:
            # Always check connectivity first
            results.append(check_connectivity())

            if args.account:
                results.append(get_account_info())

            if args.orders:
                results.append(get_open_orders())

            if args.positions:
                results.append(get_positions())

        exit_code = print_results(results, verbose=args.verbose)
        sys.exit(exit_code)
    except Exception as e:
        print(f"[ERROR] Broker status check failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
