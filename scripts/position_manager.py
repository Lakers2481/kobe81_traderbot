#!/usr/bin/env python3
"""
Position Manager for Kobe Trading System (Scheduler v2)

Intraday position monitoring and lifecycle management.
Runs every 5-15 minutes during market hours.

Features:
- Fetch current positions from Alpaca
- Track bars held since entry
- Check if stop hit or time stop expired
- Update trailing stops via TrailingStopManager
- Execute required exits
- Reconcile OMS with broker
- Alert on threshold breaches

Usage:
    python scripts/position_manager.py --dotenv ./.env
    python scripts/position_manager.py --check-only  # No exits, just report
    python scripts/position_manager.py --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from risk.policy_gate import PolicyGate
from risk.trailing_stops import get_trailing_stop_manager, StopUpdate
from risk.weekly_exposure_gate import get_weekly_exposure_gate
from oms.order_state import OrderRecord, OrderStatus
from core.kill_switch import is_kill_switch_active
from core.structured_log import get_logger

# Setup logging
logger = get_logger(__name__)

# Strategy-specific time stops (in bars/days)
TIME_STOPS = {
    "IBS_RSI": 7,        # IBS+RSI: 7-bar time stop
    "TURTLE_SOUP": 3,    # Turtle Soup: 3-bar time stop
    "ICT": 3,            # ICT strategies: 3-bar time stop
    "DEFAULT": 5         # Default: 5-bar time stop
}

ALPACA_TIMEOUT = 10
STATE_FILE = ROOT / "state" / "position_state.json"


@dataclass
class PositionState:
    """Tracked state for a position."""
    symbol: str
    entry_date: str
    entry_price: float
    qty: int
    side: str
    stop_loss: float
    initial_stop: float
    strategy: str
    bars_held: int
    last_check: str
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    r_multiple: float = 0.0
    stop_state: str = "initial"
    should_exit: bool = False
    exit_reason: Optional[str] = None


def get_alpaca_config() -> Dict[str, str]:
    """Get Alpaca API configuration from environment."""
    return {
        "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/"),
        "key_id": os.getenv("ALPACA_API_KEY_ID", os.getenv("APCA_API_KEY_ID", "")),
        "secret": os.getenv("ALPACA_API_SECRET_KEY", os.getenv("APCA_API_SECRET_KEY", ""))
    }


def get_auth_headers(config: Dict[str, str]) -> Dict[str, str]:
    """Get authentication headers for Alpaca API."""
    return {
        "APCA-API-KEY-ID": config["key_id"],
        "APCA-API-SECRET-KEY": config["secret"],
        "Content-Type": "application/json"
    }


def alpaca_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
    """Make a request to the Alpaca API."""
    config = get_alpaca_config()
    url = f"{config['base_url']}{endpoint}"
    headers = get_auth_headers(config)

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=ALPACA_TIMEOUT)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=ALPACA_TIMEOUT)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=ALPACA_TIMEOUT)
        else:
            response = requests.request(method, url, headers=headers, json=data, timeout=ALPACA_TIMEOUT)

        return {
            "success": response.status_code in (200, 201, 204),
            "status_code": response.status_code,
            "data": response.json() if response.content and response.status_code != 204 else None,
            "error": None
        }
    except requests.exceptions.Timeout:
        return {"success": False, "status_code": None, "data": None, "error": "Request timed out"}
    except requests.exceptions.ConnectionError as e:
        return {"success": False, "status_code": None, "data": None, "error": f"Connection error: {e}"}
    except Exception as e:
        return {"success": False, "status_code": None, "data": None, "error": str(e)}


def fetch_positions() -> List[Dict[str, Any]]:
    """Fetch current positions from Alpaca."""
    result = alpaca_request("/v2/positions")
    if not result["success"]:
        logger.error(f"Failed to fetch positions: {result['error']}")
        return []
    return result["data"] or []


def fetch_quote(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """Fetch current bid/ask for a symbol."""
    config = get_alpaca_config()
    data_base = config["base_url"].replace("paper-api", "data").replace("api.", "data.")
    if "data.alpaca.markets" not in data_base:
        data_base = "https://data.alpaca.markets"

    url = f"{data_base}/v2/stocks/quotes?symbols={symbol}"
    headers = get_auth_headers(config)

    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code != 200:
            return None, None
        data = r.json()
        quotes = data.get("quotes", {}).get(symbol, [])
        if not quotes:
            return None, None
        q = quotes[-1] if isinstance(quotes, list) else quotes
        bid = q.get("bp") or q.get("bid_price")
        ask = q.get("ap") or q.get("ask_price")
        return float(bid) if bid else None, float(ask) if ask else None
    except Exception as e:
        logger.debug(f"Quote fetch failed for {symbol}: {e}")
        return None, None


def load_position_state() -> Dict[str, PositionState]:
    """Load persisted position state from file."""
    if not STATE_FILE.exists():
        return {}

    try:
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
        return {k: PositionState(**v) for k, v in data.items()}
    except Exception as e:
        logger.warning(f"Failed to load position state: {e}")
        return {}


def save_position_state(states: Dict[str, PositionState]):
    """Save position state to file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = {k: asdict(v) for k, v in states.items()}
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save position state: {e}")


def calculate_bars_held(entry_date_str: str) -> int:
    """Calculate trading days (bars) held since entry."""
    try:
        entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
        today = date.today()

        # Count trading days (weekdays) between entry and today
        bars = 0
        current = entry_date
        while current <= today:
            if current.weekday() < 5:  # Mon-Fri
                bars += 1
            current += timedelta(days=1)
        return max(0, bars - 1)  # Subtract 1 (entry day doesn't count)
    except Exception:
        return 0


def get_time_stop_bars(strategy: str) -> int:
    """Get time stop threshold for a strategy."""
    strategy_upper = strategy.upper().replace(" ", "_").replace("-", "_")
    for key in TIME_STOPS:
        if key in strategy_upper:
            return TIME_STOPS[key]
    return TIME_STOPS["DEFAULT"]


def close_position(symbol: str, qty: int, side: str) -> bool:
    """Close a position via market order."""
    if is_kill_switch_active():
        logger.warning(f"Kill switch active - cannot close {symbol}")
        return False

    # Determine exit side
    exit_side = "sell" if side.lower() in ("long", "buy") else "buy"

    payload = {
        "symbol": symbol.upper(),
        "qty": str(abs(int(qty))),
        "side": exit_side,
        "type": "market",
        "time_in_force": "day"
    }

    logger.info(f"Closing position: {symbol} {exit_side} {qty} shares")
    result = alpaca_request("/v2/orders", method="POST", data=payload)

    if result["success"]:
        logger.info(f"Position close submitted for {symbol}")
        return True
    else:
        logger.error(f"Failed to close {symbol}: {result['error']}")
        return False


class PositionManager:
    """Manages intraday position lifecycle."""

    def __init__(self, check_only: bool = False, verbose: bool = False):
        self.check_only = check_only
        self.verbose = verbose
        self.tsm = get_trailing_stop_manager()
        self.policy_gate = PolicyGate.from_config()
        self.weekly_gate = get_weekly_exposure_gate()  # Professional Portfolio Allocation
        self.position_states: Dict[str, PositionState] = load_position_state()

    def run_cycle(self) -> Dict[str, Any]:
        """Run a full position management cycle."""
        logger.info("=" * 60)
        logger.info("POSITION MANAGER CYCLE")
        logger.info(f"Timestamp: {datetime.utcnow().isoformat()}Z")
        logger.info(f"Mode: {'CHECK ONLY' if self.check_only else 'LIVE'}")
        logger.info("=" * 60)

        # Check kill switch
        if is_kill_switch_active():
            logger.warning("Kill switch is ACTIVE - no exits will be executed")
            self.check_only = True

        # Fetch current positions
        positions = fetch_positions()
        logger.info(f"Found {len(positions)} open positions")

        if not positions:
            logger.info("No positions to manage")
            # Clean up stale state
            self.position_states = {}
            save_position_state(self.position_states)
            return {"positions": 0, "exits": 0, "updates": 0}

        # Process each position
        exits_needed = []
        stop_updates = []
        position_reports = []

        for pos in positions:
            symbol = pos.get("symbol")
            qty = int(float(pos.get("qty", 0)))
            side = pos.get("side", "long")
            entry_price = float(pos.get("avg_entry_price", 0))
            current_price = float(pos.get("current_price", 0))
            unrealized_pnl = float(pos.get("unrealized_pl", 0))

            # Get or create position state
            if symbol not in self.position_states:
                # New position - initialize state
                self.position_states[symbol] = PositionState(
                    symbol=symbol,
                    entry_date=date.today().isoformat(),
                    entry_price=entry_price,
                    qty=qty,
                    side=side,
                    stop_loss=entry_price * 0.95,  # Default 5% stop if unknown
                    initial_stop=entry_price * 0.95,
                    strategy="UNKNOWN",
                    bars_held=0,
                    last_check=datetime.utcnow().isoformat()
                )
                logger.info(f"New position detected: {symbol}")

            state = self.position_states[symbol]
            state.current_price = current_price
            state.unrealized_pnl = unrealized_pnl
            state.last_check = datetime.utcnow().isoformat()

            # Calculate bars held
            state.bars_held = calculate_bars_held(state.entry_date)

            # Check time stop
            time_stop_bars = get_time_stop_bars(state.strategy)
            if state.bars_held >= time_stop_bars:
                state.should_exit = True
                state.exit_reason = f"TIME_STOP ({state.bars_held} bars >= {time_stop_bars})"
                exits_needed.append(state)
                logger.warning(f"{symbol}: TIME STOP triggered ({state.bars_held} bars)")

            # Check price stop
            elif side.lower() in ("long", "buy") and current_price <= state.stop_loss:
                state.should_exit = True
                state.exit_reason = f"STOP_LOSS (${current_price:.2f} <= ${state.stop_loss:.2f})"
                exits_needed.append(state)
                logger.warning(f"{symbol}: STOP LOSS hit (${current_price:.2f})")

            elif side.lower() in ("short", "sell") and current_price >= state.stop_loss:
                state.should_exit = True
                state.exit_reason = f"STOP_LOSS (${current_price:.2f} >= ${state.stop_loss:.2f})"
                exits_needed.append(state)
                logger.warning(f"{symbol}: STOP LOSS hit (${current_price:.2f})")

            else:
                # Update trailing stop
                position_dict = {
                    "symbol": symbol,
                    "entry_price": state.entry_price,
                    "stop_loss": state.stop_loss,
                    "initial_stop": state.initial_stop,
                    "side": side
                }

                update = self.tsm.update_stop(
                    position=position_dict,
                    current_price=current_price,
                    bars_held=state.bars_held
                )

                state.r_multiple = update.r_multiple
                state.stop_state = update.state.value

                if update.should_update:
                    state.stop_loss = update.new_stop
                    stop_updates.append(update)
                    logger.info(f"{symbol}: Stop updated {update.old_stop:.2f} -> {update.new_stop:.2f} ({update.reason})")

            # Build report
            position_reports.append({
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "entry_price": f"${state.entry_price:.2f}",
                "current_price": f"${current_price:.2f}",
                "unrealized_pnl": f"${unrealized_pnl:+.2f}",
                "bars_held": state.bars_held,
                "time_stop_at": time_stop_bars,
                "stop_loss": f"${state.stop_loss:.2f}",
                "r_multiple": f"{state.r_multiple:.2f}R",
                "stop_state": state.stop_state,
                "should_exit": state.should_exit,
                "exit_reason": state.exit_reason
            })

        # Print position summary
        logger.info("\nPOSITION SUMMARY:")
        logger.info("-" * 80)
        for report in position_reports:
            logger.info(
                f"  {report['symbol']:6} | {report['qty']:4} | P&L: {report['unrealized_pnl']:>10} | "
                f"Bars: {report['bars_held']}/{report['time_stop_at']} | "
                f"Stop: {report['stop_loss']} | R: {report['r_multiple']} | "
                f"Exit: {report['exit_reason'] or 'NO'}"
            )

        # Execute exits
        exits_executed = 0
        if exits_needed and not self.check_only:
            logger.info(f"\nEXECUTING {len(exits_needed)} EXITS:")
            for state in exits_needed:
                success = close_position(state.symbol, state.qty, state.side)
                if success:
                    exits_executed += 1
                    # Record exit with weekly exposure gate (Professional Portfolio Allocation)
                    # This frees budget for next scan
                    self.weekly_gate.record_exit(
                        symbol=state.symbol,
                        exit_reason=state.exit_reason or "CLOSED"
                    )
                    logger.info(f"  {state.symbol}: Budget freed (available at next scan)")
                    # Remove from state tracking
                    self.tsm.reset_position(state.symbol)
                    if state.symbol in self.position_states:
                        del self.position_states[state.symbol]
        elif exits_needed and self.check_only:
            logger.info(f"\n[CHECK ONLY] Would execute {len(exits_needed)} exits")

        # Clean up closed positions
        current_symbols = {p.get("symbol") for p in positions}
        for symbol in list(self.position_states.keys()):
            if symbol not in current_symbols:
                logger.info(f"Position {symbol} no longer exists - removing from state")
                # Record exit with weekly exposure gate (position closed externally - target/stop hit)
                self.weekly_gate.record_exit(
                    symbol=symbol,
                    exit_reason="CLOSED_EXTERNAL"
                )
                logger.info(f"  {symbol}: Budget freed (closed externally)")
                self.tsm.reset_position(symbol)
                del self.position_states[symbol]

        # Save state
        save_position_state(self.position_states)

        # Update policy gate position count
        self.policy_gate.update_position_count(len(positions))

        # Summary
        weekly_status = self.weekly_gate.get_status()
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "positions": len(positions),
            "exits_needed": len(exits_needed),
            "exits_executed": exits_executed,
            "stop_updates": len(stop_updates),
            "policy_gate": self.policy_gate.get_status(),
            "weekly_gate": weekly_status
        }

        logger.info("\nCYCLE SUMMARY:")
        logger.info(f"  Positions: {summary['positions']}")
        logger.info(f"  Exits needed: {summary['exits_needed']}")
        logger.info(f"  Exits executed: {summary['exits_executed']}")
        logger.info(f"  Stop updates: {summary['stop_updates']}")
        logger.info(f"  Trading mode: {summary['policy_gate']['trading_mode']}")
        logger.info(f"  Daily remaining: ${summary['policy_gate']['daily_remaining']:.2f}")
        logger.info(f"\nWEEKLY BUDGET:")
        logger.info(f"  Current Exposure: {weekly_status['exposure']['current_pct']}")
        logger.info(f"  Daily Entries Today: {weekly_status['daily']['entries_today']}/{weekly_status['daily']['max_per_day']}")
        logger.info(f"  Budget Freed Pending: ${weekly_status['budget'].get('freed_pending', 0):,.0f}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Kobe Position Manager - Intraday position lifecycle management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/position_manager.py --dotenv ./.env
    python scripts/position_manager.py --check-only
    python scripts/position_manager.py --verbose
        """
    )
    parser.add_argument("--dotenv", type=str, default="./.env",
                        help="Path to .env file")
    parser.add_argument("--check-only", action="store_true",
                        help="Report only, no exits executed")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run position manager
    try:
        pm = PositionManager(check_only=args.check_only, verbose=args.verbose)
        summary = pm.run_cycle()

        # Exit code based on results
        if summary.get("exits_needed", 0) > summary.get("exits_executed", 0):
            sys.exit(1)  # Some exits failed
        sys.exit(0)
    except Exception as e:
        logger.error(f"Position manager failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
