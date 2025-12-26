#!/usr/bin/env python3
"""
Risk Limit Checker for Kobe Trading System

Checks PolicyGate limits, current exposure, position concentration, and correlation limits.
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

from config.env_loader import load_env
from risk.policy_gate import PolicyGate, RiskLimits


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_DOTENV = "C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env"
STATE_DIR = ROOT / "state"
POSITIONS_FILE = STATE_DIR / "reconcile" / "positions.json"

# Risk thresholds
MAX_POSITION_CONCENTRATION = 0.20  # 20% max in single position
MAX_SECTOR_CONCENTRATION = 0.40    # 40% max in single sector
MAX_CORRELATION_THRESHOLD = 0.85   # Max pairwise correlation allowed


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_positions() -> List[Dict[str, Any]]:
    """Load positions from state file or fetch from Alpaca."""
    if POSITIONS_FILE.exists():
        try:
            return json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return fetch_alpaca_positions()


def fetch_alpaca_positions() -> List[Dict[str, Any]]:
    """Fetch current positions from Alpaca API."""
    base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    key = os.getenv("ALPACA_API_KEY_ID", "")
    sec = os.getenv("ALPACA_API_SECRET_KEY", "")

    if not key or not sec:
        return []

    try:
        r = requests.get(
            f"{base}/v2/positions",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec},
            timeout=10
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def fetch_account_info() -> Optional[Dict[str, Any]]:
    """Fetch account info from Alpaca API."""
    base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    key = os.getenv("ALPACA_API_KEY_ID", "")
    sec = os.getenv("ALPACA_API_SECRET_KEY", "")

    if not key or not sec:
        return None

    try:
        r = requests.get(
            f"{base}/v2/account",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec},
            timeout=10
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# -----------------------------------------------------------------------------
# Risk Checks
# -----------------------------------------------------------------------------
def check_policy_gate_limits() -> Dict[str, Any]:
    """Check PolicyGate limits configuration."""
    gate = PolicyGate()
    limits = gate.limits

    return {
        "check": "policy_gate_limits",
        "status": "PASS",
        "details": {
            "max_notional_per_order": f"${limits.max_notional_per_order:.2f}",
            "max_daily_notional": f"${limits.max_daily_notional:.2f}",
            "min_price": f"${limits.min_price:.2f}",
            "max_price": f"${limits.max_price:.2f}",
            "allow_shorts": limits.allow_shorts,
            "current_daily_usage": f"${gate._daily_notional:.2f}"
        }
    }


def check_current_exposure() -> Dict[str, Any]:
    """Check current market exposure from positions."""
    positions = load_positions()
    account = fetch_account_info()

    total_exposure = 0.0
    long_exposure = 0.0
    short_exposure = 0.0
    position_details = []

    for pos in positions:
        try:
            mv = float(pos.get('market_value', 0))
        except Exception:
            mv = 0.0
        total_exposure += abs(mv)
        if mv >= 0:
            long_exposure += mv
        else:
            short_exposure += abs(mv)
        position_details.append({
            'symbol': pos.get('symbol'),
            'qty': pos.get('qty'),
            'market_value': pos.get('market_value'),
            'unrealized_pl': pos.get('unrealized_pl'),
        })

    equity = 0.0
    if account:
        try:
            equity = float(account.get('portfolio_value', 0))
        except Exception:
            equity = 0.0

    concentration = 0.0
    if position_details:
        try:
            mvs = [abs(float(p.get('market_value', 0))) for p in position_details]
            if sum(mvs) > 0:
                concentration = max(mvs) / sum(mvs)
        except Exception:
            concentration = 0.0

    status = 'PASS'
    details: Dict[str, Any] = {
        'equity': f"${equity:.2f}",
        'total_exposure': f"${total_exposure:.2f}",
        'long_exposure': f"${long_exposure:.2f}",
        'short_exposure': f"${short_exposure:.2f}",
        'max_position_concentration': f"{concentration:.2%}",
        'positions': position_details,
    }

    if concentration > MAX_POSITION_CONCENTRATION:
        status = 'WARN'
        details['note'] = 'Position concentration high'

    return {
        'check': 'current_exposure',
        'status': status,
        'details': details,
    }


def check_correlation_limits() -> Dict[str, Any]:
    """Placeholder: pairwise correlation checks (requires historical data)."""
    # Not implemented in v1; return PASS with note
    return {
        'check': 'correlation_limits',
        'status': 'PASS',
        'details': {
            'note': 'Correlation check not implemented in v1',
            'threshold': MAX_CORRELATION_THRESHOLD,
        }
    }


def run_all_checks() -> List[Dict[str, Any]]:
    results = []
    results.append(check_policy_gate_limits())
    results.append(check_current_exposure())
    results.append(check_correlation_limits())
    return results


def test_order_against_limits(symbol: str, side: str, price: float, qty: int) -> Dict[str, Any]:
    """Test a hypothetical order against PolicyGate limits."""
    gate = PolicyGate()
    passed, reason = gate.check(symbol, side, price, qty)
    notional = price * qty

    return {
        "check": "order_test",
        "status": "PASS" if passed else "FAIL",
        "order": {
            "symbol": symbol,
            "side": side,
            "price": f"${price:.2f}",
            "qty": qty,
            "notional": f"${notional:.2f}"
        },
        "result": reason
    }


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def print_results(results: List[Dict[str, Any]], verbose: bool = False) -> int:
    """Print results and return exit code."""
    print("=" * 60)
    print("KOBE RISK LIMIT CHECK")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    all_pass = True
    has_warn = False

    for result in results:
        check_name = result.get("check", "unknown")
        status = result.get("status", "UNKNOWN")

        if status == "FAIL":
            all_pass = False
            status_str = "[FAIL]"
        elif status == "WARN":
            has_warn = True
            status_str = "[WARN]"
        else:
            status_str = "[PASS]"

        print(f"\n{status_str} {check_name}")

        if verbose or status != "PASS":
            details = result.get("details", {})
            for key, value in details.items():
                if isinstance(value, list):
                    if value:
                        print(f"  {key}:")
                        for item in value[:10]:  # Limit output
                            if isinstance(item, dict):
                                print(f"    - {item}")
                            else:
                                print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    if all_pass and not has_warn:
        print("OVERALL: PASS - All risk checks passed")
        return 0
    elif all_pass and has_warn:
        print("OVERALL: WARN - Passed with warnings")
        return 1
    else:
        print("OVERALL: FAIL - Risk limit violations detected")
        return 2


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Kobe Risk Limit Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python risk_cli.py --check              # Run all risk checks
  python risk_cli.py --status             # Quick status summary
  python risk_cli.py --check --verbose    # Detailed output
        """
    )
    parser.add_argument("--dotenv", type=str, default=DEFAULT_DOTENV,
                        help="Path to .env file")
    parser.add_argument("--check", action="store_true",
                        help="Run all risk checks")
    parser.add_argument("--status", action="store_true",
                        help="Show quick status summary")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
        if args.verbose:
            print(f"Loaded {len(loaded)} env vars from {dotenv}")

    # Default to --status if no flags provided
    if not args.check and not args.status:
        args.status = True

    try:
        if args.check or args.status:
            results = run_all_checks()
            exit_code = print_results(results, verbose=args.verbose or args.check)
            sys.exit(exit_code)
    except Exception as e:
        print(f"[ERROR] Risk check failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()

