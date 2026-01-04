#!/usr/bin/env python3
"""
Kobe Hedge Suggestions - Analyze positions and suggest protective hedges.

Usage:
    python hedge.py --analyze
    python hedge.py --suggest
    python hedge.py --suggest --max-cost 500
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.env_loader import load_env

STATE_DIR = ROOT / "state"
RECONCILE_DIR = STATE_DIR / "reconcile"


def fetch_current_positions() -> List[Dict[str, Any]]:
    """
    Fetch current positions from Alpaca API or local state.
    Returns list of position dictionaries.
    """
    import requests

    # First try API
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    key_id = os.getenv("ALPACA_API_KEY_ID", "")
    secret = os.getenv("ALPACA_API_SECRET_KEY", "")

    if key_id and secret:
        headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret,
        }
        try:
            resp = requests.get(f"{base_url}/v2/positions", headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

    # Fall back to local state
    positions_file = RECONCILE_DIR / "positions.json"
    if positions_file.exists():
        try:
            return json.loads(positions_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    return []


def analyze_position_risk(position: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze risk metrics for a single position.
    """
    symbol = position.get("symbol", "UNKNOWN")
    qty = float(position.get("qty", 0))
    avg_entry = float(position.get("avg_entry_price", 0))
    current_price = float(position.get("current_price", avg_entry))
    market_value = float(position.get("market_value", qty * current_price))
    unrealized_pl = float(position.get("unrealized_pl", 0))
    unrealized_plpc = float(position.get("unrealized_plpc", 0))

    # Calculate risk metrics
    position_cost = abs(qty * avg_entry)
    at_risk_5pct = position_cost * 0.05
    at_risk_10pct = position_cost * 0.10
    at_risk_20pct = position_cost * 0.20

    return {
        "symbol": symbol,
        "qty": qty,
        "avg_entry": avg_entry,
        "current_price": current_price,
        "market_value": market_value,
        "unrealized_pl": unrealized_pl,
        "unrealized_plpc": unrealized_plpc,
        "position_cost": position_cost,
        "at_risk_5pct": at_risk_5pct,
        "at_risk_10pct": at_risk_10pct,
        "at_risk_20pct": at_risk_20pct,
        "side": "long" if qty > 0 else "short",
    }


def estimate_put_cost(
    symbol: str,
    strike_pct: float,  # e.g., 0.95 for 5% OTM put
    current_price: float,
    position_size: float,
) -> Dict[str, Any]:
    """
    Estimate cost of protective put.
    Uses simplified Black-Scholes approximation.

    Note: This is a rough estimate. Real pricing requires:
    - Actual options chain data
    - Current IV
    - Time to expiration
    """
    strike = current_price * strike_pct
    contracts_needed = max(1, int(position_size / 100))

    # Simplified premium estimate
    # Assumes ~30 DTE, moderate IV (~30%), ATM put ~3% of stock price
    # OTM puts are cheaper
    otm_factor = 1.0 - strike_pct  # How far OTM
    base_premium_pct = 0.03  # ~3% for ATM
    premium_pct = base_premium_pct * (1 - otm_factor * 2)  # Reduce for OTM
    premium_pct = max(0.005, premium_pct)  # Floor at 0.5%

    premium_per_share = current_price * premium_pct
    premium_per_contract = premium_per_share * 100
    total_cost = premium_per_contract * contracts_needed

    # Protection provided
    protection_per_share = current_price - strike
    total_protection = protection_per_share * (contracts_needed * 100)

    return {
        "symbol": symbol,
        "strike": strike,
        "strike_pct": strike_pct,
        "contracts": contracts_needed,
        "estimated_premium": premium_per_contract,
        "total_cost": total_cost,
        "protection_value": total_protection,
        "cost_vs_protection": total_cost / total_protection if total_protection > 0 else 0,
        "note": "Estimate only - actual prices may vary significantly",
    }


def suggest_hedges(
    positions: List[Dict[str, Any]],
    max_cost: Optional[float] = None,
    protection_level: float = 0.10,  # 10% protection by default
) -> List[Dict[str, Any]]:
    """
    Generate hedge suggestions for all positions.
    """
    suggestions = []

    for pos in positions:
        risk = analyze_position_risk(pos)

        # Only hedge long positions (shorts are already bearish)
        if risk["side"] != "long":
            continue

        # Skip small positions
        if risk["position_cost"] < 500:
            continue

        current_price = risk["current_price"]
        position_size = abs(risk["qty"])

        # Suggest protective put at specified protection level
        strike_pct = 1.0 - protection_level
        put_estimate = estimate_put_cost(
            risk["symbol"],
            strike_pct,
            current_price,
            position_size,
        )

        # Check budget constraint
        if max_cost and put_estimate["total_cost"] > max_cost:
            # Suggest smaller hedge
            reduced_contracts = max(1, int(max_cost / put_estimate["estimated_premium"]))
            put_estimate["contracts"] = reduced_contracts
            put_estimate["total_cost"] = put_estimate["estimated_premium"] * reduced_contracts
            put_estimate["note"] += f" (reduced to fit ${max_cost} budget)"

        suggestion = {
            "position": risk,
            "hedge_type": "protective_put",
            "put_details": put_estimate,
            "alternative_hedges": [
                {
                    "type": "collar",
                    "description": f"Buy {strike_pct:.0%} put, sell {1 + protection_level:.0%} call",
                    "note": "Caps upside but reduces/eliminates hedge cost",
                },
                {
                    "type": "stop_loss",
                    "description": f"Set stop at ${current_price * strike_pct:.2f}",
                    "note": "Free but may get stopped out on volatility",
                },
                {
                    "type": "reduce_position",
                    "description": f"Sell {int(position_size * 0.5)} shares",
                    "note": "Reduces exposure without options cost",
                },
            ],
        }
        suggestions.append(suggestion)

    return suggestions


def print_analysis(positions: List[Dict[str, Any]]) -> None:
    """Print position risk analysis."""
    if not positions:
        print("No positions found.")
        return

    print("\nPosition Risk Analysis")
    print("=" * 80)

    total_value = 0
    total_at_risk = 0

    for pos in positions:
        risk = analyze_position_risk(pos)
        total_value += risk["market_value"]
        total_at_risk += risk["at_risk_10pct"]

        pl_sign = "+" if risk["unrealized_pl"] >= 0 else ""
        pl_pct = risk["unrealized_plpc"] * 100

        print(f"\n{risk['symbol']} ({risk['side'].upper()})")
        print(f"  Position: {risk['qty']} shares @ ${risk['avg_entry']:.2f}")
        print(f"  Current:  ${risk['current_price']:.2f} ({pl_sign}${risk['unrealized_pl']:.2f}, {pl_sign}{pl_pct:.1f}%)")
        print(f"  Value:    ${risk['market_value']:.2f}")
        print(f"  At Risk (10% drop): ${risk['at_risk_10pct']:.2f}")

    print("\n" + "-" * 80)
    print(f"Total Portfolio Value: ${total_value:,.2f}")
    print(f"Total at Risk (10%):   ${total_at_risk:,.2f}")


def print_suggestions(suggestions: List[Dict[str, Any]]) -> None:
    """Print hedge suggestions."""
    if not suggestions:
        print("No hedge suggestions (no hedgeable positions).")
        return

    print("\nHedge Suggestions")
    print("=" * 80)

    total_hedge_cost = 0

    for i, sug in enumerate(suggestions, 1):
        pos = sug["position"]
        put = sug["put_details"]
        total_hedge_cost += put["total_cost"]

        print(f"\n{i}. {pos['symbol']}")
        print(f"   Position: {pos['qty']} shares, ${pos['market_value']:.2f} value")
        print(f"   Current Price: ${pos['current_price']:.2f}")
        print("\n   RECOMMENDED: Protective Put")
        print(f"   Strike: ${put['strike']:.2f} ({put['strike_pct']:.0%} of current)")
        print(f"   Contracts: {put['contracts']}")
        print(f"   Estimated Cost: ${put['total_cost']:.2f}")
        print(f"   Protection Value: ${put['protection_value']:.2f}")
        print(f"   Note: {put['note']}")

        print("\n   Alternative Hedges:")
        for alt in sug["alternative_hedges"]:
            print(f"   - {alt['type'].upper()}: {alt['description']}")
            print(f"     ({alt['note']})")

    print("\n" + "-" * 80)
    print(f"Total Estimated Hedge Cost: ${total_hedge_cost:,.2f}")
    print("\nDISCLAIMER: These are estimates only. Actual options prices depend on")
    print("current IV, time to expiry, and market conditions. Consult actual")
    print("options chain before trading.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kobe Hedge Suggestions")
    ap.add_argument("--dotenv", type=str, default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
                    help="Path to .env file")

    # Actions
    ap.add_argument("--analyze", action="store_true", help="Analyze current position risk")
    ap.add_argument("--suggest", action="store_true", help="Suggest hedges for positions")

    # Options
    ap.add_argument("--max-cost", type=float, help="Maximum total hedge cost budget")
    ap.add_argument("--protection", type=float, default=0.10,
                    help="Protection level (0.10 = 10%% OTM put)")

    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    if not args.analyze and not args.suggest:
        print("Specify --analyze or --suggest")
        print("Use --help for usage information")
        sys.exit(1)

    # Fetch positions
    print("Fetching current positions...")
    positions = fetch_current_positions()

    if not positions:
        print("\nNo positions found.")
        print("Make sure Alpaca API credentials are set and you have open positions,")
        print("or run reconcile_alpaca.py first to cache position data.")
        sys.exit(0)

    print(f"Found {len(positions)} position(s)")

    if args.analyze:
        print_analysis(positions)

    if args.suggest:
        suggestions = suggest_hedges(
            positions,
            max_cost=args.max_cost,
            protection_level=args.protection,
        )
        print_suggestions(suggestions)


if __name__ == "__main__":
    main()
