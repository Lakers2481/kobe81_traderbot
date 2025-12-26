#!/usr/bin/env python3
"""
Kobe AI Suggestions - Analyze system state and suggest next actions.

Usage:
    python suggest.py --analyze
    python suggest.py --analyze --verbose
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.env_loader import load_env

STATE_DIR = ROOT / "state"
LOGS_DIR = ROOT / "logs"
RECONCILE_DIR = STATE_DIR / "reconcile"


def get_system_status() -> Dict[str, Any]:
    """Gather current system status."""
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "api_keys_set": {},
        "positions": [],
        "recent_orders": [],
        "logs_available": False,
        "state_files": [],
    }

    # Check API keys
    status["api_keys_set"]["polygon"] = bool(os.getenv("POLYGON_API_KEY"))
    status["api_keys_set"]["alpaca_key"] = bool(os.getenv("ALPACA_API_KEY_ID"))
    status["api_keys_set"]["alpaca_secret"] = bool(os.getenv("ALPACA_API_SECRET_KEY"))

    # Check state files
    for file in STATE_DIR.glob("*.json"):
        status["state_files"].append(str(file.name))
    for file in STATE_DIR.glob("*.jsonl"):
        status["state_files"].append(str(file.name))

    # Check logs
    status["logs_available"] = LOGS_DIR.exists() and any(LOGS_DIR.glob("*.jsonl"))

    # Get positions from local state or API
    positions_file = RECONCILE_DIR / "positions.json"
    if positions_file.exists():
        try:
            status["positions"] = json.loads(positions_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Get recent orders
    orders_file = RECONCILE_DIR / "orders_all.json"
    if orders_file.exists():
        try:
            all_orders = json.loads(orders_file.read_text(encoding="utf-8"))
            status["recent_orders"] = all_orders[:10] if len(all_orders) > 10 else all_orders
        except Exception:
            pass

    return status


def analyze_positions(positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze current positions for actionable insights."""
    insights = []

    if not positions:
        insights.append({
            "type": "info",
            "category": "positions",
            "message": "No open positions. Consider running a scan for new signals.",
            "action": "Run: python scripts/runner.py --once --mode paper --universe universes/sp500.csv",
        })
        return insights

    total_value = 0
    winners = []
    losers = []

    for pos in positions:
        qty = float(pos.get("qty", 0))
        market_value = float(pos.get("market_value", 0))
        unrealized_pl = float(pos.get("unrealized_pl", 0))
        unrealized_plpc = float(pos.get("unrealized_plpc", 0))

        total_value += abs(market_value)

        if unrealized_pl > 0:
            winners.append({
                "symbol": pos.get("symbol"),
                "pl": unrealized_pl,
                "pct": unrealized_plpc,
            })
        elif unrealized_pl < 0:
            losers.append({
                "symbol": pos.get("symbol"),
                "pl": unrealized_pl,
                "pct": unrealized_plpc,
            })

    # Sort by P&L
    winners.sort(key=lambda x: x["pl"], reverse=True)
    losers.sort(key=lambda x: x["pl"])

    # Generate insights
    if winners:
        best = winners[0]
        if best["pct"] > 0.05:  # >5% gain
            insights.append({
                "type": "opportunity",
                "category": "take_profit",
                "message": f"{best['symbol']} is up {best['pct']*100:.1f}% (+${best['pl']:.2f}). Consider taking partial profits.",
                "action": f"Review exit rules for {best['symbol']}",
            })

    if losers:
        worst = losers[0]
        if worst["pct"] < -0.03:  # >3% loss
            insights.append({
                "type": "warning",
                "category": "risk",
                "message": f"{worst['symbol']} is down {abs(worst['pct'])*100:.1f}% (${worst['pl']:.2f}). Review stop loss.",
                "action": f"Check if stop loss should be tightened for {worst['symbol']}",
            })

    # Concentration warning
    if len(positions) > 0 and len(positions) < 3:
        insights.append({
            "type": "info",
            "category": "diversification",
            "message": f"Only {len(positions)} position(s). Consider diversifying.",
            "action": "Run signal scan to find additional opportunities",
        })

    return insights


def analyze_orders(orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze recent orders for patterns."""
    insights = []

    if not orders:
        return insights

    # Check for rejected orders
    rejected = [o for o in orders if o.get("status") == "rejected"]
    if rejected:
        insights.append({
            "type": "warning",
            "category": "orders",
            "message": f"{len(rejected)} rejected order(s) in history. Check order parameters.",
            "action": "Review logs for rejection reasons",
        })

    # Check for unfilled orders
    unfilled = [o for o in orders if o.get("status") in ["new", "accepted", "pending_new"]]
    if unfilled:
        insights.append({
            "type": "info",
            "category": "orders",
            "message": f"{len(unfilled)} order(s) still pending. Monitor for fills.",
            "action": "Check order status in broker dashboard",
        })

    return insights


def check_market_conditions() -> List[Dict[str, Any]]:
    """Check basic market conditions."""
    insights = []
    now = datetime.utcnow()

    # Weekend check
    if now.weekday() >= 5:
        insights.append({
            "type": "info",
            "category": "market",
            "message": "Market is closed (weekend). Use this time to review and plan.",
            "action": "Review past week's trades and update journal",
        })
    else:
        # Market hours check (approximate - 9:30 AM to 4:00 PM ET)
        # UTC offset for ET is -5 or -4 depending on DST
        et_hour = (now.hour - 5) % 24  # Rough approximation
        if et_hour < 9 or et_hour >= 16:
            insights.append({
                "type": "info",
                "category": "market",
                "message": "Market is likely closed. Check pre-market for gaps.",
                "action": "Review overnight news and pre-market movers",
            })
        elif 9 <= et_hour < 10:
            insights.append({
                "type": "opportunity",
                "category": "market",
                "message": "Market open hour - higher volatility expected.",
                "action": "Watch for signal triggers but be cautious of whipsaws",
            })
        elif 15 <= et_hour < 16:
            insights.append({
                "type": "info",
                "category": "market",
                "message": "Final hour of trading. MOC orders affect price.",
                "action": "Consider closing day trades before close",
            })

    return insights


def check_system_health() -> List[Dict[str, Any]]:
    """Check system configuration and health."""
    insights = []

    # Check API keys
    if not os.getenv("POLYGON_API_KEY"):
        insights.append({
            "type": "error",
            "category": "config",
            "message": "POLYGON_API_KEY not set. Cannot fetch market data.",
            "action": "Set POLYGON_API_KEY in .env file",
        })

    if not os.getenv("ALPACA_API_KEY_ID") or not os.getenv("ALPACA_API_SECRET_KEY"):
        insights.append({
            "type": "error",
            "category": "config",
            "message": "Alpaca API credentials not set. Cannot trade.",
            "action": "Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY in .env file",
        })

    # Check state directory
    if not STATE_DIR.exists():
        insights.append({
            "type": "warning",
            "category": "setup",
            "message": "State directory does not exist. System state may not persist.",
            "action": "Run any script to auto-create state directory",
        })

    # Check for recent logs
    if LOGS_DIR.exists():
        log_files = list(LOGS_DIR.glob("*.jsonl"))
        if log_files:
            newest = max(log_files, key=lambda f: f.stat().st_mtime)
            age = datetime.now().timestamp() - newest.stat().st_mtime
            if age > 86400:  # More than 24 hours
                insights.append({
                    "type": "warning",
                    "category": "logs",
                    "message": "No recent log activity (>24h). System may not be running.",
                    "action": "Check if scheduled jobs are active",
                })

    return insights


def generate_priority_actions(all_insights: List[Dict[str, Any]]) -> List[str]:
    """Generate prioritized action list from insights."""
    actions = []

    # Errors first
    for i in all_insights:
        if i["type"] == "error":
            actions.append(f"[CRITICAL] {i['action']}")

    # Warnings next
    for i in all_insights:
        if i["type"] == "warning":
            actions.append(f"[HIGH] {i['action']}")

    # Opportunities
    for i in all_insights:
        if i["type"] == "opportunity":
            actions.append(f"[OPPORTUNITY] {i['action']}")

    # Info items
    for i in all_insights:
        if i["type"] == "info" and i["action"] not in [a.split("] ")[1] for a in actions if "] " in a]:
            actions.append(f"[INFO] {i['action']}")

    return actions[:10]  # Top 10 actions


def main() -> None:
    ap = argparse.ArgumentParser(description="Kobe AI Suggestions")
    ap.add_argument("--dotenv", type=str, default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
                    help="Path to .env file")

    # Actions
    ap.add_argument("--analyze", action="store_true", help="Analyze system and generate suggestions")
    ap.add_argument("--verbose", action="store_true", help="Show detailed analysis")

    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    if not args.analyze:
        print("Specify --analyze to run analysis")
        print("Use --help for usage information")
        sys.exit(1)

    print("Kobe AI Suggestions")
    print("=" * 70)
    print(f"Analysis Time: {datetime.utcnow().isoformat()}")

    # Gather status
    status = get_system_status()

    # Collect all insights
    all_insights: List[Dict[str, Any]] = []

    # System health
    print("\nChecking system health...")
    health_insights = check_system_health()
    all_insights.extend(health_insights)

    # Market conditions
    print("Checking market conditions...")
    market_insights = check_market_conditions()
    all_insights.extend(market_insights)

    # Position analysis
    print("Analyzing positions...")
    position_insights = analyze_positions(status["positions"])
    all_insights.extend(position_insights)

    # Order analysis
    print("Analyzing orders...")
    order_insights = analyze_orders(status["recent_orders"])
    all_insights.extend(order_insights)

    # Generate priority actions
    priority_actions = generate_priority_actions(all_insights)

    # Output
    print("\n" + "=" * 70)
    print("SUGGESTED ACTIONS (Priority Order)")
    print("=" * 70)

    if not priority_actions:
        print("\nNo specific actions suggested. System appears healthy.")
        print("Consider running a signal scan or reviewing recent performance.")
    else:
        for i, action in enumerate(priority_actions, 1):
            print(f"\n{i}. {action}")

    if args.verbose:
        print("\n" + "=" * 70)
        print("DETAILED INSIGHTS")
        print("=" * 70)

        for insight in all_insights:
            type_icon = {
                "error": "[!]",
                "warning": "[W]",
                "opportunity": "[O]",
                "info": "[i]",
            }.get(insight["type"], "[-]")

            print(f"\n{type_icon} {insight['category'].upper()}: {insight['message']}")

        print("\n" + "-" * 70)
        print("SYSTEM STATUS")
        print("-" * 70)
        print(f"Positions: {len(status['positions'])}")
        print(f"Recent Orders: {len(status['recent_orders'])}")
        print(f"State Files: {', '.join(status['state_files']) if status['state_files'] else 'none'}")
        print(f"Logs Available: {status['logs_available']}")
        print(f"API Keys Set: {json.dumps(status['api_keys_set'])}")

    print("\n" + "=" * 70)
    print("Run with --verbose for detailed analysis")


if __name__ == "__main__":
    main()
