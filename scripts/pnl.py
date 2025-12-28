#!/usr/bin/env python3
"""
Kobe Trading System - P&L Summary

Display realized P&L summaries:
- Today's realized P&L
- Week/Month/Total P&L
- P&L by strategy
- Win/loss breakdown

Usage:
    python pnl.py                    # Default: show all periods
    python pnl.py --period today     # Today's P&L only
    python pnl.py --period week      # This week's P&L
    python pnl.py --period month     # This month's P&L
    python pnl.py --strategy ibs_rsi    # Filter by strategy
    python pnl.py --dotenv /path/to/.env
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.env_loader import load_env

# Default paths
STATE_DIR = ROOT / "state"
TRADES_FILE = STATE_DIR / "trades.json"
ORDERS_FILE = STATE_DIR / "orders.json"
LOGS_DIR = ROOT / "logs"
DEFAULT_DOTENV = "C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env"


def load_trades() -> List[Dict[str, Any]]:
    """Load completed trades from state file."""
    trades: List[Dict[str, Any]] = []

    # Try trades.json first
    if TRADES_FILE.exists():
        try:
            data = json.loads(TRADES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                trades.extend(data)
            elif isinstance(data, dict):
                trades.extend(data.get("trades", []))
        except Exception as e:
            print(f"[WARN] Failed to load trades.json: {e}")

    # Also check orders.json for filled orders
    if ORDERS_FILE.exists():
        try:
            data = json.loads(ORDERS_FILE.read_text(encoding="utf-8"))
            orders = data if isinstance(data, list) else data.get("orders", [])
            for order in orders:
                if order.get("status") in ("FILLED", "CLOSED"):
                    trades.append(order)
        except Exception as e:
            print(f"[WARN] Failed to load orders.json: {e}")

    # Parse log files for trade events
    events_file = LOGS_DIR / "events.jsonl"
    if events_file.exists():
        try:
            for line in events_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if event.get("event") in ("trade_closed", "order_filled", "position_closed"):
                        trades.append(event)
                except Exception:
                    continue
        except Exception as e:
            print(f"[WARN] Failed to parse events.jsonl: {e}")

    return trades


def parse_timestamp(ts: Any) -> Optional[datetime]:
    """Parse various timestamp formats."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        ts_str = str(ts)
        # Try ISO format
        if "T" in ts_str:
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00").replace("+00:00", ""))
        # Try date only
        if len(ts_str) >= 10:
            return datetime.strptime(ts_str[:10], "%Y-%m-%d")
    except Exception:
        pass
    return None


def filter_trades_by_period(
    trades: List[Dict[str, Any]], period: Optional[str]
) -> List[Dict[str, Any]]:
    """Filter trades by time period."""
    if not period or period == "all":
        return trades

    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if period == "today":
        cutoff = today_start
    elif period == "week":
        # Start of current week (Monday)
        days_since_monday = now.weekday()
        cutoff = today_start - timedelta(days=days_since_monday)
    elif period == "month":
        cutoff = today_start.replace(day=1)
    elif period == "ytd":
        cutoff = today_start.replace(month=1, day=1)
    else:
        return trades

    filtered = []
    for trade in trades:
        ts = parse_timestamp(
            trade.get("timestamp")
            or trade.get("ts")
            or trade.get("closed_at")
            or trade.get("filled_at")
        )
        if ts and ts >= cutoff:
            filtered.append(trade)
    return filtered


def filter_trades_by_strategy(
    trades: List[Dict[str, Any]], strategy: Optional[str]
) -> List[Dict[str, Any]]:
    """Filter trades by strategy name."""
    if not strategy:
        return trades

    strategy_lower = strategy.lower()
    filtered = []
    for trade in trades:
        trade_strat = (
            trade.get("strategy", "")
            or trade.get("signal_source", "")
            or trade.get("reason", "")
        ).lower()

        # Match by keyword
        if strategy_lower in trade_strat:
            filtered.append(trade)
        elif strategy_lower == "ibs_rsi" and ("ibs" in trade_strat or "rsi" in trade_strat):
            filtered.append(trade)
        elif strategy_lower == "TURTLE_SOUP" and "TURTLE_SOUP" in trade_strat:
            filtered.append(trade)
    return filtered


def calculate_trade_pnl(trade: Dict[str, Any]) -> Tuple[float, bool]:
    """Calculate P&L for a trade. Returns (pnl, is_win)."""
    # Check if P&L is directly available
    pnl = trade.get("pnl") or trade.get("realized_pnl") or trade.get("profit")
    if pnl is not None:
        pnl = float(pnl)
        return pnl, pnl >= 0

    # Calculate from entry/exit
    entry = trade.get("entry_price") or trade.get("avg_entry_price") or trade.get("avg_cost")
    exit_price = trade.get("exit_price") or trade.get("close_price") or trade.get("fill_price")
    qty = trade.get("qty") or trade.get("quantity", 0)
    side = (trade.get("side", "") or trade.get("direction", "")).lower()

    if entry and exit_price and qty:
        entry = float(entry)
        exit_price = float(exit_price)
        qty = abs(int(qty))
        if side in ("long", "buy"):
            pnl = (exit_price - entry) * qty
        elif side in ("short", "sell"):
            pnl = (entry - exit_price) * qty
        else:
            pnl = (exit_price - entry) * qty  # Assume long
        return pnl, pnl >= 0

    return 0.0, False


def compute_pnl_stats(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute P&L statistics from trades."""
    if not trades:
        return {
            "total_pnl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "trade_count": 0,
        }

    total_pnl = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0
    win_pnls: List[float] = []
    loss_pnls: List[float] = []

    for trade in trades:
        pnl, is_win = calculate_trade_pnl(trade)
        total_pnl += pnl
        if is_win:
            wins += 1
            gross_profit += pnl
            win_pnls.append(pnl)
        else:
            losses += 1
            gross_loss += abs(pnl)
            loss_pnls.append(pnl)

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    return {
        "total_pnl": total_pnl,
        "gross_profit": gross_profit,
        "gross_loss": -gross_loss,  # Keep as negative for display
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": sum(win_pnls) / len(win_pnls) if win_pnls else 0.0,
        "avg_loss": sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0,
        "largest_win": max(win_pnls) if win_pnls else 0.0,
        "largest_loss": min(loss_pnls) if loss_pnls else 0.0,
        "trade_count": total_trades,
    }


def compute_pnl_by_strategy(trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group and compute P&L by strategy."""
    by_strategy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for trade in trades:
        strategy = (
            trade.get("strategy")
            or trade.get("signal_source")
            or "Unknown"
        )
        # Normalize strategy names
        strategy_lower = strategy.lower()
        if "rsi" in strategy_lower:
            strategy = "RSI-2"
        elif "TURTLE_SOUP" in strategy_lower:
            strategy = "TURTLE_SOUP"
        else:
            strategy = strategy or "Unknown"
        by_strategy[strategy].append(trade)

    result = {}
    for strat, strat_trades in by_strategy.items():
        result[strat] = compute_pnl_stats(strat_trades)

    return result


def format_currency(value: float, width: int = 12) -> str:
    """Format currency with sign."""
    if value >= 0:
        return f"+${value:,.2f}".rjust(width)
    return f"-${abs(value):,.2f}".rjust(width)


def print_summary_section(title: str, stats: Dict[str, Any]):
    """Print a P&L summary section."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    print()
    print(f"  {'Total P&L:':<25} {format_currency(stats['total_pnl'])}")
    print(f"  {'Gross Profit:':<25} {format_currency(stats['gross_profit'])}")
    print(f"  {'Gross Loss:':<25} {format_currency(stats['gross_loss'])}")
    print()
    print(f"  {'Wins:':<25} {stats['wins']:>12}")
    print(f"  {'Losses:':<25} {stats['losses']:>12}")
    print(f"  {'Win Rate:':<25} {stats['win_rate']:>11.1f}%")
    print(f"  {'Profit Factor:':<25} {stats['profit_factor']:>12.2f}")
    print()
    print(f"  {'Average Win:':<25} {format_currency(stats['avg_win'])}")
    print(f"  {'Average Loss:':<25} {format_currency(stats['avg_loss'])}")
    print(f"  {'Largest Win:':<25} {format_currency(stats['largest_win'])}")
    print(f"  {'Largest Loss:':<25} {format_currency(stats['largest_loss'])}")


def print_strategy_breakdown(by_strategy: Dict[str, Dict[str, Any]]):
    """Print P&L breakdown by strategy."""
    if not by_strategy:
        print("\n[INFO] No strategy breakdown available.")
        return

    print(f"\n{'='*80}")
    print(" P&L by Strategy")
    print(f"{'='*80}")
    print()
    print(f"  {'Strategy':<15} {'Trades':>8} {'Win Rate':>10} {'P&L':>14} {'Profit Factor':>14}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*14} {'-'*14}")

    total_pnl = 0.0
    total_trades = 0

    for strat, stats in sorted(by_strategy.items()):
        pf_str = f"{stats['profit_factor']:.2f}" if stats["profit_factor"] != float("inf") else "inf"
        pnl_str = format_currency(stats["total_pnl"]).strip()
        print(
            f"  {strat:<15} {stats['trade_count']:>8} "
            f"{stats['win_rate']:>9.1f}% {pnl_str:>14} {pf_str:>14}"
        )
        total_pnl += stats["total_pnl"]
        total_trades += stats["trade_count"]

    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*14} {'-'*14}")
    print(f"  {'TOTAL':<15} {total_trades:>8} {'':<10} {format_currency(total_pnl).strip():>14}")


def print_period_summary(trades: List[Dict[str, Any]]):
    """Print P&L summary by time period."""
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=now.weekday())
    month_start = today_start.replace(day=1)
    year_start = today_start.replace(month=1, day=1)

    periods = [
        ("Today", today_start),
        ("This Week", week_start),
        ("This Month", month_start),
        ("YTD", year_start),
        ("All Time", None),
    ]

    print(f"\n{'='*80}")
    print(" P&L by Period")
    print(f"{'='*80}")
    print()
    print(f"  {'Period':<15} {'Trades':>8} {'Wins':>8} {'Losses':>8} {'Win %':>8} {'P&L':>16}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*16}")

    for period_name, cutoff in periods:
        if cutoff:
            period_trades = [
                t for t in trades
                if parse_timestamp(
                    t.get("timestamp") or t.get("ts") or t.get("closed_at")
                ) and parse_timestamp(
                    t.get("timestamp") or t.get("ts") or t.get("closed_at")
                ) >= cutoff
            ]
        else:
            period_trades = trades

        stats = compute_pnl_stats(period_trades)
        pnl_str = format_currency(stats["total_pnl"]).strip()
        print(
            f"  {period_name:<15} {stats['trade_count']:>8} {stats['wins']:>8} "
            f"{stats['losses']:>8} {stats['win_rate']:>7.1f}% {pnl_str:>16}"
        )


def show_pnl(period: Optional[str] = None, strategy: Optional[str] = None):
    """Main P&L display function."""
    print(f"\n[INFO] Loading trade data...")
    trades = load_trades()

    if not trades:
        print("\n[INFO] No trade data found.")
        print("       Checked: state/trades.json, state/orders.json, logs/events.jsonl")
        return

    print(f"[INFO] Loaded {len(trades)} trade records")

    # Apply filters
    if strategy:
        trades = filter_trades_by_strategy(trades, strategy)
        print(f"[INFO] Filtered to {len(trades)} trades for strategy: {strategy}")

    filtered_trades = filter_trades_by_period(trades, period)

    if not filtered_trades:
        print(f"\n[INFO] No trades found for period: {period or 'all'}")
        return

    # Compute and display
    print(f"\n{'#'*80}")
    print(f"#  KOBE TRADING SYSTEM - P&L SUMMARY")
    print(f"#  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if period:
        print(f"#  Period: {period.upper()}")
    if strategy:
        print(f"#  Strategy: {strategy}")
    print(f"{'#'*80}")

    # Overall stats for filtered period
    stats = compute_pnl_stats(filtered_trades)
    period_title = f"{period.upper()} P&L Summary" if period else "Overall P&L Summary"
    print_summary_section(period_title, stats)

    # Period breakdown (only if not already filtered to specific period)
    if not period:
        print_period_summary(trades)

    # Strategy breakdown
    if not strategy:
        by_strategy = compute_pnl_by_strategy(filtered_trades)
        print_strategy_breakdown(by_strategy)

    print()


def main():
    ap = argparse.ArgumentParser(description="Kobe Trading System - P&L Summary")
    ap.add_argument(
        "--dotenv",
        type=str,
        default=DEFAULT_DOTENV,
        help="Path to .env file",
    )
    ap.add_argument(
        "--period",
        type=str,
        choices=["today", "week", "month", "ytd", "all"],
        default=None,
        help="Time period filter (today, week, month, ytd, all)",
    )
    ap.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Filter by strategy name (e.g., ibs_rsi, TURTLE_SOUP)",
    )
    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
        print(f"[INFO] Loaded {len(loaded)} env vars from {dotenv}")

    show_pnl(period=args.period, strategy=args.strategy)


if __name__ == "__main__":
    main()



