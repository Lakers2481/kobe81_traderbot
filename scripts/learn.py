#!/usr/bin/env python3
"""
Show what Kobe learned from recent trades.
Analyzes patterns, mistakes, and improvements.
Usage: python scripts/learn.py [--trades N|--period DAYS|--strategy NAME]
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


def load_trade_history(limit: int = 100, days: int = 30) -> List[Dict]:
    """Load recent trade history."""
    trades = []

    # Load from order history
    order_file = Path("state/order_history.json")
    if order_file.exists():
        with open(order_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                trades.extend(data)
            elif isinstance(data, dict):
                trades.extend(data.values())

    # Load from trade logs
    trade_log = Path("logs/trades.jsonl")
    if trade_log.exists():
        with open(trade_log) as f:
            for line in f:
                try:
                    trades.append(json.loads(line.strip()))
                except (json.JSONDecodeError, ValueError):
                    pass

    # Filter by date
    cutoff = datetime.now() - timedelta(days=days)
    filtered = []
    for trade in trades:
        trade_time = trade.get("timestamp") or trade.get("created_at") or trade.get("time")
        if trade_time:
            try:
                if isinstance(trade_time, str):
                    dt = datetime.fromisoformat(trade_time.replace("Z", "+00:00"))
                    if dt.replace(tzinfo=None) >= cutoff:
                        filtered.append(trade)
            except ValueError:
                filtered.append(trade)  # Include if can't parse

    # Sort by time and limit
    filtered.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return filtered[:limit]


def analyze_win_loss_patterns(trades: List[Dict]) -> Dict:
    """Analyze win/loss patterns."""
    patterns = {
        "by_day_of_week": defaultdict(lambda: {"wins": 0, "losses": 0}),
        "by_hour": defaultdict(lambda: {"wins": 0, "losses": 0}),
        "by_holding_period": defaultdict(lambda: {"wins": 0, "losses": 0}),
        "by_symbol": defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0}),
    }

    for trade in trades:
        pnl = float(trade.get("pnl", trade.get("realized_pnl", 0)))
        symbol = trade.get("symbol", "UNKNOWN")
        is_win = pnl > 0

        # By day of week
        timestamp = trade.get("timestamp") or trade.get("created_at")
        if timestamp:
            try:
                dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                day = dt.strftime("%A")
                patterns["by_day_of_week"][day]["wins" if is_win else "losses"] += 1

                hour = dt.hour
                period = "Pre-Market" if hour < 9 else "Morning" if hour < 12 else "Afternoon" if hour < 16 else "After-Hours"
                patterns["by_hour"][period]["wins" if is_win else "losses"] += 1
            except ValueError:
                pass

        # By symbol
        patterns["by_symbol"][symbol]["wins" if is_win else "losses"] += 1
        patterns["by_symbol"][symbol]["pnl"] += pnl

        # By holding period
        hold_time = trade.get("holding_period", trade.get("hold_bars", "Unknown"))
        if isinstance(hold_time, (int, float)):
            if hold_time <= 1:
                period = "Same Day"
            elif hold_time <= 5:
                period = "1-5 Days"
            else:
                period = "5+ Days"
            patterns["by_holding_period"][period]["wins" if is_win else "losses"] += 1

    return patterns


def identify_mistakes(trades: List[Dict]) -> List[Dict]:
    """Identify trading mistakes and lessons."""
    mistakes = []

    for trade in trades:
        pnl = float(trade.get("pnl", trade.get("realized_pnl", 0)))
        symbol = trade.get("symbol", "")
        exit_reason = trade.get("exit_reason", trade.get("reason", ""))

        # Stop loss hit
        if "stop" in str(exit_reason).lower() and pnl < 0:
            mistakes.append({
                "type": "Stop Loss Hit",
                "symbol": symbol,
                "pnl": pnl,
                "lesson": "Consider tighter entry criteria or wider stops",
            })

        # Time stop (held too long)
        if "time" in str(exit_reason).lower():
            mistakes.append({
                "type": "Time Stop",
                "symbol": symbol,
                "pnl": pnl,
                "lesson": "Trade didn't move as expected - review entry timing",
            })

        # Large loss
        if pnl < -50:
            mistakes.append({
                "type": "Large Loss",
                "symbol": symbol,
                "pnl": pnl,
                "lesson": "Review position sizing and risk limits",
            })

    return mistakes


def calculate_improvements(trades: List[Dict]) -> Dict:
    """Calculate potential improvements."""
    improvements = {
        "entry_timing": {"current": 0, "potential": 0},
        "exit_timing": {"current": 0, "potential": 0},
        "position_sizing": {"current": 0, "potential": 0},
    }

    total_pnl = sum(float(t.get("pnl", t.get("realized_pnl", 0))) for t in trades)
    win_trades = [t for t in trades if float(t.get("pnl", t.get("realized_pnl", 0))) > 0]
    loss_trades = [t for t in trades if float(t.get("pnl", t.get("realized_pnl", 0))) < 0]

    if trades:
        win_rate = len(win_trades) / len(trades) * 100
        avg_win = sum(float(t.get("pnl", 0)) for t in win_trades) / len(win_trades) if win_trades else 0
        avg_loss = sum(float(t.get("pnl", 0)) for t in loss_trades) / len(loss_trades) if loss_trades else 0

        improvements["summary"] = {
            "total_trades": len(trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_pnl": total_pnl,
        }

        # Improvement suggestions
        if win_rate < 60:
            improvements["suggestions"] = [
                "Win rate below 60% - consider stricter entry filters",
                f"Current: {win_rate:.1f}%, Target: 65%+",
            ]
        elif avg_loss and abs(avg_loss) > avg_win * 1.5:
            improvements["suggestions"] = [
                "Average loss too large relative to wins",
                "Consider tighter stop losses or smaller position sizes",
            ]
        else:
            improvements["suggestions"] = [
                "Performance within acceptable range",
                "Continue current strategy parameters",
            ]

    return improvements


def show_learnings(trades: List[Dict], strategy: Optional[str] = None):
    """Display learning summary."""
    print("\n" + "=" * 50)
    print("     KOBE LEARNING SUMMARY")
    print("=" * 50)

    if strategy:
        trades = [t for t in trades if t.get("strategy", "").lower() == strategy.lower()]
        print(f"\nStrategy Filter: {strategy}")

    if not trades:
        print("\nNo trades found to analyze.")
        print("Complete some trades first, then run this again.")
        return

    print(f"\nAnalyzing {len(trades)} recent trades...\n")

    # Win/Loss Patterns
    patterns = analyze_win_loss_patterns(trades)

    print("--- Day of Week Performance ---")
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        data = patterns["by_day_of_week"].get(day, {"wins": 0, "losses": 0})
        total = data["wins"] + data["losses"]
        if total > 0:
            wr = data["wins"] / total * 100
            print(f"  {day:<12}: {data['wins']:>3}W / {data['losses']:>3}L  ({wr:.0f}% WR)")

    print("\n--- Time of Day Performance ---")
    for period in ["Pre-Market", "Morning", "Afternoon", "After-Hours"]:
        data = patterns["by_hour"].get(period, {"wins": 0, "losses": 0})
        total = data["wins"] + data["losses"]
        if total > 0:
            wr = data["wins"] / total * 100
            print(f"  {period:<12}: {data['wins']:>3}W / {data['losses']:>3}L  ({wr:.0f}% WR)")

    print("\n--- Top Symbols ---")
    symbol_data = sorted(patterns["by_symbol"].items(), key=lambda x: x[1]["pnl"], reverse=True)
    for symbol, data in symbol_data[:5]:
        total = data["wins"] + data["losses"]
        wr = data["wins"] / total * 100 if total > 0 else 0
        pnl_str = f"+${data['pnl']:.0f}" if data["pnl"] >= 0 else f"-${abs(data['pnl']):.0f}"
        print(f"  {symbol:<6}: {data['wins']:>2}W / {data['losses']:>2}L  {pnl_str:>8}  ({wr:.0f}% WR)")

    # Mistakes
    print("\n--- Recent Mistakes & Lessons ---")
    mistakes = identify_mistakes(trades)
    if mistakes:
        for i, m in enumerate(mistakes[:5], 1):
            print(f"  {i}. [{m['type']}] {m['symbol']}: ${m['pnl']:.0f}")
            print(f"     Lesson: {m['lesson']}")
    else:
        print("  No significant mistakes identified!")

    # Improvements
    print("\n--- Improvement Opportunities ---")
    improvements = calculate_improvements(trades)
    if "summary" in improvements:
        s = improvements["summary"]
        print(f"  Total Trades: {s['total_trades']}")
        print(f"  Win Rate: {s['win_rate']:.1f}%")
        print(f"  Avg Win: ${s['avg_win']:.2f}")
        print(f"  Avg Loss: ${s['avg_loss']:.2f}")
        print(f"  Total P&L: ${s['total_pnl']:.2f}")

    if "suggestions" in improvements:
        print("\n  Suggestions:")
        for suggestion in improvements["suggestions"]:
            print(f"    - {suggestion}")

    # Key Learnings
    print("\n--- Key Learnings ---")
    print("  1. Best performing days and times identified above")
    print("  2. Strongest and weakest symbols tracked")
    print("  3. Mistake patterns logged for improvement")
    print("  4. Continue tracking for more insights")

    print("\n" + "=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Show Kobe's trading learnings")
    parser.add_argument("--trades", type=int, default=100, help="Number of trades to analyze")
    parser.add_argument("--period", type=int, default=30, help="Days to look back")
    parser.add_argument("--strategy", type=str, help="Filter by strategy name")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")

    args = parser.parse_args()

    trades = load_trade_history(limit=args.trades, days=args.period)

    if args.json:
        patterns = analyze_win_loss_patterns(trades)
        mistakes = identify_mistakes(trades)
        improvements = calculate_improvements(trades)

        output = {
            "timestamp": datetime.now().isoformat(),
            "trades_analyzed": len(trades),
            "patterns": {k: dict(v) for k, v in patterns.items()},
            "mistakes": mistakes[:10],
            "improvements": improvements,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        show_learnings(trades, args.strategy)


if __name__ == "__main__":
    main()
