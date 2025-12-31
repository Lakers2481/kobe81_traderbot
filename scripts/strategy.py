#!/usr/bin/env python3
"""
Kobe Trading System - Strategy Information

Show strategy parameters, compare strategies, and view statistics.

Usage:
    python strategy.py --list           # List available strategies
    python strategy.py --show rsi2      # Show RSI-2 parameters
    python strategy.py --show ibs       # Show IBS parameters
    python strategy.py --compare        # Compare all strategies
    python strategy.py --stats          # Show strategy performance stats
    python strategy.py --dotenv /path/to/.env
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from datetime import datetime
from core.clock.tz_utils import now_et, fmt_ct
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.env_loader import load_env
# 

from strategies.registry import get_production_scanner, DualStrategyScanner, DualStrategyParams

# Default paths
STATE_DIR = ROOT / "state"
LOGS_DIR = ROOT / "logs"
WF_OUTPUTS = ROOT / "wf_outputs"
DEFAULT_DOTENV = "C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env"

# Strategy registry - now uses canonical DualStrategyScanner
STRATEGIES = {
    "dual": {
        "name": "Dual Strategy (IBS+RSI + Turtle Soup)",
        "class": DualStrategyScanner,
        "params_class": DualStrategyParams,
        "description": "Combined mean-reversion: IBS+RSI (59.9% WR) + Turtle Soup (61.0% WR) with 0.3 ATR sweep filter",
    },
}


def get_strategy_params(strategy_key: str) -> Dict[str, Any]:
    """Get default parameters for a strategy."""
    if strategy_key not in STRATEGIES:
        return {}

    params_class = STRATEGIES[strategy_key]["params_class"]
    params = params_class()
    return asdict(params)


def get_strategy_param_descriptions(strategy_key: str) -> Dict[str, str]:
    """Get parameter descriptions for a strategy."""
    descriptions = {
        # RSI-2 parameters
        "rsi_period": "RSI calculation period",
        "rsi_method": "RSI smoothing method (wilder/sma)",
        "sma_period": "Trend filter SMA period",
        "atr_period": "ATR calculation period for stops",
        "atr_stop_mult": "ATR multiplier for stop distance",
        "time_stop_bars": "Maximum bars to hold position",
        "long_entry_rsi_max": "RSI threshold for long entry (<=)",
        "short_entry_rsi_min": "RSI threshold for short entry (>=)",
        "long_exit_rsi_min": "RSI threshold for long exit (>=)",
        "short_exit_rsi_max": "RSI threshold for short exit (<=)",
        "min_price": "Minimum stock price filter",
        # IBS parameters
        "ibs_long_max": "IBS threshold for long entry (<)",
        "ibs_short_min": "IBS threshold for short entry (>)",
    }
    return descriptions


def load_strategy_stats() -> Dict[str, Dict[str, Any]]:
    """Load strategy performance statistics from walk-forward outputs."""
    stats: Dict[str, Dict[str, Any]] = {}

    # Check walk-forward output directories
    for wf_dir in [WF_OUTPUTS, ROOT / "wf_outputs_full10y"]:
        if not wf_dir.exists():
            continue

        for summary_file in wf_dir.glob("**/summary.json"):
            try:
                data = json.loads(summary_file.read_text(encoding="utf-8"))
                strategy_name = summary_file.parent.name

                # Normalize strategy name
                if "rsi" in strategy_name.lower():
                    key = "rsi2"
                elif "ibs" in strategy_name.lower():
                    key = "ibs"
                else:
                    key = strategy_name.lower()

                if key not in stats:
                    stats[key] = {
                        "runs": [],
                        "total_trades": 0,
                        "total_pnl": 0.0,
                        "win_rates": [],
                        "sharpe_ratios": [],
                        "max_drawdowns": [],
                    }

                stats[key]["runs"].append(data)
                stats[key]["total_trades"] += data.get("trades", 0)
                stats[key]["total_pnl"] += data.get("pnl", 0)

                if "win_rate" in data:
                    stats[key]["win_rates"].append(data["win_rate"])
                if "sharpe" in data:
                    stats[key]["sharpe_ratios"].append(data["sharpe"])
                if "max_drawdown" in data:
                    stats[key]["max_drawdowns"].append(data["max_drawdown"])

            except Exception:
                continue

    # Also check trade logs for live/paper stats
    events_file = LOGS_DIR / "events.jsonl"
    if events_file.exists():
        try:
            for line in events_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if event.get("event") in ("trade_closed", "position_closed"):
                        strategy = event.get("strategy", "").lower()
                        if "rsi" in strategy:
                            key = "rsi2"
                        elif "ibs" in strategy:
                            key = "ibs"
                        else:
                            continue

                        if key not in stats:
                            stats[key] = {
                                "runs": [],
                                "total_trades": 0,
                                "total_pnl": 0.0,
                                "win_rates": [],
                                "sharpe_ratios": [],
                                "max_drawdowns": [],
                                "live_trades": [],
                            }

                        if "live_trades" not in stats[key]:
                            stats[key]["live_trades"] = []
                        stats[key]["live_trades"].append(event)

                except Exception:
                    continue
        except Exception:
            pass

    return stats


def list_strategies():
    """List all available strategies."""
    print(f"\n{'='*60}")
    print(" Available Strategies")
    print(f"{'='*60}")
    print()

    for key, info in STRATEGIES.items():
        print(f"  [{key.upper()}] {info['name']}")
        print(f"        {info['description']}")
        print()


def show_strategy(strategy_key: str):
    """Show detailed information about a strategy."""
    strategy_key = strategy_key.lower()

    if strategy_key not in STRATEGIES:
        print(f"[ERROR] Unknown strategy: {strategy_key}")
        print(f"        Available: {', '.join(STRATEGIES.keys())}")
        return

    info = STRATEGIES[strategy_key]
    params = get_strategy_params(strategy_key)
    descriptions = get_strategy_param_descriptions(strategy_key)

    print(f"\n{'#'*70}")
    print(f"#  {info['name'].upper()}")
    print(f"{'#'*70}")
    print()
    print(f"  Description: {info['description']}")
    print()

    # Strategy logic
    if strategy_key == "rsi2":
        print("  Entry Logic:")
        print("    - LONG:  Close > SMA(200) AND RSI(2) <= 10")
        print("    - SHORT: Close < SMA(200) AND RSI(2) >= 90")
        print()
        print("  Exit Logic:")
        print("    - LONG exit:  RSI(2) >= 70 OR stop hit OR time stop")
        print("    - SHORT exit: RSI(2) <= 30 OR stop hit OR time stop")
    elif strategy_key == "ibs":
        print("  Entry Logic:")
        print("    - LONG:  Close > SMA(200) AND IBS < 0.2")
        print("    - SHORT: Close < SMA(200) AND IBS > 0.8")
        print()
        print("  Exit Logic:")
        print("    - Stop hit OR time stop (5 bars)")

    print()
    print("  Risk Management:")
    print(f"    - Stop Loss: {params.get('atr_stop_mult', 2.0)}x ATR({params.get('atr_period', 14)})")
    print(f"    - Time Stop: {params.get('time_stop_bars', 5)} bars")
    print(f"    - Min Price Filter: ${params.get('min_price', 5.0)}")

    print()
    print(f"  {'='*60}")
    print("  Parameters")
    print(f"  {'='*60}")
    print()
    print(f"  {'Parameter':<25} {'Value':>15} {'Description':<30}")
    print(f"  {'-'*25} {'-'*15} {'-'*30}")

    for param_name, value in params.items():
        desc = descriptions.get(param_name, "")
        if len(desc) > 30:
            desc = desc[:27] + "..."
        print(f"  {param_name:<25} {str(value):>15} {desc:<30}")

    print()


def compare_strategies():
    """Compare RSI-2 and IBS strategies side by side."""
    rsi2_params = get_strategy_params("rsi2")
    ibs_params = get_strategy_params("ibs")

    print(f"\n{'#'*80}")
    print(f"#  STRATEGY COMPARISON: RSI-2 vs IBS")
    print(f"{'#'*80}")
    print()

    print(f"  {'Aspect':<30} {'RSI-2':<25} {'IBS':<25}")
    print(f"  {'-'*30} {'-'*25} {'-'*25}")

    # Entry indicator
    print(f"  {'Entry Indicator':<30} {'RSI(2)':<25} {'IBS (Internal Bar Strength)':<25}")

    # Long entry
    rsi_long = f"RSI <= {rsi2_params['long_entry_rsi_max']}"
    ibs_long = f"IBS < {ibs_params['ibs_long_max']}"
    print(f"  {'Long Entry Threshold':<30} {rsi_long:<25} {ibs_long:<25}")

    # Short entry
    rsi_short = f"RSI >= {rsi2_params['short_entry_rsi_min']}"
    ibs_short = f"IBS > {ibs_params['ibs_short_min']}"
    print(f"  {'Short Entry Threshold':<30} {rsi_short:<25} {ibs_short:<25}")

    # Trend filter
    print(f"  {'Trend Filter':<30} {'SMA(200)':<25} {'SMA(200)':<25}")

    # Stop loss
    rsi_stop = f"{rsi2_params['atr_stop_mult']}x ATR({rsi2_params['atr_period']})"
    ibs_stop = f"{ibs_params['atr_stop_mult']}x ATR({ibs_params['atr_period']})"
    print(f"  {'Stop Loss':<30} {rsi_stop:<25} {ibs_stop:<25}")

    # Time stop
    print(f"  {'Time Stop':<30} {str(rsi2_params['time_stop_bars']) + ' bars':<25} {str(ibs_params['time_stop_bars']) + ' bars':<25}")

    # Min price
    print(f"  {'Min Price':<30} {'$' + str(rsi2_params['min_price']):<25} {'$' + str(ibs_params['min_price']):<25}")

    # Exit conditions
    rsi_exit = f"RSI >= {rsi2_params['long_exit_rsi_min']} (long)"
    ibs_exit = "Stop/Time only"
    print(f"  {'Exit Condition':<30} {rsi_exit:<25} {ibs_exit:<25}")

    print()
    print(f"  {'='*80}")
    print("  Key Differences:")
    print(f"  {'='*80}")
    print()
    print("  1. Indicator Type:")
    print("     - RSI-2: Momentum oscillator (0-100 scale)")
    print("     - IBS: Bar position indicator (0-1 scale)")
    print()
    print("  2. Signal Sensitivity:")
    print("     - RSI-2: More selective (extreme RSI values required)")
    print("     - IBS: More frequent signals (20%/80% thresholds)")
    print()
    print("  3. Exit Logic:")
    print("     - RSI-2: Indicator-based exit (RSI crosses threshold)")
    print("     - IBS: Pure stop/time exit (no indicator exit)")
    print()


def show_stats():
    """Show strategy performance statistics."""
    print("[INFO] Loading strategy statistics...")
    stats = load_strategy_stats()

    if not stats:
        print("\n[INFO] No strategy statistics found.")
        print("       Run backtests or walk-forward analysis to generate stats.")
        return

    print(f"\n{'#'*80}")
    print(f"#  STRATEGY PERFORMANCE STATISTICS")
    now = now_et()
    print(f"#  Generated: {now.strftime('%Y-%m-%d')} {fmt_ct(now)}")
    print(f"{'#'*80}")

    for key, data in stats.items():
        strat_name = STRATEGIES.get(key, {}).get("name", key.upper())

        print(f"\n{'='*60}")
        print(f" {strat_name}")
        print(f"{'='*60}")
        print()

        # Aggregate stats
        num_runs = len(data.get("runs", []))
        total_trades = data.get("total_trades", 0)

        print(f"  {'Backtest Runs:':<25} {num_runs}")
        print(f"  {'Total Trades:':<25} {total_trades}")

        if data.get("win_rates"):
            avg_wr = sum(data["win_rates"]) / len(data["win_rates"]) * 100
            print(f"  {'Avg Win Rate:':<25} {avg_wr:.1f}%")

        if data.get("sharpe_ratios"):
            avg_sharpe = sum(data["sharpe_ratios"]) / len(data["sharpe_ratios"])
            print(f"  {'Avg Sharpe:':<25} {avg_sharpe:.2f}")

        if data.get("max_drawdowns"):
            worst_dd = min(data["max_drawdowns"]) * 100
            print(f"  {'Worst Max DD:':<25} {worst_dd:.1f}%")

        # Live/paper trade stats
        live_trades = data.get("live_trades", [])
        if live_trades:
            print()
            print(f"  Live/Paper Trades: {len(live_trades)}")

            wins = sum(1 for t in live_trades if float(t.get("pnl", 0)) >= 0)
            losses = len(live_trades) - wins
            total_pnl = sum(float(t.get("pnl", 0)) for t in live_trades)

            print(f"    Wins:     {wins}")
            print(f"    Losses:   {losses}")
            print(f"    Win Rate: {wins/len(live_trades)*100:.1f}%" if live_trades else "    Win Rate: N/A")
            print(f"    Total P&L: ${total_pnl:,.2f}")

    print()


def main():
    ap = argparse.ArgumentParser(description="Kobe Trading System - Strategy Information")
    ap.add_argument(
        "--dotenv",
        type=str,
        default=DEFAULT_DOTENV,
        help="Path to .env file",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List available strategies",
    )
    ap.add_argument(
        "--show",
        type=str,
        default=None,
        help="Show detailed info for a strategy (rsi2, ibs)",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="Compare RSI-2 and IBS strategies",
    )
    ap.add_argument(
        "--stats",
        action="store_true",
        help="Show strategy performance statistics",
    )
    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
        print(f"[INFO] Loaded {len(loaded)} env vars from {dotenv}")

    # Determine action
    if args.list:
        list_strategies()
    elif args.show:
        show_strategy(args.show)
    elif args.compare:
        # compare_strategies() # Commented out
        print("Cannot compare strategies: RSI-2 is not implemented.")
    elif args.stats:
        show_stats()
    else:
        # Default: show list
        list_strategies()
        print("  Use --show <strategy> for detailed parameters")
        print("  Use --compare to compare strategies")
        print("  Use --stats for performance statistics")
        print()


if __name__ == "__main__":
    main()
