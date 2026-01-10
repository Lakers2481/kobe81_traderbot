#!/usr/bin/env python3
"""
Live vs Backtest Reconciliation Script

Compares live paper trading results against walk-forward backtest for the same period.
Part of The Gauntlet forward testing protocol.

Usage:
    python scripts/live_vs_backtest_reconcile.py --live-start 2026-01-07
    python scripts/live_vs_backtest_reconcile.py --live-start 2026-01-07 --backtest-file wf_outputs/wf_summary.csv
    python scripts/live_vs_backtest_reconcile.py --report-only

Author: Kobe Trading System
Date: 2026-01-07
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.structured_log import jlog

logger = logging.getLogger(__name__)

# Paths
LIVE_TRADES_LOG = ROOT / "logs" / "paper_trades.jsonl"
BACKTEST_SUMMARY = ROOT / "wf_outputs" / "wf_summary.csv"
RECONCILE_OUTPUT = ROOT / "reports" / "live_vs_backtest"
STATE_DIR = ROOT / "state"


@dataclass
class PerformanceMetrics:
    """Performance metrics for comparison."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_slippage_bps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "avg_pnl_per_trade": round(self.avg_pnl_per_trade, 2),
            "profit_factor": round(self.profit_factor, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown": round(self.max_drawdown, 4),
            "avg_slippage_bps": round(self.avg_slippage_bps, 2),
        }


@dataclass
class ReconciliationResult:
    """Result of live vs backtest comparison."""
    live_metrics: PerformanceMetrics
    backtest_metrics: PerformanceMetrics
    divergences: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    verdict: str = "UNKNOWN"
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "verdict": self.verdict,
            "live_metrics": self.live_metrics.to_dict(),
            "backtest_metrics": self.backtest_metrics.to_dict(),
            "divergences": {k: round(v, 4) for k, v in self.divergences.items()},
            "alerts": self.alerts,
        }


class LiveVsBacktestReconciler:
    """
    Reconciles live paper trading performance against backtest expectations.

    Compares key metrics and flags significant divergences that may indicate:
    - Execution issues (slippage, partial fills)
    - Data quality problems
    - Edge decay
    - Overfitting in backtest
    """

    # Acceptable divergence thresholds
    THRESHOLDS = {
        "win_rate": 0.10,        # +/- 10%
        "sharpe_ratio": 0.5,     # +/- 0.5
        "avg_pnl": 0.25,         # +/- 25%
        "profit_factor": 0.3,    # +/- 0.3
        "max_drawdown": 1.0,     # Can be up to 2x backtest
    }

    def __init__(
        self,
        live_trades_path: Path = LIVE_TRADES_LOG,
        backtest_path: Path = BACKTEST_SUMMARY,
    ):
        self.live_trades_path = live_trades_path
        self.backtest_path = backtest_path

    def load_live_trades(self, start_date: Optional[datetime] = None) -> pd.DataFrame:
        """Load live paper trades from JSONL log."""
        if not self.live_trades_path.exists():
            logger.warning(f"Live trades log not found: {self.live_trades_path}")
            return pd.DataFrame()

        trades = []
        with open(self.live_trades_path, 'r') as f:
            for line in f:
                try:
                    trade = json.loads(line.strip())
                    trades.append(trade)
                except json.JSONDecodeError:
                    continue

        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)

        # Parse timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if start_date:
                df = df[df['timestamp'] >= start_date]

        return df

    def load_backtest_summary(self) -> Dict[str, Any]:
        """Load backtest summary metrics."""
        if not self.backtest_path.exists():
            logger.warning(f"Backtest summary not found: {self.backtest_path}")
            return {}

        try:
            df = pd.read_csv(self.backtest_path)
            # Get the most recent or aggregate row
            if len(df) > 0:
                # Assuming last row is aggregate or use mean
                row = df.iloc[-1] if len(df) == 1 else df.mean(numeric_only=True)
                return row.to_dict()
        except Exception as e:
            logger.error(f"Failed to load backtest summary: {e}")

        return {}

    def calculate_live_metrics(self, trades_df: pd.DataFrame) -> PerformanceMetrics:
        """Calculate performance metrics from live trades."""
        metrics = PerformanceMetrics()

        if trades_df.empty:
            return metrics

        metrics.total_trades = len(trades_df)

        # Calculate wins/losses
        if 'pnl' in trades_df.columns:
            metrics.wins = (trades_df['pnl'] > 0).sum()
            metrics.losses = (trades_df['pnl'] <= 0).sum()
            metrics.win_rate = metrics.wins / metrics.total_trades if metrics.total_trades > 0 else 0
            metrics.total_pnl = trades_df['pnl'].sum()
            metrics.avg_pnl_per_trade = trades_df['pnl'].mean()

            # Profit factor
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Sharpe (annualized, assuming daily returns)
            if len(trades_df) > 1:
                returns = trades_df['pnl'].values
                if returns.std() > 0:
                    metrics.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

            # Max drawdown
            cumulative = trades_df['pnl'].cumsum()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max.replace(0, 1)
            metrics.max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0

        # Slippage
        if 'slippage_bps' in trades_df.columns:
            metrics.avg_slippage_bps = trades_df['slippage_bps'].mean()
        elif 'expected_price' in trades_df.columns and 'fill_price' in trades_df.columns:
            slippage = abs(trades_df['fill_price'] - trades_df['expected_price']) / trades_df['expected_price'] * 10000
            metrics.avg_slippage_bps = slippage.mean()

        return metrics

    def calculate_backtest_metrics(self, summary: Dict[str, Any]) -> PerformanceMetrics:
        """Convert backtest summary to PerformanceMetrics."""
        metrics = PerformanceMetrics()

        # Map common column names
        metrics.total_trades = int(summary.get('total_trades', summary.get('n_trades', 0)))
        metrics.wins = int(summary.get('wins', summary.get('n_wins', 0)))
        metrics.losses = int(summary.get('losses', summary.get('n_losses', 0)))
        metrics.win_rate = float(summary.get('win_rate', summary.get('wr', 0)))
        metrics.total_pnl = float(summary.get('total_pnl', summary.get('pnl', 0)))
        metrics.avg_pnl_per_trade = float(summary.get('avg_pnl', summary.get('avg_pnl_per_trade', 0)))
        metrics.profit_factor = float(summary.get('profit_factor', summary.get('pf', 1)))
        metrics.sharpe_ratio = float(summary.get('sharpe', summary.get('sharpe_ratio', 0)))
        metrics.max_drawdown = float(summary.get('max_drawdown', summary.get('max_dd', 0)))

        return metrics

    def compare_metrics(
        self,
        live: PerformanceMetrics,
        backtest: PerformanceMetrics
    ) -> ReconciliationResult:
        """Compare live and backtest metrics, identify divergences."""
        result = ReconciliationResult(
            live_metrics=live,
            backtest_metrics=backtest,
            generated_at=datetime.now().isoformat()
        )

        # Calculate divergences
        if backtest.win_rate > 0:
            wr_div = abs(live.win_rate - backtest.win_rate) / backtest.win_rate
            result.divergences["win_rate"] = wr_div
            if wr_div > self.THRESHOLDS["win_rate"]:
                result.alerts.append(
                    f"WIN RATE DIVERGENCE: Live {live.win_rate:.1%} vs Backtest {backtest.win_rate:.1%} "
                    f"(delta: {wr_div:.1%})"
                )

        if backtest.sharpe_ratio != 0:
            sharpe_div = abs(live.sharpe_ratio - backtest.sharpe_ratio)
            result.divergences["sharpe_ratio"] = sharpe_div
            if sharpe_div > self.THRESHOLDS["sharpe_ratio"]:
                result.alerts.append(
                    f"SHARPE DIVERGENCE: Live {live.sharpe_ratio:.2f} vs Backtest {backtest.sharpe_ratio:.2f} "
                    f"(delta: {sharpe_div:.2f})"
                )

        if backtest.avg_pnl_per_trade != 0:
            pnl_div = abs(live.avg_pnl_per_trade - backtest.avg_pnl_per_trade) / abs(backtest.avg_pnl_per_trade)
            result.divergences["avg_pnl"] = pnl_div
            if pnl_div > self.THRESHOLDS["avg_pnl"]:
                result.alerts.append(
                    f"AVG P&L DIVERGENCE: Live ${live.avg_pnl_per_trade:.2f} vs Backtest ${backtest.avg_pnl_per_trade:.2f} "
                    f"(delta: {pnl_div:.1%})"
                )

        if backtest.profit_factor > 0:
            pf_div = abs(live.profit_factor - backtest.profit_factor)
            result.divergences["profit_factor"] = pf_div
            if pf_div > self.THRESHOLDS["profit_factor"]:
                result.alerts.append(
                    f"PROFIT FACTOR DIVERGENCE: Live {live.profit_factor:.2f} vs Backtest {backtest.profit_factor:.2f} "
                    f"(delta: {pf_div:.2f})"
                )

        if backtest.max_drawdown > 0:
            dd_ratio = live.max_drawdown / backtest.max_drawdown
            result.divergences["max_drawdown_ratio"] = dd_ratio
            if dd_ratio > (1 + self.THRESHOLDS["max_drawdown"]):
                result.alerts.append(
                    f"DRAWDOWN EXCEEDED: Live {live.max_drawdown:.1%} vs Backtest {backtest.max_drawdown:.1%} "
                    f"(ratio: {dd_ratio:.1f}x)"
                )

        # Check slippage
        if live.avg_slippage_bps > 25:
            result.alerts.append(
                f"HIGH SLIPPAGE: {live.avg_slippage_bps:.1f} BPS (threshold: 25 BPS)"
            )

        # Determine verdict
        if len(result.alerts) == 0:
            result.verdict = "PASS - All metrics within acceptable range"
        elif len(result.alerts) <= 2:
            result.verdict = "WARNING - Minor divergences detected, continue monitoring"
        else:
            result.verdict = "FAIL - Significant divergences, consider pausing Gauntlet"

        return result

    def run_reconciliation(
        self,
        start_date: Optional[datetime] = None
    ) -> ReconciliationResult:
        """Run full reconciliation analysis."""
        logger.info("Starting live vs backtest reconciliation...")

        # Load data
        live_trades = self.load_live_trades(start_date)
        backtest_summary = self.load_backtest_summary()

        # Calculate metrics
        live_metrics = self.calculate_live_metrics(live_trades)
        backtest_metrics = self.calculate_backtest_metrics(backtest_summary)

        # Compare
        result = self.compare_metrics(live_metrics, backtest_metrics)

        logger.info(f"Reconciliation complete: {result.verdict}")
        return result

    def save_report(self, result: ReconciliationResult) -> Path:
        """Save reconciliation report to file."""
        RECONCILE_OUTPUT.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = RECONCILE_OUTPUT / f"reconcile_{timestamp}.json"

        with open(report_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Report saved to: {report_path}")
        return report_path

    def print_report(self, result: ReconciliationResult) -> None:
        """Print formatted report to console."""
        print("\n" + "=" * 70)
        print("LIVE VS BACKTEST RECONCILIATION REPORT")
        print("=" * 70)
        print(f"Generated: {result.generated_at}")
        print()

        print("-" * 70)
        print(f"{'METRIC':<25} {'LIVE':>15} {'BACKTEST':>15} {'DIVERGENCE':>12}")
        print("-" * 70)

        live = result.live_metrics
        bt = result.backtest_metrics

        print(f"{'Total Trades':<25} {live.total_trades:>15} {bt.total_trades:>15}")
        print(f"{'Win Rate':<25} {live.win_rate:>14.1%} {bt.win_rate:>14.1%} {result.divergences.get('win_rate', 0):>11.1%}")
        print(f"{'Sharpe Ratio':<25} {live.sharpe_ratio:>15.2f} {bt.sharpe_ratio:>15.2f} {result.divergences.get('sharpe_ratio', 0):>12.2f}")
        print(f"{'Profit Factor':<25} {live.profit_factor:>15.2f} {bt.profit_factor:>15.2f} {result.divergences.get('profit_factor', 0):>12.2f}")
        print(f"{'Avg P&L/Trade':<25} ${live.avg_pnl_per_trade:>14.2f} ${bt.avg_pnl_per_trade:>14.2f} {result.divergences.get('avg_pnl', 0):>11.1%}")
        print(f"{'Max Drawdown':<25} {live.max_drawdown:>14.1%} {bt.max_drawdown:>14.1%}")
        print(f"{'Avg Slippage (BPS)':<25} {live.avg_slippage_bps:>15.1f} {'N/A':>15}")

        print()
        print("-" * 70)
        print("ALERTS")
        print("-" * 70)

        if result.alerts:
            for alert in result.alerts:
                print(f"  [!] {alert}")
        else:
            print("  No alerts - all metrics within acceptable range")

        print()
        print("=" * 70)
        print(f"VERDICT: {result.verdict}")
        print("=" * 70)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare live paper trading vs backtest performance"
    )
    parser.add_argument(
        "--live-start",
        type=str,
        help="Start date for live trades (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--backtest-file",
        type=str,
        default=str(BACKTEST_SUMMARY),
        help="Path to backtest summary CSV",
    )
    parser.add_argument(
        "--live-file",
        type=str,
        default=str(LIVE_TRADES_LOG),
        help="Path to live trades JSONL",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Show most recent report without running new reconciliation",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save report to file",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if args.report_only:
        # Find most recent report
        if RECONCILE_OUTPUT.exists():
            reports = sorted(RECONCILE_OUTPUT.glob("reconcile_*.json"))
            if reports:
                with open(reports[-1], 'r') as f:
                    data = json.load(f)
                print(f"Loading report from: {reports[-1]}")
                print(json.dumps(data, indent=2))
                return
        print("No previous reports found")
        return

    # Run reconciliation
    reconciler = LiveVsBacktestReconciler(
        live_trades_path=Path(args.live_file),
        backtest_path=Path(args.backtest_file),
    )

    start_date = None
    if args.live_start:
        start_date = datetime.strptime(args.live_start, "%Y-%m-%d")

    result = reconciler.run_reconciliation(start_date)

    # Print report
    reconciler.print_report(result)

    # Save if requested
    if args.save:
        reconciler.save_report(result)

    # Log structured event
    try:
        jlog(
            "reconciliation_complete",
            verdict=result.verdict,
            alerts=len(result.alerts),
            live_trades=result.live_metrics.total_trades,
        )
    except Exception:
        pass  # jlog may not be available in all contexts

    # Exit code based on verdict
    if "FAIL" in result.verdict:
        sys.exit(1)
    elif "WARNING" in result.verdict:
        sys.exit(0)  # Warning is acceptable
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
