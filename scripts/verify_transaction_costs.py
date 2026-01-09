"""
Verify Transaction Costs vs Paper Trading (Jim Simons Standard)
================================================================

Compares backtest slippage assumptions against actual paper trading fills.

Key Metrics:
    - Actual slippage (bps): (fill_price - decision_price) / decision_price * 10000
    - Spread capture (bps): How much of bid-ask spread was captured
    - Fill rate: % of signals that resulted in fills
    - Adverse selection: Do we get worse fills than expected?

Backtest Assumption (backtest/engine.py line 29):
    - slippage_bps = 10.0 (0.10%)

Jim Simons Standard:
    - If real slippage > backtest assumption → Backtest is overly optimistic
    - If real slippage < backtest assumption → Backtest is conservative (good!)
    - Typical slippage for liquid stocks: 2-10 bps
    - If slippage > 15 bps → Execution quality issue

Usage:
    python scripts/verify_transaction_costs.py
    python scripts/verify_transaction_costs.py --lookback-days 30
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.structured_log import jlog


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FillAnalysis:
    """Analysis of a single fill."""
    timestamp: datetime
    symbol: str
    side: str
    qty: int
    decision_price: Optional[float]
    limit_price: float
    fill_price: float
    bid_at_execution: Optional[float]
    ask_at_execution: Optional[float]
    strategy: Optional[str]

    @property
    def slippage_bps(self) -> Optional[float]:
        """Calculate slippage in basis points vs decision price."""
        if self.decision_price is None or self.decision_price == 0:
            return None

        if self.side.upper() == "BUY":
            # For BUY: positive slippage = paid more than expected (bad)
            return ((self.fill_price - self.decision_price) / self.decision_price) * 10000
        else:  # SELL
            # For SELL: positive slippage = got less than expected (bad)
            return ((self.decision_price - self.fill_price) / self.decision_price) * 10000

    @property
    def limit_slippage_bps(self) -> Optional[float]:
        """Calculate slippage in basis points vs limit price."""
        if self.limit_price == 0:
            return None

        if self.side.upper() == "BUY":
            return ((self.fill_price - self.limit_price) / self.limit_price) * 10000
        else:
            return ((self.limit_price - self.fill_price) / self.limit_price) * 10000

    @property
    def spread_bps(self) -> Optional[float]:
        """Calculate bid-ask spread in basis points."""
        if self.bid_at_execution is None or self.ask_at_execution is None:
            return None
        if self.bid_at_execution == 0:
            return None

        return ((self.ask_at_execution - self.bid_at_execution) / self.bid_at_execution) * 10000

    @property
    def spread_capture_bps(self) -> Optional[float]:
        """How much of the spread was captured (positive = good fill)."""
        if self.bid_at_execution is None or self.ask_at_execution is None:
            return None

        mid_price = (self.bid_at_execution + self.ask_at_execution) / 2
        if mid_price == 0:
            return None

        if self.side.upper() == "BUY":
            # Buy below mid = good
            return ((mid_price - self.fill_price) / mid_price) * 10000
        else:
            # Sell above mid = good
            return ((self.fill_price - mid_price) / mid_price) * 10000


@dataclass
class TransactionCostReport:
    """Full transaction cost analysis report."""
    total_fills: int
    total_rejections: int
    fill_rate: float

    avg_slippage_bps: Optional[float]
    median_slippage_bps: Optional[float]
    p90_slippage_bps: Optional[float]
    p95_slippage_bps: Optional[float]

    avg_spread_bps: Optional[float]
    avg_spread_capture_bps: Optional[float]

    backtest_slippage_bps: float = 10.0

    fills_analyzed: int = 0
    fills_with_decision_price: int = 0
    fills_with_spread_data: int = 0

    @property
    def passed(self) -> bool:
        """Check if transaction costs are within acceptable limits."""
        if self.avg_slippage_bps is None:
            return False

        # PASS if real slippage <= backtest assumption
        # This means backtest is conservative (not overly optimistic)
        return self.avg_slippage_bps <= self.backtest_slippage_bps

    @property
    def quality_grade(self) -> str:
        """Grade execution quality."""
        if self.avg_slippage_bps is None:
            return "UNKNOWN"

        if self.avg_slippage_bps < 5:
            return "EXCELLENT"
        elif self.avg_slippage_bps < 10:
            return "GOOD"
        elif self.avg_slippage_bps < 15:
            return "ACCEPTABLE"
        else:
            return "POOR"

    @property
    def recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recs = []

        if self.avg_slippage_bps is None:
            recs.append("Need more fills with decision_price data to analyze slippage")
            return recs

        if self.avg_slippage_bps > self.backtest_slippage_bps:
            excess = self.avg_slippage_bps - self.backtest_slippage_bps
            recs.append(
                f"Real slippage ({self.avg_slippage_bps:.2f} bps) exceeds "
                f"backtest assumption ({self.backtest_slippage_bps:.2f} bps) by {excess:.2f} bps"
            )
            recs.append("Action: Increase backtest slippage_bps to match reality")
        else:
            recs.append(
                f"Backtest slippage ({self.backtest_slippage_bps:.2f} bps) is conservative "
                f"vs actual ({self.avg_slippage_bps:.2f} bps) - GOOD!"
            )

        if self.fill_rate < 0.90:
            recs.append(
                f"Low fill rate ({self.fill_rate:.1%}) - many signals rejected. "
                "This is expected with quality gates."
            )

        if self.avg_spread_capture_bps is not None and self.avg_spread_capture_bps < 0:
            recs.append(
                f"Negative spread capture ({self.avg_spread_capture_bps:.2f} bps) - "
                "filling outside the spread (expected with IOC LIMIT)"
            )

        return recs


# ============================================================================
# Analysis Functions
# ============================================================================

def load_trades(lookback_days: int = 30) -> List[Dict]:
    """Load trades from logs/trades.jsonl."""
    trades_file = Path("logs/trades.jsonl")
    if not trades_file.exists():
        print(f"\n[WARNING] No trades file found at {trades_file}")
        return []

    # Make cutoff timezone-aware (UTC)
    from datetime import timezone
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    trades = []

    with open(trades_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                trade = json.loads(line)
                ts = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                if ts >= cutoff:
                    trades.append(trade)
            except Exception as e:
                # Silently skip unparseable trades
                pass

    return trades


def analyze_fills(trades: List[Dict]) -> List[FillAnalysis]:
    """Extract filled orders and analyze slippage."""
    fills = []

    for trade in trades:
        if trade['status'] != 'FILLED':
            continue

        # Skip if missing critical data
        if trade.get('fill_price') is None:
            continue

        fills.append(FillAnalysis(
            timestamp=datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00')),
            symbol=trade['symbol'],
            side=trade['side'],
            qty=trade['qty'],
            decision_price=trade.get('entry_price_decision'),
            limit_price=trade['limit_price'],
            fill_price=trade['fill_price'],
            bid_at_execution=trade.get('market_bid_at_execution'),
            ask_at_execution=trade.get('market_ask_at_execution'),
            strategy=trade.get('strategy_used'),
        ))

    return fills


def generate_report(fills: List[FillAnalysis], total_trades: int) -> TransactionCostReport:
    """Generate comprehensive transaction cost report."""
    total_fills = len(fills)
    total_rejections = total_trades - total_fills
    fill_rate = total_fills / total_trades if total_trades > 0 else 0.0

    # Calculate slippage metrics (only for fills with decision_price)
    slippages = [f.slippage_bps for f in fills if f.slippage_bps is not None]
    spreads = [f.spread_bps for f in fills if f.spread_bps is not None]
    spread_captures = [f.spread_capture_bps for f in fills if f.spread_capture_bps is not None]

    return TransactionCostReport(
        total_fills=total_fills,
        total_rejections=total_rejections,
        fill_rate=fill_rate,
        avg_slippage_bps=float(pd.Series(slippages).mean()) if slippages else None,
        median_slippage_bps=float(pd.Series(slippages).median()) if slippages else None,
        p90_slippage_bps=float(pd.Series(slippages).quantile(0.90)) if slippages else None,
        p95_slippage_bps=float(pd.Series(slippages).quantile(0.95)) if slippages else None,
        avg_spread_bps=float(pd.Series(spreads).mean()) if spreads else None,
        avg_spread_capture_bps=float(pd.Series(spread_captures).mean()) if spread_captures else None,
        fills_analyzed=total_fills,
        fills_with_decision_price=len(slippages),
        fills_with_spread_data=len(spreads),
    )


def print_report(report: TransactionCostReport, fills: List[FillAnalysis]):
    """Pretty print transaction cost report."""
    print("\n" + "=" * 80)
    print("TRANSACTION COST ANALYSIS")
    print("Jim Simons / Renaissance Technologies Standard")
    print("=" * 80)

    print(f"\nFill Statistics:")
    print(f"  Total Fills: {report.total_fills}")
    print(f"  Total Rejections: {report.total_rejections}")
    print(f"  Fill Rate: {report.fill_rate:.1%}")

    print(f"\nSlippage Analysis (vs Decision Price):")
    if report.avg_slippage_bps is not None:
        print(f"  Fills with decision_price: {report.fills_with_decision_price}")
        print(f"  Average Slippage: {report.avg_slippage_bps:.2f} bps")
        print(f"  Median Slippage: {report.median_slippage_bps:.2f} bps")
        print(f"  90th Percentile: {report.p90_slippage_bps:.2f} bps")
        print(f"  95th Percentile: {report.p95_slippage_bps:.2f} bps")
    else:
        print(f"  [WARN] No fills with decision_price - cannot calculate slippage")

    print(f"\nSpread Analysis:")
    if report.avg_spread_bps is not None:
        print(f"  Fills with spread data: {report.fills_with_spread_data}")
        print(f"  Average Bid-Ask Spread: {report.avg_spread_bps:.2f} bps")
        print(f"  Average Spread Capture: {report.avg_spread_capture_bps:.2f} bps")
    else:
        print(f"  [WARN] No fills with bid/ask data")

    print(f"\nBacktest Comparison:")
    print(f"  Backtest Slippage Assumption: {report.backtest_slippage_bps:.2f} bps")
    if report.avg_slippage_bps is not None:
        print(f"  Actual Slippage: {report.avg_slippage_bps:.2f} bps")
        diff = report.avg_slippage_bps - report.backtest_slippage_bps
        if diff > 0:
            print(f"  [FAIL] Backtest is {diff:.2f} bps TOO OPTIMISTIC")
        else:
            print(f"  [OK] Backtest is {abs(diff):.2f} bps CONSERVATIVE")

    print(f"\nExecution Quality: {report.quality_grade}")

    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")

    # Sample fills table
    if fills:
        print(f"\nSample Fills (Last 10):")
        print(f"  {'Symbol':<8} {'Side':<5} {'Decision':<10} {'Limit':<10} {'Fill':<10} {'Slip (bps)':<12} {'Spread (bps)'}")
        print(f"  {'-'*80}")
        for fill in fills[-10:]:
            dec_price = f"${fill.decision_price:.2f}" if fill.decision_price else "N/A"
            slip = f"{fill.slippage_bps:+.2f}" if fill.slippage_bps is not None else "N/A"
            spread = f"{fill.spread_bps:.2f}" if fill.spread_bps is not None else "N/A"
            print(f"  {fill.symbol:<8} {fill.side:<5} {dec_price:<10} "
                  f"${fill.limit_price:<9.2f} ${fill.fill_price:<9.2f} {slip:<12} {spread}")

    # Verdict
    print("\n" + "=" * 80)
    if report.passed:
        print("[OK] PASSED - Transaction costs within acceptable limits")
    else:
        print("[FAIL] NEEDS ATTENTION - Transaction costs exceed backtest assumptions")
    print("=" * 80)


def save_report(report: TransactionCostReport, fills: List[FillAnalysis]):
    """Save report to file."""
    output_file = Path("reports/TRANSACTION_COST_VERIFICATION.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("# Transaction Cost Verification Report\n")
        f.write("**Jim Simons / Renaissance Technologies Standard**\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Status:** {'PASSED' if report.passed else 'NEEDS ATTENTION'}\n\n")
        f.write("---\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Fills:** {report.total_fills}\n")
        f.write(f"- **Fill Rate:** {report.fill_rate:.1%}\n")
        if report.avg_slippage_bps is not None:
            f.write(f"- **Average Slippage:** {report.avg_slippage_bps:.2f} bps\n")
            f.write(f"- **Backtest Assumption:** {report.backtest_slippage_bps:.2f} bps\n")
            f.write(f"- **Execution Quality:** {report.quality_grade}\n")
        f.write("\n---\n\n")

        f.write("## Slippage Analysis\n\n")
        if report.avg_slippage_bps is not None:
            f.write(f"- **Fills Analyzed:** {report.fills_with_decision_price}\n")
            f.write(f"- **Average:** {report.avg_slippage_bps:.2f} bps\n")
            f.write(f"- **Median:** {report.median_slippage_bps:.2f} bps\n")
            f.write(f"- **90th Percentile:** {report.p90_slippage_bps:.2f} bps\n")
            f.write(f"- **95th Percentile:** {report.p95_slippage_bps:.2f} bps\n")
        else:
            f.write("No fills with decision_price data available.\n")
        f.write("\n---\n\n")

        f.write("## Recommendations\n\n")
        for rec in report.recommendations:
            f.write(f"- {rec}\n")
        f.write("\n---\n\n")

        f.write("## Sample Fills\n\n")
        f.write("| Symbol | Side | Decision | Limit | Fill | Slippage (bps) | Spread (bps) |\n")
        f.write("|--------|------|----------|-------|------|----------------|---------------|\n")
        for fill in fills[-20:]:
            dec = f"${fill.decision_price:.2f}" if fill.decision_price else "N/A"
            slip = f"{fill.slippage_bps:+.2f}" if fill.slippage_bps is not None else "N/A"
            spread = f"{fill.spread_bps:.2f}" if fill.spread_bps is not None else "N/A"
            f.write(f"| {fill.symbol} | {fill.side} | {dec} | ${fill.limit_price:.2f} | "
                   f"${fill.fill_price:.2f} | {slip} | {spread} |\n")

        f.write("\n---\n\n")
        f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Verification Standard:** Jim Simons / Renaissance Technologies\n")

    print(f"\n[OK] Report saved to {output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify transaction costs vs paper trading (Jim Simons standard)"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Number of days to analyze (default: 30)",
    )

    args = parser.parse_args()

    # Load trades
    print(f"\nLoading trades from last {args.lookback_days} days...")
    trades = load_trades(lookback_days=args.lookback_days)

    if not trades:
        print("[ERROR] No trades found in logs/trades.jsonl")
        print("Run paper trading first to collect fill data.")
        sys.exit(1)

    print(f"[OK] Loaded {len(trades)} trades")

    # Analyze fills
    fills = analyze_fills(trades)
    print(f"[OK] Found {len(fills)} fills to analyze")

    # Generate report
    report = generate_report(fills, total_trades=len(trades))

    # Print results
    print_report(report, fills)

    # Save to file
    save_report(report, fills)

    # Exit code
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
