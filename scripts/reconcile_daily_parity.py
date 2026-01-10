#!/usr/bin/env python3
"""
Daily Parity Reconciliation Script

FIX (2026-01-08): Created as part of Phase 4.3 - Daily reconciliation comparing
expected (backtest) vs actual (live) execution.

This script:
1. Loads today's signals from logs/signals.jsonl
2. Loads today's executed trades from logs/trades.jsonl
3. Compares signal prices vs fill prices (slippage analysis)
4. Checks for missed signals (signal generated but not executed)
5. Checks for unexpected trades (executed but no matching signal)
6. Reports parity metrics and alerts

Usage:
    python scripts/reconcile_daily_parity.py
    python scripts/reconcile_daily_parity.py --date 2026-01-07
    python scripts/reconcile_daily_parity.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.structured_log import jlog

# Paths
SIGNALS_LOG = ROOT / "logs" / "signals.jsonl"
TRADES_LOG = ROOT / "logs" / "trades.jsonl"
RECONCILE_OUTPUT = ROOT / "reports" / "daily_parity"


@dataclass
class SignalRecord:
    """A signal from the scanner."""
    timestamp: str
    symbol: str
    side: str
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    strategy: str
    conf_score: Optional[float] = None


@dataclass
class TradeRecord:
    """An executed trade from the broker."""
    timestamp: str
    symbol: str
    side: str
    qty: int
    limit_price: float
    fill_price: Optional[float]
    status: str
    decision_id: str
    strategy_used: Optional[str] = None


@dataclass
class ParityReport:
    """Daily parity reconciliation report."""
    date: str
    signals_count: int = 0
    trades_count: int = 0
    matched_count: int = 0
    missed_signals: List[str] = field(default_factory=list)
    unexpected_trades: List[str] = field(default_factory=list)
    avg_slippage_bps: float = 0.0
    max_slippage_bps: float = 0.0
    fill_rate: float = 0.0
    alerts: List[str] = field(default_factory=list)
    verdict: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "signals_count": self.signals_count,
            "trades_count": self.trades_count,
            "matched_count": self.matched_count,
            "missed_signals": self.missed_signals,
            "unexpected_trades": self.unexpected_trades,
            "avg_slippage_bps": round(self.avg_slippage_bps, 2),
            "max_slippage_bps": round(self.max_slippage_bps, 2),
            "fill_rate": round(self.fill_rate, 4),
            "alerts": self.alerts,
            "verdict": self.verdict,
        }


def load_signals_for_date(target_date: date) -> List[SignalRecord]:
    """Load signals generated on a specific date."""
    signals = []
    if not SIGNALS_LOG.exists():
        return signals

    date_str = target_date.strftime("%Y-%m-%d")

    with open(SIGNALS_LOG, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                ts = data.get('timestamp', '')
                if ts.startswith(date_str):
                    signals.append(SignalRecord(
                        timestamp=ts,
                        symbol=data.get('symbol', ''),
                        side=data.get('side', 'long'),
                        entry_price=float(data.get('entry_price', 0)),
                        stop_loss=float(data['stop_loss']) if data.get('stop_loss') else None,
                        take_profit=float(data['take_profit']) if data.get('take_profit') else None,
                        strategy=data.get('strategy', 'unknown'),
                        conf_score=float(data['conf_score']) if data.get('conf_score') else None,
                    ))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    return signals


def load_trades_for_date(target_date: date) -> List[TradeRecord]:
    """Load executed trades on a specific date."""
    trades = []
    if not TRADES_LOG.exists():
        return trades

    date_str = target_date.strftime("%Y-%m-%d")

    with open(TRADES_LOG, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                ts = data.get('timestamp', '')
                if ts.startswith(date_str):
                    trades.append(TradeRecord(
                        timestamp=ts,
                        symbol=data.get('symbol', ''),
                        side=data.get('side', 'BUY'),
                        qty=int(data.get('qty', 0)),
                        limit_price=float(data.get('limit_price', 0)),
                        fill_price=float(data['fill_price']) if data.get('fill_price') else None,
                        status=data.get('status', 'UNKNOWN'),
                        decision_id=data.get('decision_id', ''),
                        strategy_used=data.get('strategy_used'),
                    ))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    return trades


def calculate_slippage_bps(signal_price: float, fill_price: float, side: str) -> float:
    """Calculate slippage in basis points."""
    if signal_price <= 0:
        return 0.0
    # For buys, positive slippage = paid more than expected
    # For sells, positive slippage = received less than expected
    if side.upper() in ('BUY', 'LONG'):
        slip = (fill_price - signal_price) / signal_price * 10000
    else:
        slip = (signal_price - fill_price) / signal_price * 10000
    return slip


def reconcile_signals_and_trades(
    signals: List[SignalRecord],
    trades: List[TradeRecord]
) -> ParityReport:
    """Reconcile signals against executed trades."""
    report = ParityReport(date=datetime.now().strftime("%Y-%m-%d"))
    report.signals_count = len(signals)
    report.trades_count = len(trades)

    # Build lookup of signals by symbol
    signal_by_symbol: Dict[str, SignalRecord] = {}
    for sig in signals:
        signal_by_symbol[sig.symbol] = sig

    # Build lookup of filled trades by symbol
    filled_trades: Dict[str, TradeRecord] = {}
    for trade in trades:
        if trade.status.upper() == 'FILLED' and trade.fill_price:
            filled_trades[trade.symbol] = trade

    # Calculate matches and slippage
    slippages = []
    matched_symbols = set()

    for symbol, sig in signal_by_symbol.items():
        if symbol in filled_trades:
            trade = filled_trades[symbol]
            matched_symbols.add(symbol)

            # Calculate slippage
            if trade.fill_price and sig.entry_price > 0:
                slip = calculate_slippage_bps(sig.entry_price, trade.fill_price, sig.side)
                slippages.append(slip)

    report.matched_count = len(matched_symbols)

    # Find missed signals (signal but no fill)
    for symbol, sig in signal_by_symbol.items():
        if symbol not in filled_trades:
            report.missed_signals.append(f"{symbol} ({sig.strategy})")

    # Find unexpected trades (fill but no signal)
    for symbol, trade in filled_trades.items():
        if symbol not in signal_by_symbol:
            report.unexpected_trades.append(f"{symbol} ({trade.status})")

    # Calculate slippage statistics
    if slippages:
        report.avg_slippage_bps = sum(slippages) / len(slippages)
        report.max_slippage_bps = max(abs(s) for s in slippages)

    # Calculate fill rate
    if report.signals_count > 0:
        report.fill_rate = report.matched_count / report.signals_count

    # Generate alerts
    if report.avg_slippage_bps > 15:
        report.alerts.append(f"HIGH SLIPPAGE: Avg {report.avg_slippage_bps:.1f} bps > 15 bps threshold")

    if report.max_slippage_bps > 50:
        report.alerts.append(f"EXTREME SLIPPAGE: Max {report.max_slippage_bps:.1f} bps > 50 bps")

    if report.fill_rate < 0.5 and report.signals_count >= 3:
        report.alerts.append(f"LOW FILL RATE: {report.fill_rate:.1%} < 50%")

    if report.unexpected_trades:
        report.alerts.append(f"UNEXPECTED TRADES: {len(report.unexpected_trades)} trades without signals")

    # Determine verdict
    if not report.alerts:
        report.verdict = "PASS"
    elif len(report.alerts) == 1:
        report.verdict = "WARNING"
    else:
        report.verdict = "FAIL"

    return report


def print_report(report: ParityReport) -> None:
    """Print human-readable report."""
    print("\n" + "=" * 70)
    print("          DAILY PARITY RECONCILIATION REPORT")
    print("=" * 70)
    print(f"Date: {report.date}")
    print(f"Verdict: {report.verdict}")
    print("-" * 70)

    print(f"\nSIGNALS & TRADES:")
    print(f"  Signals generated: {report.signals_count}")
    print(f"  Trades executed:   {report.trades_count}")
    print(f"  Matched:           {report.matched_count}")
    print(f"  Fill rate:         {report.fill_rate:.1%}")

    print(f"\nSLIPPAGE:")
    print(f"  Average: {report.avg_slippage_bps:.1f} bps")
    print(f"  Maximum: {report.max_slippage_bps:.1f} bps")
    print(f"  Expected: ~10 bps (IOC LIMIT)")

    if report.missed_signals:
        print(f"\nMISSED SIGNALS ({len(report.missed_signals)}):")
        for s in report.missed_signals[:5]:
            print(f"  - {s}")
        if len(report.missed_signals) > 5:
            print(f"  ... and {len(report.missed_signals) - 5} more")

    if report.unexpected_trades:
        print(f"\nUNEXPECTED TRADES ({len(report.unexpected_trades)}):")
        for t in report.unexpected_trades[:5]:
            print(f"  - {t}")
        if len(report.unexpected_trades) > 5:
            print(f"  ... and {len(report.unexpected_trades) - 5} more")

    if report.alerts:
        print(f"\nALERTS ({len(report.alerts)}):")
        for alert in report.alerts:
            print(f"  [!] {alert}")
    else:
        print("\n[OK] No alerts - parity looks good!")

    print("\n" + "=" * 70)


def save_report(report: ParityReport, output_dir: Path) -> Path:
    """Save report to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"parity_{report.date}.json"
    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Daily parity reconciliation")
    parser.add_argument("--date", type=str, help="Date to reconcile (YYYY-MM-DD), default today")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--save", action="store_true", help="Save report to file")
    args = parser.parse_args()

    # Parse date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"Invalid date format: {args.date}. Use YYYY-MM-DD.")
            sys.exit(1)
    else:
        target_date = date.today()

    # Load data
    signals = load_signals_for_date(target_date)
    trades = load_trades_for_date(target_date)

    # Reconcile
    report = reconcile_signals_and_trades(signals, trades)
    report.date = target_date.strftime("%Y-%m-%d")

    # Output
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report)

    # Save if requested
    if args.save:
        filepath = save_report(report, RECONCILE_OUTPUT)
        print(f"\nReport saved to: {filepath}")

    # Log result
    jlog("daily_parity_reconcile",
         date=report.date,
         signals=report.signals_count,
         trades=report.trades_count,
         matched=report.matched_count,
         avg_slippage_bps=report.avg_slippage_bps,
         verdict=report.verdict)

    # Exit code based on verdict
    if report.verdict == "PASS":
        sys.exit(0)
    elif report.verdict == "WARNING":
        sys.exit(0)  # Warnings don't fail
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
