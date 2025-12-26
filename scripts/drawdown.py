#!/usr/bin/env python3
"""
Drawdown Analysis Script for Kobe Trading System

Analyzes portfolio drawdowns:
- Calculate current drawdown from peak
- Show max drawdown history
- Time to recovery statistics
- Underwater curve data

Usage:
    python drawdown.py --dotenv /path/to/.env
    python drawdown.py --current
    python drawdown.py --history --days 252
    python drawdown.py --stats
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.env_loader import load_env


@dataclass
class DrawdownEvent:
    """Container for a drawdown event."""
    start_date: str
    trough_date: str
    recovery_date: Optional[str]
    peak_value: float
    trough_value: float
    max_drawdown: float  # As negative percentage
    duration_days: int
    recovery_days: Optional[int]
    is_recovered: bool


@dataclass
class DrawdownSummary:
    """Container for drawdown summary statistics."""
    current_drawdown: float
    current_drawdown_start: Optional[str]
    days_underwater: int
    max_drawdown: float
    max_drawdown_date: Optional[str]
    avg_drawdown: float
    total_drawdown_events: int
    avg_recovery_days: Optional[float]
    longest_drawdown_days: int
    current_peak: float
    current_value: float
    timestamp: str


def get_alpaca_account(api_key: str, api_secret: str, base_url: str) -> Optional[Dict]:
    """Fetch account info from Alpaca."""
    url = f"{base_url}/v2/account"
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': api_secret,
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        print(f"Error fetching Alpaca account: {e}")
        return None


def get_alpaca_portfolio_history(
    api_key: str,
    api_secret: str,
    base_url: str,
    period: str = '1A',
    timeframe: str = '1D'
) -> pd.DataFrame:
    """Fetch portfolio history from Alpaca."""
    url = f"{base_url}/v2/account/portfolio/history"
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': api_secret,
    }
    params = {
        'period': period,
        'timeframe': timeframe,
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json()
        timestamps = data.get('timestamp', [])
        equity = data.get('equity', [])

        if not timestamps or not equity:
            return pd.DataFrame()

        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps, unit='s'),
            'equity': equity,
        })
        df.set_index('timestamp', inplace=True)
        df = df[df['equity'].notna() & (df['equity'] > 0)]

        return df

    except Exception as e:
        print(f"Error fetching portfolio history: {e}")
        return pd.DataFrame()


def load_equity_curve(equity_file: Path) -> pd.DataFrame:
    """Load equity curve from local file."""
    if not equity_file.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(equity_file, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error loading equity file: {e}")
        return pd.DataFrame()


class DrawdownAnalyzer:
    """Portfolio drawdown analysis engine."""

    def __init__(self):
        self.equity_curve: Optional[pd.DataFrame] = None
        self.drawdown_series: Optional[pd.Series] = None
        self.drawdown_events: List[DrawdownEvent] = []

    def load_equity_data(
        self,
        use_broker: bool = False,
        equity_file: Optional[Path] = None,
        period: str = '1A'
    ) -> bool:
        """Load equity curve from broker or file."""

        if use_broker:
            api_key = os.getenv('ALPACA_API_KEY_ID', '')
            api_secret = os.getenv('ALPACA_API_SECRET_KEY', '')
            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')

            if api_key and api_secret:
                self.equity_curve = get_alpaca_portfolio_history(
                    api_key, api_secret, base_url, period=period
                )
                if not self.equity_curve.empty:
                    return True

        if equity_file:
            self.equity_curve = load_equity_curve(equity_file)
            if not self.equity_curve.empty:
                return True

        return False

    def generate_sample_data(self, days: int = 252) -> None:
        """Generate sample equity curve for demonstration."""
        print("No equity data available. Generating sample data for demonstration...")

        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Simulate equity curve with realistic volatility
        initial_equity = 100000
        daily_return_mean = 0.0003  # ~7.5% annual
        daily_return_std = 0.012   # ~19% annual vol

        returns = np.random.normal(daily_return_mean, daily_return_std, days)

        # Add some drawdown events
        returns[50:60] -= 0.015  # Small drawdown
        returns[120:150] -= 0.02  # Medium drawdown
        returns[200:210] -= 0.025  # Another drawdown

        equity = initial_equity * np.cumprod(1 + returns)

        self.equity_curve = pd.DataFrame({
            'equity': equity
        }, index=dates)

    def calculate_drawdown_series(self) -> pd.Series:
        """Calculate the drawdown series from equity curve."""
        if self.equity_curve is None or self.equity_curve.empty:
            return pd.Series()

        equity = self.equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max

        self.drawdown_series = drawdown
        return drawdown

    def identify_drawdown_events(self, threshold: float = -0.01) -> List[DrawdownEvent]:
        """Identify distinct drawdown events."""
        if self.drawdown_series is None:
            self.calculate_drawdown_series()

        if self.drawdown_series is None or self.drawdown_series.empty:
            return []

        events = []
        equity = self.equity_curve['equity']
        drawdown = self.drawdown_series

        in_drawdown = False
        event_start = None
        event_peak = None
        event_trough = None
        event_trough_value = None

        for i, (date, dd) in enumerate(drawdown.items()):
            if dd < threshold and not in_drawdown:
                # Start of new drawdown
                in_drawdown = True
                # Find the peak (last day at 0 drawdown)
                for j in range(i - 1, -1, -1):
                    if drawdown.iloc[j] >= 0:
                        event_start = drawdown.index[j]
                        event_peak = equity.iloc[j]
                        break
                else:
                    event_start = drawdown.index[0]
                    event_peak = equity.iloc[0]
                event_trough = date
                event_trough_value = dd

            elif in_drawdown:
                if dd < event_trough_value:
                    # Deeper trough
                    event_trough = date
                    event_trough_value = dd

                if dd >= 0:
                    # Recovery
                    recovery_date = date
                    duration = (event_trough - event_start).days if event_start else 0
                    recovery_days = (recovery_date - event_trough).days

                    events.append(DrawdownEvent(
                        start_date=event_start.strftime('%Y-%m-%d') if event_start else '',
                        trough_date=event_trough.strftime('%Y-%m-%d'),
                        recovery_date=recovery_date.strftime('%Y-%m-%d'),
                        peak_value=float(event_peak) if event_peak else 0,
                        trough_value=float(equity.loc[event_trough]),
                        max_drawdown=float(event_trough_value),
                        duration_days=duration,
                        recovery_days=recovery_days,
                        is_recovered=True
                    ))

                    in_drawdown = False
                    event_start = None
                    event_peak = None
                    event_trough = None
                    event_trough_value = None

        # Handle ongoing drawdown
        if in_drawdown and event_trough is not None:
            duration = (event_trough - event_start).days if event_start else 0
            events.append(DrawdownEvent(
                start_date=event_start.strftime('%Y-%m-%d') if event_start else '',
                trough_date=event_trough.strftime('%Y-%m-%d'),
                recovery_date=None,
                peak_value=float(event_peak) if event_peak else 0,
                trough_value=float(equity.loc[event_trough]),
                max_drawdown=float(event_trough_value),
                duration_days=duration,
                recovery_days=None,
                is_recovered=False
            ))

        self.drawdown_events = events
        return events

    def get_current_drawdown(self) -> Tuple[float, Optional[str], int]:
        """Get current drawdown, start date, and days underwater."""
        if self.drawdown_series is None:
            self.calculate_drawdown_series()

        if self.drawdown_series is None or self.drawdown_series.empty:
            return 0.0, None, 0

        current_dd = self.drawdown_series.iloc[-1]

        if current_dd >= 0:
            return 0.0, None, 0

        # Find when drawdown started
        days_underwater = 0
        start_date = None

        for i in range(len(self.drawdown_series) - 1, -1, -1):
            if self.drawdown_series.iloc[i] >= 0:
                days_underwater = len(self.drawdown_series) - i - 1
                if i > 0:
                    start_date = self.drawdown_series.index[i].strftime('%Y-%m-%d')
                break
        else:
            days_underwater = len(self.drawdown_series)
            start_date = self.drawdown_series.index[0].strftime('%Y-%m-%d')

        return float(current_dd), start_date, days_underwater

    def get_max_drawdown(self) -> Tuple[float, Optional[str]]:
        """Get maximum drawdown and date."""
        if self.drawdown_series is None:
            self.calculate_drawdown_series()

        if self.drawdown_series is None or self.drawdown_series.empty:
            return 0.0, None

        max_dd_idx = self.drawdown_series.idxmin()
        max_dd = self.drawdown_series.min()

        return float(max_dd), max_dd_idx.strftime('%Y-%m-%d') if max_dd_idx else None

    def get_summary(self) -> DrawdownSummary:
        """Get comprehensive drawdown summary."""
        current_dd, dd_start, days_underwater = self.get_current_drawdown()
        max_dd, max_dd_date = self.get_max_drawdown()

        if not self.drawdown_events:
            self.identify_drawdown_events()

        # Calculate average drawdown (below 0)
        avg_dd = 0.0
        if self.drawdown_series is not None:
            underwater = self.drawdown_series[self.drawdown_series < 0]
            avg_dd = underwater.mean() if len(underwater) > 0 else 0.0

        # Calculate recovery statistics
        recovered_events = [e for e in self.drawdown_events if e.is_recovered]
        avg_recovery = None
        if recovered_events:
            recovery_days = [e.recovery_days for e in recovered_events if e.recovery_days is not None]
            avg_recovery = np.mean(recovery_days) if recovery_days else None

        # Longest drawdown
        longest_dd = 0
        for event in self.drawdown_events:
            total_days = event.duration_days + (event.recovery_days or 0)
            if total_days > longest_dd:
                longest_dd = total_days

        # Current peak and value
        equity = self.equity_curve['equity'] if self.equity_curve is not None else pd.Series([0])
        current_peak = float(equity.expanding().max().iloc[-1])
        current_value = float(equity.iloc[-1])

        return DrawdownSummary(
            current_drawdown=current_dd,
            current_drawdown_start=dd_start,
            days_underwater=days_underwater,
            max_drawdown=max_dd,
            max_drawdown_date=max_dd_date,
            avg_drawdown=float(avg_dd),
            total_drawdown_events=len(self.drawdown_events),
            avg_recovery_days=float(avg_recovery) if avg_recovery else None,
            longest_drawdown_days=longest_dd,
            current_peak=current_peak,
            current_value=current_value,
            timestamp=datetime.now().isoformat()
        )

    def get_underwater_curve(self) -> pd.DataFrame:
        """Get underwater curve data for charting."""
        if self.drawdown_series is None:
            self.calculate_drawdown_series()

        if self.drawdown_series is None:
            return pd.DataFrame()

        return pd.DataFrame({
            'date': self.drawdown_series.index,
            'drawdown': self.drawdown_series.values,
            'equity': self.equity_curve['equity'].values if self.equity_curve is not None else []
        })


def print_current_drawdown(analyzer: DrawdownAnalyzer) -> None:
    """Print current drawdown status."""
    current_dd, dd_start, days_underwater = analyzer.get_current_drawdown()

    print("\n" + "=" * 60)
    print("            CURRENT DRAWDOWN STATUS")
    print("=" * 60)

    if current_dd >= 0:
        print("\n  Portfolio is at or above its peak!")
        print("  Current drawdown: 0.00%")
        print("  Status: AT NEW HIGHS")
    else:
        equity = analyzer.equity_curve['equity']
        current_peak = equity.expanding().max().iloc[-1]
        current_value = equity.iloc[-1]

        print(f"\n  Current Drawdown:     {current_dd:.2%}")
        print(f"  Days Underwater:      {days_underwater}")
        print(f"  Drawdown Started:     {dd_start}")
        print(f"\n  Peak Equity:          ${current_peak:,.2f}")
        print(f"  Current Equity:       ${current_value:,.2f}")
        print(f"  Loss from Peak:       ${current_value - current_peak:,.2f}")

        # Recovery projection (simple linear)
        if days_underwater > 0 and current_dd < 0:
            avg_daily_dd = current_dd / days_underwater
            if avg_daily_dd < 0:
                est_recovery = int(abs(current_dd / 0.0005))  # Assume 0.05% daily recovery
                print(f"\n  Est. Days to Recovery: ~{est_recovery} days (at 0.05%/day)")

    print("\n" + "=" * 60)


def print_drawdown_history(events: List[DrawdownEvent]) -> None:
    """Print drawdown event history."""
    print("\n" + "=" * 80)
    print("                        DRAWDOWN EVENT HISTORY")
    print("=" * 80)

    if not events:
        print("\n  No significant drawdown events recorded.")
        print()
        return

    print(f"\n{'#':>3s} | {'Start Date':>12s} | {'Trough Date':>12s} | {'Max DD':>10s} | "
          f"{'Duration':>10s} | {'Recovery':>10s} | Status")
    print("-" * 80)

    for i, event in enumerate(events, 1):
        status = "RECOVERED" if event.is_recovered else "ONGOING"
        recovery_str = f"{event.recovery_days}d" if event.recovery_days else "N/A"
        print(f"{i:>3d} | {event.start_date:>12s} | {event.trough_date:>12s} | "
              f"{event.max_drawdown:>9.2%} | {event.duration_days:>9d}d | "
              f"{recovery_str:>10s} | {status}")

    print("-" * 80)

    # Summary statistics
    recovered = [e for e in events if e.is_recovered]
    ongoing = [e for e in events if not e.is_recovered]

    print(f"\n  Total Events: {len(events)}")
    print(f"  Recovered: {len(recovered)}")
    print(f"  Ongoing: {len(ongoing)}")

    if recovered:
        avg_dd = np.mean([e.max_drawdown for e in recovered])
        avg_recovery = np.mean([e.recovery_days for e in recovered if e.recovery_days])
        print(f"\n  Avg Drawdown (recovered): {avg_dd:.2%}")
        print(f"  Avg Recovery Time: {avg_recovery:.0f} days")

    print()


def print_drawdown_stats(summary: DrawdownSummary) -> None:
    """Print comprehensive drawdown statistics."""
    print("\n" + "=" * 70)
    print("              DRAWDOWN STATISTICS SUMMARY")
    print("=" * 70)

    print("\n  CURRENT STATUS")
    print("  " + "-" * 40)
    if summary.current_drawdown >= 0:
        print("    Portfolio at peak equity")
    else:
        print(f"    Current Drawdown:    {summary.current_drawdown:.2%}")
        print(f"    Drawdown Started:    {summary.current_drawdown_start}")
        print(f"    Days Underwater:     {summary.days_underwater}")

    print(f"\n    Current Peak:        ${summary.current_peak:,.2f}")
    print(f"    Current Value:       ${summary.current_value:,.2f}")

    print("\n  HISTORICAL STATISTICS")
    print("  " + "-" * 40)
    print(f"    Maximum Drawdown:    {summary.max_drawdown:.2%}")
    print(f"    Max DD Date:         {summary.max_drawdown_date}")
    print(f"    Average Drawdown:    {summary.avg_drawdown:.2%}")
    print(f"    Total DD Events:     {summary.total_drawdown_events}")
    print(f"    Longest DD Period:   {summary.longest_drawdown_days} days")

    if summary.avg_recovery_days:
        print(f"    Avg Recovery Time:   {summary.avg_recovery_days:.1f} days")

    print("\n  RISK METRICS")
    print("  " + "-" * 40)

    # Calmar-like ratio (simplified)
    if summary.max_drawdown < 0:
        # Estimate annual return from equity curve
        if summary.current_peak > 0 and summary.current_value > 0:
            # Simple return estimate
            annual_return_est = 0.10  # Placeholder - would need more data
            calmar = abs(annual_return_est / summary.max_drawdown)
            print(f"    Est. Calmar Ratio:   {calmar:.2f} (higher is better)")

    # Underwater ratio
    # How much time spent underwater in the period
    print(f"    Days Underwater:     {summary.days_underwater}")

    # Risk classification
    if summary.max_drawdown >= 0:
        risk = "MINIMAL"
    elif summary.max_drawdown > -0.05:
        risk = "LOW"
    elif summary.max_drawdown > -0.10:
        risk = "MODERATE"
    elif summary.max_drawdown > -0.20:
        risk = "ELEVATED"
    else:
        risk = "HIGH"

    print(f"\n    Risk Classification: {risk}")

    print("\n" + "=" * 70)


def print_underwater_chart(analyzer: DrawdownAnalyzer) -> None:
    """Print ASCII underwater chart."""
    if analyzer.drawdown_series is None or analyzer.drawdown_series.empty:
        print("No drawdown data available for chart.")
        return

    print("\n" + "=" * 70)
    print("                UNDERWATER CURVE (ASCII)")
    print("=" * 70)

    dd = analyzer.drawdown_series

    # Resample to weekly for display
    weekly = dd.resample('W').last().dropna()

    if len(weekly) == 0:
        print("  Not enough data for chart.")
        return

    # Chart dimensions
    width = 60
    height = 15

    min_dd = dd.min()
    max_dd = 0

    if min_dd >= 0:
        print("  No drawdowns to display - portfolio at highs!")
        return

    # Scale factor
    scale = height / abs(min_dd) if min_dd != 0 else 1

    print(f"\n  0.00% |{'=' * width}")

    # Create chart rows
    for row in range(height):
        dd_level = -(row + 1) / scale
        row_str = f"{dd_level:>6.1%} |"

        # Sample points for this row
        for i, val in enumerate(np.linspace(0, len(weekly) - 1, width)):
            idx = int(val)
            if idx < len(weekly):
                dd_val = weekly.iloc[idx]
                if dd_val <= dd_level:
                    row_str += "#"
                else:
                    row_str += " "
            else:
                row_str += " "

        print(row_str)

    print(f"        |{'-' * width}")

    # Date labels
    start_date = weekly.index[0].strftime('%Y-%m')
    end_date = weekly.index[-1].strftime('%Y-%m')
    print(f"        {start_date}{' ' * (width - len(start_date) - len(end_date))}{end_date}")

    print(f"\n  Max Drawdown: {min_dd:.2%}")
    print()


def main():
    ap = argparse.ArgumentParser(description='Drawdown Analysis for Kobe Trading System')
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env',
                    help='Path to .env file')
    ap.add_argument('--equity-file', type=str, default='state/equity_curve.csv',
                    help='Path to equity curve CSV file')
    ap.add_argument('--broker', action='store_true',
                    help='Fetch equity history from broker')
    ap.add_argument('--current', action='store_true',
                    help='Show only current drawdown status')
    ap.add_argument('--history', action='store_true',
                    help='Show drawdown event history')
    ap.add_argument('--stats', action='store_true',
                    help='Show comprehensive statistics')
    ap.add_argument('--chart', action='store_true',
                    help='Show ASCII underwater chart')
    ap.add_argument('--period', type=str, default='1A',
                    help='History period for broker (1D, 1W, 1M, 3M, 1A)')
    ap.add_argument('--json', action='store_true',
                    help='Output as JSON')
    ap.add_argument('--sample', action='store_true',
                    help='Use sample data for demonstration')
    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # Initialize analyzer
    analyzer = DrawdownAnalyzer()

    # Load equity data
    equity_file = Path(args.equity_file) if args.equity_file else None

    if args.sample:
        analyzer.generate_sample_data(days=252)
        print("Using sample data for demonstration.")
    elif not analyzer.load_equity_data(use_broker=args.broker, equity_file=equity_file, period=args.period):
        print("\n" + "=" * 60)
        print("  No equity data available.")
        print("=" * 60)
        print("\n  Checked locations:")
        print(f"    - File: {equity_file}")
        if args.broker:
            print("    - Alpaca portfolio history API")
        print("\n  To demonstrate, run with --sample flag.")
        print()
        sys.exit(0)

    # Calculate drawdowns
    analyzer.calculate_drawdown_series()
    analyzer.identify_drawdown_events()

    # Output based on flags
    if args.json:
        summary = analyzer.get_summary()
        output = {
            'summary': asdict(summary),
            'events': [asdict(e) for e in analyzer.drawdown_events],
            'underwater_curve': {
                'dates': [d.isoformat() if hasattr(d, 'isoformat') else str(d)
                          for d in analyzer.drawdown_series.index.tolist()],
                'values': analyzer.drawdown_series.tolist()
            } if analyzer.drawdown_series is not None else {}
        }
        print(json.dumps(output, indent=2, default=str))
    elif args.current:
        print_current_drawdown(analyzer)
    elif args.history:
        print_drawdown_history(analyzer.drawdown_events)
    elif args.stats:
        summary = analyzer.get_summary()
        print_drawdown_stats(summary)
    elif args.chart:
        print_underwater_chart(analyzer)
    else:
        # Full report
        print_current_drawdown(analyzer)
        print_drawdown_history(analyzer.drawdown_events)
        summary = analyzer.get_summary()
        print_drawdown_stats(summary)
        print_underwater_chart(analyzer)


if __name__ == '__main__':
    main()
