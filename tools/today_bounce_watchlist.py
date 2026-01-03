#!/usr/bin/env python3
"""
Today's Bounce Watchlist Generator

Generates daily bounce watchlist by:
1. Identifying tickers with current_streak >= 3
2. Looking up bounce profile (5Y preferred → 10Y fallback)
3. Applying gates, computing BounceScore
4. Ranking by BounceScore descending

Usage:
    python tools/today_bounce_watchlist.py
    python tools/today_bounce_watchlist.py --prefer 5 --fallback 10 --min_events 20
    python tools/today_bounce_watchlist.py --min_streak 4 --top 10

Options:
    --prefer: Preferred window (5 or 10, default: 5)
    --fallback: Fallback window (5 or 10, default: 10)
    --min_events: Minimum events required (default: 20)
    --min_streak: Minimum streak level (default: 3)
    --top: Number of top signals to show (default: 20)
    --output: Output directory (default: reports/bounce)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import time

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from bounce.data_loader import load_universe_data
from bounce.streak_analyzer import get_current_streaks
from bounce.strategy_integration import BounceIntegration, create_bounce_watchlist
from data.universe.loader import load_universe


def generate_watchlist(
    prefer: int = 5,
    fallback: int = 10,
    min_events: int = 20,
    min_streak: int = 3,
    top: int = 20,
    output_dir: Path = None,
    cap: int = None,
):
    """
    Generate today's bounce watchlist.

    Args:
        prefer: Preferred window years
        fallback: Fallback window years
        min_events: Minimum events required
        min_streak: Minimum streak level
        top: Number of top signals
        output_dir: Output directory
        cap: Limit universe size
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "reports" / "bounce"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TODAY'S BOUNCE WATCHLIST")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load per-stock summaries
    reports_dir = PROJECT_ROOT / "reports" / "bounce"
    per_stock_5y_path = reports_dir / "week_down_then_bounce_per_stock_5y.csv"
    per_stock_10y_path = reports_dir / "week_down_then_bounce_per_stock_10y.csv"

    per_stock_5y = None
    per_stock_10y = None

    if per_stock_5y_path.exists():
        per_stock_5y = pd.read_csv(per_stock_5y_path)
        print(f"Loaded 5Y per-stock: {len(per_stock_5y)} rows")
    else:
        print("WARNING: 5Y per-stock not found")

    if per_stock_10y_path.exists():
        per_stock_10y = pd.read_csv(per_stock_10y_path)
        print(f"Loaded 10Y per-stock: {len(per_stock_10y)} rows")
    else:
        print("WARNING: 10Y per-stock not found")

    if per_stock_5y is None and per_stock_10y is None:
        print("ERROR: No bounce data available. Run build_bounce_db.py first.")
        sys.exit(1)

    # Load universe
    print()
    print("Loading universe and current streak data...")
    symbols = load_universe(cap=cap or 900)
    print(f"  Universe: {len(symbols)} tickers")

    # Get current streaks (need to fetch fresh data)
    # For now, use cached data or fetch minimal data
    cache_dir = PROJECT_ROOT / "cache" / "polygon"

    # Try to load from recent scan or fetch fresh
    try:
        from data.providers.polygon_eod import fetch_daily_bars_polygon
        from datetime import timedelta

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        print("  Fetching recent data for streak calculation...")
        ticker_data = {}

        for i, symbol in enumerate(symbols):
            try:
                df = fetch_daily_bars_polygon(
                    symbol=symbol,
                    start=start_date,
                    end=end_date,
                    cache_dir=cache_dir,
                )
                if df is not None and len(df) > 0:
                    ticker_data[symbol] = df
            except Exception:
                pass

            if (i + 1) % 100 == 0:
                print(f"    Loaded {i+1}/{len(symbols)}...")

        print(f"  Loaded: {len(ticker_data)} tickers")

        # Calculate current streaks
        current_streaks_df = get_current_streaks(ticker_data)
        print(f"  Calculated streaks for: {len(current_streaks_df)} tickers")

    except Exception as e:
        print(f"  Error loading data: {e}")
        print("  Using empty streak data")
        current_streaks_df = pd.DataFrame(columns=['ticker', 'current_streak', 'last_close', 'last_date'])

    # Filter for minimum streak
    if len(current_streaks_df) > 0:
        candidates = current_streaks_df[current_streaks_df['current_streak'] >= min_streak].copy()
        print(f"\nCandidates with streak >= {min_streak}: {len(candidates)}")
    else:
        candidates = pd.DataFrame()
        print("\nNo streak data available")

    # Initialize bounce integration
    bounce_integration = BounceIntegration(
        per_stock_5y=per_stock_5y if prefer == 5 else per_stock_10y,
        per_stock_10y=per_stock_10y if fallback == 10 else per_stock_5y,
        min_events=min_events,
        min_bounce_score=40.0,  # Lower threshold for watchlist
        require_gate_pass=False,  # Show all for watchlist
    )

    # Create watchlist
    if len(candidates) > 0:
        watchlist = create_bounce_watchlist(
            current_streaks_df=candidates,
            bounce_integration=bounce_integration,
            min_streak=min_streak,
            max_signals=top * 2,  # Get more to filter
        )
    else:
        watchlist = pd.DataFrame()

    # Generate output
    today_str = datetime.now().strftime("%Y%m%d")

    if len(watchlist) > 0:
        # Sort by bounce score and take top
        watchlist = watchlist.sort_values('bounce_score', ascending=False).head(top)

        # Save CSV
        csv_path = output_dir / f"today_bounce_watchlist_{today_str}.csv"
        watchlist.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        # Save latest (overwrites)
        latest_path = output_dir / "today_bounce_watchlist.csv"
        watchlist.to_csv(latest_path, index=False)

        # Generate markdown
        md_lines = []
        md_lines.append(f"# Bounce Watchlist - {datetime.now().strftime('%Y-%m-%d')}")
        md_lines.append("")
        md_lines.append(f"**Generated:** {datetime.now().strftime('%H:%M:%S')}")
        md_lines.append(f"**Minimum Streak:** {min_streak}")
        md_lines.append(f"**Candidates:** {len(candidates)}")
        md_lines.append(f"**Filtered:** {len(watchlist)}")
        md_lines.append("")
        md_lines.append("## Top Bounce Candidates")
        md_lines.append("")
        md_lines.append("| Rank | Ticker | Streak | BounceScore | Window | Recovery | Avg Days | Gate |")
        md_lines.append("|------|--------|--------|-------------|--------|----------|----------|------|")

        for i, (_, row) in enumerate(watchlist.iterrows(), 1):
            ticker = row.get('ticker', '-')
            streak = row.get('streak', '-')
            score = row.get('bounce_score', 0)
            window = row.get('bounce_window_used', '-')
            recovery = row.get('bounce_recovery_rate')
            avg_days = row.get('bounce_avg_days')
            gate = row.get('bounce_gate_passed', False)

            recovery_str = f"{recovery:.0%}" if pd.notna(recovery) else "-"
            days_str = f"{avg_days:.1f}" if pd.notna(avg_days) else "-"
            gate_str = "✓" if gate else "✗"

            md_lines.append(f"| {i} | **{ticker}** | {streak} | {score:.0f} | {window} | {recovery_str} | {days_str} | {gate_str} |")

        md_lines.append("")
        md_lines.append("---")
        md_lines.append("*Generated by Kobe Bounce Analysis System*")

        md_path = output_dir / f"today_bounce_watchlist_{today_str}.md"
        md_path.write_text("\n".join(md_lines))
        print(f"Saved: {md_path}")

        # Also save latest
        latest_md_path = output_dir / "today_bounce_watchlist.md"
        latest_md_path.write_text("\n".join(md_lines))

        # Print watchlist
        print()
        print("=" * 60)
        print("TOP BOUNCE CANDIDATES")
        print("=" * 60)
        print()
        print(f"{'Rank':<5} {'Ticker':<8} {'Streak':<7} {'Score':<7} {'Window':<7} {'Recovery':<10} {'Days':<6} {'Gate'}")
        print("-" * 70)

        for i, (_, row) in enumerate(watchlist.iterrows(), 1):
            ticker = row.get('ticker', '-')
            streak = row.get('streak', '-')
            score = row.get('bounce_score', 0)
            window = row.get('bounce_window_used', '-')
            recovery = row.get('bounce_recovery_rate')
            avg_days = row.get('bounce_avg_days')
            gate = row.get('bounce_gate_passed', False)

            recovery_str = f"{recovery:.0%}" if pd.notna(recovery) else "-"
            days_str = f"{avg_days:.1f}" if pd.notna(avg_days) else "-"
            gate_str = "PASS" if gate else "FAIL"

            print(f"{i:<5} {ticker:<8} {streak:<7} {score:<7.0f} {window:<7} {recovery_str:<10} {days_str:<6} {gate_str}")

    else:
        print("\nNo candidates found meeting criteria.")

        # Save empty watchlist
        csv_path = output_dir / f"today_bounce_watchlist_{today_str}.csv"
        pd.DataFrame().to_csv(csv_path, index=False)

    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate today's bounce watchlist"
    )
    parser.add_argument(
        "--prefer",
        type=int,
        default=5,
        choices=[5, 10],
        help="Preferred window (5 or 10)",
    )
    parser.add_argument(
        "--fallback",
        type=int,
        default=10,
        choices=[5, 10],
        help="Fallback window (5 or 10)",
    )
    parser.add_argument(
        "--min_events",
        type=int,
        default=20,
        help="Minimum events required",
    )
    parser.add_argument(
        "--min_streak",
        type=int,
        default=3,
        help="Minimum streak level",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top signals to show",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Limit universe size (for testing)",
    )

    args = parser.parse_args()

    generate_watchlist(
        prefer=args.prefer,
        fallback=args.fallback,
        min_events=args.min_events,
        min_streak=args.min_streak,
        top=args.top,
        cap=args.cap,
    )


if __name__ == "__main__":
    main()
