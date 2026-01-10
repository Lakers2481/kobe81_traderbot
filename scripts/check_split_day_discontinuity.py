"""
Check Split Day Discontinuity
==============================

Verifies the exact overnight return from day before split to split day.

For NVDA 10:1 split on 2024-06-10:
- If UNADJUSTED: Jun 9 close ~$1,200 → Jun 10 open ~$120 (90% drop)
- If ADJUSTED: Jun 9 close and Jun 10 open should be similar (normal overnight move)

Usage:
    python scripts/check_split_day_discontinuity.py --symbol NVDA --split-date 2024-06-10
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.providers.polygon_eod import fetch_daily_bars_polygon


def check_split_day_discontinuity(symbol: str, split_date: str):
    """Check for price discontinuity on split day."""
    split_dt = pd.to_datetime(split_date)
    # Fetch 10 days before and after to ensure we have the data
    start_date = (split_dt - timedelta(days=10)).strftime("%Y-%m-%d")
    end_date = (split_dt + timedelta(days=10)).strftime("%Y-%m-%d")

    print(f"\nChecking {symbol} split day discontinuity on {split_date}")
    print("=" * 80)

    df = fetch_daily_bars_polygon(
        symbol=symbol,
        start=start_date,
        end=end_date,
        cache_dir=Path("data/cache"),
        ignore_cache_ttl=False,
    )

    if df.empty:
        print("ERROR: No data")
        return

    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    split_date_obj = split_dt.date()

    # Find the day before and day of split
    df = df.sort_values("date")

    # Get split day index
    split_idx = df[df["date"] == split_date_obj].index
    if len(split_idx) == 0:
        print(f"ERROR: No data for split date {split_date}")
        return

    split_idx = split_idx[0]
    df_loc = df.index.get_loc(split_idx)

    if df_loc == 0:
        print("ERROR: No prior day data")
        return

    # Get prior day
    prior_idx = df.index[df_loc - 1]
    prior_row = df.loc[prior_idx]
    split_row = df.loc[split_idx]

    # Calculate overnight return
    prior_close = prior_row["close"]
    split_open = split_row["open"]
    split_close = split_row["close"]

    overnight_return = (split_open - prior_close) / prior_close * 100
    split_day_return = (split_close - split_open) / split_open * 100

    print(f"\nDay BEFORE Split ({prior_row['date']}):")
    print(f"  Close: ${prior_close:.2f}")
    print(f"  Volume: {prior_row['volume']:,.0f}")

    print(f"\nSplit Day ({split_row['date']}):")
    print(f"  Open: ${split_open:.2f}")
    print(f"  Close: ${split_close:.2f}")
    print(f"  Volume: {split_row['volume']:,.0f}")

    print(f"\nCalculations:")
    print(f"  Overnight Return: {overnight_return:.2f}%")
    print(f"  (Split Open - Prior Close) / Prior Close")
    print(f"  = ({split_open:.2f} - {prior_close:.2f}) / {prior_close:.2f}")

    print(f"\n  Split Day Return: {split_day_return:.2f}%")
    print(f"  (Split Close - Split Open) / Split Open")

    print("\n" + "=" * 80)
    print("VERDICT:")
    print("=" * 80)

    if abs(overnight_return) > 30:
        print(
            f"[FAIL] UNADJUSTED: Overnight return of {overnight_return:.2f}% indicates split is NOT adjusted in data"
        )
        print("   This is a CRITICAL data quality issue!")
    elif abs(overnight_return) < 10:
        print(
            f"[OK] ADJUSTED: Overnight return of {overnight_return:.2f}% is normal (data IS adjusted)"
        )
        print("   Polygon is correctly providing split-adjusted data.")
    else:
        print(
            f"[WARN] UNCLEAR: Overnight return of {overnight_return:.2f}% is in gray zone"
        )
        print("   Could be adjusted data with high volatility, or partial adjustment.")

    # Also check the 5 days before and after for context
    context_start_idx = max(0, df_loc - 5)
    context_end_idx = min(len(df) - 1, df_loc + 5)
    context_df = df.iloc[context_start_idx : context_end_idx + 1][
        ["date", "open", "close", "volume"]
    ]

    print("\n5 Days Before and After (for context):")
    print(context_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Check split day discontinuity")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--split-date", type=str, required=True)

    args = parser.parse_args()
    check_split_day_discontinuity(args.symbol, args.split_date)


if __name__ == "__main__":
    main()
