"""
Deep Dive: Volume Adjustment Verification
==========================================

The initial test showed NVDA volume ratio of 0.56 instead of expected 10.0.
This could mean:
1. Polygon doesn't adjust volume for splits (CRITICAL BUG)
2. Volume naturally decreased after split (normal market behavior)
3. We're interpreting adjusted data incorrectly

This script investigates further by:
- Fetching longer history to see volume trends
- Comparing volume patterns before/after split
- Checking if unadjusted volume shows the expected jump

Usage:
    python scripts/verify_volume_adjustment.py --symbol NVDA --split-date 2024-06-10
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.providers.polygon_eod import fetch_daily_bars_polygon
from core.structured_log import jlog


def analyze_volume_around_split(
    symbol: str,
    split_date: str,
    split_ratio: float,
    window_days: int = 30,
):
    """
    Analyze volume behavior around a stock split.

    Args:
        symbol: Stock ticker
        split_date: Split date (YYYY-MM-DD)
        split_ratio: Split ratio (e.g., 10.0 for 10:1)
        window_days: Days before/after to analyze
    """
    split_dt = pd.to_datetime(split_date)
    start_date = (split_dt - timedelta(days=window_days * 2)).strftime("%Y-%m-%d")
    end_date = (split_dt + timedelta(days=window_days * 2)).strftime("%Y-%m-%d")

    print(f"\nAnalyzing {symbol} volume around {split_date} ({split_ratio}:1 split)")
    print("=" * 80)

    # Fetch adjusted data
    df = fetch_daily_bars_polygon(
        symbol=symbol,
        start=start_date,
        end=end_date,
        cache_dir=Path("data/cache"),
        ignore_cache_ttl=False,
    )

    if df.empty:
        print("ERROR: No data returned")
        return

    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    split_date_obj = split_dt.date()

    # Split into pre/post periods
    pre_split = df[df["date"] < split_date_obj]
    post_split = df[df["date"] >= split_date_obj]

    if pre_split.empty or post_split.empty:
        print("ERROR: Insufficient data")
        return

    # Calculate statistics
    pre_avg_volume = pre_split["volume"].mean()
    post_avg_volume = post_split["volume"].mean()
    volume_ratio = post_avg_volume / pre_avg_volume

    pre_avg_price = pre_split["close"].mean()
    post_avg_price = post_split["close"].mean()
    price_ratio = pre_avg_price / post_avg_price

    print(f"\nPre-Split Period ({len(pre_split)} days):")
    print(f"  Avg Volume: {pre_avg_volume:,.0f}")
    print(f"  Avg Price: ${pre_avg_price:.2f}")

    print(f"\nPost-Split Period ({len(post_split)} days):")
    print(f"  Avg Volume: {post_avg_volume:,.0f}")
    print(f"  Avg Price: ${post_avg_price:.2f}")

    print(f"\nRatios:")
    print(f"  Volume Ratio (post/pre): {volume_ratio:.2f}x")
    print(f"  Price Ratio (pre/post): {price_ratio:.2f}x")

    print(f"\nExpected (if unadjusted):")
    print(f"  Volume Ratio: {split_ratio:.2f}x (more shares outstanding)")
    print(f"  Price Ratio: {split_ratio:.2f}x (lower price per share)")

    print(f"\nExpected (if adjusted):")
    print(f"  Volume Ratio: ~1.0x (natural market activity)")
    print(f"  Price Ratio: ~1.0x (no discontinuity)")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)

    if abs(price_ratio - 1.0) < 0.2:
        print("[OK] Prices are ADJUSTED (no discontinuity)")
    else:
        print(f"WARNING: Prices show {price_ratio:.2f}x ratio (possibly UNADJUSTED)")

    if abs(volume_ratio - 1.0) < 0.5:
        print(
            "[OK] Volume appears ADJUSTED or following natural market behavior"
        )
        print(
            f"    (Ratio {volume_ratio:.2f}x is close to 1.0, suggesting adjusted data or stable interest)"
        )
    elif abs(volume_ratio - split_ratio) < split_ratio * 0.3:
        print(
            f"WARNING: Volume shows {volume_ratio:.2f}x ratio (close to split ratio {split_ratio:.2f}x)"
        )
        print("    This suggests volume might be UNADJUSTED")
    else:
        print(
            f"INFO: Volume ratio {volume_ratio:.2f}x differs from both 1.0 and split ratio {split_ratio:.2f}x"
        )
        print("    This likely reflects natural market behavior (interest decreased after split)")

    # Check for discontinuity on split day
    split_day = df[df["date"] == split_date_obj]
    if not split_day.empty:
        split_volume = split_day.iloc[0]["volume"]
        split_price = split_day.iloc[0]["close"]

        print(f"\nOn Split Day ({split_date}):")
        print(f"  Volume: {split_volume:,.0f}")
        print(f"  Close: ${split_price:.2f}")

        # Compare to surrounding days
        surrounding = df[
            (df["date"] >= (split_date_obj - timedelta(days=5)))
            & (df["date"] <= (split_date_obj + timedelta(days=5)))
        ]
        avg_surrounding_volume = surrounding["volume"].mean()
        spike_ratio = split_volume / avg_surrounding_volume

        print(f"  Avg surrounding volume: {avg_surrounding_volume:,.0f}")
        print(f"  Spike ratio: {spike_ratio:.2f}x")

        if spike_ratio > 2.0:
            print("  [INFO] Volume spike on split day (typical for corporate actions)")

    # Plot
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot price
        ax1.plot(df["date"], df["close"], label="Close Price", color="blue")
        ax1.axvline(split_date_obj, color="red", linestyle="--", label="Split Date")
        ax1.set_ylabel("Price ($)")
        ax1.set_title(f"{symbol} Price Around {split_date} Split")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot volume
        ax2.bar(df["date"], df["volume"], label="Volume", color="green", alpha=0.6)
        ax2.axvline(split_date_obj, color="red", linestyle="--", label="Split Date")
        ax2.set_ylabel("Volume")
        ax2.set_xlabel("Date")
        ax2.set_title(f"{symbol} Volume Around {split_date} Split")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = f"reports/{symbol}_split_analysis_{split_date}.png"
        Path("reports").mkdir(exist_ok=True)
        plt.savefig(output_path)
        print(f"\nPlot saved to: {output_path}")

    except Exception as e:
        print(f"\nCould not generate plot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze volume adjustment around splits")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol")
    parser.add_argument(
        "--split-date", type=str, required=True, help="Split date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--ratio", type=float, required=True, help="Split ratio (e.g., 10.0)"
    )
    parser.add_argument(
        "--window", type=int, default=30, help="Days before/after to analyze"
    )

    args = parser.parse_args()

    analyze_volume_around_split(
        symbol=args.symbol,
        split_date=args.split_date,
        split_ratio=args.ratio,
        window_days=args.window,
    )


if __name__ == "__main__":
    main()
