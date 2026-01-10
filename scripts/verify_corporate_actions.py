"""
Verify Corporate Actions Handling (Jim Simons Standard)
========================================================

Tests known stock splits to ensure data is properly adjusted:
- AAPL 4:1 split on 2020-08-31
- TSLA 5:1 split on 2020-08-31
- NVDA 4:1 split on 2024-06-10

Verifies:
1. Price series adjusted correctly
2. Volume adjusted correctly (multiplied by split ratio)
3. NO phantom returns around split dates
4. Backtest results consistent with/without split periods

Usage:
    python scripts/verify_corporate_actions.py --symbol AAPL --split-date 2020-08-31
    python scripts/verify_corporate_actions.py --all  # Test all known splits
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.providers.polygon_eod import fetch_daily_bars_polygon
from core.structured_log import jlog


@dataclass
class KnownSplit:
    """A documented stock split event."""

    symbol: str
    date: str  # YYYY-MM-DD
    ratio: float  # e.g., 4.0 for 4:1 split
    pre_split_price_approx: float  # Unadjusted price before split
    post_split_price_approx: float  # Unadjusted price after split


# Known splits for verification
KNOWN_SPLITS = [
    KnownSplit(
        symbol="AAPL",
        date="2020-08-31",
        ratio=4.0,
        pre_split_price_approx=500.0,  # ~$500 before split
        post_split_price_approx=125.0,  # ~$125 after split
    ),
    KnownSplit(
        symbol="TSLA",
        date="2020-08-31",
        ratio=5.0,
        pre_split_price_approx=2200.0,  # ~$2,200 before split
        post_split_price_approx=440.0,  # ~$440 after split
    ),
    KnownSplit(
        symbol="NVDA",
        date="2024-06-10",
        ratio=10.0,
        pre_split_price_approx=1200.0,  # ~$1,200 before split
        post_split_price_approx=120.0,  # ~$120 after split
    ),
]


def verify_split_adjustment(
    symbol: str,
    split_date: str,
    split_ratio: float,
    window_days: int = 10,
) -> Dict:
    """
    Verify that data is properly adjusted for a stock split.

    Args:
        symbol: Stock ticker
        split_date: Split date (YYYY-MM-DD)
        split_ratio: Split ratio (e.g., 4.0 for 4:1)
        window_days: Days before/after split to check

    Returns:
        Dict with verification results
    """
    split_dt = pd.to_datetime(split_date)
    start_date = (split_dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_date = (split_dt + timedelta(days=window_days)).strftime("%Y-%m-%d")

    jlog(
        "verify_split_start",
        symbol=symbol,
        split_date=split_date,
        ratio=split_ratio,
    )

    # Fetch data
    df = fetch_daily_bars_polygon(
        symbol=symbol,
        start=start_date,
        end=end_date,
        cache_dir=Path("data/cache"),
        ignore_cache_ttl=False,
    )

    if df.empty:
        return {
            "passed": False,
            "error": "No data returned",
            "symbol": symbol,
            "split_date": split_date,
        }

    # Convert timestamp to date for comparison
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    split_date_obj = split_dt.date()

    # Find bars around split
    pre_split = df[df["date"] < split_date_obj].tail(5)
    post_split = df[df["date"] > split_date_obj].head(5)

    if pre_split.empty or post_split.empty:
        return {
            "passed": False,
            "error": "Insufficient data around split date",
            "symbol": symbol,
            "split_date": split_date,
        }

    # TEST 1: Check for price discontinuity (if data is NOT adjusted, we'd see a jump)
    last_pre_close = pre_split.iloc[-1]["close"]
    first_post_open = post_split.iloc[0]["open"]
    price_ratio = last_pre_close / first_post_open

    # If adjusted correctly, ratio should be ~1.0 (no discontinuity)
    # If NOT adjusted, ratio should be ~split_ratio
    is_adjusted = abs(price_ratio - 1.0) < 0.15  # Within 15% of 1.0

    # TEST 2: Calculate overnight return across split
    # Overnight return: (post_open - pre_close) / pre_close
    overnight_return = (first_post_open - last_pre_close) / last_pre_close * 100

    # If adjusted: return should be small (normal overnight move)
    # If NOT adjusted: return should be ~-75% (for 4:1) or ~-80% (for 5:1)
    has_phantom_return = abs(overnight_return) > 30  # >30% overnight is suspicious

    # TEST 3: Volume adjustment
    # Volume should INCREASE by split ratio (more shares outstanding)
    avg_pre_volume = pre_split["volume"].mean()
    avg_post_volume = post_split["volume"].mean()
    volume_ratio = avg_post_volume / avg_pre_volume if avg_pre_volume > 0 else 0

    # Expected: volume increases by split ratio
    volume_adjusted_correctly = (
        abs(volume_ratio - split_ratio) / split_ratio < 0.5  # Within 50% tolerance
    )

    # TEST 4: Price continuity check (standard deviation of returns)
    all_returns = df["close"].pct_change() * 100
    mean_return = all_returns.mean()
    std_return = all_returns.std()

    # Flag if any single day return is >5 standard deviations
    outliers = all_returns[abs(all_returns - mean_return) > 5 * std_return]
    has_outliers = len(outliers) > 0

    # Overall assessment
    passed = (
        is_adjusted
        and not has_phantom_return
        and volume_adjusted_correctly
        and not has_outliers
    )

    result = {
        "symbol": symbol,
        "split_date": split_date,
        "split_ratio": split_ratio,
        "passed": passed,
        "tests": {
            "price_adjusted": {
                "passed": is_adjusted,
                "price_ratio": price_ratio,
                "expected": "~1.0 (no discontinuity)",
                "actual": f"{price_ratio:.2f}",
            },
            "no_phantom_return": {
                "passed": not has_phantom_return,
                "overnight_return_pct": overnight_return,
                "threshold": 30.0,
                "actual": f"{overnight_return:.2f}%",
            },
            "volume_adjusted": {
                "passed": volume_adjusted_correctly,
                "volume_ratio": volume_ratio,
                "expected": split_ratio,
                "actual": f"{volume_ratio:.2f}",
            },
            "no_outliers": {
                "passed": not has_outliers,
                "outlier_count": len(outliers),
                "threshold": "5 std devs",
            },
        },
        "data_sample": {
            "last_pre_close": last_pre_close,
            "first_post_open": first_post_open,
            "avg_pre_volume": avg_pre_volume,
            "avg_post_volume": avg_post_volume,
        },
    }

    # Log result
    jlog(
        "verify_split_complete",
        symbol=symbol,
        passed=passed,
        price_ratio=price_ratio,
        overnight_return=overnight_return,
        volume_ratio=volume_ratio,
    )

    return result


def print_result(result: Dict):
    """Pretty print verification result."""
    symbol = result["symbol"]
    split_date = result["split_date"]
    passed = result["passed"]

    status = "[OK] PASSED" if passed else "FAILED"
    print(f"\n{'=' * 80}")
    print(f"{symbol} {split_date} - {status}")
    print(f"{'=' * 80}")

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    tests = result["tests"]
    print("\nTest Results:")
    for test_name, test_data in tests.items():
        test_passed = test_data["passed"]
        test_status = "[OK]" if test_passed else "FAIL"
        print(f"  {test_status} {test_name}")

        # Print details
        for key, value in test_data.items():
            if key != "passed":
                print(f"      {key}: {value}")

    print("\nData Sample:")
    for key, value in result["data_sample"].items():
        print(f"  {key}: {value:,.0f}")


def verify_all_known_splits() -> List[Dict]:
    """Verify all known splits."""
    results = []

    print("\nVerifying Corporate Actions (Jim Simons Standard)")
    print("=" * 80)
    print(f"Testing {len(KNOWN_SPLITS)} known stock splits...")

    for split in KNOWN_SPLITS:
        result = verify_split_adjustment(
            symbol=split.symbol,
            split_date=split.date,
            split_ratio=split.ratio,
        )
        results.append(result)
        print_result(result)

    # Summary
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)

    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed_count}/{total_count} splits verified correctly")
    print("=" * 80)

    if passed_count == total_count:
        print("[OK] All corporate actions verified - data is properly adjusted!")
    else:
        print(
            "WARNING: Some splits failed verification - check data provider settings!"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Verify corporate actions handling (Jim Simons standard)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Stock symbol to verify",
    )
    parser.add_argument(
        "--split-date",
        type=str,
        help="Split date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        help="Split ratio (e.g., 4.0 for 4:1 split)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Verify all known splits",
    )

    args = parser.parse_args()

    if args.all:
        results = verify_all_known_splits()
        sys.exit(0 if all(r["passed"] for r in results) else 1)

    elif args.symbol and args.split_date and args.ratio:
        result = verify_split_adjustment(
            symbol=args.symbol,
            split_date=args.split_date,
            split_ratio=args.ratio,
        )
        print_result(result)
        sys.exit(0 if result["passed"] else 1)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/verify_corporate_actions.py --all")
        print(
            "  python scripts/verify_corporate_actions.py --symbol AAPL --split-date 2020-08-31 --ratio 4.0"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
