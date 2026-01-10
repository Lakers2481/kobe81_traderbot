"""
Verify Survivorship Bias (Jim Simons Standard)
===============================================

Checks if the trading universe suffers from survivorship bias:
- Does it only include stocks that survived to present day?
- Are delisted/bankrupt companies excluded from backtests?
- Is the universe static (same symbols for all periods) or dynamic (point-in-time)?

Survivorship bias can inflate backtest results by 1-2% annually!

Usage:
    python scripts/verify_survivorship_bias.py --universe data/universe/optionable_liquid_900.csv
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.universe.loader import load_universe
from core.structured_log import jlog


def check_universe_construction(universe_file: Path) -> Dict:
    """
    Analyze how the universe was constructed.

    Returns:
        Dict with universe metadata
    """
    print("\n" + "=" * 80)
    print("PART 1: Universe Construction Analysis")
    print("=" * 80)

    # Load universe
    symbols = load_universe(universe_file, cap=None)
    print(f"\nUniverse file: {universe_file}")
    print(f"Total symbols: {len(symbols)}")
    print(f"Sample symbols: {symbols[:10]}")

    # Check for .full.csv with metadata
    full_file = universe_file.with_suffix(".full.csv")
    has_metadata = full_file.exists()

    result = {
        "universe_file": str(universe_file),
        "symbol_count": len(symbols),
        "has_metadata": has_metadata,
        "symbols_sample": symbols[:20],
    }

    if has_metadata:
        print(f"\nMetadata file found: {full_file}")
        df_full = pd.read_csv(full_file)
        print(f"\nColumns: {list(df_full.columns)}")
        print(f"\nSummary stats:")
        print(df_full.describe())

        # Check earliest/latest dates
        if "earliest" in df_full.columns and "latest" in df_full.columns:
            df_full["earliest"] = pd.to_datetime(df_full["earliest"])
            df_full["latest"] = pd.to_datetime(df_full["latest"])

            earliest_overall = df_full["earliest"].min()
            latest_overall = df_full["latest"].max()

            print(f"\nData range:")
            print(f"  Earliest data: {earliest_overall.date()}")
            print(f"  Latest data: {latest_overall.date()}")

            # Check if all symbols have data to "today" (survivorship bias indicator)
            today = datetime.now().date()
            stale_symbols = df_full[
                df_full["latest"] < (datetime.now() - timedelta(days=30))
            ]

            if len(stale_symbols) == 0:
                print(
                    "\n[WARNING] All symbols have recent data (within 30 days)"
                )
                print("         This suggests SURVIVORSHIP BIAS - only survivors are included!")
            else:
                print(
                    f"\n[OK] {len(stale_symbols)} symbols have stale data (potential delisted stocks)"
                )

            result["earliest_data"] = str(earliest_overall.date())
            result["latest_data"] = str(latest_overall.date())
            result["stale_count"] = len(stale_symbols)

    return result


def check_for_known_delistings(symbols: List[str]) -> Dict:
    """
    Check if universe includes known delisted stocks.

    Returns:
        Dict with findings
    """
    print("\n" + "=" * 80)
    print("PART 2: Known Delisted Stocks Check")
    print("=" * 80)

    # Known major delistings/bankruptcies from 2015-2024
    KNOWN_DELISTINGS = {
        # Symbol: (Reason, Year)
        "GMCR": ("Acquired by JAB Holding, delisted", 2016),
        "TWX": ("Acquired by AT&T, became T", 2018),
        "VIAB": ("Merged with CBS, became VIAC then PARA", 2019),
        "TMUS": ("Survived - should be in list", 2024),  # Control (should exist)
        "GE": ("Survived but restructured", 2024),  # Control (should exist)
        "TSLA": ("Survived", 2024),  # Control (should exist)
    }

    print("\nChecking for known delisted/changed stocks:")
    print(f"Testing {len(KNOWN_DELISTINGS)} symbols...")

    found = {}
    for symbol, (reason, year) in KNOWN_DELISTINGS.items():
        is_in_universe = symbol in symbols

        status = "[FOUND]" if is_in_universe else "[MISSING]"
        print(f"  {status} {symbol:6s} - {reason} ({year})")

        found[symbol] = {
            "in_universe": is_in_universe,
            "reason": reason,
            "year": year,
        }

    # Count how many delisted stocks are missing vs present
    delisted_symbols = [
        s
        for s, (r, y) in KNOWN_DELISTINGS.items()
        if "delisted" in r.lower() or "acquired" in r.lower()
    ]
    delisted_missing = sum(1 for s in delisted_symbols if not found[s]["in_universe"])
    survivor_symbols = [
        s for s, (r, y) in KNOWN_DELISTINGS.items() if "survived" in r.lower()
    ]
    survivor_present = sum(1 for s in survivor_symbols if found[s]["in_universe"])

    print("\n" + "-" * 80)
    print(f"Delisted stocks MISSING: {delisted_missing}/{len(delisted_symbols)}")
    print(f"Survivor stocks PRESENT: {survivor_present}/{len(survivor_symbols)}")

    if delisted_missing == len(delisted_symbols):
        print(
            "\n[CRITICAL] ALL delisted stocks are missing from universe!"
        )
        print("          This is clear evidence of SURVIVORSHIP BIAS.")
    elif delisted_missing > 0:
        print(
            f"\n[WARNING] {delisted_missing} delisted stocks missing - likely survivorship bias."
        )
    else:
        print(
            "\n[OK] Delisted stocks are included - good point-in-time universe."
        )

    return {
        "tested_count": len(KNOWN_DELISTINGS),
        "delisted_missing_count": delisted_missing,
        "survivor_present_count": survivor_present,
        "details": found,
    }


def estimate_survivorship_bias_impact() -> Dict:
    """
    Estimate the impact of survivorship bias on backtest results.

    Based on academic research:
    - Survivorship bias inflates returns by 1-2% annually
    - Impact is higher for small-cap stocks
    - Impact is higher over longer periods
    """
    print("\n" + "=" * 80)
    print("PART 3: Survivorship Bias Impact Estimation")
    print("=" * 80)

    print("\nAcademic Research Findings:")
    print("  - Survivorship bias inflates returns by 1-2% annually (Elton et al., 1996)")
    print("  - Impact is higher for small-cap stocks (+2-3% annually)")
    print("  - Cumulative impact over 10 years: 10-20% overestimation")

    print("\nExample Calculations:")
    print("  Backtest Period: 2015-2024 (10 years)")
    print("  Reported WR: 65%")
    print("  Reported Annual Return: 15%")

    print("\nWith Survivorship Bias Correction:")
    print("  Actual WR: 63-64% (1-2% lower)")
    print("  Actual Annual Return: 13-14% (1-2% lower)")
    print("  Cumulative Return Impact: 10-20% overestimation")

    return {
        "annual_bias_pct": 1.5,
        "wr_bias_pct": 1.5,
        "cumulative_10yr_bias_pct": 15.0,
    }


def recommend_mitigation_strategies():
    """Print mitigation strategies for survivorship bias."""
    print("\n" + "=" * 80)
    print("PART 4: Mitigation Strategies")
    print("=" * 80)

    print("\n[1] Accept the Bias (Current State)")
    print("    - Acknowledge 1-2% annual return overestimation")
    print("    - Be conservative in live trading expectations")
    print("    - Treat backtest results as upper bound")
    print("    - PRO: Simple, no code changes")
    print("    - CON: Less accurate backtest results")

    print("\n[2] Use Point-in-Time Universe")
    print("    - Reconstruct universe at each backtest date")
    print("    - Include stocks that were tradeable THEN (even if delisted now)")
    print("    - Requires historical delisting data")
    print("    - PRO: Eliminates survivorship bias")
    print("    - CON: Requires significant engineering effort")

    print("\n[3] Apply Survivorship Adjustment Factor")
    print("    - Subtract 1-2% from backtest annual returns")
    print("    - Reduce backtest WR by 1-2 percentage points")
    print("    - Document adjustment in reports")
    print("    - PRO: Easy to implement, academically supported")
    print("    - CON: Approximation, not exact")

    print("\n[4] Use Only Recent Data")
    print("    - Backtest only 2020-2024 (short period)")
    print("    - Fewer delistings in recent years")
    print("    - PRO: Reduces bias impact")
    print("    - CON: Less data for validation")

    print("\nRECOMMENDED: Strategy #3 (Survivorship Adjustment Factor)")
    print("  - Document: 'Backtest results may overestimate by 1-2% annually due to survivorship bias'")
    print("  - Adjust targets: If backtest shows 65% WR, expect 63-64% in live trading")
    print("  - Apply 20% discount to cumulative returns over 10 years")


def main():
    parser = argparse.ArgumentParser(
        description="Verify survivorship bias (Jim Simons standard)"
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="data/universe/optionable_liquid_900.csv",
        help="Universe CSV file",
    )

    args = parser.parse_args()

    universe_file = Path(args.universe)
    if not universe_file.exists():
        print(f"ERROR: Universe file not found: {universe_file}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("SURVIVORSHIP BIAS VERIFICATION")
    print("Jim Simons / Renaissance Technologies Standard")
    print("=" * 80)
    print(f"\nTarget Universe: {universe_file}")
    print(f"Date: {datetime.now().date()}")

    # Load symbols
    symbols = load_universe(universe_file, cap=None)

    # Run checks
    universe_info = check_universe_construction(universe_file)
    delisting_check = check_for_known_delistings(symbols)
    impact_estimate = estimate_survivorship_bias_impact()
    recommend_mitigation_strategies()

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    stale_count = universe_info.get("stale_count", 0)
    delisted_missing = delisting_check["delisted_missing_count"]

    if stale_count == 0 and delisted_missing > 0:
        verdict = "[FAIL] SURVIVORSHIP BIAS DETECTED"
        print(f"\n{verdict}")
        print("\nEvidence:")
        print(f"  - All {universe_info['symbol_count']} symbols have recent data")
        print(
            f"  - {delisted_missing} known delisted stocks are missing from universe"
        )
        print("  - Universe appears to be constructed from 'survivors only'")
        print("\nImpact:")
        print("  - Backtest results likely OVERESTIMATE by 1-2% annually")
        print("  - Win rate likely OVERESTIMATED by 1-2 percentage points")
        print("\nRecommendation:")
        print("  - Document this limitation")
        print("  - Apply -1.5% annual adjustment to backtest returns")
        print("  - Expect lower performance in live trading")

        sys.exit(1)  # Fail exit code

    elif stale_count > 0:
        print("\n[WARN] Possible survivorship bias")
        print(f"  - {stale_count} symbols with stale data (good!)")
        print("  - But some delisted stocks still missing")
        print("\nRecommendation: Review universe construction method")

        sys.exit(0)  # Pass with warning

    else:
        print("\n[OK] No obvious survivorship bias detected")
        print("  - Universe includes non-survivors")
        print("  - Point-in-time methodology appears sound")

        sys.exit(0)  # Pass


if __name__ == "__main__":
    main()
