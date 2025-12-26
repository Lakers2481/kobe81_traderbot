#!/usr/bin/env python3
"""
Universe Management - Kobe Trading System

Manages the trading universe: list symbols, show statistics,
add/remove symbols, and validate data history.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.env_loader import load_env
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon


# Sector mappings for common symbols (partial list for demonstration)
# In production, this would come from an API or more complete mapping file
SECTOR_MAPPINGS = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "GOOG": "Technology",
    "META": "Technology", "AMZN": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "INTC": "Technology", "CRM": "Technology", "ORCL": "Technology", "ADBE": "Technology",
    "CSCO": "Technology", "IBM": "Technology", "NOW": "Technology", "AVGO": "Technology",
    "QCOM": "Technology", "TXN": "Technology", "MU": "Technology", "AMAT": "Technology",
    "LRCX": "Technology", "KLAC": "Technology", "MRVL": "Technology", "ASML": "Technology",
    "TSM": "Technology", "PLTR": "Technology", "SMCI": "Technology",
    # Financial
    "JPM": "Financial", "BAC": "Financial", "WFC": "Financial", "GS": "Financial",
    "MS": "Financial", "C": "Financial", "BLK": "Financial", "SCHW": "Financial",
    "AXP": "Financial", "V": "Financial", "MA": "Financial", "COF": "Financial",
    "BRK.B": "Financial", "USB": "Financial", "PNC": "Financial",
    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "PFE": "Healthcare", "MRK": "Healthcare",
    "ABBV": "Healthcare", "LLY": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    "DHR": "Healthcare", "BMY": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare",
    "CVS": "Healthcare", "CI": "Healthcare", "HUM": "Healthcare",
    # Consumer
    "WMT": "Consumer", "COST": "Consumer", "HD": "Consumer", "TGT": "Consumer",
    "LOW": "Consumer", "NKE": "Consumer", "SBUX": "Consumer", "MCD": "Consumer",
    "KO": "Consumer", "PEP": "Consumer", "PG": "Consumer", "PM": "Consumer",
    "NFLX": "Consumer", "DIS": "Consumer",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "PXD": "Energy", "MPC": "Energy", "VLO": "Energy",
    "PSX": "Energy", "OXY": "Energy",
    # Industrial
    "BA": "Industrial", "CAT": "Industrial", "DE": "Industrial", "HON": "Industrial",
    "UNP": "Industrial", "RTX": "Industrial", "LMT": "Industrial", "GE": "Industrial",
    "MMM": "Industrial", "UPS": "Industrial", "FDX": "Industrial",
    # Telecom
    "T": "Telecom", "VZ": "Telecom", "TMUS": "Telecom",
    # Automotive
    "TSLA": "Automotive", "F": "Automotive", "GM": "Automotive", "RIVN": "Automotive",
    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate", "SPG": "Real Estate",
    "EQIX": "Real Estate", "O": "Real Estate",
    # ETFs
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF", "TLT": "ETF",
    "GLD": "ETF", "HYG": "ETF", "LQD": "ETF", "XLF": "ETF", "XLE": "ETF",
    "XLK": "ETF", "XLV": "ETF", "TQQQ": "ETF", "SOXL": "ETF", "FXI": "ETF",
    "SMH": "ETF", "EEM": "ETF", "VTI": "ETF", "VOO": "ETF", "ARKK": "ETF",
    # Crypto-related
    "MSTR": "Crypto", "COIN": "Crypto",
    # Chinese ADRs
    "BABA": "China ADR", "JD": "China ADR", "PDD": "China ADR", "NIO": "China ADR",
    "BIDU": "China ADR", "LI": "China ADR", "XPEV": "China ADR",
}


def get_sector(symbol: str) -> str:
    """Get sector for a symbol, defaulting to 'Other' if unknown."""
    return SECTOR_MAPPINGS.get(symbol.upper(), "Other")


def load_universe_with_details(path: Path) -> Tuple[List[str], Optional[pd.DataFrame]]:
    """
    Load universe from CSV, returning both symbol list and full dataframe if available.
    """
    if not path.exists():
        return [], None

    df = pd.read_csv(path)
    symbols = load_universe(path)

    # Check if we have a .full.csv version with more details
    full_path = path.with_suffix(".full.csv")
    if full_path.exists():
        try:
            full_df = pd.read_csv(full_path)
            return symbols, full_df
        except Exception:
            pass

    return symbols, df if len(df.columns) > 1 else None


def get_universe_stats(symbols: List[str], details_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Calculate universe statistics.
    """
    stats = {
        "total_symbols": len(symbols),
        "sectors": Counter(),
        "etf_count": 0,
        "stock_count": 0,
    }

    for sym in symbols:
        sector = get_sector(sym)
        stats["sectors"][sector] += 1
        if sector == "ETF":
            stats["etf_count"] += 1
        else:
            stats["stock_count"] += 1

    # If we have details, add more stats
    if details_df is not None and not details_df.empty:
        if "adv_dollar" in details_df.columns:
            stats["total_adv_dollar"] = details_df["adv_dollar"].sum()
            stats["avg_adv_dollar"] = details_df["adv_dollar"].mean()
            stats["median_adv_dollar"] = details_df["adv_dollar"].median()

            # ADV distribution buckets
            adv = details_df["adv_dollar"]
            stats["adv_distribution"] = {
                ">$1B": int((adv >= 1e9).sum()),
                "$500M-$1B": int(((adv >= 5e8) & (adv < 1e9)).sum()),
                "$100M-$500M": int(((adv >= 1e8) & (adv < 5e8)).sum()),
                "$50M-$100M": int(((adv >= 5e7) & (adv < 1e8)).sum()),
                "<$50M": int((adv < 5e7).sum()),
            }

        if "years" in details_df.columns:
            stats["avg_history_years"] = details_df["years"].mean()
            stats["min_history_years"] = details_df["years"].min()
            stats["max_history_years"] = details_df["years"].max()

        if "has_options" in details_df.columns:
            stats["optionable_count"] = int(details_df["has_options"].sum())

    return stats


def validate_symbol_history(
    symbols: List[str],
    cache_dir: Path,
    min_years: float = 10.0,
    sample_size: int = 0,
) -> Dict[str, Any]:
    """
    Validate that symbols have sufficient data history.
    """
    import random

    if sample_size > 0 and sample_size < len(symbols):
        check_symbols = random.sample(symbols, sample_size)
    else:
        check_symbols = symbols

    results = {
        "total": len(symbols),
        "checked": len(check_symbols),
        "valid": 0,
        "insufficient": 0,
        "no_data": 0,
        "errors": 0,
        "insufficient_symbols": [],
        "no_data_symbols": [],
    }

    end_date = datetime.now().date().isoformat()
    start_date = "2010-01-01"

    for i, sym in enumerate(check_symbols, 1):
        if i % 50 == 0:
            print(f"  Validated {i}/{len(check_symbols)}...")

        try:
            df = fetch_daily_bars_polygon(sym, start_date, end_date, cache_dir=cache_dir)

            if df.empty:
                results["no_data"] += 1
                results["no_data_symbols"].append(sym)
            else:
                df = df.sort_values("timestamp")
                earliest = pd.to_datetime(df.iloc[0]["timestamp"]).date()
                latest = pd.to_datetime(df.iloc[-1]["timestamp"]).date()
                years = (latest - earliest).days / 365.25

                if years >= min_years:
                    results["valid"] += 1
                else:
                    results["insufficient"] += 1
                    results["insufficient_symbols"].append((sym, round(years, 2)))
        except Exception as e:
            results["errors"] += 1
            results["no_data_symbols"].append(f"{sym} (error)")

    return results


def add_symbols(universe_path: Path, symbols_to_add: List[str]) -> Tuple[int, List[str]]:
    """
    Add symbols to the universe CSV.
    Returns (added_count, already_existed).
    """
    existing = load_universe(universe_path) if universe_path.exists() else []
    existing_set = set(s.upper() for s in existing)

    added = []
    already_existed = []

    for sym in symbols_to_add:
        sym_upper = sym.upper().strip()
        if not sym_upper:
            continue
        if sym_upper in existing_set:
            already_existed.append(sym_upper)
        else:
            added.append(sym_upper)
            existing_set.add(sym_upper)

    if added:
        # Append to file
        new_symbols = existing + added
        df = pd.DataFrame({"symbol": new_symbols})
        universe_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(universe_path, index=False)

    return len(added), already_existed


def remove_symbols(universe_path: Path, symbols_to_remove: List[str]) -> Tuple[int, List[str]]:
    """
    Remove symbols from the universe CSV.
    Returns (removed_count, not_found).
    """
    if not universe_path.exists():
        return 0, symbols_to_remove

    existing = load_universe(universe_path)
    remove_set = set(s.upper().strip() for s in symbols_to_remove if s.strip())

    removed = []
    not_found = []
    remaining = []

    for sym in existing:
        if sym.upper() in remove_set:
            removed.append(sym)
        else:
            remaining.append(sym)

    for sym in remove_set:
        if sym not in [r.upper() for r in removed]:
            not_found.append(sym)

    if removed:
        df = pd.DataFrame({"symbol": remaining})
        df.to_csv(universe_path, index=False)

    return len(removed), not_found


def format_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Simple table formatter."""
    if not rows:
        return "No data."

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in col_widths)
    data_lines = [" | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)) for row in rows]

    return "\n".join([header_line, separator] + data_lines)


def format_number(n: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1e9:
        return f"${n/1e9:.1f}B"
    elif n >= 1e6:
        return f"${n/1e6:.1f}M"
    elif n >= 1e3:
        return f"${n/1e3:.1f}K"
    else:
        return f"${n:.0f}"


def main():
    ap = argparse.ArgumentParser(
        description="Universe Management - Kobe Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/universe.py --list
  python scripts/universe.py --stats
  python scripts/universe.py --validate --sample 50
  python scripts/universe.py --add AAPL,MSFT,GOOGL
  python scripts/universe.py --remove XYZ,ABC
        """,
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file",
    )
    ap.add_argument(
        "--universe",
        type=str,
        default="data/universe/optionable_liquid_final.csv",
        help="Universe CSV file",
    )
    ap.add_argument(
        "--cache",
        type=str,
        default="data/cache",
        help="Cache directory path",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List all symbols in universe",
    )
    ap.add_argument(
        "--stats",
        action="store_true",
        help="Show universe statistics",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Validate all symbols have sufficient history",
    )
    ap.add_argument(
        "--add",
        type=str,
        default=None,
        help="Add symbols (comma-separated)",
    )
    ap.add_argument(
        "--remove",
        type=str,
        default=None,
        help="Remove symbols (comma-separated)",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample size for validation (0 = all)",
    )
    ap.add_argument(
        "--min-years",
        type=float,
        default=10.0,
        help="Minimum years of history required",
    )
    ap.add_argument(
        "--columns",
        type=int,
        default=10,
        help="Number of columns for symbol list display",
    )
    args = ap.parse_args()

    # Load environment
    dotenv_path = Path(args.dotenv)
    if dotenv_path.exists():
        loaded = load_env(dotenv_path)
        print(f"Loaded {len(loaded)} env vars from {dotenv_path}")

    universe_path = Path(args.universe)
    if not universe_path.is_absolute():
        universe_path = ROOT / universe_path

    cache_dir = Path(args.cache)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir

    # Determine what to show
    show_default = not (args.list or args.stats or args.validate or args.add or args.remove)

    # Add symbols
    if args.add:
        print("\n" + "=" * 60)
        print("ADD SYMBOLS")
        print("=" * 60)

        symbols_to_add = [s.strip() for s in args.add.split(",") if s.strip()]
        if symbols_to_add:
            added, existed = add_symbols(universe_path, symbols_to_add)
            print(f"Added: {added} symbols")
            if existed:
                print(f"Already existed: {', '.join(existed)}")
        else:
            print("No valid symbols provided.")

    # Remove symbols
    if args.remove:
        print("\n" + "=" * 60)
        print("REMOVE SYMBOLS")
        print("=" * 60)

        symbols_to_remove = [s.strip() for s in args.remove.split(",") if s.strip()]
        if symbols_to_remove:
            removed, not_found = remove_symbols(universe_path, symbols_to_remove)
            print(f"Removed: {removed} symbols")
            if not_found:
                print(f"Not found: {', '.join(not_found)}")
        else:
            print("No valid symbols provided.")

    # Load universe
    if not universe_path.exists():
        print(f"\nUniverse file not found: {universe_path}")
        if show_default or args.list or args.stats or args.validate:
            return

    symbols, details_df = load_universe_with_details(universe_path)

    # List symbols
    if args.list or show_default:
        print("\n" + "=" * 60)
        print("UNIVERSE SYMBOLS")
        print("=" * 60)
        print(f"File: {universe_path}")
        print(f"Total: {len(symbols)} symbols\n")

        # Display in columns
        cols = args.columns
        rows = []
        for i in range(0, len(symbols), cols):
            row = symbols[i : i + cols]
            # Pad row to full width
            while len(row) < cols:
                row.append("")
            rows.append(row)

        # Calculate column widths
        col_width = max(len(s) for s in symbols) + 2
        for row in rows:
            print(" ".join(s.ljust(col_width) for s in row))

    # Show statistics
    if args.stats or show_default:
        print("\n" + "=" * 60)
        print("UNIVERSE STATISTICS")
        print("=" * 60)

        stats = get_universe_stats(symbols, details_df)

        print(f"Total Symbols:    {stats['total_symbols']}")
        print(f"Stocks:           {stats['stock_count']}")
        print(f"ETFs:             {stats['etf_count']}")

        if "optionable_count" in stats:
            print(f"Optionable:       {stats['optionable_count']}")

        print("\nSector Distribution:")
        print("-" * 40)
        sector_counts = sorted(stats["sectors"].items(), key=lambda x: -x[1])
        for sector, count in sector_counts:
            pct = 100 * count / stats["total_symbols"]
            bar = "#" * int(pct / 2)
            print(f"  {sector:15s} {count:4d} ({pct:5.1f}%) {bar}")

        if "adv_distribution" in stats:
            print("\nADV (Dollar Volume) Distribution:")
            print("-" * 40)
            for bucket, count in stats["adv_distribution"].items():
                pct = 100 * count / stats["total_symbols"]
                print(f"  {bucket:15s} {count:4d} ({pct:5.1f}%)")

            print(f"\n  Total ADV:      {format_number(stats['total_adv_dollar'])}")
            print(f"  Average ADV:    {format_number(stats['avg_adv_dollar'])}")
            print(f"  Median ADV:     {format_number(stats['median_adv_dollar'])}")

        if "avg_history_years" in stats:
            print("\nHistory Statistics:")
            print("-" * 40)
            print(f"  Average:        {stats['avg_history_years']:.1f} years")
            print(f"  Minimum:        {stats['min_history_years']:.1f} years")
            print(f"  Maximum:        {stats['max_history_years']:.1f} years")

    # Validate history
    if args.validate:
        print("\n" + "=" * 60)
        print("HISTORY VALIDATION")
        print("=" * 60)
        print(f"Minimum Required: {args.min_years} years")
        print(f"Sample Size: {args.sample if args.sample > 0 else 'ALL'}")
        print()

        results = validate_symbol_history(
            symbols, cache_dir, min_years=args.min_years, sample_size=args.sample
        )

        print("\nValidation Results:")
        print("-" * 40)
        print(f"Checked:          {results['checked']}/{results['total']}")
        print(f"Valid:            {results['valid']} ({100*results['valid']/results['checked']:.1f}%)")
        print(f"Insufficient:     {results['insufficient']}")
        print(f"No Data:          {results['no_data']}")
        print(f"Errors:           {results['errors']}")

        if results["insufficient_symbols"]:
            print("\nSymbols with insufficient history:")
            for sym, years in results["insufficient_symbols"][:20]:
                print(f"  {sym}: {years} years")
            if len(results["insufficient_symbols"]) > 20:
                print(f"  ... and {len(results['insufficient_symbols']) - 20} more")

        if results["no_data_symbols"]:
            print("\nSymbols with no data:")
            for sym in results["no_data_symbols"][:20]:
                print(f"  {sym}")
            if len(results["no_data_symbols"]) > 20:
                print(f"  ... and {len(results['no_data_symbols']) - 20} more")

        # Summary
        target = 950
        if results["valid"] >= target:
            print(f"\n[PASS] >= {target} symbols have sufficient history")
        else:
            print(f"\n[FAIL] Only {results['valid']} symbols have sufficient history (need {target})")

    print("\n" + "=" * 60)
    print("Universe management complete.")


if __name__ == "__main__":
    main()
