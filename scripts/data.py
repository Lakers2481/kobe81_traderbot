#!/usr/bin/env python3
"""
Data Status Dashboard - Kobe Trading System

Shows cache statistics, data freshness per symbol, API rate limit status,
and lists symbols with stale/missing data.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from data.universe.loader import load_universe


def get_cache_files(cache_dir: Path) -> List[Path]:
    """Get all CSV cache files from the cache directory."""
    if not cache_dir.exists():
        return []
    return list(cache_dir.glob("*.csv"))


def get_cache_stats(cache_dir: Path) -> Dict[str, Any]:
    """
    Calculate cache statistics: total files, total size, oldest/newest files.
    """
    files = get_cache_files(cache_dir)
    if not files:
        return {
            "total_files": 0,
            "total_size_mb": 0.0,
            "oldest_file": None,
            "newest_file": None,
            "oldest_age_days": None,
            "newest_age_days": None,
        }

    total_size = sum(f.stat().st_size for f in files)
    mtimes = [(f, f.stat().st_mtime) for f in files]
    mtimes.sort(key=lambda x: x[1])

    oldest = mtimes[0]
    newest = mtimes[-1]
    now = datetime.now().timestamp()

    return {
        "total_files": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "oldest_file": oldest[0].name,
        "newest_file": newest[0].name,
        "oldest_age_days": round((now - oldest[1]) / 86400, 1),
        "newest_age_days": round((now - newest[1]) / 86400, 1),
    }


def parse_cache_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse cache filename pattern: SYMBOL_STARTDATE_ENDDATE.csv
    Returns dict with symbol, start, end or None if pattern doesn't match.
    """
    if not filename.endswith(".csv"):
        return None
    base = filename[:-4]
    parts = base.rsplit("_", 2)
    if len(parts) != 3:
        return None
    return {"symbol": parts[0], "start": parts[1], "end": parts[2]}


def get_symbol_freshness(cache_dir: Path, symbols: List[str]) -> pd.DataFrame:
    """
    Check data freshness for each symbol in the universe.
    Returns DataFrame with symbol, has_cache, latest_cache_date, file_age_days, is_stale.
    """
    files = get_cache_files(cache_dir)
    now = datetime.now()
    today = now.date()

    # Build lookup: symbol -> list of (end_date, file_mtime, file_path)
    symbol_files: Dict[str, List[tuple]] = {}
    for f in files:
        parsed = parse_cache_filename(f.name)
        if parsed:
            sym = parsed["symbol"].upper()
            if sym not in symbol_files:
                symbol_files[sym] = []
            symbol_files[sym].append((parsed["end"], f.stat().st_mtime, f))

    rows = []
    for sym in symbols:
        sym_upper = sym.upper()
        if sym_upper not in symbol_files:
            rows.append({
                "symbol": sym_upper,
                "has_cache": False,
                "latest_cache_date": None,
                "file_age_days": None,
                "is_stale": True,
                "status": "MISSING",
            })
        else:
            # Find the most recent cache file for this symbol
            entries = symbol_files[sym_upper]
            entries.sort(key=lambda x: x[0], reverse=True)
            latest_end, mtime, fpath = entries[0]
            file_age_days = (now.timestamp() - mtime) / 86400

            try:
                cache_date = datetime.strptime(latest_end, "%Y-%m-%d").date()
                days_behind = (today - cache_date).days
                # Consider stale if cache is more than 3 days old (allowing for weekends)
                is_stale = days_behind > 3
            except ValueError:
                cache_date = None
                days_behind = None
                is_stale = True

            status = "OK" if not is_stale else "STALE"
            rows.append({
                "symbol": sym_upper,
                "has_cache": True,
                "latest_cache_date": latest_end,
                "file_age_days": round(file_age_days, 1),
                "days_behind": days_behind,
                "is_stale": is_stale,
                "status": status,
            })

    return pd.DataFrame(rows)


def check_api_rate_limits() -> Dict[str, Any]:
    """
    Check Polygon API rate limit status.
    Note: Polygon's rate limits depend on subscription tier.
    Free tier: 5 req/min, Starter: 100 req/min, etc.
    """
    import requests

    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return {
            "status": "NO_API_KEY",
            "message": "POLYGON_API_KEY not set",
        }

    # Make a lightweight API call to check connectivity and get rate limit headers
    try:
        url = "https://api.polygon.io/v3/reference/tickers"
        params = {"limit": 1, "apiKey": api_key}
        resp = requests.get(url, params=params, timeout=10)

        # Polygon returns rate limit info in headers
        headers = resp.headers
        rate_limit = headers.get("X-RateLimit-Limit", "N/A")
        rate_remaining = headers.get("X-RateLimit-Remaining", "N/A")
        rate_reset = headers.get("X-RateLimit-Reset", "N/A")

        if resp.status_code == 200:
            return {
                "status": "OK",
                "http_status": resp.status_code,
                "rate_limit": rate_limit,
                "rate_remaining": rate_remaining,
                "rate_reset": rate_reset,
            }
        elif resp.status_code == 401:
            return {
                "status": "INVALID_API_KEY",
                "http_status": resp.status_code,
                "message": "API key is invalid or expired",
            }
        elif resp.status_code == 429:
            return {
                "status": "RATE_LIMITED",
                "http_status": resp.status_code,
                "rate_limit": rate_limit,
                "rate_remaining": rate_remaining,
                "rate_reset": rate_reset,
            }
        else:
            return {
                "status": "ERROR",
                "http_status": resp.status_code,
                "message": resp.text[:200],
            }
    except requests.exceptions.RequestException as e:
        return {
            "status": "CONNECTION_ERROR",
            "message": str(e),
        }


def format_table(headers: List[str], rows: List[List[Any]], max_width: int = 80) -> str:
    """Simple table formatter without external dependencies."""
    if not rows:
        return "No data to display."

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    # Format header
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in col_widths)

    # Format rows
    data_lines = []
    for row in rows:
        line = " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row))
        data_lines.append(line)

    return "\n".join([header_line, separator] + data_lines)


def main():
    ap = argparse.ArgumentParser(
        description="Kobe Data Status Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/data.py --cache-stats
  python scripts/data.py --check-freshness --universe data/universe/optionable_liquid_800.csv
  python scripts/data.py --check-freshness --cache-stats --show-stale
        """,
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file",
    )
    ap.add_argument(
        "--cache",
        type=str,
        default="data/cache",
        help="Cache directory path",
    )
    ap.add_argument(
        "--universe",
        type=str,
        default="data/universe/optionable_liquid_800.csv",
        help="Universe CSV file",
    )
    ap.add_argument(
        "--cache-stats",
        action="store_true",
        help="Show cache statistics",
    )
    ap.add_argument(
        "--check-freshness",
        action="store_true",
        help="Check data freshness per symbol",
    )
    ap.add_argument(
        "--api-status",
        action="store_true",
        help="Show API rate limit status",
    )
    ap.add_argument(
        "--show-stale",
        action="store_true",
        help="Only show stale/missing symbols (with --check-freshness)",
    )
    ap.add_argument(
        "--show-all",
        action="store_true",
        help="Show all symbols (with --check-freshness)",
    )
    args = ap.parse_args()

    # Load environment
    dotenv_path = Path(args.dotenv)
    if dotenv_path.exists():
        loaded = load_env(dotenv_path)
        print(f"Loaded {len(loaded)} env vars from {dotenv_path}")
    else:
        print(f"Warning: .env file not found at {dotenv_path}")

    cache_dir = Path(args.cache)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir

    universe_path = Path(args.universe)
    if not universe_path.is_absolute():
        universe_path = ROOT / universe_path

    # If no specific flags, show all
    show_all = not (args.cache_stats or args.check_freshness or args.api_status)

    # Cache Statistics
    if args.cache_stats or show_all:
        print("\n" + "=" * 60)
        print("CACHE STATISTICS")
        print("=" * 60)

        if not cache_dir.exists():
            print(f"Cache directory does not exist: {cache_dir}")
        else:
            stats = get_cache_stats(cache_dir)
            print(f"Cache Directory: {cache_dir}")
            print(f"Total Files:     {stats['total_files']}")
            print(f"Total Size:      {stats['total_size_mb']} MB")
            if stats["oldest_file"]:
                print(f"Oldest File:     {stats['oldest_file']} ({stats['oldest_age_days']} days ago)")
                print(f"Newest File:     {stats['newest_file']} ({stats['newest_age_days']} days ago)")

    # API Rate Limit Status
    if args.api_status or show_all:
        print("\n" + "=" * 60)
        print("API RATE LIMIT STATUS")
        print("=" * 60)

        rate_info = check_api_rate_limits()
        print(f"Status:          {rate_info['status']}")
        if "http_status" in rate_info:
            print(f"HTTP Status:     {rate_info['http_status']}")
        if "rate_limit" in rate_info:
            print(f"Rate Limit:      {rate_info['rate_limit']}")
            print(f"Rate Remaining:  {rate_info['rate_remaining']}")
            print(f"Rate Reset:      {rate_info['rate_reset']}")
        if "message" in rate_info:
            print(f"Message:         {rate_info['message']}")

    # Data Freshness Check
    if args.check_freshness or show_all:
        print("\n" + "=" * 60)
        print("DATA FRESHNESS CHECK")
        print("=" * 60)

        if not universe_path.exists():
            print(f"Universe file not found: {universe_path}")
        else:
            symbols = load_universe(universe_path)
            print(f"Universe: {universe_path} ({len(symbols)} symbols)")

            freshness_df = get_symbol_freshness(cache_dir, symbols)

            # Summary stats
            total = len(freshness_df)
            missing = (freshness_df["status"] == "MISSING").sum()
            stale = (freshness_df["status"] == "STALE").sum()
            ok = (freshness_df["status"] == "OK").sum()

            print("\nSummary:")
            print(f"  OK:      {ok:4d} ({100*ok/total:.1f}%)")
            print(f"  STALE:   {stale:4d} ({100*stale/total:.1f}%)")
            print(f"  MISSING: {missing:4d} ({100*missing/total:.1f}%)")

            # Show details
            if args.show_stale:
                df_show = freshness_df[freshness_df["is_stale"]]
                print(f"\nStale/Missing Symbols ({len(df_show)}):")
            elif args.show_all:
                df_show = freshness_df
                print(f"\nAll Symbols ({len(df_show)}):")
            else:
                # Show first 20 by default
                df_show = freshness_df.head(20)
                print("\nFirst 20 Symbols (use --show-all for complete list):")

            if not df_show.empty:
                headers = ["Symbol", "Status", "Latest Date", "Days Behind", "File Age"]
                rows = []
                for _, row in df_show.iterrows():
                    rows.append([
                        row["symbol"],
                        row["status"],
                        row.get("latest_cache_date", "N/A") or "N/A",
                        row.get("days_behind", "N/A") if pd.notna(row.get("days_behind")) else "N/A",
                        f"{row['file_age_days']:.1f}d" if pd.notna(row.get("file_age_days")) else "N/A",
                    ])
                print(format_table(headers, rows))

            # List symbols needing refresh
            if missing > 0 or stale > 0:
                needs_refresh = freshness_df[freshness_df["is_stale"]]["symbol"].tolist()
                if len(needs_refresh) <= 20:
                    print(f"\nSymbols needing refresh: {', '.join(needs_refresh)}")
                else:
                    print(f"\nSymbols needing refresh: {len(needs_refresh)} symbols")
                    print(f"  First 10: {', '.join(needs_refresh[:10])}")

    print("\n" + "=" * 60)
    print("Dashboard complete.")


if __name__ == "__main__":
    main()
