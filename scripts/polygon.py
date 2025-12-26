#!/usr/bin/env python3
"""
Polygon Data Source Validation - Kobe Trading System

Validates Polygon API configuration, checks rate limits,
validates 950-symbol coverage, and compares cache vs fresh fetches.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from data.universe.loader import load_universe
from data.providers.polygon_eod import fetch_daily_bars_polygon, PolygonConfig


POLYGON_TICKERS_URL = "https://api.polygon.io/v3/reference/tickers"
POLYGON_ACCOUNT_URL = "https://api.polygon.io/v1/account"


def verify_api_key() -> Dict[str, Any]:
    """
    Verify the Polygon API key is valid and check subscription tier.
    """
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return {
            "valid": False,
            "error": "POLYGON_API_KEY not set in environment",
        }

    # Test with a simple API call
    try:
        url = POLYGON_TICKERS_URL
        params = {"limit": 1, "apiKey": api_key}
        resp = requests.get(url, params=params, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            return {
                "valid": True,
                "status": "OK",
                "http_status": resp.status_code,
                "results_count": len(data.get("results", [])),
                "request_id": data.get("request_id", "N/A"),
            }
        elif resp.status_code == 401:
            return {
                "valid": False,
                "error": "Invalid API key (401 Unauthorized)",
                "http_status": resp.status_code,
            }
        elif resp.status_code == 403:
            return {
                "valid": False,
                "error": "API key lacks required permissions (403 Forbidden)",
                "http_status": resp.status_code,
            }
        elif resp.status_code == 429:
            return {
                "valid": True,
                "status": "RATE_LIMITED",
                "http_status": resp.status_code,
                "message": "Rate limit exceeded - key is valid but throttled",
            }
        else:
            return {
                "valid": False,
                "error": f"Unexpected response: {resp.status_code}",
                "http_status": resp.status_code,
                "response": resp.text[:200],
            }
    except requests.exceptions.Timeout:
        return {
            "valid": False,
            "error": "Request timed out - check network connection",
        }
    except requests.exceptions.RequestException as e:
        return {
            "valid": False,
            "error": f"Connection error: {str(e)}",
        }


def check_rate_limits() -> Dict[str, Any]:
    """
    Check current rate limit status by making a test request.
    """
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return {"status": "NO_API_KEY"}

    try:
        url = POLYGON_TICKERS_URL
        params = {"limit": 1, "apiKey": api_key}
        resp = requests.get(url, params=params, timeout=15)

        # Extract rate limit headers
        headers = resp.headers
        return {
            "status": "OK" if resp.status_code == 200 else "ERROR",
            "http_status": resp.status_code,
            "rate_limit": headers.get("X-RateLimit-Limit", "N/A"),
            "rate_remaining": headers.get("X-RateLimit-Remaining", "N/A"),
            "rate_reset": headers.get("X-RateLimit-Reset", "N/A"),
        }
    except requests.exceptions.RequestException as e:
        return {"status": "ERROR", "message": str(e)}


def validate_symbol_coverage(
    symbols: List[str],
    cache_dir: Path,
    min_years: float = 10.0,
    sample_size: int = 0,
) -> Dict[str, Any]:
    """
    Validate that symbols have sufficient data coverage.
    If sample_size > 0, only check a random sample of symbols.
    """
    if sample_size > 0 and sample_size < len(symbols):
        check_symbols = random.sample(symbols, sample_size)
    else:
        check_symbols = symbols

    results = {
        "total_symbols": len(symbols),
        "checked_symbols": len(check_symbols),
        "has_data": 0,
        "no_data": 0,
        "sufficient_history": 0,
        "insufficient_history": 0,
        "errors": 0,
        "details": [],
    }

    # Use historical date range for checking
    end_date = datetime.now().date().isoformat()
    start_date = "2010-01-01"

    for i, sym in enumerate(check_symbols, 1):
        if i % 50 == 0:
            print(f"  Checking {i}/{len(check_symbols)}...")

        try:
            df = fetch_daily_bars_polygon(
                sym, start_date, end_date, cache_dir=cache_dir
            )

            if df.empty:
                results["no_data"] += 1
                results["details"].append({
                    "symbol": sym,
                    "status": "NO_DATA",
                    "rows": 0,
                })
            else:
                results["has_data"] += 1
                df = df.sort_values("timestamp")
                earliest = pd.to_datetime(df.iloc[0]["timestamp"]).date()
                latest = pd.to_datetime(df.iloc[-1]["timestamp"]).date()
                years = (latest - earliest).days / 365.25

                if years >= min_years:
                    results["sufficient_history"] += 1
                    status = "OK"
                else:
                    results["insufficient_history"] += 1
                    status = "INSUFFICIENT"

                results["details"].append({
                    "symbol": sym,
                    "status": status,
                    "rows": len(df),
                    "earliest": str(earliest),
                    "latest": str(latest),
                    "years": round(years, 2),
                })
        except Exception as e:
            results["errors"] += 1
            results["details"].append({
                "symbol": sym,
                "status": "ERROR",
                "error": str(e)[:100],
            })

    # Calculate percentages
    checked = results["checked_symbols"]
    if checked > 0:
        results["coverage_pct"] = round(100 * results["has_data"] / checked, 1)
        results["sufficient_pct"] = round(100 * results["sufficient_history"] / checked, 1)

    return results


def compare_cache_vs_fresh(
    symbols: List[str],
    cache_dir: Path,
    sample_size: int = 5,
) -> List[Dict[str, Any]]:
    """
    Compare cached data vs fresh fetch for a sample of symbols.
    Useful for detecting stale or corrupted cache files.
    """
    if sample_size > len(symbols):
        sample_size = len(symbols)

    sample_symbols = random.sample(symbols, sample_size)
    results = []

    end_date = datetime.now().date().isoformat()
    start_date = (datetime.now() - timedelta(days=30)).date().isoformat()

    for sym in sample_symbols:
        print(f"  Comparing {sym}...")

        # Read from cache
        cached_df = fetch_daily_bars_polygon(sym, start_date, end_date, cache_dir=cache_dir)

        # Force fresh fetch by using a temp cache dir
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            fresh_df = fetch_daily_bars_polygon(
                sym, start_date, end_date, cache_dir=Path(tmpdir)
            )

        result = {
            "symbol": sym,
            "cache_rows": len(cached_df),
            "fresh_rows": len(fresh_df),
            "rows_match": len(cached_df) == len(fresh_df),
        }

        if not cached_df.empty and not fresh_df.empty:
            # Compare last close price
            cache_last = cached_df.sort_values("timestamp").iloc[-1]
            fresh_last = fresh_df.sort_values("timestamp").iloc[-1]

            result["cache_last_date"] = str(cache_last["timestamp"])[:10]
            result["fresh_last_date"] = str(fresh_last["timestamp"])[:10]
            result["cache_last_close"] = round(cache_last["close"], 2)
            result["fresh_last_close"] = round(fresh_last["close"], 2)
            result["close_match"] = abs(cache_last["close"] - fresh_last["close"]) < 0.01
            result["status"] = "OK" if result["rows_match"] and result["close_match"] else "MISMATCH"
        else:
            result["status"] = "MISSING_DATA"

        results.append(result)
        time.sleep(0.5)  # Rate limiting

    return results


def generate_coverage_report(
    coverage: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a text coverage report.
    """
    lines = [
        "=" * 70,
        "POLYGON DATA COVERAGE REPORT",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 70,
        "",
        "SUMMARY",
        "-" * 40,
        f"Total Universe Symbols:  {coverage['total_symbols']}",
        f"Symbols Checked:         {coverage['checked_symbols']}",
        f"With Data:               {coverage['has_data']} ({coverage.get('coverage_pct', 0):.1f}%)",
        f"No Data:                 {coverage['no_data']}",
        f"Sufficient History:      {coverage['sufficient_history']} ({coverage.get('sufficient_pct', 0):.1f}%)",
        f"Insufficient History:    {coverage['insufficient_history']}",
        f"Errors:                  {coverage['errors']}",
        "",
    ]

    # Group by status
    details = coverage.get("details", [])
    if details:
        ok_symbols = [d for d in details if d.get("status") == "OK"]
        insufficient = [d for d in details if d.get("status") == "INSUFFICIENT"]
        no_data = [d for d in details if d.get("status") == "NO_DATA"]
        errors = [d for d in details if d.get("status") == "ERROR"]

        if insufficient:
            lines.append("SYMBOLS WITH INSUFFICIENT HISTORY")
            lines.append("-" * 40)
            for d in insufficient[:20]:
                lines.append(f"  {d['symbol']}: {d.get('years', 0):.1f} years ({d.get('earliest', 'N/A')} to {d.get('latest', 'N/A')})")
            if len(insufficient) > 20:
                lines.append(f"  ... and {len(insufficient) - 20} more")
            lines.append("")

        if no_data:
            lines.append("SYMBOLS WITH NO DATA")
            lines.append("-" * 40)
            syms = [d["symbol"] for d in no_data]
            for i in range(0, len(syms), 10):
                lines.append("  " + ", ".join(syms[i : i + 10]))
            lines.append("")

        if errors:
            lines.append("SYMBOLS WITH ERRORS")
            lines.append("-" * 40)
            for d in errors[:10]:
                lines.append(f"  {d['symbol']}: {d.get('error', 'Unknown error')}")
            lines.append("")

    lines.append("=" * 70)
    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

    return report


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


def main():
    ap = argparse.ArgumentParser(
        description="Polygon Data Source Validation - Kobe Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/polygon.py --verify-key
  python scripts/polygon.py --check-coverage --universe data/universe/optionable_liquid_final.csv
  python scripts/polygon.py --compare-cache --sample 10
  python scripts/polygon.py --full-validation --report outputs/polygon_report.txt
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
        default="data/universe/optionable_liquid_final.csv",
        help="Universe CSV file",
    )
    ap.add_argument(
        "--verify-key",
        action="store_true",
        help="Verify API key is valid",
    )
    ap.add_argument(
        "--check-limits",
        action="store_true",
        help="Check rate limit status",
    )
    ap.add_argument(
        "--check-coverage",
        action="store_true",
        help="Validate symbol coverage (uses cache)",
    )
    ap.add_argument(
        "--compare-cache",
        action="store_true",
        help="Compare cache vs fresh fetch for sample symbols",
    )
    ap.add_argument(
        "--full-validation",
        action="store_true",
        help="Run full validation (all checks)",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Sample size for coverage check (0 = all)",
    )
    ap.add_argument(
        "--min-years",
        type=float,
        default=10.0,
        help="Minimum years of history required",
    )
    ap.add_argument(
        "--report",
        type=str,
        default=None,
        help="Output coverage report to file",
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

    # If no flags, run all checks
    run_all = args.full_validation or not (
        args.verify_key or args.check_limits or args.check_coverage or args.compare_cache
    )

    # 1. Verify API Key
    if args.verify_key or run_all:
        print("\n" + "=" * 60)
        print("API KEY VERIFICATION")
        print("=" * 60)

        result = verify_api_key()
        if result["valid"]:
            print("Status:     VALID")
            print(f"HTTP:       {result.get('http_status', 'N/A')}")
            print(f"Request ID: {result.get('request_id', 'N/A')}")
        else:
            print("Status:     INVALID")
            print(f"Error:      {result.get('error', 'Unknown error')}")
            if not run_all:
                sys.exit(1)

    # 2. Check Rate Limits
    if args.check_limits or run_all:
        print("\n" + "=" * 60)
        print("RATE LIMIT STATUS")
        print("=" * 60)

        limits = check_rate_limits()
        print(f"Status:          {limits.get('status', 'N/A')}")
        print(f"Rate Limit:      {limits.get('rate_limit', 'N/A')}")
        print(f"Rate Remaining:  {limits.get('rate_remaining', 'N/A')}")
        print(f"Rate Reset:      {limits.get('rate_reset', 'N/A')}")

    # 3. Check Coverage
    if args.check_coverage or run_all:
        print("\n" + "=" * 60)
        print("SYMBOL COVERAGE VALIDATION")
        print("=" * 60)

        if not universe_path.exists():
            print(f"Error: Universe file not found: {universe_path}")
        else:
            symbols = load_universe(universe_path)
            print(f"Universe: {universe_path}")
            print(f"Total Symbols: {len(symbols)}")
            print(f"Sample Size: {args.sample if args.sample > 0 else 'ALL'}")
            print(f"Min Years Required: {args.min_years}")
            print()

            coverage = validate_symbol_coverage(
                symbols,
                cache_dir,
                min_years=args.min_years,
                sample_size=args.sample,
            )

            print("\nCoverage Results:")
            print(f"  Checked:            {coverage['checked_symbols']}")
            print(f"  With Data:          {coverage['has_data']} ({coverage.get('coverage_pct', 0):.1f}%)")
            print(f"  Sufficient History: {coverage['sufficient_history']} ({coverage.get('sufficient_pct', 0):.1f}%)")
            print(f"  Insufficient:       {coverage['insufficient_history']}")
            print(f"  No Data:            {coverage['no_data']}")
            print(f"  Errors:             {coverage['errors']}")

            # Check 950 symbol requirement
            target = 950
            if coverage["sufficient_history"] >= target:
                print(f"\n  [PASS] >= {target} symbols with sufficient history")
            else:
                print(f"\n  [FAIL] Only {coverage['sufficient_history']} symbols meet {args.min_years}y requirement (need {target})")

            # Generate report if requested
            if args.report:
                report_path = Path(args.report)
                if not report_path.is_absolute():
                    report_path = ROOT / report_path
                report = generate_coverage_report(coverage, report_path)
                print(f"\nReport written to: {report_path}")

    # 4. Compare Cache vs Fresh
    if args.compare_cache or run_all:
        print("\n" + "=" * 60)
        print("CACHE VS FRESH COMPARISON")
        print("=" * 60)

        if not universe_path.exists():
            print(f"Error: Universe file not found: {universe_path}")
        else:
            symbols = load_universe(universe_path)
            sample_size = min(5, len(symbols))
            print(f"Comparing {sample_size} random symbols...")
            print()

            comparisons = compare_cache_vs_fresh(symbols, cache_dir, sample_size)

            headers = ["Symbol", "Cache Rows", "Fresh Rows", "Match", "Cache Date", "Fresh Date", "Status"]
            rows = []
            for c in comparisons:
                rows.append([
                    c["symbol"],
                    c["cache_rows"],
                    c["fresh_rows"],
                    "Yes" if c.get("rows_match") else "No",
                    c.get("cache_last_date", "N/A"),
                    c.get("fresh_last_date", "N/A"),
                    c["status"],
                ])
            print(format_table(headers, rows))

            # Summary
            ok_count = sum(1 for c in comparisons if c["status"] == "OK")
            print(f"\nCache Integrity: {ok_count}/{len(comparisons)} OK")

    print("\n" + "=" * 60)
    print("Polygon validation complete.")


if __name__ == "__main__":
    main()
