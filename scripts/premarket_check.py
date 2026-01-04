#!/usr/bin/env python3
"""
Premarket Data Check for Kobe Trading System (Scheduler v2)

Runs before market open (6:45 AM ET) to verify data quality.

Checks:
1. Data staleness (last update within 24h)
2. Missing bars for active symbols
3. Corporate actions (splits, dividends) overnight
4. Universe file integrity
5. Cache health

Usage:
    python scripts/premarket_check.py --dotenv ./.env
    python scripts/premarket_check.py --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "optionable_liquid_900.csv"
CACHE_DIR = ROOT / "data" / "cache"
MANIFEST_FILE = ROOT / "state" / "data_manifest.json"
CHECK_REPORT = ROOT / "reports" / "premarket_check.json"

# Thresholds
MAX_STALENESS_HOURS = 24
MIN_HISTORY_DAYS = 252  # 1 year minimum


def load_universe(universe_path: Path) -> List[str]:
    """Load symbols from universe file."""
    if not universe_path.exists():
        logger.error(f"Universe file not found: {universe_path}")
        return []

    symbols = []
    with open(universe_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if line.lower().startswith('symbol'):
                    continue
                symbol = line.split(',')[0].strip().upper()
                if symbol and symbol.isalpha():
                    symbols.append(symbol)

    return symbols


def get_last_trading_day() -> date:
    """Get the most recent trading day (excluding weekends)."""
    today = date.today()
    if today.weekday() == 0:  # Monday
        return today - timedelta(days=3)  # Friday
    elif today.weekday() == 6:  # Sunday
        return today - timedelta(days=2)  # Friday
    else:
        return today - timedelta(days=1)


def check_cache_file(symbol: str, cache_dir: Path) -> Dict[str, Any]:
    """Check a single cache file for issues."""
    cache_file = cache_dir / f"{symbol}.csv"

    result = {
        "symbol": symbol,
        "exists": False,
        "rows": 0,
        "last_date": None,
        "staleness_days": None,
        "issues": []
    }

    if not cache_file.exists():
        result["issues"].append("missing_cache_file")
        return result

    result["exists"] = True

    try:
        with open(cache_file, 'r') as f:
            lines = f.readlines()

        # Count data rows (skip header)
        data_lines = [l for l in lines if l.strip() and not l.startswith('date')]
        result["rows"] = len(data_lines)

        if result["rows"] < MIN_HISTORY_DAYS:
            result["issues"].append(f"insufficient_history ({result['rows']} days)")

        # Check last date
        if data_lines:
            last_line = data_lines[-1]
            last_date_str = last_line.split(',')[0]
            try:
                last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
                result["last_date"] = last_date.isoformat()

                # Calculate staleness
                today = date.today()
                staleness = (today - last_date).days
                result["staleness_days"] = staleness

                # Check if too stale (accounting for weekends)
                expected_last = get_last_trading_day()
                if last_date < expected_last - timedelta(days=1):
                    result["issues"].append(f"stale_data (last: {last_date}, expected: {expected_last})")
            except Exception:
                result["issues"].append("invalid_date_format")

    except Exception as e:
        result["issues"].append(f"read_error: {str(e)}")

    return result


def check_corporate_actions(symbols: List[str], target_date: date) -> List[Dict]:
    """Check for corporate actions on target date."""
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        logger.warning("POLYGON_API_KEY not set - skipping corporate action check")
        return []

    import requests

    actions = []
    date_str = target_date.strftime("%Y-%m-%d")

    # Check splits
    try:
        url = "https://api.polygon.io/v3/reference/splits"
        params = {
            "execution_date": date_str,
            "apiKey": api_key,
            "limit": 100
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for split in data.get("results", []):
                ticker = split.get("ticker", "")
                if ticker in symbols:
                    actions.append({
                        "symbol": ticker,
                        "type": "split",
                        "date": date_str,
                        "split_from": split.get("split_from"),
                        "split_to": split.get("split_to")
                    })
    except Exception as e:
        logger.warning(f"Split check failed: {e}")

    # Check dividends
    try:
        url = "https://api.polygon.io/v3/reference/dividends"
        params = {
            "ex_dividend_date": date_str,
            "apiKey": api_key,
            "limit": 100
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for div in data.get("results", []):
                ticker = div.get("ticker", "")
                if ticker in symbols:
                    actions.append({
                        "symbol": ticker,
                        "type": "dividend",
                        "date": date_str,
                        "cash_amount": div.get("cash_amount"),
                        "declaration_date": div.get("declaration_date")
                    })
    except Exception as e:
        logger.warning(f"Dividend check failed: {e}")

    return actions


class PremarketChecker:
    """Runs premarket data quality checks."""

    def __init__(
        self,
        universe_path: Path,
        cache_dir: Path,
        verbose: bool = False
    ):
        self.universe_path = universe_path
        self.cache_dir = cache_dir
        self.verbose = verbose

    def run_checks(self) -> Dict[str, Any]:
        """Run all premarket checks."""
        logger.info("=" * 60)
        logger.info("PREMARKET DATA CHECK")
        logger.info(f"Timestamp: {datetime.utcnow().isoformat()}Z")
        logger.info("=" * 60)

        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "pass",
            "checks": {},
            "warnings": [],
            "errors": []
        }

        # 1. Universe integrity
        logger.info("\n[1] Checking universe integrity...")
        symbols = load_universe(self.universe_path)
        results["checks"]["universe"] = {
            "status": "pass" if symbols else "fail",
            "symbol_count": len(symbols),
            "file": str(self.universe_path)
        }
        if not symbols:
            results["errors"].append("Universe file empty or missing")
            results["status"] = "fail"
        logger.info(f"    Universe: {len(symbols)} symbols")

        # 2. Cache health
        logger.info("\n[2] Checking cache health...")
        missing = []
        stale = []
        insufficient = []

        sample_size = min(100, len(symbols))  # Sample for speed
        import random
        sample_symbols = random.sample(symbols, sample_size) if len(symbols) > sample_size else symbols

        for symbol in sample_symbols:
            check = check_cache_file(symbol, self.cache_dir)
            if "missing_cache_file" in check.get("issues", []):
                missing.append(symbol)
            elif any("stale_data" in i for i in check.get("issues", [])):
                stale.append(symbol)
            elif any("insufficient_history" in i for i in check.get("issues", [])):
                insufficient.append(symbol)

        results["checks"]["cache"] = {
            "status": "pass" if not missing and not stale else ("warn" if stale else "fail"),
            "symbols_checked": len(sample_symbols),
            "missing": len(missing),
            "stale": len(stale),
            "insufficient_history": len(insufficient)
        }

        if missing:
            msg = f"{len(missing)} symbols missing cache files"
            results["warnings"].append(msg)
            logger.warning(f"    {msg}")
            if self.verbose:
                logger.warning(f"    Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}")

        if stale:
            msg = f"{len(stale)} symbols with stale data"
            results["warnings"].append(msg)
            logger.warning(f"    {msg}")
            if self.verbose:
                logger.warning(f"    Stale: {stale[:10]}{'...' if len(stale) > 10 else ''}")

        if not missing and not stale:
            logger.info(f"    Cache OK: {len(sample_symbols)} symbols checked")

        # 3. Corporate actions
        logger.info("\n[3] Checking corporate actions...")
        today = date.today()
        actions = check_corporate_actions(symbols, today)

        results["checks"]["corporate_actions"] = {
            "status": "warn" if actions else "pass",
            "date": today.isoformat(),
            "actions_count": len(actions),
            "actions": actions[:20]  # Limit output
        }

        if actions:
            logger.warning(f"    {len(actions)} corporate action(s) today!")
            for action in actions[:5]:
                logger.warning(f"    - {action['symbol']}: {action['type']}")
            results["warnings"].append(f"{len(actions)} corporate actions today")
        else:
            logger.info("    No corporate actions today")

        # 4. Manifest check
        logger.info("\n[4] Checking data manifest...")
        if MANIFEST_FILE.exists():
            try:
                with open(MANIFEST_FILE, 'r') as f:
                    manifest = json.load(f)
                last_finalized = manifest.get("last_finalized")
                results["checks"]["manifest"] = {
                    "status": "pass",
                    "last_finalized": last_finalized,
                    "symbols_count": manifest.get("symbols_count")
                }
                logger.info(f"    Last finalized: {last_finalized}")
            except Exception as e:
                results["checks"]["manifest"] = {"status": "warn", "error": str(e)}
                results["warnings"].append("Cannot read manifest")
        else:
            results["checks"]["manifest"] = {"status": "warn", "error": "not_found"}
            results["warnings"].append("Manifest file not found")
            logger.warning("    Manifest file not found")

        # 5. API connectivity
        logger.info("\n[5] Checking API connectivity...")
        api_key = os.getenv("POLYGON_API_KEY", "")
        if api_key:
            import requests
            try:
                url = "https://api.polygon.io/v2/aggs/ticker/SPY/prev"
                r = requests.get(url, params={"apiKey": api_key}, timeout=10)
                if r.status_code == 200:
                    results["checks"]["api"] = {"status": "pass", "latency_ms": r.elapsed.total_seconds() * 1000}
                    logger.info(f"    Polygon API OK ({r.elapsed.total_seconds() * 1000:.0f}ms)")
                else:
                    results["checks"]["api"] = {"status": "fail", "http_code": r.status_code}
                    results["errors"].append(f"Polygon API error: HTTP {r.status_code}")
            except Exception as e:
                results["checks"]["api"] = {"status": "fail", "error": str(e)}
                results["errors"].append(f"Polygon API error: {e}")
        else:
            results["checks"]["api"] = {"status": "warn", "error": "no_api_key"}
            results["warnings"].append("POLYGON_API_KEY not set")
            logger.warning("    POLYGON_API_KEY not set")

        # Overall status
        if results["errors"]:
            results["status"] = "fail"
        elif results["warnings"]:
            results["status"] = "warn"

        # Save report
        CHECK_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with open(CHECK_REPORT, 'w') as f:
            json.dump(results, f, indent=2)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("PREMARKET CHECK SUMMARY")
        logger.info(f"  Status: {results['status'].upper()}")
        logger.info(f"  Errors: {len(results['errors'])}")
        logger.info(f"  Warnings: {len(results['warnings'])}")
        logger.info(f"  Report: {CHECK_REPORT}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Kobe Premarket Data Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/premarket_check.py --dotenv ./.env
    python scripts/premarket_check.py --verbose
        """
    )
    parser.add_argument("--dotenv", type=str, default="./.env",
                        help="Path to .env file")
    parser.add_argument("--universe", type=str, default=str(DEFAULT_UNIVERSE),
                        help="Path to universe file")
    parser.add_argument("--cache", type=str, default=str(CACHE_DIR),
                        help="Path to cache directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON summary")

    args = parser.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run checks
    try:
        checker = PremarketChecker(
            universe_path=Path(args.universe),
            cache_dir=Path(args.cache),
            verbose=args.verbose
        )
        results = checker.run_checks()

        if args.json:
            print(json.dumps(results, indent=2))

        if results["status"] == "fail":
            sys.exit(2)
        elif results["status"] == "warn":
            sys.exit(1)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Premarket check failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
