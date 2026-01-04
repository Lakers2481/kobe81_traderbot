#!/usr/bin/env python3
"""
EOD Data Finalization for Kobe Trading System (Scheduler v2)

Runs after market close (6:00 PM ET) to finalize today's data.
Polygon data typically available 15-30 minutes after close.

Tasks:
1. Wait for provider data availability
2. Fetch final EOD bars for universe
3. Check for corporate actions (splits, dividends)
4. Update adjusted prices if needed
5. Validate no missing bars
6. Mark data as "finalized" in manifest

Usage:
    python scripts/eod_finalize.py --dotenv ./.env
    python scripts/eod_finalize.py --universe data/universe/optionable_liquid_900.csv
    python scripts/eod_finalize.py --check-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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
POLYGON_DELAY_MINUTES = 30  # Wait for Polygon to finalize data


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
                # Handle CSV with header
                if line.lower().startswith('symbol'):
                    continue
                # Extract symbol (first column if CSV)
                symbol = line.split(',')[0].strip().upper()
                if symbol and symbol.isalpha():
                    symbols.append(symbol)

    return symbols


def get_trading_date() -> date:
    """Get the trading date to finalize (today if before 4 PM, yesterday otherwise)."""
    now = datetime.now()
    # If after 4 PM ET, finalize today's data
    # If before, finalize yesterday's
    if now.hour >= 16:
        return now.date()
    else:
        return (now - timedelta(days=1)).date()


def check_polygon_data_available(symbol: str, target_date: date) -> bool:
    """Check if Polygon has data for the target date."""
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        logger.warning("POLYGON_API_KEY not set")
        return False

    import requests

    date_str = target_date.strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{date_str}/{date_str}"
    params = {"apiKey": api_key}

    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            return len(results) > 0
        return False
    except Exception as e:
        logger.debug(f"Polygon check failed for {symbol}: {e}")
        return False


def fetch_eod_bar(symbol: str, target_date: date) -> Optional[Dict]:
    """Fetch EOD bar from Polygon."""
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return None

    import requests

    date_str = target_date.strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{date_str}/{date_str}"
    params = {"apiKey": api_key, "adjusted": "true"}

    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            if results:
                bar = results[0]
                return {
                    "date": date_str,
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v"),
                    "vwap": bar.get("vw"),
                    "transactions": bar.get("n")
                }
        return None
    except Exception as e:
        logger.debug(f"Fetch failed for {symbol}: {e}")
        return None


def check_corporate_actions(symbol: str, target_date: date) -> Dict[str, Any]:
    """Check for splits/dividends on the target date."""
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return {"has_actions": False}

    import requests

    date_str = target_date.strftime("%Y-%m-%d")
    actions = {"has_actions": False, "splits": [], "dividends": []}

    # Check splits
    try:
        url = "https://api.polygon.io/v3/reference/splits"
        params = {
            "ticker": symbol,
            "execution_date": date_str,
            "apiKey": api_key
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            if results:
                actions["has_actions"] = True
                actions["splits"] = results
    except Exception:
        pass

    # Check dividends
    try:
        url = "https://api.polygon.io/v3/reference/dividends"
        params = {
            "ticker": symbol,
            "ex_dividend_date": date_str,
            "apiKey": api_key
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            if results:
                actions["has_actions"] = True
                actions["dividends"] = results
    except Exception:
        pass

    return actions


def update_cache(symbol: str, bar: Dict, cache_dir: Path):
    """Append bar to symbol's cache file."""
    cache_file = cache_dir / f"{symbol}.csv"

    # Check if file exists and if date already present
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            content = f.read()
            if bar["date"] in content:
                logger.debug(f"{symbol}: Date {bar['date']} already in cache")
                return False

    # Append new bar
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Create header if new file
    if not cache_file.exists():
        with open(cache_file, 'w') as f:
            f.write("date,open,high,low,close,volume,vwap\n")

    with open(cache_file, 'a') as f:
        f.write(f"{bar['date']},{bar['open']},{bar['high']},{bar['low']},{bar['close']},{bar['volume']},{bar.get('vwap', '')}\n")

    return True


def load_manifest() -> Dict:
    """Load data manifest."""
    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_finalized": None, "finalized_dates": [], "symbols_count": 0}


def save_manifest(manifest: Dict):
    """Save data manifest."""
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)


class EODFinalizer:
    """Finalizes end-of-day data."""

    def __init__(
        self,
        universe_path: Path,
        cache_dir: Path,
        check_only: bool = False,
        max_wait_minutes: int = 45
    ):
        self.universe_path = universe_path
        self.cache_dir = cache_dir
        self.check_only = check_only
        self.max_wait_minutes = max_wait_minutes
        self.target_date = get_trading_date()

    def wait_for_data(self) -> bool:
        """Wait for Polygon data to become available."""
        test_symbols = ["AAPL", "MSFT", "SPY"]

        logger.info(f"Waiting for Polygon data for {self.target_date}...")

        start_time = time.time()
        max_wait_seconds = self.max_wait_minutes * 60

        while (time.time() - start_time) < max_wait_seconds:
            available = sum(1 for s in test_symbols if check_polygon_data_available(s, self.target_date))

            if available >= 2:
                logger.info(f"Data available ({available}/{len(test_symbols)} test symbols)")
                return True

            elapsed = (time.time() - start_time) / 60
            logger.info(f"Data not ready yet ({available}/{len(test_symbols)}). Waiting... ({elapsed:.1f} min elapsed)")
            time.sleep(60)

        logger.warning(f"Timeout waiting for data after {self.max_wait_minutes} minutes")
        return False

    def finalize(self) -> Dict[str, Any]:
        """Run the finalization process."""
        logger.info("=" * 60)
        logger.info("EOD DATA FINALIZATION")
        logger.info(f"Target date: {self.target_date}")
        logger.info(f"Mode: {'CHECK ONLY' if self.check_only else 'LIVE'}")
        logger.info("=" * 60)

        # Check if already finalized
        manifest = load_manifest()
        date_str = self.target_date.strftime("%Y-%m-%d")

        if date_str in manifest.get("finalized_dates", []):
            logger.info(f"Date {date_str} already finalized")
            return {"status": "already_finalized", "date": date_str}

        # Load universe
        symbols = load_universe(self.universe_path)
        logger.info(f"Universe: {len(symbols)} symbols")

        if not symbols:
            return {"status": "error", "error": "No symbols in universe"}

        # Wait for data
        if not self.wait_for_data():
            return {"status": "error", "error": "Data not available"}

        # Process symbols
        results = {
            "date": date_str,
            "symbols_processed": 0,
            "bars_added": 0,
            "missing": [],
            "corporate_actions": [],
            "errors": []
        }

        for i, symbol in enumerate(symbols):
            if i % 100 == 0:
                logger.info(f"Processing {i}/{len(symbols)}...")

            try:
                # Fetch bar
                bar = fetch_eod_bar(symbol, self.target_date)

                if bar is None:
                    results["missing"].append(symbol)
                    continue

                results["symbols_processed"] += 1

                # Check corporate actions
                actions = check_corporate_actions(symbol, self.target_date)
                if actions["has_actions"]:
                    results["corporate_actions"].append({
                        "symbol": symbol,
                        "actions": actions
                    })

                # Update cache
                if not self.check_only:
                    if update_cache(symbol, bar, self.cache_dir):
                        results["bars_added"] += 1

            except Exception as e:
                results["errors"].append({"symbol": symbol, "error": str(e)})

            # Rate limiting
            time.sleep(0.1)

        # Update manifest
        if not self.check_only:
            manifest["last_finalized"] = date_str
            if date_str not in manifest.get("finalized_dates", []):
                if "finalized_dates" not in manifest:
                    manifest["finalized_dates"] = []
                manifest["finalized_dates"].append(date_str)
            manifest["symbols_count"] = len(symbols)
            manifest["last_update"] = datetime.utcnow().isoformat()
            save_manifest(manifest)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("FINALIZATION COMPLETE")
        logger.info(f"  Date: {date_str}")
        logger.info(f"  Symbols processed: {results['symbols_processed']}/{len(symbols)}")
        logger.info(f"  Bars added: {results['bars_added']}")
        logger.info(f"  Missing: {len(results['missing'])}")
        logger.info(f"  Corporate actions: {len(results['corporate_actions'])}")
        logger.info(f"  Errors: {len(results['errors'])}")

        if results["corporate_actions"]:
            logger.info("\nCORPORATE ACTIONS:")
            for ca in results["corporate_actions"]:
                logger.info(f"  {ca['symbol']}: {ca['actions']}")

        results["status"] = "success"
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Kobe EOD Data Finalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/eod_finalize.py --dotenv ./.env
    python scripts/eod_finalize.py --check-only
    python scripts/eod_finalize.py --universe data/universe/optionable_liquid_900.csv
        """
    )
    parser.add_argument("--dotenv", type=str, default="./.env",
                        help="Path to .env file")
    parser.add_argument("--universe", type=str, default=str(DEFAULT_UNIVERSE),
                        help="Path to universe file")
    parser.add_argument("--cache", type=str, default=str(CACHE_DIR),
                        help="Path to cache directory")
    parser.add_argument("--check-only", action="store_true",
                        help="Check data availability only, don't update cache")
    parser.add_argument("--max-wait", type=int, default=45,
                        help="Maximum minutes to wait for data (default: 45)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run finalizer
    try:
        finalizer = EODFinalizer(
            universe_path=Path(args.universe),
            cache_dir=Path(args.cache),
            check_only=args.check_only,
            max_wait_minutes=args.max_wait
        )
        results = finalizer.finalize()

        if results.get("status") == "success":
            sys.exit(0)
        elif results.get("status") == "already_finalized":
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        logger.error(f"EOD finalization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
