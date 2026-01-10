#!/usr/bin/env python3
"""
Divergence Monitor for Kobe Trading System (Scheduler v2)

Continuous monitoring for system divergences and data integrity.
Runs every 5-15 minutes during market hours.

Checks:
1. Order fill status (partial/unfilled orders)
2. Broker vs OMS position mismatch
3. Quote staleness (last update > 5 min)
4. Spread widening alerts
5. Model output availability
6. API health status

Usage:
    python monitor/divergence_monitor.py --dotenv ./.env
    python monitor/divergence_monitor.py --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from enum import Enum

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class DivergenceLevel(Enum):
    """Severity level for divergences."""
    OK = "ok"           # No issue
    WARN = "warn"       # Warning, should investigate
    CRITICAL = "critical"  # Critical issue, may need action
    ERROR = "error"     # System error


@dataclass
class DivergenceCheck:
    """Result of a divergence check."""
    check_name: str
    level: DivergenceLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


ALPACA_TIMEOUT = 10
STATE_DIR = ROOT / "state"
OMS_STATE_FILE = STATE_DIR / "order_state.json"
DIVERGENCE_LOG = ROOT / "logs" / "divergence.jsonl"


def get_alpaca_config() -> Dict[str, str]:
    """Get Alpaca API configuration from environment."""
    return {
        "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/"),
        "key_id": os.getenv("ALPACA_API_KEY_ID", os.getenv("APCA_API_KEY_ID", "")),
        "secret": os.getenv("ALPACA_API_SECRET_KEY", os.getenv("APCA_API_SECRET_KEY", ""))
    }


def get_auth_headers(config: Dict[str, str]) -> Dict[str, str]:
    """Get authentication headers for Alpaca API."""
    return {
        "APCA-API-KEY-ID": config["key_id"],
        "APCA-API-SECRET-KEY": config["secret"],
        "Content-Type": "application/json"
    }


def alpaca_request(endpoint: str, timeout: int = ALPACA_TIMEOUT) -> Dict[str, Any]:
    """Make a GET request to the Alpaca API."""
    config = get_alpaca_config()
    url = f"{config['base_url']}{endpoint}"
    headers = get_auth_headers(config)

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        return {
            "success": response.status_code in (200, 201),
            "status_code": response.status_code,
            "data": response.json() if response.content else None,
            "error": None,
            "latency_ms": response.elapsed.total_seconds() * 1000
        }
    except requests.exceptions.Timeout:
        return {"success": False, "status_code": None, "data": None, "error": "timeout", "latency_ms": None}
    except requests.exceptions.ConnectionError as e:
        return {"success": False, "status_code": None, "data": None, "error": f"connection: {e}", "latency_ms": None}
    except Exception as e:
        return {"success": False, "status_code": None, "data": None, "error": str(e), "latency_ms": None}


def log_divergence(check: DivergenceCheck):
    """Log divergence to file."""
    DIVERGENCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        record = asdict(check)
        record["level"] = check.level.value
        with open(DIVERGENCE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to log divergence: {e}")


class DivergenceMonitor:
    """Monitors for system divergences."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.checks: List[DivergenceCheck] = []

    def check_api_health(self) -> DivergenceCheck:
        """Check Alpaca API connectivity and latency."""
        result = alpaca_request("/v2/account")

        if not result["success"]:
            return DivergenceCheck(
                check_name="api_health",
                level=DivergenceLevel.CRITICAL,
                message=f"API unreachable: {result['error']}",
                details={"error": result["error"]}
            )

        latency = result.get("latency_ms", 0)
        if latency > 2000:
            return DivergenceCheck(
                check_name="api_health",
                level=DivergenceLevel.WARN,
                message=f"High API latency: {latency:.0f}ms",
                details={"latency_ms": latency}
            )

        return DivergenceCheck(
            check_name="api_health",
            level=DivergenceLevel.OK,
            message=f"API healthy ({latency:.0f}ms)",
            details={"latency_ms": latency}
        )

    def check_order_fill_status(self) -> DivergenceCheck:
        """Check for open/pending orders that should have been filled."""
        result = alpaca_request("/v2/orders?status=open")

        if not result["success"]:
            return DivergenceCheck(
                check_name="order_fill_status",
                level=DivergenceLevel.ERROR,
                message=f"Cannot fetch orders: {result['error']}",
                details={"error": result["error"]}
            )

        orders = result["data"] or []

        if not orders:
            return DivergenceCheck(
                check_name="order_fill_status",
                level=DivergenceLevel.OK,
                message="No pending orders",
                details={"pending_count": 0}
            )

        # Check for stale orders (submitted > 5 minutes ago)
        stale_orders = []
        now = datetime.utcnow()

        for order in orders:
            created_at = order.get("created_at", "")
            if created_at:
                try:
                    # Parse ISO timestamp
                    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    age_minutes = (now - created.replace(tzinfo=None)).total_seconds() / 60
                    if age_minutes > 5:
                        stale_orders.append({
                            "symbol": order.get("symbol"),
                            "side": order.get("side"),
                            "type": order.get("type"),
                            "age_minutes": round(age_minutes, 1)
                        })
                except Exception:
                    pass

        if stale_orders:
            return DivergenceCheck(
                check_name="order_fill_status",
                level=DivergenceLevel.WARN,
                message=f"{len(stale_orders)} stale order(s) detected",
                details={"stale_orders": stale_orders}
            )

        return DivergenceCheck(
            check_name="order_fill_status",
            level=DivergenceLevel.OK,
            message=f"{len(orders)} active order(s), none stale",
            details={"active_count": len(orders)}
        )

    def check_position_mismatch(self) -> DivergenceCheck:
        """Compare broker positions with OMS state."""
        # Fetch broker positions
        result = alpaca_request("/v2/positions")
        if not result["success"]:
            return DivergenceCheck(
                check_name="position_mismatch",
                level=DivergenceLevel.ERROR,
                message=f"Cannot fetch positions: {result['error']}",
                details={"error": result["error"]}
            )

        broker_positions = {p["symbol"]: int(float(p["qty"])) for p in (result["data"] or [])}

        # Load OMS state
        oms_positions = {}
        if OMS_STATE_FILE.exists():
            try:
                with open(OMS_STATE_FILE, 'r') as f:
                    oms_data = json.load(f)
                oms_positions = {k: v.get("qty", 0) for k, v in oms_data.items() if v.get("status") == "FILLED"}
            except Exception as e:
                logger.warning(f"Cannot read OMS state: {e}")

        # Compare
        mismatches = []
        all_symbols = set(broker_positions.keys()) | set(oms_positions.keys())

        for symbol in all_symbols:
            broker_qty = broker_positions.get(symbol, 0)
            oms_qty = oms_positions.get(symbol, 0)
            if broker_qty != oms_qty:
                mismatches.append({
                    "symbol": symbol,
                    "broker_qty": broker_qty,
                    "oms_qty": oms_qty,
                    "diff": broker_qty - oms_qty
                })

        if mismatches:
            return DivergenceCheck(
                check_name="position_mismatch",
                level=DivergenceLevel.CRITICAL,
                message=f"{len(mismatches)} position mismatch(es) detected",
                details={"mismatches": mismatches}
            )

        return DivergenceCheck(
            check_name="position_mismatch",
            level=DivergenceLevel.OK,
            message=f"Positions synced ({len(broker_positions)} positions)",
            details={"position_count": len(broker_positions)}
        )

    def check_quote_staleness(self) -> DivergenceCheck:
        """Check if quote data is stale (for open positions)."""
        # Fetch positions
        result = alpaca_request("/v2/positions")
        if not result["success"] or not result["data"]:
            return DivergenceCheck(
                check_name="quote_staleness",
                level=DivergenceLevel.OK,
                message="No positions to check",
                details={}
            )

        symbols = [p["symbol"] for p in result["data"]]

        # Fetch quotes via data API
        config = get_alpaca_config()
        data_base = config["base_url"].replace("paper-api", "data").replace("api.", "data.")
        if "data.alpaca.markets" not in data_base:
            data_base = "https://data.alpaca.markets"

        stale_quotes = []
        now = datetime.utcnow()

        for symbol in symbols[:10]:  # Limit to 10 symbols
            try:
                url = f"{data_base}/v2/stocks/quotes?symbols={symbol}"
                headers = get_auth_headers(config)
                r = requests.get(url, headers=headers, timeout=5)

                if r.status_code == 200:
                    data = r.json()
                    quotes = data.get("quotes", {}).get(symbol, [])
                    if quotes:
                        q = quotes[-1] if isinstance(quotes, list) else quotes
                        quote_time = q.get("t", "")
                        if quote_time:
                            try:
                                qt = datetime.fromisoformat(quote_time.replace("Z", "+00:00"))
                                age_seconds = (now - qt.replace(tzinfo=None)).total_seconds()
                                if age_seconds > 300:  # 5 minutes
                                    stale_quotes.append({
                                        "symbol": symbol,
                                        "age_seconds": round(age_seconds)
                                    })
                            except Exception:
                                pass
            except Exception as e:
                logger.debug(f"Quote check failed for {symbol}: {e}")

        if stale_quotes:
            return DivergenceCheck(
                check_name="quote_staleness",
                level=DivergenceLevel.WARN,
                message=f"{len(stale_quotes)} stale quote(s) detected",
                details={"stale_quotes": stale_quotes}
            )

        return DivergenceCheck(
            check_name="quote_staleness",
            level=DivergenceLevel.OK,
            message=f"Quotes fresh for {len(symbols)} symbols",
            details={"symbols_checked": len(symbols)}
        )

    def check_spread_widening(self) -> DivergenceCheck:
        """Check for abnormally wide spreads on positions."""
        # Fetch positions
        result = alpaca_request("/v2/positions")
        if not result["success"] or not result["data"]:
            return DivergenceCheck(
                check_name="spread_widening",
                level=DivergenceLevel.OK,
                message="No positions to check",
                details={}
            )

        positions = result["data"]
        config = get_alpaca_config()
        data_base = config["base_url"].replace("paper-api", "data").replace("api.", "data.")
        if "data.alpaca.markets" not in data_base:
            data_base = "https://data.alpaca.markets"

        wide_spreads = []

        for pos in positions[:10]:
            symbol = pos["symbol"]
            try:
                url = f"{data_base}/v2/stocks/quotes?symbols={symbol}"
                headers = get_auth_headers(config)
                r = requests.get(url, headers=headers, timeout=5)

                if r.status_code == 200:
                    data = r.json()
                    quotes = data.get("quotes", {}).get(symbol, [])
                    if quotes:
                        q = quotes[-1] if isinstance(quotes, list) else quotes
                        bid = float(q.get("bp") or 0)
                        ask = float(q.get("ap") or 0)
                        if bid > 0 and ask > 0:
                            spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
                            if spread_pct > 0.5:  # 0.5% threshold
                                wide_spreads.append({
                                    "symbol": symbol,
                                    "bid": bid,
                                    "ask": ask,
                                    "spread_pct": round(spread_pct, 3)
                                })
            except Exception as e:
                logger.debug(f"Spread check failed for {symbol}: {e}")

        if wide_spreads:
            level = DivergenceLevel.CRITICAL if any(s["spread_pct"] > 2.0 for s in wide_spreads) else DivergenceLevel.WARN
            return DivergenceCheck(
                check_name="spread_widening",
                level=level,
                message=f"{len(wide_spreads)} wide spread(s) detected",
                details={"wide_spreads": wide_spreads}
            )

        return DivergenceCheck(
            check_name="spread_widening",
            level=DivergenceLevel.OK,
            message="Spreads normal",
            details={"positions_checked": len(positions)}
        )

    def check_model_outputs(self) -> DivergenceCheck:
        """Check if model output files are fresh."""
        model_files = [
            ROOT / "state" / "cognitive" / "self_model.json",
            ROOT / "state" / "cognitive" / "curiosity_state.json",
            ROOT / "logs" / "daily_insights.json",
        ]

        stale_files = []
        missing_files = []
        now = datetime.utcnow()

        for f in model_files:
            if not f.exists():
                missing_files.append(str(f.name))
            else:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                age_hours = (now - mtime).total_seconds() / 3600
                if age_hours > 24:
                    stale_files.append({
                        "file": str(f.name),
                        "age_hours": round(age_hours, 1)
                    })

        if missing_files:
            return DivergenceCheck(
                check_name="model_outputs",
                level=DivergenceLevel.WARN,
                message=f"{len(missing_files)} model file(s) missing",
                details={"missing": missing_files}
            )

        if stale_files:
            return DivergenceCheck(
                check_name="model_outputs",
                level=DivergenceLevel.WARN,
                message=f"{len(stale_files)} model file(s) stale",
                details={"stale_files": stale_files}
            )

        return DivergenceCheck(
            check_name="model_outputs",
            level=DivergenceLevel.OK,
            message="Model outputs fresh",
            details={"files_checked": len(model_files)}
        )

    def check_kill_switch(self) -> DivergenceCheck:
        """Check if kill switch is active."""
        kill_switch_file = STATE_DIR / "KILL_SWITCH"
        if kill_switch_file.exists():
            return DivergenceCheck(
                check_name="kill_switch",
                level=DivergenceLevel.CRITICAL,
                message="KILL SWITCH IS ACTIVE",
                details={"file": str(kill_switch_file)}
            )

        return DivergenceCheck(
            check_name="kill_switch",
            level=DivergenceLevel.OK,
            message="Kill switch inactive",
            details={}
        )

    def run_all_checks(self) -> List[DivergenceCheck]:
        """Run all divergence checks."""
        logger.info("=" * 60)
        logger.info("DIVERGENCE MONITOR")
        logger.info(f"Timestamp: {datetime.utcnow().isoformat()}Z")
        logger.info("=" * 60)

        self.checks = [
            self.check_kill_switch(),
            self.check_api_health(),
            self.check_order_fill_status(),
            self.check_position_mismatch(),
            self.check_quote_staleness(),
            self.check_spread_widening(),
            self.check_model_outputs(),
        ]

        # Log and print results
        for check in self.checks:
            log_divergence(check)

            # Print with color coding
            level_str = f"[{check.level.value.upper():^8}]"
            if check.level == DivergenceLevel.CRITICAL:
                logger.error(f"{level_str} {check.check_name}: {check.message}")
            elif check.level == DivergenceLevel.WARN:
                logger.warning(f"{level_str} {check.check_name}: {check.message}")
            elif check.level == DivergenceLevel.ERROR:
                logger.error(f"{level_str} {check.check_name}: {check.message}")
            else:
                logger.info(f"{level_str} {check.check_name}: {check.message}")

            if self.verbose and check.details:
                logger.info(f"          Details: {json.dumps(check.details, indent=2)}")

        # Summary
        critical_count = sum(1 for c in self.checks if c.level == DivergenceLevel.CRITICAL)
        warn_count = sum(1 for c in self.checks if c.level == DivergenceLevel.WARN)
        error_count = sum(1 for c in self.checks if c.level == DivergenceLevel.ERROR)

        logger.info("\n" + "-" * 60)
        if critical_count > 0:
            logger.error(f"SUMMARY: {critical_count} CRITICAL, {warn_count} WARN, {error_count} ERROR")
        elif warn_count > 0 or error_count > 0:
            logger.warning(f"SUMMARY: {critical_count} CRITICAL, {warn_count} WARN, {error_count} ERROR")
        else:
            logger.info("SUMMARY: All checks passed")

        return self.checks

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all checks."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_checks": len(self.checks),
            "critical": sum(1 for c in self.checks if c.level == DivergenceLevel.CRITICAL),
            "warn": sum(1 for c in self.checks if c.level == DivergenceLevel.WARN),
            "error": sum(1 for c in self.checks if c.level == DivergenceLevel.ERROR),
            "ok": sum(1 for c in self.checks if c.level == DivergenceLevel.OK),
            "checks": [asdict(c) for c in self.checks]
        }


def main():
    parser = argparse.ArgumentParser(
        description="Kobe Divergence Monitor - System health and sync checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python monitor/divergence_monitor.py --dotenv ./.env
    python monitor/divergence_monitor.py --verbose
        """
    )
    parser.add_argument("--dotenv", type=str, default="./.env",
                        help="Path to .env file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output with details")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON summary")

    args = parser.parse_args()

    # Load environment
    from config.env_loader import load_env
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # Run monitor
    try:
        monitor = DivergenceMonitor(verbose=args.verbose)
        checks = monitor.run_all_checks()

        if args.json:
            print(json.dumps(monitor.get_summary(), indent=2))

        # Exit code based on results
        critical_count = sum(1 for c in checks if c.level == DivergenceLevel.CRITICAL)
        if critical_count > 0:
            sys.exit(2)

        error_count = sum(1 for c in checks if c.level == DivergenceLevel.ERROR)
        if error_count > 0:
            sys.exit(1)

        sys.exit(0)
    except Exception as e:
        logger.error(f"Divergence monitor failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
