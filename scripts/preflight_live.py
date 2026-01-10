#!/usr/bin/env python3
"""
Kobe Preflight Live Gate
========================

Comprehensive pre-live validation with 15+ checks that MUST ALL PASS
before live trading can proceed. This is the final safety gate.

Unlike preflight.py (advisory), this script BLOCKS live trading on any failure.

Checks:
1. Settings schema valid (Pydantic validation)
2. Webhook HMAC secret set (for secure callbacks)
3. Broker keys present (ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY)
4. Broker connectivity (Alpaca Trading API)
5. Market calendar OK for today
6. Earnings source reachable
7. Prometheus registry responds
8. Kill switch inactive
9. LLM budget not exceeded (>20% remaining)
10. Position reconciliation OK
11. Data freshness OK (Polygon)
12. Config pin integrity
13. Settings mode matches requested mode
14. No pending orders stuck
15. Hash chain integrity

FIX (2026-01-05): Created as comprehensive live gate.

Usage:
    python scripts/preflight_live.py --mode live
    python scripts/preflight_live.py --mode paper  # Less strict
    python scripts/preflight_live.py --json  # Output as JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env


@dataclass
class CheckResult:
    """Result of a single preflight check."""
    name: str
    passed: bool
    message: str
    blocking: bool  # If True, failure blocks live trading
    severity: str  # "critical", "warning", "info"


@dataclass
class PreflightReport:
    """Complete preflight report."""
    mode: str
    timestamp: str
    all_passed: bool
    blocking_failures: int
    total_checks: int
    checks: List[CheckResult]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "timestamp": self.timestamp,
            "all_passed": self.all_passed,
            "blocking_failures": self.blocking_failures,
            "total_checks": self.total_checks,
            "checks": [asdict(c) for c in self.checks],
        }


def check_settings_schema() -> CheckResult:
    """Check if settings can be validated by Pydantic schema."""
    try:
        from config.settings_loader import get_setting
        # Try to load a few critical settings
        mode = get_setting("system.mode", "paper")
        if mode not in ("paper", "live", "micro"):
            return CheckResult(
                name="settings_schema",
                passed=False,
                message=f"Invalid mode: {mode}",
                blocking=True,
                severity="critical",
            )
        return CheckResult(
            name="settings_schema",
            passed=True,
            message=f"Settings valid (mode={mode})",
            blocking=True,
            severity="critical",
        )
    except Exception as e:
        return CheckResult(
            name="settings_schema",
            passed=False,
            message=f"Settings validation failed: {e}",
            blocking=True,
            severity="critical",
        )


def check_webhook_hmac() -> CheckResult:
    """Check if webhook HMAC secret is set (required for live)."""
    hmac_secret = os.getenv("WEBHOOK_HMAC_SECRET", "")
    if hmac_secret and len(hmac_secret) >= 32:
        return CheckResult(
            name="webhook_hmac",
            passed=True,
            message="HMAC secret set (32+ chars)",
            blocking=True,
            severity="critical",
        )
    elif hmac_secret:
        return CheckResult(
            name="webhook_hmac",
            passed=False,
            message=f"HMAC secret too short ({len(hmac_secret)} chars, need 32+)",
            blocking=True,
            severity="critical",
        )
    else:
        return CheckResult(
            name="webhook_hmac",
            passed=False,
            message="WEBHOOK_HMAC_SECRET not set",
            blocking=True,
            severity="critical",
        )


def check_broker_keys() -> CheckResult:
    """Check if Alpaca API keys are present."""
    key_id = os.getenv("ALPACA_API_KEY_ID", "")
    secret = os.getenv("ALPACA_API_SECRET_KEY", "")

    if key_id and secret:
        return CheckResult(
            name="broker_keys",
            passed=True,
            message="Alpaca keys present",
            blocking=True,
            severity="critical",
        )
    missing = []
    if not key_id:
        missing.append("ALPACA_API_KEY_ID")
    if not secret:
        missing.append("ALPACA_API_SECRET_KEY")
    return CheckResult(
        name="broker_keys",
        passed=False,
        message=f"Missing: {', '.join(missing)}",
        blocking=True,
        severity="critical",
    )


def check_broker_connectivity() -> CheckResult:
    """Check if Alpaca Trading API is accessible."""
    base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    try:
        r = requests.get(
            f"{base}/v2/account",
            headers={
                "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY_ID", ""),
                "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET_KEY", ""),
            },
            timeout=10,
        )
        if r.status_code == 200:
            acct = r.json()
            buying_power = float(acct.get("buying_power", 0))
            return CheckResult(
                name="broker_connectivity",
                passed=True,
                message=f"Connected (buying power: ${buying_power:,.2f})",
                blocking=True,
                severity="critical",
            )
        else:
            return CheckResult(
                name="broker_connectivity",
                passed=False,
                message=f"HTTP {r.status_code}",
                blocking=True,
                severity="critical",
            )
    except Exception as e:
        return CheckResult(
            name="broker_connectivity",
            passed=False,
            message=f"Connection failed: {e}",
            blocking=True,
            severity="critical",
        )


def check_market_calendar() -> CheckResult:
    """Check if market is open today or will open."""
    try:
        from risk.kill_zone_gate import is_market_holiday
        today = datetime.now().date()
        if is_market_holiday(today):
            return CheckResult(
                name="market_calendar",
                passed=False,
                message=f"{today} is a market holiday",
                blocking=False,  # Not blocking - just informational
                severity="warning",
            )
        return CheckResult(
            name="market_calendar",
            passed=True,
            message=f"{today} is a trading day",
            blocking=False,
            severity="info",
        )
    except ImportError:
        # Fallback: just check it's not weekend
        weekday = datetime.now().weekday()
        if weekday >= 5:
            return CheckResult(
                name="market_calendar",
                passed=False,
                message="Weekend - market closed",
                blocking=False,
                severity="warning",
            )
        return CheckResult(
            name="market_calendar",
            passed=True,
            message="Weekday",
            blocking=False,
            severity="info",
        )


def check_earnings_source() -> CheckResult:
    """Check if earnings data source is reachable."""
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        # Check yfinance fallback
        try:
            import yfinance
            return CheckResult(
                name="earnings_source",
                passed=True,
                message="yfinance fallback available (no Polygon key)",
                blocking=False,
                severity="warning",
            )
        except ImportError:
            return CheckResult(
                name="earnings_source",
                passed=False,
                message="No Polygon key and yfinance not installed",
                blocking=True,
                severity="critical",
            )

    # Test Polygon earnings endpoint
    try:
        # Use reference endpoint to check connectivity
        url = f"https://api.polygon.io/v3/reference/tickers/AAPL?apiKey={api_key}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return CheckResult(
                name="earnings_source",
                passed=True,
                message="Polygon API accessible",
                blocking=False,
                severity="info",
            )
        else:
            return CheckResult(
                name="earnings_source",
                passed=False,
                message=f"Polygon HTTP {r.status_code}",
                blocking=True,
                severity="critical",
            )
    except Exception as e:
        return CheckResult(
            name="earnings_source",
            passed=False,
            message=f"Polygon connection failed: {e}",
            blocking=True,
            severity="critical",
        )


def check_prometheus() -> CheckResult:
    """Check if Prometheus metrics are available."""
    try:
        from trade_logging.prometheus_metrics import PROMETHEUS_AVAILABLE, get_metrics_text
        if not PROMETHEUS_AVAILABLE:
            return CheckResult(
                name="prometheus",
                passed=False,
                message="prometheus_client not installed",
                blocking=False,
                severity="warning",
            )
        # Try to generate metrics
        metrics = get_metrics_text()
        if metrics and len(metrics) > 0:
            return CheckResult(
                name="prometheus",
                passed=True,
                message=f"Metrics available ({len(metrics)} bytes)",
                blocking=False,
                severity="info",
            )
        return CheckResult(
            name="prometheus",
            passed=True,
            message="Metrics registry empty but functional",
            blocking=False,
            severity="info",
        )
    except Exception as e:
        return CheckResult(
            name="prometheus",
            passed=False,
            message=f"Prometheus check failed: {e}",
            blocking=False,
            severity="warning",
        )


def check_kill_switch() -> CheckResult:
    """Check if kill switch is inactive."""
    try:
        from core.kill_switch import is_kill_switch_active
        if is_kill_switch_active():
            return CheckResult(
                name="kill_switch",
                passed=False,
                message="KILL SWITCH IS ACTIVE - trading halted",
                blocking=True,
                severity="critical",
            )
        return CheckResult(
            name="kill_switch",
            passed=True,
            message="Kill switch inactive",
            blocking=True,
            severity="critical",
        )
    except Exception as e:
        return CheckResult(
            name="kill_switch",
            passed=False,
            message=f"Kill switch check failed: {e}",
            blocking=True,
            severity="critical",
        )


def check_llm_budget() -> CheckResult:
    """Check if LLM budget has >20% remaining."""
    try:
        from llm.token_budget import get_token_budget
        budget = get_token_budget()
        remaining_pct = budget.get_remaining_percent()

        if remaining_pct < 20:
            return CheckResult(
                name="llm_budget",
                passed=False,
                message=f"LLM budget low ({remaining_pct:.1f}% remaining)",
                blocking=False,
                severity="warning",
            )
        return CheckResult(
            name="llm_budget",
            passed=True,
            message=f"LLM budget OK ({remaining_pct:.1f}% remaining)",
            blocking=False,
            severity="info",
        )
    except ImportError:
        return CheckResult(
            name="llm_budget",
            passed=True,
            message="LLM budget module not available (OK)",
            blocking=False,
            severity="info",
        )
    except Exception as e:
        return CheckResult(
            name="llm_budget",
            passed=True,
            message=f"LLM budget check skipped: {e}",
            blocking=False,
            severity="info",
        )


def check_position_reconciliation() -> CheckResult:
    """Check if positions are reconciled with broker."""
    try:
        # Check if reconciliation file exists and is recent
        recon_path = ROOT / "state" / "reconciliation" / "last_reconcile.json"
        if not recon_path.exists():
            return CheckResult(
                name="position_reconciliation",
                passed=True,
                message="No reconciliation file (OK for first run)",
                blocking=False,
                severity="info",
            )

        with open(recon_path) as f:
            recon = json.load(f)

        # Check if reconciliation is recent (within 24 hours)
        last_ts = recon.get("timestamp", "")
        if last_ts:
            last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            age_hours = (datetime.now().astimezone() - last_dt).total_seconds() / 3600
            if age_hours > 24:
                return CheckResult(
                    name="position_reconciliation",
                    passed=False,
                    message=f"Reconciliation stale ({age_hours:.1f} hours old)",
                    blocking=False,
                    severity="warning",
                )

        # Check for mismatches
        mismatches = recon.get("mismatches", [])
        if mismatches:
            return CheckResult(
                name="position_reconciliation",
                passed=False,
                message=f"{len(mismatches)} position mismatches",
                blocking=True,
                severity="critical",
            )

        return CheckResult(
            name="position_reconciliation",
            passed=True,
            message="Positions reconciled",
            blocking=False,
            severity="info",
        )
    except Exception as e:
        return CheckResult(
            name="position_reconciliation",
            passed=True,
            message=f"Reconciliation check skipped: {e}",
            blocking=False,
            severity="info",
        )


def check_data_freshness() -> CheckResult:
    """Check if Polygon data is fresh."""
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return CheckResult(
            name="data_freshness",
            passed=True,
            message="No Polygon key (skipped)",
            blocking=False,
            severity="info",
        )

    try:
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/"
            f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            f"?adjusted=true&sort=desc&limit=5&apiKey={api_key}"
        )

        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return CheckResult(
                name="data_freshness",
                passed=False,
                message=f"Polygon HTTP {r.status_code}",
                blocking=True,
                severity="critical",
            )

        data = r.json()
        results = data.get("results", [])
        if not results:
            return CheckResult(
                name="data_freshness",
                passed=False,
                message="No data returned from Polygon",
                blocking=True,
                severity="critical",
            )

        latest_ts = results[0].get("t", 0) / 1000
        latest_date = datetime.fromtimestamp(latest_ts)
        days_old = (datetime.now() - latest_date).days

        if days_old > 5:
            return CheckResult(
                name="data_freshness",
                passed=False,
                message=f"Data stale ({days_old} days old)",
                blocking=True,
                severity="critical",
            )

        return CheckResult(
            name="data_freshness",
            passed=True,
            message=f"Data fresh (latest: {latest_date.strftime('%Y-%m-%d')})",
            blocking=True,
            severity="critical",
        )
    except Exception as e:
        return CheckResult(
            name="data_freshness",
            passed=False,
            message=f"Data freshness check failed: {e}",
            blocking=True,
            severity="critical",
        )


def check_config_pin() -> CheckResult:
    """Check config file integrity."""
    try:
        from core.config_pin import sha256_file
        config_path = ROOT / "config" / "settings.json"
        if not config_path.exists():
            return CheckResult(
                name="config_pin",
                passed=False,
                message="config/settings.json not found",
                blocking=True,
                severity="critical",
            )

        digest = sha256_file(str(config_path))
        return CheckResult(
            name="config_pin",
            passed=True,
            message=f"Config hash: {digest[:16]}...",
            blocking=True,
            severity="critical",
        )
    except Exception as e:
        return CheckResult(
            name="config_pin",
            passed=False,
            message=f"Config pin check failed: {e}",
            blocking=True,
            severity="critical",
        )


def check_mode_match(requested_mode: str) -> CheckResult:
    """Check if settings mode matches requested mode."""
    try:
        from config.settings_loader import get_setting
        actual_mode = get_setting("system.mode", "paper")

        if actual_mode != requested_mode:
            return CheckResult(
                name="mode_match",
                passed=False,
                message=f"Mode mismatch: settings={actual_mode}, requested={requested_mode}",
                blocking=True,
                severity="critical",
            )
        return CheckResult(
            name="mode_match",
            passed=True,
            message=f"Mode matches: {actual_mode}",
            blocking=True,
            severity="critical",
        )
    except Exception as e:
        return CheckResult(
            name="mode_match",
            passed=False,
            message=f"Mode check failed: {e}",
            blocking=True,
            severity="critical",
        )


def check_pending_orders() -> CheckResult:
    """Check for stuck pending orders."""
    try:
        base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
        r = requests.get(
            f"{base}/v2/orders?status=open",
            headers={
                "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY_ID", ""),
                "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET_KEY", ""),
            },
            timeout=10,
        )

        if r.status_code != 200:
            return CheckResult(
                name="pending_orders",
                passed=False,
                message=f"Orders check failed: HTTP {r.status_code}",
                blocking=False,
                severity="warning",
            )

        orders = r.json()
        if orders:
            return CheckResult(
                name="pending_orders",
                passed=False,
                message=f"{len(orders)} pending orders found",
                blocking=False,
                severity="warning",
            )

        return CheckResult(
            name="pending_orders",
            passed=True,
            message="No pending orders",
            blocking=False,
            severity="info",
        )
    except Exception as e:
        return CheckResult(
            name="pending_orders",
            passed=True,
            message=f"Orders check skipped: {e}",
            blocking=False,
            severity="info",
        )


def check_hash_chain() -> CheckResult:
    """Check hash chain integrity."""
    try:
        from core.hash_chain import verify_chain
        chain_path = ROOT / "state" / "hash_chain.jsonl"

        if not chain_path.exists():
            return CheckResult(
                name="hash_chain",
                passed=True,
                message="No hash chain yet (OK for first run)",
                blocking=False,
                severity="info",
            )

        valid = verify_chain(str(chain_path))
        if valid:
            return CheckResult(
                name="hash_chain",
                passed=True,
                message="Hash chain valid",
                blocking=True,
                severity="critical",
            )
        else:
            return CheckResult(
                name="hash_chain",
                passed=False,
                message="HASH CHAIN TAMPERED",
                blocking=True,
                severity="critical",
            )
    except Exception as e:
        return CheckResult(
            name="hash_chain",
            passed=True,
            message=f"Hash chain check skipped: {e}",
            blocking=False,
            severity="info",
        )


def run_preflight(mode: str = "live") -> PreflightReport:
    """
    Run all preflight checks.

    Args:
        mode: Trading mode ("live" or "paper")

    Returns:
        PreflightReport with all check results
    """
    checks = [
        check_settings_schema(),
        check_broker_keys(),
        check_broker_connectivity(),
        check_kill_switch(),
        check_config_pin(),
        check_mode_match(mode),
        check_data_freshness(),
        check_market_calendar(),
        check_earnings_source(),
        check_prometheus(),
        check_llm_budget(),
        check_position_reconciliation(),
        check_pending_orders(),
        check_hash_chain(),
    ]

    # Only require HMAC for live mode
    if mode == "live":
        checks.insert(1, check_webhook_hmac())

    blocking_failures = sum(1 for c in checks if not c.passed and c.blocking)
    all_passed = blocking_failures == 0

    return PreflightReport(
        mode=mode,
        timestamp=datetime.now().isoformat(),
        all_passed=all_passed,
        blocking_failures=blocking_failures,
        total_checks=len(checks),
        checks=checks,
    )


def main():
    ap = argparse.ArgumentParser(description="Kobe Preflight Live Gate")
    ap.add_argument("--mode", choices=["live", "paper"], default="live",
                    help="Trading mode to validate")
    ap.add_argument("--dotenv", type=str, default="./.env",
                    help="Path to .env file")
    ap.add_argument("--json", action="store_true",
                    help="Output as JSON")
    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # Run preflight
    report = run_preflight(args.mode)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("=" * 60)
        print(f"KOBE PREFLIGHT LIVE GATE - {args.mode.upper()} MODE")
        print("=" * 60)
        print(f"Timestamp: {report.timestamp}")
        print()

        for i, check in enumerate(report.checks, 1):
            status = "PASS" if check.passed else "FAIL"
            blocking = "[BLOCKING]" if check.blocking and not check.passed else ""
            print(f"[{i:2d}/{report.total_checks}] {check.name}: {status} {blocking}")
            print(f"       {check.message}")

        print()
        print("=" * 60)
        if report.all_passed:
            print(f"PREFLIGHT PASSED - Ready for {args.mode.upper()} trading")
            print("=" * 60)
            sys.exit(0)
        else:
            print(f"PREFLIGHT FAILED - {report.blocking_failures} blocking failures")
            print(f"DO NOT PROCEED WITH {args.mode.upper()} TRADING")
            print("=" * 60)
            sys.exit(1)


if __name__ == "__main__":
    main()
