#!/usr/bin/env python3
"""
Backtest vs Live Parity Validation Script

FIX (2026-01-08): Critical validation to ensure backtest results match live execution.

This script:
1. Documents all known differences between backtest and live execution
2. Validates configuration alignment
3. Runs parity tests
4. Provides recommendations for alignment

Usage:
    python scripts/validate_backtest_live_parity.py
    python scripts/validate_backtest_live_parity.py --fix  # Apply recommended fixes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.structured_log import jlog

# Known parity gaps between backtest and live
PARITY_GAPS = [
    {
        "id": "GAP-001",
        "name": "Fill Price Assumption",
        "severity": "MEDIUM",
        "backtest": "Next bar OPEN + 5 bps slippage",
        "live": "IOC LIMIT at best_ask × 1.001 (10 bps)",
        "impact_wr": "-2% to -5%",
        "fix": "Align slippage_bps in backtest to 10 bps",
        "file": "backtest/engine.py:29",
    },
    {
        "id": "GAP-002",
        "name": "Stop Loss Execution",
        "severity": "CRITICAL",
        "backtest": "Auto-exits at stop price when low <= stop",
        "live": "Requires explicit stop order placement (BUG FIXED 2026-01-08)",
        "impact_wr": "-3% to -8%",
        "fix": "Stop coverage bug fixed in runner.py - now detects missing stops",
        "file": "scripts/runner.py:425-455",
    },
    {
        "id": "GAP-003",
        "name": "Time Stop Calculation",
        "severity": "HIGH",
        "backtest": "Counts trading bars in historical data",
        "live": "Was using calendar days (BUG FIXED 2026-01-08)",
        "impact_wr": "-2% to -4%",
        "fix": "Fixed in exit_manager.py and position_manager.py to use NYSE calendar",
        "file": "scripts/exit_manager.py:170-218",
    },
    {
        "id": "GAP-004",
        "name": "Position Sizing",
        "severity": "HIGH",
        "backtest": "0.5% risk per trade (config default)",
        "live": "2.5% risk per trade (medium mode)",
        "impact_wr": "-1% to -3%",
        "fix": "Align backtest to use same risk% as live mode",
        "file": "backtest/engine.py:278",
    },
    {
        "id": "GAP-005",
        "name": "Kill Zone Enforcement",
        "severity": "MEDIUM",
        "backtest": "Allows signals at any time",
        "live": "Blocks 9:30-10:00 (opening range) and 11:30-14:30 (lunch)",
        "impact_wr": "+2% to +5% (live is BETTER)",
        "fix": "Add kill zone filter to backtest signal generation",
        "file": "backtest/engine.py",
    },
    {
        "id": "GAP-006",
        "name": "Quality Gate Thresholds",
        "severity": "MEDIUM",
        "backtest": "Score >= 70, Confidence >= 0.60 enforced",
        "live": "Same thresholds but may be bypassed in some code paths",
        "impact_wr": "-2% to -4%",
        "fix": "Audit all entry points to ensure quality gate is enforced",
        "file": "strategies/signal_quality_gate.py",
    },
]


@dataclass
class ParityCheckResult:
    """Result of a single parity check."""
    gap_id: str
    name: str
    passed: bool
    backtest_value: str
    live_value: str
    recommendation: str


def load_config() -> Dict:
    """Load base configuration."""
    import yaml
    config_path = ROOT / "config" / "base.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def check_slippage_parity() -> ParityCheckResult:
    """Check if backtest slippage matches live slippage."""
    # Read actual backtest slippage from engine.py
    engine_path = ROOT / "backtest" / "engine.py"
    backtest_slippage_bps = 5.0  # default
    if engine_path.exists():
        content = engine_path.read_text()
        import re
        match = re.search(r'slippage_bps:\s*float\s*=\s*([\d.]+)', content)
        if match:
            backtest_slippage_bps = float(match.group(1))

    # Live uses best_ask × 1.001 = ~10 bps
    live_slippage_bps = 10.0

    passed = abs(backtest_slippage_bps - live_slippage_bps) < 2

    return ParityCheckResult(
        gap_id="GAP-001",
        name="Fill Price Slippage",
        passed=passed,
        backtest_value=f"{backtest_slippage_bps} bps",
        live_value=f"~{live_slippage_bps} bps (best_ask x 1.001)",
        recommendation="Update backtest/engine.py slippage_bps to 10" if not passed else "OK"
    )


def check_risk_pct_parity() -> ParityCheckResult:
    """Check if backtest risk% matches live risk%."""
    config = load_config()

    # Backtest uses sizing.risk_per_trade_pct (default 0.5%)
    sizing_config = config.get("sizing", {})
    backtest_risk_pct = sizing_config.get("risk_per_trade_pct", 0.005) * 100

    # Live paper uses policy.modes.medium.risk_per_trade_pct (2.5%)
    policy_config = config.get("policy", {})
    modes = policy_config.get("modes", {})
    medium_mode = modes.get("medium", {})
    live_risk_pct = medium_mode.get("risk_per_trade_pct", 0.025) * 100

    passed = abs(backtest_risk_pct - live_risk_pct) < 0.5

    return ParityCheckResult(
        gap_id="GAP-004",
        name="Position Sizing Risk%",
        passed=passed,
        backtest_value=f"{backtest_risk_pct:.1f}%",
        live_value=f"{live_risk_pct:.1f}%",
        recommendation=f"Update sizing.risk_per_trade_pct to {live_risk_pct/100}" if not passed else "OK"
    )


def check_time_stop_parity() -> ParityCheckResult:
    """Check if time stop calculation is fixed."""
    # Check if market_calendar has get_trading_days_between
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from market_calendar import get_trading_days_between
        has_function = True
    except ImportError:
        has_function = False

    return ParityCheckResult(
        gap_id="GAP-003",
        name="Time Stop Trading Days",
        passed=has_function,
        backtest_value="Uses trading bars",
        live_value="NYSE calendar (FIXED)" if has_function else "Calendar days (BUG)",
        recommendation="OK - Fixed 2026-01-08" if has_function else "Add get_trading_days_between to market_calendar.py"
    )


def check_stop_coverage_parity() -> ParityCheckResult:
    """Check if stop coverage bug is fixed."""
    # Check runner.py for symbols_with_stop_coverage
    runner_path = ROOT / "scripts" / "runner.py"
    if runner_path.exists():
        content = runner_path.read_text()
        has_fix = "symbols_with_stop_coverage" in content
    else:
        has_fix = False

    return ParityCheckResult(
        gap_id="GAP-002",
        name="Stop Loss Coverage Logic",
        passed=has_fix,
        backtest_value="Auto-exits at stop",
        live_value="STOP orders tracked separately (FIXED)" if has_fix else "All sells counted (BUG)",
        recommendation="OK - Fixed 2026-01-08" if has_fix else "Fix runner.py to track STOP orders only"
    )


def check_paper_mode_assertion() -> ParityCheckResult:
    """Check if paper mode URL assertion exists."""
    runner_path = ROOT / "scripts" / "runner.py"
    if runner_path.exists():
        content = runner_path.read_text()
        has_assertion = "paper_mode_url_mismatch" in content or "SAFETY VIOLATION" in content
    else:
        has_assertion = False

    return ParityCheckResult(
        gap_id="SAFETY-001",
        name="Paper Mode URL Assertion",
        passed=has_assertion,
        backtest_value="N/A",
        live_value="Runtime assertion (ADDED)" if has_assertion else "No assertion (RISK)",
        recommendation="OK - Added 2026-01-08" if has_assertion else "Add paper URL check to runner.py"
    )


def check_kill_zone_parity() -> ParityCheckResult:
    """Check if backtest applies kill zone filtering."""
    engine_path = ROOT / "backtest" / "engine.py"
    if engine_path.exists():
        content = engine_path.read_text()
        has_filter = "_filter_signals_by_kill_zone" in content
    else:
        has_filter = False

    return ParityCheckResult(
        gap_id="GAP-005",
        name="Kill Zone Enforcement",
        passed=has_filter,
        backtest_value="Kill zones applied (10:00-11:30, 14:30-15:30)" if has_filter else "All times allowed",
        live_value="10:00-11:30, 14:30-15:30 only",
        recommendation="OK - Kill zone filter added 2026-01-08" if has_filter else "Add kill zone filter to backtest/engine.py"
    )


def check_quality_gate_parity() -> ParityCheckResult:
    """Check if paper trade applies quality gate."""
    paper_path = ROOT / "scripts" / "run_paper_trade.py"
    if paper_path.exists():
        content = paper_path.read_text()
        has_gate = "filter_to_best_signals" in content
    else:
        has_gate = False

    return ParityCheckResult(
        gap_id="GAP-006",
        name="Quality Gate in Paper Trade",
        passed=has_gate,
        backtest_value="Score >= 70 enforced",
        live_value="Quality gate applied" if has_gate else "BYPASSED (cognitive only)",
        recommendation="OK - Quality gate added 2026-01-08" if has_gate else "Add filter_to_best_signals to run_paper_trade.py"
    )


def run_all_checks() -> List[ParityCheckResult]:
    """Run all parity checks."""
    checks = [
        check_slippage_parity(),
        check_risk_pct_parity(),
        check_time_stop_parity(),
        check_stop_coverage_parity(),
        check_paper_mode_assertion(),
        check_kill_zone_parity(),       # FIX (2026-01-08): Added for GAP-005
        check_quality_gate_parity(),    # FIX (2026-01-08): Added for GAP-006
    ]
    return checks


def print_parity_report(checks: List[ParityCheckResult]) -> None:
    """Print formatted parity report."""
    print("\n" + "=" * 70)
    print("          BACKTEST vs LIVE PARITY VALIDATION REPORT")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    passed_count = sum(1 for c in checks if c.passed)
    total_count = len(checks)

    # Summary
    print(f"SUMMARY: {passed_count}/{total_count} checks passed")
    print("-" * 70)

    # Details
    for check in checks:
        status = "[PASS]" if check.passed else "[FAIL]"
        print(f"\n[{check.gap_id}] {check.name}: {status}")
        print(f"  Backtest: {check.backtest_value}")
        print(f"  Live:     {check.live_value}")
        if not check.passed:
            print(f"  FIX:      {check.recommendation}")

    # Known gaps
    print("\n" + "=" * 70)
    print("          KNOWN PARITY GAPS (Documentation)")
    print("=" * 70)

    for gap in PARITY_GAPS:
        severity_icon = {"CRITICAL": "[!!!]", "HIGH": "[!!]", "MEDIUM": "[!]", "LOW": "[.]"}.get(gap["severity"], "[?]")
        print(f"\n{severity_icon} [{gap['id']}] {gap['name']} ({gap['severity']})")
        print(f"  Backtest: {gap['backtest']}")
        print(f"  Live:     {gap['live']}")
        print(f"  Impact:   {gap['impact_wr']} WR")
        print(f"  Fix:      {gap['fix']}")
        print(f"  File:     {gap['file']}")

    # Recommendations
    print("\n" + "=" * 70)
    print("          RECOMMENDATIONS")
    print("=" * 70)

    failed_checks = [c for c in checks if not c.passed]
    if failed_checks:
        print("\nCRITICAL FIXES NEEDED:")
        for i, check in enumerate(failed_checks, 1):
            print(f"  {i}. [{check.gap_id}] {check.recommendation}")
    else:
        print("\n[OK] All automated checks passed!")

    print("\nMANUAL ACTIONS:")
    print("  1. Place STOP orders on TSLA ($425.45) and CFG ($56.33) via Alpaca UI")
    print("  2. Verify bars counter shows correct values (2-3/5 for Jan 5 entries)")
    print("  3. Run runner.py and check for CRITICAL_NO_STOP_PROTECTION warnings")

    print("\n" + "=" * 70)


def apply_fixes(dry_run: bool = True) -> None:
    """Apply recommended fixes to align backtest with live."""
    print("\n" + "=" * 70)
    print("          APPLYING PARITY FIXES" + (" (DRY RUN)" if dry_run else ""))
    print("=" * 70)

    # Fix 1: Update backtest slippage to 10 bps
    engine_path = ROOT / "backtest" / "engine.py"
    if engine_path.exists():
        content = engine_path.read_text()
        if "slippage_bps: float = 5.0" in content:
            if dry_run:
                print("\n[DRY RUN] Would update backtest/engine.py slippage_bps: 5.0 -> 10.0")
            else:
                new_content = content.replace(
                    "slippage_bps: float = 5.0",
                    "slippage_bps: float = 10.0  # FIX (2026-01-08): Aligned with live IOC LIMIT"
                )
                engine_path.write_text(new_content)
                print("\n[OK] Updated backtest/engine.py slippage_bps: 5.0 -> 10.0")

    # Fix 2: Update config risk_per_trade_pct
    config_path = ROOT / "config" / "base.yaml"
    if config_path.exists():
        content = config_path.read_text()
        if "risk_per_trade_pct: 0.005" in content:
            if dry_run:
                print("[DRY RUN] Would update config/base.yaml sizing.risk_per_trade_pct: 0.005 -> 0.025")
            else:
                # Only update the sizing section, not policy modes
                # This is tricky with YAML, so we'll note it needs manual update
                print("\n[WARNING] Manual action needed: Update config/base.yaml sizing.risk_per_trade_pct to 0.025")

    if dry_run:
        print("\nRun with --fix to apply changes (without --dry-run)")


def main():
    parser = argparse.ArgumentParser(description="Validate backtest vs live parity")
    parser.add_argument("--fix", action="store_true", help="Apply recommended fixes")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Show what would be changed")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Run checks
    checks = run_all_checks()

    if args.json:
        output = {
            "timestamp": datetime.now().isoformat(),
            "checks": [
                {
                    "gap_id": c.gap_id,
                    "name": c.name,
                    "passed": c.passed,
                    "backtest_value": c.backtest_value,
                    "live_value": c.live_value,
                    "recommendation": c.recommendation,
                }
                for c in checks
            ],
            "known_gaps": PARITY_GAPS,
            "passed_count": sum(1 for c in checks if c.passed),
            "total_count": len(checks),
        }
        print(json.dumps(output, indent=2))
    else:
        print_parity_report(checks)

        if args.fix:
            apply_fixes(dry_run=not args.fix)

    # Log result
    passed_count = sum(1 for c in checks if c.passed)
    jlog("parity_validation",
         passed=passed_count,
         total=len(checks),
         failed=[c.gap_id for c in checks if not c.passed])

    # Exit code based on results
    sys.exit(0 if passed_count == len(checks) else 1)


if __name__ == "__main__":
    main()
