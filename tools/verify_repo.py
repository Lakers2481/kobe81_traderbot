#!/usr/bin/env python3
"""
verify_repo.py - One-Command Repository Verification

Verifies that the Kobe trading robot is properly configured and all
critical components are wired correctly.

Usage:
    python tools/verify_repo.py
    python tools/verify_repo.py --verbose
    python tools/verify_repo.py --fix  # Attempt auto-fixes
"""

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def check_mark(passed: bool) -> str:
    """Return check mark or X based on pass/fail."""
    if passed:
        return f"{Colors.GREEN}[PASS]{Colors.RESET}"
    return f"{Colors.RED}[FAIL]{Colors.RESET}"


def verify_documentation() -> Tuple[int, int, List[str]]:
    """Verify all required documentation exists."""
    required_docs = [
        "CLAUDE.md",
        "docs/STATUS.md",
        "docs/READINESS.md",
        "docs/ARCHITECTURE.md",
        "docs/REPO_MAP.md",
        "docs/ENTRYPOINTS.md",
        "docs/KNOWN_GAPS.md",
        "docs/RISK_REGISTER.md",
        "docs/START_HERE.md",
        "docs/ROBOT_MANUAL.md",
        "docs/WORKLOG.md",
        "docs/CHANGELOG.md",
        "docs/CONTRIBUTING.md",
        "docs/JOBS_AND_SCHEDULER.md",
    ]

    passed = 0
    failed = 0
    issues = []

    for doc in required_docs:
        path = PROJECT_ROOT / doc
        if path.exists():
            passed += 1
        else:
            failed += 1
            issues.append(f"Missing: {doc}")

    return passed, failed, issues


def verify_critical_modules() -> Tuple[int, int, List[str]]:
    """Verify critical Python modules can be imported."""
    critical_modules = [
        ("strategies.dual_strategy.combined", "DualStrategyScanner"),
        ("strategies.registry", "get_production_scanner"),
        ("risk.policy_gate", "PolicyGate"),
        ("risk.kill_zone_gate", "can_trade_now"),
        ("risk.equity_sizer", "calculate_position_size"),
        ("execution.broker_alpaca", "AlpacaBroker"),
        ("oms.idempotency_store", "IdempotencyStore"),
        ("core.kill_switch", "is_kill_switch_active"),
        ("core.structured_log", "jlog"),
        ("data.providers.polygon_eod", "fetch_daily_bars_polygon"),
    ]

    passed = 0
    failed = 0
    issues = []

    for module_name, symbol in critical_modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, symbol):
                passed += 1
            else:
                failed += 1
                issues.append(f"Missing symbol: {module_name}.{symbol}")
        except ImportError as e:
            failed += 1
            issues.append(f"Import error: {module_name} - {e}")

    return passed, failed, issues


def verify_state_directories() -> Tuple[int, int, List[str]]:
    """Verify state directories exist."""
    required_dirs = [
        "state",
        "state/watchlist",
        "logs",
        "cache",
        "data/universe",
    ]

    passed = 0
    failed = 0
    issues = []

    for dir_path in required_dirs:
        path = PROJECT_ROOT / dir_path
        if path.exists() and path.is_dir():
            passed += 1
        else:
            failed += 1
            issues.append(f"Missing directory: {dir_path}")

    return passed, failed, issues


def verify_critical_files() -> Tuple[int, int, List[str]]:
    """Verify critical data files exist."""
    critical_files = [
        "data/universe/optionable_liquid_900.csv",
        "config/base.yaml",
        ".gitignore",
    ]

    passed = 0
    failed = 0
    issues = []

    for file_path in critical_files:
        path = PROJECT_ROOT / file_path
        if path.exists():
            passed += 1
        else:
            failed += 1
            issues.append(f"Missing file: {file_path}")

    return passed, failed, issues


def verify_no_standalone_strategies() -> Tuple[int, int, List[str]]:
    """Verify no standalone strategy imports in scripts.

    Allows intentional legacy imports in specific files that need individual
    strategy access for optimization or comparison purposes:
    - optimize.py: Parameter optimization (needs individual strategy instances)
    - run_showdown_crypto.py: Strategy comparison
    - run_wf_crypto.py: Crypto walk-forward comparison
    """
    bad_imports = [
        "from strategies.ibs_rsi.strategy import IbsRsiStrategy",
        "from strategies.ict.turtle_soup import TurtleSoupStrategy",
    ]

    # Files that are allowed to use deprecated imports (with documented reason)
    allowed_files = {
        "optimize.py",  # Parameter optimization requires individual strategy instances
        "run_showdown_crypto.py",  # Strategy comparison
        "run_wf_crypto.py",  # Crypto walk-forward comparison
    }

    scripts_dir = PROJECT_ROOT / "scripts"

    passed = 0
    failed = 0
    issues = []

    if not scripts_dir.exists():
        return 0, 1, ["Scripts directory not found"]

    for script in scripts_dir.glob("*.py"):
        try:
            # Skip allowed files
            if script.name in allowed_files:
                passed += 1
                continue

            content = script.read_text(encoding="utf-8")
            found_bad = False
            for bad_import in bad_imports:
                if bad_import in content:
                    failed += 1
                    issues.append(f"Bad import in {script.name}: {bad_import[:50]}...")
                    found_bad = True
                    break
            if not found_bad:
                passed += 1
        except Exception as e:
            issues.append(f"Error reading {script.name}: {e}")

    return passed, failed, issues


def verify_kill_switch_not_active() -> Tuple[int, int, List[str]]:
    """Verify kill switch is not accidentally active."""
    kill_switch_path = PROJECT_ROOT / "state" / "KILL_SWITCH"

    if kill_switch_path.exists():
        return 0, 1, ["KILL_SWITCH is active! Run: python scripts/resume.py --confirm"]
    return 1, 0, []


def verify_env_template() -> Tuple[int, int, List[str]]:
    """Verify .env.template exists (not .env which is secret)."""
    env_path = PROJECT_ROOT / ".env"
    template_path = PROJECT_ROOT / ".env.template"

    issues = []
    passed = 0
    failed = 0

    if template_path.exists():
        passed += 1
    else:
        failed += 1
        issues.append("Missing .env.template for new developers")

    if env_path.exists():
        passed += 1
    else:
        # Not a failure, just informational
        issues.append("INFO: .env not found (expected for new clones)")

    return passed, failed, issues


def run_verification(verbose: bool = False) -> Dict:
    """Run all verification checks."""

    checks = [
        ("Documentation", verify_documentation),
        ("Critical Modules", verify_critical_modules),
        ("State Directories", verify_state_directories),
        ("Critical Files", verify_critical_files),
        ("Strategy Imports", verify_no_standalone_strategies),
        ("Kill Switch", verify_kill_switch_not_active),
        ("Environment", verify_env_template),
    ]

    total_passed = 0
    total_failed = 0
    all_issues = []

    print(f"\n{Colors.BOLD}=== Kobe Trading Robot Verification ==={Colors.RESET}\n")

    for name, check_func in checks:
        passed, failed, issues = check_func()
        total_passed += passed
        total_failed += failed
        all_issues.extend(issues)

        status = check_mark(failed == 0)
        print(f"{status} {name}: {passed} passed, {failed} failed")

        if verbose and issues:
            for issue in issues:
                print(f"       - {issue}")

    print(f"\n{Colors.BOLD}=== Summary ==={Colors.RESET}")
    print(f"Total Checks: {total_passed + total_failed}")
    print(f"Passed: {Colors.GREEN}{total_passed}{Colors.RESET}")
    print(f"Failed: {Colors.RED}{total_failed}{Colors.RESET}")

    if total_failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All checks passed! System is ready.{Colors.RESET}")
    else:
        print(f"\n{Colors.YELLOW}Issues found. Run with --verbose for details.{Colors.RESET}")

    return {
        "passed": total_passed,
        "failed": total_failed,
        "issues": all_issues,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify Kobe trading robot configuration"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed issue descriptions"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to auto-fix issues (creates missing directories)"
    )

    args = parser.parse_args()

    if args.fix:
        # Create missing directories
        dirs_to_create = [
            "state",
            "state/watchlist",
            "logs",
            "cache",
        ]
        for dir_path in dirs_to_create:
            path = PROJECT_ROOT / dir_path
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                print(f"Created: {dir_path}")

    results = run_verification(verbose=args.verbose)

    # Exit with error code if failures
    sys.exit(1 if results["failed"] > 0 else 0)


if __name__ == "__main__":
    main()
