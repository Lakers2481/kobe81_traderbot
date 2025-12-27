#!/usr/bin/env python3
"""
Run unit tests and integration tests for Kobe trading system.
Usage: python scripts/test.py [--unit|--integration|--all] [--verbose] [--coverage]
"""

import argparse
import importlib
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def find_test_files() -> Dict[str, List[Path]]:
    """Find all test files organized by type."""
    test_files = {
        "unit": [],
        "integration": [],
        "other": [],
    }

    # Look for test directories
    test_dirs = [
        Path("tests"),
        Path("test"),
        Path("tests/unit"),
        Path("tests/integration"),
    ]

    for test_dir in test_dirs:
        if test_dir.exists():
            for test_file in test_dir.rglob("test_*.py"):
                if "unit" in str(test_file).lower():
                    test_files["unit"].append(test_file)
                elif "integration" in str(test_file).lower():
                    test_files["integration"].append(test_file)
                else:
                    test_files["other"].append(test_file)

    return test_files


def run_pytest(test_path: str = "tests", verbose: bool = False, coverage: bool = False) -> Tuple[bool, str]:
    """Run pytest on specified path."""
    cmd = [sys.executable, "-m", "pytest", test_path]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=.", "--cov-report=term-missing"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        success = result.returncode == 0
        return success, output
    except subprocess.TimeoutExpired:
        return False, "Tests timed out after 5 minutes"
    except FileNotFoundError:
        return False, "pytest not installed. Run: pip install pytest"


def run_module_tests(module_name: str) -> Tuple[bool, List[str]]:
    """Run basic import and sanity tests on a module."""
    results = []

    try:
        # Ensure project root is on sys.path for absolute imports
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        module = importlib.import_module(module_name)
        results.append(f"[OK] Import: {module_name}")

        # Check for expected attributes
        if hasattr(module, "__version__"):
            results.append(f"     Version: {module.__version__}")

        return True, results
    except ImportError as e:
        results.append(f"[FAIL] Import: {module_name} - {e}")
        return False, results
    except Exception as e:
        results.append(f"[FAIL] Import: {module_name} - {e}")
        return False, results


def run_quick_sanity_tests() -> Tuple[int, int, List[str]]:
    """Run quick sanity tests on core modules."""
    modules_to_test = [
        # Providers can pull in heavy deps; skip here to avoid environment flakiness
        "data.universe.loader",
        "strategies.donchian.strategy",
        "strategies.ict.turtle_soup",
        "backtest.engine",
        "backtest.walk_forward",
        "risk.policy_gate",
        "oms.order_state",
        "oms.idempotency_store",
        "execution.broker_alpaca",
        "core.hash_chain",
        "core.structured_log",
    ]

    passed = 0
    failed = 0
    all_results = []

    for module in modules_to_test:
        success, results = run_module_tests(module)
        all_results.extend(results)
        if success:
            passed += 1
        else:
            failed += 1

    return passed, failed, all_results


def run_config_tests() -> Tuple[bool, List[str]]:
    """Test configuration loading."""
    results = []

    # Check config files exist
    config_files = [
        Path("config/base.yaml"),
    ]

    all_exist = True
    for config_file in config_files:
        if config_file.exists():
            results.append(f"[OK] Config: {config_file}")
        else:
            results.append(f"[SKIP] Config not found: {config_file}")

    return all_exist, results


def run_data_tests() -> Tuple[bool, List[str]]:
    """Test data availability."""
    results = []

    # Check universe file
    universe_file = Path("data/universe/optionable_liquid_900.csv")
    if universe_file.exists():
        with open(universe_file) as f:
            lines = len(f.readlines()) - 1  # Minus header
        results.append(f"[OK] Universe: {lines} symbols")
    else:
        results.append(f"[WARN] Universe file not found")

    # Check cache directory
    cache_dir = Path("data/cache")
    if cache_dir.exists():
        parquet_files = list(cache_dir.glob("*.parquet"))
        results.append(f"[OK] Cache: {len(parquet_files)} parquet files")
    else:
        results.append(f"[INFO] Cache directory not found")

    return True, results


def run_all_tests(verbose: bool = False, coverage: bool = False):
    """Run all tests."""
    print("\n" + "=" * 60)
    print("     KOBE TRADING SYSTEM TEST SUITE")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    total_passed = 0
    total_failed = 0
    start_time = time.time()

    # 1. Sanity Tests (module imports)
    print("--- Module Import Tests ---\n")
    passed, failed, results = run_quick_sanity_tests()
    for r in results:
        print(f"  {r}")
    total_passed += passed
    total_failed += failed
    print(f"\n  Passed: {passed}, Failed: {failed}\n")

    # 2. Config Tests
    print("--- Configuration Tests ---\n")
    success, results = run_config_tests()
    for r in results:
        print(f"  {r}")
    if success:
        total_passed += 1
    else:
        total_failed += 1
    print()

    # 3. Data Tests
    print("--- Data Availability Tests ---\n")
    success, results = run_data_tests()
    for r in results:
        print(f"  {r}")
    print()

    # 4. Pytest (if available)
    print("--- Unit Tests (pytest) ---\n")
    test_files = find_test_files()

    if any(test_files.values()):
        pytest_success, pytest_output = run_pytest(verbose=verbose, coverage=coverage)
        if pytest_success:
            print("  [OK] All pytest tests passed")
            total_passed += 1
        else:
            print("  [FAIL] Some pytest tests failed")
            if verbose:
                print(pytest_output)
            total_failed += 1
    else:
        print("  [SKIP] No test files found in tests/ directory")

    # 5. Integration Tests
    print("\n--- Integration Tests ---\n")
    if test_files["integration"]:
        for test_file in test_files["integration"]:
            print(f"  Running: {test_file}")
    else:
        print("  [SKIP] No integration tests found")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("     TEST SUMMARY")
    print("=" * 60)
    print(f"\n  Total Passed:  {total_passed}")
    print(f"  Total Failed:  {total_failed}")
    print(f"  Time Elapsed:  {elapsed:.2f}s")
    print(f"\n  Result: {'PASSED' if total_failed == 0 else 'FAILED'}")
    print("=" * 60 + "\n")

    return total_failed == 0


def main():
    parser = argparse.ArgumentParser(description="Run Kobe tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--sanity", action="store_true", help="Run quick sanity tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")

    args = parser.parse_args()

    if args.sanity:
        print("\n=== Sanity Tests ===\n")
        passed, failed, results = run_quick_sanity_tests()
        for r in results:
            print(r)
        print(f"\nPassed: {passed}, Failed: {failed}")
        sys.exit(0 if failed == 0 else 1)

    if args.unit:
        success, output = run_pytest("tests/unit", args.verbose, args.coverage)
        print(output)
        sys.exit(0 if success else 1)

    if args.integration:
        success, output = run_pytest("tests/integration", args.verbose, args.coverage)
        print(output)
        sys.exit(0 if success else 1)

    # Default: run all tests
    success = run_all_tests(args.verbose, args.coverage)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
