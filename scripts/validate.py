#!/usr/bin/env python3
"""
Kobe Validation Script

Runs tests and type checks:
- pytest if tests/ directory exists
- mypy type checking on key modules
- Outputs pass/fail summary
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.env_loader import load_env


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: Optional[str] = None
    skipped: bool = False


# -----------------------------------------------------------------------------
# Check functions
# -----------------------------------------------------------------------------
def run_pytest(verbose: bool = False) -> CheckResult:
    """Run pytest if tests/ directory exists."""
    tests_dir = ROOT / "tests"

    if not tests_dir.exists():
        return CheckResult(
            name="pytest",
            passed=True,
            message="No tests/ directory found (skipped)",
            skipped=True,
        )

    # Check if pytest is available
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        if result.returncode != 0:
            return CheckResult(
                name="pytest",
                passed=False,
                message="pytest not installed",
                details="Install with: pip install pytest",
            )
    except FileNotFoundError:
        return CheckResult(
            name="pytest",
            passed=False,
            message="pytest not installed",
            details="Install with: pip install pytest",
        )
    except Exception as e:
        return CheckResult(
            name="pytest",
            passed=False,
            message=f"Error checking pytest: {e}",
        )

    # Run pytest
    cmd = [sys.executable, "-m", "pytest", str(tests_dir), "-v", "--tb=short"]
    if not verbose:
        cmd.append("-q")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=300,  # 5 minute timeout
        )

        output = result.stdout + result.stderr
        passed = result.returncode == 0

        # Extract summary line
        summary = ""
        for line in output.split("\n"):
            if "passed" in line or "failed" in line or "error" in line:
                summary = line.strip()
                break

        return CheckResult(
            name="pytest",
            passed=passed,
            message=summary if summary else ("All tests passed" if passed else "Tests failed"),
            details=output if verbose or not passed else None,
        )

    except subprocess.TimeoutExpired:
        return CheckResult(
            name="pytest",
            passed=False,
            message="Tests timed out (>5 minutes)",
        )
    except Exception as e:
        return CheckResult(
            name="pytest",
            passed=False,
            message=f"Error running pytest: {e}",
        )


def run_mypy(verbose: bool = False) -> CheckResult:
    """Run mypy type checking on key modules."""
    # Key modules to type check
    modules = [
        "strategies",
        "data",
        "backtest",
        "risk",
        "oms",
        "execution",
        "core",
        "configs",
        "monitor",
    ]

    # Check if mypy is available
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--version"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        if result.returncode != 0:
            return CheckResult(
                name="mypy",
                passed=True,
                message="mypy not installed (skipped)",
                details="Install with: pip install mypy",
                skipped=True,
            )
    except FileNotFoundError:
        return CheckResult(
            name="mypy",
            passed=True,
            message="mypy not installed (skipped)",
            details="Install with: pip install mypy",
            skipped=True,
        )
    except Exception as e:
        return CheckResult(
            name="mypy",
            passed=False,
            message=f"Error checking mypy: {e}",
        )

    # Find existing modules
    existing_modules = []
    for mod in modules:
        mod_path = ROOT / mod
        if mod_path.exists():
            existing_modules.append(str(mod_path))

    if not existing_modules:
        return CheckResult(
            name="mypy",
            passed=True,
            message="No modules to check",
            skipped=True,
        )

    # Run mypy
    cmd = [
        sys.executable,
        "-m",
        "mypy",
        "--ignore-missing-imports",
        "--no-error-summary",
        "--show-error-codes",
        *existing_modules,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=180,  # 3 minute timeout
        )

        output = result.stdout + result.stderr
        passed = result.returncode == 0

        # Count errors
        error_count = 0
        error_lines: List[str] = []
        for line in output.split("\n"):
            if ": error:" in line:
                error_count += 1
                error_lines.append(line.strip())

        if passed:
            message = f"Type check passed ({len(existing_modules)} modules)"
        else:
            message = f"Type check failed: {error_count} error(s)"

        details = None
        if verbose or not passed:
            if error_lines:
                details = "\n".join(error_lines[:20])  # Show first 20 errors
                if len(error_lines) > 20:
                    details += f"\n... and {len(error_lines) - 20} more errors"

        return CheckResult(
            name="mypy",
            passed=passed,
            message=message,
            details=details,
        )

    except subprocess.TimeoutExpired:
        return CheckResult(
            name="mypy",
            passed=False,
            message="Type check timed out (>3 minutes)",
        )
    except Exception as e:
        return CheckResult(
            name="mypy",
            passed=False,
            message=f"Error running mypy: {e}",
        )


def run_import_check() -> CheckResult:
    """Check that key modules can be imported."""
    modules_to_check = [
        ("configs.env_loader", "load_env"),
        ("data.universe.loader", "load_universe"),
        ("data.providers.polygon_eod", "fetch_daily_bars_polygon"),
        ("strategies.connors_rsi2.strategy", "ConnorsRSI2Strategy"),
        ("strategies.ibs.strategy", "IBSStrategy"),
        ("backtest.engine", "Backtester"),
        ("risk.policy_gate", "PolicyGate"),
        ("oms.order_state", "OrderRecord"),
        ("execution.broker_alpaca", "place_ioc_limit"),
        ("core.structured_log", "jlog"),
    ]

    errors: List[str] = []
    passed_count = 0

    for module_name, attr_name in modules_to_check:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            if hasattr(module, attr_name):
                passed_count += 1
            else:
                errors.append(f"{module_name}: missing {attr_name}")
        except ImportError as e:
            errors.append(f"{module_name}: {e}")
        except Exception as e:
            errors.append(f"{module_name}: {e}")

    if not errors:
        return CheckResult(
            name="imports",
            passed=True,
            message=f"All {passed_count} module imports successful",
        )
    else:
        return CheckResult(
            name="imports",
            passed=False,
            message=f"Import errors: {len(errors)} failed, {passed_count} passed",
            details="\n".join(errors),
        )


def run_syntax_check() -> CheckResult:
    """Check Python syntax of all .py files."""
    py_files = list(ROOT.rglob("*.py"))

    # Exclude common directories
    excluded = {"__pycache__", ".git", "venv", ".venv", "env", ".env"}
    py_files = [
        f for f in py_files if not any(exc in f.parts for exc in excluded)
    ]

    errors: List[str] = []

    for py_file in py_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            compile(source, str(py_file), "exec")
        except SyntaxError as e:
            errors.append(f"{py_file.relative_to(ROOT)}:{e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"{py_file.relative_to(ROOT)}: {e}")

    if not errors:
        return CheckResult(
            name="syntax",
            passed=True,
            message=f"All {len(py_files)} Python files have valid syntax",
        )
    else:
        return CheckResult(
            name="syntax",
            passed=False,
            message=f"Syntax errors in {len(errors)} file(s)",
            details="\n".join(errors),
        )


def run_requirements_check() -> CheckResult:
    """Check that required packages are installed."""
    requirements_file = ROOT / "requirements.txt"

    if not requirements_file.exists():
        return CheckResult(
            name="requirements",
            passed=True,
            message="No requirements.txt found (skipped)",
            skipped=True,
        )

    try:
        with open(requirements_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return CheckResult(
            name="requirements",
            passed=False,
            message=f"Error reading requirements.txt: {e}",
        )

    missing: List[str] = []
    installed: List[str] = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Parse package name (handle various formats)
        pkg = line.split("==")[0].split(">=")[0].split("<=")[0].split("<")[0].split(">")[0].strip()
        if not pkg:
            continue

        # Check if installed
        try:
            __import__(pkg.replace("-", "_"))
            installed.append(pkg)
        except ImportError:
            # Try alternative import names
            try:
                # Some packages have different import names
                alt_name = pkg.lower().replace("-", "_")
                __import__(alt_name)
                installed.append(pkg)
            except ImportError:
                missing.append(pkg)

    if not missing:
        return CheckResult(
            name="requirements",
            passed=True,
            message=f"All {len(installed)} required packages installed",
        )
    else:
        return CheckResult(
            name="requirements",
            passed=False,
            message=f"Missing packages: {len(missing)} of {len(installed) + len(missing)}",
            details=f"Missing: {', '.join(missing)}",
        )


# -----------------------------------------------------------------------------
# Output formatting
# -----------------------------------------------------------------------------
def print_result(result: CheckResult, verbose: bool = False) -> None:
    """Print a check result."""
    if result.skipped:
        status = "[SKIP]"
    elif result.passed:
        status = "[PASS]"
    else:
        status = "[FAIL]"

    print(f"  {status} {result.name}: {result.message}")

    if result.details and (verbose or not result.passed):
        print()
        for line in result.details.split("\n"):
            print(f"         {line}")
        print()


def print_summary(results: List[CheckResult]) -> Tuple[int, int, int]:
    """Print summary and return (passed, failed, skipped) counts."""
    passed = sum(1 for r in results if r.passed and not r.skipped)
    failed = sum(1 for r in results if not r.passed)
    skipped = sum(1 for r in results if r.skipped)

    print("=" * 60)
    print(f"  SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return passed, failed, skipped


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Kobe Validation Script - Run tests and type checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate.py              # Run all checks
  python scripts/validate.py --verbose    # Verbose output
  python scripts/validate.py --quick      # Skip slow checks (mypy)
        """,
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output with full details",
    )
    ap.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Quick mode: skip slow checks like mypy",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = ap.parse_args()

    # Load environment (for any checks that might need it)
    dotenv_path = Path(args.dotenv)
    if dotenv_path.exists():
        load_env(dotenv_path)

    print("=" * 60)
    print("  KOBE VALIDATION")
    print("=" * 60)
    print()

    results: List[CheckResult] = []

    # Run checks
    print("Running validation checks...")
    print()

    # 1. Syntax check (fast, always run)
    print("  Checking Python syntax...")
    result = run_syntax_check()
    results.append(result)
    print_result(result, args.verbose)

    # 2. Import check (fast, always run)
    print("  Checking module imports...")
    result = run_import_check()
    results.append(result)
    print_result(result, args.verbose)

    # 3. Requirements check (fast)
    print("  Checking requirements...")
    result = run_requirements_check()
    results.append(result)
    print_result(result, args.verbose)

    # 4. Pytest (can be slow)
    print("  Running pytest...")
    result = run_pytest(args.verbose)
    results.append(result)
    print_result(result, args.verbose)

    # 5. Mypy (slow, skip in quick mode)
    if not args.quick:
        print("  Running mypy type check...")
        result = run_mypy(args.verbose)
        results.append(result)
        print_result(result, args.verbose)
    else:
        results.append(
            CheckResult(
                name="mypy",
                passed=True,
                message="Skipped (quick mode)",
                skipped=True,
            )
        )
        print("  [SKIP] mypy: Skipped (quick mode)")

    print()

    # Output
    if args.json:
        import json

        output = [
            {
                "name": r.name,
                "passed": r.passed,
                "skipped": r.skipped,
                "message": r.message,
                "details": r.details,
            }
            for r in results
        ]
        print(json.dumps(output, indent=2))
        return 0

    # Summary
    passed, failed, skipped = print_summary(results)

    # Exit code
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
