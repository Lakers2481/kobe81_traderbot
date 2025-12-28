#!/usr/bin/env python3
"""
Kobe Trading System - Quality Check Script

Comprehensive quality checks across:
- Code quality: syntax errors, flake8 linting
- Data quality: missing/stale data, OHLCV validation
- Test quality: pytest execution and coverage
- System quality: disk space, memory, API connectivity

Usage:
    python quality_check.py --full
    python quality_check.py --area code
    python quality_check.py --area data --strict
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env


# =============================================================================
# Score Definitions
# =============================================================================

GRADE_THRESHOLDS = [
    (95, 'A+'), (90, 'A'), (85, 'A-'),
    (80, 'B+'), (75, 'B'), (70, 'B-'),
    (65, 'C+'), (60, 'C'), (55, 'C-'),
    (50, 'D+'), (45, 'D'), (40, 'D-'),
    (0, 'F'),
]


def score_to_grade(score: float) -> str:
    """Convert numeric score (0-100) to letter grade."""
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return 'F'


@dataclass
class CheckResult:
    name: str
    passed: bool
    score: float  # 0-100
    message: str
    details: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = 'PASS' if self.passed else 'FAIL'
        return f"[{status}] {self.name}: {self.message} (Score: {self.score:.1f})"


@dataclass
class AreaReport:
    area: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.checks:
            return 0.0
        return sum(c.score for c in self.checks) / len(self.checks)

    @property
    def grade(self) -> str:
        return score_to_grade(self.score)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)


# =============================================================================
# Code Quality Checks
# =============================================================================

def check_python_syntax(root: Path) -> CheckResult:
    """Check all Python files for syntax errors."""
    errors: List[str] = []
    total_files = 0

    for py_file in root.rglob('*.py'):
        if '__pycache__' in str(py_file) or '.git' in str(py_file):
            continue
        total_files += 1
        try:
            with open(py_file, 'r', encoding='utf-8', errors='replace') as f:
                source = f.read()
            ast.parse(source)
        except SyntaxError as e:
            errors.append(f"{py_file.relative_to(root)}: Line {e.lineno}: {e.msg}")

    if not errors:
        return CheckResult(
            name="Python Syntax",
            passed=True,
            score=100.0,
            message=f"All {total_files} Python files have valid syntax"
        )
    else:
        score = max(0, 100 - (len(errors) / total_files) * 100)
        return CheckResult(
            name="Python Syntax",
            passed=False,
            score=score,
            message=f"{len(errors)} syntax errors in {total_files} files",
            details=errors[:10]  # Limit to first 10
        )


def check_flake8(root: Path) -> CheckResult:
    """Run flake8 linting if available."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'flake8', '--count', '--select=E9,F63,F7,F82',
             '--show-source', '--statistics', str(root)],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Check critical errors only (E9, F63, F7, F82)
        lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
        error_lines = [l for l in lines if l.strip() and not l.startswith(' ')]

        if result.returncode == 0 or len(error_lines) == 0:
            return CheckResult(
                name="Flake8 Critical",
                passed=True,
                score=100.0,
                message="No critical linting errors (E9, F63, F7, F82)"
            )
        else:
            # Penalize based on number of errors
            error_count = len(error_lines)
            score = max(0, 100 - error_count * 5)
            return CheckResult(
                name="Flake8 Critical",
                passed=error_count == 0,
                score=score,
                message=f"{error_count} critical linting issues",
                details=error_lines[:10]
            )

    except FileNotFoundError:
        return CheckResult(
            name="Flake8 Critical",
            passed=True,
            score=80.0,
            message="Flake8 not installed - skipped (install with: pip install flake8)"
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="Flake8 Critical",
            passed=False,
            score=50.0,
            message="Flake8 timed out"
        )
    except Exception as e:
        return CheckResult(
            name="Flake8 Critical",
            passed=False,
            score=50.0,
            message=f"Flake8 error: {e}"
        )


def check_imports(root: Path) -> CheckResult:
    """Check that critical modules can be imported."""
    critical_modules = [
        'strategies.ibs_rsi.strategy',
        'strategies.ict.turtle_soup',
        'backtest.engine',
        'data.providers.polygon_eod',
        'config.env_loader',
        'core.structured_log',
    ]

    errors: List[str] = []
    for mod_path in critical_modules:
        try:
            spec = importlib.util.find_spec(mod_path)
            if spec is None:
                errors.append(f"Module not found: {mod_path}")
        except ModuleNotFoundError as e:
            errors.append(f"{mod_path}: {e}")
        except Exception as e:
            errors.append(f"{mod_path}: {e}")

    if not errors:
        return CheckResult(
            name="Critical Imports",
            passed=True,
            score=100.0,
            message=f"All {len(critical_modules)} critical modules importable"
        )
    else:
        score = max(0, 100 - (len(errors) / len(critical_modules)) * 100)
        return CheckResult(
            name="Critical Imports",
            passed=False,
            score=score,
            message=f"{len(errors)} import failures",
            details=errors
        )


def run_code_quality_checks(root: Path) -> AreaReport:
    """Run all code quality checks."""
    report = AreaReport(area="Code Quality")
    report.checks.append(check_python_syntax(root))
    report.checks.append(check_flake8(root))
    report.checks.append(check_imports(root))
    return report


# =============================================================================
# Data Quality Checks
# =============================================================================

def check_data_freshness(cache_dir: Path, max_age_days: int = 7) -> CheckResult:
    """Check if cached data is stale."""
    if not cache_dir.exists():
        return CheckResult(
            name="Data Freshness",
            passed=False,
            score=0.0,
            message=f"Cache directory not found: {cache_dir}"
        )

    csv_files = list(cache_dir.glob('*.csv'))
    if not csv_files:
        return CheckResult(
            name="Data Freshness",
            passed=False,
            score=0.0,
            message="No cached data files found"
        )

    now = datetime.now()
    stale_files: List[str] = []
    fresh_files = 0

    for f in csv_files:
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        age_days = (now - mtime).days
        if age_days > max_age_days:
            stale_files.append(f"{f.name} ({age_days} days old)")
        else:
            fresh_files += 1

    total = len(csv_files)
    fresh_pct = (fresh_files / total) * 100 if total else 0

    if not stale_files:
        return CheckResult(
            name="Data Freshness",
            passed=True,
            score=100.0,
            message=f"All {total} cache files are fresh (< {max_age_days} days)"
        )
    else:
        return CheckResult(
            name="Data Freshness",
            passed=fresh_pct >= 80,
            score=fresh_pct,
            message=f"{len(stale_files)}/{total} files are stale (> {max_age_days} days)",
            details=stale_files[:10]
        )


def check_ohlcv_validity(cache_dir: Path, sample_size: int = 10) -> CheckResult:
    """Validate OHLCV data integrity on sample files."""
    csv_files = list(cache_dir.glob('*.csv')) if cache_dir.exists() else []

    if not csv_files:
        return CheckResult(
            name="OHLCV Validity",
            passed=False,
            score=0.0,
            message="No data files to validate"
        )

    import random
    sample = random.sample(csv_files, min(sample_size, len(csv_files)))

    errors: List[str] = []
    valid_files = 0

    for f in sample:
        try:
            df = pd.read_csv(f)

            # Check required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [c for c in required if c not in df.columns]
            if missing_cols:
                errors.append(f"{f.name}: Missing columns {missing_cols}")
                continue

            file_errors: List[str] = []

            # High >= Low
            invalid_hl = (df['high'] < df['low']).sum()
            if invalid_hl > 0:
                file_errors.append(f"High < Low: {invalid_hl} rows")

            # High >= Close and High >= Open
            invalid_hc = (df['high'] < df['close']).sum()
            invalid_ho = (df['high'] < df['open']).sum()
            if invalid_hc > 0:
                file_errors.append(f"High < Close: {invalid_hc} rows")
            if invalid_ho > 0:
                file_errors.append(f"High < Open: {invalid_ho} rows")

            # Low <= Close and Low <= Open
            invalid_lc = (df['low'] > df['close']).sum()
            invalid_lo = (df['low'] > df['open']).sum()
            if invalid_lc > 0:
                file_errors.append(f"Low > Close: {invalid_lc} rows")
            if invalid_lo > 0:
                file_errors.append(f"Low > Open: {invalid_lo} rows")

            # Volume >= 0
            negative_vol = (df['volume'] < 0).sum()
            if negative_vol > 0:
                file_errors.append(f"Negative volume: {negative_vol} rows")

            # Prices > 0
            zero_prices = ((df['close'] <= 0) | (df['open'] <= 0)).sum()
            if zero_prices > 0:
                file_errors.append(f"Zero/negative prices: {zero_prices} rows")

            if file_errors:
                errors.append(f"{f.name}: {'; '.join(file_errors)}")
            else:
                valid_files += 1

        except Exception as e:
            errors.append(f"{f.name}: Parse error - {e}")

    score = (valid_files / len(sample)) * 100 if sample else 0

    if not errors:
        return CheckResult(
            name="OHLCV Validity",
            passed=True,
            score=100.0,
            message=f"All {len(sample)} sampled files have valid OHLCV data"
        )
    else:
        return CheckResult(
            name="OHLCV Validity",
            passed=score >= 80,
            score=score,
            message=f"{len(errors)} files with OHLCV issues",
            details=errors[:10]
        )


def check_missing_data(cache_dir: Path, sample_size: int = 10) -> CheckResult:
    """Check for missing data (NaN values, gaps)."""
    csv_files = list(cache_dir.glob('*.csv')) if cache_dir.exists() else []

    if not csv_files:
        return CheckResult(
            name="Missing Data",
            passed=False,
            score=0.0,
            message="No data files to check"
        )

    import random
    sample = random.sample(csv_files, min(sample_size, len(csv_files)))

    issues: List[str] = []
    clean_files = 0

    for f in sample:
        try:
            df = pd.read_csv(f, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(f, nrows=0).columns else None)

            file_issues: List[str] = []

            # Check for NaN in price columns
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    nan_count = df[col].isna().sum()
                    if nan_count > 0:
                        file_issues.append(f"NaN in {col}: {nan_count}")

            # Check for large gaps in trading days (> 5 business days)
            if 'timestamp' in df.columns:
                df_sorted = df.sort_values('timestamp')
                ts = pd.to_datetime(df_sorted['timestamp'])
                gaps = ts.diff().dt.days
                large_gaps = (gaps > 7).sum()  # More than a week gap
                if large_gaps > 0:
                    file_issues.append(f"Large gaps (>7 days): {large_gaps}")

            if file_issues:
                issues.append(f"{f.name}: {'; '.join(file_issues)}")
            else:
                clean_files += 1

        except Exception as e:
            issues.append(f"{f.name}: {e}")

    score = (clean_files / len(sample)) * 100 if sample else 0

    if not issues:
        return CheckResult(
            name="Missing Data",
            passed=True,
            score=100.0,
            message=f"No missing data issues in {len(sample)} sampled files"
        )
    else:
        return CheckResult(
            name="Missing Data",
            passed=score >= 70,
            score=score,
            message=f"{len(issues)} files with missing data issues",
            details=issues[:10]
        )


def run_data_quality_checks(root: Path) -> AreaReport:
    """Run all data quality checks."""
    report = AreaReport(area="Data Quality")
    cache_dir = root / 'data' / 'cache'

    report.checks.append(check_data_freshness(cache_dir))
    report.checks.append(check_ohlcv_validity(cache_dir))
    report.checks.append(check_missing_data(cache_dir))

    return report


# =============================================================================
# Test Quality Checks
# =============================================================================

def check_pytest(root: Path) -> CheckResult:
    """Run pytest and report results."""
    tests_dir = root / 'tests'

    if not tests_dir.exists():
        return CheckResult(
            name="Pytest",
            passed=True,
            score=70.0,
            message="No tests directory found - consider adding tests"
        )

    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', str(tests_dir), '-v', '--tb=short', '-q'],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(root)
        )

        # Parse pytest output
        output = result.stdout + result.stderr

        # Look for summary line like "5 passed, 2 failed"
        import re
        match = re.search(r'(\d+) passed', output)
        passed = int(match.group(1)) if match else 0

        match = re.search(r'(\d+) failed', output)
        failed = int(match.group(1)) if match else 0

        match = re.search(r'(\d+) error', output)
        errors = int(match.group(1)) if match else 0

        total = passed + failed + errors

        if total == 0:
            return CheckResult(
                name="Pytest",
                passed=True,
                score=70.0,
                message="No tests found to run"
            )

        score = (passed / total) * 100 if total else 0

        if failed == 0 and errors == 0:
            return CheckResult(
                name="Pytest",
                passed=True,
                score=100.0,
                message=f"All {passed} tests passed"
            )
        else:
            return CheckResult(
                name="Pytest",
                passed=False,
                score=score,
                message=f"{passed} passed, {failed} failed, {errors} errors",
                details=output.split('\n')[-20:]  # Last 20 lines
            )

    except FileNotFoundError:
        return CheckResult(
            name="Pytest",
            passed=True,
            score=60.0,
            message="Pytest not installed (pip install pytest)"
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="Pytest",
            passed=False,
            score=30.0,
            message="Pytest timed out after 5 minutes"
        )
    except Exception as e:
        return CheckResult(
            name="Pytest",
            passed=False,
            score=0.0,
            message=f"Pytest error: {e}"
        )


def check_coverage(root: Path) -> CheckResult:
    """Check test coverage if pytest-cov is available."""
    tests_dir = root / 'tests'

    if not tests_dir.exists():
        return CheckResult(
            name="Test Coverage",
            passed=True,
            score=50.0,
            message="No tests directory - coverage check skipped"
        )

    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', str(tests_dir),
             '--cov=' + str(root), '--cov-report=term-missing', '-q'],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(root)
        )

        # Parse coverage percentage
        import re
        match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', result.stdout)

        if match:
            coverage_pct = int(match.group(1))
            return CheckResult(
                name="Test Coverage",
                passed=coverage_pct >= 60,
                score=float(coverage_pct),
                message=f"Code coverage: {coverage_pct}%"
            )
        else:
            return CheckResult(
                name="Test Coverage",
                passed=True,
                score=50.0,
                message="Could not parse coverage report"
            )

    except FileNotFoundError:
        return CheckResult(
            name="Test Coverage",
            passed=True,
            score=50.0,
            message="pytest-cov not installed (pip install pytest-cov)"
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="Test Coverage",
            passed=False,
            score=30.0,
            message="Coverage check timed out"
        )
    except Exception as e:
        return CheckResult(
            name="Test Coverage",
            passed=True,
            score=50.0,
            message=f"Coverage check error: {e}"
        )


def run_test_quality_checks(root: Path) -> AreaReport:
    """Run all test quality checks."""
    report = AreaReport(area="Test Quality")
    report.checks.append(check_pytest(root))
    report.checks.append(check_coverage(root))
    return report


# =============================================================================
# System Quality Checks
# =============================================================================

def check_disk_space(min_gb: float = 1.0) -> CheckResult:
    """Check available disk space."""
    try:
        total, used, free = shutil.disk_usage(ROOT)
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        used_pct = (used / total) * 100

        if free_gb >= min_gb:
            score = min(100, (free_gb / min_gb) * 50 + 50)
            return CheckResult(
                name="Disk Space",
                passed=True,
                score=score,
                message=f"{free_gb:.1f} GB free ({100 - used_pct:.1f}% available)"
            )
        else:
            score = (free_gb / min_gb) * 50
            return CheckResult(
                name="Disk Space",
                passed=False,
                score=score,
                message=f"Low disk space: {free_gb:.2f} GB (need >= {min_gb} GB)"
            )
    except Exception as e:
        return CheckResult(
            name="Disk Space",
            passed=False,
            score=0.0,
            message=f"Could not check disk space: {e}"
        )


def check_memory() -> CheckResult:
    """Check available memory."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        used_pct = mem.percent

        if available_gb >= 1.0:
            score = min(100, 100 - used_pct + 20)
            return CheckResult(
                name="Memory",
                passed=True,
                score=score,
                message=f"{available_gb:.1f} GB available ({100 - used_pct:.1f}% free)"
            )
        else:
            return CheckResult(
                name="Memory",
                passed=False,
                score=max(0, 100 - used_pct),
                message=f"Low memory: {available_gb:.2f} GB available"
            )
    except ImportError:
        return CheckResult(
            name="Memory",
            passed=True,
            score=70.0,
            message="psutil not installed - memory check skipped"
        )
    except Exception as e:
        return CheckResult(
            name="Memory",
            passed=True,
            score=50.0,
            message=f"Memory check error: {e}"
        )


def check_polygon_api() -> CheckResult:
    """Check Polygon API connectivity."""
    api_key = os.getenv('POLYGON_API_KEY', '')
    if not api_key:
        return CheckResult(
            name="Polygon API",
            passed=False,
            score=0.0,
            message="POLYGON_API_KEY not set"
        )

    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={api_key}"
        resp = requests.get(url, timeout=10)

        if resp.status_code == 200:
            return CheckResult(
                name="Polygon API",
                passed=True,
                score=100.0,
                message="Polygon API responding normally"
            )
        elif resp.status_code == 401:
            return CheckResult(
                name="Polygon API",
                passed=False,
                score=0.0,
                message="Invalid Polygon API key"
            )
        else:
            return CheckResult(
                name="Polygon API",
                passed=False,
                score=30.0,
                message=f"Polygon API returned status {resp.status_code}"
            )
    except requests.Timeout:
        return CheckResult(
            name="Polygon API",
            passed=False,
            score=20.0,
            message="Polygon API timeout"
        )
    except Exception as e:
        return CheckResult(
            name="Polygon API",
            passed=False,
            score=0.0,
            message=f"Polygon API error: {e}"
        )


def check_alpaca_api() -> CheckResult:
    """Check Alpaca API connectivity."""
    api_key = os.getenv('ALPACA_API_KEY_ID', '')
    api_secret = os.getenv('ALPACA_API_SECRET_KEY', '')

    if not api_key or not api_secret:
        return CheckResult(
            name="Alpaca API",
            passed=False,
            score=0.0,
            message="Alpaca API credentials not set"
        )

    try:
        base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
        resp = requests.get(
            f"{base}/v2/account",
            headers={
                'APCA-API-KEY-ID': api_key,
                'APCA-API-SECRET-KEY': api_secret,
            },
            timeout=10
        )

        if resp.status_code == 200:
            return CheckResult(
                name="Alpaca API",
                passed=True,
                score=100.0,
                message="Alpaca API responding normally"
            )
        elif resp.status_code in (401, 403):
            return CheckResult(
                name="Alpaca API",
                passed=False,
                score=0.0,
                message="Invalid Alpaca API credentials"
            )
        else:
            return CheckResult(
                name="Alpaca API",
                passed=False,
                score=30.0,
                message=f"Alpaca API returned status {resp.status_code}"
            )
    except requests.Timeout:
        return CheckResult(
            name="Alpaca API",
            passed=False,
            score=20.0,
            message="Alpaca API timeout"
        )
    except Exception as e:
        return CheckResult(
            name="Alpaca API",
            passed=False,
            score=0.0,
            message=f"Alpaca API error: {e}"
        )


def run_system_quality_checks() -> AreaReport:
    """Run all system quality checks."""
    report = AreaReport(area="System Quality")
    report.checks.append(check_disk_space())
    report.checks.append(check_memory())
    report.checks.append(check_polygon_api())
    report.checks.append(check_alpaca_api())
    return report


# =============================================================================
# Report Generation
# =============================================================================

def print_report(reports: List[AreaReport], strict: bool = False) -> int:
    """Print quality report and return exit code."""
    print("\n" + "=" * 70)
    print("KOBE TRADING SYSTEM - QUALITY CHECK REPORT")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70 + "\n")

    overall_scores: List[float] = []
    all_passed = True

    for report in reports:
        print(f"\n{'=' * 50}")
        print(f"  {report.area.upper()}")
        print(f"  Grade: {report.grade} ({report.score:.1f}/100)")
        print(f"{'=' * 50}\n")

        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"  [{status}] {check.name}")
            print(f"         {check.message}")
            print(f"         Score: {check.score:.1f}")

            if check.details:
                print("         Details:")
                for detail in check.details[:5]:
                    print(f"           - {detail}")
                if len(check.details) > 5:
                    print(f"           ... and {len(check.details) - 5} more")
            print()

            if not check.passed:
                all_passed = False

        overall_scores.append(report.score)

    # Overall summary
    overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    overall_grade = score_to_grade(overall_score)

    print("\n" + "=" * 70)
    print("OVERALL QUALITY SCORE")
    print("=" * 70)
    print(f"\n  Grade: {overall_grade}")
    print(f"  Score: {overall_score:.1f}/100")
    print(f"  Status: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print("\n" + "=" * 70 + "\n")

    # Exit code
    if strict:
        return 0 if all_passed else 1
    else:
        return 0 if overall_score >= 60 else 1


def main():
    parser = argparse.ArgumentParser(
        description='Kobe Trading System - Comprehensive Quality Check'
    )
    parser.add_argument(
        '--dotenv', type=str,
        default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env',
        help='Path to .env file'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Run all quality checks'
    )
    parser.add_argument(
        '--area', type=str, choices=['code', 'data', 'test', 'system'],
        help='Run checks for specific area only'
    )
    parser.add_argument(
        '--strict', action='store_true',
        help='Fail if any check fails (otherwise fail only if score < 60)'
    )

    args = parser.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
        print(f"Loaded {len(loaded)} environment variables from {dotenv}")

    # Determine which checks to run
    reports: List[AreaReport] = []

    if args.full or args.area is None:
        # Run all checks
        reports.append(run_code_quality_checks(ROOT))
        reports.append(run_data_quality_checks(ROOT))
        reports.append(run_test_quality_checks(ROOT))
        reports.append(run_system_quality_checks())
    else:
        # Run specific area
        if args.area == 'code':
            reports.append(run_code_quality_checks(ROOT))
        elif args.area == 'data':
            reports.append(run_data_quality_checks(ROOT))
        elif args.area == 'test':
            reports.append(run_test_quality_checks(ROOT))
        elif args.area == 'system':
            reports.append(run_system_quality_checks())

    # Print report and exit
    exit_code = print_report(reports, strict=args.strict)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

