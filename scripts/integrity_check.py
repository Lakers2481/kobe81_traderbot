#!/usr/bin/env python3
"""
Kobe Trading System - Data Integrity Validation

Comprehensive integrity checks:
- Lookahead bias in indicators (verify proper shift)
- OHLCV data validity (High >= Low, etc.)
- Future timestamp detection
- Signal generation logic verification
- Backtest result sanity checks (detect impossible returns)

Usage:
    python integrity_check.py --full
    python integrity_check.py --area ohlcv
    python integrity_check.py --area lookahead --fix
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.env_loader import load_env


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class IntegrityIssue:
    severity: str  # 'CRITICAL', 'WARNING', 'INFO'
    category: str
    description: str
    location: str
    fix_available: bool = False
    fix_description: str = ""

    def __str__(self) -> str:
        return f"[{self.severity}] {self.category}: {self.description} ({self.location})"


@dataclass
class IntegrityReport:
    area: str
    issues: List[IntegrityIssue] = field(default_factory=list)
    passed: bool = True
    summary: str = ""

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == 'CRITICAL')

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == 'WARNING')


# =============================================================================
# Lookahead Bias Checks
# =============================================================================

def check_indicator_shifts(file_path: Path) -> List[IntegrityIssue]:
    """Check that indicators are properly shifted to avoid lookahead bias."""
    issues: List[IntegrityIssue] = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception as e:
        issues.append(IntegrityIssue(
            severity='WARNING',
            category='File Read',
            description=f"Could not read file: {e}",
            location=str(file_path)
        ))
        return issues

    # Patterns that indicate potential lookahead bias
    indicator_patterns = [
        (r"df\['(\w+)'\]\s*=\s*(rsi|sma|ema|atr|ibs)\(", "indicator_computed"),
        (r"\.rolling\(.*?\)\.mean\(\)", "rolling_mean"),
        (r"\.ewm\(.*?\)\.mean\(\)", "ewm"),
    ]

    # Check for shift patterns
    shift_pattern = re.compile(r"df\['\w+_sig'\]\s*=\s*df\['\w+'\]\.shift\(1\)")

    has_indicators = False
    has_shifts = False

    for i, line in enumerate(lines, 1):
        # Check for indicator calculations
        for pattern, _ in indicator_patterns:
            if re.search(pattern, line):
                has_indicators = True

        # Check for shift
        if shift_pattern.search(line):
            has_shifts = True

    # If we have indicators but no shifts, flag it
    if has_indicators and not has_shifts:
        # Check if this is a strategy file
        if 'strategy' in str(file_path).lower():
            # More detailed check - look for signal generation without shift
            if re.search(r"generate_signals|scan_signals", content):
                # Look for proper shifting in _compute_indicators
                if "_compute_indicators" in content:
                    # Find the method
                    method_match = re.search(
                        r"def _compute_indicators.*?(?=def |\Z)",
                        content,
                        re.DOTALL
                    )
                    if method_match:
                        method_content = method_match.group()
                        if '.shift(1)' not in method_content:
                            issues.append(IntegrityIssue(
                                severity='CRITICAL',
                                category='Lookahead Bias',
                                description="Indicators not shifted in _compute_indicators - potential lookahead bias",
                                location=f"{file_path.name}:_compute_indicators",
                                fix_available=True,
                                fix_description="Add .shift(1) to indicator columns before signal generation"
                            ))

    # Check for direct use of non-shifted indicators in conditions
    condition_patterns = [
        (r"df\['(rsi\d*|sma\d*|ema\d*|atr\d*|ibs)'\]\s*[<>=]", "direct_indicator_comparison"),
        (r"row\['(rsi\d*|sma\d*|ema\d*|atr\d*|ibs)'\]", "row_indicator_access"),
    ]

    for i, line in enumerate(lines, 1):
        for pattern, pattern_type in condition_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                # Check if this is accessing the shifted version
                if f"{match}_sig" not in line and "shift" not in line:
                    # Check context - is this in signal generation?
                    context_start = max(0, i - 20)
                    context = '\n'.join(lines[context_start:i])
                    if 'generate_signals' in context or 'scan_signals' in context:
                        issues.append(IntegrityIssue(
                            severity='CRITICAL',
                            category='Lookahead Bias',
                            description=f"Using non-shifted indicator '{match}' in signal logic",
                            location=f"{file_path.name}:{i}",
                            fix_available=True,
                            fix_description=f"Use '{match}_sig' (shifted) instead of '{match}'"
                        ))

    return issues


def check_future_data_access(file_path: Path) -> List[IntegrityIssue]:
    """Check for patterns that might access future data."""
    issues: List[IntegrityIssue] = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception:
        return issues

    # Dangerous patterns that might indicate future data access
    dangerous_patterns = [
        (r"\.shift\(-\d+\)", "Negative shift (looking forward)"),
        (r"\.iloc\[\d+:\]", "Forward slicing without check"),
        (r"df\.loc\[.*?\+\s*\d+", "Forward index arithmetic"),
    ]

    for i, line in enumerate(lines, 1):
        for pattern, description in dangerous_patterns:
            if re.search(pattern, line):
                # Check if it's in a legitimate context (e.g., computing future returns for labels)
                if 'label' not in line.lower() and 'target' not in line.lower():
                    issues.append(IntegrityIssue(
                        severity='WARNING',
                        category='Future Data Access',
                        description=description,
                        location=f"{file_path.name}:{i}",
                        fix_available=False,
                        fix_description="Review this line for potential lookahead bias"
                    ))

    return issues


def run_lookahead_checks(root: Path) -> IntegrityReport:
    """Run all lookahead bias checks."""
    report = IntegrityReport(area="Lookahead Bias")

    strategy_files = list(root.glob('strategies/**/*.py'))
    indicator_files = list(root.glob('**/indicators.py'))
    backtest_files = list(root.glob('backtest/**/*.py'))

    all_files = set(strategy_files + indicator_files + backtest_files)

    for file_path in all_files:
        if '__pycache__' in str(file_path):
            continue

        issues = check_indicator_shifts(file_path)
        issues.extend(check_future_data_access(file_path))

        report.issues.extend(issues)

    # Determine pass/fail
    report.passed = report.critical_count == 0
    report.summary = f"Found {report.critical_count} critical and {report.warning_count} warning issues"

    return report


# =============================================================================
# OHLCV Validity Checks
# =============================================================================

def validate_ohlcv_file(file_path: Path) -> List[IntegrityIssue]:
    """Validate OHLCV data in a single file."""
    issues: List[IntegrityIssue] = []

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        issues.append(IntegrityIssue(
            severity='CRITICAL',
            category='File Parse',
            description=f"Could not parse CSV: {e}",
            location=str(file_path.name)
        ))
        return issues

    # Check required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(IntegrityIssue(
            severity='CRITICAL',
            category='Missing Columns',
            description=f"Missing columns: {missing}",
            location=str(file_path.name)
        ))
        return issues

    # High >= Low
    invalid_hl = (df['high'] < df['low'])
    if invalid_hl.any():
        count = invalid_hl.sum()
        first_idx = invalid_hl.idxmax()
        issues.append(IntegrityIssue(
            severity='CRITICAL',
            category='OHLCV Validity',
            description=f"High < Low in {count} rows",
            location=f"{file_path.name}:row {first_idx}",
            fix_available=True,
            fix_description="Swap high and low values where high < low"
        ))

    # High >= Close and High >= Open
    invalid_hc = (df['high'] < df['close'])
    invalid_ho = (df['high'] < df['open'])
    if invalid_hc.any():
        issues.append(IntegrityIssue(
            severity='CRITICAL',
            category='OHLCV Validity',
            description=f"High < Close in {invalid_hc.sum()} rows",
            location=str(file_path.name),
            fix_available=True,
            fix_description="Set high = max(high, close)"
        ))
    if invalid_ho.any():
        issues.append(IntegrityIssue(
            severity='CRITICAL',
            category='OHLCV Validity',
            description=f"High < Open in {invalid_ho.sum()} rows",
            location=str(file_path.name),
            fix_available=True,
            fix_description="Set high = max(high, open)"
        ))

    # Low <= Close and Low <= Open
    invalid_lc = (df['low'] > df['close'])
    invalid_lo = (df['low'] > df['open'])
    if invalid_lc.any():
        issues.append(IntegrityIssue(
            severity='CRITICAL',
            category='OHLCV Validity',
            description=f"Low > Close in {invalid_lc.sum()} rows",
            location=str(file_path.name),
            fix_available=True,
            fix_description="Set low = min(low, close)"
        ))
    if invalid_lo.any():
        issues.append(IntegrityIssue(
            severity='CRITICAL',
            category='OHLCV Validity',
            description=f"Low > Open in {invalid_lo.sum()} rows",
            location=str(file_path.name),
            fix_available=True,
            fix_description="Set low = min(low, open)"
        ))

    # Non-positive prices
    for col in ['open', 'high', 'low', 'close']:
        non_positive = (df[col] <= 0)
        if non_positive.any():
            issues.append(IntegrityIssue(
                severity='CRITICAL',
                category='Invalid Prices',
                description=f"Non-positive {col} in {non_positive.sum()} rows",
                location=str(file_path.name)
            ))

    # Negative volume
    if (df['volume'] < 0).any():
        issues.append(IntegrityIssue(
            severity='WARNING',
            category='Invalid Volume',
            description=f"Negative volume in {(df['volume'] < 0).sum()} rows",
            location=str(file_path.name),
            fix_available=True,
            fix_description="Set volume = abs(volume)"
        ))

    # Extreme price changes (> 50% in one day)
    if len(df) > 1:
        pct_change = df['close'].pct_change().abs()
        extreme_moves = (pct_change > 0.5)
        if extreme_moves.any():
            count = extreme_moves.sum()
            if count > 10:  # More than 10 is suspicious
                issues.append(IntegrityIssue(
                    severity='WARNING',
                    category='Extreme Moves',
                    description=f"{count} days with >50% price change (may indicate stock splits not adjusted)",
                    location=str(file_path.name)
                ))

    return issues


def fix_ohlcv_file(file_path: Path) -> bool:
    """Attempt to fix OHLCV issues in a file."""
    try:
        df = pd.read_csv(file_path)

        modified = False

        # Fix High < Low
        mask = df['high'] < df['low']
        if mask.any():
            df.loc[mask, ['high', 'low']] = df.loc[mask, ['low', 'high']].values
            modified = True

        # Fix High < Close
        mask = df['high'] < df['close']
        if mask.any():
            df.loc[mask, 'high'] = df.loc[mask, 'close']
            modified = True

        # Fix High < Open
        mask = df['high'] < df['open']
        if mask.any():
            df.loc[mask, 'high'] = df.loc[mask, 'open']
            modified = True

        # Fix Low > Close
        mask = df['low'] > df['close']
        if mask.any():
            df.loc[mask, 'low'] = df.loc[mask, 'close']
            modified = True

        # Fix Low > Open
        mask = df['low'] > df['open']
        if mask.any():
            df.loc[mask, 'low'] = df.loc[mask, 'open']
            modified = True

        # Fix negative volume
        if (df['volume'] < 0).any():
            df['volume'] = df['volume'].abs()
            modified = True

        if modified:
            df.to_csv(file_path, index=False)
            return True

        return False

    except Exception:
        return False


def run_ohlcv_checks(root: Path, sample_size: int = 50, fix: bool = False) -> IntegrityReport:
    """Run OHLCV validity checks."""
    report = IntegrityReport(area="OHLCV Validity")

    cache_dir = root / 'data' / 'cache'
    if not cache_dir.exists():
        report.issues.append(IntegrityIssue(
            severity='WARNING',
            category='Missing Directory',
            description="Data cache directory not found",
            location=str(cache_dir)
        ))
        report.passed = True
        report.summary = "No data to check"
        return report

    csv_files = list(cache_dir.glob('*.csv'))
    if not csv_files:
        report.summary = "No CSV files found in cache"
        report.passed = True
        return report

    import random
    sample = random.sample(csv_files, min(sample_size, len(csv_files)))

    fixed_count = 0
    for file_path in sample:
        issues = validate_ohlcv_file(file_path)

        # Attempt fix if requested
        if fix and issues:
            fixable = [i for i in issues if i.fix_available]
            if fixable:
                if fix_ohlcv_file(file_path):
                    fixed_count += 1
                    # Re-validate after fix
                    issues = validate_ohlcv_file(file_path)

        report.issues.extend(issues)

    report.passed = report.critical_count == 0
    report.summary = f"Checked {len(sample)}/{len(csv_files)} files. " \
                    f"Found {report.critical_count} critical, {report.warning_count} warnings"

    if fix and fixed_count > 0:
        report.summary += f". Fixed {fixed_count} files"

    return report


# =============================================================================
# Future Timestamp Checks
# =============================================================================

def check_future_timestamps(file_path: Path) -> List[IntegrityIssue]:
    """Check for future timestamps in data file."""
    issues: List[IntegrityIssue] = []

    try:
        df = pd.read_csv(file_path)
    except Exception:
        return issues

    if 'timestamp' not in df.columns:
        return issues

    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception:
        return issues

    now = datetime.now()
    future_mask = df['timestamp'] > now

    if future_mask.any():
        count = future_mask.sum()
        max_future = df.loc[future_mask, 'timestamp'].max()
        issues.append(IntegrityIssue(
            severity='CRITICAL',
            category='Future Timestamp',
            description=f"{count} rows with future timestamps (max: {max_future})",
            location=str(file_path.name),
            fix_available=True,
            fix_description="Remove rows with future timestamps"
        ))

    return issues


def run_timestamp_checks(root: Path, sample_size: int = 50) -> IntegrityReport:
    """Run future timestamp checks."""
    report = IntegrityReport(area="Timestamp Validity")

    cache_dir = root / 'data' / 'cache'
    if not cache_dir.exists():
        report.summary = "No data cache directory"
        report.passed = True
        return report

    csv_files = list(cache_dir.glob('*.csv'))
    if not csv_files:
        report.summary = "No CSV files found"
        report.passed = True
        return report

    import random
    sample = random.sample(csv_files, min(sample_size, len(csv_files)))

    for file_path in sample:
        issues = check_future_timestamps(file_path)
        report.issues.extend(issues)

    report.passed = report.critical_count == 0
    report.summary = f"Checked {len(sample)} files for future timestamps"

    return report


# =============================================================================
# Signal Logic Verification
# =============================================================================

def verify_signal_logic(root: Path) -> IntegrityReport:
    """Verify signal generation logic for consistency."""
    report = IntegrityReport(area="Signal Logic")

    strategy_files = list(root.glob('strategies/**/strategy.py'))

    for file_path in strategy_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue

        strategy_name = file_path.parent.name

        # Check for required methods
        required_methods = ['generate_signals', '_compute_indicators']
        for method in required_methods:
            if f"def {method}" not in content:
                report.issues.append(IntegrityIssue(
                    severity='WARNING',
                    category='Missing Method',
                    description=f"Missing {method} method",
                    location=f"{strategy_name}/strategy.py"
                ))

        # Check for proper signal column output
        signal_columns = ['timestamp', 'symbol', 'side', 'entry_price', 'stop_loss']
        for col in signal_columns:
            if f"'{col}'" not in content and f'"{col}"' not in content:
                report.issues.append(IntegrityIssue(
                    severity='WARNING',
                    category='Missing Column',
                    description=f"Signal output may be missing '{col}' column",
                    location=f"{strategy_name}/strategy.py"
                ))

        # Check for entry/exit logic symmetry
        has_long_entry = 'long' in content and 'entry' in content.lower()
        has_stop_loss = 'stop_loss' in content or 'stop' in content

        if has_long_entry and not has_stop_loss:
            report.issues.append(IntegrityIssue(
                severity='WARNING',
                category='Missing Risk Control',
                description="Has entry logic but no stop_loss definition",
                location=f"{strategy_name}/strategy.py"
            ))

    report.passed = report.critical_count == 0
    report.summary = f"Verified {len(strategy_files)} strategy files"

    return report


# =============================================================================
# Backtest Result Sanity Checks
# =============================================================================

def check_backtest_results(root: Path) -> IntegrityReport:
    """Check backtest results for impossibly good performance."""
    report = IntegrityReport(area="Backtest Sanity")

    # Find backtest output directories
    output_patterns = [
        'smoke_outputs',
        'backtest_results',
        'wf_outputs',
    ]

    result_files: List[Path] = []
    for pattern in output_patterns:
        result_files.extend(root.glob(f'{pattern}/**/summary.json'))
        result_files.extend(root.glob(f'{pattern}/**/metrics.json'))

    if not result_files:
        report.summary = "No backtest result files found"
        report.passed = True
        return report

    for file_path in result_files:
        try:
            import json
            with open(file_path, 'r') as f:
                metrics = json.load(f)
        except Exception:
            continue

        # Check for impossibly good returns
        # Assuming annual return > 500% is a bug
        sharpe = metrics.get('sharpe', 0)
        pf = metrics.get('profit_factor', 0)
        wr = metrics.get('win_rate', 0)
        max_dd = abs(metrics.get('max_drawdown', 0))
        trades = metrics.get('trades', 0)

        issues_found: List[str] = []

        # Sharpe > 5 is extremely rare and likely a bug
        if sharpe > 5:
            issues_found.append(f"Sharpe {sharpe:.2f} > 5 (extremely rare in practice)")

        # Profit factor > 10 with significant trades is suspicious
        if pf > 10 and trades > 50:
            issues_found.append(f"Profit factor {pf:.2f} > 10 with {trades} trades")

        # Win rate > 90% with > 100 trades is suspicious
        if wr > 0.9 and trades > 100:
            issues_found.append(f"Win rate {wr*100:.1f}% > 90% with {trades} trades")

        # No drawdown with significant trading is impossible
        if max_dd == 0 and trades > 20:
            issues_found.append(f"Zero max drawdown with {trades} trades")

        # Perfect profit factor (infinity) with trades
        if pf == float('inf') and trades > 10:
            issues_found.append("Infinite profit factor (no losing trades)")

        if issues_found:
            for issue in issues_found:
                report.issues.append(IntegrityIssue(
                    severity='CRITICAL',
                    category='Impossible Results',
                    description=issue,
                    location=str(file_path.relative_to(root))
                ))

    report.passed = report.critical_count == 0
    report.summary = f"Checked {len(result_files)} backtest result files"

    return report


# =============================================================================
# Report Generation
# =============================================================================

def print_report(reports: List[IntegrityReport]) -> int:
    """Print integrity report and return exit code."""
    print("\n" + "=" * 70)
    print("KOBE TRADING SYSTEM - DATA INTEGRITY REPORT")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70 + "\n")

    all_passed = True
    total_critical = 0
    total_warnings = 0

    for report in reports:
        status = "PASS" if report.passed else "FAIL"
        print(f"\n{'=' * 50}")
        print(f"  {report.area.upper()} - [{status}]")
        print(f"  {report.summary}")
        print(f"{'=' * 50}\n")

        if not report.passed:
            all_passed = False

        total_critical += report.critical_count
        total_warnings += report.warning_count

        # Group issues by severity
        critical_issues = [i for i in report.issues if i.severity == 'CRITICAL']
        warning_issues = [i for i in report.issues if i.severity == 'WARNING']
        info_issues = [i for i in report.issues if i.severity == 'INFO']

        if critical_issues:
            print("  CRITICAL ISSUES:")
            for issue in critical_issues[:10]:
                print(f"    - {issue.category}: {issue.description}")
                print(f"      Location: {issue.location}")
                if issue.fix_available:
                    print(f"      Fix: {issue.fix_description}")
            if len(critical_issues) > 10:
                print(f"    ... and {len(critical_issues) - 10} more")
            print()

        if warning_issues:
            print("  WARNINGS:")
            for issue in warning_issues[:5]:
                print(f"    - {issue.category}: {issue.description}")
                print(f"      Location: {issue.location}")
            if len(warning_issues) > 5:
                print(f"    ... and {len(warning_issues) - 5} more")
            print()

    # Summary
    print("\n" + "=" * 70)
    print("INTEGRITY CHECK SUMMARY")
    print("=" * 70)
    print(f"\n  Status: {'ALL CHECKS PASSED' if all_passed else 'ISSUES FOUND'}")
    print(f"  Critical Issues: {total_critical}")
    print(f"  Warnings: {total_warnings}")
    print("\n" + "=" * 70 + "\n")

    return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(
        description='Kobe Trading System - Data Integrity Validation'
    )
    parser.add_argument(
        '--dotenv', type=str,
        default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env',
        help='Path to .env file'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Run all integrity checks'
    )
    parser.add_argument(
        '--area', type=str,
        choices=['lookahead', 'ohlcv', 'timestamp', 'signals', 'backtest'],
        help='Run checks for specific area only'
    )
    parser.add_argument(
        '--fix', action='store_true',
        help='Attempt to fix issues where possible'
    )
    parser.add_argument(
        '--sample-size', type=int, default=50,
        help='Number of files to sample for data checks'
    )

    args = parser.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
        print(f"Loaded {len(loaded)} environment variables from {dotenv}")

    # Determine which checks to run
    reports: List[IntegrityReport] = []

    if args.full or args.area is None:
        # Run all checks
        print("\nRunning full integrity check...")
        reports.append(run_lookahead_checks(ROOT))
        reports.append(run_ohlcv_checks(ROOT, sample_size=args.sample_size, fix=args.fix))
        reports.append(run_timestamp_checks(ROOT, sample_size=args.sample_size))
        reports.append(verify_signal_logic(ROOT))
        reports.append(check_backtest_results(ROOT))
    else:
        # Run specific area
        if args.area == 'lookahead':
            reports.append(run_lookahead_checks(ROOT))
        elif args.area == 'ohlcv':
            reports.append(run_ohlcv_checks(ROOT, sample_size=args.sample_size, fix=args.fix))
        elif args.area == 'timestamp':
            reports.append(run_timestamp_checks(ROOT, sample_size=args.sample_size))
        elif args.area == 'signals':
            reports.append(verify_signal_logic(ROOT))
        elif args.area == 'backtest':
            reports.append(check_backtest_results(ROOT))

    # Print report and exit
    exit_code = print_report(reports)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
