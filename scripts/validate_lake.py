#!/usr/bin/env python3
"""
Validate Frozen Data Lake
==========================

CLI tool to verify data lake integrity, coverage, and quality.

Features:
- Hash verification (detect data drift/corruption)
- Coverage validation (min years, symbol count)
- Gap detection (max acceptable gap in trading days)
- Report generation

Usage:
    # Validate all datasets
    python scripts/validate_lake.py

    # Validate specific dataset
    python scripts/validate_lake.py --dataset-id stooq_1d_2015_2024_abc123

    # Validate with specific requirements
    python scripts/validate_lake.py --min-years 5 --max-gap 5 --min-symbols 100

    # Windows:
    python scripts\\validate_lake.py --dataset-id YOUR_DATASET_ID

    # Linux/macOS:
    python scripts/validate_lake.py --dataset-id YOUR_DATASET_ID
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.lake import (
    LakeReader,
    DatasetManifest,
    find_manifest,
    list_manifests,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_integrity(
    manifest: DatasetManifest,
    lake_dir: Path,
) -> Dict:
    """Validate file hashes match manifest."""
    dataset_dir = lake_dir / manifest.dataset_id
    return manifest.verify_integrity(dataset_dir)


def validate_coverage(
    manifest: DatasetManifest,
    min_years: float = 5.0,
    min_symbols: int = 10,
) -> Dict:
    """Validate data coverage meets requirements."""
    stats = manifest.get_coverage_stats()

    result = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'coverage': stats,
    }

    if stats['years'] < min_years:
        result['valid'] = False
        result['errors'].append(
            f"Insufficient coverage: {stats['years']:.1f} years < {min_years} required"
        )

    if stats['symbols'] < min_symbols:
        result['valid'] = False
        result['errors'].append(
            f"Insufficient symbols: {stats['symbols']} < {min_symbols} required"
        )

    return result


def validate_gaps(
    reader: LakeReader,
    dataset_id: str,
    max_gap_days: int = 5,
    sample_symbols: int = 10,
) -> Dict:
    """Validate no unacceptable gaps in trading days."""
    result = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'symbols_checked': 0,
        'max_gap_found': 0,
    }

    try:
        # Load data
        df = reader.load_dataset(dataset_id)

        if df.empty:
            result['valid'] = False
            result['errors'].append("Dataset is empty")
            return result

        # Get unique symbols
        symbols = df['symbol'].unique()

        # Sample if too many
        if len(symbols) > sample_symbols:
            import numpy as np
            np.random.seed(42)
            symbols = np.random.choice(symbols, sample_symbols, replace=False)

        # Check gaps for each symbol
        max_gap = 0

        for symbol in symbols:
            sym_data = df[df['symbol'] == symbol].sort_values('timestamp')
            if len(sym_data) < 2:
                continue

            timestamps = pd.to_datetime(sym_data['timestamp'])
            gaps = timestamps.diff().dropna()

            # Find max gap in calendar days
            max_sym_gap = gaps.max().days if len(gaps) > 0 else 0

            if max_sym_gap > max_gap:
                max_gap = max_sym_gap

            if max_sym_gap > max_gap_days:
                result['warnings'].append(
                    f"{symbol}: gap of {max_sym_gap} days exceeds max {max_gap_days}"
                )

            result['symbols_checked'] += 1

        result['max_gap_found'] = max_gap

        if max_gap > max_gap_days * 2:  # Only fail for very large gaps
            result['valid'] = False
            result['errors'].append(
                f"Excessive gap detected: {max_gap} days (max allowed: {max_gap_days})"
            )

    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Failed to validate gaps: {e}")

    return result


def validate_dataset(
    dataset_id: str,
    lake_dir: Path,
    manifest_dir: Path,
    min_years: float = 5.0,
    min_symbols: int = 10,
    max_gap_days: int = 5,
    check_gaps: bool = True,
) -> Dict:
    """Full validation of a dataset."""
    result = {
        'dataset_id': dataset_id,
        'valid': True,
        'integrity': None,
        'coverage': None,
        'gaps': None,
        'errors': [],
        'warnings': [],
    }

    # Find manifest
    manifest = find_manifest(dataset_id, manifest_dir)
    if manifest is None:
        result['valid'] = False
        result['errors'].append(f"Manifest not found for: {dataset_id}")
        return result

    # Integrity check
    logger.info(f"Checking integrity for {dataset_id}...")
    integrity = validate_integrity(manifest, lake_dir)
    result['integrity'] = integrity
    if not integrity['valid']:
        result['valid'] = False
        result['errors'].extend(integrity['errors'])

    # Coverage check
    logger.info(f"Checking coverage for {dataset_id}...")
    coverage = validate_coverage(manifest, min_years, min_symbols)
    result['coverage'] = coverage
    if not coverage['valid']:
        result['valid'] = False
        result['errors'].extend(coverage['errors'])

    # Gap check (optional, slower)
    if check_gaps:
        logger.info(f"Checking gaps for {dataset_id}...")
        reader = LakeReader(lake_dir, manifest_dir)
        gaps = validate_gaps(reader, dataset_id, max_gap_days)
        result['gaps'] = gaps
        if not gaps['valid']:
            result['valid'] = False
            result['errors'].extend(gaps['errors'])
        result['warnings'].extend(gaps.get('warnings', []))

    return result


def print_validation_report(result: Dict):
    """Print formatted validation report."""
    print("\n" + "=" * 60)
    print(f"DATASET VALIDATION: {result['dataset_id']}")
    print("=" * 60)

    status = "PASS" if result['valid'] else "FAIL"
    print(f"\nOverall Status: {status}")

    # Integrity
    if result['integrity']:
        int_status = "PASS" if result['integrity']['valid'] else "FAIL"
        print(f"\nIntegrity Check: {int_status}")
        print(f"  Files checked: {result['integrity'].get('files_checked', 0)}")
        if result['integrity'].get('files_missing', 0) > 0:
            print(f"  Files missing: {result['integrity']['files_missing']}")

    # Coverage
    if result['coverage']:
        cov = result['coverage']
        cov_status = "PASS" if cov['valid'] else "FAIL"
        print(f"\nCoverage Check: {cov_status}")
        if 'coverage' in cov:
            print(f"  Years: {cov['coverage']['years']:.1f}")
            print(f"  Symbols: {cov['coverage']['symbols']}")
            print(f"  Total rows: {cov['coverage']['total_rows']:,}")

    # Gaps
    if result['gaps']:
        gap_status = "PASS" if result['gaps']['valid'] else "FAIL"
        print(f"\nGap Check: {gap_status}")
        print(f"  Symbols checked: {result['gaps'].get('symbols_checked', 0)}")
        print(f"  Max gap found: {result['gaps'].get('max_gap_found', 0)} days")

    # Errors
    if result['errors']:
        print("\nERRORS:")
        for err in result['errors']:
            print(f"  - {err}")

    # Warnings
    if result['warnings']:
        print("\nWARNINGS:")
        for warn in result['warnings'][:10]:  # Limit displayed warnings
            print(f"  - {warn}")
        if len(result['warnings']) > 10:
            print(f"  ... and {len(result['warnings']) - 10} more")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate frozen data lake datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate all datasets
    python scripts/validate_lake.py

    # Validate specific dataset
    python scripts/validate_lake.py --dataset-id stooq_1d_2015_2024_abc123

    # Validate with stricter requirements
    python scripts/validate_lake.py --min-years 10 --min-symbols 500

    # Skip gap checking (faster)
    python scripts/validate_lake.py --no-gap-check

    # Output JSON report
    python scripts/validate_lake.py --json --output validation_report.json
"""
    )

    parser.add_argument(
        '--dataset-id',
        help='Specific dataset ID to validate (default: validate all)',
    )
    parser.add_argument(
        '--lake-dir',
        default='data/lake',
        help='Path to data lake directory (default: data/lake)',
    )
    parser.add_argument(
        '--manifest-dir',
        default='data/manifests',
        help='Path to manifests directory (default: data/manifests)',
    )
    parser.add_argument(
        '--min-years',
        type=float,
        default=5.0,
        help='Minimum years of coverage required (default: 5)',
    )
    parser.add_argument(
        '--min-symbols',
        type=int,
        default=10,
        help='Minimum number of symbols required (default: 10)',
    )
    parser.add_argument(
        '--max-gap',
        type=int,
        default=5,
        help='Maximum acceptable gap in trading days (default: 5)',
    )
    parser.add_argument(
        '--no-gap-check',
        action='store_true',
        help='Skip gap checking (faster)',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON',
    )
    parser.add_argument(
        '--output',
        help='Output file for JSON report',
    )

    args = parser.parse_args()

    lake_dir = Path(args.lake_dir)
    manifest_dir = Path(args.manifest_dir)

    if not manifest_dir.exists():
        print(f"ERROR: Manifest directory not found: {manifest_dir}")
        print("No datasets have been frozen yet.")
        print("\nTo freeze equities data, run:")
        print("  python scripts/freeze_equities_eod.py --universe data/universe/optionable_liquid_900.csv")
        sys.exit(1)

    # Get datasets to validate
    if args.dataset_id:
        dataset_ids = [args.dataset_id]
    else:
        manifests = list_manifests(manifest_dir)
        dataset_ids = [m.dataset_id for m in manifests]

    if not dataset_ids:
        print("No datasets found to validate.")
        sys.exit(0)

    # Validate each dataset
    all_results = []
    all_valid = True

    for dataset_id in dataset_ids:
        result = validate_dataset(
            dataset_id=dataset_id,
            lake_dir=lake_dir,
            manifest_dir=manifest_dir,
            min_years=args.min_years,
            min_symbols=args.min_symbols,
            max_gap_days=args.max_gap,
            check_gaps=not args.no_gap_check,
        )

        all_results.append(result)

        if not result['valid']:
            all_valid = False

        if not args.json:
            print_validation_report(result)

    # Output
    if args.json or args.output:
        report = {
            'validated_at': datetime.now().isoformat(),
            'all_valid': all_valid,
            'datasets_checked': len(all_results),
            'results': all_results,
        }

        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2))
            print(f"Report saved to: {args.output}")
        else:
            print(json.dumps(report, indent=2))

    # Summary
    if not args.json:
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Datasets validated: {len(all_results)}")
        passed = sum(1 for r in all_results if r['valid'])
        failed = len(all_results) - passed
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"\nOverall: {'PASS' if all_valid else 'FAIL'}")

    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()
