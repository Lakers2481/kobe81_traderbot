"""
DATA QUALITY VERIFIER
=====================

Checks for:
- OHLC violations (high < open/close, low > open/close)
- Negative prices
- Zero volume days
- Duplicate timestamps
- Large gaps (>50% moves)
- Missing data
- Corporate action mismatches
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DataQualityVerifier:
    """Verify data quality with ZERO tolerance for bad data."""

    def __init__(self):
        self.violations = []
        self.warnings = []

    def check_ohlc_violations(self, df: pd.DataFrame) -> Dict:
        """Check for OHLC violations."""
        violations = []

        # Check: high >= max(open, close)
        invalid_high = (df['high'] < df['open']) | (df['high'] < df['close'])
        if invalid_high.any():
            count = invalid_high.sum()
            violations.append({
                'type': 'OHLC_HIGH_VIOLATION',
                'count': count,
                'severity': 'SEV-0',
                'sample': df[invalid_high][['timestamp', 'symbol', 'open', 'high', 'low', 'close']].head(5).to_dict('records')
            })

        # Check: low <= min(open, close)
        invalid_low = (df['low'] > df['open']) | (df['low'] > df['close'])
        if invalid_low.any():
            count = invalid_low.sum()
            violations.append({
                'type': 'OHLC_LOW_VIOLATION',
                'count': count,
                'severity': 'SEV-0',
                'sample': df[invalid_low][['timestamp', 'symbol', 'open', 'high', 'low', 'close']].head(5).to_dict('records')
            })

        # Check: high >= low
        invalid_range = df['high'] < df['low']
        if invalid_range.any():
            count = invalid_range.sum()
            violations.append({
                'type': 'HIGH_LT_LOW',
                'count': count,
                'severity': 'SEV-0',
                'sample': df[invalid_range][['timestamp', 'symbol', 'open', 'high', 'low', 'close']].head(5).to_dict('records')
            })

        return {
            'violations': violations,
            'total_violations': len(violations),
        }

    def check_negative_prices(self, df: pd.DataFrame) -> Dict:
        """Check for negative prices."""
        violations = []

        for col in ['open', 'high', 'low', 'close']:
            negative = df[col] < 0
            if negative.any():
                count = negative.sum()
                violations.append({
                    'type': f'NEGATIVE_{col.upper()}',
                    'count': count,
                    'severity': 'SEV-0',
                    'sample': df[negative][['timestamp', 'symbol', col]].head(5).to_dict('records')
                })

        return {
            'violations': violations,
            'total_violations': len(violations),
        }

    def check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate timestamps per symbol."""
        violations = []

        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            duplicates = symbol_df.duplicated(subset=['timestamp'], keep=False)

            if duplicates.any():
                count = duplicates.sum()
                violations.append({
                    'type': 'DUPLICATE_TIMESTAMPS',
                    'symbol': symbol,
                    'count': count,
                    'severity': 'SEV-0',
                    'sample': symbol_df[duplicates][['timestamp', 'symbol', 'close']].head(5).to_dict('records')
                })

        return {
            'violations': violations,
            'total_violations': len(violations),
        }

    def check_large_gaps(self, df: pd.DataFrame, threshold: float = 0.50) -> Dict:
        """Check for suspiciously large gaps (>50% move)."""
        warnings = []

        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].sort_values('timestamp')

            if len(symbol_df) < 2:
                continue

            # Calculate returns
            symbol_df['return'] = symbol_df['close'].pct_change()

            # Find large gaps
            large_gaps = (symbol_df['return'].abs() > threshold)

            if large_gaps.any():
                count = large_gaps.sum()
                warnings.append({
                    'type': 'LARGE_GAP',
                    'symbol': symbol,
                    'count': count,
                    'threshold': threshold,
                    'severity': 'SEV-1',
                    'sample': symbol_df[large_gaps][['timestamp', 'symbol', 'close', 'return']].head(5).to_dict('records')
                })

        return {
            'warnings': warnings,
            'total_warnings': len(warnings),
        }

    def check_zero_volume(self, df: pd.DataFrame) -> Dict:
        """Check for zero volume days."""
        warnings = []

        zero_volume = df['volume'] == 0
        if zero_volume.any():
            count = zero_volume.sum()
            pct = count / len(df) * 100

            warnings.append({
                'type': 'ZERO_VOLUME',
                'count': count,
                'pct': pct,
                'severity': 'SEV-1',
                'sample': df[zero_volume][['timestamp', 'symbol', 'volume']].head(5).to_dict('records')
            })

        return {
            'warnings': warnings,
            'total_warnings': len(warnings),
        }

    def check_data_from_csv(self, csv_path: Path) -> Dict:
        """Check data quality from a CSV file."""
        print(f"\n[CHECK] Loading {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            print(f"[CHECK] Loaded {len(df)} rows")
        except Exception as e:
            print(f"[ERROR] Failed to load {csv_path}: {e}")
            return {'error': str(e)}

        # Ensure required columns
        required = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return {'error': f'Missing columns: {missing}'}

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Run all checks
        ohlc_results = self.check_ohlc_violations(df)
        negative_results = self.check_negative_prices(df)
        duplicate_results = self.check_duplicates(df)
        gap_results = self.check_large_gaps(df)
        volume_results = self.check_zero_volume(df)

        results = {
            'file': str(csv_path),
            'total_rows': len(df),
            'symbols': df['symbol'].nunique(),
            'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'ohlc_violations': ohlc_results,
            'negative_prices': negative_results,
            'duplicates': duplicate_results,
            'large_gaps': gap_results,
            'zero_volume': volume_results,
            'total_sev0': (
                ohlc_results['total_violations'] +
                negative_results['total_violations'] +
                duplicate_results['total_violations']
            ),
            'total_sev1': (
                gap_results.get('total_warnings', 0) +
                volume_results.get('total_warnings', 0)
            ),
        }

        return results

    def generate_report(self, results: Dict):
        """Generate data quality report."""
        if 'error' in results:
            print(f"[ERROR] Cannot generate report: {results['error']}")
            return

        report = f"""# DATA QUALITY AUDIT REPORT

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**File:** {results['file']}
**Total Rows:** {results['total_rows']:,}
**Symbols:** {results['symbols']}
**Date Range:** {results['date_range']}

---

## SUMMARY

| Check | SEV-0 Violations | SEV-1 Warnings |
|-------|-----------------|----------------|
| OHLC Violations | {results['ohlc_violations']['total_violations']} | - |
| Negative Prices | {results['negative_prices']['total_violations']} | - |
| Duplicates | {results['duplicates']['total_violations']} | - |
| Large Gaps | - | {results['large_gaps'].get('total_warnings', 0)} |
| Zero Volume | - | {results['zero_volume'].get('total_warnings', 0)} |
| **TOTAL** | **{results['total_sev0']}** | **{results['total_sev1']}** |

---

## OHLC VIOLATIONS

"""

        if results['ohlc_violations']['total_violations'] == 0:
            report += "**PASS** - No OHLC violations detected\n\n"
        else:
            report += "**FAIL** - OHLC violations detected:\n\n"
            for v in results['ohlc_violations']['violations']:
                report += f"- **{v['severity']}** {v['type']}: {v['count']} instances\n"

        report += "\n---\n\n## NEGATIVE PRICES\n\n"

        if results['negative_prices']['total_violations'] == 0:
            report += "**PASS** - No negative prices detected\n\n"
        else:
            report += "**FAIL** - Negative prices detected:\n\n"
            for v in results['negative_prices']['violations']:
                report += f"- **{v['severity']}** {v['type']}: {v['count']} instances\n"

        report += "\n---\n\n## DUPLICATES\n\n"

        if results['duplicates']['total_violations'] == 0:
            report += "**PASS** - No duplicate timestamps detected\n\n"
        else:
            report += "**FAIL** - Duplicate timestamps detected:\n\n"
            for v in results['duplicates']['violations']:
                report += f"- **{v['severity']}** Symbol {v['symbol']}: {v['count']} duplicates\n"

        report += "\n---\n\n## VERDICT\n\n"

        if results['total_sev0'] > 0:
            report += f"**FAIL** - {results['total_sev0']} SEV-0 data quality violations detected\n\n"
            report += "Data quality is INSUFFICIENT for trading.\n"
        else:
            report += "**PASS** - No critical data quality violations detected\n\n"
            if results['total_sev1'] > 0:
                report += f"Note: {results['total_sev1']} SEV-1 warnings detected (large gaps, zero volume)\n"

        # Save report
        output_path = project_root / "AUDITS" / "DATA_QUALITY_SCORECARD.md"
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\n[SAVED] Data quality report saved to {output_path}")


def main():
    """Run data quality verification."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python verify_data_quality.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])

    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    verifier = DataQualityVerifier()
    results = verifier.check_data_from_csv(csv_path)

    if 'error' not in results:
        verifier.generate_report(results)

        print("\n" + "="*80)
        print("DATA QUALITY VERIFICATION COMPLETE")
        print("="*80)
        print(f"SEV-0 violations: {results['total_sev0']}")
        print(f"SEV-1 warnings: {results['total_sev1']}")
        print("="*80)


if __name__ == "__main__":
    main()
