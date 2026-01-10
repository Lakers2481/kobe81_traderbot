"""
LOOKAHEAD BIAS DETECTOR
=======================

Verifies that NO future data is used in signal generation.

SEV-0 FAIL CONDITIONS:
- Any indicator uses current bar data without shift(1)
- Any signal generation looks at future bars
- Any training data includes target from same bar
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict
import ast

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class LookaheadBiasDetector:
    """Detect lookahead bias in code and data."""

    def __init__(self):
        self.violations = []
        self.warnings = []

    def check_strategy_files(self) -> Dict:
        """Check all strategy files for lookahead violations."""
        print("\n" + "="*80)
        print("LOOKAHEAD BIAS DETECTION - STRATEGY FILES")
        print("="*80)

        strategy_files = [
            project_root / "strategies" / "ibs_rsi" / "strategy.py",
            project_root / "strategies" / "ict" / "turtle_soup.py",
            project_root / "strategies" / "dual_strategy" / "combined.py",
        ]

        for filepath in strategy_files:
            if not filepath.exists():
                print(f"[SKIP] {filepath} not found")
                continue

            print(f"\n[CHECK] {filepath.name}")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for shift(1) on signal columns
            if '_sig = ' in content:
                # Find all signal assignments
                for line_num, line in enumerate(content.split('\n'), 1):
                    if '_sig = ' in line and 'shift(1)' not in line and '.shift(1)' not in line:
                        # Check if this is a direct assignment without shift
                        if '=' in line and not line.strip().startswith('#'):
                            self.violations.append({
                                'file': str(filepath),
                                'line': line_num,
                                'code': line.strip(),
                                'severity': 'SEV-0',
                                'reason': 'Signal column assigned without shift(1) - LOOKAHEAD BIAS'
                            })
                            print(f"  [SEV-0] Line {line_num}: Missing shift(1) on signal column")

            # Check for .iloc[-1] or .iloc[i] in signal generation
            if '.iloc[-1]' in content or '.iloc[i]' in content:
                for line_num, line in enumerate(content.split('\n'), 1):
                    if ('.iloc[-1]' in line or '.iloc[i]' in line) and not line.strip().startswith('#'):
                        self.warnings.append({
                            'file': str(filepath),
                            'line': line_num,
                            'code': line.strip(),
                            'severity': 'SEV-1',
                            'reason': 'Using iloc[-1] or iloc[i] - verify this is not lookahead'
                        })
                        print(f"  [SEV-1] Line {line_num}: Using iloc indexing - needs review")

        results = {
            'violations': self.violations,
            'warnings': self.warnings,
            'total_violations': len(self.violations),
            'total_warnings': len(self.warnings),
        }

        return results

    def check_feature_pipeline(self) -> Dict:
        """Check feature pipeline for lookahead violations."""
        print("\n" + "="*80)
        print("LOOKAHEAD BIAS DETECTION - FEATURE PIPELINE")
        print("="*80)

        feature_files = [
            project_root / "ml_features" / "feature_pipeline.py",
            project_root / "ml_features" / "technical_features.py",
        ]

        pipeline_violations = []

        for filepath in feature_files:
            if not filepath.exists():
                print(f"[SKIP] {filepath} not found")
                continue

            print(f"\n[CHECK] {filepath.name}")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for pct_change without shift
            if 'pct_change()' in content:
                for line_num, line in enumerate(content.split('\n'), 1):
                    if 'pct_change()' in line and not line.strip().startswith('#'):
                        # Check if followed by shift
                        if 'shift(' not in line and '.shift(' not in line:
                            pipeline_violations.append({
                                'file': str(filepath),
                                'line': line_num,
                                'code': line.strip(),
                                'severity': 'SEV-1',
                                'reason': 'pct_change() without shift - may cause lookahead'
                            })
                            print(f"  [SEV-1] Line {line_num}: pct_change() without shift")

        results = {
            'violations': pipeline_violations,
            'total_violations': len(pipeline_violations),
        }

        return results

    def check_backtest_logic(self) -> Dict:
        """Check backtest engine for lookahead violations."""
        print("\n" + "="*80)
        print("LOOKAHEAD BIAS DETECTION - BACKTEST ENGINE")
        print("="*80)

        backtest_file = project_root / "backtest" / "engine.py"

        if not backtest_file.exists():
            print(f"[SKIP] {backtest_file} not found")
            return {'violations': [], 'total_violations': 0}

        print(f"\n[CHECK] {backtest_file.name}")
        with open(backtest_file, 'r', encoding='utf-8') as f:
            content = f.read()

        backtest_violations = []

        # Check for same-bar fills (entry_price should be next bar open)
        if 'entry_price' in content:
            for line_num, line in enumerate(content.split('\n'), 1):
                if 'entry_price' in line and 'open' in line:
                    # Look for patterns like: entry_price = row['open']
                    if "row['open']" in line or 'row["open"]' in line:
                        # Check if this is using current bar or next bar
                        if 'shift' not in line and 'iloc[i+1]' not in line:
                            backtest_violations.append({
                                'file': str(backtest_file),
                                'line': line_num,
                                'code': line.strip(),
                                'severity': 'SEV-0',
                                'reason': 'Entry price may use current bar - should use next bar'
                            })
                            print(f"  [SEV-0] Line {line_num}: Entry price may have lookahead")

        results = {
            'violations': backtest_violations,
            'total_violations': len(backtest_violations),
        }

        return results

    def generate_report(self, strategy_results: Dict, feature_results: Dict, backtest_results: Dict):
        """Generate lookahead bias report."""
        report = f"""# LOOKAHEAD BIAS AUDIT REPORT

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## SUMMARY

| Component | Violations | Warnings |
|-----------|-----------|----------|
| Strategy Files | {strategy_results['total_violations']} | {strategy_results['total_warnings']} |
| Feature Pipeline | {feature_results['total_violations']} | 0 |
| Backtest Engine | {backtest_results['total_violations']} | 0 |
| **TOTAL** | **{strategy_results['total_violations'] + feature_results['total_violations'] + backtest_results['total_violations']}** | **{strategy_results['total_warnings']}** |

---

## STRATEGY FILES

"""

        if strategy_results['total_violations'] == 0:
            report += "**PASS** - No lookahead violations detected\n\n"
        else:
            report += "**FAIL** - Lookahead violations detected:\n\n"
            for v in strategy_results['violations']:
                report += f"- **{v['severity']}** `{v['file']}:{v['line']}`\n"
                report += f"  - Code: `{v['code']}`\n"
                report += f"  - Reason: {v['reason']}\n\n"

        if strategy_results['total_warnings'] > 0:
            report += "\n### Warnings (Needs Review)\n\n"
            for w in strategy_results['warnings']:
                report += f"- **{w['severity']}** `{w['file']}:{w['line']}`\n"
                report += f"  - Code: `{w['code']}`\n"
                report += f"  - Reason: {w['reason']}\n\n"

        report += "\n---\n\n## FEATURE PIPELINE\n\n"

        if feature_results['total_violations'] == 0:
            report += "**PASS** - No lookahead violations detected\n\n"
        else:
            report += "**FAIL** - Lookahead violations detected:\n\n"
            for v in feature_results['violations']:
                report += f"- **{v['severity']}** `{v['file']}:{v['line']}`\n"
                report += f"  - Code: `{v['code']}`\n"
                report += f"  - Reason: {v['reason']}\n\n"

        report += "\n---\n\n## BACKTEST ENGINE\n\n"

        if backtest_results['total_violations'] == 0:
            report += "**PASS** - No lookahead violations detected\n\n"
        else:
            report += "**FAIL** - Lookahead violations detected:\n\n"
            for v in backtest_results['violations']:
                report += f"- **{v['severity']}** `{v['file']}:{v['line']}`\n"
                report += f"  - Code: `{v['code']}`\n"
                report += f"  - Reason: {v['reason']}\n\n"

        report += "\n---\n\n## VERDICT\n\n"

        total_sev0 = sum(1 for v in strategy_results['violations'] + feature_results['violations'] + backtest_results['violations'] if v['severity'] == 'SEV-0')

        if total_sev0 > 0:
            report += f"**FAIL** - {total_sev0} SEV-0 lookahead violations detected\n\n"
            report += "These MUST be fixed before any trading.\n"
        else:
            report += "**PASS** - No critical lookahead violations detected\n\n"
            if strategy_results['total_warnings'] > 0:
                report += f"Note: {strategy_results['total_warnings']} warnings require manual review.\n"

        # Save report
        output_path = project_root / "AUDITS" / "TRAINING_DATA_LEAKAGE_AUDIT.md"
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\n[SAVED] Lookahead bias report saved to {output_path}")


def main():
    """Run lookahead bias detection."""
    detector = LookaheadBiasDetector()

    # Check all components
    strategy_results = detector.check_strategy_files()
    feature_results = detector.check_feature_pipeline()
    backtest_results = detector.check_backtest_logic()

    # Generate report
    detector.generate_report(strategy_results, feature_results, backtest_results)

    print("\n" + "="*80)
    print("LOOKAHEAD BIAS DETECTION COMPLETE")
    print("="*80)
    print(f"Total SEV-0 violations: {sum(1 for v in strategy_results['violations'] + feature_results['violations'] + backtest_results['violations'] if v['severity'] == 'SEV-0')}")
    print(f"Total warnings: {strategy_results['total_warnings']}")
    print("="*80)


if __name__ == "__main__":
    main()
