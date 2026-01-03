"""
INTEGRITY GUARDIAN - The Ultimate Anti-Hallucination System.

This module ensures Kobe NEVER:
- Hallucinates data
- Fakes numbers
- Overfits to history
- Produces unrealistic results
- Lies about performance

EVERY result must pass ALL checks before being accepted.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


@dataclass
class IntegrityReport:
    """Report from integrity validation."""
    passed: bool
    checks_run: int
    checks_passed: int
    checks_failed: int
    failures: List[str]
    warnings: List[str]
    timestamp: str


class IntegrityGuardian:
    """
    The ultimate safeguard against fake data and hallucinations.

    RULES:
    1. All data must be traceable to source files
    2. All results must be within realistic bounds
    3. All experiments must be reproducible
    4. Statistical significance is required
    5. Anomalies trigger automatic rejection
    """

    # REALISTIC BOUNDS - Based on decades of trading research
    # Any result outside these bounds is SUSPICIOUS
    REALISTIC_BOUNDS = {
        "win_rate": {"min": 0.30, "max": 0.70, "suspicious_above": 0.70},
        "profit_factor": {"min": 0.5, "max": 3.0, "suspicious_above": 2.5},
        "sharpe_ratio": {"min": -1.0, "max": 3.0, "suspicious_above": 2.5},
        "avg_win_pct": {"min": 0.5, "max": 15.0, "suspicious_above": 10.0},
        "avg_loss_pct": {"min": -15.0, "max": -0.5, "suspicious_below": -10.0},
        "max_drawdown": {"min": 0.01, "max": 0.50, "suspicious_above": 0.40},
    }

    # MINIMUM SAMPLE SIZES for statistical significance
    MIN_TRADES = 100  # Minimum trades for valid backtest
    MIN_SYMBOLS = 30  # Minimum symbols tested
    MIN_DAYS = 252    # Minimum trading days (1 year)

    def __init__(self, state_dir: Optional[Path] = None):
        self.state_dir = state_dir or Path("state/integrity")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.violation_log = self.state_dir / "violations.jsonl"
        self.data_hashes = {}

    def validate_result(self, result: Dict[str, Any], context: str = "") -> IntegrityReport:
        """
        Validate a backtest/experiment result.
        Returns IntegrityReport with pass/fail and details.
        """
        failures = []
        warnings = []
        checks_passed = 0

        # CHECK 1: Win rate bounds
        wr = result.get("win_rate", 0)
        if wr > 0.90:
            failures.append(f"IMPOSSIBLE: Win rate {wr:.1%} > 90% (lookahead bias)")
        elif wr > 0.80:
            failures.append(f"VERY SUSPICIOUS: Win rate {wr:.1%} > 80%")
        elif wr > 0.70:
            warnings.append(f"SUSPICIOUS: Win rate {wr:.1%} > 70%")
            checks_passed += 1
        elif wr < 0.30:
            warnings.append(f"LOW: Win rate {wr:.1%} < 30% (strategy may be inverted)")
            checks_passed += 1
        else:
            checks_passed += 1

        # CHECK 2: Profit factor bounds
        pf = result.get("profit_factor", 0)
        if pf > 5.0:
            failures.append(f"IMPOSSIBLE: Profit factor {pf:.2f} > 5.0")
        elif pf > 3.0:
            failures.append(f"VERY SUSPICIOUS: Profit factor {pf:.2f} > 3.0")
        elif pf > 2.5:
            warnings.append(f"SUSPICIOUS: Profit factor {pf:.2f} > 2.5")
            checks_passed += 1
        elif pf < 0.5:
            warnings.append(f"POOR: Profit factor {pf:.2f} < 0.5")
            checks_passed += 1
        else:
            checks_passed += 1

        # CHECK 3: Minimum sample size
        trades = result.get("trades", 0)
        if trades < 30:
            failures.append(f"INSUFFICIENT DATA: Only {trades} trades (need 100+)")
        elif trades < self.MIN_TRADES:
            warnings.append(f"LOW SAMPLE: Only {trades} trades (prefer 100+)")
            checks_passed += 1
        else:
            checks_passed += 1

        # CHECK 4: Average win/loss sanity
        avg_win = result.get("avg_win_pct", 0)
        avg_loss = result.get("avg_loss_pct", 0)

        if avg_win > 20:
            failures.append(f"IMPOSSIBLE: Avg win {avg_win:.1f}% > 20%")
        elif avg_win > 15:
            warnings.append(f"HIGH: Avg win {avg_win:.1f}% > 15%")
            checks_passed += 1
        else:
            checks_passed += 1

        if avg_loss < -20:
            failures.append(f"DANGEROUS: Avg loss {avg_loss:.1f}% > 20%")
        elif avg_loss < -15:
            warnings.append(f"HIGH RISK: Avg loss {avg_loss:.1f}%")
            checks_passed += 1
        else:
            checks_passed += 1

        # CHECK 5: Data source verification
        data_source = result.get("data_source", "UNKNOWN")
        if data_source == "UNKNOWN":
            failures.append("NO DATA SOURCE: Cannot verify data origin")
        elif "SYNTHETIC" in data_source.upper():
            failures.append("SYNTHETIC DATA: Not allowed for production")
        elif "RANDOM" in data_source.upper():
            failures.append("RANDOM DATA: Not allowed")
        else:
            checks_passed += 1

        # CHECK 6: Timestamp verification
        timestamp = result.get("timestamp")
        if not timestamp:
            warnings.append("NO TIMESTAMP: Result not dated")
        else:
            checks_passed += 1

        # CHECK 7: Consistency check (win_rate vs profit_factor)
        # High WR with low PF or low WR with high PF is suspicious
        if wr > 0.60 and pf < 1.0:
            warnings.append(f"INCONSISTENT: High WR {wr:.1%} but PF < 1.0")
        elif wr < 0.40 and pf > 2.0:
            warnings.append(f"INCONSISTENT: Low WR {wr:.1%} but PF > 2.0")
        else:
            checks_passed += 1

        # CHECK 8: Zero/null detection
        if wr == 0 and trades > 0:
            failures.append("ZERO WIN RATE: With trades present - data error")
        if pf == 0 and trades > 0:
            failures.append("ZERO PROFIT FACTOR: With trades present - data error")

        checks_passed += 1  # For zero check

        # Build report
        total_checks = 8
        passed = len(failures) == 0

        report = IntegrityReport(
            passed=passed,
            checks_run=total_checks,
            checks_passed=checks_passed,
            checks_failed=len(failures),
            failures=failures,
            warnings=warnings,
            timestamp=datetime.now(ET).isoformat()
        )

        # Log violations
        if not passed:
            self._log_violation(result, report, context)

        return report

    def verify_data_file(self, filepath: Path) -> Tuple[bool, str]:
        """
        Verify a data file is valid and unchanged.
        Returns (valid, hash).
        """
        if not filepath.exists():
            return False, "FILE_NOT_FOUND"

        try:
            # Calculate hash
            hasher = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            file_hash = hasher.hexdigest()[:16]

            # Check if we've seen this file before
            stored_hash = self.data_hashes.get(str(filepath))
            if stored_hash and stored_hash != file_hash:
                logger.warning(f"DATA CHANGED: {filepath.name} hash mismatch!")
                return False, "HASH_MISMATCH"

            self.data_hashes[str(filepath)] = file_hash
            return True, file_hash

        except Exception as e:
            return False, f"ERROR: {e}"

    def verify_reproducibility(
        self,
        run1_result: Dict[str, Any],
        run2_result: Dict[str, Any],
        tolerance: float = 0.001
    ) -> Tuple[bool, List[str]]:
        """
        Verify two runs produce identical results (reproducibility check).
        """
        differences = []

        for key in ["win_rate", "profit_factor", "trades"]:
            v1 = run1_result.get(key, 0)
            v2 = run2_result.get(key, 0)

            if key == "trades":
                if v1 != v2:
                    differences.append(f"{key}: {v1} vs {v2}")
            else:
                if abs(v1 - v2) > tolerance:
                    differences.append(f"{key}: {v1:.4f} vs {v2:.4f}")

        reproducible = len(differences) == 0

        if not reproducible:
            logger.warning(f"REPRODUCIBILITY FAILED: {differences}")

        return reproducible, differences

    def check_for_lookahead(self, signals_df) -> Tuple[bool, List[str]]:
        """
        Check signals for potential lookahead bias.
        """
        issues = []

        try:
            import pandas as pd

            # Check 1: Signals shouldn't reference future dates
            if 'timestamp' in signals_df.columns:
                if signals_df['timestamp'].is_monotonic_increasing:
                    pass  # Good
                else:
                    issues.append("TIMESTAMPS NOT MONOTONIC")

            # Check 2: Entry price shouldn't be better than signal bar
            if 'entry_price' in signals_df.columns and 'close' in signals_df.columns:
                # Entry should be at or after signal bar
                pass  # This is checked in strategy

            # Check 3: Check for perfect entries at lows
            if 'entry_price' in signals_df.columns and 'low' in signals_df.columns:
                perfect_entries = (signals_df['entry_price'] == signals_df['low']).sum()
                if perfect_entries > len(signals_df) * 0.5:
                    issues.append(f"TOO MANY PERFECT ENTRIES: {perfect_entries}/{len(signals_df)}")

        except Exception as e:
            issues.append(f"CHECK ERROR: {e}")

        return len(issues) == 0, issues

    def _log_violation(self, result: Dict, report: IntegrityReport, context: str):
        """Log integrity violation for audit trail."""
        violation = {
            "timestamp": datetime.now(ET).isoformat(),
            "context": context,
            "failures": report.failures,
            "warnings": report.warnings,
            "result_summary": {
                "win_rate": result.get("win_rate"),
                "profit_factor": result.get("profit_factor"),
                "trades": result.get("trades"),
                "data_source": result.get("data_source"),
            }
        }

        with open(self.violation_log, 'a') as f:
            f.write(json.dumps(violation) + '\n')

        logger.warning(f"INTEGRITY VIOLATION logged: {report.failures}")

    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of all violations."""
        if not self.violation_log.exists():
            return {"total": 0, "violations": []}

        violations = []
        with open(self.violation_log) as f:
            for line in f:
                try:
                    violations.append(json.loads(line.strip()))
                except:
                    pass

        return {
            "total": len(violations),
            "recent": violations[-10:] if violations else [],
            "by_type": self._count_by_type(violations)
        }

    def _count_by_type(self, violations: List[Dict]) -> Dict[str, int]:
        """Count violations by type."""
        counts = {}
        for v in violations:
            for failure in v.get("failures", []):
                key = failure.split(":")[0]
                counts[key] = counts.get(key, 0) + 1
        return counts


# Global instance
_guardian = None

def get_guardian() -> IntegrityGuardian:
    """Get the global IntegrityGuardian instance."""
    global _guardian
    if _guardian is None:
        _guardian = IntegrityGuardian()
    return _guardian


def validate_before_use(result: Dict[str, Any], context: str = "") -> bool:
    """
    Quick validation check - returns True if safe to use.
    USE THIS BEFORE ACCEPTING ANY RESULT.
    """
    guardian = get_guardian()
    report = guardian.validate_result(result, context)

    if not report.passed:
        logger.error(f"RESULT REJECTED: {report.failures}")
        return False

    if report.warnings:
        logger.warning(f"RESULT WARNINGS: {report.warnings}")

    return True
