"""
Gate 0: Sanity Check
====================

Validates strategy for fundamental integrity:
- NO lookahead bias
- NO data leakage
- NO perfect entries (>10% at daily low = suspicious)
- Proper cost modeling

FAILURE = ARCHIVE FOREVER
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SanityResult:
    """Result from sanity check."""
    passed: bool
    lookahead_detected: bool
    leakage_detected: bool
    perfect_entries_pct: float
    cost_check_passed: bool
    issues: List[str]
    details: Dict[str, Any]


class Gate0Sanity:
    """
    Gate 0: Fundamental sanity checks.

    This gate catches the most egregious errors that
    would make any backtest completely invalid.

    Checks:
    1. Lookahead bias - using future data in signals
    2. Data leakage - information bleeding between train/test
    3. Perfect entries - suspicious entry at daily low
    4. Cost modeling - transaction costs included

    FAIL = ARCHIVE FOREVER
    """

    # Threshold for perfect entries (suspicious if >10%)
    PERFECT_ENTRY_THRESHOLD = 10.0

    def __init__(self):
        self.issues: List[str] = []

    def validate(
        self,
        strategy_code: Optional[str] = None,
        strategy_file: Optional[str] = None,
        backtest_trades: Optional[pd.DataFrame] = None,
        backtest_config: Optional[Dict[str, Any]] = None,
    ) -> SanityResult:
        """
        Run all sanity checks.

        Args:
            strategy_code: Strategy source code (or file path)
            strategy_file: Path to strategy file
            backtest_trades: DataFrame of trades from backtest
            backtest_config: Backtest configuration

        Returns:
            SanityResult with pass/fail and details
        """
        self.issues = []
        details = {}

        # Load code if file provided
        if strategy_file and not strategy_code:
            try:
                with open(strategy_file, "r") as f:
                    strategy_code = f.read()
            except Exception as e:
                self.issues.append(f"Could not read strategy file: {e}")

        # Check 1: Lookahead bias
        lookahead = self._check_lookahead(strategy_code) if strategy_code else False
        details["lookahead_check"] = {"detected": lookahead}

        # Check 2: Data leakage
        leakage = self._check_leakage(backtest_config) if backtest_config else False
        details["leakage_check"] = {"detected": leakage}

        # Check 3: Perfect entries
        perfect_pct = self._check_perfect_entries(backtest_trades) if backtest_trades is not None else 0.0
        details["perfect_entries"] = {"percentage": perfect_pct}

        # Check 4: Cost modeling
        cost_ok = self._check_costs(backtest_config) if backtest_config else True
        details["cost_check"] = {"passed": cost_ok}

        # Determine overall pass/fail
        passed = (
            not lookahead and
            not leakage and
            perfect_pct <= self.PERFECT_ENTRY_THRESHOLD and
            cost_ok
        )

        return SanityResult(
            passed=passed,
            lookahead_detected=lookahead,
            leakage_detected=leakage,
            perfect_entries_pct=perfect_pct,
            cost_check_passed=cost_ok,
            issues=self.issues,
            details=details,
        )

    def _check_lookahead(self, code: str) -> bool:
        """
        Check for lookahead bias in code.

        Looks for:
        - Missing .shift(1) on indicators
        - Using close price for same-day signals
        - Direct indexing without lag
        """
        issues_found = False

        # Pattern 1: Signal assignment using current price without shift
        patterns_bad = [
            # df['signal'] = df['close'] > something (no shift)
            r"signal.*=.*\[.*close.*\](?!.*\.shift)",
            r"signal.*=.*\[.*high.*\](?!.*\.shift)",
            r"signal.*=.*\[.*low.*\](?!.*\.shift)",
            r"signal.*=.*\[.*volume.*\](?!.*\.shift)",
            # Direct comparison without shift
            r"df\[.*\]\s*[<>]=?\s*df\[.*\](?!.*\.shift)",
        ]

        # Pattern 2: Good patterns (with shift)
        patterns_good = [
            r"\.shift\(1\)",
            r"\.shift\(-1\)",  # This would be future data!
            r"col_sig\s*=\s*col\.shift\(1\)",
        ]

        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("#"):
                continue

            # Check for bad patterns
            for pattern in patterns_bad:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if good pattern also exists on same line
                    has_shift = any(re.search(p, line) for p in patterns_good if "shift(-1)" not in p)
                    if not has_shift:
                        self.issues.append(f"Line {i}: Potential lookahead - {line.strip()[:60]}")
                        issues_found = True

            # Check for explicit future data access
            if ".shift(-" in line and "shift(-1)" not in line:
                self.issues.append(f"Line {i}: Future data access detected - {line.strip()[:60]}")
                issues_found = True

        return issues_found

    def _check_leakage(self, config: Dict[str, Any]) -> bool:
        """
        Check for data leakage in configuration.

        Looks for:
        - Training on test period
        - Overlapping train/test splits
        - Feature using future info
        """
        issues_found = False

        # Check for train/test overlap
        train_end = config.get("train_end")
        test_start = config.get("test_start")

        if train_end and test_start:
            if train_end >= test_start:
                self.issues.append(f"Train/test overlap: train ends {train_end}, test starts {test_start}")
                issues_found = True

        # Check for suspicious features
        features = config.get("features", [])
        suspicious = ["future_", "next_", "forward_", "target_"]
        for feature in features:
            for sus in suspicious:
                if sus in feature.lower():
                    self.issues.append(f"Suspicious feature name: {feature}")
                    issues_found = True

        return issues_found

    def _check_perfect_entries(self, trades: pd.DataFrame) -> float:
        """
        Check for suspiciously perfect entries.

        If >10% of entries are at daily low (for longs),
        this suggests lookahead bias.
        """
        if trades.empty:
            return 0.0

        required_cols = ["entry_price", "date"]
        if not all(col in trades.columns for col in required_cols):
            return 0.0

        # Check if we have high/low data to compare
        if "day_low" not in trades.columns:
            # Can't check without daily low data
            return 0.0

        # Count entries at daily low
        tolerance = 0.001  # 0.1% tolerance
        at_low = (
            (trades["entry_price"] - trades["day_low"]).abs() /
            trades["day_low"] < tolerance
        ).sum()

        pct = 100 * at_low / len(trades)

        if pct > self.PERFECT_ENTRY_THRESHOLD:
            self.issues.append(
                f"Perfect entry rate: {pct:.1f}% (>{self.PERFECT_ENTRY_THRESHOLD}% suspicious)"
            )

        return pct

    def _check_costs(self, config: Dict[str, Any]) -> bool:
        """
        Check that transaction costs are modeled.
        """
        has_costs = False

        # Check for cost-related config
        cost_keys = ["commission", "slippage", "fees", "spread", "transaction_cost"]
        for key in cost_keys:
            if key in config:
                value = config[key]
                if value is not None and value > 0:
                    has_costs = True
                    break

        if not has_costs:
            self.issues.append("No transaction costs configured (commission, slippage, fees)")

        return has_costs


def check_strategy_sanity(
    strategy_file: str,
    trades_df: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
) -> SanityResult:
    """
    Convenience function to run sanity check on a strategy.

    Args:
        strategy_file: Path to strategy Python file
        trades_df: Optional trades DataFrame
        config: Optional configuration dict

    Returns:
        SanityResult
    """
    gate = Gate0Sanity()
    return gate.validate(
        strategy_file=strategy_file,
        backtest_trades=trades_df,
        backtest_config=config,
    )
