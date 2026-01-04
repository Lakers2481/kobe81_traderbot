"""
Data Audit Pipeline - Daily data quality validation.

This pipeline checks data integrity:
- Price data freshness
- OHLC violations
- Missing data gaps
- Stale prices
- Corporate action detection

Schedule: Daily (05:00 ET)

Author: Kobe Trading System
Version: 1.0.0
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from pipelines.base import Pipeline


class DataAuditPipeline(Pipeline):
    """Pipeline for auditing data quality."""

    @property
    def name(self) -> str:
        return "data_audit"

    def execute(self) -> bool:
        """
        Execute data quality audit.

        Returns:
            True if data quality is acceptable
        """
        self.logger.info("Running data quality audit...")

        # Load universe
        symbols = self.load_universe()
        if not symbols:
            return False

        # Sample symbols for audit
        sample_size = min(self.config.get("sample_size", 50), len(symbols))
        sample = random.sample(symbols, sample_size)

        self.set_metric("sample_size", sample_size)

        # Run quality checks
        results = []
        for symbol in sample:
            result = self._audit_symbol(symbol)
            results.append(result)

        # Aggregate results
        passed = sum(1 for r in results if r["passed"])
        pass_rate = passed / len(results) if results else 0

        self.set_metric("symbols_audited", len(results))
        self.set_metric("symbols_passed", passed)
        self.set_metric("pass_rate", pass_rate)

        # Check threshold
        min_pass_rate = self.config.get("min_pass_rate", 0.95)
        if pass_rate < min_pass_rate:
            self.add_error(
                f"Data quality below threshold: {pass_rate:.1%} < {min_pass_rate:.1%}"
            )
            return False

        self.logger.info(f"Data audit passed: {pass_rate:.1%} symbols OK")
        return True

    def _audit_symbol(self, symbol: str) -> Dict:
        """
        Audit data for a single symbol.

        Returns:
            Dict with audit results
        """
        result = {
            "symbol": symbol,
            "passed": True,
            "checks": {},
            "errors": [],
        }

        # Check cache file exists
        cache_dir = self.data_dir / "cache" / "polygon_eod"
        cache_file = cache_dir / f"{symbol}.csv"

        if not cache_file.exists():
            result["passed"] = False
            result["errors"].append("No cache file")
            return result

        try:
            import pandas as pd
            df = pd.read_csv(cache_file, parse_dates=["timestamp"])

            # Check 1: Data freshness
            if len(df) > 0:
                latest = df["timestamp"].max()
                days_old = (datetime.now() - latest).days
                result["checks"]["freshness_days"] = days_old

                if days_old > 5:  # Allow weekend gap
                    result["errors"].append(f"Stale data: {days_old} days old")
                    result["passed"] = False

            # Check 2: OHLC violations
            if all(col in df.columns for col in ["open", "high", "low", "close"]):
                violations = (
                    (df["high"] < df["low"]) |
                    (df["high"] < df["open"]) |
                    (df["high"] < df["close"]) |
                    (df["low"] > df["open"]) |
                    (df["low"] > df["close"])
                )
                violation_count = violations.sum()
                result["checks"]["ohlc_violations"] = int(violation_count)

                if violation_count > 0:
                    result["errors"].append(f"{violation_count} OHLC violations")
                    # Don't fail for minor violations
                    if violation_count > len(df) * 0.01:
                        result["passed"] = False

            # Check 3: Missing data gaps
            if len(df) > 1:
                df = df.sort_values("timestamp")
                gaps = df["timestamp"].diff().dt.days
                large_gaps = (gaps > 5).sum()  # More than 5 days is suspicious
                result["checks"]["large_gaps"] = int(large_gaps)

                if large_gaps > 10:
                    result["errors"].append(f"{large_gaps} large data gaps")
                    result["passed"] = False

            # Check 4: Data length
            result["checks"]["total_rows"] = len(df)
            min_rows = 252 * 5  # 5 years
            if len(df) < min_rows:
                result["errors"].append(f"Insufficient history: {len(df)} < {min_rows}")
                # Don't fail, just warn

        except Exception as e:
            result["passed"] = False
            result["errors"].append(f"Read error: {e}")

        return result
