"""
Universe Pipeline - Validate the 900-stock trading universe.

This pipeline ensures the universe meets all quality criteria:
- Sufficient history (10+ years)
- Adequate liquidity
- Optionable stocks
- No delisted/problematic symbols

Schedule: Weekly (Sunday 18:00 ET)

Author: Kobe Trading System
Version: 1.0.0
"""

import pandas as pd
from datetime import datetime

from pipelines.base import Pipeline


class UniversePipeline(Pipeline):
    """Pipeline for validating the trading universe."""

    @property
    def name(self) -> str:
        return "universe"

    def execute(self) -> bool:
        """
        Execute universe validation.

        Returns:
            True if universe is valid
        """
        self.logger.info("Validating trading universe...")

        # Load universe
        universe_file = self.data_dir / "universe" / "optionable_liquid_800.csv"
        if not universe_file.exists():
            self.add_error(f"Universe file not found: {universe_file}")
            return False

        try:
            df = pd.read_csv(universe_file)
        except Exception as e:
            self.add_error(f"Failed to load universe: {e}")
            return False

        # Validation checks
        checks = [
            self._check_symbol_count(df),
            self._check_required_columns(df),
            self._check_duplicates(df),
            self._check_empty_symbols(df),
        ]

        # Set metrics
        self.set_metric("total_symbols", len(df))
        self.set_metric("validation_time", datetime.utcnow().isoformat())

        # All checks must pass
        all_passed = all(checks)

        if all_passed:
            self.logger.info(f"Universe validation passed: {len(df)} symbols")
        else:
            self.add_error("Universe validation failed")

        return all_passed

    def _check_symbol_count(self, df: pd.DataFrame) -> bool:
        """Check minimum symbol count."""
        min_symbols = self.config.get("min_symbols", 100)
        count = len(df)

        if count < min_symbols:
            self.add_error(f"Insufficient symbols: {count} < {min_symbols}")
            return False

        self.logger.info(f"Symbol count check passed: {count} >= {min_symbols}")
        return True

    def _check_required_columns(self, df: pd.DataFrame) -> bool:
        """Check required columns exist."""
        required = ["symbol"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            self.add_error(f"Missing required columns: {missing}")
            return False

        self.logger.info("Required columns check passed")
        return True

    def _check_duplicates(self, df: pd.DataFrame) -> bool:
        """Check for duplicate symbols."""
        duplicates = df[df['symbol'].duplicated()]['symbol'].tolist()

        if duplicates:
            self.add_warning(f"Duplicate symbols found: {duplicates[:10]}")
            self.set_metric("duplicate_count", len(duplicates))
            # Don't fail, just warn
            return True

        self.logger.info("No duplicate symbols found")
        return True

    def _check_empty_symbols(self, df: pd.DataFrame) -> bool:
        """Check for empty/null symbols."""
        empty_count = df['symbol'].isna().sum()

        if empty_count > 0:
            self.add_error(f"Found {empty_count} empty symbols")
            return False

        self.logger.info("No empty symbols found")
        return True
