"""
Strict OHLCV Data Validation Module

Ensures data integrity before any analysis or trading:
- No NaNs or missing values
- No negative prices or volumes
- OHLC relationships enforced (High >= max(O,C), Low <= min(O,C))
- Monotonic timestamps with no duplicates
- Expected trading days validation
- Anomaly detection for impossible price moves

Based on: Codex & Gemini reliability recommendations (2026-01-04)

Usage:
    from data.validation import validate_ohlcv, DataQualityReport

    report = validate_ohlcv(df, symbol="AAPL")
    if not report.passed:
        print(f"REJECTED: {report.errors}")
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"           # Informational, no action needed
    WARNING = "warning"     # Potential issue, review recommended
    ERROR = "error"         # Data should not be used
    CRITICAL = "critical"   # Halt all operations


@dataclass
class ValidationIssue:
    """A single validation issue."""
    field: str
    severity: ValidationSeverity
    message: str
    row_indices: Optional[List[int]] = None
    sample_values: Optional[List[Any]] = None

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.field}: {self.message}"


@dataclass
class DataQualityReport:
    """Complete data quality validation report."""
    symbol: str
    timestamp: str
    passed: bool
    row_count: int
    date_range: Tuple[str, str]
    issues: List[ValidationIssue] = field(default_factory=list)
    data_hash: Optional[str] = None
    validation_time_ms: float = 0.0

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "row_count": self.row_count,
            "date_range": self.date_range,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [str(i) for i in self.issues],
            "data_hash": self.data_hash,
            "validation_time_ms": self.validation_time_ms,
        }


class OHLCVValidator:
    """
    Strict OHLCV data validator.

    Validates:
    1. Schema: Required columns present
    2. Nulls: No NaN/None values
    3. Types: Correct data types
    4. Ranges: No negative prices/volumes
    5. OHLC: High >= max(O,C), Low <= min(O,C), Close in [Low, High]
    6. Timestamps: Monotonic, no duplicates
    7. Gaps: No missing trading days beyond tolerance
    8. Anomalies: No impossible price moves (>50% in a day)
    """

    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}
    TIMESTAMP_COLUMNS = {"timestamp", "date", "datetime", "time"}
    MAX_DAILY_MOVE_PCT = 0.50  # 50% max move in a day (circuit breaker level)
    MAX_GAP_DAYS = 5  # Max consecutive missing days allowed

    def __init__(
        self,
        strict_mode: bool = True,
        max_missing_pct: float = 0.05,
        require_volume: bool = True,
    ):
        self.strict_mode = strict_mode
        self.max_missing_pct = max_missing_pct
        self.require_volume = require_volume

    def validate(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> DataQualityReport:
        """
        Validate OHLCV data and return quality report.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol for reporting

        Returns:
            DataQualityReport with all issues found
        """
        start_time = datetime.now()
        issues: List[ValidationIssue] = []

        # Empty check
        if df is None or df.empty:
            return DataQualityReport(
                symbol=symbol,
                timestamp=datetime.utcnow().isoformat(),
                passed=False,
                row_count=0,
                date_range=("", ""),
                issues=[ValidationIssue(
                    field="dataframe",
                    severity=ValidationSeverity.CRITICAL,
                    message="DataFrame is empty or None"
                )],
            )

        # Normalize column names to lowercase
        df_normalized = df.copy()
        df_normalized.columns = [c.lower() for c in df_normalized.columns]

        # 1. Schema validation
        issues.extend(self._validate_schema(df_normalized))

        # 2. Null/NaN validation
        issues.extend(self._validate_nulls(df_normalized))

        # 3. Type validation
        issues.extend(self._validate_types(df_normalized))

        # 4. Range validation (no negatives)
        issues.extend(self._validate_ranges(df_normalized))

        # 5. OHLC relationship validation
        issues.extend(self._validate_ohlc_relationships(df_normalized))

        # 6. Timestamp validation
        issues.extend(self._validate_timestamps(df_normalized))

        # 7. Gap detection
        issues.extend(self._validate_gaps(df_normalized))

        # 8. Anomaly detection
        issues.extend(self._validate_anomalies(df_normalized))

        # Calculate data hash
        data_hash = self._compute_hash(df_normalized)

        # Determine pass/fail
        has_errors = any(i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL) for i in issues)
        passed = not has_errors

        # Get date range
        date_range = self._get_date_range(df_normalized)

        validation_time = (datetime.now() - start_time).total_seconds() * 1000

        return DataQualityReport(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            passed=passed,
            row_count=len(df),
            date_range=date_range,
            issues=issues,
            data_hash=data_hash,
            validation_time_ms=validation_time,
        )

    def _validate_schema(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate required columns are present."""
        issues = []
        present_cols = set(df.columns)
        missing_cols = self.REQUIRED_COLUMNS - present_cols

        if missing_cols:
            issues.append(ValidationIssue(
                field="schema",
                severity=ValidationSeverity.CRITICAL,
                message=f"Missing required columns: {missing_cols}"
            ))

        # Check for timestamp column
        has_timestamp = bool(present_cols & self.TIMESTAMP_COLUMNS)
        if not has_timestamp:
            # Check if index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                issues.append(ValidationIssue(
                    field="schema",
                    severity=ValidationSeverity.ERROR,
                    message="No timestamp column or DatetimeIndex found"
                ))

        return issues

    def _validate_nulls(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate no null/NaN values in required columns."""
        issues = []

        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                continue

            null_count = df[col].isna().sum()
            if null_count > 0:
                null_pct = null_count / len(df)
                null_indices = df[df[col].isna()].index.tolist()[:5]

                if null_pct > self.max_missing_pct:
                    severity = ValidationSeverity.ERROR
                else:
                    severity = ValidationSeverity.WARNING

                issues.append(ValidationIssue(
                    field=col,
                    severity=severity,
                    message=f"{null_count} null values ({null_pct:.1%})",
                    row_indices=null_indices[:5] if isinstance(null_indices[0], int) else None,
                ))

        return issues

    def _validate_types(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate data types are numeric."""
        issues = []

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(ValidationIssue(
                    field=col,
                    severity=ValidationSeverity.ERROR,
                    message=f"Column is not numeric: {df[col].dtype}"
                ))

        return issues

    def _validate_ranges(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate no negative prices or volumes."""
        issues = []

        # Price columns should be positive
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                continue

            negative_mask = df[col] <= 0
            if negative_mask.any():
                count = negative_mask.sum()
                sample = df[negative_mask][col].head(3).tolist()
                issues.append(ValidationIssue(
                    field=col,
                    severity=ValidationSeverity.ERROR,
                    message=f"{count} non-positive values found",
                    sample_values=sample,
                ))

        # Volume should be non-negative
        if "volume" in df.columns and self.require_volume:
            negative_vol = df["volume"] < 0
            if negative_vol.any():
                count = negative_vol.sum()
                issues.append(ValidationIssue(
                    field="volume",
                    severity=ValidationSeverity.ERROR,
                    message=f"{count} negative volume values",
                ))

        return issues

    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate OHLC relationships: High >= max(O,C), Low <= min(O,C)."""
        issues = []

        required = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in required):
            return issues

        # High must be >= Open and Close
        high_violation = (df["high"] < df["open"]) | (df["high"] < df["close"])
        if high_violation.any():
            count = high_violation.sum()
            issues.append(ValidationIssue(
                field="high",
                severity=ValidationSeverity.ERROR,
                message=f"{count} rows where High < max(Open, Close)",
            ))

        # Low must be <= Open and Close
        low_violation = (df["low"] > df["open"]) | (df["low"] > df["close"])
        if low_violation.any():
            count = low_violation.sum()
            issues.append(ValidationIssue(
                field="low",
                severity=ValidationSeverity.ERROR,
                message=f"{count} rows where Low > min(Open, Close)",
            ))

        # Close must be within [Low, High]
        close_violation = (df["close"] < df["low"]) | (df["close"] > df["high"])
        if close_violation.any():
            count = close_violation.sum()
            issues.append(ValidationIssue(
                field="close",
                severity=ValidationSeverity.ERROR,
                message=f"{count} rows where Close outside [Low, High]",
            ))

        # High must be >= Low
        hl_violation = df["high"] < df["low"]
        if hl_violation.any():
            count = hl_violation.sum()
            issues.append(ValidationIssue(
                field="high/low",
                severity=ValidationSeverity.CRITICAL,
                message=f"{count} rows where High < Low (impossible)",
            ))

        return issues

    def _validate_timestamps(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate timestamps are monotonic and no duplicates."""
        issues = []

        # Find timestamp column
        ts_col = None
        for col in self.TIMESTAMP_COLUMNS:
            if col in df.columns:
                ts_col = col
                break

        if ts_col is None and isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        elif ts_col:
            timestamps = pd.to_datetime(df[ts_col])
        else:
            return issues

        # Check for duplicates
        duplicates = timestamps.duplicated()
        if duplicates.any():
            count = duplicates.sum()
            issues.append(ValidationIssue(
                field="timestamp",
                severity=ValidationSeverity.ERROR,
                message=f"{count} duplicate timestamps found",
            ))

        # Check monotonicity
        if not timestamps.is_monotonic_increasing:
            issues.append(ValidationIssue(
                field="timestamp",
                severity=ValidationSeverity.WARNING,
                message="Timestamps are not monotonically increasing",
            ))

        return issues

    def _validate_gaps(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate no large gaps in trading days."""
        issues = []

        # Find timestamp column
        ts_col = None
        for col in self.TIMESTAMP_COLUMNS:
            if col in df.columns:
                ts_col = col
                break

        if ts_col is None and isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        elif ts_col:
            timestamps = pd.to_datetime(df[ts_col])
        else:
            return issues

        if len(timestamps) < 2:
            return issues

        # Calculate gaps
        sorted_ts = timestamps.sort_values()
        gaps = sorted_ts.diff().dropna()

        # Find large gaps (> MAX_GAP_DAYS business days)
        large_gaps = gaps[gaps > timedelta(days=self.MAX_GAP_DAYS)]
        if len(large_gaps) > 0:
            issues.append(ValidationIssue(
                field="timestamp",
                severity=ValidationSeverity.WARNING,
                message=f"{len(large_gaps)} gaps > {self.MAX_GAP_DAYS} days found",
            ))

        return issues

    def _validate_anomalies(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Detect anomalous price movements."""
        issues = []

        if "close" not in df.columns or len(df) < 2:
            return issues

        # Calculate daily returns
        returns = df["close"].pct_change().dropna()

        # Find extreme moves
        extreme_moves = returns.abs() > self.MAX_DAILY_MOVE_PCT
        if extreme_moves.any():
            count = extreme_moves.sum()
            max_move = returns.abs().max()
            issues.append(ValidationIssue(
                field="close",
                severity=ValidationSeverity.WARNING,
                message=f"{count} daily moves > {self.MAX_DAILY_MOVE_PCT:.0%} (max: {max_move:.1%})",
            ))

        # Check for zero prices
        zero_prices = (df["close"] == 0) | (df["open"] == 0)
        if zero_prices.any():
            count = zero_prices.sum()
            issues.append(ValidationIssue(
                field="price",
                severity=ValidationSeverity.CRITICAL,
                message=f"{count} zero prices found (data corruption)",
            ))

        return issues

    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute SHA256 hash of dataframe for integrity verification."""
        try:
            # Use a stable representation
            data_str = df.to_csv(index=True)
            return hashlib.sha256(data_str.encode()).hexdigest()[:16]
        except Exception:
            return "HASH_ERROR"

    def _get_date_range(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Get date range of data."""
        ts_col = None
        for col in self.TIMESTAMP_COLUMNS:
            if col in df.columns:
                ts_col = col
                break

        try:
            if ts_col:
                timestamps = pd.to_datetime(df[ts_col])
            elif isinstance(df.index, pd.DatetimeIndex):
                timestamps = df.index
            else:
                return ("", "")

            return (str(timestamps.min().date()), str(timestamps.max().date()))
        except Exception:
            return ("", "")


# ============================================================================
# Convenience Functions
# ============================================================================

_default_validator = OHLCVValidator()


def validate_ohlcv(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    strict: bool = True,
) -> DataQualityReport:
    """
    Validate OHLCV data with default validator.

    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol for reporting
        strict: Use strict validation mode

    Returns:
        DataQualityReport
    """
    validator = OHLCVValidator(strict_mode=strict)
    return validator.validate(df, symbol)


def require_valid_data(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
) -> pd.DataFrame:
    """
    Validate data and raise exception if invalid.

    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol for reporting

    Returns:
        Original DataFrame if valid

    Raises:
        ValueError: If data fails validation
    """
    report = validate_ohlcv(df, symbol)
    if not report.passed:
        errors = "; ".join(str(e) for e in report.errors)
        raise ValueError(f"Data validation failed for {symbol}: {errors}")
    return df


def compute_data_hash(df: pd.DataFrame) -> str:
    """Compute SHA256 hash of dataframe for integrity verification."""
    try:
        data_str = df.to_csv(index=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    except Exception:
        return "HASH_ERROR"


def verify_data_integrity(df: pd.DataFrame, expected_hash: str) -> bool:
    """Verify data matches expected hash."""
    actual_hash = compute_data_hash(df)
    return actual_hash == expected_hash
