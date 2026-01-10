"""
Pandera schemas for OHLCV data validation.

Production-grade DataFrame validation using Pandera v0.28.0+.
Type-safe schemas with statistical checks for trading data.

Usage:
    from data.schemas.ohlcv_schema import OHLCVSchema, ohlcv_schema

    # Validate DataFrame
    df = fetch_daily_bars_polygon(...)
    validated_df = ohlcv_schema.validate(df, lazy=True)

    # Use as decorator
    @pa.check_input(ohlcv_schema)
    def process_bars(df: pd.DataFrame) -> pd.DataFrame:
        ...

Features:
- Schema validation (required columns, types)
- Null checks (zero NaNs allowed)
- OHLC relationships (High >= max(O,C), Low <= min(O,C))
- Range validation (prices > 0, volume >= 0)
- Statistical bounds (optional - for research)
- Timestamp monotonicity and uniqueness

Based on:
- Pandera best practices 2025
- OHLCV validation research
- Kobe's existing validation.py

Author: Claude Opus 4.5
Date: 2026-01-07
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera import Column, DataFrameSchema
from datetime import datetime
from typing import Optional
import pandas as pd


class OHLCVSchema(pa.DataFrameModel):
    """
    Pandera schema for OHLCV data.

    Validates:
    - Schema: Required columns with correct types
    - Nulls: Zero NaNs allowed in OHLCV columns
    - OHLC: High >= max(O,C), Low <= min(O,C), Close in [Low, High]
    - Ranges: All prices > 0, volume >= 0
    - Timestamps: Unique, monotonically increasing
    """

    timestamp: pa.Timestamp = pa.Field(
        nullable=False,
        unique=True,  # No duplicate timestamps
        coerce=True,  # Auto-convert to datetime
    )

    symbol: pa.String = pa.Field(
        nullable=False,
        str_length={"min_value": 1, "max_value": 10},
    )

    open: pa.Float = pa.Field(
        nullable=False,
        gt=0,  # Must be positive
        lt=1_000_000,  # Sanity: no stock > $1M
    )

    high: pa.Float = pa.Field(
        nullable=False,
        gt=0,
        lt=1_000_000,
    )

    low: pa.Float = pa.Field(
        nullable=False,
        gt=0,
        lt=1_000_000,
    )

    close: pa.Float = pa.Field(
        nullable=False,
        gt=0,
        lt=1_000_000,
    )

    volume: pa.Float = pa.Field(
        nullable=False,
        ge=0,  # Volume can be 0 (halts, pre-market)
    )

    class Config:
        strict = True  # No extra columns allowed
        coerce = True  # Auto-convert dtypes

    # DataFrame-level checks (relationships between columns)
    @pa.dataframe_check
    def high_gte_open_and_close(cls, df: pd.DataFrame) -> pd.Series:
        """High must be >= both Open and Close"""
        return (df["high"] >= df["open"]) & (df["high"] >= df["close"])

    @pa.dataframe_check
    def low_lte_open_and_close(cls, df: pd.DataFrame) -> pd.Series:
        """Low must be <= both Open and Close"""
        return (df["low"] <= df["open"]) & (df["low"] <= df["close"])

    @pa.dataframe_check
    def close_in_range(cls, df: pd.DataFrame) -> pd.Series:
        """Close must be in [Low, High]"""
        return (df["close"] >= df["low"]) & (df["close"] <= df["high"])

    @pa.dataframe_check
    def high_gte_low(cls, df: pd.DataFrame) -> pd.Series:
        """High must be >= Low (physically impossible otherwise)"""
        return df["high"] >= df["low"]

    @pa.dataframe_check
    def timestamp_monotonic(cls, df: pd.DataFrame) -> bool:
        """Timestamps must be monotonically increasing"""
        if len(df) < 2:
            return True
        return df["timestamp"].is_monotonic_increasing


class OHLCVStatsSchema(OHLCVSchema):
    """
    Extended schema with statistical checks.

    Use for research validation, not production (too strict).
    Adds checks for:
    - Daily returns within circuit breaker bounds [-50%, +50%]
    - Volume stability (CV < 300%)
    - Price continuity (no exact duplicates for 5+ days)
    """

    @pa.dataframe_check
    def returns_in_bounds(cls, df: pd.DataFrame) -> pd.Series:
        """Daily returns must be within [-50%, +50%] (circuit breaker)"""
        if len(df) < 2:
            return pd.Series([True])
        returns = df["close"].pct_change().dropna()
        return returns.abs() <= 0.50

    @pa.dataframe_check
    def volume_stability(cls, df: pd.DataFrame) -> bool:
        """Coefficient of variation < 300% (sanity check)"""
        if len(df) < 10:
            return True  # Not enough data
        cv = df["volume"].std() / df["volume"].mean()
        return cv < 3.0

    @pa.dataframe_check
    def no_extended_duplicates(cls, df: pd.DataFrame) -> bool:
        """No exact duplicate prices for 5+ consecutive days"""
        if len(df) < 5:
            return True
        # Check if any 5-day window has all identical closes
        for i in range(len(df) - 4):
            window = df["close"].iloc[i:i+5]
            if window.nunique() == 1:
                return False
        return True


# Pre-built schemas for common use cases
ohlcv_schema = OHLCVSchema.to_schema()
ohlcv_stats_schema = OHLCVStatsSchema.to_schema()


# Convenience function for quick validation
def validate_ohlcv_pandera(
    df: pd.DataFrame,
    lazy: bool = True,
    stats: bool = False,
) -> pd.DataFrame:
    """
    Validate OHLCV DataFrame with Pandera.

    Args:
        df: DataFrame with OHLCV data
        lazy: If True, collect all errors before raising (better UX)
        stats: If True, use extended statistical checks

    Returns:
        Validated DataFrame (with type coercion applied)

    Raises:
        pa.errors.SchemaError: If validation fails
    """
    schema = ohlcv_stats_schema if stats else ohlcv_schema
    return schema.validate(df, lazy=lazy)
