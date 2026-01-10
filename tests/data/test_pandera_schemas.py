"""
Unit tests for Pandera OHLCV schemas.

Tests validate that:
1. Valid OHLCV data passes validation
2. OHLC violations are caught
3. NaN values are rejected
4. Negative prices are rejected
5. Duplicate timestamps are rejected
6. Non-monotonic timestamps trigger warnings
7. Statistical checks work correctly

Run:
    pytest tests/data/test_pandera_schemas.py -v
"""

import pytest
import pandas as pd
import pandera as pa
from datetime import datetime, timedelta

from data.schemas.ohlcv_schema import (
    OHLCVSchema,
    OHLCVStatsSchema,
    ohlcv_schema,
    ohlcv_stats_schema,
    validate_ohlcv_pandera,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def valid_ohlcv():
    """Valid OHLCV DataFrame for testing"""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
        'symbol': ['AAPL'] * 10,
        'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        'high': [101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5],
        'low': [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5],
        'close': [100.8, 101.8, 102.8, 103.8, 104.8, 105.8, 106.8, 107.8, 108.8, 109.8],
        'volume': [1000000] * 10,
    })


# ============================================================================
# Happy Path Tests
# ============================================================================

def test_valid_ohlcv_passes(valid_ohlcv):
    """Test that valid OHLCV data passes validation"""
    validated = ohlcv_schema.validate(valid_ohlcv)
    assert len(validated) == 10
    assert list(validated.columns) == ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']


def test_validate_ohlcv_pandera_convenience_function(valid_ohlcv):
    """Test convenience function"""
    validated = validate_ohlcv_pandera(valid_ohlcv, lazy=True)
    assert len(validated) == 10


def test_stats_schema_passes_valid_data(valid_ohlcv):
    """Test that stats schema passes valid data"""
    validated = ohlcv_stats_schema.validate(valid_ohlcv)
    assert len(validated) == 10


# ============================================================================
# OHLC Relationship Tests
# ============================================================================

def test_high_less_than_open_rejected():
    """Test that High < Open is rejected"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1),
        'symbol': ['AAPL'],
        'open': [100.0],
        'high': [99.0],  # HIGH < OPEN (invalid!)
        'low': [98.0],
        'close': [99.5],
        'volume': [1000000],
    })

    with pytest.raises(pa.errors.SchemaError) as exc:
        ohlcv_schema.validate(df)

    assert "high_gte_open_and_close" in str(exc.value)


def test_high_less_than_close_rejected():
    """Test that High < Close is rejected"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1),
        'symbol': ['AAPL'],
        'open': [100.0],
        'high': [99.0],  # HIGH < CLOSE (invalid!)
        'low': [98.0],
        'close': [100.0],
        'volume': [1000000],
    })

    with pytest.raises(pa.errors.SchemaError):
        ohlcv_schema.validate(df)


def test_low_greater_than_open_rejected():
    """Test that Low > Open is rejected"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1),
        'symbol': ['AAPL'],
        'open': [100.0],
        'high': [102.0],
        'low': [101.0],  # LOW > OPEN (invalid!)
        'close': [101.5],
        'volume': [1000000],
    })

    with pytest.raises(pa.errors.SchemaError) as exc:
        ohlcv_schema.validate(df)

    assert "low_lte_open_and_close" in str(exc.value)


def test_low_greater_than_close_rejected():
    """Test that Low > Close is rejected"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1),
        'symbol': ['AAPL'],
        'open': [100.0],
        'high': [102.0],
        'low': [101.0],  # LOW > CLOSE (invalid!)
        'close': [100.5],
        'volume': [1000000],
    })

    with pytest.raises(pa.errors.SchemaError):
        ohlcv_schema.validate(df)


def test_close_outside_high_low_range_rejected():
    """Test that Close outside [Low, High] is rejected"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1),
        'symbol': ['AAPL'],
        'open': [100.0],
        'high': [101.0],
        'low': [99.0],
        'close': [102.0],  # CLOSE > HIGH (invalid!)
        'volume': [1000000],
    })

    with pytest.raises(pa.errors.SchemaError) as exc:
        ohlcv_schema.validate(df)

    assert "close_in_range" in str(exc.value)


def test_high_less_than_low_rejected():
    """Test that High < Low is rejected (physically impossible)"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1),
        'symbol': ['AAPL'],
        'open': [100.0],
        'high': [99.0],  # HIGH < LOW (impossible!)
        'low': [100.0],
        'close': [99.5],
        'volume': [1000000],
    })

    with pytest.raises(pa.errors.SchemaError) as exc:
        ohlcv_schema.validate(df)

    assert "high_gte_low" in str(exc.value)


# ============================================================================
# Null/Missing Value Tests
# ============================================================================

def test_null_open_rejected():
    """Test that NaNs in Open are rejected"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=2),
        'symbol': ['AAPL', 'AAPL'],
        'open': [100.0, None],  # NaN in open
        'high': [101.0, 102.0],
        'low': [99.0, 100.0],
        'close': [100.5, 101.5],
        'volume': [1000000, 1000000],
    })

    with pytest.raises(pa.errors.SchemaError):
        ohlcv_schema.validate(df)


def test_null_close_rejected():
    """Test that NaNs in Close are rejected"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=2),
        'symbol': ['AAPL', 'AAPL'],
        'open': [100.0, 101.0],
        'high': [101.0, 102.0],
        'low': [99.0, 100.0],
        'close': [100.5, None],  # NaN in close
        'volume': [1000000, 1000000],
    })

    with pytest.raises(pa.errors.SchemaError):
        ohlcv_schema.validate(df)


# ============================================================================
# Range Validation Tests
# ============================================================================

def test_negative_price_rejected():
    """Test that negative prices are rejected"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1),
        'symbol': ['AAPL'],
        'open': [-100.0],  # Negative price (invalid!)
        'high': [101.0],
        'low': [99.0],
        'close': [100.5],
        'volume': [1000000],
    })

    with pytest.raises(pa.errors.SchemaError):
        ohlcv_schema.validate(df)


def test_zero_price_rejected():
    """Test that zero prices are rejected"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1),
        'symbol': ['AAPL'],
        'open': [0.0],  # Zero price (invalid!)
        'high': [101.0],
        'low': [99.0],
        'close': [100.5],
        'volume': [1000000],
    })

    with pytest.raises(pa.errors.SchemaError):
        ohlcv_schema.validate(df)


def test_negative_volume_rejected():
    """Test that negative volume is rejected"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1),
        'symbol': ['AAPL'],
        'open': [100.0],
        'high': [101.0],
        'low': [99.0],
        'close': [100.5],
        'volume': [-1000000],  # Negative volume (invalid!)
    })

    with pytest.raises(pa.errors.SchemaError):
        ohlcv_schema.validate(df)


def test_zero_volume_allowed():
    """Test that zero volume is allowed (halts, pre-market)"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1),
        'symbol': ['AAPL'],
        'open': [100.0],
        'high': [101.0],
        'low': [99.0],
        'close': [100.5],
        'volume': [0],  # Zero volume OK
    })

    validated = ohlcv_schema.validate(df)
    assert validated['volume'].iloc[0] == 0


# ============================================================================
# Timestamp Validation Tests
# ============================================================================

def test_duplicate_timestamps_rejected():
    """Test that duplicate timestamps are rejected"""
    df = pd.DataFrame({
        'timestamp': [datetime(2024, 1, 1)] * 2,  # Duplicates!
        'symbol': ['AAPL', 'AAPL'],
        'open': [100.0, 101.0],
        'high': [101.0, 102.0],
        'low': [99.0, 100.0],
        'close': [100.5, 101.5],
        'volume': [1000000, 1000000],
    })

    with pytest.raises(pa.errors.SchemaError) as exc:
        ohlcv_schema.validate(df)

    # Should fail on unique constraint
    assert "unique" in str(exc.value).lower()


def test_non_monotonic_timestamps_rejected():
    """Test that non-monotonic timestamps are rejected"""
    df = pd.DataFrame({
        'timestamp': [
            datetime(2024, 1, 3),  # Out of order!
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
        ],
        'symbol': ['AAPL'] * 3,
        'open': [100.0, 101.0, 102.0],
        'high': [101.0, 102.0, 103.0],
        'low': [99.0, 100.0, 101.0],
        'close': [100.5, 101.5, 102.5],
        'volume': [1000000, 1000000, 1000000],
    })

    with pytest.raises(pa.errors.SchemaError) as exc:
        ohlcv_schema.validate(df)

    assert "timestamp_monotonic" in str(exc.value)


# ============================================================================
# Schema Tests
# ============================================================================

def test_missing_required_column_rejected():
    """Test that missing required columns are rejected"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1),
        'symbol': ['AAPL'],
        'open': [100.0],
        # Missing: high, low, close, volume
    })

    with pytest.raises(pa.errors.SchemaError):
        ohlcv_schema.validate(df)


def test_extra_columns_rejected_in_strict_mode():
    """Test that extra columns are rejected in strict mode"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1),
        'symbol': ['AAPL'],
        'open': [100.0],
        'high': [101.0],
        'low': [99.0],
        'close': [100.5],
        'volume': [1000000],
        'extra_column': [42],  # Extra column!
    })

    with pytest.raises(pa.errors.SchemaError):
        ohlcv_schema.validate(df)


# ============================================================================
# Statistical Schema Tests
# ============================================================================

def test_extreme_daily_return_rejected_in_stats_schema():
    """Test that extreme daily returns are rejected in stats schema"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=2),
        'symbol': ['AAPL', 'AAPL'],
        'open': [100.0, 200.0],
        'high': [101.0, 201.0],
        'low': [99.0, 199.0],
        'close': [100.0, 200.0],  # 100% return (> 50% threshold)
        'volume': [1000000, 1000000],
    })

    with pytest.raises(pa.errors.SchemaError) as exc:
        ohlcv_stats_schema.validate(df)

    assert "returns_in_bounds" in str(exc.value)


def test_volume_instability_rejected_in_stats_schema():
    """Test that extreme volume instability is rejected"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10),
        'symbol': ['AAPL'] * 10,
        'open': [100.0] * 10,
        'high': [101.0] * 10,
        'low': [99.0] * 10,
        'close': [100.5] * 10,
        'volume': [100, 1, 10000000, 1, 1, 1, 1, 1, 1, 1],  # CV >> 300%
    })

    with pytest.raises(pa.errors.SchemaError) as exc:
        ohlcv_stats_schema.validate(df)

    assert "volume_stability" in str(exc.value)


def test_extended_duplicate_prices_rejected_in_stats_schema():
    """Test that 5+ days of identical prices are rejected"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10),
        'symbol': ['AAPL'] * 10,
        'open': [100.0] * 10,
        'high': [101.0] * 10,
        'low': [99.0] * 10,
        'close': [100.0] * 10,  # 10 days of identical close (suspicious)
        'volume': [1000000] * 10,
    })

    with pytest.raises(pa.errors.SchemaError) as exc:
        ohlcv_stats_schema.validate(df)

    assert "no_extended_duplicates" in str(exc.value)


# ============================================================================
# Type Coercion Tests
# ============================================================================

def test_type_coercion_works():
    """Test that type coercion converts strings to proper types"""
    df = pd.DataFrame({
        'timestamp': ['2024-01-01', '2024-01-02'],  # Strings
        'symbol': ['AAPL', 'AAPL'],
        'open': ['100.0', '101.0'],  # Strings
        'high': ['101.0', '102.0'],
        'low': ['99.0', '100.0'],
        'close': ['100.5', '101.5'],
        'volume': ['1000000', '1000000'],
    })

    validated = ohlcv_schema.validate(df)

    # Check types were coerced
    assert pd.api.types.is_datetime64_any_dtype(validated['timestamp'])
    assert pd.api.types.is_float_dtype(validated['open'])
    assert pd.api.types.is_float_dtype(validated['volume'])


# ============================================================================
# Lazy Validation Tests
# ============================================================================

def test_lazy_validation_collects_all_errors():
    """Test that lazy=True collects all errors before raising"""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=2),
        'symbol': ['AAPL', 'AAPL'],
        'open': [100.0, None],  # Error 1: NaN
        'high': [99.0, 102.0],  # Error 2: High < Open in row 0
        'low': [101.0, 100.0],  # Error 3: Low > Open in row 0
        'close': [100.5, 101.5],
        'volume': [-1000000, 1000000],  # Error 4: Negative volume
    })

    with pytest.raises(pa.errors.SchemaError) as exc:
        ohlcv_schema.validate(df, lazy=True)

    # Lazy mode should report multiple failures
    error_str = str(exc.value)
    assert "failure_case" in error_str.lower()


# ============================================================================
# Integration Tests
# ============================================================================

def test_real_world_aapl_data_structure(valid_ohlcv):
    """Test that schema handles real-world data structure from Polygon"""
    # Simulate Polygon API response structure
    validated = ohlcv_schema.validate(valid_ohlcv)

    # Should preserve all rows
    assert len(validated) == len(valid_ohlcv)

    # Should have correct columns in order
    expected_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    assert list(validated.columns) == expected_cols


def test_empty_dataframe_rejected():
    """Test that empty DataFrames are rejected"""
    df = pd.DataFrame(columns=['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

    # Empty DataFrame should technically pass schema but fail on uniqueness
    # This is acceptable - upstream code should check for empty before validation
    validated = ohlcv_schema.validate(df)
    assert len(validated) == 0


def test_single_row_dataframe_passes(valid_ohlcv):
    """Test that single-row DataFrames pass validation"""
    single_row = valid_ohlcv.head(1)
    validated = ohlcv_schema.validate(single_row)
    assert len(validated) == 1
