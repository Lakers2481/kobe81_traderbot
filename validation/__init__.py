"""
Validation Module - Data Quality and Fake Data Detection

This module provides validation tools for the trading system:
- Fake data detection (hardcoded placeholders)
- Data quality validation
- Signal integrity checks
"""

from .fake_data_detector import (
    detect_fake_data,
    FakeDataAlert,
    validate_signals_before_trading,
)

__all__ = [
    'detect_fake_data',
    'FakeDataAlert',
    'validate_signals_before_trading',
]
