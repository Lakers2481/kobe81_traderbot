"""
Data Quality Module.

FIX (2026-01-05): Added for data quality monitoring and canary checks.

Provides:
- Corporate actions canary (split/dividend detection)
- Price discontinuity detection
"""
from __future__ import annotations

from data.quality.corporate_actions_canary import (
    DiscontinuityEvent,
    CanaryResult,
    check_symbol_for_splits,
    run_price_discontinuity_check,
    check_recent_data,
    get_canary_summary,
)

__all__ = [
    "DiscontinuityEvent",
    "CanaryResult",
    "check_symbol_for_splits",
    "run_price_discontinuity_check",
    "check_recent_data",
    "get_canary_summary",
]
