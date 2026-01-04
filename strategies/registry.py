"""
CANONICAL STRATEGY REGISTRY - SINGLE SOURCE OF TRUTH
=====================================================

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!                    CRITICAL - READ THIS FIRST                           !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

This registry is the ONLY place to get production strategies.
DO NOT import strategies directly from their individual modules.

CORRECT:
    from strategies.registry import get_production_scanner, validate_strategy_import

WRONG:
    from strategies.ict.turtle_soup import TurtleSoupStrategy  # DEPRECATED
    from strategies.ibs_rsi.strategy import IbsRsiStrategy      # DEPRECATED

WHY THIS MATTERS:
- DualStrategyScanner has ts_min_sweep_strength=0.3 filter = 61% WR
- TurtleSoupStrategy (standalone) has NO filter = 48% WR
- Using the wrong strategy costs you 13% win rate!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Optional

# Canonical import - the ONLY strategy to use
from strategies.dual_strategy.combined import DualStrategyScanner, DualStrategyParams


# =============================================================================
# CANONICAL STRATEGY ACCESS
# =============================================================================

def get_production_scanner(params: Optional[DualStrategyParams] = None) -> DualStrategyScanner:
    """
    Get the production-ready strategy scanner.

    This is the ONLY way to get a strategy for production use.

    Returns:
        DualStrategyScanner with verified v2.2 parameters
        - IBS+RSI: 59.9% WR, 1.46 PF, 867 trades
        - Turtle Soup: 61.0% WR, 1.37 PF, 305 trades
        - Combined: 60.2% WR, 1.44 PF, 1172 trades

    Example:
        from strategies.registry import get_production_scanner
        scanner = get_production_scanner()
        signals = scanner.scan_signals_over_time(df)
    """
    if params is None:
        params = DualStrategyParams()

    # Validate critical parameters
    if params.ts_min_sweep_strength < 0.3:
        warnings.warn(
            f"WARNING: ts_min_sweep_strength={params.ts_min_sweep_strength} is below 0.3! "
            f"This will DEGRADE Turtle Soup win rate from 61% to ~48%. "
            f"Use default DualStrategyParams() for verified performance.",
            UserWarning,
            stacklevel=2
        )

    return DualStrategyScanner(params)


def get_default_params() -> DualStrategyParams:
    """
    Get verified v2.2 default parameters.

    These parameters produced:
    - IBS+RSI: 59.9% WR, 1.46 PF
    - Turtle Soup: 61.0% WR, 1.37 PF
    - Combined: 60.2% WR, 1.44 PF

    DO NOT MODIFY without re-running full backtest verification.
    """
    return DualStrategyParams()


# =============================================================================
# DEPRECATION GUARDS
# =============================================================================

_DEPRECATED_IMPORTS = {
    'strategies.ict.turtle_soup': 'TurtleSoupStrategy',
    'strategies.ibs_rsi.strategy': 'IbsRsiStrategy',
}

_WARNING_ISSUED = set()


def validate_strategy_import():
    """
    Call this at startup to check for deprecated strategy imports.

    This scans sys.modules for any deprecated standalone strategy imports
    and issues warnings. It SKIPS warnings if DualStrategyScanner is loaded
    (since it legitimately uses the sub-strategies internally).

    Usage:
        from strategies.registry import validate_strategy_import
        validate_strategy_import()  # Call at startup
    """
    # If DualStrategyScanner is loaded, the sub-strategies are legitimate internal imports
    # Don't warn in this case - the user is using the correct registry pattern
    if 'strategies.dual_strategy.combined' in sys.modules:
        return  # Legitimate use via DualStrategyScanner

    for module_name, class_name in _DEPRECATED_IMPORTS.items():
        if module_name in sys.modules and module_name not in _WARNING_ISSUED:
            _WARNING_ISSUED.add(module_name)
            warnings.warn(
                f"\n"
                f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                f"!!! DEPRECATED STRATEGY IMPORT DETECTED !!!\n"
                f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                f"\n"
                f"You imported: {module_name}.{class_name}\n"
                f"\n"
                f"This is WRONG for production use!\n"
                f"- Standalone {class_name} produces ~48-59% win rate\n"
                f"- DualStrategyScanner produces 60-61% win rate\n"
                f"\n"
                f"CORRECT USAGE:\n"
                f"    from strategies.registry import get_production_scanner\n"
                f"    scanner = get_production_scanner()\n"
                f"\n"
                f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
                DeprecationWarning,
                stacklevel=2
            )


def assert_no_deprecated_strategies():
    """
    STRICT validation - raises error if deprecated strategies are imported.

    Use this in production scripts to BLOCK execution if wrong strategies loaded.

    Usage:
        from strategies.registry import assert_no_deprecated_strategies
        assert_no_deprecated_strategies()  # Raises RuntimeError if bad imports
    """
    bad_imports = []
    for module_name, class_name in _DEPRECATED_IMPORTS.items():
        if module_name in sys.modules:
            bad_imports.append(f"{module_name}.{class_name}")

    if bad_imports:
        raise RuntimeError(
            "\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "!!! FATAL: DEPRECATED STRATEGY IMPORTS DETECTED !!!\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "\n"
            "Found deprecated imports:\n"
            + "\n".join(f"  - {imp}" for imp in bad_imports) +
            "\n\n"
            "These strategies do NOT have critical filters and will\n"
            "produce inferior results (48% WR instead of 61% WR).\n"
            "\n"
            "REQUIRED: Use strategies.registry.get_production_scanner()\n"
            "\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        )


# =============================================================================
# STRATEGY INFO
# =============================================================================

VERIFIED_PERFORMANCE = {
    "version": "v2.2",
    "frozen_date": "2025-12-30",
    "ibs_rsi": {
        "trades": 867,
        "win_rate": 0.599,
        "profit_factor": 1.46,
    },
    "turtle_soup": {
        "trades": 305,
        "win_rate": 0.610,
        "profit_factor": 1.37,
    },
    "combined": {
        "trades": 1172,
        "win_rate": 0.602,
        "profit_factor": 1.44,
    },
}


def print_strategy_info():
    """Print verified strategy performance information."""
    print("\n" + "="*60)
    print("KOBE81 VERIFIED STRATEGY PERFORMANCE (v2.2)")
    print("="*60)
    print("\nIBS+RSI Mean Reversion:")
    print(f"  - Trades: {VERIFIED_PERFORMANCE['ibs_rsi']['trades']}")
    print(f"  - Win Rate: {VERIFIED_PERFORMANCE['ibs_rsi']['win_rate']:.1%}")
    print(f"  - Profit Factor: {VERIFIED_PERFORMANCE['ibs_rsi']['profit_factor']:.2f}")
    print("\nICT Turtle Soup:")
    print(f"  - Trades: {VERIFIED_PERFORMANCE['turtle_soup']['trades']}")
    print(f"  - Win Rate: {VERIFIED_PERFORMANCE['turtle_soup']['win_rate']:.1%}")
    print(f"  - Profit Factor: {VERIFIED_PERFORMANCE['turtle_soup']['profit_factor']:.2f}")
    print("\nCombined:")
    print(f"  - Trades: {VERIFIED_PERFORMANCE['combined']['trades']}")
    print(f"  - Win Rate: {VERIFIED_PERFORMANCE['combined']['win_rate']:.1%}")
    print(f"  - Profit Factor: {VERIFIED_PERFORMANCE['combined']['profit_factor']:.2f}")
    print("\n" + "="*60)
    print("USE: from strategies.registry import get_production_scanner")
    print("="*60 + "\n")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'get_production_scanner',
    'get_default_params',
    'validate_strategy_import',
    'assert_no_deprecated_strategies',
    'print_strategy_info',
    'DualStrategyScanner',
    'DualStrategyParams',
    'VERIFIED_PERFORMANCE',
]
