"""
Execution module for Kobe Trading System.
Provides broker integration and order execution via Alpaca.

Key functions:
- execute_signal(): High-level signal execution with all safety checks
- place_order_with_liquidity_check(): Order placement with liquidity validation
- place_ioc_limit(): Direct IOC LIMIT order (no liquidity check)
"""
from __future__ import annotations

from .broker_alpaca import (
    AlpacaConfig,
    OrderResult,
    get_best_ask,
    get_best_bid,
    get_quote_with_sizes,
    get_avg_volume,
    place_ioc_limit,
    place_order_with_liquidity_check,
    execute_signal,
    construct_decision,
    check_liquidity_for_order,
    get_liquidity_gate,
    set_liquidity_gate,
    enable_liquidity_gate,
    is_liquidity_gate_enabled,
)

__all__ = [
    # Config
    "AlpacaConfig",
    "OrderResult",
    # Quote functions
    "get_best_ask",
    "get_best_bid",
    "get_quote_with_sizes",
    "get_avg_volume",
    # Order placement
    "place_ioc_limit",
    "place_order_with_liquidity_check",
    "execute_signal",
    "construct_decision",
    # Liquidity gate
    "check_liquidity_for_order",
    "get_liquidity_gate",
    "set_liquidity_gate",
    "enable_liquidity_gate",
    "is_liquidity_gate_enabled",
]
