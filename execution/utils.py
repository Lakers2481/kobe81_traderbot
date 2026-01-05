"""
Execution Utilities
===================

Common utilities for the execution layer.

Contains:
- Side normalization (buy/sell/long/short -> BUY/SELL)
- Order type utilities
"""

from __future__ import annotations

from typing import Literal


def normalize_side(value: str) -> Literal["BUY", "SELL"]:
    """
    Normalize trade side to Alpaca-compatible format.

    Handles variations:
    - "long", "Long", "LONG", " long " -> "BUY"
    - "buy", "Buy", "BUY", " buy " -> "BUY"
    - "short", "Short", "SHORT", " short " -> "SELL"
    - "sell", "Sell", "SELL", " sell " -> "SELL"

    Args:
        value: Trade side string in any format

    Returns:
        "BUY" or "SELL" (uppercase, Alpaca-compatible)

    Raises:
        ValueError: If value is not a valid side

    Examples:
        >>> normalize_side("long")
        'BUY'
        >>> normalize_side("SHORT")
        'SELL'
        >>> normalize_side(" Buy ")
        'BUY'
    """
    if not isinstance(value, str):
        raise ValueError(f"Side must be a string, got {type(value).__name__}")

    v = value.strip().upper()

    if v in ("BUY", "LONG"):
        return "BUY"
    elif v in ("SELL", "SHORT"):
        return "SELL"
    else:
        raise ValueError(f"Invalid side: {value!r}. Expected: buy, sell, long, or short")


def normalize_side_lowercase(value: str) -> Literal["buy", "sell"]:
    """
    Normalize trade side to lowercase format (for Alpaca API payload).

    Args:
        value: Trade side string in any format

    Returns:
        "buy" or "sell" (lowercase, for API payloads)

    Examples:
        >>> normalize_side_lowercase("LONG")
        'buy'
        >>> normalize_side_lowercase("short")
        'sell'
    """
    return normalize_side(value).lower()  # type: ignore


def is_buy_side(value: str) -> bool:
    """
    Check if the side represents a buy/long position.

    Args:
        value: Trade side string

    Returns:
        True if buy/long, False if sell/short

    Examples:
        >>> is_buy_side("long")
        True
        >>> is_buy_side("SHORT")
        False
    """
    return normalize_side(value) == "BUY"


def is_sell_side(value: str) -> bool:
    """
    Check if the side represents a sell/short position.

    Args:
        value: Trade side string

    Returns:
        True if sell/short, False if buy/long

    Examples:
        >>> is_sell_side("short")
        True
        >>> is_sell_side("BUY")
        False
    """
    return normalize_side(value) == "SELL"
