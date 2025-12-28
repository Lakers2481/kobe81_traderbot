"""
Position Limit Gate for Kobe Trading System.

Enforces maximum number of concurrent open positions to prevent
over-exposure and concentration risk.

Usage:
    from risk.position_limit_gate import PositionLimitGate, get_position_limit_gate

    gate = get_position_limit_gate()
    allowed, reason = gate.check("AAPL")
    if not allowed:
        print(f"Blocked: {reason}")
"""
from __future__ import annotations

import logging
import os
import requests
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PositionLimits:
    """Configuration for position limits."""
    max_positions: int = 5  # Maximum concurrent open positions
    max_per_symbol: int = 1  # Maximum positions per symbol (prevent doubling down)
    max_sector_concentration: float = 0.40  # Max 40% in any single sector


class PositionLimitGate:
    """
    Gate that enforces position limits before order placement.

    Checks:
    1. Total number of open positions vs max_positions
    2. Whether we already have a position in the symbol
    3. (Optional) Sector concentration limits
    """

    def __init__(self, limits: Optional[PositionLimits] = None):
        self.limits = limits or PositionLimits()
        self._cached_positions: Optional[List[Dict]] = None
        self._cache_timestamp: float = 0
        self._cache_ttl_seconds: float = 5.0  # Cache positions for 5 seconds

    def _get_alpaca_config(self) -> Tuple[str, str, str]:
        """Get Alpaca API configuration from environment."""
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
        key_id = os.getenv("ALPACA_API_KEY_ID", "") or os.getenv("APCA_API_KEY_ID", "")
        secret = os.getenv("ALPACA_API_SECRET_KEY", "") or os.getenv("APCA_API_SECRET_KEY", "")
        return base_url, key_id, secret

    def _fetch_positions(self, timeout: int = 5) -> List[Dict]:
        """Fetch current open positions from Alpaca."""
        import time

        # Check cache
        now = time.time()
        if self._cached_positions is not None and (now - self._cache_timestamp) < self._cache_ttl_seconds:
            return self._cached_positions

        base_url, key_id, secret = self._get_alpaca_config()

        if not key_id or not secret:
            logger.warning("Alpaca credentials not configured, cannot check positions")
            return []

        headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret,
        }

        try:
            url = f"{base_url}/v2/positions"
            response = requests.get(url, headers=headers, timeout=timeout)

            if response.status_code == 200:
                positions = response.json()
                # Update cache
                self._cached_positions = positions
                self._cache_timestamp = now
                return positions
            else:
                logger.warning(f"Failed to fetch positions: HTTP {response.status_code}")
                return self._cached_positions or []

        except Exception as e:
            logger.warning(f"Error fetching positions: {e}")
            return self._cached_positions or []

    def get_open_position_count(self) -> int:
        """Get the number of currently open positions."""
        positions = self._fetch_positions()
        return len(positions)

    def get_open_symbols(self) -> List[str]:
        """Get list of symbols with open positions."""
        positions = self._fetch_positions()
        return [p.get("symbol", "").upper() for p in positions]

    def has_position_in_symbol(self, symbol: str) -> bool:
        """Check if we already have a position in this symbol."""
        open_symbols = self.get_open_symbols()
        return symbol.upper() in open_symbols

    def check(self, symbol: str, side: str = "long") -> Tuple[bool, str]:
        """
        Check if a new position is allowed.

        Args:
            symbol: Stock symbol to check
            side: "long" or "short" (for future use)

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        symbol = symbol.upper()

        # Check if we already have a position in this symbol
        if self.has_position_in_symbol(symbol):
            if self.limits.max_per_symbol <= 1:
                return False, f"already_have_position:{symbol}"

        # Check total position count
        current_count = self.get_open_position_count()
        if current_count >= self.limits.max_positions:
            return False, f"max_positions_reached:{current_count}/{self.limits.max_positions}"

        return True, "ok"

    def get_status(self) -> Dict[str, Any]:
        """Get current position limit status."""
        positions = self._fetch_positions()
        symbols = [p.get("symbol", "") for p in positions]

        return {
            "open_positions": len(positions),
            "max_positions": self.limits.max_positions,
            "positions_available": max(0, self.limits.max_positions - len(positions)),
            "open_symbols": symbols,
            "max_per_symbol": self.limits.max_per_symbol,
        }

    def clear_cache(self) -> None:
        """Clear the position cache to force a fresh fetch."""
        self._cached_positions = None
        self._cache_timestamp = 0


# Singleton instance
_position_limit_gate: Optional[PositionLimitGate] = None


def get_position_limit_gate(limits: Optional[PositionLimits] = None) -> PositionLimitGate:
    """Get the singleton PositionLimitGate instance."""
    global _position_limit_gate
    if _position_limit_gate is None:
        _position_limit_gate = PositionLimitGate(limits)
    return _position_limit_gate


def reset_position_limit_gate() -> None:
    """Reset the singleton instance (for testing)."""
    global _position_limit_gate
    _position_limit_gate = None
