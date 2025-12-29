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


_MOCK_SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "JPM": "Financials", "GS": "Financials",
    "XOM": "Energy", "CVX": "Energy",
    "UNH": "Healthcare", "JNJ": "Healthcare",
    "SPG": "Real Estate",
    "WMT": "Consumer Staples",
    # Add more as needed for testing or demo purposes
}

def _get_symbol_sector(symbol: str) -> Optional[str]:
    """
    Placeholder to get the sector for a given symbol.
    In a real system, this would query a data source or an internal mapping.
    """
    return _MOCK_SECTOR_MAP.get(symbol.upper(), None)


class PositionLimitGate:
    """
    Gate that enforces position limits before order placement.

    Checks:
    1. Total number of open positions vs max_positions
    2. Whether we already have a position in the symbol
    3. Sector concentration limits
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
                # Add estimated market_value and sector to each position for checks
                for p in positions:
                    qty = int(p.get("qty", 0))
                    current_price = float(p.get("current_price", 0.0))
                    p["market_value"] = abs(qty * current_price)
                    p["sector"] = _get_symbol_sector(p.get("symbol", ""))

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

    def check(self, symbol: str, side: str, price: float, qty: int) -> Tuple[bool, str]:
        """
        Check if a new position is allowed.

        Args:
            symbol: Stock symbol to check
            side: "long" or "short" (for future use)
            price: Current price of the symbol for notional calculation
            qty: Quantity of shares for the proposed trade

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        symbol = symbol.upper()

        # 1. Check if we already have a position in this symbol
        if self.has_position_in_symbol(symbol):
            if self.limits.max_per_symbol <= 1:
                return False, f"already_have_position:{symbol}"

        # 2. Check total position count
        current_count = self.get_open_position_count()
        if current_count >= self.limits.max_positions:
            return False, f"max_positions_reached:{current_count}/{self.limits.max_positions}"

        # 3. Check sector concentration
        if self.limits.max_sector_concentration > 0:
            proposed_sector = _get_symbol_sector(symbol)
            if proposed_sector:
                open_positions = self._fetch_positions()
                total_portfolio_notional = sum(p["market_value"] for p in open_positions) + (price * qty) # Including proposed trade
                
                sector_notionals: Dict[str, float] = {}
                for p in open_positions:
                    sector = p.get("sector")
                    if sector:
                        sector_notionals[sector] = sector_notionals.get(sector, 0.0) + p["market_value"]
                
                sector_notionals[proposed_sector] = sector_notionals.get(proposed_sector, 0.0) + (price * qty)

                if total_portfolio_notional > 0:
                    for sector, notional in sector_notionals.items():
                        concentration = notional / total_portfolio_notional
                        if concentration > self.limits.max_sector_concentration:
                            return False, f"exceeds_sector_concentration:{sector}:{concentration:.2f}"
            else:
                logger.warning(f"Could not determine sector for {symbol}. Sector concentration check skipped.")

        return True, "ok"

    def get_status(self) -> Dict[str, Any]:
        """Get current position limit status."""
        positions = self._fetch_positions()
        symbols = [p.get("symbol", "") for p in positions]
        
        # Calculate sector concentration for status
        total_portfolio_notional = sum(p["market_value"] for p in positions)
        sector_notionals: Dict[str, float] = {}
        for p in positions:
            sector = p.get("sector")
            if sector:
                sector_notionals[sector] = sector_notionals.get(sector, 0.0) + p["market_value"]
        
        sector_concentrations: Dict[str, float] = {
            sector: (notional / total_portfolio_notional if total_portfolio_notional > 0 else 0.0)
            for sector, notional in sector_notionals.items()
        }

        return {
            "open_positions": len(positions),
            "max_positions": self.limits.max_positions,
            "positions_available": max(0, self.limits.max_positions - len(positions)),
            "open_symbols": symbols,
            "max_per_symbol": self.limits.max_per_symbol,
            "sector_concentrations": {s: f"{c:.2%}" for s, c in sector_concentrations.items()},
            "max_sector_concentration": f"{self.limits.max_sector_concentration:.2%}",
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
