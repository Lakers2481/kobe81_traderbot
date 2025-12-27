"""
Prohibited Symbols List
========================

Manages list of symbols that cannot be traded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ProhibitionReason(Enum):
    """Reason for prohibition."""
    EARNINGS = "earnings"           # Around earnings
    NEWS = "news"                   # Major news event
    VOLATILITY = "volatility"       # Extreme volatility
    LIQUIDITY = "liquidity"         # Insufficient liquidity
    REGULATORY = "regulatory"       # Regulatory restriction
    RISK = "risk"                   # Risk management
    MANUAL = "manual"               # Manual prohibition
    SECTOR = "sector"               # Sector restriction


@dataclass
class Prohibition:
    """Record of a symbol prohibition."""
    symbol: str
    reason: ProhibitionReason
    added_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    notes: str = ""

    def is_expired(self) -> bool:
        """Check if prohibition has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'reason': self.reason.value,
            'added_at': self.added_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'notes': self.notes,
        }


class ProhibitedList:
    """
    Manages list of prohibited trading symbols.
    """

    def __init__(self, data_file: Optional[Path] = None):
        self.data_file = data_file or Path("data/prohibited_symbols.json")
        self._prohibitions: Dict[str, Prohibition] = {}
        self._load()
        logger.info(f"ProhibitedList initialized with {len(self._prohibitions)} symbols")

    def _load(self):
        """Load from file."""
        if not self.data_file.exists():
            return
        try:
            with open(self.data_file) as f:
                data = json.load(f)
            for item in data.get('prohibitions', []):
                self._prohibitions[item['symbol']] = Prohibition(
                    symbol=item['symbol'],
                    reason=ProhibitionReason(item['reason']),
                    notes=item.get('notes', ''),
                )
        except Exception as e:
            logger.warning(f"Failed to load prohibited list: {e}")

    def _save(self):
        """Save to file."""
        try:
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'prohibitions': [p.to_dict() for p in self._prohibitions.values()],
                'updated_at': datetime.now().isoformat(),
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save prohibited list: {e}")

    def add(
        self,
        symbol: str,
        reason: ProhibitionReason,
        expires_at: Optional[datetime] = None,
        notes: str = "",
    ):
        """Add symbol to prohibited list."""
        prohibition = Prohibition(
            symbol=symbol.upper(),
            reason=reason,
            expires_at=expires_at,
            notes=notes,
        )
        self._prohibitions[symbol.upper()] = prohibition
        self._save()
        logger.info(f"Added {symbol} to prohibited list: {reason.value}")

    def remove(self, symbol: str):
        """Remove symbol from prohibited list."""
        symbol = symbol.upper()
        if symbol in self._prohibitions:
            del self._prohibitions[symbol]
            self._save()
            logger.info(f"Removed {symbol} from prohibited list")

    def is_prohibited(self, symbol: str) -> bool:
        """Check if symbol is prohibited."""
        symbol = symbol.upper()
        if symbol not in self._prohibitions:
            return False

        prohibition = self._prohibitions[symbol]
        if prohibition.is_expired():
            self.remove(symbol)
            return False

        return True

    def check(self, symbol: str) -> Optional[Prohibition]:
        """Check symbol and get prohibition details if prohibited."""
        symbol = symbol.upper()
        if not self.is_prohibited(symbol):
            return None
        return self._prohibitions.get(symbol)

    def list_all(self) -> List[Prohibition]:
        """List all active prohibitions."""
        # Clean expired
        expired = [s for s, p in self._prohibitions.items() if p.is_expired()]
        for s in expired:
            self.remove(s)
        return list(self._prohibitions.values())

    def list_by_reason(self, reason: ProhibitionReason) -> List[Prohibition]:
        """List prohibitions by reason."""
        return [p for p in self.list_all() if p.reason == reason]

    def clear_expired(self):
        """Clear all expired prohibitions."""
        expired = [s for s, p in self._prohibitions.items() if p.is_expired()]
        for s in expired:
            self.remove(s)


# Global instance
_list: Optional[ProhibitedList] = None


def get_list() -> ProhibitedList:
    """Get global prohibited list."""
    global _list
    if _list is None:
        _list = ProhibitedList()
    return _list


def check_symbol(symbol: str) -> Optional[Prohibition]:
    """Check if symbol is prohibited."""
    return get_list().check(symbol)


def add_prohibition(symbol: str, reason: ProhibitionReason, **kwargs):
    """Add symbol to prohibited list."""
    get_list().add(symbol, reason, **kwargs)


def is_prohibited(symbol: str) -> bool:
    """Check if symbol is prohibited."""
    return get_list().is_prohibited(symbol)
