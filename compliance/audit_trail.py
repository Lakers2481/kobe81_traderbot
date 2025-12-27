"""
Audit Trail for Compliance
============================

Maintains detailed audit log of all trading activities.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class AuditAction(Enum):
    """Type of auditable action."""
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    RULE_VIOLATION = "rule_violation"
    CONFIG_CHANGE = "config_change"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class AuditEntry:
    """Single audit log entry."""
    action: AuditAction
    timestamp: datetime = field(default_factory=datetime.now)
    user: str = "system"
    symbol: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    entry_hash: str = ""

    def __post_init__(self):
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash for integrity."""
        data = f"{self.action.value}{self.timestamp.isoformat()}{self.symbol}{json.dumps(self.details, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action.value,
            'timestamp': self.timestamp.isoformat(),
            'user': self.user,
            'symbol': self.symbol,
            'details': self.details,
            'hash': self.entry_hash,
        }

    def to_line(self) -> str:
        """Convert to log line."""
        return json.dumps(self.to_dict())


class AuditTrail:
    """
    Maintains audit trail for compliance and forensics.
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        auto_persist: bool = True,
    ):
        self.log_dir = log_dir or Path("logs/audit")
        self.auto_persist = auto_persist
        self._entries: List[AuditEntry] = []
        self._last_hash = ""

        if auto_persist:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info("AuditTrail initialized")

    def log(
        self,
        action: AuditAction,
        symbol: str = "",
        details: Optional[Dict[str, Any]] = None,
        user: str = "system",
    ) -> AuditEntry:
        """Log an audit entry."""
        entry = AuditEntry(
            action=action,
            symbol=symbol,
            details=details or {},
            user=user,
        )

        self._entries.append(entry)
        self._last_hash = entry.entry_hash

        if self.auto_persist:
            self._persist(entry)

        logger.debug(f"Audit: {action.value} {symbol}")

        return entry

    def _persist(self, entry: AuditEntry):
        """Persist entry to log file."""
        try:
            date_str = entry.timestamp.strftime('%Y%m%d')
            filepath = self.log_dir / f"audit_{date_str}.jsonl"

            with open(filepath, 'a') as f:
                f.write(entry.to_line() + '\n')

        except Exception as e:
            logger.error(f"Failed to persist audit entry: {e}")

    def log_order(
        self,
        action: AuditAction,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order_id: str = "",
    ):
        """Log order-related action."""
        self.log(
            action=action,
            symbol=symbol,
            details={
                'side': side,
                'quantity': quantity,
                'price': price,
                'order_id': order_id,
            },
        )

    def log_violation(
        self,
        symbol: str,
        rule_name: str,
        details: str,
    ):
        """Log rule violation."""
        self.log(
            action=AuditAction.RULE_VIOLATION,
            symbol=symbol,
            details={
                'rule': rule_name,
                'details': details,
            },
        )

    def get_history(
        self,
        action: Optional[AuditAction] = None,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Get audit history with filters."""
        entries = self._entries

        if action:
            entries = [e for e in entries if e.action == action]

        if symbol:
            entries = [e for e in entries if e.symbol == symbol]

        if since:
            entries = [e for e in entries if e.timestamp >= since]

        return entries[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get audit statistics."""
        if not self._entries:
            return {'total_entries': 0}

        action_counts = {}
        for entry in self._entries:
            action = entry.action.value
            action_counts[action] = action_counts.get(action, 0) + 1

        return {
            'total_entries': len(self._entries),
            'actions': action_counts,
            'first_entry': self._entries[0].timestamp.isoformat(),
            'last_entry': self._entries[-1].timestamp.isoformat(),
        }

    def verify_integrity(self) -> bool:
        """Verify integrity of audit trail."""
        for entry in self._entries:
            expected_hash = entry._compute_hash()
            if entry.entry_hash != expected_hash:
                logger.error(f"Audit integrity check failed for entry {entry.timestamp}")
                return False
        return True


# Global instance
_trail: Optional[AuditTrail] = None


def get_trail() -> AuditTrail:
    """Get global audit trail."""
    global _trail
    if _trail is None:
        _trail = AuditTrail()
    return _trail


def log_audit(
    action: AuditAction,
    symbol: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> AuditEntry:
    """Log an audit entry."""
    return get_trail().log(action, symbol, details)


def get_audit_history(limit: int = 100) -> List[AuditEntry]:
    """Get recent audit history."""
    return get_trail().get_history(limit=limit)
