"""
Signal Queue for External Signal Ingestion.

Provides a thread-safe queue for receiving signals from external sources
like TradingView webhooks, custom alerting systems, etc.

Features:
- Thread-safe signal queueing
- JSONL persistence for audit trail
- Signal deduplication via idempotency keys
- Expiration handling for stale signals
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SignalSource(Enum):
    """Signal source types."""
    TRADINGVIEW = auto()
    CUSTOM_WEBHOOK = auto()
    TELEGRAM = auto()
    EMAIL = auto()
    INTERNAL = auto()


class SignalStatus(Enum):
    """Signal processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    EXECUTED = "executed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    DUPLICATE = "duplicate"


@dataclass
class ExternalSignal:
    """
    External trading signal from webhook or other source.

    Designed to be compatible with internal signal format while
    capturing additional context from external sources.
    """
    # Core signal data
    symbol: str
    side: str  # "buy", "sell", "long", "short"
    action: str = "open"  # "open", "close", "scale_in", "scale_out"

    # Signal identification
    signal_id: str = field(default_factory=lambda: f"EXT_{uuid.uuid4().hex[:12].upper()}")
    idempotency_key: Optional[str] = None  # For deduplication

    # Source metadata
    source: SignalSource = SignalSource.CUSTOM_WEBHOOK
    source_name: str = ""  # e.g., "TradingView", "MyAlert"

    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None  # Signal expires after this time

    # Optional trade parameters
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    qty: Optional[int] = None
    risk_pct: Optional[float] = None  # Alternative to qty - risk-based sizing

    # Strategy context
    strategy: Optional[str] = None  # Strategy name if applicable
    timeframe: Optional[str] = None  # e.g., "1h", "4h", "1d"
    confidence: Optional[float] = None  # 0-1 confidence score

    # Processing status
    status: SignalStatus = SignalStatus.PENDING
    processed_at: Optional[datetime] = None
    result_notes: str = ""

    # Raw payload for debugging
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize fields after initialization."""
        self.symbol = self.symbol.upper().strip()
        self.side = self.side.lower().strip()
        self.action = self.action.lower().strip()

        # Generate idempotency key if not provided
        if self.idempotency_key is None:
            # Hash of symbol + side + action + timestamp (minute precision)
            ts_minute = self.timestamp.strftime("%Y%m%d%H%M")
            self.idempotency_key = f"{self.symbol}_{self.side}_{self.action}_{ts_minute}"

        # Set default expiration (5 minutes)
        if self.expires_at is None:
            self.expires_at = self.timestamp + timedelta(minutes=5)

    @property
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get signal age in seconds."""
        return (datetime.utcnow() - self.timestamp).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["source"] = self.source.name
        d["status"] = self.status.value
        d["timestamp"] = self.timestamp.isoformat()
        d["expires_at"] = self.expires_at.isoformat() if self.expires_at else None
        d["processed_at"] = self.processed_at.isoformat() if self.processed_at else None
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExternalSignal":
        """Create from dictionary."""
        # Parse enums
        if isinstance(data.get("source"), str):
            data["source"] = SignalSource[data["source"]]
        if isinstance(data.get("status"), str):
            data["status"] = SignalStatus(data["status"])

        # Parse datetimes
        for dt_field in ["timestamp", "expires_at", "processed_at"]:
            if data.get(dt_field) and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])

        return cls(**data)


class SignalQueue:
    """
    Thread-safe queue for external trading signals.

    Supports:
    - FIFO signal processing
    - Deduplication via idempotency keys
    - Persistence to JSONL file
    - Callbacks for signal processing
    - Automatic expiration handling
    """

    DEFAULT_QUEUE_PATH = Path("logs/signal_queue.jsonl")
    DEFAULT_ARCHIVE_PATH = Path("logs/signal_archive.jsonl")
    MAX_QUEUE_SIZE = 1000  # Max pending signals

    def __init__(
        self,
        queue_path: Optional[Path] = None,
        archive_path: Optional[Path] = None,
        max_queue_size: int = MAX_QUEUE_SIZE,
        persist: bool = True,
    ):
        """
        Initialize signal queue.

        Args:
            queue_path: Path for pending signals JSONL
            archive_path: Path for processed signals JSONL
            max_queue_size: Maximum pending signals
            persist: Whether to persist to disk
        """
        self.queue_path = queue_path or self.DEFAULT_QUEUE_PATH
        self.archive_path = archive_path or self.DEFAULT_ARCHIVE_PATH
        self.max_queue_size = max_queue_size
        self.persist = persist

        # Thread-safe internals
        self._lock = threading.RLock()
        self._queue: deque[ExternalSignal] = deque(maxlen=max_queue_size)
        self._seen_keys: Set[str] = set()
        self._callbacks: List[Callable[[ExternalSignal], None]] = []

        # Ensure directories exist
        if persist:
            self.queue_path.parent.mkdir(parents=True, exist_ok=True)
            self.archive_path.parent.mkdir(parents=True, exist_ok=True)

    def enqueue(self, signal: ExternalSignal) -> bool:
        """
        Add a signal to the queue.

        Args:
            signal: ExternalSignal to queue

        Returns:
            True if queued, False if duplicate or expired
        """
        with self._lock:
            # Check for duplicate
            if signal.idempotency_key in self._seen_keys:
                signal.status = SignalStatus.DUPLICATE
                signal.result_notes = "Duplicate idempotency key"
                logger.debug(f"Duplicate signal rejected: {signal.idempotency_key}")
                self._persist_to_archive(signal)
                return False

            # Check for expiration
            if signal.is_expired:
                signal.status = SignalStatus.EXPIRED
                signal.result_notes = f"Signal expired (age: {signal.age_seconds:.1f}s)"
                logger.debug(f"Expired signal rejected: {signal.signal_id}")
                self._persist_to_archive(signal)
                return False

            # Add to queue
            self._queue.append(signal)
            self._seen_keys.add(signal.idempotency_key)

            # Persist
            self._persist_signal(signal)

            logger.info(
                f"Signal queued: {signal.symbol} {signal.side} {signal.action} "
                f"(source={signal.source.name}, id={signal.signal_id})"
            )

            # Trigger callbacks
            for callback in self._callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    logger.error(f"Signal callback error: {e}")

            return True

    def dequeue(self) -> Optional[ExternalSignal]:
        """
        Get next signal from queue.

        Returns:
            ExternalSignal or None if queue is empty
        """
        with self._lock:
            # Skip expired signals
            while self._queue:
                signal = self._queue.popleft()

                if signal.is_expired:
                    signal.status = SignalStatus.EXPIRED
                    signal.result_notes = f"Expired while in queue (age: {signal.age_seconds:.1f}s)"
                    self._persist_to_archive(signal)
                    continue

                signal.status = SignalStatus.PROCESSING
                return signal

            return None

    def peek(self) -> Optional[ExternalSignal]:
        """Peek at next signal without removing."""
        with self._lock:
            return self._queue[0] if self._queue else None

    def mark_processed(
        self,
        signal: ExternalSignal,
        status: SignalStatus = SignalStatus.EXECUTED,
        notes: str = "",
    ) -> None:
        """
        Mark a signal as processed.

        Args:
            signal: The signal that was processed
            status: Final status
            notes: Processing notes
        """
        with self._lock:
            signal.status = status
            signal.processed_at = datetime.utcnow()
            signal.result_notes = notes
            self._persist_to_archive(signal)

    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.size() == 0

    def clear(self) -> int:
        """Clear queue and return count of cleared signals."""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count

    def register_callback(self, callback: Callable[[ExternalSignal], None]) -> None:
        """Register callback for new signals."""
        with self._lock:
            self._callbacks.append(callback)

    def get_pending(self) -> List[ExternalSignal]:
        """Get all pending signals (snapshot)."""
        with self._lock:
            return list(self._queue)

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            pending = list(self._queue)

            by_source = {}
            by_symbol = {}
            for s in pending:
                by_source[s.source.name] = by_source.get(s.source.name, 0) + 1
                by_symbol[s.symbol] = by_symbol.get(s.symbol, 0) + 1

            return {
                "pending_count": len(pending),
                "seen_keys_count": len(self._seen_keys),
                "max_queue_size": self.max_queue_size,
                "by_source": by_source,
                "by_symbol": by_symbol,
                "oldest_signal_age_s": pending[0].age_seconds if pending else 0,
            }

    def _persist_signal(self, signal: ExternalSignal) -> None:
        """Persist signal to queue file."""
        if not self.persist:
            return

        try:
            with open(self.queue_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(signal.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist signal: {e}")

    def _persist_to_archive(self, signal: ExternalSignal) -> None:
        """Persist processed signal to archive."""
        if not self.persist:
            return

        try:
            with open(self.archive_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(signal.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to archive signal: {e}")


# Global singleton
_signal_queue: Optional[SignalQueue] = None
_queue_lock = threading.Lock()


def get_signal_queue() -> SignalQueue:
    """Get or create the global signal queue."""
    global _signal_queue
    with _queue_lock:
        if _signal_queue is None:
            _signal_queue = SignalQueue()
        return _signal_queue


def set_signal_queue(queue: SignalQueue) -> None:
    """Set the global signal queue (for testing)."""
    global _signal_queue
    with _queue_lock:
        _signal_queue = queue
