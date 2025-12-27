"""
Decision Tracker for Audit and Transparency
=============================================

Records trading decisions with full context for audit,
review, and machine learning purposes.

Tracks:
- What decision was made (entry, exit, skip)
- Why it was made (indicators, filters, model outputs)
- When it was made (timestamp, market conditions)
- Outcome (if known)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Type of trading decision."""
    ENTRY = "entry"           # Enter a position
    EXIT = "exit"             # Exit a position
    SKIP = "skip"             # Skip a signal
    HOLD = "hold"             # Continue holding
    SCALE_IN = "scale_in"     # Add to position
    SCALE_OUT = "scale_out"   # Reduce position


class DecisionReason(Enum):
    """Reason category for the decision."""
    SIGNAL = "signal"         # Strategy signal triggered
    STOP_LOSS = "stop_loss"   # Stop loss hit
    TAKE_PROFIT = "take_profit"  # Take profit hit
    TIME_STOP = "time_stop"   # Time-based exit
    FILTER = "filter"         # Filtered out
    RISK = "risk"             # Risk limit
    MANUAL = "manual"         # Manual intervention


@dataclass
class DecisionContext:
    """Context at the time of decision."""
    # Market state
    symbol: str
    price: float
    timestamp: datetime = field(default_factory=datetime.now)

    # Indicator values
    indicators: Dict[str, float] = field(default_factory=dict)

    # Model outputs
    model_confidence: Optional[float] = None
    model_prediction: Optional[str] = None

    # Market regime
    regime: Optional[str] = None
    volatility: Optional[float] = None

    # Position state (if relevant)
    current_position: Optional[float] = None
    unrealized_pnl: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'indicators': self.indicators,
            'model_confidence': self.model_confidence,
            'model_prediction': self.model_prediction,
            'regime': self.regime,
            'volatility': self.volatility,
            'current_position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl,
        }


@dataclass
class DecisionRecord:
    """A recorded trading decision."""
    decision_id: str
    decision_type: DecisionType
    reason: DecisionReason
    context: DecisionContext
    created_at: datetime = field(default_factory=datetime.now)

    # Decision details
    action_taken: str = ""
    rationale: str = ""

    # Filters that were applied
    filters_passed: List[str] = field(default_factory=list)
    filters_failed: List[str] = field(default_factory=list)

    # Outcome (filled in later)
    outcome_pnl: Optional[float] = None
    outcome_duration: Optional[int] = None  # bars held
    outcome_recorded_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'decision_id': self.decision_id,
            'decision_type': self.decision_type.value,
            'reason': self.reason.value,
            'context': self.context.to_dict(),
            'created_at': self.created_at.isoformat(),
            'action_taken': self.action_taken,
            'rationale': self.rationale,
            'filters_passed': self.filters_passed,
            'filters_failed': self.filters_failed,
            'outcome_pnl': self.outcome_pnl,
            'outcome_duration': self.outcome_duration,
            'outcome_recorded_at': self.outcome_recorded_at.isoformat() if self.outcome_recorded_at else None,
        }

    @property
    def was_successful(self) -> Optional[bool]:
        """Whether the decision was successful (profitable)."""
        if self.outcome_pnl is None:
            return None
        return self.outcome_pnl > 0


class DecisionTracker:
    """
    Tracks trading decisions for audit and review.

    Records full context for each decision to enable:
    - Audit trail for compliance
    - Post-trade review and learning
    - ML training data generation
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        auto_persist: bool = True,
        max_records: int = 10000,
    ):
        """
        Initialize the decision tracker.

        Args:
            log_dir: Directory for persisting decisions
            auto_persist: Whether to auto-save decisions
            max_records: Maximum records to keep in memory
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs/decisions")
        self.auto_persist = auto_persist
        self.max_records = max_records

        # In-memory records
        self._records: List[DecisionRecord] = []
        self._decision_counter = 0

        # Create log directory
        if auto_persist:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DecisionTracker initialized, log_dir={self.log_dir}")

    def _generate_id(self, context: DecisionContext) -> str:
        """Generate a unique decision ID."""
        self._decision_counter += 1
        hash_input = f"{context.symbol}{context.timestamp.isoformat()}{self._decision_counter}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"D{self._decision_counter:06d}_{short_hash}"

    def record(
        self,
        decision_type: DecisionType,
        reason: DecisionReason,
        context: DecisionContext,
        action_taken: str = "",
        rationale: str = "",
        filters_passed: Optional[List[str]] = None,
        filters_failed: Optional[List[str]] = None,
    ) -> DecisionRecord:
        """
        Record a trading decision.

        Args:
            decision_type: Type of decision (entry, exit, skip)
            reason: Reason for the decision
            context: Market and model context
            action_taken: Description of action
            rationale: Explanation for decision
            filters_passed: Filters that passed
            filters_failed: Filters that failed

        Returns:
            Created DecisionRecord
        """
        decision_id = self._generate_id(context)

        record = DecisionRecord(
            decision_id=decision_id,
            decision_type=decision_type,
            reason=reason,
            context=context,
            action_taken=action_taken,
            rationale=rationale,
            filters_passed=filters_passed or [],
            filters_failed=filters_failed or [],
        )

        self._records.append(record)

        # Trim if over limit
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]

        # Auto-persist
        if self.auto_persist:
            self._persist_record(record)

        logger.debug(
            f"Recorded decision {decision_id}: {decision_type.value} {context.symbol}"
        )

        return record

    def record_entry(
        self,
        symbol: str,
        price: float,
        indicators: Dict[str, float],
        reason: str = "",
        confidence: Optional[float] = None,
    ) -> DecisionRecord:
        """Convenience method to record an entry decision."""
        context = DecisionContext(
            symbol=symbol,
            price=price,
            indicators=indicators,
            model_confidence=confidence,
        )

        return self.record(
            decision_type=DecisionType.ENTRY,
            reason=DecisionReason.SIGNAL,
            context=context,
            action_taken=f"LONG {symbol} @ {price:.2f}",
            rationale=reason,
        )

    def record_exit(
        self,
        symbol: str,
        price: float,
        reason: DecisionReason,
        pnl: Optional[float] = None,
        rationale: str = "",
    ) -> DecisionRecord:
        """Convenience method to record an exit decision."""
        context = DecisionContext(
            symbol=symbol,
            price=price,
            unrealized_pnl=pnl,
        )

        record = self.record(
            decision_type=DecisionType.EXIT,
            reason=reason,
            context=context,
            action_taken=f"EXIT {symbol} @ {price:.2f}",
            rationale=rationale,
        )

        # Record outcome immediately if known
        if pnl is not None:
            record.outcome_pnl = pnl
            record.outcome_recorded_at = datetime.now()

        return record

    def record_skip(
        self,
        symbol: str,
        price: float,
        reason: str,
        filters_failed: Optional[List[str]] = None,
    ) -> DecisionRecord:
        """Convenience method to record a skip decision."""
        context = DecisionContext(
            symbol=symbol,
            price=price,
        )

        return self.record(
            decision_type=DecisionType.SKIP,
            reason=DecisionReason.FILTER,
            context=context,
            action_taken=f"SKIP {symbol}",
            rationale=reason,
            filters_failed=filters_failed,
        )

    def record_outcome(
        self,
        decision_id: str,
        pnl: float,
        duration: Optional[int] = None,
    ) -> bool:
        """
        Record the outcome of a decision.

        Args:
            decision_id: ID of the decision
            pnl: Realized P&L
            duration: Duration in bars

        Returns:
            True if record was found and updated
        """
        for record in reversed(self._records):
            if record.decision_id == decision_id:
                record.outcome_pnl = pnl
                record.outcome_duration = duration
                record.outcome_recorded_at = datetime.now()

                if self.auto_persist:
                    self._persist_record(record)

                return True

        logger.warning(f"Decision {decision_id} not found for outcome update")
        return False

    def _persist_record(self, record: DecisionRecord):
        """Persist a record to disk."""
        try:
            date_str = record.created_at.strftime('%Y%m%d')
            filepath = self.log_dir / f"decisions_{date_str}.jsonl"

            with open(filepath, 'a') as f:
                f.write(json.dumps(record.to_dict()) + '\n')

        except Exception as e:
            logger.error(f"Failed to persist decision: {e}")

    def get_history(
        self,
        symbol: Optional[str] = None,
        decision_type: Optional[DecisionType] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[DecisionRecord]:
        """
        Get decision history with optional filters.

        Args:
            symbol: Filter by symbol
            decision_type: Filter by decision type
            since: Filter by datetime
            limit: Maximum records to return

        Returns:
            List of matching DecisionRecords
        """
        records = self._records

        if symbol:
            records = [r for r in records if r.context.symbol == symbol]

        if decision_type:
            records = [r for r in records if r.decision_type == decision_type]

        if since:
            records = [r for r in records if r.created_at >= since]

        return records[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self._records:
            return {
                'total_decisions': 0,
                'entries': 0,
                'exits': 0,
                'skips': 0,
            }

        entries = [r for r in self._records if r.decision_type == DecisionType.ENTRY]
        exits = [r for r in self._records if r.decision_type == DecisionType.EXIT]
        skips = [r for r in self._records if r.decision_type == DecisionType.SKIP]

        # Calculate success rate for completed trades
        completed = [r for r in self._records if r.outcome_pnl is not None]
        if completed:
            wins = sum(1 for r in completed if r.outcome_pnl > 0)
            win_rate = wins / len(completed)
        else:
            win_rate = None

        return {
            'total_decisions': len(self._records),
            'entries': len(entries),
            'exits': len(exits),
            'skips': len(skips),
            'completed_trades': len(completed),
            'win_rate': win_rate,
        }

    def clear(self):
        """Clear all in-memory records."""
        self._records = []
        logger.info("Decision tracker cleared")


def record_decision(
    decision_type: DecisionType,
    symbol: str,
    price: float,
    reason: str = "",
    **kwargs,
) -> DecisionRecord:
    """Convenience function to record a decision."""
    tracker = get_tracker()
    context = DecisionContext(
        symbol=symbol,
        price=price,
        indicators=kwargs.get('indicators', {}),
        model_confidence=kwargs.get('confidence'),
    )

    return tracker.record(
        decision_type=decision_type,
        reason=DecisionReason.SIGNAL,
        context=context,
        rationale=reason,
    )


def get_decision_history(
    symbol: Optional[str] = None,
    limit: int = 100,
) -> List[DecisionRecord]:
    """Convenience function to get decision history."""
    tracker = get_tracker()
    return tracker.get_history(symbol=symbol, limit=limit)


# Module-level tracker instance
_tracker: Optional[DecisionTracker] = None


def get_tracker() -> DecisionTracker:
    """Get or create the global tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = DecisionTracker()
    return _tracker
