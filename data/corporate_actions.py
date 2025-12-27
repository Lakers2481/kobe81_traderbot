"""
Corporate actions tracking for backtesting.

Tracks stock splits, dividends, and other corporate actions.
Polygon data is already split-adjusted, so this module primarily
warns about potential issues rather than re-adjusting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of corporate actions."""
    SPLIT = auto()
    REVERSE_SPLIT = auto()
    DIVIDEND = auto()
    SPINOFF = auto()
    MERGER = auto()
    SYMBOL_CHANGE = auto()


@dataclass
class CorporateAction:
    """A corporate action event."""
    symbol: str
    action_type: ActionType
    effective_date: date
    factor: Optional[float] = None  # Split ratio or dividend amount
    old_symbol: Optional[str] = None  # For symbol changes
    new_symbol: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action_type": self.action_type.name,
            "effective_date": self.effective_date.isoformat(),
            "factor": self.factor,
            "old_symbol": self.old_symbol,
            "new_symbol": self.new_symbol,
            "description": self.description,
        }


class CorporateActionsTracker:
    """
    Tracks corporate actions for backtest awareness.

    Since Polygon provides split-adjusted data, we don't re-adjust.
    Instead, we:
    1. Detect potential splits from price jumps
    2. Log warnings during backtest periods with splits
    3. Maintain a registry of known actions
    """

    def __init__(self, actions_path: Optional[Path] = None):
        self.actions: List[CorporateAction] = []
        self.actions_by_symbol: Dict[str, List[CorporateAction]] = {}
        self.actions_path = actions_path or Path("data/corporate_actions.json")

        if self.actions_path.exists():
            self._load_actions()

    def _load_actions(self) -> None:
        """Load known corporate actions from file."""
        try:
            with open(self.actions_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data.get("actions", []):
                action = CorporateAction(
                    symbol=item["symbol"],
                    action_type=ActionType[item["action_type"]],
                    effective_date=date.fromisoformat(item["effective_date"]),
                    factor=item.get("factor"),
                    old_symbol=item.get("old_symbol"),
                    new_symbol=item.get("new_symbol"),
                    description=item.get("description", ""),
                )
                self.actions.append(action)

                symbol = action.symbol.upper()
                if symbol not in self.actions_by_symbol:
                    self.actions_by_symbol[symbol] = []
                self.actions_by_symbol[symbol].append(action)

        except Exception as e:
            logger.warning(f"Could not load corporate actions: {e}")

    def save_actions(self) -> None:
        """Save corporate actions to file."""
        self.actions_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "updated": datetime.utcnow().isoformat(),
            "actions": [a.to_dict() for a in self.actions],
        }

        with open(self.actions_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_action(self, action: CorporateAction) -> None:
        """Add a corporate action to the registry."""
        self.actions.append(action)

        symbol = action.symbol.upper()
        if symbol not in self.actions_by_symbol:
            self.actions_by_symbol[symbol] = []
        self.actions_by_symbol[symbol].append(action)

    def get_actions_in_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> List[CorporateAction]:
        """Get actions for a symbol within a date range."""
        symbol = symbol.upper()
        if symbol not in self.actions_by_symbol:
            return []

        return [
            a for a in self.actions_by_symbol[symbol]
            if start_date <= a.effective_date <= end_date
        ]

    def has_splits_in_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> bool:
        """Check if symbol had splits in date range."""
        actions = self.get_actions_in_range(symbol, start_date, end_date)
        return any(
            a.action_type in (ActionType.SPLIT, ActionType.REVERSE_SPLIT)
            for a in actions
        )

    def detect_potential_split(
        self,
        symbol: str,
        prev_close: float,
        current_open: float,
        threshold: float = 0.40,
    ) -> Optional[CorporateAction]:
        """
        Detect potential split from price jump.

        If price jumps by more than threshold (40% default), it may
        indicate a split that wasn't in our registry.

        Returns detected action or None.
        """
        if prev_close <= 0 or current_open <= 0:
            return None

        ratio = current_open / prev_close

        # Check for significant jump
        if ratio > (1 + threshold):
            # Price jumped up significantly - possible reverse split or data issue
            factor = round(ratio)
            if factor >= 2:
                action = CorporateAction(
                    symbol=symbol,
                    action_type=ActionType.REVERSE_SPLIT,
                    effective_date=date.today(),
                    factor=factor,
                    description=f"Detected {factor}:1 reverse split (price jumped {ratio:.2f}x)",
                )
                logger.warning(
                    f"Potential reverse split detected for {symbol}: "
                    f"price jumped from ${prev_close:.2f} to ${current_open:.2f}"
                )
                return action

        elif ratio < (1 - threshold):
            # Price dropped significantly - possible split
            factor = round(1 / ratio)
            if factor >= 2:
                action = CorporateAction(
                    symbol=symbol,
                    action_type=ActionType.SPLIT,
                    effective_date=date.today(),
                    factor=factor,
                    description=f"Detected 1:{factor} split (price dropped to {ratio:.2f}x)",
                )
                logger.warning(
                    f"Potential split detected for {symbol}: "
                    f"price dropped from ${prev_close:.2f} to ${current_open:.2f}"
                )
                return action

        return None

    def get_warnings_for_backtest(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> List[str]:
        """
        Get warning messages for a backtest period.

        Returns list of warning strings about corporate actions
        that may affect backtest results.
        """
        warnings = []

        for symbol in symbols:
            actions = self.get_actions_in_range(symbol, start_date, end_date)
            for action in actions:
                if action.action_type == ActionType.SPLIT:
                    warnings.append(
                        f"WARNING: {symbol} had a {action.factor}:1 split on "
                        f"{action.effective_date}. Data should be split-adjusted."
                    )
                elif action.action_type == ActionType.REVERSE_SPLIT:
                    warnings.append(
                        f"WARNING: {symbol} had a 1:{action.factor} reverse split on "
                        f"{action.effective_date}. Data should be split-adjusted."
                    )
                elif action.action_type == ActionType.SYMBOL_CHANGE:
                    warnings.append(
                        f"WARNING: {symbol} changed symbol from {action.old_symbol} "
                        f"to {action.new_symbol} on {action.effective_date}."
                    )

        return warnings


# Global tracker instance
_tracker: Optional[CorporateActionsTracker] = None


def get_tracker() -> CorporateActionsTracker:
    """Get global corporate actions tracker."""
    global _tracker
    if _tracker is None:
        _tracker = CorporateActionsTracker()
    return _tracker


def check_for_splits(
    symbol: str,
    start_date: date,
    end_date: date,
) -> bool:
    """Convenience function to check for splits."""
    return get_tracker().has_splits_in_range(symbol, start_date, end_date)


def get_backtest_warnings(
    symbols: List[str],
    start_date: date,
    end_date: date,
) -> List[str]:
    """Convenience function to get backtest warnings."""
    return get_tracker().get_warnings_for_backtest(symbols, start_date, end_date)


def log_backtest_warnings(
    symbols: List[str],
    start_date: date,
    end_date: date,
) -> None:
    """Log any corporate action warnings for the backtest period."""
    warnings = get_backtest_warnings(symbols, start_date, end_date)
    for warning in warnings:
        logger.warning(warning)
