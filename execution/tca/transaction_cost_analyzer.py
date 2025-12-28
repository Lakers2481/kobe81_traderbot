"""
Transaction Cost Analyzer (TCA)
=================================

This module is responsible for analyzing the costs associated with trade
execution. It quantifies "slippage" - the difference between the expected
price of a trade and the actual price at which it was executed. TCA data
is crucial feedback for the `OrderManager` and the `SelfModel` of the
Cognitive Brain to improve execution strategies and refine the AI's
understanding of its own capabilities.

Features:
- Records detailed execution events.
- Calculates various forms of slippage (e.g., market impact, spread capture).
- Aggregates TCA metrics for performance reporting.
- Feeds back insights into the AI's learning mechanisms.

Usage:
    from execution.tca.transaction_cost_analyzer import get_tca_analyzer
    from oms.order_state import OrderRecord, OrderStatus

    tca_analyzer = get_tca_analyzer()

    # After an order is filled, record its execution details
    order = OrderRecord(symbol="AAPL", side="BUY", qty=100)
    fill_details = {
        'order': order,
        'fill_price': 150.10,
        'expected_price': 150.00,
        'timestamp': datetime.now(),
        'market_conditions': {'vix': 20.0, 'volume': 100000},
    }
    tca_analyzer.record_execution(fill_details)

    # Get aggregated TCA metrics
    daily_tca = tca_analyzer.get_daily_summary()
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from cognitive.self_model import get_self_model
from cognitive.global_workspace import get_workspace
from oms.order_state import OrderRecord, OrderStatus

logger = logging.getLogger(__name__)


@dataclass
class TCARecord:
    """Represents a single recorded trade execution for TCA."""
    execution_id: str  # Link to the order execution
    symbol: str
    side: str
    qty: int
    entry_price_decision: float  # Price at which the decision was made
    fill_price: float
    market_bid_at_execution: float
    market_ask_at_execution: float
    timestamp: datetime
    market_context: Dict[str, Any] = field(default_factory=dict)
    strategy: str = "unknown"
    status: OrderStatus = OrderStatus.PENDING

    @property
    def slippage_bps(self) -> float:
        """
        Calculates slippage in basis points.
        Positive slippage means the fill was worse than expected.
        """
        if self.side == "BUY":
            # For buy, actual fill price > expected price means negative slippage
            return ((self.fill_price - self.entry_price_decision) / self.entry_price_decision) * 10000
        else: # SELL
            # For sell, actual fill price < expected price means negative slippage
            return ((self.entry_price_decision - self.fill_price) / self.entry_price_decision) * 10000

    @property
    def spread_capture_bps(self) -> float:
        """
        Calculates how much of the bid-ask spread was captured.
        Positive means filled within the spread (good), negative means outside.
        """
        spread = self.market_ask_at_execution - self.market_bid_at_execution
        if spread <= 0: return 0.0

        mid_price = (self.market_bid_at_execution + self.market_ask_at_execution) / 2

        if self.side == "BUY":
            return ((mid_price - self.fill_price) / mid_price) * 10000
        else: # SELL
            return ((self.fill_price - mid_price) / mid_price) * 10000
            
    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d['timestamp'] = self.timestamp.isoformat()
        d['side'] = self.side.value if isinstance(self.side, Enum) else self.side
        d['status'] = self.status.value
        d['slippage_bps'] = self.slippage_bps
        d['spread_capture_bps'] = self.spread_capture_bps
        return d


class TransactionCostAnalyzer:
    """
    Manages the collection and analysis of trade execution costs (slippage).
    Feeds back insights to the AI's SelfModel and potentially SemanticMemory.
    """

    def __init__(self, storage_dir: str = "state/tca", auto_persist: bool = True):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.auto_persist = auto_persist

        self._tca_records: List[TCARecord] = []
        self._max_records = 10000 # Keep a rolling window of records

        # Lazy load dependencies
        self._self_model = None
        self._workspace = None

        self._load_records()
        logger.info(f"TransactionCostAnalyzer initialized with {len(self._tca_records)} records.")

    @property
    def self_model(self):
        """Lazy-loads the SelfModel."""
        if self._self_model is None:
            self._self_model = get_self_model()
        return self._self_model

    @property
    def workspace(self):
        """Lazy-loads the GlobalWorkspace."""
        if self._workspace is None:
            self._workspace = get_workspace()
        return self._workspace

    def record_execution(
        self,
        order: OrderRecord,
        fill_price: float,
        market_bid_at_execution: float,
        market_ask_at_execution: float,
        entry_price_decision: float, # Price at time decision was made (benchmark)
        market_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Records the details of a filled order for later TCA.

        Args:
            order: The original OrderRecord submitted.
            fill_price: The actual price at which the order was filled.
            market_bid_at_execution: Best bid price in market at time of execution.
            market_ask_at_execution: Best ask price in market at time of execution.
            entry_price_decision: The price the AI decided to enter at (benchmark).
            market_context: Relevant market conditions at the time of execution.
        """
        record = TCARecord(
            execution_id=order.execution_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            entry_price_decision=entry_price_decision,
            fill_price=fill_price,
            market_bid_at_execution=market_bid_at_execution,
            market_ask_at_execution=market_ask_at_execution,
            timestamp=datetime.now(), # Use current time for TCA record timestamp
            market_context=market_context or {},
            strategy=order.strategy_used,
            status=order.status,
        )
        self._tca_records.append(record)
        
        # Keep list within max_records
        if len(self._tca_records) > self._max_records:
            self._tca_records.pop(0)

        if self.auto_persist:
            self._save_records()

        # Feedback to SelfModel
        slippage_bps = record.slippage_bps
        self.self_model.record_tca_feedback(
            strategy=record.strategy,
            regime=record.market_context.get('regime', 'unknown'),
            slippage_bps=slippage_bps,
            market_impact=(record.fill_price - entry_price_decision) # crude market impact for now
        )

        logger.info(
            f"TCA Recorded for {record.symbol} ({record.side}): "
            f"Slippage: {slippage_bps:.2f} bps, "
            f"Spread Capture: {record.spread_capture_bps:.2f} bps"
        )

    def get_recent_tca(self, limit: int = 100) -> List[TCARecord]:
        """Returns recent TCA records."""
        return self._tca_records[-limit:]

    def get_summary_tca_metrics(self, lookback_days: int = 7) -> Dict[str, Any]:
        """
        Aggregates TCA metrics over a specified lookback period.
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_records = [r for r in self._tca_records if r.timestamp >= cutoff]

        if not recent_records:
            return {
                'total_trades': 0,
                'avg_slippage_bps': 0.0,
                'avg_spread_capture_bps': 0.0,
                'total_cost_usd': 0.0,
            }

        total_trades = len(recent_records)
        avg_slippage_bps = sum(r.slippage_bps for r in recent_records) / total_trades
        avg_spread_capture_bps = sum(r.spread_capture_bps for r in recent_records) / total_trades
        # For BUY, cost = (fill - entry) positive when overpaid
        # For SELL, cost = (entry - fill) positive when underpaid (got less)
        def calc_cost(r):
            if r.side == "BUY":
                return (r.fill_price - r.entry_price_decision) * r.qty
            else:  # SELL
                return (r.entry_price_decision - r.fill_price) * r.qty
        total_cost_usd = sum(calc_cost(r) for r in recent_records)

        return {
            'total_trades': total_trades,
            'avg_slippage_bps': round(avg_slippage_bps, 2),
            'avg_spread_capture_bps': round(avg_spread_capture_bps, 2),
            'total_cost_usd': round(total_cost_usd, 2),
        }
        
    def introspect(self) -> str:
        """Generates an introspection report for the TransactionCostAnalyzer."""
        summary = self.get_summary_tca_metrics(lookback_days=7)
        
        lines = [
            "--- Transaction Cost Analyzer Introspection ---",
            "My purpose is to constantly monitor and quantify execution quality.",
            f"Over the last 7 days, I recorded {summary['total_trades']} trades.",
            f"Average Slippage: {summary['avg_slippage_bps']:.2f} bps",
            f"Average Spread Capture: {summary['avg_spread_capture_bps']:.2f} bps",
            f"Total Estimated Execution Cost: ${summary['total_cost_usd']:.2f}",
        ]
        if summary['avg_slippage_bps'] > 5: # More than 5 bps slippage is often considered high
            lines.append("Recommendation: High slippage detected. Consider reviewing execution strategies, broker quality, or reducing order sizes in current market conditions.")
        
        return "\n".join(lines)

    def _save_records(self) -> None:
        """Saves TCA records to a JSON file."""
        try:
            with open(self.storage_dir / "tca_records.json", 'w') as f:
                json.dump([r.to_dict() for r in self._tca_records], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save TCA records: {e}")

    def _load_records(self) -> None:
        """Loads TCA records from a JSON file."""
        tca_file = self.storage_dir / "tca_records.json"
        if not tca_file.exists(): return

        try:
            with open(tca_file, 'r') as f:
                data = json.load(f)
            self._tca_records = [TCARecord(**r) for r in data]
        except Exception as e:
            logger.warning(f"Failed to load TCA records from {tca_file}: {e}")
            self._tca_records = []


# Singleton instance
_tca_analyzer: Optional[TransactionCostAnalyzer] = None

def get_tca_analyzer(storage_dir: str = "state/tca") -> TransactionCostAnalyzer:
    """Factory function to get the singleton instance of the TransactionCostAnalyzer."""
    global _tca_analyzer
    if _tca_analyzer is None:
        _tca_analyzer = TransactionCostAnalyzer(storage_dir=storage_dir)
    return _tca_analyzer
