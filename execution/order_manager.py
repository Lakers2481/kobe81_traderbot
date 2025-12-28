"""
Order Manager - Intelligent Order Execution Strategies
======================================================

This module provides advanced order execution capabilities beyond simple
market or limit orders. It aims to minimize market impact and optimize
fill prices by implementing intelligent strategies like Time-Weighted
Average Price (TWAP) and Volume-Weighted Average Price (VWAP).

The OrderManager acts as an intermediary between the IntelligentExecutor
and the Broker (Alpaca), deciding the best way to execute a given order
based on its size, market conditions, and specified strategy.

Features:
- **TWAP Execution:** Breaks down large orders into smaller chunks over time.
- **VWAP Execution:** Distributes orders according to historical volume profiles.
- **Dynamic Order Routing:** Chooses between simple IOC/Limit or advanced strategies.
- **Transaction Cost Analysis (TCA) Integration:** Records execution details
  for post-trade analysis.

Usage:
    from execution.order_manager import get_order_manager
    from oms.order_state import OrderRecord

    order_manager = get_order_manager()

    # Submit an order with a specified execution strategy
    order = OrderRecord(symbol="AAPL", side="BUY", qty=1000, order_type="TWAP")
    execution_id = order_manager.submit_order(order)

    # Or a simple limit order
    order = OrderRecord(symbol="GOOG", side="SELL", qty=50, order_type="LIMIT", limit_price=135.50)
    execution_id = order_manager.submit_order(order)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import time
import pandas as pd

from oms.order_state import OrderRecord, OrderStatus
from execution.broker_alpaca import place_ioc_limit, get_best_ask, get_best_bid, BrokerExecutionResult
from execution.tca.transaction_cost_analyzer import get_tca_analyzer

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Manages the intelligent execution of trades, applying various algorithms
    to optimize fills and minimize market impact.
    """

    def __init__(self, default_execution_strategy: str = "LIMIT"):
        self.default_execution_strategy = default_execution_strategy
        # Lazy load TCA analyzer
        self._tca_analyzer = None
        logger.info(f"OrderManager initialized with default strategy: {self.default_execution_strategy}")

    @property
    def tca_analyzer(self):
        """Lazy-loads the TransactionCostAnalyzer."""
        if self._tca_analyzer is None:
            self._tca_analyzer = get_tca_analyzer()
        return self._tca_analyzer

    def submit_order(self, order: OrderRecord, execution_strategy: Optional[str] = None) -> str:
        """
        Submits an order for intelligent execution.

        Args:
            order: The OrderRecord object containing order details.
            execution_strategy: The desired execution strategy (e.g., "TWAP", "VWAP", "LIMIT", "MARKET").
                                If None, uses the order.order_type or default.

        Returns:
            A unique execution ID for tracking.
        """
        strategy = execution_strategy or order.order_type or self.default_execution_strategy
        execution_id = f"EXEC-{order.symbol}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{strategy.upper()}"
        order.execution_id = execution_id
        
        logger.info(f"Submitting order {execution_id} for {order.symbol} with strategy: {strategy}")

        try:
            # Fetch current market bid/ask once for the order, to be used as benchmark for simple IOC/LIMIT
            market_bid_at_submission = get_best_bid(order.symbol)
            market_ask_at_submission = get_best_ask(order.symbol)

            if market_bid_at_submission is None or market_ask_at_submission is None:
                raise ValueError(f"Could not get current market quotes for {order.symbol} at order submission.")

            if strategy.upper() == "TWAP":
                self._execute_twap(order, market_bid_at_submission, market_ask_at_submission)
            elif strategy.upper() == "VWAP":
                self._execute_vwap(order, market_bid_at_submission, market_ask_at_submission)
            elif strategy.upper() == "MARKET":
                logger.warning(f"Market order strategy not fully implemented. Using IOC LIMIT for {order.symbol}.")
                # For market order simulation, set limit to a slightly unfavorable price to ensure fill
                if order.side.upper() == "BUY":
                    order.limit_price = market_ask_at_submission * 1.001
                else: # SELL
                    order.limit_price = market_bid_at_submission * 0.999
                self._execute_simple_ioc_limit(order, market_bid_at_submission, market_ask_at_submission)
            elif strategy.upper() == "LIMIT":
                self._execute_simple_ioc_limit(order, market_bid_at_submission, market_ask_at_submission)
            else:
                logger.warning(f"Unknown execution strategy '{strategy}'. Defaulting to LIMIT for {order.symbol}.")
                self._execute_simple_ioc_limit(order, market_bid_at_submission, market_ask_at_submission)
        except Exception as e:
            logger.error(f"Error submitting order {execution_id} for {order.symbol}: {e}")
            order.status = OrderStatus.FAILED
            order.notes = str(e) # Use notes for error message
        finally:
            # Ensure the order is recorded even if it failed submission, for audit.
            if order.status == OrderStatus.PENDING: # If status wasn't updated by execution method
                order.status = OrderStatus.FAILED
                order.notes = order.notes or "Unknown error during submission"

        return execution_id

    def _execute_simple_ioc_limit(self, order: OrderRecord, market_bid_at_submission: Optional[float] = None, market_ask_at_submission: Optional[float] = None):
        """
        Executes a simple Immediate-Or-Cancel Limit order and records TCA.
        """
        if not order.limit_price:
            # Attempt to set a reasonable limit price if not provided
            if order.side.upper() == "BUY":
                order.limit_price = market_ask_at_submission * 1.001 if market_ask_at_submission else get_best_ask(order.symbol) * 1.001
            else: # SELL
                order.limit_price = market_bid_at_submission * 0.999 if market_bid_at_submission else get_best_bid(order.symbol) * 0.999
            if not order.limit_price:
                raise ValueError(f"Could not determine limit price for {order.symbol}")

        logger.debug(f"Placing IOC Limit order for {order.qty} {order.symbol} at {order.limit_price}")
        
        broker_result: BrokerExecutionResult = place_ioc_limit(order)
        
        # Record execution for TCA
        if order.entry_price_decision: # Only record if benchmark price is set
            self.tca_analyzer.record_execution(
                order=broker_result.order,
                fill_price=broker_result.order.fill_price or order.limit_price, # Use fill price if available, else limit
                market_bid_at_execution=broker_result.market_bid_at_execution,
                market_ask_at_execution=broker_result.market_ask_at_execution,
                entry_price_decision=order.entry_price_decision,
            )
        
        # Update original order record status
        order.update_status(
            broker_result.order.status,
            broker_result.order.notes,
            broker_result.order.filled_qty,
            broker_result.order.fill_price
        )
        return broker_result

    def _execute_twap(self, order: OrderRecord, market_bid_at_submission: float, market_ask_at_submission: float, duration_minutes: int = 60, slice_count: int = 10):
        """
        Executes a Time-Weighted Average Price (TWAP) strategy.
        Breaks a large order into smaller slices over a specified duration.
        """
        total_qty = order.qty
        slice_qty = total_qty // slice_count
        remaining_qty = total_qty % slice_count
        interval_seconds = duration_minutes * 60 / slice_count

        logger.info(f"Executing TWAP for {total_qty} {order.symbol} over {duration_minutes} min, in {slice_count} slices.")

        for i in range(slice_count):
            qty_to_execute = slice_qty + (remaining_qty if i == slice_count - 1 else 0)
            if qty_to_execute <= 0:
                continue

            slice_order = OrderRecord(
                decision_id=order.decision_id, # Same decision ID for all slices
                signal_id=order.signal_id,
                symbol=order.symbol,
                side=order.side,
                qty=qty_to_execute,
                limit_price=order.limit_price, # Use original limit price for slices or dynamic
                tif="IOC",
                order_type="LIMIT",
                idempotency_key=f"{order.idempotency_key}-TWAP-slice-{i+1}",
                created_at=datetime.now(),
                entry_price_decision=order.entry_price_decision,
                strategy_used=order.strategy_used,
                execution_id=f"{order.execution_id}-TWAP-slice-{i+1}"
            )
            
            try:
                broker_result = self._execute_simple_ioc_limit(slice_order, market_bid_at_submission, market_ask_at_submission)
                logger.debug(f"TWAP slice {i+1} for {order.symbol} executed: {broker_result.order.status.value}")
                
                # Aggregate filled quantity and average price for the parent order
                order.filled_qty = (order.filled_qty or 0) + (broker_result.order.filled_qty or 0)
                if broker_result.order.filled_qty and broker_result.order.fill_price:
                    if order.filled_qty > 0:
                        order.fill_price = ((order.fill_price or 0) * (order.filled_qty - broker_result.order.filled_qty) + 
                                            (broker_result.order.fill_price * broker_result.order.filled_qty)) / order.filled_qty
                    else:
                        order.fill_price = broker_result.order.fill_price

            except Exception as e:
                logger.warning(f"TWAP slice {i+1} failed for {order.symbol}: {e}")
            
            if i < slice_count - 1:
                time.sleep(interval_seconds) # Wait for the next slice

        order.update_status(OrderStatus.FILLED if (order.filled_qty or 0) > 0 else OrderStatus.FAILED)
        logger.info(f"TWAP for {order.symbol} completed. Filled {order.filled_qty or 0}/{total_qty} shares.")

    def _execute_vwap(self, order: OrderRecord, market_bid_at_submission: float, market_ask_at_submission: float, lookback_hours: int = 2):
        """
        Executes a Volume-Weighted Average Price (VWAP) strategy.
        This is a more complex strategy and will be a basic placeholder for now.
        It typically requires historical intraday volume profiles.
        """
        logger.info(f"Simulating VWAP for {order.qty} {order.symbol}. This is a basic placeholder.")
        # For a true VWAP, you would need to:
        # 1. Fetch historical intraday volume profile for the lookback_hours.
        # 2. Distribute the order slices proportional to the volume profile.
        # 3. Execute slices over time, adapting to real-time volume.
        
        # For now, it will simply execute as a TWAP over 30 minutes, 5 slices.
        self._execute_twap(order, market_bid_at_submission, market_ask_at_submission, duration_minutes=30, slice_count=5)
        logger.info(f"Basic VWAP simulation for {order.symbol} completed.")


# Singleton instance
_order_manager: Optional[OrderManager] = None

def get_order_manager(default_strategy: str = "LIMIT") -> OrderManager:
    """Get or create singleton OrderManager."""
    global _order_manager
    if _order_manager is None:
        _order_manager = OrderManager(default_strategy=default_strategy)
    return _order_manager

