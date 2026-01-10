"""
Paper Trading Broker for Kobe Trading System.

Simulates order execution without real money for testing.
Uses realistic slippage and fill models.

This broker is useful for:
- Testing strategies before live trading
- Development and debugging
- Demo mode
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from execution.broker_base import (
    BrokerBase,
    BrokerType,
    Quote,
    Position,
    Account,
    Order,
    OrderResult,
    OrderSide,
    OrderType,
    BrokerOrderStatus,
)
from execution.broker_factory import register_broker
from safety.execution_choke import evaluate_safety_gates

logger = logging.getLogger(__name__)


class PaperBroker(BrokerBase):
    """
    Paper trading broker for testing.

    Simulates order execution with configurable fill probability and slippage.
    Maintains virtual positions and P&L.
    """

    def __init__(
        self,
        initial_equity: float = 100000.0,
        fill_probability: float = 1.0,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.0,
        min_commission: float = 0.0,
    ):
        """
        Initialize paper broker.

        Args:
            initial_equity: Starting account equity
            fill_probability: Probability of order fill (0.0 to 1.0)
            slippage_bps: Slippage in basis points
            commission_per_share: Commission per share
            min_commission: Minimum commission per order
        """
        self.initial_equity = initial_equity
        self.fill_probability = fill_probability
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

        # Account state
        self._equity = initial_equity
        self._cash = initial_equity
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._connected = False

        # Simulated market data (can be overridden)
        self._quotes: Dict[str, Quote] = {}

    # === Properties ===

    @property
    def broker_type(self) -> BrokerType:
        return BrokerType.PAPER

    @property
    def name(self) -> str:
        return "Paper"

    @property
    def supports_extended_hours(self) -> bool:
        return True

    @property
    def is_24_7(self) -> bool:
        return True  # Paper can trade anytime

    # === Connection ===

    def connect(self) -> bool:
        self._connected = True
        logger.info(f"Paper broker connected with ${self.initial_equity:,.2f} equity")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("Paper broker disconnected")

    def is_connected(self) -> bool:
        return self._connected

    # === Market Data ===

    def set_quote(self, symbol: str, bid: float, ask: float, volume: int = 1000000) -> None:
        """Set simulated quote for a symbol (for testing)."""
        self._quotes[symbol.upper()] = Quote(
            symbol=symbol.upper(),
            bid=bid,
            ask=ask,
            bid_size=1000,
            ask_size=1000,
            last=(bid + ask) / 2,
            volume=volume,
            timestamp=datetime.now(),
        )

    def get_quote(self, symbol: str, timeout: int = 5) -> Optional[Quote]:
        symbol = symbol.upper()

        # Return cached quote if available
        if symbol in self._quotes:
            return self._quotes[symbol]

        # Try to get real quote from Polygon or return simulated
        try:
            from data.providers.polygon_eod import get_latest_price
            price = get_latest_price(symbol)
            if price:
                # Simulate spread
                spread = price * 0.001  # 0.1% spread
                return Quote(
                    symbol=symbol,
                    bid=price - spread / 2,
                    ask=price + spread / 2,
                    bid_size=1000,
                    ask_size=1000,
                    last=price,
                    volume=1000000,
                    timestamp=datetime.now(),
                )
        except Exception:
            pass

        # Default simulated quote
        return Quote(
            symbol=symbol,
            bid=100.0,
            ask=100.05,
            bid_size=1000,
            ask_size=1000,
            last=100.025,
            volume=1000000,
            timestamp=datetime.now(),
        )

    def get_quotes(self, symbols: List[str], timeout: int = 10) -> Dict[str, Quote]:
        return {s: self.get_quote(s, timeout) for s in symbols if self.get_quote(s)}

    def is_market_open(self) -> bool:
        return True  # Always open for paper trading

    # === Account & Positions ===

    def get_account(self) -> Optional[Account]:
        # Recalculate equity from positions
        self._update_equity()

        return Account(
            account_id="PAPER",
            equity=self._equity,
            cash=self._cash,
            buying_power=self._cash,
            currency="USD",
            pattern_day_trader=False,
            trading_blocked=False,
            transfers_blocked=False,
            account_blocked=False,
            portfolio_value=self._equity,
        )

    def get_positions(self) -> List[Position]:
        self._update_positions()
        return list(self._positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        self._update_positions()
        return self._positions.get(symbol.upper())

    def _update_equity(self) -> None:
        """Recalculate equity from cash and positions."""
        self._update_positions()
        position_value = sum(p.market_value for p in self._positions.values())
        self._equity = self._cash + position_value

    def _update_positions(self) -> None:
        """Update position values with current prices."""
        for symbol, pos in list(self._positions.items()):
            quote = self.get_quote(symbol)
            if quote and quote.last:
                pos.current_price = quote.last
                pos.market_value = pos.qty * pos.current_price
                if pos.side == "long":
                    pos.unrealized_pnl = pos.qty * (pos.current_price - pos.avg_price)
                else:
                    pos.unrealized_pnl = pos.qty * (pos.avg_price - pos.current_price)

                if pos.avg_price > 0:
                    pos.unrealized_pnl_pct = (pos.unrealized_pnl / (pos.avg_price * abs(pos.qty))) * 100

    # === Orders ===

    def place_order(self, order: Order, ack_token: str = None) -> OrderResult:
        """
        Place a simulated order with safety gate enforcement.

        Args:
            order: The order to place
            ack_token: Runtime acknowledgment token (for consistency with live brokers)

        Returns:
            OrderResult with simulated execution details
        """
        # === PAPER MODE GUARD - MUST BE FIRST (defense in depth) ===
        from safety.paper_guard import ensure_paper_mode_or_die
        ensure_paper_mode_or_die(context=f"PaperBroker.place_order:{order.symbol}")

        import random

        # UNIFIED SAFETY GATE CHECK - Paper broker is always paper mode
        gate_result = evaluate_safety_gates(
            is_paper_order=True,  # Paper broker is always paper
            ack_token=ack_token,
            context=f"paper_place_order:{order.symbol}"
        )

        if not gate_result.allowed:
            logger.warning(
                f"Safety gate blocked paper order for {order.symbol}: {gate_result.reason}"
            )
            return OrderResult(
                success=False,
                broker_order_id=None,
                status=BrokerOrderStatus.REJECTED,
                filled_qty=0,
                fill_price=None,
                error_message=f"safety_gate_blocked: {gate_result.reason}",
            )

        symbol = order.symbol.upper()

        # Get current quote
        quote = self.get_quote(symbol)
        if quote is None or quote.bid is None:
            return OrderResult(
                success=False,
                broker_order_id=None,
                status=BrokerOrderStatus.REJECTED,
                filled_qty=0,
                fill_price=None,
                error_message="no_quote_available",
            )

        # Simulate fill probability
        if random.random() > self.fill_probability:
            return OrderResult(
                success=False,
                broker_order_id=None,
                status=BrokerOrderStatus.CANCELLED,
                filled_qty=0,
                fill_price=None,
                error_message="simulated_no_fill",
                market_bid_at_execution=quote.bid,
                market_ask_at_execution=quote.ask,
            )

        # Generate order ID
        broker_order_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"

        # Determine fill price
        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                base_price = quote.ask
            else:
                base_price = quote.bid
        else:
            base_price = order.limit_price or quote.mid

        # Apply slippage
        slippage = base_price * (self.slippage_bps / 10000)
        if order.side == OrderSide.BUY:
            fill_price = base_price + slippage
        else:
            fill_price = base_price - slippage

        fill_price = round(fill_price, 2)

        # Calculate commission
        commission = max(
            self.min_commission,
            order.qty * self.commission_per_share,
        )

        # Calculate notional
        notional = fill_price * order.qty

        # Check if we have enough cash for buys
        if order.side == OrderSide.BUY:
            if notional + commission > self._cash:
                return OrderResult(
                    success=False,
                    broker_order_id=None,
                    status=BrokerOrderStatus.REJECTED,
                    filled_qty=0,
                    fill_price=None,
                    error_message="insufficient_buying_power",
                    market_bid_at_execution=quote.bid,
                    market_ask_at_execution=quote.ask,
                )

        # Execute order
        if order.side == OrderSide.BUY:
            self._cash -= (notional + commission)
            self._add_to_position(symbol, order.qty, fill_price)
        else:
            self._cash += (notional - commission)
            self._reduce_position(symbol, order.qty, fill_price)

        # Store order
        self._orders[broker_order_id] = {
            "id": broker_order_id,
            "client_order_id": order.client_order_id,
            "symbol": symbol,
            "side": order.side.value,
            "qty": order.qty,
            "filled_qty": order.qty,
            "fill_price": fill_price,
            "status": "filled",
            "commission": commission,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Paper order filled: {order.side.value} {order.qty} {symbol} @ ${fill_price:.2f}"
        )

        return OrderResult(
            success=True,
            broker_order_id=broker_order_id,
            client_order_id=order.client_order_id,
            status=BrokerOrderStatus.FILLED,
            filled_qty=order.qty,
            fill_price=fill_price,
            error_message=None,
            market_bid_at_execution=quote.bid,
            market_ask_at_execution=quote.ask,
        )

    def _add_to_position(self, symbol: str, qty: int, price: float) -> None:
        """Add shares to position (or create new)."""
        symbol = symbol.upper()

        if symbol in self._positions:
            pos = self._positions[symbol]
            # Average in
            total_qty = pos.qty + qty
            total_cost = (pos.avg_price * pos.qty) + (price * qty)
            pos.qty = total_qty
            pos.avg_price = total_cost / total_qty
            pos.current_price = price
            pos.market_value = pos.qty * price
            pos.cost_basis = total_cost
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                qty=qty,
                avg_price=price,
                current_price=price,
                market_value=qty * price,
                unrealized_pnl=0.0,
                side="long",
                cost_basis=qty * price,
            )

    def _reduce_position(self, symbol: str, qty: int, price: float) -> None:
        """Reduce position (or close)."""
        symbol = symbol.upper()

        if symbol in self._positions:
            pos = self._positions[symbol]
            pos.qty -= qty

            if pos.qty <= 0:
                # Position closed
                del self._positions[symbol]
            else:
                pos.market_value = pos.qty * price
                pos.current_price = price

    def cancel_order(self, broker_order_id: str) -> bool:
        if broker_order_id in self._orders:
            order = self._orders[broker_order_id]
            if order.get("status") == "filled":
                return False  # Can't cancel filled orders

            order["status"] = "cancelled"
            return True
        return False

    def get_order_status(self, broker_order_id: str) -> Optional[BrokerOrderStatus]:
        if broker_order_id in self._orders:
            status = self._orders[broker_order_id].get("status", "")
            mapping = {
                "filled": BrokerOrderStatus.FILLED,
                "cancelled": BrokerOrderStatus.CANCELLED,
                "rejected": BrokerOrderStatus.REJECTED,
            }
            return mapping.get(status, BrokerOrderStatus.PENDING)
        return None

    def get_orders(
        self,
        status: str = "all",
        limit: int = 100,
        after: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        orders = list(self._orders.values())[-limit:]

        if status != "all":
            orders = [o for o in orders if o.get("status") == status]

        return orders

    # === Paper-specific methods ===

    def reset(self) -> None:
        """Reset broker to initial state."""
        self._equity = self.initial_equity
        self._cash = self.initial_equity
        self._positions.clear()
        self._orders.clear()
        self._quotes.clear()
        logger.info("Paper broker reset")

    def get_pnl(self) -> Dict[str, float]:
        """Get P&L summary."""
        self._update_equity()

        for order in self._orders.values():
            if order.get("status") == "filled":
                side = order.get("side")
                order.get("fill_price", 0)
                order.get("qty", 0)

                if side == "sell" and order.get("symbol") not in self._positions:
                    # This was a closing trade
                    # Would need to track entry prices properly
                    pass

        unrealized_pnl = sum(p.unrealized_pnl for p in self._positions.values())
        total_pnl = self._equity - self.initial_equity

        return {
            "total_pnl": total_pnl,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": total_pnl - unrealized_pnl,
            "initial_equity": self.initial_equity,
            "current_equity": self._equity,
            "return_pct": (total_pnl / self.initial_equity) * 100,
        }


# Auto-register
register_broker("paper", PaperBroker)
