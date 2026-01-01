"""
Order State Machine for Kobe Trading System.

Implements a formal state machine for order lifecycle management with:
- Valid state transitions
- Partial fill handling with replacement orders
- Full audit trail of state changes

This is a SAFETY-CRITICAL component for real money trading.

State Diagram:
    PENDING -> SUBMITTED -> PARTIALLY_FILLED -> FILLED -> CLOSED
                   |               |                |
                   v               v                v
              REJECTED        REPLACED         CANCELLED
                   |               |                |
                   v               v                v
                CLOSED         SUBMITTED         CLOSED
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import uuid
import yaml

logger = logging.getLogger(__name__)


class OrderState(str, Enum):
    """Order lifecycle states."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    REPLACED = "REPLACED"
    EXPIRED = "EXPIRED"
    CLOSED = "CLOSED"


# Valid state transitions
VALID_TRANSITIONS: Dict[OrderState, set] = {
    OrderState.PENDING: {OrderState.SUBMITTED, OrderState.CANCELLED, OrderState.REJECTED},
    OrderState.SUBMITTED: {
        OrderState.PARTIALLY_FILLED,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
    },
    OrderState.PARTIALLY_FILLED: {
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REPLACED,
        OrderState.EXPIRED,
    },
    OrderState.FILLED: {OrderState.CLOSED},
    OrderState.CANCELLED: {OrderState.CLOSED},
    OrderState.REJECTED: {OrderState.CLOSED},
    OrderState.REPLACED: {OrderState.SUBMITTED},  # New order created
    OrderState.EXPIRED: {OrderState.CLOSED},
    OrderState.CLOSED: set(),  # Terminal state
}


@dataclass
class OrderStateTransition:
    """Record of a state transition."""
    from_state: OrderState
    to_state: OrderState
    timestamp: datetime
    reason: str
    filled_qty_delta: int = 0
    fill_price: Optional[float] = None
    broker_order_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "filled_qty_delta": self.filled_qty_delta,
            "fill_price": self.fill_price,
            "broker_order_id": self.broker_order_id,
        }


@dataclass
class ManagedOrder:
    """An order managed by the state machine."""
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    original_qty: int
    limit_price: float
    state: OrderState = OrderState.PENDING
    filled_qty: int = 0
    avg_fill_price: Optional[float] = None
    broker_order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None  # For replaced orders
    replace_count: int = 0
    transitions: List[OrderStateTransition] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_update: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_qty(self) -> int:
        return self.original_qty - self.filled_qty

    @property
    def fill_pct(self) -> float:
        if self.original_qty == 0:
            return 0.0
        return (self.filled_qty / self.original_qty) * 100

    @property
    def is_terminal(self) -> bool:
        return self.state in {
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.CLOSED,
        }

    @property
    def is_active(self) -> bool:
        return self.state in {
            OrderState.PENDING,
            OrderState.SUBMITTED,
            OrderState.PARTIALLY_FILLED,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "original_qty": self.original_qty,
            "limit_price": self.limit_price,
            "state": self.state.value,
            "filled_qty": self.filled_qty,
            "remaining_qty": self.remaining_qty,
            "fill_pct": round(self.fill_pct, 2),
            "avg_fill_price": self.avg_fill_price,
            "broker_order_id": self.broker_order_id,
            "client_order_id": self.client_order_id,
            "parent_order_id": self.parent_order_id,
            "replace_count": self.replace_count,
            "created_at": self.created_at.isoformat(),
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "transitions": [t.to_dict() for t in self.transitions],
        }


@dataclass
class OrderStateMachineConfig:
    """Configuration for order state machine."""
    partial_fill_threshold: float = 0.5   # Accept partial if >= 50% filled
    max_replace_attempts: int = 2         # Max replacement orders
    replace_price_adjustment_pct: float = 0.001  # 0.1% more aggressive on replace
    timeout_seconds: float = 60.0         # Consider stale after this
    log_transitions: bool = True          # Log all state transitions


class OrderStateMachine:
    """
    Manages order lifecycle with formal state transitions.

    Key features:
    - Validates all state transitions
    - Handles partial fills with replacement orders
    - Maintains full audit trail
    - Thread-safe order tracking

    Usage:
        sm = OrderStateMachine()
        order = sm.create_order("AAPL", "buy", 100, 150.0)
        sm.transition(order.order_id, OrderState.SUBMITTED, "initial_submit")
        # ... broker returns fill ...
        sm.handle_fill(order.order_id, filled_qty=50, fill_price=149.95)
    """

    def __init__(
        self,
        config: Optional[OrderStateMachineConfig] = None,
    ):
        self.config = config or OrderStateMachineConfig()
        self._orders: Dict[str, ManagedOrder] = {}
        self._by_broker_id: Dict[str, str] = {}  # broker_order_id -> order_id

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> 'OrderStateMachine':
        """Create OrderStateMachine from config file."""
        if config_path is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "base.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            return cls()

        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)

            osm_cfg = cfg.get('order_state_machine', {})

            config = OrderStateMachineConfig(
                partial_fill_threshold=osm_cfg.get('partial_fill_threshold', 0.5),
                max_replace_attempts=osm_cfg.get('max_replace_attempts', 2),
                replace_price_adjustment_pct=osm_cfg.get('replace_price_adjustment_pct', 0.001),
                timeout_seconds=osm_cfg.get('timeout_seconds', 60.0),
                log_transitions=osm_cfg.get('log_transitions', True),
            )

            return cls(config=config)

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return cls()

    def create_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        limit_price: float,
        client_order_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ManagedOrder:
        """
        Create a new managed order.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            qty: Quantity
            limit_price: Limit price
            client_order_id: Optional client-specified order ID
            metadata: Optional metadata

        Returns:
            ManagedOrder in PENDING state
        """
        order_id = f"OSM-{uuid.uuid4().hex[:12].upper()}"

        order = ManagedOrder(
            order_id=order_id,
            symbol=symbol.upper(),
            side=side.lower(),
            original_qty=qty,
            limit_price=limit_price,
            client_order_id=client_order_id,
            metadata=metadata or {},
        )

        self._orders[order_id] = order

        if self.config.log_transitions:
            logger.info(
                f"Created order {order_id}: {side} {qty} {symbol} @ {limit_price}"
            )

        return order

    def transition(
        self,
        order_id: str,
        to_state: OrderState,
        reason: str,
        filled_qty_delta: int = 0,
        fill_price: Optional[float] = None,
        broker_order_id: Optional[str] = None,
    ) -> ManagedOrder:
        """
        Execute a state transition.

        Args:
            order_id: Order ID
            to_state: Target state
            reason: Reason for transition
            filled_qty_delta: Quantity filled in this transition
            fill_price: Fill price (if applicable)
            broker_order_id: Broker's order ID (if applicable)

        Returns:
            Updated ManagedOrder

        Raises:
            ValueError: If order not found
            InvalidTransitionError: If transition is not valid
        """
        if order_id not in self._orders:
            raise ValueError(f"Order {order_id} not found")

        order = self._orders[order_id]
        from_state = order.state

        # Validate transition
        if to_state not in VALID_TRANSITIONS.get(from_state, set()):
            raise InvalidTransitionError(
                f"Invalid transition: {from_state.value} -> {to_state.value}"
            )

        # Create transition record
        transition = OrderStateTransition(
            from_state=from_state,
            to_state=to_state,
            timestamp=datetime.now(),
            reason=reason,
            filled_qty_delta=filled_qty_delta,
            fill_price=fill_price,
            broker_order_id=broker_order_id,
        )

        # Update order
        order.state = to_state
        order.transitions.append(transition)
        order.last_update = datetime.now()

        if broker_order_id:
            order.broker_order_id = broker_order_id
            self._by_broker_id[broker_order_id] = order_id

        if filled_qty_delta > 0:
            order.filled_qty += filled_qty_delta
            # Update average fill price
            if fill_price:
                if order.avg_fill_price is None:
                    order.avg_fill_price = fill_price
                else:
                    # Weighted average
                    old_qty = order.filled_qty - filled_qty_delta
                    total_qty = order.filled_qty
                    order.avg_fill_price = (
                        (order.avg_fill_price * old_qty + fill_price * filled_qty_delta)
                        / total_qty
                    )

        if self.config.log_transitions:
            logger.info(
                f"Order {order_id}: {from_state.value} -> {to_state.value} ({reason})"
            )

        return order

    def handle_fill(
        self,
        order_id: str,
        filled_qty: int,
        fill_price: float,
        broker_order_id: Optional[str] = None,
    ) -> Tuple[ManagedOrder, Optional[ManagedOrder]]:
        """
        Handle a fill event from broker.

        Determines whether fill is complete, partial (accept), or partial (replace).

        Args:
            order_id: Order ID
            filled_qty: Total filled quantity (cumulative)
            fill_price: Average fill price
            broker_order_id: Broker's order ID

        Returns:
            Tuple of (updated_order, replacement_order or None)
        """
        if order_id not in self._orders:
            raise ValueError(f"Order {order_id} not found")

        order = self._orders[order_id]

        # Calculate delta from last known fill
        qty_delta = filled_qty - order.filled_qty
        if qty_delta <= 0:
            # No new fills
            return order, None

        # Full fill?
        if filled_qty >= order.original_qty:
            self.transition(
                order_id,
                OrderState.FILLED,
                "full_fill",
                filled_qty_delta=qty_delta,
                fill_price=fill_price,
                broker_order_id=broker_order_id,
            )
            return order, None

        # Partial fill
        fill_pct = filled_qty / order.original_qty

        # Update to partially filled state
        if order.state == OrderState.SUBMITTED:
            self.transition(
                order_id,
                OrderState.PARTIALLY_FILLED,
                f"partial_fill_{fill_pct:.0%}",
                filled_qty_delta=qty_delta,
                fill_price=fill_price,
                broker_order_id=broker_order_id,
            )

        # Decide: accept partial or replace?
        if fill_pct >= self.config.partial_fill_threshold:
            # Accept partial fill, don't replace
            self.transition(
                order_id,
                OrderState.CANCELLED,
                f"partial_accepted_{fill_pct:.0%}",
            )
            return order, None

        # Check if we can replace
        if order.replace_count >= self.config.max_replace_attempts:
            # Max retries reached, accept what we have
            self.transition(
                order_id,
                OrderState.CANCELLED,
                "max_replace_attempts_reached",
            )
            return order, None

        # Create replacement order
        replacement = self._create_replacement_order(order)
        return order, replacement

    def _create_replacement_order(
        self,
        original: ManagedOrder,
    ) -> ManagedOrder:
        """
        Create a replacement order for remaining quantity.

        Args:
            original: Original order with partial fill

        Returns:
            New ManagedOrder for remaining qty
        """
        # Transition original to REPLACED
        self.transition(
            original.order_id,
            OrderState.REPLACED,
            "creating_replacement",
        )

        # Calculate new price (more aggressive)
        if original.side == "buy":
            # Increase limit for buys
            new_price = original.limit_price * (1 + self.config.replace_price_adjustment_pct)
        else:
            # Decrease limit for sells
            new_price = original.limit_price * (1 - self.config.replace_price_adjustment_pct)

        # Create new order
        replacement = self.create_order(
            symbol=original.symbol,
            side=original.side,
            qty=original.remaining_qty,
            limit_price=round(new_price, 2),
            client_order_id=f"{original.client_order_id or original.order_id}_R{original.replace_count + 1}",
            metadata={
                **original.metadata,
                "is_replacement": True,
                "parent_order_id": original.order_id,
            },
        )

        replacement.parent_order_id = original.order_id
        replacement.replace_count = original.replace_count + 1

        logger.info(
            f"Created replacement order {replacement.order_id} for {original.order_id}: "
            f"{replacement.remaining_qty} shares @ {replacement.limit_price}"
        )

        return replacement

    def handle_rejection(
        self,
        order_id: str,
        reason: str,
        broker_order_id: Optional[str] = None,
    ) -> ManagedOrder:
        """Handle order rejection from broker."""
        return self.transition(
            order_id,
            OrderState.REJECTED,
            f"rejected:{reason}",
            broker_order_id=broker_order_id,
        )

    def handle_cancellation(
        self,
        order_id: str,
        reason: str = "user_cancelled",
    ) -> ManagedOrder:
        """Handle order cancellation."""
        order = self._orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        if order.state == OrderState.FILLED:
            raise InvalidTransitionError("Cannot cancel a filled order")

        return self.transition(order_id, OrderState.CANCELLED, reason)

    def handle_expiration(
        self,
        order_id: str,
    ) -> ManagedOrder:
        """Handle order expiration (e.g., IOC not filled)."""
        return self.transition(order_id, OrderState.EXPIRED, "order_expired")

    def close_order(
        self,
        order_id: str,
    ) -> ManagedOrder:
        """Move terminal order to CLOSED state."""
        order = self._orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        if order.state not in {
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        }:
            raise InvalidTransitionError(
                f"Cannot close order in state {order.state.value}"
            )

        return self.transition(order_id, OrderState.CLOSED, "order_closed")

    def get_order(self, order_id: str) -> Optional[ManagedOrder]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_order_by_broker_id(self, broker_order_id: str) -> Optional[ManagedOrder]:
        """Get order by broker's order ID."""
        order_id = self._by_broker_id.get(broker_order_id)
        if order_id:
            return self._orders.get(order_id)
        return None

    def get_active_orders(self) -> List[ManagedOrder]:
        """Get all active (non-terminal) orders."""
        return [o for o in self._orders.values() if o.is_active]

    def get_order_history(self, order_id: str) -> List[OrderStateTransition]:
        """Get transition history for an order."""
        order = self._orders.get(order_id)
        if order:
            return order.transitions
        return []

    def get_status(self) -> Dict[str, Any]:
        """Get state machine status."""
        orders_by_state = {}
        for order in self._orders.values():
            state = order.state.value
            orders_by_state[state] = orders_by_state.get(state, 0) + 1

        return {
            "total_orders": len(self._orders),
            "active_orders": len(self.get_active_orders()),
            "orders_by_state": orders_by_state,
            "config": {
                "partial_fill_threshold": self.config.partial_fill_threshold,
                "max_replace_attempts": self.config.max_replace_attempts,
            },
        }


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass
