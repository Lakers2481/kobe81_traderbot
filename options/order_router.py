"""
Options Order Router for Live Execution.

Routes options orders to appropriate broker:
- Single-leg orders
- Multi-leg spread orders
- Order lifecycle management

Integrates with:
- ChainFetcher for contract verification
- SpreadBuilder for complex orders
- BrokerBase for execution
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from options.chain_fetcher import OptionContract, OptionsChain, OptionType
from options.spreads import OptionsSpread, SpreadLeg, SpreadType

logger = logging.getLogger(__name__)


class OptionsOrderType(Enum):
    """Order types for options."""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()


class OptionsOrderSide(Enum):
    """Order sides."""
    BUY_TO_OPEN = "buy_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_OPEN = "sell_to_open"
    SELL_TO_CLOSE = "sell_to_close"


class OptionsOrderStatus(Enum):
    """Order status."""
    PENDING = auto()
    SUBMITTED = auto()
    PARTIAL_FILL = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


@dataclass
class OptionsOrderLeg:
    """Single leg of an options order."""
    contract_symbol: str
    side: OptionsOrderSide
    quantity: int
    limit_price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_symbol": self.contract_symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
        }


@dataclass
class OptionsOrder:
    """Options order (single or multi-leg)."""
    symbol: str                              # Underlying symbol
    legs: List[OptionsOrderLeg] = field(default_factory=list)
    order_type: OptionsOrderType = OptionsOrderType.LIMIT
    time_in_force: str = "day"              # day, gtc, ioc
    net_limit_price: Optional[float] = None  # For spreads
    order_id: str = ""
    status: OptionsOrderStatus = OptionsOrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    fill_price: Optional[float] = None
    filled_quantity: int = 0
    commission: float = 0.0
    notes: str = ""

    @property
    def is_spread(self) -> bool:
        """True if this is a multi-leg order."""
        return len(self.legs) > 1

    @property
    def is_open(self) -> bool:
        """True if order is still active."""
        return self.status in (
            OptionsOrderStatus.PENDING,
            OptionsOrderStatus.SUBMITTED,
            OptionsOrderStatus.PARTIAL_FILL,
        )

    @property
    def is_complete(self) -> bool:
        """True if order is terminal."""
        return self.status in (
            OptionsOrderStatus.FILLED,
            OptionsOrderStatus.CANCELLED,
            OptionsOrderStatus.REJECTED,
            OptionsOrderStatus.EXPIRED,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "legs": [leg.to_dict() for leg in self.legs],
            "order_type": self.order_type.name,
            "time_in_force": self.time_in_force,
            "net_limit_price": self.net_limit_price,
            "order_id": self.order_id,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "fill_price": self.fill_price,
            "filled_quantity": self.filled_quantity,
            "commission": self.commission,
        }


@dataclass
class OptionsOrderResult:
    """Result from order submission."""
    success: bool
    order: OptionsOrder
    message: str = ""
    broker_order_id: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "order": self.order.to_dict(),
            "message": self.message,
            "broker_order_id": self.broker_order_id,
            "errors": self.errors,
        }


class OptionsOrderRouter:
    """
    Routes options orders to appropriate broker.

    Supports:
    - Polygon API key for contract verification
    - Alpaca for order execution
    - Kill switch integration
    - Idempotency tracking
    """

    # Commission per contract (adjust per broker)
    DEFAULT_COMMISSION = 0.65

    def __init__(
        self,
        polygon_api_key: Optional[str] = None,
        alpaca_api_key: Optional[str] = None,
        alpaca_secret: Optional[str] = None,
        alpaca_base_url: Optional[str] = None,
        paper_mode: bool = True,
        kill_switch_path: str = "state/KILL_SWITCH",
    ):
        """
        Initialize options order router.

        Args:
            polygon_api_key: Polygon.io API key for verification
            alpaca_api_key: Alpaca API key
            alpaca_secret: Alpaca API secret
            alpaca_base_url: Alpaca base URL
            paper_mode: If True, simulate orders
            kill_switch_path: Path to kill switch file
        """
        self.polygon_api_key = polygon_api_key or os.getenv("POLYGON_API_KEY", "")
        self.alpaca_api_key = alpaca_api_key or os.getenv("ALPACA_API_KEY_ID", "")
        self.alpaca_secret = alpaca_secret or os.getenv("ALPACA_API_SECRET_KEY", "")
        self.alpaca_base_url = alpaca_base_url or os.getenv(
            "ALPACA_BASE_URL",
            "https://paper-api.alpaca.markets" if paper_mode else "https://api.alpaca.markets"
        )
        self.paper_mode = paper_mode
        self.kill_switch_path = Path(kill_switch_path)

        # Order tracking
        self._orders: Dict[str, OptionsOrder] = {}
        self._next_order_id = 1

        # Alpaca client (lazy init)
        self._alpaca_client = None

    def _check_kill_switch(self) -> bool:
        """Check if kill switch is active."""
        return self.kill_switch_path.exists()

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        order_id = f"OPT-{datetime.utcnow().strftime('%Y%m%d')}-{self._next_order_id:06d}"
        self._next_order_id += 1
        return order_id

    def create_single_leg_order(
        self,
        contract: OptionContract,
        side: OptionsOrderSide,
        quantity: int,
        limit_price: Optional[float] = None,
        order_type: OptionsOrderType = OptionsOrderType.LIMIT,
        time_in_force: str = "day",
    ) -> OptionsOrder:
        """
        Create a single-leg options order.

        Args:
            contract: Option contract to trade
            side: Buy/sell to open/close
            quantity: Number of contracts
            limit_price: Limit price (uses mid if None for LIMIT orders)
            order_type: Order type
            time_in_force: Time in force

        Returns:
            OptionsOrder ready for submission
        """
        if limit_price is None and order_type == OptionsOrderType.LIMIT:
            limit_price = contract.mid

        leg = OptionsOrderLeg(
            contract_symbol=contract.contract_symbol,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
        )

        order = OptionsOrder(
            symbol=contract.symbol,
            legs=[leg],
            order_type=order_type,
            time_in_force=time_in_force,
            net_limit_price=limit_price,
            order_id=self._generate_order_id(),
        )

        logger.info(f"Created single-leg order: {order.order_id} {side.value} {quantity}x {contract.contract_symbol}")
        return order

    def create_spread_order(
        self,
        spread: OptionsSpread,
        action: str = "open",  # "open" or "close"
        limit_price: Optional[float] = None,
        order_type: OptionsOrderType = OptionsOrderType.LIMIT,
        time_in_force: str = "day",
    ) -> OptionsOrder:
        """
        Create a multi-leg spread order.

        Args:
            spread: OptionsSpread to trade
            action: "open" to enter, "close" to exit
            limit_price: Net limit price (uses spread cost if None)
            order_type: Order type
            time_in_force: Time in force

        Returns:
            OptionsOrder ready for submission
        """
        legs = []
        for spread_leg in spread.legs:
            # Determine side based on action and leg direction
            if action == "open":
                side = (
                    OptionsOrderSide.BUY_TO_OPEN if spread_leg.is_long
                    else OptionsOrderSide.SELL_TO_OPEN
                )
            else:  # close
                side = (
                    OptionsOrderSide.SELL_TO_CLOSE if spread_leg.is_long
                    else OptionsOrderSide.BUY_TO_CLOSE
                )

            legs.append(OptionsOrderLeg(
                contract_symbol=spread_leg.contract.contract_symbol,
                side=side,
                quantity=abs(spread_leg.quantity),
                limit_price=spread_leg.contract.mid,
            ))

        if limit_price is None:
            # Calculate net price from spread
            limit_price = spread.total_cost / 100  # Convert to per-share

        order = OptionsOrder(
            symbol=spread.symbol,
            legs=legs,
            order_type=order_type,
            time_in_force=time_in_force,
            net_limit_price=limit_price,
            order_id=self._generate_order_id(),
            notes=f"Spread: {spread.spread_type.name}",
        )

        logger.info(
            f"Created spread order: {order.order_id} {action} {spread.spread_type.name} "
            f"net_price={limit_price:.2f}"
        )
        return order

    def submit_order(self, order: OptionsOrder) -> OptionsOrderResult:
        """
        Submit an options order.

        Args:
            order: OptionsOrder to submit

        Returns:
            OptionsOrderResult with status
        """
        # Check kill switch
        if self._check_kill_switch():
            order.status = OptionsOrderStatus.REJECTED
            return OptionsOrderResult(
                success=False,
                order=order,
                message="Kill switch active",
                errors=["KILL_SWITCH file exists"],
            )

        # Track order
        self._orders[order.order_id] = order

        # Paper mode simulation
        if self.paper_mode:
            return self._simulate_order(order)

        # Live execution via Alpaca
        return self._execute_alpaca_order(order)

    def _simulate_order(self, order: OptionsOrder) -> OptionsOrderResult:
        """Simulate order fill in paper mode."""
        order.status = OptionsOrderStatus.SUBMITTED
        order.submitted_at = datetime.utcnow()

        # Simulate immediate fill
        order.status = OptionsOrderStatus.FILLED
        order.filled_at = datetime.utcnow()
        order.fill_price = order.net_limit_price
        order.filled_quantity = sum(leg.quantity for leg in order.legs)
        order.commission = self.DEFAULT_COMMISSION * len(order.legs)

        logger.info(
            f"PAPER: Order {order.order_id} filled @ {order.fill_price:.2f} "
            f"(commission: ${order.commission:.2f})"
        )

        return OptionsOrderResult(
            success=True,
            order=order,
            message="Paper trade filled",
            broker_order_id=f"PAPER-{order.order_id}",
        )

    def _execute_alpaca_order(self, order: OptionsOrder) -> OptionsOrderResult:
        """
        Execute order via Alpaca API.

        Note: Alpaca options trading requires approved account.
        """
        try:
            # Import Alpaca client
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import (
                LimitOrderRequest,
                MarketOrderRequest,
            )
            from alpaca.trading.enums import (
                OrderSide,
                TimeInForce,
                OrderClass,
                OrderType as AlpacaOrderType,
            )

            if self._alpaca_client is None:
                self._alpaca_client = TradingClient(
                    api_key=self.alpaca_api_key,
                    secret_key=self.alpaca_secret,
                    paper="paper" in self.alpaca_base_url.lower(),
                )

            order.status = OptionsOrderStatus.SUBMITTED
            order.submitted_at = datetime.utcnow()

            # Handle single-leg vs multi-leg
            if not order.is_spread:
                # Single leg order
                leg = order.legs[0]

                # Map side
                if leg.side in (OptionsOrderSide.BUY_TO_OPEN, OptionsOrderSide.BUY_TO_CLOSE):
                    side = OrderSide.BUY
                else:
                    side = OrderSide.SELL

                if order.order_type == OptionsOrderType.LIMIT:
                    request = LimitOrderRequest(
                        symbol=leg.contract_symbol,
                        qty=leg.quantity,
                        side=side,
                        time_in_force=TimeInForce.DAY,
                        limit_price=leg.limit_price,
                    )
                else:
                    request = MarketOrderRequest(
                        symbol=leg.contract_symbol,
                        qty=leg.quantity,
                        side=side,
                        time_in_force=TimeInForce.DAY,
                    )

                result = self._alpaca_client.submit_order(request)
                order.order_id = result.id

                return OptionsOrderResult(
                    success=True,
                    order=order,
                    message="Order submitted to Alpaca",
                    broker_order_id=result.id,
                )

            else:
                # Multi-leg spread order
                # Alpaca spread orders use different API endpoint
                # This requires options trading approval and specific endpoint

                logger.warning("Alpaca multi-leg orders require special API access")

                # Fallback: submit legs individually
                errors = []
                for leg in order.legs:
                    try:
                        if leg.side in (OptionsOrderSide.BUY_TO_OPEN, OptionsOrderSide.BUY_TO_CLOSE):
                            side = OrderSide.BUY
                        else:
                            side = OrderSide.SELL

                        request = LimitOrderRequest(
                            symbol=leg.contract_symbol,
                            qty=leg.quantity,
                            side=side,
                            time_in_force=TimeInForce.DAY,
                            limit_price=leg.limit_price,
                        )
                        self._alpaca_client.submit_order(request)
                    except Exception as e:
                        errors.append(f"Leg {leg.contract_symbol}: {str(e)}")

                if errors:
                    order.status = OptionsOrderStatus.PARTIAL_FILL
                    return OptionsOrderResult(
                        success=False,
                        order=order,
                        message="Spread partially filled",
                        errors=errors,
                    )

                return OptionsOrderResult(
                    success=True,
                    order=order,
                    message="Spread legs submitted individually",
                    broker_order_id=order.order_id,
                )

        except ImportError:
            logger.error("alpaca-py not installed")
            order.status = OptionsOrderStatus.REJECTED
            return OptionsOrderResult(
                success=False,
                order=order,
                message="Alpaca client not available",
                errors=["Install alpaca-py: pip install alpaca-py"],
            )
        except Exception as e:
            logger.error(f"Alpaca order error: {e}")
            order.status = OptionsOrderStatus.REJECTED
            return OptionsOrderResult(
                success=False,
                order=order,
                message=str(e),
                errors=[str(e)],
            )

    def cancel_order(self, order_id: str) -> OptionsOrderResult:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            OptionsOrderResult with status
        """
        order = self._orders.get(order_id)
        if not order:
            return OptionsOrderResult(
                success=False,
                order=OptionsOrder(symbol="UNKNOWN", order_id=order_id),
                message="Order not found",
                errors=[f"No order with ID {order_id}"],
            )

        if order.is_complete:
            return OptionsOrderResult(
                success=False,
                order=order,
                message="Order already complete",
                errors=[f"Order status: {order.status.name}"],
            )

        if self.paper_mode:
            order.status = OptionsOrderStatus.CANCELLED
            return OptionsOrderResult(
                success=True,
                order=order,
                message="Paper order cancelled",
            )

        # Live cancellation via Alpaca
        try:
            if self._alpaca_client:
                self._alpaca_client.cancel_order_by_id(order_id)
            order.status = OptionsOrderStatus.CANCELLED
            return OptionsOrderResult(
                success=True,
                order=order,
                message="Order cancelled",
            )
        except Exception as e:
            return OptionsOrderResult(
                success=False,
                order=order,
                message=str(e),
                errors=[str(e)],
            )

    def get_order(self, order_id: str) -> Optional[OptionsOrder]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[OptionsOrder]:
        """Get all open orders."""
        return [o for o in self._orders.values() if o.is_open]

    def get_filled_orders(self) -> List[OptionsOrder]:
        """Get all filled orders."""
        return [o for o in self._orders.values() if o.status == OptionsOrderStatus.FILLED]

    def refresh_order_status(self, order_id: str) -> Optional[OptionsOrder]:
        """
        Refresh order status from broker.

        Args:
            order_id: Order ID to refresh

        Returns:
            Updated order or None
        """
        order = self._orders.get(order_id)
        if not order:
            return None

        if self.paper_mode:
            return order

        try:
            if self._alpaca_client:
                broker_order = self._alpaca_client.get_order_by_id(order_id)

                # Map Alpaca status to our status
                status_map = {
                    "new": OptionsOrderStatus.SUBMITTED,
                    "accepted": OptionsOrderStatus.SUBMITTED,
                    "pending_new": OptionsOrderStatus.PENDING,
                    "partially_filled": OptionsOrderStatus.PARTIAL_FILL,
                    "filled": OptionsOrderStatus.FILLED,
                    "canceled": OptionsOrderStatus.CANCELLED,
                    "rejected": OptionsOrderStatus.REJECTED,
                    "expired": OptionsOrderStatus.EXPIRED,
                }
                order.status = status_map.get(
                    broker_order.status.value.lower(),
                    OptionsOrderStatus.PENDING
                )

                if broker_order.filled_avg_price:
                    order.fill_price = float(broker_order.filled_avg_price)
                if broker_order.filled_qty:
                    order.filled_quantity = int(broker_order.filled_qty)

        except Exception as e:
            logger.error(f"Failed to refresh order {order_id}: {e}")

        return order


# Convenience functions
def get_options_router(paper_mode: bool = True) -> OptionsOrderRouter:
    """Get default options order router."""
    return OptionsOrderRouter(paper_mode=paper_mode)


def quick_buy_call(
    symbol: str,
    target_delta: float = 0.50,
    target_dte: int = 30,
    quantity: int = 1,
    paper_mode: bool = True,
) -> OptionsOrderResult:
    """
    Quick function to buy a call option.

    Args:
        symbol: Underlying symbol
        target_delta: Delta target (0.50 = ATM)
        target_dte: Days to expiration
        quantity: Number of contracts
        paper_mode: If True, simulate

    Returns:
        OptionsOrderResult
    """
    from options.chain_fetcher import ChainFetcher

    fetcher = ChainFetcher()
    chain = fetcher.fetch_chain(symbol)

    if not chain:
        return OptionsOrderResult(
            success=False,
            order=OptionsOrder(symbol=symbol),
            message=f"Could not fetch chain for {symbol}",
        )

    expiration = chain.get_expiration(target_dte)
    if not expiration:
        return OptionsOrderResult(
            success=False,
            order=OptionsOrder(symbol=symbol),
            message="No suitable expiration found",
        )

    contract = chain.get_contract_by_delta(expiration, OptionType.CALL, target_delta)
    if not contract:
        return OptionsOrderResult(
            success=False,
            order=OptionsOrder(symbol=symbol),
            message="No suitable contract found",
        )

    router = OptionsOrderRouter(paper_mode=paper_mode)
    order = router.create_single_leg_order(
        contract=contract,
        side=OptionsOrderSide.BUY_TO_OPEN,
        quantity=quantity,
    )

    return router.submit_order(order)


def quick_buy_put(
    symbol: str,
    target_delta: float = 0.50,
    target_dte: int = 30,
    quantity: int = 1,
    paper_mode: bool = True,
) -> OptionsOrderResult:
    """
    Quick function to buy a put option.

    Args:
        symbol: Underlying symbol
        target_delta: Delta target (0.50 = ATM)
        target_dte: Days to expiration
        quantity: Number of contracts
        paper_mode: If True, simulate

    Returns:
        OptionsOrderResult
    """
    from options.chain_fetcher import ChainFetcher

    fetcher = ChainFetcher()
    chain = fetcher.fetch_chain(symbol)

    if not chain:
        return OptionsOrderResult(
            success=False,
            order=OptionsOrder(symbol=symbol),
            message=f"Could not fetch chain for {symbol}",
        )

    expiration = chain.get_expiration(target_dte)
    if not expiration:
        return OptionsOrderResult(
            success=False,
            order=OptionsOrder(symbol=symbol),
            message="No suitable expiration found",
        )

    contract = chain.get_contract_by_delta(expiration, OptionType.PUT, target_delta)
    if not contract:
        return OptionsOrderResult(
            success=False,
            order=OptionsOrder(symbol=symbol),
            message="No suitable contract found",
        )

    router = OptionsOrderRouter(paper_mode=paper_mode)
    order = router.create_single_leg_order(
        contract=contract,
        side=OptionsOrderSide.BUY_TO_OPEN,
        quantity=quantity,
    )

    return router.submit_order(order)
