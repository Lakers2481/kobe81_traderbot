"""
Broker Abstraction Layer for Kobe Trading System.

Provides a unified interface for order execution, position management,
and market data across different brokers (Alpaca, IBKR, crypto exchanges).

This is a SAFETY-CRITICAL component for real money trading.
All broker implementations MUST implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class BrokerType(Enum):
    """Supported broker types."""
    ALPACA = auto()
    IBKR = auto()
    CRYPTO = auto()
    PAPER = auto()


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    """Time in force options."""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    OPG = "opg"  # At open
    CLS = "cls"  # At close


class BrokerOrderStatus(Enum):
    """Standardized order status across brokers."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Quote:
    """Market quote data."""
    symbol: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    timestamp: Optional[datetime] = None

    @property
    def mid(self) -> Optional[float]:
        """Get mid price."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        """Get spread as percentage of mid."""
        mid = self.mid
        spread = self.spread
        if mid and spread and mid > 0:
            return (spread / mid) * 100
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "last": self.last,
            "volume": self.volume,
            "mid": self.mid,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class Position:
    """Position data from broker."""
    symbol: str
    qty: int
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    side: str  # "long" or "short"
    cost_basis: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "avg_price": self.avg_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "side": self.side,
            "cost_basis": self.cost_basis,
        }


@dataclass
class Account:
    """Account information from broker."""
    account_id: str
    equity: float
    cash: float
    buying_power: float
    currency: str = "USD"
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    transfers_blocked: bool = False
    account_blocked: bool = False
    portfolio_value: Optional[float] = None
    last_equity: Optional[float] = None
    multiplier: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "equity": self.equity,
            "cash": self.cash,
            "buying_power": self.buying_power,
            "currency": self.currency,
            "pattern_day_trader": self.pattern_day_trader,
            "trading_blocked": self.trading_blocked,
            "portfolio_value": self.portfolio_value,
        }


@dataclass
class Order:
    """Order to submit to broker."""
    symbol: str
    side: OrderSide
    qty: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    client_order_id: Optional[str] = None
    extended_hours: bool = False

    # Bracket order fields
    take_profit_limit: Optional[float] = None
    stop_loss_price: Optional[float] = None
    stop_loss_limit: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "qty": self.qty,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "client_order_id": self.client_order_id,
            "extended_hours": self.extended_hours,
            "take_profit_limit": self.take_profit_limit,
            "stop_loss_price": self.stop_loss_price,
        }


@dataclass
class OrderResult:
    """Result of order submission."""
    success: bool
    broker_order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: BrokerOrderStatus = BrokerOrderStatus.PENDING
    filled_qty: int = 0
    fill_price: Optional[float] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    # Market context for TCA
    market_bid_at_execution: Optional[float] = None
    market_ask_at_execution: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "broker_order_id": self.broker_order_id,
            "client_order_id": self.client_order_id,
            "status": self.status.value,
            "filled_qty": self.filled_qty,
            "fill_price": self.fill_price,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "market_bid": self.market_bid_at_execution,
            "market_ask": self.market_ask_at_execution,
            "timestamp": self.timestamp.isoformat(),
        }


class BrokerBase(ABC):
    """
    Abstract base class for all broker implementations.

    All broker implementations MUST inherit from this class and
    implement ALL abstract methods. This ensures a consistent
    interface across different brokers.

    SAFETY: All methods that interact with real money must:
    1. Check for kill switch before execution
    2. Validate inputs
    3. Log all operations
    4. Handle errors gracefully
    """

    # === Properties (must be implemented) ===

    @property
    @abstractmethod
    def broker_type(self) -> BrokerType:
        """Return the broker type."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable broker name."""
        pass

    @property
    @abstractmethod
    def supports_extended_hours(self) -> bool:
        """Whether broker supports extended hours trading."""
        pass

    @property
    @abstractmethod
    def is_24_7(self) -> bool:
        """Whether broker operates 24/7 (e.g., crypto)."""
        pass

    # === Connection Management ===

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to broker.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        pass

    # === Market Data ===

    @abstractmethod
    def get_quote(self, symbol: str, timeout: int = 5) -> Optional[Quote]:
        """
        Get current market quote.

        Args:
            symbol: Symbol to quote
            timeout: Request timeout in seconds

        Returns:
            Quote object or None if unavailable
        """
        pass

    @abstractmethod
    def get_quotes(self, symbols: List[str], timeout: int = 10) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of symbols
            timeout: Request timeout

        Returns:
            Dict mapping symbol to Quote
        """
        pass

    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        pass

    # === Account & Positions ===

    @abstractmethod
    def get_account(self) -> Optional[Account]:
        """Get account information."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get all current positions."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        pass

    # === Orders ===

    @abstractmethod
    def place_order(self, order: Order) -> OrderResult:
        """
        Place an order.

        Args:
            order: Order to place

        Returns:
            OrderResult with execution details
        """
        pass

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            broker_order_id: Broker's order ID

        Returns:
            True if cancellation request successful
        """
        pass

    @abstractmethod
    def get_order_status(self, broker_order_id: str) -> Optional[BrokerOrderStatus]:
        """
        Get order status.

        Args:
            broker_order_id: Broker's order ID

        Returns:
            OrderStatus or None if not found
        """
        pass

    @abstractmethod
    def get_orders(
        self,
        status: str = "all",
        limit: int = 100,
        after: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get orders from broker.

        Args:
            status: Filter by status ("open", "closed", "all")
            limit: Maximum orders to return
            after: Only orders after this time

        Returns:
            List of order dictionaries
        """
        pass

    # === Convenience Methods (non-abstract, using abstract methods) ===

    def get_best_bid(self, symbol: str, timeout: int = 5) -> Optional[float]:
        """Get best bid price."""
        quote = self.get_quote(symbol, timeout)
        return quote.bid if quote else None

    def get_best_ask(self, symbol: str, timeout: int = 5) -> Optional[float]:
        """Get best ask price."""
        quote = self.get_quote(symbol, timeout)
        return quote.ask if quote else None

    def get_mid_price(self, symbol: str, timeout: int = 5) -> Optional[float]:
        """Get mid price."""
        quote = self.get_quote(symbol, timeout)
        return quote.mid if quote else None

    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """
        Convenience method for market orders.

        Args:
            symbol: Symbol to trade
            side: Buy or sell
            qty: Quantity
            client_order_id: Optional client order ID

        Returns:
            OrderResult
        """
        # === PAPER MODE GUARD - Defense in depth ===
        from safety.paper_guard import ensure_paper_mode_or_die
        ensure_paper_mode_or_die(context=f"BaseBroker.place_market_order:{symbol}")

        order = Order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        return self.place_order(order)

    def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        limit_price: float,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """
        Convenience method for limit orders.

        Args:
            symbol: Symbol to trade
            side: Buy or sell
            qty: Quantity
            limit_price: Limit price
            time_in_force: Time in force
            client_order_id: Optional client order ID

        Returns:
            OrderResult
        """
        # === PAPER MODE GUARD - Defense in depth ===
        from safety.paper_guard import ensure_paper_mode_or_die
        ensure_paper_mode_or_die(context=f"BaseBroker.place_limit_order:{symbol}")

        order = Order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )
        return self.place_order(order)

    def place_ioc_limit(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        limit_price: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """
        Convenience method for IOC LIMIT orders.

        Args:
            symbol: Symbol to trade
            side: Buy or sell
            qty: Quantity
            limit_price: Limit price
            client_order_id: Optional client order ID

        Returns:
            OrderResult
        """
        # === PAPER MODE GUARD - Defense in depth ===
        from safety.paper_guard import ensure_paper_mode_or_die
        ensure_paper_mode_or_die(context=f"BaseBroker.place_ioc_limit:{symbol}")

        order = Order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            time_in_force=TimeInForce.IOC,
            client_order_id=client_order_id,
        )
        return self.place_order(order)

    def place_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        limit_price: float,
        take_profit: float,
        stop_loss: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """
        Convenience method for bracket (OCO) orders.

        Args:
            symbol: Symbol to trade
            side: Buy or sell
            qty: Quantity
            limit_price: Entry limit price
            take_profit: Take profit price
            stop_loss: Stop loss price
            client_order_id: Optional client order ID

        Returns:
            OrderResult
        """
        # === PAPER MODE GUARD - Defense in depth ===
        from safety.paper_guard import ensure_paper_mode_or_die
        ensure_paper_mode_or_die(context=f"BaseBroker.place_bracket_order:{symbol}")

        order = Order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
            take_profit_limit=take_profit,
            stop_loss_price=stop_loss,
        )
        return self.place_order(order)

    def get_equity(self) -> float:
        """Get account equity."""
        account = self.get_account()
        return account.equity if account else 0.0

    def get_buying_power(self) -> float:
        """Get buying power."""
        account = self.get_account()
        return account.buying_power if account else 0.0

    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in symbol."""
        pos = self.get_position(symbol)
        return pos is not None and pos.qty != 0
