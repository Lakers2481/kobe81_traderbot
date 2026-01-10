"""
Alpaca Crypto Broker for Kobe Trading System.

Uses Alpaca's native crypto trading API - NO CCXT NEEDED.
Same API keys as your equity trading account.

Supports 24/7 crypto trading with:
- BTC, ETH, and other major pairs
- Real-time quotes
- Order execution with safety gates

Author: Kobe Trading System
Version: 1.0.0
Date: 2026-01-06
"""

from __future__ import annotations

import logging
import os
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
    TimeInForce,
    BrokerOrderStatus,
)
from execution.broker_factory import register_broker
from safety.execution_choke import evaluate_safety_gates

logger = logging.getLogger(__name__)

# Check for alpaca-py
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import (
        OrderSide as AlpacaOrderSide,
        TimeInForce as AlpacaTIF,
        OrderType as AlpacaOrderType,
        AssetClass,
    )
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoLatestQuoteRequest, CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed. Install with: pip install alpaca-py")


class AlpacaCryptoBroker(BrokerBase):
    """
    Alpaca-based cryptocurrency broker.

    Uses same API keys as your equity account.
    Supports 24/7 trading on major crypto pairs.

    Supported pairs (vs USD):
    - BTC/USD, ETH/USD, SOL/USD, DOGE/USD
    - AVAX/USD, LINK/USD, DOT/USD, and more
    """

    # Major crypto pairs supported by Alpaca
    SUPPORTED_PAIRS = [
        "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD",
        "AVAX/USD", "LINK/USD", "DOT/USD", "SHIB/USD",
        "LTC/USD", "UNI/USD", "AAVE/USD", "BCH/USD",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
    ):
        """
        Initialize Alpaca crypto broker.

        Args:
            api_key: Alpaca API key (or from env: ALPACA_API_KEY_ID)
            api_secret: Alpaca API secret (or from env: ALPACA_API_SECRET_KEY)
            paper: Use paper trading mode (default True for safety)
        """
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is required for Alpaca crypto trading. "
                "Install with: pip install alpaca-py"
            )

        self.api_key = api_key or os.getenv("ALPACA_API_KEY_ID", "")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET_KEY", "")
        self.paper = paper

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY_ID and "
                "ALPACA_API_SECRET_KEY environment variables."
            )

        # Trading client
        self._trading_client: Optional[TradingClient] = None

        # Data client (no auth needed for crypto data)
        self._data_client: Optional[CryptoHistoricalDataClient] = None

        self._connected = False

    # === Properties ===

    @property
    def broker_type(self) -> BrokerType:
        return BrokerType.CRYPTO

    @property
    def name(self) -> str:
        mode = "Paper" if self.paper else "Live"
        return f"Alpaca Crypto ({mode})"

    @property
    def supports_extended_hours(self) -> bool:
        return True  # 24/7

    @property
    def is_24_7(self) -> bool:
        return True

    # === Connection ===

    def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            # Initialize trading client
            self._trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.paper,
            )

            # Initialize data client (no auth needed)
            self._data_client = CryptoHistoricalDataClient()

            # Test connection by getting account
            account = self._trading_client.get_account()
            if account:
                self._connected = True
                logger.info(
                    f"Connected to Alpaca Crypto ({'paper' if self.paper else 'live'})"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        self._connected = False
        self._trading_client = None
        self._data_client = None
        logger.info("Disconnected from Alpaca Crypto")

    def is_connected(self) -> bool:
        return self._connected and self._trading_client is not None

    # === Market Data ===

    def get_quote(self, symbol: str, timeout: int = 5) -> Optional[Quote]:
        """Get current crypto quote."""
        try:
            if not self._data_client:
                return None

            # Normalize symbol (BTC -> BTC/USD)
            symbol = self._normalize_symbol(symbol)

            # Get latest quote
            request = CryptoLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self._data_client.get_crypto_latest_quote(request)

            if symbol in quotes:
                q = quotes[symbol]
                return Quote(
                    symbol=symbol,
                    bid=float(q.bid_price) if q.bid_price else None,
                    ask=float(q.ask_price) if q.ask_price else None,
                    bid_size=int(q.bid_size) if q.bid_size else None,
                    ask_size=int(q.ask_size) if q.ask_size else None,
                    last=None,  # Use mid as last
                    volume=None,
                    timestamp=q.timestamp if hasattr(q, 'timestamp') else datetime.now(),
                )

            return None

        except Exception as e:
            logger.debug(f"Quote fetch failed for {symbol}: {e}")
            return None

    def get_quotes(self, symbols: List[str], timeout: int = 10) -> Dict[str, Quote]:
        """Get quotes for multiple crypto pairs."""
        quotes = {}
        for symbol in symbols:
            quote = self.get_quote(symbol, timeout)
            if quote:
                quotes[symbol] = quote
        return quotes

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to Alpaca format (BTC/USD)."""
        symbol = symbol.upper()

        # Already in correct format
        if "/" in symbol:
            return symbol

        # Add /USD suffix
        if symbol in ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "DOT", "LTC"]:
            return f"{symbol}/USD"

        # Remove USDT/USDC suffix and add USD
        for suffix in ["USDT", "USDC"]:
            if symbol.endswith(suffix):
                base = symbol[:-len(suffix)]
                return f"{base}/USD"

        return f"{symbol}/USD"

    def is_market_open(self) -> bool:
        """Crypto is always open."""
        return True

    # === Account & Positions ===

    def get_account(self) -> Optional[Account]:
        """Get account information."""
        try:
            if not self._trading_client:
                return None

            acct = self._trading_client.get_account()

            return Account(
                account_id=acct.id,
                equity=float(acct.equity),
                cash=float(acct.cash),
                buying_power=float(acct.buying_power),
                currency="USD",
                pattern_day_trader=acct.pattern_day_trader,
                trading_blocked=acct.trading_blocked,
                transfers_blocked=acct.transfers_blocked,
                account_blocked=acct.account_blocked,
            )

        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return None

    def get_positions(self) -> List[Position]:
        """Get all crypto positions."""
        try:
            if not self._trading_client:
                return []

            positions = self._trading_client.get_all_positions()

            result = []
            for pos in positions:
                # Filter to crypto only
                if hasattr(pos, 'asset_class') and pos.asset_class == AssetClass.CRYPTO:
                    result.append(Position(
                        symbol=pos.symbol,
                        qty=int(float(pos.qty)),
                        avg_price=float(pos.avg_entry_price),
                        current_price=float(pos.current_price),
                        market_value=float(pos.market_value),
                        unrealized_pnl=float(pos.unrealized_pl),
                        side="long" if float(pos.qty) > 0 else "short",
                        unrealized_pnl_pct=float(pos.unrealized_plpc) * 100 if pos.unrealized_plpc else None,
                    ))

            return result

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        symbol = self._normalize_symbol(symbol)
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol or pos.symbol == symbol.replace("/", ""):
                return pos
        return None

    # === Orders ===

    def place_order(self, order: Order, ack_token: str = None) -> OrderResult:
        """
        Place a crypto order with safety gate enforcement.

        Args:
            order: The order to place
            ack_token: Runtime acknowledgment token for live orders

        Returns:
            OrderResult with success/failure status
        """
        # === PAPER MODE GUARD - MUST BE FIRST ===
        from safety.paper_guard import ensure_paper_mode_or_die
        ensure_paper_mode_or_die(context=f"AlpacaCryptoBroker.place_order:{order.symbol}")

        try:
            # SAFETY GATE CHECK - Required for all order submissions
            gate_result = evaluate_safety_gates(
                is_paper_order=self.paper,
                ack_token=ack_token,
                context=f"alpaca_crypto_place_order:{order.symbol}"
            )

            if not gate_result.allowed:
                logger.warning(
                    f"Safety gate blocked crypto order for {order.symbol}: {gate_result.reason}"
                )
                return OrderResult(
                    success=False,
                    broker_order_id=None,
                    status=BrokerOrderStatus.REJECTED,
                    filled_qty=0,
                    fill_price=None,
                    error_message=f"safety_gate_blocked: {gate_result.reason}",
                )

            if not self._trading_client:
                return OrderResult(
                    success=False,
                    broker_order_id=None,
                    status=BrokerOrderStatus.REJECTED,
                    filled_qty=0,
                    fill_price=None,
                    error_message="Not connected to Alpaca",
                )

            symbol = self._normalize_symbol(order.symbol)

            # Get quote for TCA
            quote = self.get_quote(symbol)

            # Map order side
            alpaca_side = (
                AlpacaOrderSide.BUY if order.side == OrderSide.BUY
                else AlpacaOrderSide.SELL
            )

            # Map time in force
            tif_map = {
                TimeInForce.DAY: AlpacaTIF.DAY,
                TimeInForce.GTC: AlpacaTIF.GTC,
                TimeInForce.IOC: AlpacaTIF.IOC,
                TimeInForce.FOK: AlpacaTIF.FOK,
            }
            alpaca_tif = tif_map.get(order.time_in_force, AlpacaTIF.GTC)

            # Create order request
            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=symbol.replace("/", ""),  # Alpaca uses BTCUSD not BTC/USD
                    qty=order.qty,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                )
            else:
                request = LimitOrderRequest(
                    symbol=symbol.replace("/", ""),
                    qty=order.qty,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    limit_price=order.limit_price,
                )

            # Submit order
            logger.info(
                f"Placing crypto order (mode: {gate_result.mode}): "
                f"{alpaca_side.value} {order.qty} {symbol}"
            )
            result = self._trading_client.submit_order(request)

            return OrderResult(
                success=True,
                broker_order_id=str(result.id),
                client_order_id=order.client_order_id,
                status=self._map_alpaca_status(result.status.value),
                filled_qty=int(float(result.filled_qty)) if result.filled_qty else 0,
                fill_price=float(result.filled_avg_price) if result.filled_avg_price else None,
                error_message=None,
                market_bid_at_execution=quote.bid if quote else None,
                market_ask_at_execution=quote.ask if quote else None,
            )

        except Exception as e:
            logger.error(f"Failed to place crypto order: {e}")
            return OrderResult(
                success=False,
                broker_order_id=None,
                status=BrokerOrderStatus.REJECTED,
                filled_qty=0,
                fill_price=None,
                error_message=str(e),
            )

    def _map_alpaca_status(self, status: str) -> BrokerOrderStatus:
        """Map Alpaca status to BrokerOrderStatus."""
        mapping = {
            "new": BrokerOrderStatus.SUBMITTED,
            "accepted": BrokerOrderStatus.ACCEPTED,
            "pending_new": BrokerOrderStatus.PENDING,
            "partially_filled": BrokerOrderStatus.PARTIALLY_FILLED,
            "filled": BrokerOrderStatus.FILLED,
            "canceled": BrokerOrderStatus.CANCELLED,
            "cancelled": BrokerOrderStatus.CANCELLED,
            "expired": BrokerOrderStatus.EXPIRED,
            "rejected": BrokerOrderStatus.REJECTED,
        }
        return mapping.get(status.lower(), BrokerOrderStatus.PENDING)

    def cancel_order(self, broker_order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancel an order."""
        try:
            if not self._trading_client:
                return False

            self._trading_client.cancel_order_by_id(broker_order_id)
            logger.info(f"Cancelled crypto order {broker_order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {broker_order_id}: {e}")
            return False

    def get_order_status(
        self, broker_order_id: str, symbol: Optional[str] = None
    ) -> Optional[BrokerOrderStatus]:
        """Get status of an order."""
        try:
            if not self._trading_client:
                return None

            order = self._trading_client.get_order_by_id(broker_order_id)
            return self._map_alpaca_status(order.status.value)

        except Exception as e:
            logger.debug(f"Failed to get order status: {e}")
            return None

    def get_orders(
        self,
        status: str = "all",
        limit: int = 100,
        after: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get orders."""
        try:
            if not self._trading_client:
                return []

            # Build request
            request = GetOrdersRequest(
                status=status if status != "all" else None,
                limit=limit,
                after=after,
            )

            orders = self._trading_client.get_orders(request)

            # Filter to crypto only and convert to dict
            result = []
            for order in orders:
                if hasattr(order, 'asset_class') and order.asset_class == AssetClass.CRYPTO:
                    result.append({
                        "id": str(order.id),
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "qty": str(order.qty),
                        "filled_qty": str(order.filled_qty),
                        "status": order.status.value,
                        "created_at": order.created_at.isoformat() if order.created_at else None,
                    })

            return result

        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    # === Crypto-specific methods ===

    def get_available_pairs(self) -> List[str]:
        """Get available trading pairs."""
        return self.SUPPORTED_PAIRS.copy()


# Auto-register if alpaca-py is available
if ALPACA_AVAILABLE:
    register_broker("alpaca_crypto", AlpacaCryptoBroker)
