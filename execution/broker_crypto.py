"""
Crypto Broker for Kobe Trading System.

CCXT-based implementation supporting multiple exchanges:
- Binance
- Coinbase Pro
- Kraken
- And 100+ other exchanges via CCXT

Supports 24/7 trading with crypto-specific features.
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

logger = logging.getLogger(__name__)

# Optional CCXT import
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("CCXT not installed. Install with: pip install ccxt")


class CryptoBroker(BrokerBase):
    """
    CCXT-based cryptocurrency broker.

    Supports 24/7 trading on major exchanges.

    Supported exchanges:
    - binance (default)
    - coinbasepro
    - kraken
    - kucoin
    - ftx (deprecated)
    """

    RECOMMENDED_EXCHANGES = ["binance", "coinbasepro", "kraken", "kucoin"]

    def __init__(
        self,
        exchange: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        password: Optional[str] = None,  # Some exchanges require passphrase
        sandbox: bool = True,
        default_quote_currency: str = "USDT",
        **exchange_kwargs,
    ):
        """
        Initialize crypto broker.

        Args:
            exchange: Exchange name (binance, coinbasepro, kraken, etc.)
            api_key: API key (or from env: {EXCHANGE}_API_KEY)
            api_secret: API secret (or from env: {EXCHANGE}_API_SECRET)
            password: API passphrase if required (Coinbase Pro, KuCoin)
            sandbox: Use sandbox/testnet mode
            default_quote_currency: Default quote currency (USDT, USD, etc.)
            **exchange_kwargs: Additional exchange-specific options
        """
        if not CCXT_AVAILABLE:
            raise ImportError(
                "CCXT is required for crypto trading. "
                "Install with: pip install ccxt"
            )

        self.exchange_name = exchange.lower()
        self.sandbox = sandbox
        self.default_quote_currency = default_quote_currency

        # Load credentials from env if not provided
        env_prefix = exchange.upper()
        self.api_key = api_key or os.getenv(f"{env_prefix}_API_KEY", "")
        self.api_secret = api_secret or os.getenv(f"{env_prefix}_API_SECRET", "")
        self.password = password or os.getenv(f"{env_prefix}_PASSWORD", "")

        if self.exchange_name not in self.RECOMMENDED_EXCHANGES:
            logger.warning(
                f"Exchange '{exchange}' not in recommended list: {self.RECOMMENDED_EXCHANGES}"
            )

        # Get exchange class
        exchange_class = getattr(ccxt, self.exchange_name, None)
        if not exchange_class:
            raise ValueError(f"Unknown CCXT exchange: {exchange}")

        # Build config
        config = {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            **exchange_kwargs,
        }

        if self.password:
            config["password"] = self.password

        # Create exchange instance
        self._exchange: ccxt.Exchange = exchange_class(config)

        # Enable sandbox mode if requested
        if sandbox:
            if hasattr(self._exchange, 'set_sandbox_mode'):
                self._exchange.set_sandbox_mode(True)
            elif hasattr(self._exchange, 'sandbox'):
                self._exchange.sandbox = True

        self._connected = False
        self._markets_loaded = False

    # === Properties ===

    @property
    def broker_type(self) -> BrokerType:
        return BrokerType.CRYPTO

    @property
    def name(self) -> str:
        mode = "Sandbox" if self.sandbox else "Live"
        return f"Crypto ({self.exchange_name.capitalize()} {mode})"

    @property
    def supports_extended_hours(self) -> bool:
        return True  # 24/7

    @property
    def is_24_7(self) -> bool:
        return True

    # === Connection ===

    def connect(self) -> bool:
        try:
            # Load markets
            self._exchange.load_markets()
            self._markets_loaded = True

            # Test connection with balance fetch if we have credentials
            if self.api_key:
                self._exchange.fetch_balance()

            self._connected = True
            logger.info(
                f"Connected to {self.exchange_name} "
                f"({'sandbox' if self.sandbox else 'live'})"
            )
            return True

        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication failed for {self.exchange_name}: {e}")
            return False
        except ccxt.NetworkError as e:
            logger.error(f"Network error connecting to {self.exchange_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_name}: {e}")
            return False

    def disconnect(self) -> None:
        self._connected = False
        logger.info(f"Disconnected from {self.exchange_name}")

    def is_connected(self) -> bool:
        return self._connected

    def _ensure_markets(self) -> bool:
        """Ensure markets are loaded."""
        if not self._markets_loaded:
            try:
                self._exchange.load_markets()
                self._markets_loaded = True
            except Exception as e:
                logger.error(f"Failed to load markets: {e}")
                return False
        return True

    # === Market Data ===

    def get_quote(self, symbol: str, timeout: int = 5) -> Optional[Quote]:
        try:
            if not self._ensure_markets():
                return None

            # Normalize symbol format (e.g., BTC/USDT)
            symbol = self._normalize_symbol(symbol)

            ticker = self._exchange.fetch_ticker(symbol)

            return Quote(
                symbol=symbol,
                bid=ticker.get("bid"),
                ask=ticker.get("ask"),
                bid_size=None,  # Not always available
                ask_size=None,
                last=ticker.get("last"),
                volume=ticker.get("quoteVolume"),
                timestamp=datetime.now(),
            )

        except ccxt.BadSymbol as e:
            logger.debug(f"Invalid symbol {symbol}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Quote fetch failed for {symbol}: {e}")
            return None

    def get_quotes(self, symbols: List[str], timeout: int = 10) -> Dict[str, Quote]:
        quotes = {}
        for symbol in symbols:
            quote = self.get_quote(symbol, timeout)
            if quote:
                quotes[symbol] = quote
        return quotes

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to CCXT format (e.g., BTC/USDT)."""
        symbol = symbol.upper()

        # Already in correct format
        if "/" in symbol:
            return symbol

        # Common crypto pairs - add default quote currency
        if symbol in ["BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "DOT", "AVAX"]:
            return f"{symbol}/{self.default_quote_currency}"

        # Try to find the symbol in markets
        if self._markets_loaded:
            # Check if it's a market ID
            if symbol in self._exchange.markets_by_id:
                return self._exchange.markets_by_id[symbol]["symbol"]

            # Check for USDT pair
            usdt_pair = f"{symbol}/USDT"
            if usdt_pair in self._exchange.symbols:
                return usdt_pair

            # Check for USD pair
            usd_pair = f"{symbol}/USD"
            if usd_pair in self._exchange.symbols:
                return usd_pair

        return f"{symbol}/{self.default_quote_currency}"

    def is_market_open(self) -> bool:
        return True  # Crypto is 24/7

    # === Account & Positions ===

    def get_account(self) -> Optional[Account]:
        try:
            balance = self._exchange.fetch_balance()

            # Sum up quote currency values
            total_value = 0.0
            free_cash = 0.0

            for currency in ["USDT", "USD", "BUSD", "USDC"]:
                if currency in balance.get("total", {}):
                    total_value += float(balance["total"][currency])
                if currency in balance.get("free", {}):
                    free_cash += float(balance["free"][currency])

            return Account(
                account_id=self.exchange_name,
                equity=total_value,
                cash=free_cash,
                buying_power=free_cash,
                currency=self.default_quote_currency,
                pattern_day_trader=False,
                trading_blocked=False,
                transfers_blocked=False,
                account_blocked=False,
            )

        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return None

    def get_positions(self) -> List[Position]:
        try:
            balance = self._exchange.fetch_balance()
            positions = []

            for coin, amount in balance.get("total", {}).items():
                amount = float(amount)

                # Skip dust and quote currencies
                if amount < 0.0001:
                    continue
                if coin in ["USDT", "USD", "BUSD", "USDC"]:
                    continue

                # Get current price
                symbol = f"{coin}/{self.default_quote_currency}"
                try:
                    ticker = self._exchange.fetch_ticker(symbol)
                    price = ticker.get("last", 0)
                except Exception:
                    price = 0

                if price > 0:
                    market_value = amount * price

                    positions.append(Position(
                        symbol=coin,
                        qty=int(amount) if amount > 1 else int(amount * 100000000),  # Handle small amounts
                        avg_price=0,  # Not tracked by most exchanges
                        current_price=price,
                        market_value=market_value,
                        unrealized_pnl=0,  # Would need entry price
                        side="long",
                    ))

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        symbol = symbol.upper().replace(f"/{self.default_quote_currency}", "")
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    # === Orders ===

    def place_order(self, order: Order) -> OrderResult:
        try:
            symbol = self._normalize_symbol(order.symbol)

            # Map order type
            if order.order_type == OrderType.MARKET:
                ccxt_type = "market"
            else:
                ccxt_type = "limit"

            ccxt_side = "buy" if order.side == OrderSide.BUY else "sell"

            # Get quote for TCA
            quote = self.get_quote(symbol)

            # Place order
            result = self._exchange.create_order(
                symbol=symbol,
                type=ccxt_type,
                side=ccxt_side,
                amount=order.qty,
                price=order.limit_price if ccxt_type == "limit" else None,
            )

            broker_order_id = result.get("id")
            status = self._map_ccxt_status(result.get("status", "open"))
            filled = int(result.get("filled", 0))
            avg_price = result.get("average") or result.get("price")

            logger.info(
                f"Crypto order placed: {ccxt_side} {order.qty} {symbol} - "
                f"Status: {status.value}"
            )

            return OrderResult(
                success=True,
                broker_order_id=broker_order_id,
                client_order_id=order.client_order_id,
                status=status,
                filled_qty=filled,
                fill_price=avg_price,
                error_message=None,
                market_bid_at_execution=quote.bid if quote else None,
                market_ask_at_execution=quote.ask if quote else None,
            )

        except ccxt.InsufficientFunds as e:
            return OrderResult(
                success=False,
                broker_order_id=None,
                status=BrokerOrderStatus.REJECTED,
                filled_qty=0,
                fill_price=None,
                error_message=f"insufficient_funds: {e}",
            )
        except ccxt.InvalidOrder as e:
            return OrderResult(
                success=False,
                broker_order_id=None,
                status=BrokerOrderStatus.REJECTED,
                filled_qty=0,
                fill_price=None,
                error_message=f"invalid_order: {e}",
            )
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return OrderResult(
                success=False,
                broker_order_id=None,
                status=BrokerOrderStatus.REJECTED,
                filled_qty=0,
                fill_price=None,
                error_message=str(e),
            )

    def _map_ccxt_status(self, status: str) -> BrokerOrderStatus:
        """Map CCXT status to BrokerOrderStatus."""
        mapping = {
            "open": BrokerOrderStatus.SUBMITTED,
            "closed": BrokerOrderStatus.FILLED,
            "canceled": BrokerOrderStatus.CANCELLED,
            "cancelled": BrokerOrderStatus.CANCELLED,
            "expired": BrokerOrderStatus.EXPIRED,
            "rejected": BrokerOrderStatus.REJECTED,
        }
        return mapping.get(status.lower(), BrokerOrderStatus.PENDING)

    def cancel_order(self, broker_order_id: str, symbol: Optional[str] = None) -> bool:
        try:
            if symbol:
                symbol = self._normalize_symbol(symbol)
            self._exchange.cancel_order(broker_order_id, symbol)
            logger.info(f"Cancelled order {broker_order_id}")
            return True
        except ccxt.OrderNotFound:
            logger.warning(f"Order {broker_order_id} not found")
            return False
        except Exception as e:
            logger.error(f"Failed to cancel order {broker_order_id}: {e}")
            return False

    def get_order_status(self, broker_order_id: str, symbol: Optional[str] = None) -> Optional[BrokerOrderStatus]:
        try:
            if symbol:
                symbol = self._normalize_symbol(symbol)
            order = self._exchange.fetch_order(broker_order_id, symbol)
            return self._map_ccxt_status(order.get("status", ""))
        except Exception:
            return None

    def get_orders(
        self,
        status: str = "all",
        limit: int = 100,
        after: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        try:
            if status == "open":
                orders = self._exchange.fetch_open_orders(limit=limit)
            elif status == "closed":
                orders = self._exchange.fetch_closed_orders(limit=limit)
            else:
                # Combine open and closed
                open_orders = self._exchange.fetch_open_orders(limit=limit // 2)
                closed_orders = self._exchange.fetch_closed_orders(limit=limit // 2)
                orders = open_orders + closed_orders

            return orders

        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    # === Crypto-specific methods ===

    def get_available_symbols(self) -> List[str]:
        """Get all available trading symbols."""
        if not self._ensure_markets():
            return []
        return list(self._exchange.symbols)

    def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for a symbol."""
        try:
            symbol = self._normalize_symbol(symbol)
            market = self._exchange.market(symbol)
            return {
                "maker": market.get("maker", 0.001),
                "taker": market.get("taker", 0.001),
            }
        except Exception:
            return {"maker": 0.001, "taker": 0.001}


# Auto-register if CCXT is available
if CCXT_AVAILABLE:
    register_broker("crypto", CryptoBroker)
