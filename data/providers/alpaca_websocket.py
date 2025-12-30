"""
Alpaca WebSocket Market Data Streaming.

Real-time market data streaming via Alpaca's WebSocket API.
Provides live quotes, trades, and bars for subscribed symbols.

Usage:
    from data.providers.alpaca_websocket import AlpacaWebSocketClient

    async def on_quote(quote):
        print(f"{quote.symbol}: bid={quote.bid_price}, ask={quote.ask_price}")

    client = AlpacaWebSocketClient(
        symbols=["AAPL", "MSFT", "GOOGL"],
        on_quote=on_quote,
    )
    await client.start()
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Any

logger = logging.getLogger(__name__)

# Try importing alpaca-py WebSocket client
try:
    from alpaca.data.live import StockDataStream
    from alpaca.data.models import Quote, Trade, Bar
    ALPACA_WS_AVAILABLE = True
except ImportError:
    ALPACA_WS_AVAILABLE = False
    StockDataStream = None
    Quote = Trade = Bar = None


@dataclass
class QuoteData:
    """Normalized quote data from WebSocket stream."""
    symbol: str
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    timestamp: datetime

    @classmethod
    def from_alpaca(cls, quote: Any) -> 'QuoteData':
        """Convert Alpaca Quote to QuoteData."""
        return cls(
            symbol=quote.symbol,
            bid_price=float(quote.bid_price) if quote.bid_price else 0.0,
            bid_size=int(quote.bid_size) if quote.bid_size else 0,
            ask_price=float(quote.ask_price) if quote.ask_price else 0.0,
            ask_size=int(quote.ask_size) if quote.ask_size else 0,
            timestamp=quote.timestamp if hasattr(quote, 'timestamp') else datetime.utcnow(),
        )


@dataclass
class TradeData:
    """Normalized trade data from WebSocket stream."""
    symbol: str
    price: float
    size: int
    timestamp: datetime
    exchange: str

    @classmethod
    def from_alpaca(cls, trade: Any) -> 'TradeData':
        """Convert Alpaca Trade to TradeData."""
        return cls(
            symbol=trade.symbol,
            price=float(trade.price) if trade.price else 0.0,
            size=int(trade.size) if trade.size else 0,
            timestamp=trade.timestamp if hasattr(trade, 'timestamp') else datetime.utcnow(),
            exchange=str(trade.exchange) if hasattr(trade, 'exchange') else '',
        )


class AlpacaWebSocketClient:
    """
    Real-time market data streaming via Alpaca WebSocket.

    Provides:
    - Live quote streaming (bid/ask)
    - Live trade streaming
    - Live bar streaming (1-minute aggregated)

    Args:
        symbols: List of symbols to subscribe to
        on_quote: Callback for quote updates (receives QuoteData)
        on_trade: Callback for trade updates (receives TradeData)
        on_bar: Callback for bar updates
        api_key: Alpaca API key (defaults to env var)
        secret_key: Alpaca secret key (defaults to env var)
        feed: Data feed ('iex' for free, 'sip' for paid)
    """

    def __init__(
        self,
        symbols: List[str],
        on_quote: Optional[Callable[[QuoteData], None]] = None,
        on_trade: Optional[Callable[[TradeData], None]] = None,
        on_bar: Optional[Callable[[Any], None]] = None,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        feed: str = 'iex',
    ):
        if not ALPACA_WS_AVAILABLE:
            raise ImportError(
                "alpaca-py not installed. Install with: pip install alpaca-py"
            )

        self.symbols = [s.upper() for s in symbols]
        self.on_quote = on_quote
        self.on_trade = on_trade
        self.on_bar = on_bar
        self.feed = feed

        # Get API credentials
        self._api_key = api_key or os.getenv('ALPACA_API_KEY_ID') or os.getenv('APCA_API_KEY_ID')
        self._secret_key = secret_key or os.getenv('ALPACA_API_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')

        if not self._api_key or not self._secret_key:
            raise ValueError("Alpaca API credentials not found in environment")

        # Create stream client
        self._stream: Optional[StockDataStream] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def _create_stream(self) -> StockDataStream:
        """Create the Alpaca stream client."""
        return StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
            feed=self.feed,
        )

    async def _handle_quote(self, quote: Any) -> None:
        """Handle incoming quote from stream."""
        if self.on_quote:
            try:
                data = QuoteData.from_alpaca(quote)
                self.on_quote(data)
            except Exception as e:
                logger.error(f"Error handling quote: {e}")

    async def _handle_trade(self, trade: Any) -> None:
        """Handle incoming trade from stream."""
        if self.on_trade:
            try:
                data = TradeData.from_alpaca(trade)
                self.on_trade(data)
            except Exception as e:
                logger.error(f"Error handling trade: {e}")

    async def _handle_bar(self, bar: Any) -> None:
        """Handle incoming bar from stream."""
        if self.on_bar:
            try:
                self.on_bar(bar)
            except Exception as e:
                logger.error(f"Error handling bar: {e}")

    async def start(self) -> None:
        """
        Start the WebSocket stream.

        Subscribes to quotes, trades, and bars for configured symbols.
        This method runs indefinitely until stop() is called.
        """
        if self._running:
            logger.warning("WebSocket stream already running")
            return

        self._stream = self._create_stream()
        self._running = True

        # Subscribe to data feeds
        if self.on_quote:
            self._stream.subscribe_quotes(self._handle_quote, *self.symbols)
            logger.info(f"Subscribed to quotes: {self.symbols}")

        if self.on_trade:
            self._stream.subscribe_trades(self._handle_trade, *self.symbols)
            logger.info(f"Subscribed to trades: {self.symbols}")

        if self.on_bar:
            self._stream.subscribe_bars(self._handle_bar, *self.symbols)
            logger.info(f"Subscribed to bars: {self.symbols}")

        try:
            logger.info("Starting Alpaca WebSocket stream...")
            await self._stream._run_forever()
        except asyncio.CancelledError:
            logger.info("WebSocket stream cancelled")
        except Exception as e:
            logger.error(f"WebSocket stream error: {e}")
            raise
        finally:
            self._running = False

    def stop(self) -> None:
        """Stop the WebSocket stream."""
        if self._stream:
            try:
                self._stream.close()
            except Exception as e:
                logger.error(f"Error closing stream: {e}")
        self._running = False
        logger.info("WebSocket stream stopped")

    @property
    def is_running(self) -> bool:
        """Check if stream is currently running."""
        return self._running

    def add_symbols(self, symbols: List[str]) -> None:
        """Add symbols to the subscription (while running)."""
        new_symbols = [s.upper() for s in symbols if s.upper() not in self.symbols]
        if not new_symbols:
            return

        self.symbols.extend(new_symbols)

        if self._stream and self._running:
            if self.on_quote:
                self._stream.subscribe_quotes(self._handle_quote, *new_symbols)
            if self.on_trade:
                self._stream.subscribe_trades(self._handle_trade, *new_symbols)
            if self.on_bar:
                self._stream.subscribe_bars(self._handle_bar, *new_symbols)

            logger.info(f"Added symbols to stream: {new_symbols}")

    def remove_symbols(self, symbols: List[str]) -> None:
        """Remove symbols from the subscription (while running)."""
        remove_symbols = [s.upper() for s in symbols if s.upper() in self.symbols]
        if not remove_symbols:
            return

        for s in remove_symbols:
            self.symbols.remove(s)

        if self._stream and self._running:
            if self.on_quote:
                self._stream.unsubscribe_quotes(*remove_symbols)
            if self.on_trade:
                self._stream.unsubscribe_trades(*remove_symbols)
            if self.on_bar:
                self._stream.unsubscribe_bars(*remove_symbols)

            logger.info(f"Removed symbols from stream: {remove_symbols}")


# Convenience function for simple quote streaming
async def stream_quotes(
    symbols: List[str],
    on_quote: Callable[[QuoteData], None],
    duration_seconds: Optional[float] = None,
) -> None:
    """
    Stream quotes for a list of symbols.

    Args:
        symbols: List of stock symbols
        on_quote: Callback function for each quote
        duration_seconds: Optional duration to stream (None = forever)
    """
    client = AlpacaWebSocketClient(symbols=symbols, on_quote=on_quote)

    if duration_seconds:
        async def run_with_timeout():
            await asyncio.wait_for(client.start(), timeout=duration_seconds)

        try:
            await run_with_timeout()
        except asyncio.TimeoutError:
            client.stop()
    else:
        await client.start()


# Export availability flag
def is_websocket_available() -> bool:
    """Check if WebSocket streaming is available."""
    return ALPACA_WS_AVAILABLE
