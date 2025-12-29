"""
Alpaca Live Data Provider

Fetches current/recent market data from Alpaca Data API.
Use for:
- Real-time quotes during market hours
- Latest bar data for paper/live trading
- Filling gaps when Polygon EOD is not yet available

Historical backtesting should still use Polygon (more complete history).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class AlpacaDataConfig:
    """Alpaca Data API configuration."""
    data_url: str
    key_id: str
    secret: str


def _get_alpaca_data_config() -> AlpacaDataConfig:
    """Load Alpaca Data API config from environment."""
    # Support both ALPACA_ and APCA_ prefixes
    key_id = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY", "")

    # Data API base URL (different from trading API)
    data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")

    return AlpacaDataConfig(
        data_url=data_url.rstrip("/"),
        key_id=key_id,
        secret=secret,
    )


def _alpaca_headers() -> Dict[str, str]:
    """Get Alpaca API headers."""
    cfg = _get_alpaca_data_config()
    return {
        "APCA-API-KEY-ID": cfg.key_id,
        "APCA-API-SECRET-KEY": cfg.secret,
    }


def get_latest_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get the latest quote for a symbol.

    Returns dict with: bid_price, bid_size, ask_price, ask_size, timestamp
    Returns None on error.
    """
    cfg = _get_alpaca_data_config()
    if not cfg.key_id or not cfg.secret:
        logger.warning("Alpaca API keys not configured")
        return None

    url = f"{cfg.data_url}/v2/stocks/{symbol}/quotes/latest"

    try:
        resp = requests.get(url, headers=_alpaca_headers(), timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Alpaca quote error {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()
        quote = data.get("quote", {})
        return {
            "symbol": symbol,
            "bid_price": quote.get("bp", 0.0),
            "bid_size": quote.get("bs", 0),
            "ask_price": quote.get("ap", 0.0),
            "ask_size": quote.get("as", 0),
            "timestamp": quote.get("t"),
        }
    except Exception as e:
        logger.warning(f"Alpaca quote fetch failed for {symbol}: {e}")
        return None


def get_latest_trade(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get the latest trade for a symbol.

    Returns dict with: price, size, timestamp
    Returns None on error.
    """
    cfg = _get_alpaca_data_config()
    if not cfg.key_id or not cfg.secret:
        logger.warning("Alpaca API keys not configured")
        return None

    url = f"{cfg.data_url}/v2/stocks/{symbol}/trades/latest"

    try:
        resp = requests.get(url, headers=_alpaca_headers(), timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Alpaca trade error {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()
        trade = data.get("trade", {})
        return {
            "symbol": symbol,
            "price": trade.get("p", 0.0),
            "size": trade.get("s", 0),
            "timestamp": trade.get("t"),
        }
    except Exception as e:
        logger.warning(f"Alpaca trade fetch failed for {symbol}: {e}")
        return None


def get_latest_bar(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get the latest 1-minute bar for a symbol.

    Returns dict with: open, high, low, close, volume, timestamp
    Returns None on error.
    """
    cfg = _get_alpaca_data_config()
    if not cfg.key_id or not cfg.secret:
        logger.warning("Alpaca API keys not configured")
        return None

    url = f"{cfg.data_url}/v2/stocks/{symbol}/bars/latest"
    params = {"feed": "iex"}  # IEX feed for free tier

    try:
        resp = requests.get(url, headers=_alpaca_headers(), params=params, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Alpaca bar error {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()
        bar = data.get("bar", {})
        return {
            "symbol": symbol,
            "open": bar.get("o", 0.0),
            "high": bar.get("h", 0.0),
            "low": bar.get("l", 0.0),
            "close": bar.get("c", 0.0),
            "volume": bar.get("v", 0),
            "timestamp": bar.get("t"),
        }
    except Exception as e:
        logger.warning(f"Alpaca bar fetch failed for {symbol}: {e}")
        return None


def fetch_bars_alpaca(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1Day",
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch historical bars from Alpaca Data API.

    Args:
        symbol: Stock symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
        cache_dir: Optional cache directory

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    cfg = _get_alpaca_data_config()
    if not cfg.key_id or not cfg.secret:
        logger.warning("Alpaca API keys not configured")
        return pd.DataFrame(columns=['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

    # Check cache
    cache_file = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{symbol}_{start}_{end}_{timeframe}_alpaca.csv"
        if cache_file.exists():
            try:
                return pd.read_csv(cache_file, parse_dates=['timestamp'])
            except Exception:
                pass

    url = f"{cfg.data_url}/v2/stocks/{symbol}/bars"
    params = {
        "start": f"{start}T00:00:00Z",
        "end": f"{end}T23:59:59Z",
        "timeframe": timeframe,
        "feed": "iex",
        "limit": 10000,
    }

    all_bars = []
    page_token = None

    try:
        while True:
            if page_token:
                params["page_token"] = page_token

            resp = requests.get(url, headers=_alpaca_headers(), params=params, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"Alpaca bars error {resp.status_code}: {resp.text[:200]}")
                break

            data = resp.json()
            bars = data.get("bars", [])

            for bar in bars:
                all_bars.append({
                    "timestamp": bar.get("t"),
                    "symbol": symbol,
                    "open": bar.get("o", 0.0),
                    "high": bar.get("h", 0.0),
                    "low": bar.get("l", 0.0),
                    "close": bar.get("c", 0.0),
                    "volume": bar.get("v", 0),
                })

            page_token = data.get("next_page_token")
            if not page_token:
                break
    except Exception as e:
        logger.warning(f"Alpaca bars fetch failed for {symbol}: {e}")

    if not all_bars:
        return pd.DataFrame(columns=['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(all_bars)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Cache results
    if cache_file:
        try:
            df.to_csv(cache_file, index=False)
        except Exception:
            pass

    return df


def fetch_multi_quotes(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch latest quotes for multiple symbols in a single request.

    Returns dict mapping symbol -> quote data
    """
    cfg = _get_alpaca_data_config()
    if not cfg.key_id or not cfg.secret:
        logger.warning("Alpaca API keys not configured")
        return {}

    # Alpaca allows up to 200 symbols per request
    url = f"{cfg.data_url}/v2/stocks/quotes/latest"
    params = {"symbols": ",".join(symbols)}

    try:
        resp = requests.get(url, headers=_alpaca_headers(), params=params, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"Alpaca multi-quote error {resp.status_code}: {resp.text[:200]}")
            return {}

        data = resp.json()
        quotes = data.get("quotes", {})

        result = {}
        for sym, quote in quotes.items():
            result[sym] = {
                "symbol": sym,
                "bid_price": quote.get("bp", 0.0),
                "bid_size": quote.get("bs", 0),
                "ask_price": quote.get("ap", 0.0),
                "ask_size": quote.get("as", 0),
                "timestamp": quote.get("t"),
            }
        return result
    except Exception as e:
        logger.warning(f"Alpaca multi-quote fetch failed: {e}")
        return {}


def fetch_multi_bars(symbols: List[str], timeframe: str = "1Day") -> Dict[str, Dict[str, Any]]:
    """
    Fetch latest bars for multiple symbols in a single request.

    Returns dict mapping symbol -> bar data
    """
    cfg = _get_alpaca_data_config()
    if not cfg.key_id or not cfg.secret:
        logger.warning("Alpaca API keys not configured")
        return {}

    url = f"{cfg.data_url}/v2/stocks/bars/latest"
    params = {
        "symbols": ",".join(symbols),
        "feed": "iex",
    }

    try:
        resp = requests.get(url, headers=_alpaca_headers(), params=params, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"Alpaca multi-bar error {resp.status_code}: {resp.text[:200]}")
            return {}

        data = resp.json()
        bars = data.get("bars", {})

        result = {}
        for sym, bar in bars.items():
            result[sym] = {
                "symbol": sym,
                "open": bar.get("o", 0.0),
                "high": bar.get("h", 0.0),
                "low": bar.get("l", 0.0),
                "close": bar.get("c", 0.0),
                "volume": bar.get("v", 0),
                "timestamp": bar.get("t"),
            }
        return result
    except Exception as e:
        logger.warning(f"Alpaca multi-bar fetch failed: {e}")
        return {}


def is_market_open() -> bool:
    """Check if US equity market is currently open."""
    cfg = _get_alpaca_data_config()
    if not cfg.key_id or not cfg.secret:
        return False

    # Use trading API for clock
    trading_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    url = f"{trading_url.rstrip('/')}/v2/clock"

    try:
        resp = requests.get(url, headers=_alpaca_headers(), timeout=10)
        if resp.status_code != 200:
            return False

        data = resp.json()
        return data.get("is_open", False)
    except Exception:
        return False


def get_market_clock() -> Optional[Dict[str, Any]]:
    """Get market clock status."""
    cfg = _get_alpaca_data_config()
    if not cfg.key_id or not cfg.secret:
        return None

    trading_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    url = f"{trading_url.rstrip('/')}/v2/clock"

    try:
        resp = requests.get(url, headers=_alpaca_headers(), timeout=10)
        if resp.status_code != 200:
            return None

        return resp.json()
    except Exception:
        return None


# Convenience function for scanner
def get_current_price(symbol: str) -> Optional[float]:
    """
    Get current price for a symbol (mid of bid/ask, or last trade).

    Returns None if unavailable.
    """
    quote = get_latest_quote(symbol)
    if quote and quote.get("bid_price") and quote.get("ask_price"):
        return (quote["bid_price"] + quote["ask_price"]) / 2

    trade = get_latest_trade(symbol)
    if trade and trade.get("price"):
        return trade["price"]

    return None


if __name__ == "__main__":
    # Test the module
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing Alpaca Live Data Provider")
    print("=" * 50)

    # Test market clock
    clock = get_market_clock()
    if clock:
        print(f"Market open: {clock.get('is_open')}")
        print(f"Current time: {clock.get('timestamp')}")
        print(f"Next open: {clock.get('next_open')}")
        print(f"Next close: {clock.get('next_close')}")
    else:
        print("Could not get market clock - check API keys")

    print()

    # Test single quote
    print("SPY Quote:")
    quote = get_latest_quote("SPY")
    if quote:
        print(f"  Bid: ${quote['bid_price']:.2f} x {quote['bid_size']}")
        print(f"  Ask: ${quote['ask_price']:.2f} x {quote['ask_size']}")
    else:
        print("  Failed to get quote")

    print()

    # Test multi-quote
    print("Multi-quote (SPY, AAPL, MSFT):")
    quotes = fetch_multi_quotes(["SPY", "AAPL", "MSFT"])
    for sym, q in quotes.items():
        print(f"  {sym}: ${q['bid_price']:.2f} / ${q['ask_price']:.2f}")

    print()

    # Test current price
    print("Current prices:")
    for sym in ["SPY", "AAPL", "MSFT"]:
        price = get_current_price(sym)
        if price:
            print(f"  {sym}: ${price:.2f}")
        else:
            print(f"  {sym}: N/A")
