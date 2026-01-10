"""
Binance Public Klines Provider
===============================

Free OHLCV data from Binance public API (no API key required).

Features:
- No API key needed for public market data
- Historical klines (candlesticks) up to 1000 per request
- Supports all timeframes (1m, 5m, 15m, 1h, 4h, 1d, 1w)
- Rate limiting built-in

Usage:
    from data.providers.binance_klines import BinanceKlinesProvider

    provider = BinanceKlinesProvider()

    # Fetch hourly data
    df = provider.fetch_symbol('BTCUSDT', start='2020-01-01', end='2024-12-31', timeframe='1h')

    # Fetch and resample to daily
    df_daily = provider.fetch_symbol_daily('BTCUSDT', start='2020-01-01', end='2024-12-31')
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import List, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Binance API endpoints
BINANCE_SPOT_API = "https://api.binance.com/api/v3"
BINANCE_KLINES_ENDPOINT = f"{BINANCE_SPOT_API}/klines"

# Timeframe mappings
TIMEFRAME_MAP = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d',
    '1w': '1w',
    '1M': '1M',
}

# Milliseconds per timeframe
MS_PER_TF = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '30m': 30 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
    '1w': 7 * 24 * 60 * 60 * 1000,
}


class BinanceKlinesProvider:
    """
    Fetches OHLCV data from Binance public API (no API key required).

    Output columns: timestamp, symbol, open, high, low, close, volume
    """

    def __init__(
        self,
        rate_limit_delay: float = 0.1,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _to_ms(self, dt: Union[str, datetime]) -> int:
        """Convert datetime to milliseconds timestamp."""
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        return int(dt.timestamp() * 1000)

    def _from_ms(self, ms: int) -> datetime:
        """Convert milliseconds to datetime."""
        return datetime.fromtimestamp(ms / 1000)

    def fetch_symbol(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = '1h',
    ) -> pd.DataFrame:
        """
        Fetch OHLCV klines for a single symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            timeframe: Kline interval ('1m', '5m', '15m', '1h', '4h', '1d', '1w')

        Returns:
            DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        """
        symbol = symbol.upper()

        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(f"Invalid timeframe: {timeframe}. Valid: {list(TIMEFRAME_MAP.keys())}")

        interval = TIMEFRAME_MAP[timeframe]
        start_ms = self._to_ms(start)
        end_ms = self._to_ms(end)

        all_klines = []
        current_start = start_ms

        # Fetch in chunks (max 1000 per request)
        while current_start < end_ms:
            try:
                self._rate_limit()

                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': end_ms,
                    'limit': 1000,
                }

                response = requests.get(
                    BINANCE_KLINES_ENDPOINT,
                    params=params,
                    timeout=self.timeout,
                )

                if response.status_code == 429:
                    # Rate limited
                    logger.warning("Rate limited by Binance, waiting...")
                    time.sleep(60)
                    continue

                response.raise_for_status()
                klines = response.json()

                if not klines:
                    break

                all_klines.extend(klines)

                # Move to next chunk
                last_ts = klines[-1][0]
                current_start = last_ts + MS_PER_TF.get(timeframe, 60000)

                if len(klines) < 1000:
                    break  # No more data

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed for {symbol}: {e}")
                break

        if not all_klines:
            logger.warning(f"No data from Binance for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        # Kline format: [open_time, open, high, low, close, volume, close_time, ...]
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convert types
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df['symbol'] = symbol
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        # Select columns
        df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

        # Sort
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='last')

        logger.debug(f"Fetched {len(df)} {timeframe} klines for {symbol}")
        return df

    def fetch_symbol_daily(
        self,
        symbol: str,
        start: str,
        end: str,
        source_timeframe: str = '1h',
    ) -> pd.DataFrame:
        """
        Fetch hourly data and resample to daily for swing trading.

        Args:
            symbol: Trading pair
            start: Start date
            end: End date
            source_timeframe: Timeframe to fetch and resample from

        Returns:
            Daily OHLCV DataFrame
        """
        # Fetch hourly data
        df = self.fetch_symbol(symbol, start, end, timeframe=source_timeframe)

        if df.empty:
            return df

        # Resample to daily
        df = df.set_index('timestamp')

        daily = df.resample('D').agg({
            'symbol': 'first',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()

        daily = daily.reset_index()
        daily['symbol'] = symbol

        return daily

    def fetch_universe(
        self,
        symbols: List[str],
        start: str,
        end: str,
        timeframe: str = '1h',
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of trading pairs
            start: Start date
            end: End date
            timeframe: Kline interval
            progress_callback: Optional callback(current, total, symbol)

        Returns:
            DataFrame with all symbols' data
        """
        all_dfs = []
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(i + 1, total, symbol)

            df = self.fetch_symbol(symbol, start, end, timeframe)

            if not df.empty:
                all_dfs.append(df)
            else:
                logger.warning(f"No data for {symbol}")

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)
        result = result.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        logger.info(
            f"Fetched {len(result):,} rows for {len(all_dfs)} symbols from Binance"
        )

        return result

    def get_available_symbols(self) -> List[str]:
        """Get list of available trading pairs."""
        try:
            response = requests.get(
                f"{BINANCE_SPOT_API}/exchangeInfo",
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            symbols = [
                s['symbol'] for s in data.get('symbols', [])
                if s.get('status') == 'TRADING'
            ]

            return sorted(symbols)

        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
            return []


def fetch_binance_klines(
    symbols: List[str],
    start: str,
    end: str,
    timeframe: str = '1h',
    **kwargs,
) -> pd.DataFrame:
    """Convenience function to fetch Binance klines."""
    provider = BinanceKlinesProvider(**kwargs)
    return provider.fetch_universe(symbols, start, end, timeframe)
