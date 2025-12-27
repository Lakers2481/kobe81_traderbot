"""
Stooq EOD Data Provider
========================

Free daily OHLCV data from Stooq.

Features:
- No API key required
- Historical data available (5-10+ years for major stocks)
- Daily OHLCV with adjusted closes
- US stocks use .US suffix

Limitations:
- Rate limiting (be respectful)
- May have gaps or missing symbols
- Data quality varies

Usage:
    from data.providers.stooq_eod import StooqEODProvider

    provider = StooqEODProvider()
    df = provider.fetch_symbol('AAPL', start='2015-01-01', end='2024-12-31')

    # Fetch multiple symbols
    df = provider.fetch_universe(symbols=['AAPL', 'MSFT', 'GOOGL'], ...)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
from io import StringIO

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Stooq base URL
STOOQ_BASE_URL = "https://stooq.com/q/d/l/"


class StooqEODProvider:
    """
    Fetches daily OHLCV data from Stooq (free, no API key).

    Output columns: timestamp, symbol, open, high, low, close, volume
    """

    def __init__(
        self,
        rate_limit_delay: float = 0.5,  # Seconds between requests
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

    def _stooq_symbol(self, symbol: str) -> str:
        """Convert symbol to Stooq format (add .US for US stocks)."""
        symbol = symbol.upper().strip()

        # Skip if already has suffix
        if '.' in symbol:
            return symbol

        # Add .US suffix for US stocks
        return f"{symbol}.US"

    def fetch_symbol(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single symbol.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        """
        stooq_symbol = self._stooq_symbol(symbol)

        # Build URL
        params = {
            's': stooq_symbol,
            'd1': start.replace('-', '') if start else None,
            'd2': end.replace('-', '') if end else None,
        }
        params = {k: v for k, v in params.items() if v is not None}

        url = STOOQ_BASE_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())

        # Fetch with retries
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()

                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()

                # Parse CSV
                content = response.text

                if 'No data' in content or len(content.strip()) < 50:
                    logger.warning(f"No data from Stooq for {symbol}")
                    return pd.DataFrame()

                df = pd.read_csv(StringIO(content))

                # Stooq columns: Date,Open,High,Low,Close,Volume
                if df.empty or 'Date' not in df.columns:
                    logger.warning(f"Invalid data format for {symbol}")
                    return pd.DataFrame()

                # Rename and standardize
                df = df.rename(columns={
                    'Date': 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                })

                # Add symbol column
                df['symbol'] = symbol.upper()

                # Convert timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Ensure numeric columns
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                if 'volume' in df.columns:
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
                else:
                    df['volume'] = 0

                # Filter date range
                if start:
                    df = df[df['timestamp'] >= pd.to_datetime(start)]
                if end:
                    df = df[df['timestamp'] <= pd.to_datetime(end)]

                # Select and order columns
                df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

                # Sort by date
                df = df.sort_values('timestamp').reset_index(drop=True)

                logger.debug(f"Fetched {len(df)} rows for {symbol} from Stooq")
                return df

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue

            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                return pd.DataFrame()

        logger.error(f"Failed to fetch {symbol} after {self.max_retries} attempts")
        return pd.DataFrame()

    def fetch_universe(
        self,
        symbols: List[str],
        start: str,
        end: str,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of stock tickers
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            progress_callback: Optional callback(current, total, symbol)

        Returns:
            DataFrame with all symbols' data
        """
        all_dfs = []
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(i + 1, total, symbol)

            df = self.fetch_symbol(symbol, start, end)

            if not df.empty:
                all_dfs.append(df)
            else:
                logger.warning(f"No data for {symbol}")

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)
        result = result.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        logger.info(
            f"Fetched {len(result):,} rows for {len(all_dfs)} symbols from Stooq"
        )

        return result

    def fetch_from_universe_file(
        self,
        universe_path: Union[str, Path],
        start: str,
        end: str,
        symbol_column: str = 'symbol',
        limit: Optional[int] = None,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Fetch data for symbols from a universe CSV file.

        Args:
            universe_path: Path to universe CSV
            start: Start date
            end: End date
            symbol_column: Column name containing symbols
            limit: Optional limit on number of symbols
            progress_callback: Optional progress callback

        Returns:
            DataFrame with OHLCV data
        """
        universe_path = Path(universe_path)

        if not universe_path.exists():
            raise FileNotFoundError(f"Universe file not found: {universe_path}")

        # Read universe
        universe_df = pd.read_csv(universe_path)

        if symbol_column not in universe_df.columns:
            # Try common alternatives
            for alt in ['Symbol', 'ticker', 'Ticker', 'SYMBOL']:
                if alt in universe_df.columns:
                    symbol_column = alt
                    break
            else:
                raise ValueError(f"Symbol column '{symbol_column}' not found in universe file")

        symbols = universe_df[symbol_column].unique().tolist()

        if limit:
            symbols = symbols[:limit]

        logger.info(f"Fetching data for {len(symbols)} symbols from Stooq")

        return self.fetch_universe(symbols, start, end, progress_callback)


def fetch_stooq_eod(
    symbols: List[str],
    start: str,
    end: str,
    **kwargs,
) -> pd.DataFrame:
    """Convenience function to fetch Stooq EOD data."""
    provider = StooqEODProvider(**kwargs)
    return provider.fetch_universe(symbols, start, end)
