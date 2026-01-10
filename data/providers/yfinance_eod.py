"""
Yahoo Finance EOD Data Provider (Fallback)
==========================================

Free daily OHLCV data from Yahoo Finance via yfinance library.

WARNING: This is an UNOFFICIAL API. Yahoo may:
- Rate limit or block requests
- Change the API without notice
- Have data quality issues

Use Stooq as primary provider; this is a fallback.

Usage:
    from data.providers.yfinance_eod import YFinanceEODProvider

    provider = YFinanceEODProvider()
    df = provider.fetch_symbol('AAPL', start='2015-01-01', end='2024-12-31')
"""
from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning(
        "yfinance not installed. Install with: pip install yfinance"
    )


class YFinanceEODProvider:
    """
    Fetches daily OHLCV data from Yahoo Finance (UNOFFICIAL).

    Output columns: timestamp, symbol, open, high, low, close, volume

    WARNING: Yahoo Finance is unofficial and may break or rate limit.
    Use Stooq as primary provider.
    """

    def __init__(
        self,
        rate_limit_delay: float = 0.2,
        warn_unofficial: bool = True,
    ):
        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance not installed. Install with: pip install yfinance"
            )

        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0

        if warn_unofficial:
            warnings.warn(
                "Yahoo Finance is an UNOFFICIAL API. "
                "Data may be unreliable. Use Stooq as primary source.",
                UserWarning
            )

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

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
        try:
            self._rate_limit()

            # Create ticker object
            ticker = yf.Ticker(symbol.upper())

            # Fetch history
            df = ticker.history(start=start, end=end, auto_adjust=True)

            if df.empty:
                logger.warning(f"No data from Yahoo Finance for {symbol}")
                return pd.DataFrame()

            # Reset index to get date as column
            df = df.reset_index()

            # Rename columns
            df = df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
            })

            # Add symbol
            df['symbol'] = symbol.upper()

            # Ensure timestamp is datetime (remove timezone)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

            # Ensure numeric
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)

            # Select and order columns
            df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

            # Sort by date
            df = df.sort_values('timestamp').reset_index(drop=True)

            logger.debug(f"Fetched {len(df)} rows for {symbol} from Yahoo Finance")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol} from Yahoo Finance: {e}")
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
        failed = []

        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(i + 1, total, symbol)

            df = self.fetch_symbol(symbol, start, end)

            if not df.empty:
                all_dfs.append(df)
            else:
                failed.append(symbol)

        if failed:
            logger.warning(f"Failed to fetch {len(failed)} symbols: {failed[:10]}...")

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)
        result = result.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        logger.info(
            f"Fetched {len(result):,} rows for {len(all_dfs)} symbols from Yahoo Finance"
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

        universe_df = pd.read_csv(universe_path)

        if symbol_column not in universe_df.columns:
            for alt in ['Symbol', 'ticker', 'Ticker', 'SYMBOL']:
                if alt in universe_df.columns:
                    symbol_column = alt
                    break
            else:
                raise ValueError(f"Symbol column '{symbol_column}' not found")

        symbols = universe_df[symbol_column].unique().tolist()

        if limit:
            symbols = symbols[:limit]

        logger.info(f"Fetching data for {len(symbols)} symbols from Yahoo Finance")

        return self.fetch_universe(symbols, start, end, progress_callback)


def fetch_yfinance_eod(
    symbols: List[str],
    start: str,
    end: str,
    **kwargs,
) -> pd.DataFrame:
    """Convenience function to fetch Yahoo Finance EOD data."""
    provider = YFinanceEODProvider(**kwargs)
    return provider.fetch_universe(symbols, start, end)
