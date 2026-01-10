"""
Crypto Signal Generator for Scanner Integration.

Generates crypto trading signals using the same strategy logic as equities:
- Uses Polygon crypto data (X:BTCUSD, X:ETHUSD, etc.)
- Applies DualStrategyScanner to crypto OHLCV bars
- Returns signals in same format as equity signals

Default crypto universe:
- BTC/USD (Bitcoin)
- ETH/USD (Ethereum)
- SOL/USD (Solana)
- AVAX/USD (Avalanche)
- LINK/USD (Chainlink)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Crypto data provider
try:
    from data.providers.polygon_crypto import fetch_crypto_bars
    CRYPTO_DATA_AVAILABLE = True
except ImportError:
    logger.warning("Crypto data provider not available")
    CRYPTO_DATA_AVAILABLE = False

# Strategy imports
try:
    from strategies.dual_strategy import DualStrategyScanner, DualStrategyParams
    STRATEGY_AVAILABLE = True
except ImportError:
    logger.warning("DualStrategyScanner not available")
    STRATEGY_AVAILABLE = False


# Default crypto universe (major pairs available on Polygon)
DEFAULT_CRYPTO_UNIVERSE = [
    "X:BTCUSD",   # Bitcoin
    "X:ETHUSD",   # Ethereum
    "X:SOLUSD",   # Solana
    "X:AVAXUSD",  # Avalanche
    "X:LINKUSD", # Chainlink
    "X:DOGEUSD",  # Dogecoin
    "X:MATICUSD", # Polygon (MATIC)
    "X:ADAUSD",   # Cardano
]


def fetch_crypto_universe_data(
    symbols: List[str],
    start: str,
    end: str,
    cache_dir: Optional[Path] = None,
    timeframe: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for crypto universe.

    Args:
        symbols: List of crypto tickers (e.g., ["X:BTCUSD", "X:ETHUSD"])
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        cache_dir: Optional cache directory
        timeframe: Bar timeframe (default "1d" for daily)

    Returns:
        Combined DataFrame with all crypto bars
    """
    if not CRYPTO_DATA_AVAILABLE:
        logger.warning("Crypto data provider not available")
        return pd.DataFrame()

    all_data = []

    for symbol in symbols:
        try:
            df = fetch_crypto_bars(
                symbol=symbol,
                start=start,
                end=end,
                timeframe=timeframe,
                cache_dir=cache_dir,
            )
            if not df.empty:
                all_data.append(df)
                logger.debug(f"Fetched {len(df)} bars for {symbol}")
        except Exception as e:
            logger.debug(f"Failed to fetch {symbol}: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def generate_crypto_signals(
    price_data: pd.DataFrame = None,
    symbols: List[str] = None,
    start: str = None,
    end: str = None,
    cache_dir: Optional[Path] = None,
    max_signals: int = 3,
    preview_mode: bool = False,
) -> pd.DataFrame:
    """
    Generate crypto trading signals using DualStrategyScanner.

    Args:
        price_data: Pre-fetched OHLCV data (optional)
        symbols: Crypto tickers to scan (default: DEFAULT_CRYPTO_UNIVERSE)
        start: Start date for data fetch
        end: End date for data fetch
        cache_dir: Cache directory for data
        max_signals: Maximum signals to return
        preview_mode: Use preview mode for signal generation

    Returns:
        DataFrame of crypto signals in same format as equity signals
    """
    if not CRYPTO_DATA_AVAILABLE or not STRATEGY_AVAILABLE:
        return pd.DataFrame()

    # Use default universe if not specified
    if symbols is None:
        symbols = DEFAULT_CRYPTO_UNIVERSE

    # Fetch data if not provided
    if price_data is None or price_data.empty:
        if start is None or end is None:
            end = datetime.now().strftime('%Y-%m-%d')
            start = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')

        price_data = fetch_crypto_universe_data(
            symbols=symbols,
            start=start,
            end=end,
            cache_dir=cache_dir,
            timeframe="1d",
        )

    if price_data.empty:
        logger.warning("No crypto data available for signal generation")
        return pd.DataFrame()

    # Run DualStrategyScanner on crypto data
    try:
        # Crypto doesn't have a minimum price filter
        params = DualStrategyParams(min_price=0.0)
        scanner = DualStrategyScanner(params, preview_mode=preview_mode)

        signals = scanner.generate_signals(price_data)

        if signals.empty:
            return pd.DataFrame()

        # Add asset class marker
        signals = signals.copy()
        signals['asset_class'] = 'CRYPTO'

        # Clean up symbol names (remove X: prefix for display)
        if 'symbol' in signals.columns:
            signals['symbol'] = signals['symbol'].str.replace('X:', '', regex=False)

        # Sort by confidence and limit
        if 'conf_score' in signals.columns:
            signals = signals.sort_values('conf_score', ascending=False)
        signals = signals.head(max_signals)

        # Add strategy suffix
        if 'strategy' in signals.columns:
            signals['strategy'] = signals['strategy'].astype(str) + '_CRYPTO'

        return signals

    except Exception as e:
        logger.error(f"Crypto signal generation failed: {e}")
        return pd.DataFrame()


def scan_crypto(
    cap: int = 8,
    cache_dir: Optional[Path] = None,
    max_signals: int = 3,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Quick crypto scan using default universe.

    Args:
        cap: Maximum number of crypto pairs to scan
        cache_dir: Cache directory
        max_signals: Maximum signals to return
        verbose: Print progress

    Returns:
        DataFrame of crypto signals
    """
    symbols = DEFAULT_CRYPTO_UNIVERSE[:cap]

    if verbose:
        print(f"Scanning {len(symbols)} crypto pairs...")

    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')

    signals = generate_crypto_signals(
        symbols=symbols,
        start=start,
        end=end,
        cache_dir=cache_dir,
        max_signals=max_signals,
    )

    if verbose:
        if signals.empty:
            print("No crypto signals generated")
        else:
            print(f"Generated {len(signals)} crypto signal(s)")

    return signals
