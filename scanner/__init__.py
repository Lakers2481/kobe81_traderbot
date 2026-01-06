"""
Scanner Module - Multi-Asset Signal Generation.

Provides signal generation for:
- Equities (via DualStrategyScanner)
- Options (calls/puts derived from equity signals)
- Crypto (BTC, ETH, etc. via Polygon)
"""
from __future__ import annotations

from .options_signals import (
    OptionsSignalGenerator,
    OptionsSignal,
    generate_options_signals,
    OPTIONS_AVAILABLE,
)
from .crypto_signals import (
    generate_crypto_signals,
    scan_crypto,
    fetch_crypto_universe_data,
    DEFAULT_CRYPTO_UNIVERSE,
    CRYPTO_DATA_AVAILABLE,
)

__all__ = [
    # Options
    'OptionsSignalGenerator',
    'OptionsSignal',
    'generate_options_signals',
    'OPTIONS_AVAILABLE',
    # Crypto
    'generate_crypto_signals',
    'scan_crypto',
    'fetch_crypto_universe_data',
    'DEFAULT_CRYPTO_UNIVERSE',
    'CRYPTO_DATA_AVAILABLE',
]
