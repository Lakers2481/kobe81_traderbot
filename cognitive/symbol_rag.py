"""
Symbol RAG (Retrieval-Augmented Generation) Module
===================================================

Simple RAG implementation for symbol context lookup.
Uses local data sources (universe file, price cache, recent signals)
to provide enriched context for LLM reasoning without external vector DB.

Usage:
    from cognitive.symbol_rag import get_symbol_context, SymbolRAG

    # Get context for a symbol
    context = get_symbol_context("AAPL")

    # Or use the RAG instance directly
    rag = SymbolRAG()
    context = rag.retrieve("AAPL")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SymbolContext:
    """Enriched context for a symbol."""
    symbol: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap_tier: Optional[str] = None  # mega, large, mid, small

    # Price context
    current_price: Optional[float] = None
    price_change_1d: Optional[float] = None
    price_change_5d: Optional[float] = None
    sma_200: Optional[float] = None
    above_sma_200: Optional[bool] = None

    # Technical context
    ibs: Optional[float] = None
    rsi_2: Optional[float] = None
    atr_14: Optional[float] = None
    volatility_regime: Optional[str] = None  # low, medium, high

    # Signal history
    recent_signals: List[Dict[str, Any]] = field(default_factory=list)
    last_signal_date: Optional[str] = None
    win_rate_last_10: Optional[float] = None

    # Liquidity
    avg_volume_60d: Optional[float] = None
    avg_dollar_volume_60d: Optional[float] = None
    liquidity_tier: Optional[str] = None  # high, medium, low

    # Options
    has_options: bool = True

    # Meta
    retrieved_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_prompt_context(self) -> str:
        """Format context as text for LLM prompt."""
        lines = [f"Symbol: {self.symbol}"]

        if self.sector:
            lines.append(f"Sector: {self.sector}")
        if self.market_cap_tier:
            lines.append(f"Market Cap: {self.market_cap_tier}")

        if self.current_price:
            lines.append(f"Price: ${self.current_price:.2f}")
            if self.price_change_1d is not None:
                lines.append(f"1D Change: {self.price_change_1d:+.2f}%")

        if self.above_sma_200 is not None:
            trend = "Above" if self.above_sma_200 else "Below"
            lines.append(f"Trend: {trend} SMA(200)")

        if self.ibs is not None:
            lines.append(f"IBS: {self.ibs:.3f}")
        if self.rsi_2 is not None:
            lines.append(f"RSI(2): {self.rsi_2:.1f}")

        if self.volatility_regime:
            lines.append(f"Volatility: {self.volatility_regime}")
        if self.liquidity_tier:
            lines.append(f"Liquidity: {self.liquidity_tier}")

        if self.recent_signals:
            lines.append(f"Recent Signals: {len(self.recent_signals)} in last 30 days")
            if self.win_rate_last_10 is not None:
                lines.append(f"Win Rate (last 10): {self.win_rate_last_10:.1%}")

        return "\n".join(lines)


# =============================================================================
# Symbol RAG Implementation
# =============================================================================

class SymbolRAG:
    """
    Simple RAG for symbol context retrieval.

    Uses local data sources:
    - Universe file for sector/industry
    - Price cache for technical indicators
    - Signal logs for history
    """

    def __init__(
        self,
        universe_path: str = "data/universe/optionable_liquid_900.csv",
        cache_dir: str = "data/cache/polygon",
        signals_log: str = "logs/daily_picks.csv",
    ):
        self.universe_path = Path(universe_path)
        self.cache_dir = Path(cache_dir)
        self.signals_log = Path(signals_log)

        # Load universe metadata
        self._universe_df: Optional[pd.DataFrame] = None
        self._load_universe()

        logger.info(f"SymbolRAG initialized with {len(self._universe_df) if self._universe_df is not None else 0} symbols")

    def _load_universe(self) -> None:
        """Load universe file for symbol metadata."""
        if self.universe_path.exists():
            try:
                self._universe_df = pd.read_csv(self.universe_path)
            except Exception as e:
                logger.warning(f"Failed to load universe: {e}")
                self._universe_df = None

    def _get_symbol_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get metadata from universe file."""
        metadata = {}
        if self._universe_df is not None and 'symbol' in self._universe_df.columns:
            row = self._universe_df[self._universe_df['symbol'] == symbol]
            if not row.empty:
                row = row.iloc[0]
                metadata['sector'] = row.get('sector')
                metadata['industry'] = row.get('industry')
        return metadata

    def _get_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load recent price data from cache."""
        # Find most recent cache file
        pattern = f"{symbol}_*.csv"
        files = list(self.cache_dir.glob(pattern))
        if not files:
            return None

        # Get most recent file
        latest = max(files, key=lambda f: f.stat().st_mtime)
        try:
            df = pd.read_csv(latest)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            return df
        except Exception as e:
            logger.warning(f"Failed to load price data for {symbol}: {e}")
            return None

    def _compute_technicals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute technical indicators from price data."""
        if df.empty or len(df) < 200:
            return {}

        try:
            last = df.iloc[-1]
            close = last['close']
            high = last['high']
            low = last['low']

            # IBS
            ibs = (close - low) / (high - low) if high != low else 0.5

            # SMA 200
            sma_200 = df['close'].rolling(200).mean().iloc[-1]
            above_sma = close > sma_200

            # RSI(2) - simple calculation
            delta = df['close'].diff()
            gain = delta.clip(lower=0).rolling(2).mean()
            loss = (-delta.clip(upper=0)).rolling(2).mean()
            rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
            rsi_2 = 100 - (100 / (1 + rs))

            # ATR(14)
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift(1)).abs(),
                (df['low'] - df['close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean().iloc[-1]

            # Volatility regime
            atr_pct = atr_14 / close * 100
            if atr_pct < 2:
                vol_regime = "low"
            elif atr_pct < 4:
                vol_regime = "medium"
            else:
                vol_regime = "high"

            # Price changes
            price_1d = (close / df['close'].iloc[-2] - 1) * 100 if len(df) > 1 else None
            price_5d = (close / df['close'].iloc[-6] - 1) * 100 if len(df) > 5 else None

            # Volume
            avg_vol = df['volume'].tail(60).mean()
            avg_dollar_vol = (df['close'] * df['volume']).tail(60).mean()

            # Liquidity tier
            if avg_dollar_vol > 100_000_000:
                liq_tier = "high"
            elif avg_dollar_vol > 10_000_000:
                liq_tier = "medium"
            else:
                liq_tier = "low"

            return {
                'current_price': round(close, 2),
                'price_change_1d': round(price_1d, 2) if price_1d else None,
                'price_change_5d': round(price_5d, 2) if price_5d else None,
                'sma_200': round(sma_200, 2),
                'above_sma_200': above_sma,
                'ibs': round(ibs, 4),
                'rsi_2': round(rsi_2, 2),
                'atr_14': round(atr_14, 2),
                'volatility_regime': vol_regime,
                'avg_volume_60d': int(avg_vol),
                'avg_dollar_volume_60d': round(avg_dollar_vol, 0),
                'liquidity_tier': liq_tier,
            }
        except Exception as e:
            logger.warning(f"Failed to compute technicals: {e}")
            return {}

    def _get_recent_signals(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent signals for a symbol from log."""
        if not self.signals_log.exists():
            return []

        try:
            df = pd.read_csv(self.signals_log)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff = datetime.now() - timedelta(days=days)

            signals = df[
                (df['symbol'] == symbol) &
                (df['timestamp'] >= cutoff)
            ].to_dict('records')

            return signals
        except Exception as e:
            logger.warning(f"Failed to load signals: {e}")
            return []

    def retrieve(self, symbol: str) -> SymbolContext:
        """
        Retrieve enriched context for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            SymbolContext with all available data
        """
        context = SymbolContext(symbol=symbol)

        # Get metadata
        metadata = self._get_symbol_metadata(symbol)
        context.sector = metadata.get('sector')
        context.industry = metadata.get('industry')

        # Get price data and compute technicals
        df = self._get_price_data(symbol)
        if df is not None:
            technicals = self._compute_technicals(df)
            for key, value in technicals.items():
                if hasattr(context, key):
                    setattr(context, key, value)

        # Get recent signals
        signals = self._get_recent_signals(symbol)
        context.recent_signals = signals
        if signals:
            context.last_signal_date = signals[0].get('timestamp')

        return context

    def retrieve_batch(self, symbols: List[str]) -> Dict[str, SymbolContext]:
        """Retrieve context for multiple symbols."""
        return {symbol: self.retrieve(symbol) for symbol in symbols}

    def get_similar_symbols(self, symbol: str, n: int = 5) -> List[str]:
        """
        Find similar symbols based on sector/industry.
        Simple implementation without vector similarity.
        """
        if self._universe_df is None:
            return []

        metadata = self._get_symbol_metadata(symbol)
        sector = metadata.get('sector')
        industry = metadata.get('industry')

        if not sector:
            return []

        # Filter by same sector, prefer same industry
        same_sector = self._universe_df[self._universe_df.get('sector') == sector]
        same_sector = same_sector[same_sector['symbol'] != symbol]

        if industry and 'industry' in same_sector.columns:
            same_industry = same_sector[same_sector['industry'] == industry]
            if len(same_industry) >= n:
                return same_industry['symbol'].head(n).tolist()

        return same_sector['symbol'].head(n).tolist()


# =============================================================================
# Global Instance
# =============================================================================

_global_rag: Optional[SymbolRAG] = None


def get_symbol_rag() -> SymbolRAG:
    """Get or create global SymbolRAG instance."""
    global _global_rag
    if _global_rag is None:
        _global_rag = SymbolRAG()
    return _global_rag


def get_symbol_context(symbol: str) -> SymbolContext:
    """Convenience function to get symbol context."""
    rag = get_symbol_rag()
    return rag.retrieve(symbol)


def get_symbol_context_for_prompt(symbol: str) -> str:
    """Get symbol context formatted for LLM prompt."""
    context = get_symbol_context(symbol)
    return context.to_prompt_context()
