"""
Signal Adjudicator - Ranks quality-passed signals by 4 strategic factors.

This module COMPLEMENTS (does not replace) the SignalQualityGate.
The SignalQualityGate filters signals for quality (pass/fail).
The SignalAdjudicator ranks the survivors by 4 additional factors.

Pipeline Position:
    DualStrategyScanner -> SignalQualityGate -> SignalAdjudicator -> Top 3 Picks

Scoring Weights (total = 100):
1. Signal Strength (40%): Lower IBS = higher score, RSI tiering
2. Pattern Confluence (30%): Multiple oversold patterns aligned = bonus
3. Volatility Contraction (20%): BB width at 20th percentile = bonus
4. Relative Sector Strength (10%): Sector outperforming SPY = bonus

RSI Tiering:
- RSI(2) < 5.0 = "extreme" -> max score (100)
- RSI(2) < 10.0 = "near-extreme" -> partial score (60)
- RSI(2) >= 10.0 = no bonus (0)

Usage:
    from cognitive.signal_adjudicator import get_adjudicator, adjudicate_signals

    adjudicator = get_adjudicator()
    ranked_signals = adjudicator.adjudicate(
        signals=quality_filtered_signals,
        price_data=combined_df,
        spy_data=spy_df,
    )
    top_3 = ranked_signals.head(3)

    # Or use convenience function:
    top_3 = adjudicate_signals(signals, price_data, spy_data, max_signals=3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AdjudicatorConfig:
    """Configuration for signal adjudication weights and thresholds."""

    # Weight distribution (must sum to 100)
    signal_strength_weight: float = 40.0
    pattern_confluence_weight: float = 30.0
    volatility_contraction_weight: float = 20.0
    sector_strength_weight: float = 10.0

    # Signal Strength thresholds
    ibs_extreme_threshold: float = 0.05   # IBS < 0.05 = max score
    ibs_good_threshold: float = 0.10      # IBS < 0.10 = partial score
    rsi2_extreme_threshold: float = 5.0   # RSI2 < 5 = extreme (100 pts)
    rsi2_near_threshold: float = 10.0     # RSI2 < 10 = near-extreme (60 pts)

    # Pattern Confluence settings
    min_patterns_for_bonus: int = 2       # 2+ patterns = confluence bonus

    # Volatility Contraction thresholds
    bb_width_percentile_threshold: float = 20.0  # Below 20th percentile = bonus
    bb_lookback_days: int = 60                   # Days for percentile calc

    # Sector Strength settings
    sector_outperform_threshold: float = 0.02    # 2% above SPY = full bonus
    sector_lookback_days: int = 20               # Days for return calc

    def validate(self) -> bool:
        """Validate that weights sum to 100."""
        total = (
            self.signal_strength_weight +
            self.pattern_confluence_weight +
            self.volatility_contraction_weight +
            self.sector_strength_weight
        )
        return abs(total - 100.0) < 0.01


# ============================================================================
# Result Data Class
# ============================================================================

@dataclass
class AdjudicationResult:
    """Result of signal adjudication with full breakdown."""

    # Identity
    symbol: str
    strategy: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Final score (0-100)
    adjudication_score: float = 0.0

    # Component scores (0-100 each, weighted in final)
    signal_strength_score: float = 0.0
    pattern_confluence_score: float = 0.0
    volatility_contraction_score: float = 0.0
    sector_strength_score: float = 0.0

    # Raw values used in scoring
    ibs_value: float = 0.0
    rsi2_value: float = 0.0
    rsi2_tier: str = "normal"  # "extreme", "near_extreme", "normal"
    bb_width_percentile: float = 50.0
    sector: str = "Unknown"
    sector_return: float = 0.0
    spy_return: float = 0.0
    sector_vs_spy: float = 0.0

    # Pattern confluence details
    patterns_detected: List[str] = field(default_factory=list)
    pattern_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/storage."""
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "adjudication_score": round(self.adjudication_score, 2),
            "components": {
                "signal_strength": round(self.signal_strength_score, 2),
                "pattern_confluence": round(self.pattern_confluence_score, 2),
                "volatility_contraction": round(self.volatility_contraction_score, 2),
                "sector_strength": round(self.sector_strength_score, 2),
            },
            "raw_values": {
                "ibs": round(self.ibs_value, 4),
                "rsi2": round(self.rsi2_value, 2),
                "rsi2_tier": self.rsi2_tier,
                "bb_width_percentile": round(self.bb_width_percentile, 1),
                "sector": self.sector,
                "sector_vs_spy": round(self.sector_vs_spy * 100, 2),
            },
            "patterns": {
                "detected": self.patterns_detected,
                "count": self.pattern_count,
            },
        }


# ============================================================================
# Signal Adjudicator
# ============================================================================

class SignalAdjudicator:
    """
    Adjudicates quality-passed signals by ranking them on 4 strategic factors.

    This COMPLEMENTS the SignalQualityGate - it does NOT replace it.
    The flow is: QualityGate filters -> Adjudicator ranks survivors.

    Factors:
    1. Signal Strength (40%): How extreme is the IBS/RSI setup?
    2. Pattern Confluence (30%): How many confirming patterns?
    3. Volatility Contraction (20%): Is volatility compressed (ready to expand)?
    4. Sector Strength (10%): Is the sector outperforming?
    """

    def __init__(self, config: Optional[AdjudicatorConfig] = None):
        self.config = config or AdjudicatorConfig()
        if not self.config.validate():
            logger.warning("Adjudicator weights do not sum to 100!")

        # Lazy load sector map
        self._sector_map = None

        logger.info(
            "SignalAdjudicator initialized with weights: strength=%.0f, "
            "confluence=%.0f, volatility=%.0f, sector=%.0f",
            self.config.signal_strength_weight,
            self.config.pattern_confluence_weight,
            self.config.volatility_contraction_weight,
            self.config.sector_strength_weight,
        )

    @property
    def sector_map(self) -> Dict[str, str]:
        """Lazy load sector mapping from correlation_limits."""
        if self._sector_map is None:
            try:
                from risk.advanced.correlation_limits import SECTOR_MAP
                self._sector_map = SECTOR_MAP
            except ImportError:
                logger.warning("SECTOR_MAP not available, using empty map")
                self._sector_map = {}
        return self._sector_map

    def adjudicate(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Rank signals by 4 adjudication factors.

        Args:
            signals: Quality-filtered signals DataFrame (from SignalQualityGate)
            price_data: Combined OHLCV data for all symbols
            spy_data: SPY data for sector relative strength

        Returns:
            signals DataFrame with adjudication_score column, sorted descending
        """
        if signals.empty:
            logger.debug("No signals to adjudicate")
            return signals

        results = []

        for _, row in signals.iterrows():
            signal = row.to_dict()
            symbol = signal.get('symbol', '')

            # Get symbol-specific price data
            if 'symbol' in price_data.columns:
                symbol_data = price_data[price_data['symbol'] == symbol].copy()
            else:
                symbol_data = price_data.copy()

            if symbol_data.empty:
                logger.debug(f"No price data for {symbol}, skipping adjudication")
                continue

            # Calculate all 4 component scores
            result = self._adjudicate_single(signal, symbol_data, spy_data)
            results.append({
                **signal,
                'adjudication_score': result.adjudication_score,
                'adj_signal_strength': result.signal_strength_score,
                'adj_pattern_confluence': result.pattern_confluence_score,
                'adj_volatility_contraction': result.volatility_contraction_score,
                'adj_sector_strength': result.sector_strength_score,
                'adj_patterns_detected': ','.join(result.patterns_detected),
                'adj_rsi2_tier': result.rsi2_tier,
                'adj_bb_percentile': result.bb_width_percentile,
                'adj_sector': result.sector,
            })

        if not results:
            return signals

        result_df = pd.DataFrame(results)

        # Sort by adjudication score descending
        result_df = result_df.sort_values(
            'adjudication_score',
            ascending=False
        ).reset_index(drop=True)

        logger.info(
            "Adjudicated %d signals: top score=%.1f, bottom=%.1f",
            len(result_df),
            result_df['adjudication_score'].iloc[0] if len(result_df) > 0 else 0,
            result_df['adjudication_score'].iloc[-1] if len(result_df) > 0 else 0,
        )

        return result_df

    def _adjudicate_single(
        self,
        signal: Dict[str, Any],
        symbol_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame],
    ) -> AdjudicationResult:
        """Adjudicate a single signal."""
        symbol = signal.get('symbol', '')
        strategy = signal.get('strategy', '')

        result = AdjudicationResult(symbol=symbol, strategy=strategy)

        # 1. Signal Strength (40%)
        strength_score, ibs_val, rsi_val, rsi_tier = self._score_signal_strength(
            signal, symbol_data
        )
        result.signal_strength_score = strength_score
        result.ibs_value = ibs_val
        result.rsi2_value = rsi_val
        result.rsi2_tier = rsi_tier

        # 2. Pattern Confluence (30%)
        confluence_score, patterns = self._score_pattern_confluence(
            signal, symbol_data
        )
        result.pattern_confluence_score = confluence_score
        result.patterns_detected = patterns
        result.pattern_count = len(patterns)

        # 3. Volatility Contraction (20%)
        vol_score, bb_pct = self._score_volatility_contraction(symbol_data)
        result.volatility_contraction_score = vol_score
        result.bb_width_percentile = bb_pct

        # 4. Sector Strength (10%)
        sector_score, sector, sect_ret, spy_ret = self._score_sector_strength(
            symbol, symbol_data, spy_data
        )
        result.sector_strength_score = sector_score
        result.sector = sector
        result.sector_return = sect_ret
        result.spy_return = spy_ret
        result.sector_vs_spy = sect_ret - spy_ret

        # Calculate weighted final score
        result.adjudication_score = (
            (strength_score / 100) * self.config.signal_strength_weight +
            (confluence_score / 100) * self.config.pattern_confluence_weight +
            (vol_score / 100) * self.config.volatility_contraction_weight +
            (sector_score / 100) * self.config.sector_strength_weight
        )

        return result

    def _score_signal_strength(
        self,
        signal: Dict[str, Any],
        symbol_data: pd.DataFrame,
    ) -> Tuple[float, float, float, str]:
        """
        Score signal strength based on IBS/RSI extremity (0-100).

        RSI Tiering:
        - RSI(2) < 5.0 = "extreme" -> 100 points
        - RSI(2) < 10.0 = "near_extreme" -> 60 points
        - RSI(2) >= 10.0 = "normal" -> 0 points

        IBS Scoring:
        - IBS < 0.05 = 100 points
        - IBS 0.05-0.10 = scaled 50-100
        - IBS > 0.10 = scaled 0-50

        Final = (IBS_score + RSI_score) / 2
        """
        # Get IBS value (from signal or calculate)
        ibs_val = signal.get('ibs', None)
        if ibs_val is None and not symbol_data.empty:
            # Calculate IBS from latest bar
            try:
                close_col = 'close' if 'close' in symbol_data.columns else 'Close'
                high_col = 'high' if 'high' in symbol_data.columns else 'High'
                low_col = 'low' if 'low' in symbol_data.columns else 'Low'

                latest = symbol_data.iloc[-1]
                h, l, c = latest[high_col], latest[low_col], latest[close_col]
                ibs_val = (c - l) / (h - l + 1e-8)
            except Exception:
                ibs_val = 0.5  # Neutral default

        ibs_val = float(ibs_val) if ibs_val is not None else 0.5

        # Get RSI(2) value (from signal or calculate)
        rsi_val = signal.get('rsi2', signal.get('rsi', None))
        if rsi_val is None and not symbol_data.empty:
            try:
                close_col = 'close' if 'close' in symbol_data.columns else 'Close'
                closes = symbol_data[close_col].astype(float)
                rsi_val = self._calculate_rsi(closes, period=2)
            except Exception:
                rsi_val = 50.0  # Neutral default

        rsi_val = float(rsi_val) if rsi_val is not None else 50.0

        # Score IBS (lower = better)
        if ibs_val < self.config.ibs_extreme_threshold:
            ibs_score = 100.0
        elif ibs_val < self.config.ibs_good_threshold:
            # Scale 50-100 for IBS between 0.05 and 0.10
            ratio = (self.config.ibs_good_threshold - ibs_val) / (
                self.config.ibs_good_threshold - self.config.ibs_extreme_threshold
            )
            ibs_score = 50 + 50 * ratio
        else:
            # Scale 0-50 for IBS above 0.10
            ratio = max(0, 1 - (ibs_val - self.config.ibs_good_threshold) / 0.10)
            ibs_score = 50 * ratio

        # Score RSI with tiering
        if rsi_val < self.config.rsi2_extreme_threshold:
            rsi_score = 100.0
            rsi_tier = "extreme"
        elif rsi_val < self.config.rsi2_near_threshold:
            rsi_score = 60.0
            rsi_tier = "near_extreme"
        else:
            rsi_score = 0.0
            rsi_tier = "normal"

        # Combined score (average of IBS and RSI)
        final_score = (ibs_score + rsi_score) / 2

        return final_score, ibs_val, rsi_val, rsi_tier

    def _score_pattern_confluence(
        self,
        signal: Dict[str, Any],
        symbol_data: pd.DataFrame,
    ) -> Tuple[float, List[str]]:
        """
        Score pattern confluence - how many confirming patterns (0-100).

        Patterns checked:
        - IBS < 0.10 (oversold)
        - RSI2 < 10 (oversold)
        - Price < lower Bollinger Band (oversold)
        - Down 3+ consecutive days
        - Volume spike > 2x average (capitulation)
        - Near 20-day low (support sweep)

        Scoring:
        - 1 pattern: 30 points
        - 2 patterns: 60 points
        - 3 patterns: 80 points
        - 4+ patterns: 100 points
        """
        patterns = []

        if symbol_data.empty:
            return 0.0, patterns

        try:
            close_col = 'close' if 'close' in symbol_data.columns else 'Close'
            high_col = 'high' if 'high' in symbol_data.columns else 'High'
            low_col = 'low' if 'low' in symbol_data.columns else 'Low'
            vol_col = 'volume' if 'volume' in symbol_data.columns else 'Volume'

            latest = symbol_data.iloc[-1]
            closes = symbol_data[close_col].astype(float)

            # Pattern 1: IBS < 0.10
            h, l, c = latest[high_col], latest[low_col], latest[close_col]
            ibs = (c - l) / (h - l + 1e-8)
            if ibs < 0.10:
                patterns.append("IBS_OVERSOLD")

            # Pattern 2: RSI2 < 10
            rsi = self._calculate_rsi(closes, period=2)
            if rsi < 10:
                patterns.append("RSI2_OVERSOLD")

            # Pattern 3: Price below lower Bollinger Band
            if len(closes) >= 20:
                sma20 = closes.rolling(20).mean().iloc[-1]
                std20 = closes.rolling(20).std().iloc[-1]
                lower_bb = sma20 - 2 * std20
                if c < lower_bb:
                    patterns.append("BELOW_LOWER_BB")

            # Pattern 4: Down 3+ consecutive days
            if len(closes) >= 4:
                changes = closes.diff().tail(4).dropna()
                if (changes < 0).all():
                    patterns.append("DOWN_STREAK")

            # Pattern 5: Volume spike (> 2x 20-day average)
            if vol_col in symbol_data.columns and len(symbol_data) >= 20:
                volumes = symbol_data[vol_col].astype(float)
                avg_vol = volumes.tail(20).mean()
                latest_vol = volumes.iloc[-1]
                if latest_vol > 2 * avg_vol:
                    patterns.append("VOLUME_SPIKE")

            # Pattern 6: Near 20-day low
            if len(symbol_data) >= 20:
                low_20 = symbol_data[low_col].tail(20).min()
                if l <= low_20 * 1.01:  # Within 1% of 20-day low
                    patterns.append("NEAR_20D_LOW")

        except Exception as e:
            logger.debug(f"Pattern confluence error: {e}")

        # Score based on pattern count
        n_patterns = len(patterns)
        if n_patterns >= 4:
            score = 100.0
        elif n_patterns == 3:
            score = 80.0
        elif n_patterns == 2:
            score = 60.0
        elif n_patterns == 1:
            score = 30.0
        else:
            score = 0.0

        return score, patterns

    def _score_volatility_contraction(
        self,
        symbol_data: pd.DataFrame,
    ) -> Tuple[float, float]:
        """
        Score volatility contraction - BB width at low percentile (0-100).

        Compressed volatility often precedes explosive moves.

        Scoring:
        - BB width below 10th percentile: 100 points
        - BB width below 20th percentile: 80 points
        - BB width below 30th percentile: 60 points
        - BB width below 50th percentile: 40 points
        - BB width above 50th percentile: 0 points
        """
        if symbol_data.empty or len(symbol_data) < 20:
            return 0.0, 50.0

        try:
            close_col = 'close' if 'close' in symbol_data.columns else 'Close'
            closes = symbol_data[close_col].astype(float)

            # Calculate Bollinger Band width
            sma20 = closes.rolling(20).mean()
            std20 = closes.rolling(20).std()
            bb_width = (2 * std20) / sma20  # Width as % of middle band

            # Get lookback data for percentile
            lookback = min(self.config.bb_lookback_days, len(bb_width))
            bb_width_history = bb_width.tail(lookback).dropna()

            if len(bb_width_history) < 10:
                return 0.0, 50.0

            current_width = bb_width.iloc[-1]

            # Calculate percentile
            percentile = (bb_width_history < current_width).sum() / len(bb_width_history) * 100

            # Score based on percentile (lower = more contracted = higher score)
            if percentile <= 10:
                score = 100.0
            elif percentile <= 20:
                score = 80.0
            elif percentile <= 30:
                score = 60.0
            elif percentile <= 50:
                score = 40.0
            else:
                score = 0.0

            return score, percentile

        except Exception as e:
            logger.debug(f"Volatility contraction error: {e}")
            return 0.0, 50.0

    def _score_sector_strength(
        self,
        symbol: str,
        symbol_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame],
    ) -> Tuple[float, str, float, float]:
        """
        Score relative sector strength vs SPY (0-100).

        Signals in outperforming sectors get a bonus.

        Scoring:
        - Sector > SPY by 4%+: 100 points
        - Sector > SPY by 2-4%: 80 points
        - Sector > SPY by 0-2%: 50 points
        - Sector < SPY: 0 points
        """
        sector = self.sector_map.get(symbol, "Unknown")

        if spy_data is None or spy_data.empty or symbol_data.empty:
            return 50.0, sector, 0.0, 0.0  # Neutral if no data

        lookback = self.config.sector_lookback_days

        try:
            close_col = 'close' if 'close' in symbol_data.columns else 'Close'

            # Calculate symbol return over lookback period
            sym_closes = symbol_data[close_col].astype(float).tail(lookback + 1)
            if len(sym_closes) < 2:
                return 50.0, sector, 0.0, 0.0

            symbol_return = (sym_closes.iloc[-1] / sym_closes.iloc[0]) - 1

            # Calculate SPY return over same period
            spy_close_col = 'close' if 'close' in spy_data.columns else 'Close'
            spy_closes = spy_data[spy_close_col].astype(float).tail(lookback + 1)
            if len(spy_closes) < 2:
                return 50.0, sector, symbol_return, 0.0

            spy_return = (spy_closes.iloc[-1] / spy_closes.iloc[0]) - 1

            # Relative performance
            relative_perf = symbol_return - spy_return

            # Score based on outperformance
            if relative_perf >= 0.04:  # 4%+ outperformance
                score = 100.0
            elif relative_perf >= 0.02:  # 2-4% outperformance
                score = 80.0
            elif relative_perf >= 0:  # 0-2% outperformance
                score = 50.0
            else:  # Underperformance
                score = 0.0

            return score, sector, symbol_return, spy_return

        except Exception as e:
            logger.debug(f"Sector strength error: {e}")
            return 50.0, sector, 0.0, 0.0

    def _calculate_rsi(self, closes: pd.Series, period: int = 2) -> float:
        """Calculate RSI for given period."""
        if len(closes) < period + 1:
            return 50.0

        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    def get_top_n(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame] = None,
        n: int = 3,
    ) -> pd.DataFrame:
        """
        Convenience method to adjudicate and get top N signals.

        Args:
            signals: Quality-filtered signals DataFrame
            price_data: Combined OHLCV data
            spy_data: SPY data for sector relative strength
            n: Number of top signals to return

        Returns:
            Top N signals sorted by adjudication score
        """
        ranked = self.adjudicate(signals, price_data, spy_data)
        return ranked.head(n)


# ============================================================================
# Singleton Instance
# ============================================================================

_adjudicator: Optional[SignalAdjudicator] = None


def get_adjudicator(config: Optional[AdjudicatorConfig] = None) -> SignalAdjudicator:
    """Get or create singleton SignalAdjudicator."""
    global _adjudicator
    if _adjudicator is None or config is not None:
        _adjudicator = SignalAdjudicator(config)
    return _adjudicator


def adjudicate_signals(
    signals: pd.DataFrame,
    price_data: pd.DataFrame,
    spy_data: Optional[pd.DataFrame] = None,
    max_signals: int = 3,
    config: Optional[AdjudicatorConfig] = None,
) -> pd.DataFrame:
    """
    Convenience function to adjudicate and select top signals.

    Args:
        signals: Quality-filtered signals DataFrame
        price_data: Combined OHLCV data for all symbols
        spy_data: SPY data for sector relative strength
        max_signals: Maximum number of signals to return
        config: Optional custom configuration

    Returns:
        Top signals sorted by adjudication score

    Example:
        from cognitive.signal_adjudicator import adjudicate_signals

        # After SignalQualityGate has filtered signals:
        top_picks = adjudicate_signals(
            signals=quality_passed_signals,
            price_data=combined_df,
            spy_data=spy_df,
            max_signals=3,
        )
    """
    adjudicator = get_adjudicator(config)
    return adjudicator.get_top_n(signals, price_data, spy_data, n=max_signals)
