"""
Signal Confidence Module.

Provides ML-based confidence scoring for trading signals.
Combines multiple factors to produce a 0-1 confidence score:
- Technical alignment (momentum, trend, volatility)
- Anomaly risk (unusual market conditions)
- Historical pattern similarity
- Market regime compatibility
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.structured_log import jlog


class ConfidenceLevel(Enum):
    """Signal confidence levels."""
    VERY_LOW = "very_low"  # 0.0 - 0.2
    LOW = "low"  # 0.2 - 0.4
    MEDIUM = "medium"  # 0.4 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    VERY_HIGH = "very_high"  # 0.8 - 1.0


@dataclass
class ConfidenceConfig:
    """Configuration for signal confidence scoring."""
    # Factor weights (must sum to 1.0)
    momentum_weight: float = 0.25
    trend_weight: float = 0.25
    volatility_weight: float = 0.20
    anomaly_weight: float = 0.15
    pattern_weight: float = 0.15

    # Thresholds
    min_confidence: float = 0.0  # Floor for confidence score
    max_confidence: float = 1.0  # Ceiling for confidence score

    # RSI thresholds for long signals
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Trend requirements
    require_trend_alignment: bool = True
    sma_period: int = 200

    # Volatility
    min_volatility_percentile: float = 10.0  # Avoid too low volatility
    max_volatility_percentile: float = 90.0  # Avoid extreme volatility

    # Anomaly
    max_anomaly_score: float = 0.7  # Reduce confidence if anomaly is high


@dataclass
class ConfidenceResult:
    """Result of confidence calculation."""
    score: float  # 0-1 overall confidence
    level: ConfidenceLevel  # Categorical level
    factors: Dict[str, float]  # Individual factor scores
    reasons: List[str]  # Human-readable explanations
    details: Dict[str, Any] = field(default_factory=dict)


class SignalConfidence:
    """
    Signal confidence scorer.

    Analyzes trading signals and assigns confidence scores based on
    technical indicators, market conditions, and pattern analysis.

    Neural Integration (v2.7):
    - Episodic memory lookup for historical pattern performance
    - Confidence boost based on past similar trade outcomes
    """

    def __init__(self, config: Optional[ConfidenceConfig] = None, use_episodic_memory: bool = True):
        self.config = config or ConfidenceConfig()
        self.use_episodic_memory = use_episodic_memory
        self._episodic_memory = None  # Lazy-loaded
        self._validate_weights()

    @property
    def episodic_memory(self):
        """Lazy-load episodic memory to avoid circular imports."""
        if self._episodic_memory is None and self.use_episodic_memory:
            try:
                from cognitive.episodic_memory import get_episodic_memory
                self._episodic_memory = get_episodic_memory()
            except ImportError:
                pass
        return self._episodic_memory

    def _validate_weights(self) -> None:
        """Ensure weights sum to 1.0."""
        total = (
            self.config.momentum_weight +
            self.config.trend_weight +
            self.config.volatility_weight +
            self.config.anomaly_weight +
            self.config.pattern_weight
        )
        if abs(total - 1.0) > 0.01:
            jlog("confidence_weights_invalid", level="WARNING",
                 message=f"Weights sum to {total}, normalizing to 1.0")
            # Normalize
            self.config.momentum_weight /= total
            self.config.trend_weight /= total
            self.config.volatility_weight /= total
            self.config.anomaly_weight /= total
            self.config.pattern_weight /= total

    def score(
        self,
        signal_row: pd.Series,
        df: pd.DataFrame,
        side: str = "long"
    ) -> ConfidenceResult:
        """
        Calculate confidence score for a trading signal.

        Args:
            signal_row: Series with signal data (must include timestamp/index)
            df: DataFrame with OHLCV and feature data
            side: Signal direction ("long" or "short")

        Returns:
            ConfidenceResult with score, level, factors, and reasons
        """
        factors = {}
        reasons = []
        details = {}

        # Get data up to signal timestamp
        if 'timestamp' in signal_row:
            timestamp = signal_row['timestamp']
            mask = df.index <= timestamp if isinstance(df.index, pd.DatetimeIndex) else df['timestamp'] <= timestamp
            data = df[mask].copy()
        else:
            data = df.copy()

        if data.empty:
            return ConfidenceResult(
                score=0.0,
                level=ConfidenceLevel.VERY_LOW,
                factors={},
                reasons=["Insufficient data"],
                details={}
            )

        # Calculate individual factor scores
        factors['momentum'] = self._score_momentum(data, side)
        factors['trend'] = self._score_trend(data, side)
        factors['volatility'] = self._score_volatility(data)
        factors['anomaly'] = self._score_anomaly(data)
        factors['pattern'] = self._score_pattern(data, side)

        # Calculate weighted score
        score = (
            factors['momentum'] * self.config.momentum_weight +
            factors['trend'] * self.config.trend_weight +
            factors['volatility'] * self.config.volatility_weight +
            factors['anomaly'] * self.config.anomaly_weight +
            factors['pattern'] * self.config.pattern_weight
        )

        # Apply historical confidence boost from episodic memory (v2.7)
        historical_boost, historical_reason = self._get_historical_boost(signal_row, side)
        if historical_boost != 0.0:
            score += historical_boost
            factors['historical'] = 0.5 + historical_boost  # Normalize to 0-1 range
            details['historical_boost'] = historical_boost
            if historical_reason:
                reasons.append(historical_reason)

        # Clip to bounds
        score = np.clip(score, self.config.min_confidence, self.config.max_confidence)

        # Determine level
        level = self._score_to_level(score)

        # Generate reasons
        reasons = self._generate_reasons(factors, side, data)

        # Additional details
        details['side'] = side
        details['data_points'] = len(data)

        return ConfidenceResult(
            score=float(score),
            level=level,
            factors=factors,
            reasons=reasons,
            details=details
        )

    def _score_momentum(self, df: pd.DataFrame, side: str) -> float:
        """Score based on momentum indicators."""
        score = 0.5  # Neutral starting point

        last_row = df.iloc[-1]

        # RSI scoring
        rsi_cols = [c for c in df.columns if c.startswith('rsi_')]
        if rsi_cols:
            rsi_2 = last_row.get('rsi_2', np.nan)
            rsi_14 = last_row.get('rsi_14', np.nan)

            if side == "long":
                # For long, lower RSI = more confidence
                if pd.notna(rsi_2):
                    if rsi_2 <= 10:
                        score += 0.3
                    elif rsi_2 <= self.config.rsi_oversold:
                        score += 0.2
                    elif rsi_2 >= self.config.rsi_overbought:
                        score -= 0.2

                if pd.notna(rsi_14) and rsi_14 < 50:
                    score += 0.1
            else:
                # For short, higher RSI = more confidence
                if pd.notna(rsi_2):
                    if rsi_2 >= 90:
                        score += 0.3
                    elif rsi_2 >= self.config.rsi_overbought:
                        score += 0.2
                    elif rsi_2 <= self.config.rsi_oversold:
                        score -= 0.2

        # MACD scoring
        if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
            macd = last_row.get('MACD_12_26_9', np.nan)
            signal = last_row.get('MACDs_12_26_9', np.nan)

            if pd.notna(macd) and pd.notna(signal):
                if side == "long" and macd > signal:
                    score += 0.1
                elif side == "short" and macd < signal:
                    score += 0.1

        # Williams %R
        if 'williams_r' in df.columns:
            willr = last_row.get('williams_r', np.nan)
            if pd.notna(willr):
                if side == "long" and willr < -80:
                    score += 0.1
                elif side == "short" and willr > -20:
                    score += 0.1

        return np.clip(score, 0.0, 1.0)

    def _score_trend(self, df: pd.DataFrame, side: str) -> float:
        """Score based on trend indicators."""
        score = 0.5

        if df.empty:
            return score

        last_row = df.iloc[-1]
        close = last_row.get('close', np.nan)

        # SMA alignment
        sma_200 = last_row.get('sma_200', np.nan)
        if pd.notna(sma_200) and pd.notna(close):
            if side == "long":
                if close > sma_200:
                    score += 0.2  # Bullish trend
                else:
                    score -= 0.15  # Against trend
            else:
                if close < sma_200:
                    score += 0.2  # Bearish trend
                else:
                    score -= 0.15

        # Multiple SMA alignment
        sma_50 = last_row.get('sma_50', np.nan)
        last_row.get('sma_20', np.nan)

        if pd.notna(sma_50) and pd.notna(sma_200):
            if side == "long" and sma_50 > sma_200:
                score += 0.1
            elif side == "short" and sma_50 < sma_200:
                score += 0.1

        # ADX (trend strength)
        adx = last_row.get('ADX_14', np.nan)
        if pd.notna(adx):
            if adx > 25:
                score += 0.15  # Strong trend
            elif adx < 15:
                score -= 0.1  # Weak/ranging

        # Supertrend
        st_col = [c for c in df.columns if c.startswith('SUPERTd_')]
        if st_col:
            st_direction = last_row.get(st_col[0], np.nan)
            if pd.notna(st_direction):
                if side == "long" and st_direction == 1:
                    score += 0.1
                elif side == "short" and st_direction == -1:
                    score += 0.1

        return np.clip(score, 0.0, 1.0)

    def _score_volatility(self, df: pd.DataFrame) -> float:
        """Score based on volatility (prefer moderate volatility)."""
        score = 0.5

        if df.empty:
            return score

        # ATR percentile
        if 'natr_14' in df.columns:
            natr = df['natr_14'].dropna()
            if len(natr) > 20:
                current_natr = natr.iloc[-1]
                percentile = (natr < current_natr).sum() / len(natr) * 100

                # Prefer moderate volatility (20-80 percentile)
                if 20 <= percentile <= 80:
                    score += 0.2
                elif percentile < self.config.min_volatility_percentile:
                    score -= 0.2  # Too quiet
                elif percentile > self.config.max_volatility_percentile:
                    score -= 0.3  # Too volatile

        # Bollinger Band width
        if 'bb_width' in df.columns:
            bb_width = df['bb_width'].iloc[-1]
            if pd.notna(bb_width):
                # Moderate width is good
                if 0.02 < bb_width < 0.10:
                    score += 0.1

        return np.clip(score, 0.0, 1.0)

    def _score_anomaly(self, df: pd.DataFrame) -> float:
        """Score based on anomaly detection (penalize anomalies)."""
        score = 1.0  # Start high, reduce for anomalies

        if df.empty:
            return score

        last_row = df.iloc[-1]

        # Combined anomaly score
        anomaly_combined = last_row.get('anomaly_combined', 0.0)
        if pd.notna(anomaly_combined):
            if anomaly_combined > self.config.max_anomaly_score:
                score -= 0.5  # Significant penalty
            elif anomaly_combined > 0.5:
                score -= 0.3
            elif anomaly_combined > 0.3:
                score -= 0.1

        # Individual anomaly types
        anomaly_volume = last_row.get('anomaly_volume', 0.0)
        if pd.notna(anomaly_volume) and anomaly_volume > 0.7:
            score -= 0.1  # Volume spike warning

        return np.clip(score, 0.0, 1.0)

    def _score_pattern(self, df: pd.DataFrame, side: str) -> float:
        """Score based on price patterns."""
        score = 0.5

        if df.empty:
            return score

        last_row = df.iloc[-1]

        # IBS (Internal Bar Strength)
        ibs = last_row.get('ibs', np.nan)
        if pd.notna(ibs):
            if side == "long" and ibs < 0.2:
                score += 0.2  # Low IBS = good for long
            elif side == "short" and ibs > 0.8:
                score += 0.2  # High IBS = good for short

        # Inside bar (continuation pattern)
        inside_bar = last_row.get('inside_bar', 0.0)
        if inside_bar == 1.0:
            score += 0.1  # Breakout potential

        # Distance from recent high/low
        dist_20d_low = last_row.get('dist_from_20d_low', np.nan)
        dist_20d_high = last_row.get('dist_from_20d_high', np.nan)

        if pd.notna(dist_20d_low) and side == "long":
            if -0.05 < dist_20d_low < 0.02:
                score += 0.15  # Near 20-day low

        if pd.notna(dist_20d_high) and side == "short":
            if -0.02 < dist_20d_high < 0.05:
                score += 0.15  # Near 20-day high

        # Gap analysis
        gap_pct = last_row.get('gap_pct', np.nan)
        if pd.notna(gap_pct):
            if side == "long" and gap_pct < -0.02:
                score += 0.1  # Gap down = reversal opportunity
            elif side == "short" and gap_pct > 0.02:
                score += 0.1  # Gap up = reversal opportunity

        return np.clip(score, 0.0, 1.0)

    def _get_historical_boost(self, signal_row: pd.Series, side: str) -> Tuple[float, Optional[str]]:
        """
        Get confidence boost based on historical performance of similar trades.

        Uses episodic memory to find past trades with similar context and
        adjusts confidence based on their win rate.

        Returns:
            Tuple of (boost_value, reason_string)
            - boost_value: -0.2 to +0.2 adjustment
            - reason_string: Human-readable explanation or None
        """
        if not self.episodic_memory:
            return 0.0, None

        try:
            # Build context for similarity search
            symbol = signal_row.get('symbol', '')
            pattern_type = str(signal_row.get('strategy', '')).lower()

            if not symbol:
                return 0.0, None

            # Query episodic memory for similar past trades
            context = {
                'symbol': symbol,
                'side': side,
                'pattern_type': pattern_type,
            }

            # Find similar episodes
            similar_episodes = self.episodic_memory.find_similar(context, top_k=20)

            if not similar_episodes or len(similar_episodes) < 5:
                return 0.0, None  # Not enough history

            # Calculate historical win rate
            wins = sum(1 for ep in similar_episodes if ep.outcome and ep.outcome.get('won', False))
            total = len(similar_episodes)
            win_rate = wins / total

            # Calculate boost based on historical win rate
            # Neutral at 50%, boost for higher, penalty for lower
            boost = (win_rate - 0.5) * 0.4  # Max +/-0.2

            # Build reason string
            if boost > 0.05:
                reason = f"Historical: {win_rate:.0%} win rate on {total} similar trades (+{boost:.2f})"
            elif boost < -0.05:
                reason = f"Historical: {win_rate:.0%} win rate on {total} similar trades ({boost:.2f})"
            else:
                reason = None  # Near neutral, no reason needed

            return round(boost, 3), reason

        except Exception as e:
            # Log but don't fail - historical boost is optional
            import logging
            logging.getLogger(__name__).debug(f"Historical boost calculation error: {e}")
            return 0.0, None

    def _score_to_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to categorical level."""
        if score < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif score < 0.4:
            return ConfidenceLevel.LOW
        elif score < 0.6:
            return ConfidenceLevel.MEDIUM
        elif score < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    def _generate_reasons(self, factors: Dict[str, float], side: str, df: pd.DataFrame) -> List[str]:
        """Generate human-readable reasons for the confidence score."""
        reasons = []

        if factors.get('momentum', 0.5) > 0.7:
            reasons.append(f"Strong momentum alignment for {side}")
        elif factors.get('momentum', 0.5) < 0.3:
            reasons.append(f"Weak momentum for {side}")

        if factors.get('trend', 0.5) > 0.7:
            reasons.append(f"Trend supports {side} position")
        elif factors.get('trend', 0.5) < 0.3:
            reasons.append("Trading against prevailing trend")

        if factors.get('volatility', 0.5) < 0.3:
            reasons.append("Volatility conditions unfavorable")
        elif factors.get('volatility', 0.5) > 0.7:
            reasons.append("Favorable volatility conditions")

        if factors.get('anomaly', 1.0) < 0.5:
            reasons.append("Unusual market conditions detected")

        if factors.get('pattern', 0.5) > 0.7:
            reasons.append("Favorable price pattern")

        if not reasons:
            reasons.append("Mixed signals, moderate confidence")

        return reasons

    def score_signals(
        self,
        signals_df: pd.DataFrame,
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Score multiple signals at once.

        Args:
            signals_df: DataFrame with signals (timestamp, symbol, side columns)
            price_df: DataFrame with OHLCV and feature data

        Returns:
            signals_df with confidence columns added
        """
        if signals_df.empty:
            return signals_df

        signals_df = signals_df.copy()
        confidence_scores = []
        confidence_levels = []

        for idx, signal_row in signals_df.iterrows():
            result = self.score(
                signal_row,
                price_df,
                side=signal_row.get('side', 'long')
            )
            confidence_scores.append(result.score)
            confidence_levels.append(result.level.value)

        signals_df['confidence'] = confidence_scores
        signals_df['confidence_level'] = confidence_levels

        return signals_df


# Convenience function
def compute_signal_confidence(
    signal: pd.Series,
    df: pd.DataFrame,
    side: str = "long",
    config: Optional[ConfidenceConfig] = None
) -> float:
    """
    Compute confidence score for a single signal.

    Args:
        signal: Signal row data
        df: DataFrame with OHLCV and feature data
        side: Signal direction
        config: Optional confidence configuration

    Returns:
        Confidence score (0-1)
    """
    scorer = SignalConfidence(config)
    result = scorer.score(signal, df, side)
    return result.score
