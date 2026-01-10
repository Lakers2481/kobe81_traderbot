"""
Conviction Score Calculator - 0-100 Scoring System.

Calculates conviction scores for trading signals using 6 factors:
1. Technical setup quality (0-25 points)
2. Volume confirmation (0-15 points)
3. Market context (0-15 points)
4. Mean reversion strength (0-15 points)
5. Risk/Reward quality (0-15 points)
6. Timing quality (0-15 points)

Total: 0-100 points

Optimized for Kobe's mean-reversion strategies (RSI-2, IBS).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConvictionBreakdown:
    """Detailed breakdown of conviction score components."""
    technical_score: int      # 0-25
    volume_score: int         # 0-15
    market_score: int         # 0-15
    reversion_score: int      # 0-15
    risk_reward_score: int    # 0-15
    timing_score: int         # 0-15
    total_score: int          # 0-100
    tier: str                 # EXCEPTIONAL, EXCELLENT, GOOD, ACCEPTABLE, WEAK
    action: str               # STRONG BUY, BUY, CONSIDER, PASS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "technical_score": self.technical_score,
            "volume_score": self.volume_score,
            "market_score": self.market_score,
            "reversion_score": self.reversion_score,
            "risk_reward_score": self.risk_reward_score,
            "timing_score": self.timing_score,
            "total_score": self.total_score,
            "tier": self.tier,
            "action": self.action,
        }


class ConvictionScorer:
    """
    Conviction scoring system for mean-reversion signals.

    Provides 0-100 scoring with detailed breakdown of contributing factors.
    Optimized for RSI-2 and IBS strategies.
    """

    # Conviction tier definitions
    TIERS = {
        (90, 100): {"tier": "EXCEPTIONAL", "action": "STRONG BUY"},
        (80, 89): {"tier": "EXCELLENT", "action": "BUY"},
        (70, 79): {"tier": "GOOD", "action": "BUY"},
        (60, 69): {"tier": "ACCEPTABLE", "action": "CONSIDER"},
        (0, 59): {"tier": "WEAK", "action": "PASS"}
    }

    def __init__(self):
        """Initialize conviction scorer."""
        self.cache = {}

    def calculate_conviction(
        self,
        signal: Dict[str, Any],
        price_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame] = None,
        vix_level: Optional[float] = None
    ) -> ConvictionBreakdown:
        """
        Calculate complete conviction score for signal.

        Args:
            signal: Signal dict with symbol, entry_price, stop_loss, take_profit, etc.
            price_data: Price history DataFrame with OHLCV for the symbol
            spy_data: Optional SPY price history for market context
            vix_level: Optional current VIX level

        Returns:
            ConvictionBreakdown with detailed scoring
        """
        # Calculate each component
        technical_score = self._score_technical_setup(signal, price_data)
        volume_score = self._score_volume_confirmation(price_data)
        market_score = self._score_market_context(spy_data, vix_level)
        reversion_score = self._score_reversion_strength(signal, price_data)
        risk_reward_score = self._score_risk_reward(signal)
        timing_score = self._score_timing(signal, price_data)

        # Calculate total
        total = (technical_score + volume_score + market_score +
                 reversion_score + risk_reward_score + timing_score)

        # Determine tier
        tier_info = self._get_tier_info(total)

        return ConvictionBreakdown(
            technical_score=technical_score,
            volume_score=volume_score,
            market_score=market_score,
            reversion_score=reversion_score,
            risk_reward_score=risk_reward_score,
            timing_score=timing_score,
            total_score=total,
            tier=tier_info['tier'],
            action=tier_info['action']
        )

    def _score_technical_setup(self, signal: Dict, df: pd.DataFrame) -> int:
        """
        Score technical setup quality (0-25 points).

        Factors:
        - RSI extremity (0-10 points)
        - Price relative to SMA(200) (0-8 points)
        - ATR-normalized distance from mean (0-7 points)
        """
        score = 0

        # RSI extremity (prefer RSI-2 < 10 for entries)
        rsi_2 = signal.get('rsi_2', 50)
        if rsi_2 <= 2:
            score += 10
        elif rsi_2 <= 5:
            score += 8
        elif rsi_2 <= 10:
            score += 6
        elif rsi_2 <= 15:
            score += 4
        elif rsi_2 <= 20:
            score += 2

        # Price relative to SMA(200) - should be above for long trades
        if len(df) >= 200:
            close_col = 'close' if 'close' in df.columns else 'Close'
            close = df[close_col].iloc[-1]
            sma_200 = df[close_col].tail(200).mean()

            if close > sma_200:
                pct_above = ((close - sma_200) / sma_200) * 100
                if pct_above > 10:
                    score += 8
                elif pct_above > 5:
                    score += 6
                elif pct_above > 2:
                    score += 4
                else:
                    score += 2
        else:
            score += 4  # Neutral if insufficient data

        # ATR-normalized oversold (price below 20-day mean by > 1 ATR)
        if len(df) >= 20:
            close_col = 'close' if 'close' in df.columns else 'Close'
            high_col = 'high' if 'high' in df.columns else 'High'
            low_col = 'low' if 'low' in df.columns else 'Low'

            close = df[close_col].iloc[-1]
            mean_20 = df[close_col].tail(20).mean()

            # Calculate ATR
            high = df[high_col].tail(14)
            low = df[low_col].tail(14)
            close_prev = df[close_col].shift(1).tail(14)
            tr = pd.concat([
                high - low,
                (high - close_prev).abs(),
                (low - close_prev).abs()
            ], axis=1).max(axis=1)
            atr = tr.mean()

            if atr > 0:
                deviation = (mean_20 - close) / atr
                if deviation > 2.0:
                    score += 7
                elif deviation > 1.5:
                    score += 5
                elif deviation > 1.0:
                    score += 3
                elif deviation > 0.5:
                    score += 1
        else:
            score += 3  # Neutral

        return min(score, 25)

    def _score_volume_confirmation(self, df: pd.DataFrame) -> int:
        """
        Score volume confirmation (0-15 points).

        Factors:
        - Volume spike vs 20-day avg (0-10 points)
        - Volume trend increasing (0-5 points)
        """
        if len(df) < 20:
            return 7  # Neutral if insufficient data

        score = 0
        vol_col = 'volume' if 'volume' in df.columns else 'Volume'

        # Volume spike (capitulation volume is bullish for mean reversion)
        recent_vol = df[vol_col].iloc[-1]
        avg_vol = df[vol_col].tail(20).mean()
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

        if vol_ratio > 3.0:
            score += 10  # Strong capitulation
        elif vol_ratio > 2.0:
            score += 8
        elif vol_ratio > 1.5:
            score += 5
        elif vol_ratio > 1.0:
            score += 3
        else:
            score += 1  # Below average volume

        # Volume trend (accumulation pattern)
        if len(df) >= 5:
            recent_avg = df[vol_col].tail(5).mean()
            prev_avg = df[vol_col].iloc[-10:-5].mean() if len(df) >= 10 else avg_vol
            if recent_avg > prev_avg * 1.2:
                score += 5
            elif recent_avg > prev_avg:
                score += 3

        return min(score, 15)

    def _score_market_context(
        self,
        spy_df: Optional[pd.DataFrame],
        vix_level: Optional[float]
    ) -> int:
        """
        Score market context (0-15 points).

        Factors:
        - SPY above SMA(50) (0-7 points)
        - VIX level appropriate (0-8 points)
        """
        score = 0

        # SPY trend context
        if spy_df is not None and len(spy_df) >= 50:
            close_col = 'close' if 'close' in spy_df.columns else 'Close'
            spy_close = spy_df[close_col].iloc[-1]
            spy_sma50 = spy_df[close_col].tail(50).mean()

            if spy_close > spy_sma50:
                pct_above = ((spy_close - spy_sma50) / spy_sma50) * 100
                if pct_above > 5:
                    score += 7
                elif pct_above > 2:
                    score += 5
                else:
                    score += 3
            else:
                score += 1  # Bearish market - reduce conviction
        else:
            score += 4  # Neutral

        # VIX context (moderate VIX is best for mean reversion)
        if vix_level is not None:
            if 15 <= vix_level <= 25:
                score += 8  # Optimal VIX range
            elif 12 <= vix_level < 15:
                score += 6  # Low but OK
            elif 25 < vix_level <= 35:
                score += 5  # Elevated - opportunities but risky
            elif vix_level > 35:
                score += 3  # High fear - high reward but dangerous
            else:  # < 12
                score += 4  # Complacent market
        else:
            score += 4  # Neutral

        return min(score, 15)

    def _score_reversion_strength(self, signal: Dict, df: pd.DataFrame) -> int:
        """
        Score mean reversion strength (0-15 points).

        Factors:
        - IBS (Internal Bar Strength) (0-8 points)
        - Consecutive down days (0-7 points)
        """
        score = 0

        if len(df) < 5:
            return 7  # Neutral

        close_col = 'close' if 'close' in df.columns else 'Close'
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'

        # IBS calculation
        close = df[close_col].iloc[-1]
        high = df[high_col].iloc[-1]
        low = df[low_col].iloc[-1]

        if high != low:
            ibs = (close - low) / (high - low)
        else:
            ibs = 0.5

        if ibs <= 0.1:
            score += 8  # Strong oversold
        elif ibs <= 0.2:
            score += 6
        elif ibs <= 0.3:
            score += 4
        elif ibs <= 0.4:
            score += 2

        # Consecutive down days
        down_days = 0
        for i in range(-1, -min(6, len(df)), -1):
            if df[close_col].iloc[i] < df[close_col].iloc[i-1]:
                down_days += 1
            else:
                break

        if down_days >= 4:
            score += 7
        elif down_days >= 3:
            score += 5
        elif down_days >= 2:
            score += 3
        elif down_days >= 1:
            score += 1

        return min(score, 15)

    def _score_risk_reward(self, signal: Dict) -> int:
        """
        Score risk/reward quality (0-15 points).

        Factors:
        - R:R ratio (0-10 points)
        - Stop distance reasonable (0-5 points)
        """
        score = 0

        # Coerce potential None values to 0.0 to avoid type errors in comparisons
        try:
            entry = float(signal.get('entry_price') or 0)
        except Exception:
            entry = 0.0
        try:
            stop = float(signal.get('stop_loss') or 0)
        except Exception:
            stop = 0.0
        try:
            target = float(signal.get('take_profit') or 0)
        except Exception:
            target = 0.0

        if entry > 0 and stop > 0 and target > 0:
            risk = abs(entry - stop)
            reward = abs(target - entry)
            rr_ratio = reward / risk if risk > 0 else 0

            # R:R scoring
            if rr_ratio >= 3.0:
                score += 10
            elif rr_ratio >= 2.5:
                score += 8
            elif rr_ratio >= 2.0:
                score += 6
            elif rr_ratio >= 1.5:
                score += 4
            elif rr_ratio >= 1.0:
                score += 2

            # Stop distance (prefer tighter stops)
            stop_pct = (risk / entry) * 100 if entry > 0 else 0.0
            if 1.0 <= stop_pct <= 3.0:
                score += 5  # Optimal stop distance
            elif 0.5 <= stop_pct < 1.0:
                score += 3  # Very tight
            elif 3.0 < stop_pct <= 5.0:
                score += 3  # Acceptable
            else:
                score += 1  # Too tight or too wide

        else:
            score += 7  # Neutral if missing data

        return min(score, 15)

    def _score_timing(self, signal: Dict, df: pd.DataFrame) -> int:
        """
        Score timing quality (0-15 points).

        Factors:
        - Not at resistance (0-8 points)
        - Recent support test (0-7 points)
        """
        score = 0

        if len(df) < 20:
            return 7  # Neutral

        close_col = 'close' if 'close' in df.columns else 'Close'
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'

        current_price = df[close_col].iloc[-1]

        # Check resistance (20-day high)
        high_20 = df[high_col].tail(20).max()
        distance_from_high = (high_20 - current_price) / current_price

        if distance_from_high > 0.10:
            score += 8  # Far from resistance
        elif distance_from_high > 0.05:
            score += 6
        elif distance_from_high > 0.02:
            score += 4
        else:
            score += 2  # Near resistance

        # Check support (testing recent lows)
        low_20 = df[low_col].tail(20).min()
        distance_from_low = (current_price - low_20) / current_price

        if distance_from_low < 0.02:
            score += 7  # Near support (bounce opportunity)
        elif distance_from_low < 0.05:
            score += 5
        elif distance_from_low < 0.10:
            score += 3
        else:
            score += 1  # Far from support

        return min(score, 15)

    def _get_tier_info(self, total_score: int) -> Dict[str, str]:
        """Get tier information based on total score."""
        for (low, high), info in self.TIERS.items():
            if low <= total_score <= high:
                return info
        return {"tier": "WEAK", "action": "PASS"}

    def score_signal(
        self,
        signal: Dict[str, Any],
        price_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame] = None,
        vix_level: Optional[float] = None
    ) -> int:
        """
        Quick method to get just the total score.

        Args:
            signal: Signal dictionary
            price_data: Price history DataFrame
            spy_data: Optional SPY data
            vix_level: Optional VIX level

        Returns:
            Total conviction score (0-100)
        """
        breakdown = self.calculate_conviction(signal, price_data, spy_data, vix_level)
        return breakdown.total_score


# Singleton instance
_conviction_scorer: Optional[ConvictionScorer] = None


def get_conviction_scorer() -> ConvictionScorer:
    """Get or create the global conviction scorer instance."""
    global _conviction_scorer
    if _conviction_scorer is None:
        _conviction_scorer = ConvictionScorer()
    return _conviction_scorer
