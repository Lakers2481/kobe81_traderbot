"""
Market Mood Analyzer - Holistic Market Emotional State Assessment
================================================================

This module combines VIX levels with sentiment data to produce a holistic
assessment of the market's "emotional state". This goes beyond simple
fear/greed indicators by synthesizing multiple data sources into a
unified mood score that the cognitive system can use for decision-making.

The market mood affects:
- Risk tolerance in the MetacognitiveGovernor
- Knowledge boundary assessments (extreme moods = uncertainty)
- Strategy selection biases in the CognitiveBrain
- Position sizing recommendations

Usage:
    from altdata.market_mood_analyzer import get_market_mood_analyzer

    analyzer = get_market_mood_analyzer()
    mood = analyzer.get_market_mood(context)

    if mood['is_extreme']:
        # Consider standing down or reducing position sizes
        pass
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MoodState(Enum):
    """
    Enumerates the discrete emotional states of the market.

    These states are derived from the continuous mood_score and provide
    a human-readable categorization of market sentiment.
    """
    EXTREME_FEAR = "Extreme Fear"
    FEAR = "Fear"
    NEUTRAL = "Neutral"
    GREED = "Greed"
    EXTREME_GREED = "Extreme Greed/Euphoria"


@dataclass
class MarketMood:
    """
    Represents the analyzed emotional state of the market.

    Attributes:
        mood_score: Continuous score from -1.0 (extreme fear) to 1.0 (extreme greed).
        mood_state: Discrete categorization of the mood.
        is_extreme: True if the mood is at an extreme (fear or greed).
        confidence: Confidence in the mood assessment (0-1).
        vix_contribution: How much VIX contributed to the score.
        sentiment_contribution: How much sentiment data contributed.
        timestamp: When this mood was assessed.
        components: Breakdown of contributing factors.
    """
    mood_score: float
    mood_state: MoodState
    is_extreme: bool
    confidence: float
    vix_contribution: float = 0.0
    sentiment_contribution: float = 0.0
    timestamp: datetime = None
    components: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.components is None:
            self.components = {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'mood_score': round(self.mood_score, 3),
            'mood_state': self.mood_state.value,
            'is_extreme': self.is_extreme,
            'confidence': round(self.confidence, 3),
            'vix_contribution': round(self.vix_contribution, 3),
            'sentiment_contribution': round(self.sentiment_contribution, 3),
            'timestamp': self.timestamp.isoformat(),
            'components': self.components,
        }


class MarketMoodAnalyzer:
    """
    Analyzes market conditions to determine the overall emotional state.

    Combines VIX data with sentiment indicators to produce a unified
    mood score that represents the market's emotional state.
    """

    def __init__(
        self,
        vix_weight: float = 0.6,
        sentiment_weight: float = 0.4,
        extreme_threshold: float = 0.7,
        vix_fear_level: float = 25.0,
        vix_greed_level: float = 15.0,
        vix_extreme_fear_level: float = 35.0,
        vix_extreme_greed_level: float = 12.0,
    ):
        """
        Initialize the MarketMoodAnalyzer.

        Args:
            vix_weight: Weight given to VIX in mood calculation (0-1).
            sentiment_weight: Weight given to sentiment (0-1).
            extreme_threshold: Score threshold for extreme mood detection.
            vix_fear_level: VIX level indicating fear.
            vix_greed_level: VIX level indicating greed.
            vix_extreme_fear_level: VIX level indicating extreme fear.
            vix_extreme_greed_level: VIX level indicating extreme greed.
        """
        self.vix_weight = vix_weight
        self.sentiment_weight = sentiment_weight
        self.extreme_threshold = extreme_threshold
        self.vix_fear_level = vix_fear_level
        self.vix_greed_level = vix_greed_level
        self.vix_extreme_fear_level = vix_extreme_fear_level
        self.vix_extreme_greed_level = vix_extreme_greed_level

        # Normalize weights to sum to 1
        total_weight = self.vix_weight + self.sentiment_weight
        if total_weight > 0:
            self.vix_weight /= total_weight
            self.sentiment_weight /= total_weight

        logger.info(
            f"MarketMoodAnalyzer initialized: VIX weight={self.vix_weight:.2f}, "
            f"Sentiment weight={self.sentiment_weight:.2f}, "
            f"Extreme threshold={self.extreme_threshold}"
        )

    def get_market_mood(
        self,
        context: Dict[str, Any],
        vix_override: Optional[float] = None,
        sentiment_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the current market mood based on available data.

        Args:
            context: Market context dictionary containing 'vix' and 'sentiment' keys.
            vix_override: Optional override for VIX value (for testing).
            sentiment_override: Optional override for sentiment value (for testing).

        Returns:
            Dictionary containing mood information suitable for context enrichment.
        """
        # Extract VIX from context or use override
        vix = vix_override if vix_override is not None else context.get('vix', 20.0)
        if vix is None:
            vix = 20.0  # Default neutral VIX

        # Extract sentiment from context or use override
        # Sentiment is expected to be in range [-1, 1]
        sentiment = sentiment_override if sentiment_override is not None else context.get('sentiment', 0.0)
        if sentiment is None:
            sentiment = 0.0

        # Calculate mood score from components
        vix_mood = self._vix_to_mood_score(vix)
        sentiment_mood = self._normalize_sentiment(sentiment)

        # Weighted combination
        mood_score = (
            self.vix_weight * vix_mood +
            self.sentiment_weight * sentiment_mood
        )

        # Clamp to [-1, 1]
        mood_score = max(-1.0, min(1.0, mood_score))

        # Determine discrete state
        mood_state = self._determine_mood_state(mood_score)

        # Check for extreme conditions
        is_extreme = abs(mood_score) >= self.extreme_threshold

        # Calculate confidence based on data availability
        confidence = self._calculate_confidence(context, vix_override, sentiment_override)

        # Build the MarketMood object
        mood = MarketMood(
            mood_score=mood_score,
            mood_state=mood_state,
            is_extreme=is_extreme,
            confidence=confidence,
            vix_contribution=vix_mood * self.vix_weight,
            sentiment_contribution=sentiment_mood * self.sentiment_weight,
            components={
                'vix': vix,
                'vix_mood_score': vix_mood,
                'sentiment': sentiment,
                'sentiment_mood_score': sentiment_mood,
                'vix_weight': self.vix_weight,
                'sentiment_weight': self.sentiment_weight,
            }
        )

        logger.debug(
            f"Market mood analyzed: score={mood_score:.3f}, "
            f"state={mood_state.value}, extreme={is_extreme}"
        )

        # Return as dictionary for easy context integration
        return {
            'market_mood': mood.to_dict(),
            'market_mood_score': mood_score,
            'market_mood_state': mood_state.value,
            'is_extreme_mood': is_extreme,
        }

    def _vix_to_mood_score(self, vix: float) -> float:
        """
        Convert VIX level to a mood score.

        High VIX indicates fear (negative score), low VIX indicates greed (positive score).

        Args:
            vix: The current VIX level.

        Returns:
            Mood score from -1.0 (extreme fear) to 1.0 (extreme greed).
        """
        # Extreme fear: VIX >= 35 -> -1.0
        if vix >= self.vix_extreme_fear_level:
            # Scale from -0.7 to -1.0 for VIX 35+
            excess = min(vix - self.vix_extreme_fear_level, 15)  # Cap at VIX 50
            return -0.7 - (0.3 * excess / 15)

        # Fear zone: VIX 25-35 -> -0.3 to -0.7
        if vix >= self.vix_fear_level:
            progress = (vix - self.vix_fear_level) / (self.vix_extreme_fear_level - self.vix_fear_level)
            return -0.3 - (0.4 * progress)

        # Neutral zone: VIX 15-25 -> -0.3 to 0.3
        if vix >= self.vix_greed_level:
            # Linear interpolation between greed and fear levels
            progress = (vix - self.vix_greed_level) / (self.vix_fear_level - self.vix_greed_level)
            return 0.3 - (0.6 * progress)

        # Greed zone: VIX 12-15 -> 0.3 to 0.7
        if vix >= self.vix_extreme_greed_level:
            progress = 1 - ((vix - self.vix_extreme_greed_level) / (self.vix_greed_level - self.vix_extreme_greed_level))
            return 0.3 + (0.4 * progress)

        # Extreme greed: VIX < 12 -> 0.7 to 1.0
        if vix > 0:
            # Scale from 0.7 to 1.0 for very low VIX
            deficit = max(self.vix_extreme_greed_level - vix, 0)
            return min(0.7 + (0.3 * deficit / 4), 1.0)

        return 0.0  # Invalid VIX

    def _normalize_sentiment(self, sentiment: float) -> float:
        """
        Normalize sentiment value to the standard [-1, 1] range.

        Args:
            sentiment: Raw sentiment value (expected in [-1, 1]).

        Returns:
            Normalized sentiment score in [-1, 1].
        """
        # Clamp to valid range
        return max(-1.0, min(1.0, sentiment))

    def _determine_mood_state(self, mood_score: float) -> MoodState:
        """
        Convert a continuous mood score to a discrete MoodState.

        Args:
            mood_score: Continuous score from -1.0 to 1.0.

        Returns:
            The corresponding MoodState enum value.
        """
        if mood_score <= -0.7:
            return MoodState.EXTREME_FEAR
        elif mood_score <= -0.3:
            return MoodState.FEAR
        elif mood_score >= 0.7:
            return MoodState.EXTREME_GREED
        elif mood_score >= 0.3:
            return MoodState.GREED
        else:
            return MoodState.NEUTRAL

    def _calculate_confidence(
        self,
        context: Dict[str, Any],
        vix_override: Optional[float],
        sentiment_override: Optional[float],
    ) -> float:
        """
        Calculate confidence in the mood assessment based on data availability.

        Args:
            context: The market context dictionary.
            vix_override: Whether VIX was overridden (test mode).
            sentiment_override: Whether sentiment was overridden (test mode).

        Returns:
            Confidence score from 0 to 1.
        """
        confidence = 0.0

        # VIX data contributes to confidence
        if vix_override is not None or context.get('vix') is not None:
            confidence += 0.5 * self.vix_weight / max(self.vix_weight, 0.01)

        # Sentiment data contributes to confidence
        if sentiment_override is not None or context.get('sentiment') is not None:
            confidence += 0.5 * self.sentiment_weight / max(self.sentiment_weight, 0.01)

        # Bonus confidence if VIX data is fresh
        vix_timestamp = context.get('vix_timestamp')
        if vix_timestamp:
            try:
                if isinstance(vix_timestamp, str):
                    vix_time = datetime.fromisoformat(vix_timestamp)
                else:
                    vix_time = vix_timestamp
                age_minutes = (datetime.now() - vix_time).total_seconds() / 60
                if age_minutes < 15:
                    confidence += 0.1
                elif age_minutes < 60:
                    confidence += 0.05
            except (ValueError, TypeError):
                pass

        return min(confidence, 1.0)

    def get_mood_description(self, mood_score: float) -> str:
        """
        Generate a human-readable description of the market mood.

        Args:
            mood_score: The continuous mood score.

        Returns:
            A descriptive string about the market mood.
        """
        state = self._determine_mood_state(mood_score)

        descriptions = {
            MoodState.EXTREME_FEAR: (
                "Markets are in EXTREME FEAR. Panic selling likely. "
                "Consider reduced exposure or contrarian opportunities."
            ),
            MoodState.FEAR: (
                "Markets show FEAR. Elevated caution warranted. "
                "Volatility is elevated."
            ),
            MoodState.NEUTRAL: (
                "Markets are NEUTRAL. Normal trading conditions. "
                "Standard risk parameters apply."
            ),
            MoodState.GREED: (
                "Markets show GREED. Optimism is elevated. "
                "Watch for complacency."
            ),
            MoodState.EXTREME_GREED: (
                "Markets in EXTREME GREED/EUPHORIA. "
                "Peak optimism, potential reversal risk. Exercise caution."
            ),
        }

        return descriptions.get(state, "Unknown market mood state.")

    def introspect(self) -> str:
        """Generate a human-readable description of the analyzer configuration."""
        return (
            "--- Market Mood Analyzer Introspection ---\n"
            f"I analyze market emotional state using:\n"
            f"  - VIX (weight: {self.vix_weight:.1%})\n"
            f"  - Sentiment (weight: {self.sentiment_weight:.1%})\n"
            f"I flag conditions as 'extreme' when |mood_score| >= {self.extreme_threshold}\n"
            f"VIX thresholds: Fear={self.vix_fear_level}, "
            f"Extreme Fear={self.vix_extreme_fear_level}\n"
            f"VIX thresholds: Greed={self.vix_greed_level}, "
            f"Extreme Greed={self.vix_extreme_greed_level}"
        )


# --- Singleton Implementation ---
_market_mood_analyzer: Optional[MarketMoodAnalyzer] = None


def get_market_mood_analyzer(
    vix_weight: float = 0.6,
    sentiment_weight: float = 0.4,
    extreme_threshold: float = 0.7,
) -> MarketMoodAnalyzer:
    """
    Factory function to get the singleton instance of MarketMoodAnalyzer.

    Args:
        vix_weight: Weight for VIX in mood calculation.
        sentiment_weight: Weight for sentiment in mood calculation.
        extreme_threshold: Threshold for flagging extreme moods.

    Returns:
        The MarketMoodAnalyzer singleton instance.
    """
    global _market_mood_analyzer
    if _market_mood_analyzer is None:
        _market_mood_analyzer = MarketMoodAnalyzer(
            vix_weight=vix_weight,
            sentiment_weight=sentiment_weight,
            extreme_threshold=extreme_threshold,
        )
    return _market_mood_analyzer
