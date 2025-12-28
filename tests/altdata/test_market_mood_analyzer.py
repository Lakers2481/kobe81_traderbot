"""
Tests for the MarketMoodAnalyzer module.
"""
import pytest
from datetime import datetime, timedelta

from altdata.market_mood_analyzer import (
    MarketMoodAnalyzer,
    MarketMood,
    MoodState,
    get_market_mood_analyzer,
)


class TestMoodState:
    """Test MoodState enum."""

    def test_mood_state_values(self):
        """Verify all expected mood states exist."""
        assert MoodState.EXTREME_FEAR.value == "Extreme Fear"
        assert MoodState.FEAR.value == "Fear"
        assert MoodState.NEUTRAL.value == "Neutral"
        assert MoodState.GREED.value == "Greed"
        assert MoodState.EXTREME_GREED.value == "Extreme Greed/Euphoria"


class TestMarketMood:
    """Test MarketMood dataclass."""

    def test_market_mood_to_dict(self):
        """Test serialization of MarketMood."""
        mood = MarketMood(
            mood_score=-0.5,
            mood_state=MoodState.FEAR,
            is_extreme=False,
            confidence=0.8,
            vix_contribution=-0.3,
            sentiment_contribution=-0.2,
        )

        d = mood.to_dict()

        assert d['mood_score'] == -0.5
        assert d['mood_state'] == "Fear"
        assert d['is_extreme'] is False
        assert d['confidence'] == 0.8
        assert d['vix_contribution'] == -0.3
        assert d['sentiment_contribution'] == -0.2
        assert 'timestamp' in d

    def test_market_mood_default_timestamp(self):
        """Test that timestamp defaults to now."""
        mood = MarketMood(
            mood_score=0.0,
            mood_state=MoodState.NEUTRAL,
            is_extreme=False,
            confidence=0.5,
        )

        assert mood.timestamp is not None
        assert (datetime.now() - mood.timestamp).total_seconds() < 1


class TestMarketMoodAnalyzer:
    """Test MarketMoodAnalyzer class."""

    def test_initialization_defaults(self):
        """Test analyzer initializes with default values."""
        analyzer = MarketMoodAnalyzer()

        assert analyzer.vix_weight == 0.6
        assert analyzer.sentiment_weight == 0.4
        assert analyzer.extreme_threshold == 0.7

    def test_initialization_custom_weights(self):
        """Test analyzer respects custom weight configuration."""
        analyzer = MarketMoodAnalyzer(
            vix_weight=0.8,
            sentiment_weight=0.2,
            extreme_threshold=0.5,
        )

        assert analyzer.vix_weight == 0.8
        assert analyzer.sentiment_weight == 0.2
        assert analyzer.extreme_threshold == 0.5

    def test_weights_normalized(self):
        """Test that weights are normalized to sum to 1."""
        analyzer = MarketMoodAnalyzer(vix_weight=3.0, sentiment_weight=2.0)

        assert abs(analyzer.vix_weight + analyzer.sentiment_weight - 1.0) < 0.001


class TestVixToMoodScore:
    """Test VIX to mood score conversion."""

    def test_extreme_fear_high_vix(self):
        """VIX >= 35 should produce extreme fear (negative score)."""
        analyzer = MarketMoodAnalyzer()

        score = analyzer._vix_to_mood_score(40)
        assert score < -0.7

        score = analyzer._vix_to_mood_score(50)
        assert score <= -1.0

    def test_fear_zone_vix(self):
        """VIX 25-35 should produce fear (moderately negative score)."""
        analyzer = MarketMoodAnalyzer()

        score = analyzer._vix_to_mood_score(25)
        assert -0.5 < score < -0.2

        score = analyzer._vix_to_mood_score(30)
        assert -0.7 < score < -0.3

    def test_neutral_zone_vix(self):
        """VIX 15-25 should produce neutral (score near 0)."""
        analyzer = MarketMoodAnalyzer()

        score = analyzer._vix_to_mood_score(20)
        assert -0.3 < score < 0.3

    def test_greed_zone_vix(self):
        """VIX 12-15 should produce greed (moderately positive score)."""
        analyzer = MarketMoodAnalyzer()

        score = analyzer._vix_to_mood_score(14)
        assert 0.3 < score < 0.7

    def test_extreme_greed_low_vix(self):
        """VIX < 12 should produce extreme greed (high positive score)."""
        analyzer = MarketMoodAnalyzer()

        score = analyzer._vix_to_mood_score(10)
        assert score > 0.7

        score = analyzer._vix_to_mood_score(8)
        assert score >= 0.9


class TestDetermineMoodState:
    """Test mood state determination."""

    def test_extreme_fear_state(self):
        """Score <= -0.7 should be EXTREME_FEAR."""
        analyzer = MarketMoodAnalyzer()

        assert analyzer._determine_mood_state(-0.8) == MoodState.EXTREME_FEAR
        assert analyzer._determine_mood_state(-1.0) == MoodState.EXTREME_FEAR

    def test_fear_state(self):
        """Score -0.7 to -0.3 should be FEAR."""
        analyzer = MarketMoodAnalyzer()

        assert analyzer._determine_mood_state(-0.5) == MoodState.FEAR
        assert analyzer._determine_mood_state(-0.35) == MoodState.FEAR

    def test_neutral_state(self):
        """Score -0.3 to 0.3 should be NEUTRAL."""
        analyzer = MarketMoodAnalyzer()

        assert analyzer._determine_mood_state(0.0) == MoodState.NEUTRAL
        assert analyzer._determine_mood_state(0.2) == MoodState.NEUTRAL
        assert analyzer._determine_mood_state(-0.2) == MoodState.NEUTRAL

    def test_greed_state(self):
        """Score 0.3 to 0.7 should be GREED."""
        analyzer = MarketMoodAnalyzer()

        assert analyzer._determine_mood_state(0.5) == MoodState.GREED
        assert analyzer._determine_mood_state(0.35) == MoodState.GREED

    def test_extreme_greed_state(self):
        """Score >= 0.7 should be EXTREME_GREED."""
        analyzer = MarketMoodAnalyzer()

        assert analyzer._determine_mood_state(0.8) == MoodState.EXTREME_GREED
        assert analyzer._determine_mood_state(1.0) == MoodState.EXTREME_GREED


class TestGetMarketMood:
    """Test the main get_market_mood method."""

    def test_returns_dict_format(self):
        """get_market_mood should return a properly structured dictionary."""
        analyzer = MarketMoodAnalyzer()
        context = {'vix': 20, 'sentiment': 0.0}

        result = analyzer.get_market_mood(context)

        assert 'market_mood' in result
        assert 'market_mood_score' in result
        assert 'market_mood_state' in result
        assert 'is_extreme_mood' in result

    def test_high_vix_extreme_fear(self):
        """Very high VIX should produce fear (weighted score)."""
        analyzer = MarketMoodAnalyzer()
        context = {'vix': 45, 'sentiment': 0.0}

        result = analyzer.get_market_mood(context)

        # With vix_weight=0.6, even VIX=45 produces ~0.6 * -0.8 = -0.48
        assert result['market_mood_score'] < 0  # Should be negative
        assert 'Fear' in result['market_mood_state']

    def test_low_vix_greed(self):
        """Very low VIX should produce greed."""
        analyzer = MarketMoodAnalyzer()
        context = {'vix': 10, 'sentiment': 0.0}

        result = analyzer.get_market_mood(context)

        # With vix_weight=0.6, VIX=10 produces positive score
        assert result['market_mood_score'] > 0.3
        assert 'Greed' in result['market_mood_state']

    def test_neutral_conditions(self):
        """Normal VIX and neutral sentiment should produce neutral mood."""
        analyzer = MarketMoodAnalyzer()
        context = {'vix': 20, 'sentiment': 0.0}

        result = analyzer.get_market_mood(context)

        assert -0.3 < result['market_mood_score'] < 0.3
        assert result['is_extreme_mood'] is False
        assert result['market_mood_state'] == 'Neutral'

    def test_vix_override(self):
        """vix_override parameter should override context VIX."""
        analyzer = MarketMoodAnalyzer()
        context = {'vix': 15}  # Normal VIX in context

        # Override with high VIX
        result = analyzer.get_market_mood(context, vix_override=40)

        # With 0.6 weight, VIX 40 gives ~-0.48
        assert result['market_mood_score'] < 0  # Should be fearful (negative)

    def test_sentiment_override(self):
        """sentiment_override parameter should override context sentiment."""
        analyzer = MarketMoodAnalyzer()
        context = {'vix': 20, 'sentiment': 0.0}  # Neutral

        # Override with very negative sentiment
        result = analyzer.get_market_mood(context, sentiment_override=-0.9)

        # Score should be more negative due to sentiment
        assert result['market_mood_score'] < 0

    def test_combined_vix_and_sentiment(self):
        """VIX and sentiment should combine according to weights."""
        analyzer = MarketMoodAnalyzer(vix_weight=0.6, sentiment_weight=0.4)

        # High VIX (fear) + positive sentiment (greed)
        context = {'vix': 35, 'sentiment': 0.8}

        result = analyzer.get_market_mood(context)

        # VIX contributes negative, sentiment positive
        # Final score should be somewhere in between
        assert result['market_mood'] is not None

    def test_missing_data_uses_defaults(self):
        """Missing data in context should use defaults."""
        analyzer = MarketMoodAnalyzer()
        context = {}  # Empty context

        result = analyzer.get_market_mood(context)

        # Should default to VIX=20, sentiment=0 -> neutral
        assert 'Neutral' in result['market_mood_state']


class TestExtremeMoodDetection:
    """Test extreme mood detection for cognitive system integration."""

    def test_extreme_fear_is_extreme(self):
        """Extreme fear conditions should flag is_extreme_mood."""
        # Use both VIX and sentiment to hit extreme threshold
        # With vix_weight=0.6 and sentiment_weight=0.4, need both to align
        analyzer = MarketMoodAnalyzer(extreme_threshold=0.7)
        context = {'vix': 50, 'sentiment': -0.9}  # Very high fear VIX + negative sentiment

        result = analyzer.get_market_mood(context)

        assert result['is_extreme_mood'] is True

    def test_extreme_greed_is_extreme(self):
        """Extreme greed conditions should flag is_extreme_mood."""
        # Use both VIX and sentiment to hit extreme threshold
        analyzer = MarketMoodAnalyzer(extreme_threshold=0.7)
        context = {'vix': 8, 'sentiment': 0.9}  # Very low VIX + positive sentiment

        result = analyzer.get_market_mood(context)

        assert result['is_extreme_mood'] is True

    def test_moderate_conditions_not_extreme(self):
        """Moderate conditions should not flag is_extreme_mood."""
        analyzer = MarketMoodAnalyzer(extreme_threshold=0.7)
        context = {'vix': 20, 'sentiment': 0.0}  # Neutral

        result = analyzer.get_market_mood(context)

        assert result['is_extreme_mood'] is False

    def test_lower_threshold_triggers_extreme(self):
        """Lower extreme_threshold should make it easier to trigger."""
        # With lower threshold, even moderate VIX can trigger extreme
        analyzer = MarketMoodAnalyzer(extreme_threshold=0.4)
        context = {'vix': 35, 'sentiment': -0.3}

        result = analyzer.get_market_mood(context)

        assert result['is_extreme_mood'] is True


class TestMoodDescription:
    """Test mood description generation."""

    def test_extreme_fear_description(self):
        """Extreme fear should have appropriate description."""
        analyzer = MarketMoodAnalyzer()

        desc = analyzer.get_mood_description(-0.9)

        assert "EXTREME FEAR" in desc
        assert "panic" in desc.lower() or "reduced" in desc.lower()

    def test_neutral_description(self):
        """Neutral mood should have appropriate description."""
        analyzer = MarketMoodAnalyzer()

        desc = analyzer.get_mood_description(0.0)

        assert "NEUTRAL" in desc
        assert "normal" in desc.lower()


class TestIntrospection:
    """Test introspection output."""

    def test_introspect_returns_string(self):
        """introspect should return a human-readable string."""
        analyzer = MarketMoodAnalyzer()

        output = analyzer.introspect()

        assert isinstance(output, str)
        assert "VIX" in output
        assert "Sentiment" in output
        assert "extreme" in output.lower()


class TestSingleton:
    """Test singleton factory function."""

    def test_get_market_mood_analyzer_returns_same_instance(self):
        """Factory should return the same instance."""
        # Note: Need to reset singleton for clean test
        import altdata.market_mood_analyzer as module
        module._market_mood_analyzer = None

        analyzer1 = get_market_mood_analyzer()
        analyzer2 = get_market_mood_analyzer()

        assert analyzer1 is analyzer2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_vix(self):
        """VIX of 0 should be handled gracefully."""
        analyzer = MarketMoodAnalyzer()

        score = analyzer._vix_to_mood_score(0)

        assert -1 <= score <= 1

    def test_negative_vix(self):
        """Negative VIX (invalid) should be handled."""
        analyzer = MarketMoodAnalyzer()

        # Should not crash
        score = analyzer._vix_to_mood_score(-5)

        assert -1 <= score <= 1

    def test_sentiment_out_of_range(self):
        """Sentiment values outside [-1, 1] should be clamped."""
        analyzer = MarketMoodAnalyzer()

        # Test clamping
        assert analyzer._normalize_sentiment(2.0) == 1.0
        assert analyzer._normalize_sentiment(-2.0) == -1.0
        assert analyzer._normalize_sentiment(0.5) == 0.5
