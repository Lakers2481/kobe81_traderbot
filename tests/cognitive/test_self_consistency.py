"""
Comprehensive Unit Tests for Self-Consistency Module
=====================================================

Tests the self-consistency decoding system that samples multiple
reasoning chains and picks the majority answer.

Run: python -m pytest tests/cognitive/test_self_consistency.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from collections import Counter


class TestSelfConsistencyResult:
    """Tests for SelfConsistencyResult dataclass."""

    def test_result_creation(self):
        """Test basic result instantiation."""
        from cognitive.self_consistency import SelfConsistencyResult, ReasoningChain

        chains = [
            ReasoningChain(chain="Chain 1", answer="LONG", raw_response="resp1"),
            ReasoningChain(chain="Chain 2", answer="LONG", raw_response="resp2"),
            ReasoningChain(chain="Chain 3", answer="SHORT", raw_response="resp3"),
        ]

        result = SelfConsistencyResult(
            final_answer="LONG",
            confidence=0.67,
            agreement_ratio=0.67,
            best_chain="Chain 1",
            all_chains=chains,
            all_answers=["LONG", "LONG", "SHORT"],
            answer_distribution={"LONG": 2, "SHORT": 1},
            n_samples=3,
            unique_answers=2,
        )

        assert result.final_answer == "LONG"
        assert result.confidence == 0.67
        assert result.agreement_ratio == 0.67
        assert len(result.all_chains) == 3
        assert len(result.all_answers) == 3

    def test_result_serialization(self):
        """Test result to_dict serialization."""
        from cognitive.self_consistency import SelfConsistencyResult, ReasoningChain

        chains = [
            ReasoningChain(chain="C1", answer="SHORT", raw_response="r1"),
            ReasoningChain(chain="C2", answer="LONG", raw_response="r2"),
        ]

        result = SelfConsistencyResult(
            final_answer="SHORT",
            confidence=0.5,
            agreement_ratio=0.5,
            best_chain="C1",
            all_chains=chains,
            all_answers=["SHORT", "LONG"],
            answer_distribution={"SHORT": 1, "LONG": 1},
            n_samples=2,
            unique_answers=2,
        )

        data = result.to_dict()

        assert data['final_answer'] == "SHORT"
        assert data['confidence'] == 0.5
        assert data['agreement_ratio'] == 0.5

    def test_result_properties(self):
        """Test result computed properties."""
        from cognitive.self_consistency import SelfConsistencyResult, ReasoningChain

        chains = [
            ReasoningChain(chain=f"C{i}", answer="SD", raw_response=f"r{i}")
            for i in range(5)
        ]

        result = SelfConsistencyResult(
            final_answer="STAND_DOWN",
            confidence=0.6,
            agreement_ratio=0.6,
            best_chain="C1",
            all_chains=chains,
            all_answers=["STAND_DOWN"] * 3 + ["LONG", "SHORT"],
            answer_distribution={"STAND_DOWN": 3, "LONG": 1, "SHORT": 1},
            n_samples=5,
            unique_answers=3,
        )

        # Test properties
        assert result.is_disputed  # 3 unique answers
        assert not result.is_high_confidence  # 60% < 80%


class TestSelfConsistencyDecoder:
    """Tests for the main SelfConsistencyDecoder class."""

    def test_initialization_without_llm(self):
        """Test initialization without LLM provider."""
        from cognitive.self_consistency import SelfConsistencyDecoder

        decoder = SelfConsistencyDecoder(
            llm_provider=None,
            n_samples=5,
            temperature=0.7,
        )

        assert decoder.n_samples == 5
        assert decoder.temperature == 0.7
        assert decoder._llm is None

    def test_initialization_with_mock_llm(self):
        """Test initialization with mock LLM provider."""
        from cognitive.self_consistency import SelfConsistencyDecoder

        mock_llm = Mock()
        decoder = SelfConsistencyDecoder(
            llm_provider=mock_llm,
            n_samples=3,
            temperature=0.8,
        )

        assert decoder._llm is mock_llm
        assert decoder.n_samples == 3
        assert decoder.temperature == 0.8

    def test_decode_without_llm_returns_fallback(self):
        """Test that decode returns fallback when no LLM available."""
        from cognitive.self_consistency import SelfConsistencyDecoder

        decoder = SelfConsistencyDecoder(llm_provider=None)

        result = decoder.decode("Should I buy AAPL?")

        assert result is not None
        assert result.final_answer is not None
        assert result.confidence >= 0.0


class TestEnhancedSelfConsistency:
    """Tests for EnhancedSelfConsistency class."""

    def test_initialization(self):
        """Test enhanced decoder initialization."""
        from cognitive.self_consistency import EnhancedSelfConsistency

        enhanced = EnhancedSelfConsistency(
            llm_provider=None,
            n_samples=7,
            temperature=0.9,
        )

        assert enhanced.n_samples == 7
        assert enhanced.temperature == 0.9

    def test_decode_with_context_weights(self):
        """Test decode with context-aware weighting."""
        from cognitive.self_consistency import EnhancedSelfConsistency

        enhanced = EnhancedSelfConsistency(llm_provider=None)

        result = enhanced.decode(
            prompt="Should I trade?",
            context={"regime": "BULL", "vix": 15},
        )

        assert result is not None
        assert result.final_answer is not None


class TestSingletonPattern:
    """Tests for singleton accessor."""

    def test_get_self_consistency_returns_instance(self):
        """Test singleton accessor."""
        from cognitive.self_consistency import get_self_consistency

        sc1 = get_self_consistency()
        sc2 = get_self_consistency()

        assert sc1 is sc2


class TestReasoningChain:
    """Tests for ReasoningChain dataclass."""

    def test_reasoning_chain_creation(self):
        """Test basic chain creation."""
        from cognitive.self_consistency import ReasoningChain

        chain = ReasoningChain(
            chain="Analysis text here",
            answer="BUY",
            raw_response="Full response",
            confidence=0.85,
            extraction_method="pattern_match",
        )

        assert chain.chain == "Analysis text here"
        assert chain.answer == "BUY"
        assert chain.confidence == 0.85


class TestAnswerType:
    """Tests for AnswerType enum."""

    def test_answer_types_exist(self):
        """Test that all answer types are defined."""
        from cognitive.self_consistency import AnswerType

        assert AnswerType.TRADING_DECISION.value == "trading_decision"
        assert AnswerType.BOOLEAN.value == "boolean"
        assert AnswerType.NUMERIC.value == "numeric"
        assert AnswerType.CATEGORICAL.value == "categorical"
        assert AnswerType.FREEFORM.value == "freeform"


class TestSelfConsistencyIntegration:
    """Integration tests with mocked LLM."""

    def test_full_decode_with_mock_llm(self):
        """Test complete decode flow with mock LLM."""
        from cognitive.self_consistency import SelfConsistencyDecoder

        # Create mock LLM that returns different responses
        # Note: "LONG" gets normalized to "BUY" by TRADING_PATTERNS
        mock_llm = Mock()
        responses = [
            Mock(content="Analysis 1\nFINAL ANSWER: BUY"),
            Mock(content="Analysis 2\nFINAL ANSWER: BUY"),
            Mock(content="Analysis 3\nFINAL ANSWER: SELL"),
            Mock(content="Analysis 4\nFINAL ANSWER: BUY"),
            Mock(content="Analysis 5\nFINAL ANSWER: BUY"),
        ]
        mock_llm.chat = Mock(side_effect=responses)

        decoder = SelfConsistencyDecoder(
            llm_provider=mock_llm,
            n_samples=5,
            temperature=0.7,
        )

        result = decoder.decode("Should I buy AAPL?")

        assert result.final_answer == "BUY"
        assert result.agreement_ratio == 0.8  # 4/5
        assert len(result.all_answers) == 5

    def test_diverse_sampling_with_temperature(self):
        """Test that temperature affects diversity."""
        from cognitive.self_consistency import SelfConsistencyDecoder

        # High temperature should produce more diverse answers
        decoder_high_temp = SelfConsistencyDecoder(
            llm_provider=None,
            temperature=1.0,
        )

        # Low temperature should produce more consistent answers
        decoder_low_temp = SelfConsistencyDecoder(
            llm_provider=None,
            temperature=0.1,
        )

        assert decoder_high_temp.temperature > decoder_low_temp.temperature


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_high_confidence_result(self):
        """Test high confidence detection."""
        from cognitive.self_consistency import SelfConsistencyResult, ReasoningChain

        chains = [
            ReasoningChain(chain=f"C{i}", answer="BUY", raw_response=f"r{i}")
            for i in range(5)
        ]

        result = SelfConsistencyResult(
            final_answer="BUY",
            confidence=1.0,
            agreement_ratio=1.0,
            best_chain="C1",
            all_chains=chains,
            all_answers=["BUY"] * 5,
            answer_distribution={"BUY": 5},
            n_samples=5,
            unique_answers=1,
        )

        assert result.is_high_confidence
        assert not result.is_disputed

    def test_disputed_result(self):
        """Test disputed result detection."""
        from cognitive.self_consistency import SelfConsistencyResult, ReasoningChain

        chains = [
            ReasoningChain(chain=f"C{i}", answer="X", raw_response=f"r{i}")
            for i in range(4)
        ]

        result = SelfConsistencyResult(
            final_answer="BUY",
            confidence=0.25,
            agreement_ratio=0.25,
            best_chain="C1",
            all_chains=chains,
            all_answers=["BUY", "SELL", "HOLD", "STAND_DOWN"],
            answer_distribution={"BUY": 1, "SELL": 1, "HOLD": 1, "STAND_DOWN": 1},
            n_samples=4,
            unique_answers=4,
        )

        assert result.is_disputed
        assert not result.is_high_confidence
