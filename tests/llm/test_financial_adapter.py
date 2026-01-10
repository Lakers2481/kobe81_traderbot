"""
Comprehensive Unit Tests for Financial LLM Adapter Module
==========================================================

Tests the financial domain adaptation for LLM outputs without
full fine-tuning.

Run: python -m pytest tests/llm/test_financial_adapter.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime


class TestFinancialLLMAdapter:
    """Tests for the main FinancialLLMAdapter class."""

    def test_initialization_without_provider(self):
        """Test initialization without base provider."""
        from llm.financial_adapter import FinancialLLMAdapter

        adapter = FinancialLLMAdapter(base_provider=None)

        # Internal provider should be None until accessed
        assert adapter._provider is None
        assert adapter.FINANCIAL_SYSTEM_PROMPT is not None

    def test_initialization_with_provider(self):
        """Test initialization with mock provider."""
        from llm.financial_adapter import FinancialLLMAdapter

        mock_provider = Mock()
        adapter = FinancialLLMAdapter(base_provider=mock_provider)

        assert adapter._provider is mock_provider

    def test_financial_system_prompt_content(self):
        """Test that system prompt contains financial expertise."""
        from llm.financial_adapter import FinancialLLMAdapter

        adapter = FinancialLLMAdapter(base_provider=None)
        prompt = adapter.FINANCIAL_SYSTEM_PROMPT

        # Should contain key financial terms
        assert "technical analysis" in prompt.lower() or "trading" in prompt.lower()
        assert "risk" in prompt.lower()

    def test_few_shot_examples_exist(self):
        """Test that few-shot examples are defined."""
        from llm.financial_adapter import FinancialLLMAdapter

        adapter = FinancialLLMAdapter(base_provider=None)

        assert hasattr(adapter, 'FEW_SHOT_EXAMPLES')
        assert len(adapter.FEW_SHOT_EXAMPLES) > 0

    def test_chat_with_mock_provider(self):
        """Test that chat method works with mock provider."""
        from llm.financial_adapter import FinancialLLMAdapter
        from llm.provider_base import LLMMessage

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = "Analysis complete."
        mock_provider.chat = Mock(return_value=mock_response)

        adapter = FinancialLLMAdapter(base_provider=mock_provider, validate_outputs=False)

        messages = [LLMMessage(role="user", content="Analyze AAPL")]
        response = adapter.chat(messages)

        # Should have called underlying provider
        mock_provider.chat.assert_called_once()
        assert response.content == "Analysis complete."

    def test_is_trading_query_detection(self):
        """Test detection of trading-related queries."""
        from llm.financial_adapter import FinancialLLMAdapter
        from llm.provider_base import LLMMessage

        adapter = FinancialLLMAdapter(base_provider=None)

        # Trading queries with LLMMessage objects
        # Keywords: buy, sell, trade, stock, position, signal, bullish, bearish,
        # long, short, entry, exit, stop, target, rsi, macd, regime, vix
        trading_messages = [
            [LLMMessage(role="user", content="Should I buy AAPL?")],
            [LLMMessage(role="user", content="Is this stock a good trade?")],
            [LLMMessage(role="user", content="What's the RSI on this stock?")],
            [LLMMessage(role="user", content="Is this a good entry point?")],
        ]

        for messages in trading_messages:
            assert adapter._is_trading_query(messages), f"Should detect as trading: {messages}"

    def test_is_trading_query_non_trading(self):
        """Test that non-trading queries are not flagged."""
        from llm.financial_adapter import FinancialLLMAdapter
        from llm.provider_base import LLMMessage

        adapter = FinancialLLMAdapter(base_provider=None)

        # Non-trading queries
        non_trading = [
            [LLMMessage(role="user", content="What's the weather like?")],
            [LLMMessage(role="user", content="Tell me a joke")],
        ]

        for messages in non_trading:
            result = adapter._is_trading_query(messages)
            # Should return False for non-trading queries
            assert not result


class TestFeedbackLoop:
    """Tests for implicit feedback learning."""

    def test_record_outcome(self):
        """Test recording trade outcomes for learning."""
        from llm.financial_adapter import FinancialLLMAdapter
        from pathlib import Path
        import tempfile

        # Use temp file for feedback store
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        adapter = FinancialLLMAdapter(base_provider=None, feedback_store_path=temp_path)

        adapter.record_outcome(
            query="Should I buy AAPL?",
            response="Yes, buy AAPL at $175",
            outcome={
                "pnl": 250.0,
                "r_multiple": 1.5,
                "win": True,
            },
        )

        assert len(adapter._feedback_store) == 1
        assert adapter._feedback_store[0].outcome['win'] is True

        # Cleanup
        temp_path.unlink(missing_ok=True)

    def test_record_multiple_outcomes(self):
        """Test recording multiple outcomes."""
        from llm.financial_adapter import FinancialLLMAdapter
        from pathlib import Path
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        adapter = FinancialLLMAdapter(base_provider=None, feedback_store_path=temp_path)

        for i in range(5):
            adapter.record_outcome(
                query=f"Query {i}",
                response=f"Response {i}",
                outcome={"pnl": 100 * i, "r_multiple": 1.0, "win": i % 2 == 0},
            )

        assert len(adapter._feedback_store) == 5

        # Cleanup
        temp_path.unlink(missing_ok=True)


class TestValidation:
    """Tests for claim validation."""

    def test_validate_claims_method_exists(self):
        """Test that validate method exists."""
        from llm.financial_adapter import FinancialLLMAdapter

        adapter = FinancialLLMAdapter(base_provider=None)

        # Method should exist
        assert hasattr(adapter, '_validate_and_correct')

    def test_validate_response_passthrough(self):
        """Test that responses pass through validation."""
        from llm.financial_adapter import FinancialLLMAdapter
        from llm.provider_base import LLMMessage

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = "AAPL is at $175, RSI is 28 (oversold)."
        mock_provider.chat = Mock(return_value=mock_response)

        adapter = FinancialLLMAdapter(base_provider=mock_provider, validate_outputs=False)

        messages = [LLMMessage(role="user", content="Analyze AAPL")]
        response = adapter.chat(messages)

        # Response should pass through
        assert response is not None


class TestSingletonPattern:
    """Tests for singleton accessor."""

    def test_get_financial_adapter_returns_instance(self):
        """Test singleton accessor."""
        from llm.financial_adapter import get_financial_adapter
        import llm.financial_adapter as fa_module

        # Reset singleton
        fa_module._financial_adapter_instance = None

        adapter1 = get_financial_adapter()
        adapter2 = get_financial_adapter()

        assert adapter1 is adapter2

    def test_get_financial_adapter_parameters(self):
        """Test singleton with parameters."""
        from llm.financial_adapter import get_financial_adapter
        import llm.financial_adapter as fa_module

        # Reset singleton
        fa_module._financial_adapter_instance = None

        adapter = get_financial_adapter(use_few_shot=True, validate_outputs=True)

        assert adapter._use_few_shot is True
        assert adapter._validate_outputs is True


class TestFewShotExamples:
    """Tests for few-shot example handling."""

    def test_get_few_shot_messages(self):
        """Test generation of few-shot messages."""
        from llm.financial_adapter import FinancialLLMAdapter
        from llm.provider_base import LLMMessage

        adapter = FinancialLLMAdapter(base_provider=None)

        # Method should exist and work
        messages = [LLMMessage(role="user", content="Should I buy AAPL?")]
        few_shot = adapter._get_relevant_few_shot(messages)

        # Should return list of messages
        assert isinstance(few_shot, list)

    def test_few_shot_examples_structure(self):
        """Test structure of few-shot examples."""
        from llm.financial_adapter import FinancialLLMAdapter, FewShotExample

        adapter = FinancialLLMAdapter(base_provider=None)

        examples = adapter.FEW_SHOT_EXAMPLES

        for example in examples:
            # Each example should be a FewShotExample dataclass
            assert isinstance(example, FewShotExample)
            assert hasattr(example, 'input_query')
            assert hasattr(example, 'good_response')


class TestTemperatureHandling:
    """Tests for temperature parameter handling."""

    def test_chat_with_temperature(self):
        """Test temperature is passed to provider."""
        from llm.financial_adapter import FinancialLLMAdapter
        from llm.provider_base import LLMMessage

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = "Response"
        mock_provider.chat = Mock(return_value=mock_response)

        adapter = FinancialLLMAdapter(base_provider=mock_provider, validate_outputs=False)

        adapter.chat([LLMMessage(role="user", content="Test")], temperature=0.2)

        # Check that chat was called with temperature
        mock_provider.chat.assert_called()
        call_kwargs = mock_provider.chat.call_args[1]
        assert call_kwargs.get('temperature') == 0.2


class TestErrorHandling:
    """Tests for error handling."""

    def test_empty_messages(self):
        """Test handling of empty messages list."""
        from llm.financial_adapter import FinancialLLMAdapter

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = "Empty response"
        mock_provider.chat = Mock(return_value=mock_response)

        adapter = FinancialLLMAdapter(base_provider=mock_provider, validate_outputs=False)

        # Should handle empty messages
        result = adapter.chat([])

        # Should still call provider (with system prompt at minimum)
        mock_provider.chat.assert_called()


class TestIntegrationWithRouter:
    """Tests for integration with LLM router."""

    def test_financial_mode_attribute(self):
        """Test that financial adapter can be identified."""
        from llm.financial_adapter import FinancialLLMAdapter

        adapter = FinancialLLMAdapter(base_provider=None)

        # Should be identifiable as financial adapter
        assert hasattr(adapter, 'FINANCIAL_SYSTEM_PROMPT')

    def test_adapter_wraps_provider(self):
        """Test that adapter wraps underlying provider."""
        from llm.financial_adapter import FinancialLLMAdapter

        mock_provider = Mock()
        adapter = FinancialLLMAdapter(base_provider=mock_provider)

        # Adapter should store provider reference
        assert adapter._provider is mock_provider


class TestMessageFormatting:
    """Tests for message formatting."""

    def test_format_user_message(self):
        """Test user message formatting."""
        from llm.financial_adapter import FinancialLLMAdapter
        from llm.provider_base import LLMMessage

        adapter = FinancialLLMAdapter(base_provider=None)

        # Messages should be LLMMessage objects with trading keywords
        messages = [LLMMessage(role="user", content="Should I buy TSLA stock?")]

        # Should handle correctly
        assert adapter._is_trading_query(messages)

    def test_format_system_message_injection(self):
        """Test system message injection."""
        from llm.financial_adapter import FinancialLLMAdapter
        from llm.provider_base import LLMMessage

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = "Analysis"
        mock_provider.chat = Mock(return_value=mock_response)

        adapter = FinancialLLMAdapter(base_provider=mock_provider, validate_outputs=False)

        messages = [LLMMessage(role="user", content="Test")]
        adapter.chat(messages)

        # Get the messages passed to provider
        call_args = mock_provider.chat.call_args[0][0]

        # First message should be system prompt
        first_msg = call_args[0]
        assert first_msg.role == 'system'
        assert 'trading' in first_msg.content.lower() or 'risk' in first_msg.content.lower()


class TestTradeOutcome:
    """Tests for TradeOutcome dataclass."""

    def test_trade_outcome_creation(self):
        """Test TradeOutcome instantiation."""
        from llm.financial_adapter import TradeOutcome

        outcome = TradeOutcome(
            query="Buy AAPL?",
            response="Yes, buy at $175",
            outcome={"pnl": 100, "win": True}
        )

        assert outcome.query == "Buy AAPL?"
        assert outcome.response == "Yes, buy at $175"
        assert outcome.outcome["pnl"] == 100
        assert outcome.timestamp is not None


class TestFewShotExample:
    """Tests for FewShotExample dataclass."""

    def test_few_shot_example_creation(self):
        """Test FewShotExample instantiation."""
        from llm.financial_adapter import FewShotExample

        example = FewShotExample(
            input_query="Should I buy?",
            good_response="Here's a detailed analysis...",
            bad_response="Yes buy it.",
            category="technical"
        )

        assert example.input_query == "Should I buy?"
        assert example.category == "technical"


class TestFeedbackStats:
    """Tests for feedback statistics."""

    def test_get_feedback_stats_empty(self):
        """Test feedback stats with no records."""
        from llm.financial_adapter import FinancialLLMAdapter
        from pathlib import Path
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        adapter = FinancialLLMAdapter(base_provider=None, feedback_store_path=temp_path)

        stats = adapter.get_feedback_stats()

        assert stats['total'] == 0
        assert stats['wins'] == 0
        assert stats['losses'] == 0
        assert stats['win_rate'] == 0.0

        temp_path.unlink(missing_ok=True)

    def test_get_feedback_stats_with_records(self):
        """Test feedback stats with records."""
        from llm.financial_adapter import FinancialLLMAdapter
        from pathlib import Path
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        adapter = FinancialLLMAdapter(base_provider=None, feedback_store_path=temp_path)

        # Record some outcomes
        adapter.record_outcome("Q1", "R1", {"pnl": 100, "win": True})
        adapter.record_outcome("Q2", "R2", {"pnl": 50, "win": True})
        adapter.record_outcome("Q3", "R3", {"pnl": -30, "win": False})

        stats = adapter.get_feedback_stats()

        assert stats['total'] == 3
        assert stats['wins'] == 2
        assert stats['losses'] == 1
        assert abs(stats['win_rate'] - 0.667) < 0.01

        temp_path.unlink(missing_ok=True)
