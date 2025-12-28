import pytest
from unittest.mock import MagicMock, patch
import os
from datetime import datetime

from cognitive.llm_narrative_analyzer import (
    LLMNarrativeAnalyzer,
    LLMHypothesis,
    LLMAnalysisResult,
    StrategyIdea,
    get_llm_analyzer,
)
from cognitive.reflection_engine import Reflection


# Mock the load_dotenv to prevent actual .env loading during tests
@pytest.fixture(autouse=True)
def mock_load_dotenv():
    with patch('cognitive.llm_narrative_analyzer.load_dotenv') as mock_dotenv:
        yield mock_dotenv

# Fixture to mock os.getenv for API key
@pytest.fixture
def mock_anthropic_api_key():
    with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'mock_api_key'}):
        yield

# Fixture for a sample Reflection object
@pytest.fixture
def sample_reflection():
    return Reflection(
        scope="episode",
        timestamp=datetime.now(),
        summary="Trade of AAPL resulted in a loss.",
        what_went_well=["Identified signal correctly."],
        what_went_wrong=["Entry was too early.", "Ignored VIX spike."],
        lessons=["Wait for VIX to settle.", "Confirm entry with volume."],
        action_items=["Strengthen VIX rule.", "Add volume confirmation step."],
        llm_critique=None,
        metadata={"trade_id": "12345", "symbol": "AAPL"}
    )

# Test initialization of LLMNarrativeAnalyzer
def test_llm_analyzer_init_with_api_key(mock_anthropic_api_key):
    analyzer = LLMNarrativeAnalyzer()
    assert analyzer._client is not None

def test_llm_analyzer_init_without_api_key():
    # Ensure ANTHROPIC_API_KEY is not set for this test
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]
    analyzer = LLMNarrativeAnalyzer()
    assert analyzer._client is None

# Test _build_prompt method
def test_build_prompt(sample_reflection):
    analyzer = LLMNarrativeAnalyzer() # Client will be None, but _build_prompt doesn't use it
    prompt = analyzer._build_prompt(sample_reflection)
    assert "Trade of AAPL resulted in a loss." in prompt
    assert "What Went Well" in prompt
    assert "What Went Wrong" in prompt
    assert "Lessons Identified by the System" in prompt
    assert "YOUR ANALYSIS" in prompt
    assert "Suggested Hypotheses" in prompt
    assert "Meta-Critique" in prompt

# Test analyze_reflection with successful LLM response
@patch('anthropic.Client')
def test_analyze_reflection_success(mock_client, mock_anthropic_api_key, sample_reflection):
    # Configure the mock client's messages.create method
    mock_response_content = MagicMock()
    mock_response_content.text = "Mocked LLM analysis: interesting patterns.\nHYPOTHESIS: Test hypothesis\nCONDITION: vix > 25\nPREDICTION: win_rate > 0.6\nRATIONALE: Testing the hypothesis parsing"
    mock_client.return_value.messages.create.return_value.content = [mock_response_content]

    analyzer = LLMNarrativeAnalyzer()
    result = analyzer.analyze_reflection(sample_reflection)

    assert isinstance(result, LLMAnalysisResult)
    assert result.critique == mock_response_content.text
    assert isinstance(result.hypotheses, list)
    assert len(result.hypotheses) == 1
    assert result.hypotheses[0].description == "Test hypothesis"
    assert result.hypotheses[0].condition == "vix > 25"
    mock_client.return_value.messages.create.assert_called_once()
    args, kwargs = mock_client.return_value.messages.create.call_args
    assert kwargs['model'] == "claude-3-haiku-20240307"
    assert "Trade of AAPL resulted in a loss." in kwargs['messages'][0]['content']


# Test analyze_reflection when LLM client is not available
def test_analyze_reflection_no_client(sample_reflection):
    # Ensure ANTHROPIC_API_KEY is not set for this test
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]
    analyzer = LLMNarrativeAnalyzer()
    assert analyzer._client is None
    result = analyzer.analyze_reflection(sample_reflection)
    assert isinstance(result, LLMAnalysisResult)
    assert result.critique is None
    assert result.hypotheses == []
    assert result.strategy_ideas == []


# Test analyze_reflection with API error
@patch('anthropic.Client')
def test_analyze_reflection_api_error(mock_client, mock_anthropic_api_key, sample_reflection):
    mock_client.return_value.messages.create.side_effect = Exception("API rate limit exceeded")
    analyzer = LLMNarrativeAnalyzer()
    result = analyzer.analyze_reflection(sample_reflection)
    assert isinstance(result, LLMAnalysisResult)
    assert result.critique is None
    assert result.hypotheses == []
    mock_client.return_value.messages.create.assert_called_once()


# Test get_llm_analyzer singleton
def test_get_llm_analyzer_singleton(mock_anthropic_api_key):
    analyzer1 = get_llm_analyzer()
    analyzer2 = get_llm_analyzer()
    assert analyzer1 is analyzer2
    assert isinstance(analyzer1, LLMNarrativeAnalyzer)


# Tests for Strategy Idea functionality

class TestStrategyIdeaParsing:
    """Test strategy idea parsing from LLM responses."""

    def test_parse_strategy_ideas_single(self):
        """Test parsing a single strategy idea from LLM response."""
        analyzer = LLMNarrativeAnalyzer()
        llm_response = """
        Here is my analysis of the trading performance.

        --- STRATEGY IDEA ---
        NAME: VIX_Spike_Reversal
        CONCEPT: Buy oversold stocks after VIX spikes above 35
        MARKET_CONTEXT: high volatility regimes
        ENTRY_CONDITIONS:
        - VIX has spiked above 35 within last 5 days
        - Stock has fallen more than 10% from 20-day high
        - RSI(2) is below 10
        EXIT_CONDITIONS:
        - Stop loss at -5% from entry
        - Profit target at +10%
        - Time stop after 5 trading days
        RISK_MANAGEMENT: Position size 2% of portfolio max
        RATIONALE: High VIX creates panic selling that often reverses quickly
        --- END STRATEGY IDEA ---
        """

        ideas = analyzer._parse_strategy_ideas(llm_response)

        assert len(ideas) == 1
        idea = ideas[0]
        assert idea.name == "VIX_Spike_Reversal"
        assert "oversold stocks" in idea.concept
        assert idea.market_context == "high volatility regimes"
        assert len(idea.entry_conditions) == 3
        assert len(idea.exit_conditions) == 3
        assert "2% of portfolio" in idea.risk_management
        assert "panic selling" in idea.rationale

    def test_parse_strategy_ideas_multiple(self):
        """Test parsing multiple strategy ideas from LLM response."""
        analyzer = LLMNarrativeAnalyzer()
        llm_response = """
        --- STRATEGY IDEA ---
        NAME: Strategy_One
        CONCEPT: First strategy concept
        MARKET_CONTEXT: bull market
        ENTRY_CONDITIONS:
        - Condition 1
        EXIT_CONDITIONS:
        - Exit 1
        RISK_MANAGEMENT: Standard sizing
        RATIONALE: First rationale
        --- END STRATEGY IDEA ---

        Some text in between

        --- STRATEGY IDEA ---
        NAME: Strategy_Two
        CONCEPT: Second strategy concept
        MARKET_CONTEXT: bear market
        ENTRY_CONDITIONS:
        - Condition 2
        EXIT_CONDITIONS:
        - Exit 2
        RISK_MANAGEMENT: Conservative sizing
        RATIONALE: Second rationale
        --- END STRATEGY IDEA ---
        """

        ideas = analyzer._parse_strategy_ideas(llm_response)

        assert len(ideas) == 2
        assert ideas[0].name == "Strategy_One"
        assert ideas[1].name == "Strategy_Two"
        assert ideas[0].market_context == "bull market"
        assert ideas[1].market_context == "bear market"

    def test_parse_strategy_ideas_no_ideas(self):
        """Test parsing when no strategy ideas are present."""
        analyzer = LLMNarrativeAnalyzer()
        llm_response = "Just regular analysis without any strategy ideas."

        ideas = analyzer._parse_strategy_ideas(llm_response)

        assert len(ideas) == 0

    def test_parse_strategy_ideas_incomplete(self):
        """Test parsing strategy idea with missing required fields."""
        analyzer = LLMNarrativeAnalyzer()
        llm_response = """
        --- STRATEGY IDEA ---
        NAME: Incomplete_Strategy
        --- END STRATEGY IDEA ---
        """

        ideas = analyzer._parse_strategy_ideas(llm_response)

        # Should be empty because CONCEPT is required
        assert len(ideas) == 0


class TestShouldRequestStrategyIdeas:
    """Test the _should_request_strategy_ideas method."""

    def test_should_request_for_daily_scope(self):
        """Daily reflections should request strategy ideas."""
        analyzer = LLMNarrativeAnalyzer()
        reflection = Reflection(scope="daily")

        assert analyzer._should_request_strategy_ideas(reflection) is True

    def test_should_request_for_weekly_scope(self):
        """Weekly reflections should request strategy ideas."""
        analyzer = LLMNarrativeAnalyzer()
        reflection = Reflection(scope="weekly")

        assert analyzer._should_request_strategy_ideas(reflection) is True

    def test_should_request_for_many_failures(self):
        """Multiple failures should trigger strategy idea request."""
        analyzer = LLMNarrativeAnalyzer()
        reflection = Reflection(
            scope="episode",
            what_went_wrong=["Failure 1", "Failure 2"]
        )

        assert analyzer._should_request_strategy_ideas(reflection) is True

    def test_should_request_for_trigger_phrases(self):
        """Specific phrases in summary should trigger strategy idea request."""
        analyzer = LLMNarrativeAnalyzer()

        for phrase in ["repeated loss", "underperform", "new regime", "need different"]:
            reflection = Reflection(
                scope="episode",
                summary=f"Trade analysis shows {phrase} pattern."
            )
            assert analyzer._should_request_strategy_ideas(reflection) is True

    def test_should_not_request_for_simple_episode(self):
        """Simple episode reflections should not request strategy ideas."""
        analyzer = LLMNarrativeAnalyzer()
        reflection = Reflection(
            scope="episode",
            summary="Trade completed successfully.",
            what_went_wrong=[]
        )

        assert analyzer._should_request_strategy_ideas(reflection) is False


class TestStrategyIdeaDataclass:
    """Test StrategyIdea dataclass functionality."""

    def test_strategy_idea_to_dict(self):
        """Test serialization of StrategyIdea to dictionary."""
        idea = StrategyIdea(
            name="Test_Strategy",
            concept="Test concept",
            market_context="bull market",
            entry_conditions=["Condition 1", "Condition 2"],
            exit_conditions=["Exit 1"],
            risk_management="2% risk",
            rationale="Test rationale",
            confidence=0.7
        )

        d = idea.to_dict()

        assert d['name'] == "Test_Strategy"
        assert d['concept'] == "Test concept"
        assert d['market_context'] == "bull market"
        assert d['entry_conditions'] == ["Condition 1", "Condition 2"]
        assert d['exit_conditions'] == ["Exit 1"]
        assert d['risk_management'] == "2% risk"
        assert d['rationale'] == "Test rationale"
        assert d['confidence'] == 0.7


class TestLLMAnalysisResultDataclass:
    """Test LLMAnalysisResult dataclass functionality."""

    def test_llm_analysis_result_to_dict(self):
        """Test serialization of LLMAnalysisResult to dictionary."""
        hypothesis = LLMHypothesis(
            description="Test hypothesis",
            condition="vix > 25",
            prediction="win_rate > 0.6",
            rationale="Test rationale",
            confidence=0.5
        )

        idea = StrategyIdea(
            name="Test_Strategy",
            concept="Test concept",
            market_context="bull",
            entry_conditions=["Entry 1"],
            exit_conditions=["Exit 1"],
            risk_management="2% risk",
            rationale="Test"
        )

        result = LLMAnalysisResult(
            critique="Test critique text",
            hypotheses=[hypothesis],
            strategy_ideas=[idea]
        )

        d = result.to_dict()

        assert d['critique'] == "Test critique text"
        assert len(d['hypotheses']) == 1
        assert len(d['strategy_ideas']) == 1
        assert d['hypotheses'][0]['description'] == "Test hypothesis"
        assert d['strategy_ideas'][0]['name'] == "Test_Strategy"


@patch('anthropic.Client')
def test_analyze_reflection_with_strategy_ideas(mock_client, mock_anthropic_api_key):
    """Test that strategy ideas are parsed when present in LLM response."""
    # Create a daily reflection (which should trigger strategy idea request)
    reflection = Reflection(
        scope="daily",
        summary="Daily review: multiple losses occurred",
        what_went_wrong=["Lost on AAPL", "Lost on MSFT"]
    )

    mock_response_content = MagicMock()
    mock_response_content.text = """
    Here is my analysis.

    HYPOTHESIS: Test hypothesis
    CONDITION: vix > 30
    PREDICTION: win_rate > 0.65
    RATIONALE: High volatility creates opportunities

    --- STRATEGY IDEA ---
    NAME: VIX_Mean_Reversion
    CONCEPT: Buy oversold stocks during VIX spikes
    MARKET_CONTEXT: high volatility regimes
    ENTRY_CONDITIONS:
    - VIX above 30
    - RSI below 20
    EXIT_CONDITIONS:
    - Stop at -3%
    - Target at +8%
    RISK_MANAGEMENT: 1% position size
    RATIONALE: Fear creates buying opportunities
    --- END STRATEGY IDEA ---
    """
    mock_client.return_value.messages.create.return_value.content = [mock_response_content]

    analyzer = LLMNarrativeAnalyzer()
    result = analyzer.analyze_reflection(reflection)

    assert isinstance(result, LLMAnalysisResult)
    assert result.critique is not None
    assert len(result.hypotheses) == 1
    assert len(result.strategy_ideas) == 1
    assert result.strategy_ideas[0].name == "VIX_Mean_Reversion"
