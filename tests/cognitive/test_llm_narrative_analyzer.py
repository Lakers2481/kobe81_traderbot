import pytest
from unittest.mock import MagicMock, patch
import os
from datetime import datetime

from cognitive.llm_narrative_analyzer import LLMNarrativeAnalyzer, get_llm_analyzer
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
    mock_response_content.text = "Mocked LLM analysis: interesting patterns. Hypothesis: XYZ"
    mock_client.return_value.messages.create.return_value.content = [mock_response_content]

    analyzer = LLMNarrativeAnalyzer()
    result = analyzer.analyze_reflection(sample_reflection)

    assert result == "Mocked LLM analysis: interesting patterns. Hypothesis: XYZ"
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
    assert result is None

# Test analyze_reflection with API error
@patch('anthropic.Client')
def test_analyze_reflection_api_error(mock_client, mock_anthropic_api_key, sample_reflection):
    mock_client.return_value.messages.create.side_effect = Exception("API rate limit exceeded")
    analyzer = LLMNarrativeAnalyzer()
    result = analyzer.analyze_reflection(sample_reflection)
    assert result is None
    mock_client.return_value.messages.create.assert_called_once()


# Test get_llm_analyzer singleton
def test_get_llm_analyzer_singleton(mock_anthropic_api_key):
    analyzer1 = get_llm_analyzer()
    analyzer2 = get_llm_analyzer()
    assert analyzer1 is analyzer2
    assert isinstance(analyzer1, LLMNarrativeAnalyzer)
