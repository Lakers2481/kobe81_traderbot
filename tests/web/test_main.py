import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Import the FastAPI app directly
from web.main import app

# Create a TestClient instance for the FastAPI app
client = TestClient(app)

# Fixtures for mocking component getters
@pytest.fixture
def mock_signal_processor():
    with patch('web.main.get_signal_processor') as mock_getter:
        mock_processor = MagicMock()
        mock_processor._active_episodes = {"mock_episode_1": "abc"}
        mock_processor.get_cognitive_status.return_value = {
            "processor_active": True,
            "brain_status": {"initialized": True, "decision_count": 5},
            "active_episodes": 1,
        }
        mock_getter.return_value = mock_processor
        yield mock_processor

@pytest.fixture
def mock_reflection_engine():
    with patch('web.main.get_reflection_engine') as mock_getter:
        mock_engine = MagicMock()
        mock_reflection = MagicMock()
        mock_reflection.to_dict.return_value = {
            "scope": "episode",
            "summary": "LLM analysis: great trade!",
            "llm_critique": "Mocked LLM critique here.",
        }
        mock_engine.get_recent_reflections.return_value = [mock_reflection]
        mock_getter.return_value = mock_engine
        yield mock_engine

@pytest.fixture
def mock_tca_analyzer():
    with patch('web.main.get_tca_analyzer') as mock_getter:
        mock_analyzer = MagicMock()
        mock_analyzer.get_summary_tca_metrics.return_value = {
            "total_trades": 10,
            "avg_slippage_bps": 2.5,
            "total_cost_usd": 15.0,
        }
        mock_getter.return_value = mock_analyzer
        yield mock_analyzer

@pytest.fixture
def mock_self_model():
    with patch('web.main.get_self_model') as mock_getter:
        mock_model = MagicMock()
        mock_model.get_self_description.return_value = "I am a powerful AI bot."
        mock_getter.return_value = mock_model
        yield mock_model


class TestDashboardEndpoints:
    def test_read_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "Kobe81 Traderbot Dashboard" in response.text
        assert '<a href="/docs">' in response.text

    def test_get_bot_status(self, mock_signal_processor):
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["overall_health"] == "operational"
        assert data["active_cognitive_episodes"] == 1
        assert "timestamp" in data

    def test_get_bot_status_error(self, mock_signal_processor):
        # Mock _active_episodes to raise an error when accessed (tests error handling in /status)
        type(mock_signal_processor)._active_episodes = property(
            lambda self: (_ for _ in ()).throw(Exception("Service unavailable"))
        )
        response = client.get("/status")
        assert response.status_code == 500
        assert "Service unavailable" in response.json()["detail"]

    def test_get_cognitive_system_status(self, mock_signal_processor):
        response = client.get("/cognitive_status")
        assert response.status_code == 200
        data = response.json()
        assert data["processor_active"] is True
        assert data["brain_status"]["decision_count"] == 5
        mock_signal_processor.get_cognitive_status.assert_called_once()

    def test_get_cognitive_system_status_error(self, mock_signal_processor):
        mock_signal_processor.get_cognitive_status.side_effect = Exception("Brain error")
        response = client.get("/cognitive_status")
        assert response.status_code == 500
        assert "Brain error" in response.json()["detail"]

    def test_get_recent_reflections(self, mock_reflection_engine):
        response = client.get("/recent_reflections")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["summary"] == "LLM analysis: great trade!"
        assert data[0]["llm_critique"] == "Mocked LLM critique here."
        mock_reflection_engine.get_recent_reflections.assert_called_once_with(limit=10)

    def test_get_recent_reflections_error(self, mock_reflection_engine):
        mock_reflection_engine.get_recent_reflections.side_effect = Exception("Reflection storage down")
        response = client.get("/recent_reflections")
        assert response.status_code == 500
        assert "Reflection storage down" in response.json()["detail"]

    def test_get_recent_tca(self, mock_tca_analyzer):
        response = client.get("/recent_tca")
        assert response.status_code == 200
        data = response.json()
        assert data["total_trades"] == 10
        assert data["avg_slippage_bps"] == 2.5
        mock_tca_analyzer.get_summary_tca_metrics.assert_called_once_with(lookback_days=7)

    def test_get_recent_tca_error(self, mock_tca_analyzer):
        mock_tca_analyzer.get_summary_tca_metrics.side_effect = Exception("TCA calculation failed")
        response = client.get("/recent_tca")
        assert response.status_code == 500
        assert "TCA calculation failed" in response.json()["detail"]

    def test_get_ai_self_description(self, mock_self_model):
        response = client.get("/self_model_description")
        assert response.status_code == 200
        data = response.json()
        assert data["self_description"] == "I am a powerful AI bot."
        mock_self_model.get_self_description.assert_called_once()

    def test_get_ai_self_description_error(self, mock_self_model):
        mock_self_model.get_self_description.side_effect = Exception("Self-model corrupted")
        response = client.get("/self_model_description")
        assert response.status_code == 500
        assert "Self-model corrupted" in response.json()["detail"]
