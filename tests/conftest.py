"""
Pytest configuration and shared fixtures for Kobe trading tests.
"""

import os
import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Generate realistic price data
    base_price = 100
    returns = np.random.randn(100) * 0.02  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(100) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(100) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(100) * 0.01)),
        'close': prices,
        'volume': np.random.randint(100000, 10000000, 100),
    })

    # Ensure high >= open, close, low and low <= open, close, high
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def sample_universe():
    """Generate sample universe list."""
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'V']


@pytest.fixture
def temp_state_dir(tmp_path):
    """Create temporary state directory."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    # Create initial files
    (state_dir / "positions.json").write_text("[]")
    (state_dir / "order_history.json").write_text("[]")

    return state_dir


@pytest.fixture
def temp_logs_dir(tmp_path):
    """Create temporary logs directory."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    return logs_dir


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set mock environment variables for testing."""
    monkeypatch.setenv("POLYGON_API_KEY", "test_polygon_key")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "test_alpaca_key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "test_alpaca_secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
