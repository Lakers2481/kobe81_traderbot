"""
Pytest configuration and shared fixtures for Kobe trading tests.
"""

import os

# REPRODUCIBILITY FIX: Disable oneDNN for deterministic test results
# Must be set BEFORE TensorFlow is imported anywhere
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
    """Generate sample OHLCV data for testing with symbol column."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=250, freq='D')  # ~1 year for SMA200

    # Generate realistic price data
    base_price = 100
    returns = np.random.randn(250) * 0.02  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'TEST',  # Add symbol column
        'open': prices * (1 + np.random.randn(250) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(250) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(250) * 0.01)),
        'close': prices,
        'volume': np.random.randint(100000, 10000000, 250),
    })

    # Ensure high >= open, close, low and low <= open, close, high
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def sample_multi_symbol_data():
    """Generate sample OHLCV data for multiple symbols."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

    dfs = []
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        base_price = np.random.randint(100, 500)
        returns = np.random.randn(250) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'open': prices * (1 + np.random.randn(250) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(250) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(250) * 0.01)),
            'close': prices,
            'volume': np.random.randint(100000, 10000000, 250),
        })

        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


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
    monkeypatch.setenv("APCA_API_KEY_ID", "test_alpaca_key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test_alpaca_secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


# =============================================================================
# NEW INTEGRATION TEST FIXTURES
# =============================================================================

@pytest.fixture
def frozen_time():
    """
    Fixture for freezing time at specific moments.

    Usage:
        def test_something(frozen_time):
            with frozen_time("2026-01-06 10:00:00"):
                # Time is frozen here
                pass
    """
    try:
        from freezegun import freeze_time
        return freeze_time
    except ImportError:
        pytest.skip("freezegun not installed")


@pytest.fixture
def mock_broker(requests_mock):
    """
    Set up full Alpaca broker mock.

    Returns MockAlpacaBroker instance for assertions and state tracking.
    """
    from tests.fixtures.broker_mocks import mock_alpaca_api
    return mock_alpaca_api(requests_mock)


@pytest.fixture
def mock_all_providers(requests_mock):
    """
    Set up all data provider mocks (Polygon, Stooq).

    Returns the requests_mock for additional customization.
    """
    from tests.fixtures.provider_mocks import mock_polygon_api, mock_stooq_api
    mock_polygon_api(requests_mock)
    mock_stooq_api(requests_mock)
    return requests_mock


@pytest.fixture
def integration_state(tmp_path, mock_env_vars):
    """
    Complete integration test state.

    Creates isolated state directory with all required files.
    """
    from tests.fixtures.state_helpers import create_test_state_dir
    state_dir = create_test_state_dir(tmp_path)
    return {
        'state_dir': state_dir,
        'tmp_path': tmp_path,
    }


@pytest.fixture
def signal_triggering_data():
    """
    Generate data that will produce a DualStrategy signal.

    Returns DataFrame with OHLCV data designed to trigger signals.
    """
    from tests.fixtures.market_data import generate_signal_triggering_data
    return generate_signal_triggering_data('dual')


@pytest.fixture
def valid_signal():
    """
    Create a valid signal that passes quality gate.

    Returns signal dictionary ready for processing.
    """
    from tests.fixtures.signals import create_valid_signal
    return create_valid_signal()


@pytest.fixture
def invalid_signal():
    """
    Create an invalid signal that fails quality gate.

    Returns signal dictionary that will be rejected.
    """
    from tests.fixtures.signals import create_invalid_signal
    return create_invalid_signal()


@pytest.fixture
def signal_batch():
    """
    Create a batch of mixed valid/invalid signals.

    Returns list of 5 signals (3 valid, 2 invalid).
    """
    from tests.fixtures.signals import create_signal_batch
    return create_signal_batch(count=5, valid_ratio=0.6)


@pytest.fixture
def hash_chain_with_entries(tmp_path):
    """
    Create a hash chain with sample entries.

    Returns path to populated hash_chain.jsonl file.
    """
    from tests.fixtures.state_helpers import create_test_state_dir, create_hash_chain_file
    from datetime import datetime

    state_dir = create_test_state_dir(tmp_path)
    entries = [
        {
            "timestamp": datetime.now().isoformat(),
            "event": "order_placed",
            "symbol": "AAPL",
            "side": "buy",
            "qty": 10,
        },
        {
            "timestamp": datetime.now().isoformat(),
            "event": "order_filled",
            "symbol": "AAPL",
            "side": "buy",
            "qty": 10,
            "fill_price": 150.0,
        },
    ]
    return create_hash_chain_file(state_dir, entries)


@pytest.fixture
def idempotency_db(tmp_path):
    """
    Create idempotency database with sample entries.

    Returns path to SQLite database.
    """
    from tests.fixtures.state_helpers import create_test_state_dir, create_idempotency_db
    from datetime import datetime

    state_dir = create_test_state_dir(tmp_path)
    entries = [
        {
            "key": "test_key_1",
            "symbol": "AAPL",
            "side": "buy",
            "created_at": datetime.now().isoformat(),
            "order_id": "order_123",
            "status": "filled",
        },
    ]
    return create_idempotency_db(state_dir, entries)
