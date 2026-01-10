"""
Centralized test fixtures for Kobe trading system.

This module provides reusable fixtures for:
- Market data generation (OHLCV, patterns, signals)
- Broker mocking (Alpaca API)
- Data provider mocking (Polygon, Stooq, YFinance)
- State file helpers (positions, orders, hash chain)
- Signal generation (valid, invalid, batches)
"""

from .market_data import (
    generate_ohlcv,
    generate_multi_symbol_ohlcv,
    generate_signal_triggering_data,
    generate_gap_data,
    generate_ibs_rsi_trigger_data,
    generate_turtle_soup_trigger_data,
)
from .signals import (
    create_valid_signal,
    create_invalid_signal,
    create_signal_batch,
    create_dual_strategy_signal,
)
from .broker_mocks import (
    MockAlpacaBroker,
    mock_alpaca_api,
    mock_order_success,
    mock_order_rejected,
    mock_quote_response,
    mock_account_response,
    mock_positions_response,
)
from .provider_mocks import (
    mock_polygon_api,
    mock_stooq_api,
    failing_polygon_api,
    rate_limited_polygon_api,
    mock_polygon_response_data,
)
from .state_helpers import (
    create_test_state_dir,
    create_positions_file,
    create_order_state_file,
    create_hash_chain_file,
    create_idempotency_db,
    verify_hash_chain_integrity,
    corrupt_positions_file,
    corrupt_json_file,
    delete_state_file,
)

__all__ = [
    # Market data
    "generate_ohlcv",
    "generate_multi_symbol_ohlcv",
    "generate_signal_triggering_data",
    "generate_gap_data",
    "generate_ibs_rsi_trigger_data",
    "generate_turtle_soup_trigger_data",
    # Signals
    "create_valid_signal",
    "create_invalid_signal",
    "create_signal_batch",
    "create_dual_strategy_signal",
    # Broker mocks
    "MockAlpacaBroker",
    "mock_alpaca_api",
    "mock_order_success",
    "mock_order_rejected",
    "mock_quote_response",
    "mock_account_response",
    "mock_positions_response",
    # Provider mocks
    "mock_polygon_api",
    "mock_stooq_api",
    "failing_polygon_api",
    "rate_limited_polygon_api",
    "mock_polygon_response_data",
    # State helpers
    "create_test_state_dir",
    "create_positions_file",
    "create_order_state_file",
    "create_hash_chain_file",
    "create_idempotency_db",
    "verify_hash_chain_integrity",
    "corrupt_positions_file",
    "corrupt_json_file",
    "delete_state_file",
]
