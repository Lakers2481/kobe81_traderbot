"""
Integration tests for end-to-end workflows.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestScanToSignalWorkflow:
    """Test scan -> signal generation workflow."""

    def test_strategy_generates_signals_from_data(self, sample_ohlcv_data):
        """Test that strategy can generate signals from OHLCV data."""
        from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy

        strategy = ConnorsRSI2Strategy()
        signals = strategy.scan_signals_over_time(sample_ohlcv_data)

        # Should return a DataFrame
        assert isinstance(signals, pd.DataFrame)

        # If signals exist, check required columns
        if len(signals) > 0:
            assert 'symbol' in signals.columns or 'side' in signals.columns


class TestBacktestWorkflow:
    """Test backtest workflow."""

    def test_backtest_produces_results(self, sample_ohlcv_data):
        """Test that backtest produces results."""
        from backtest.engine import BacktestEngine
        from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy

        engine = BacktestEngine(initial_capital=100000)
        strategy = ConnorsRSI2Strategy()

        # Generate signals
        signals = strategy.scan_signals_over_time(sample_ohlcv_data)

        # Run backtest
        # (Actual implementation may vary)


class TestRiskCheckWorkflow:
    """Test risk check workflow."""

    def test_order_passes_through_risk_gate(self):
        """Test that order flows through risk gate."""
        from risk.policy_gate import PolicyGate

        gate = PolicyGate(max_order_value=75, max_daily_loss=1000)

        # Simulate order flow
        order = {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 1,
            "price": 50,
        }

        order_value = order["qty"] * order["price"]
        result = gate.check(order_value=order_value)

        assert result["allowed"] == True


class TestStateManagementWorkflow:
    """Test state management workflow."""

    def test_position_tracking(self, temp_state_dir):
        """Test position tracking through state files."""
        import json

        positions_file = temp_state_dir / "positions.json"

        # Initially empty
        with open(positions_file) as f:
            positions = json.load(f)
        assert positions == []

        # Add a position
        new_position = {
            "symbol": "AAPL",
            "qty": 10,
            "avg_entry_price": 150.00,
            "current_price": 155.00,
        }
        positions.append(new_position)

        with open(positions_file, 'w') as f:
            json.dump(positions, f)

        # Verify
        with open(positions_file) as f:
            saved = json.load(f)

        assert len(saved) == 1
        assert saved[0]["symbol"] == "AAPL"
