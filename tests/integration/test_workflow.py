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
        """Test that DualStrategyScanner can generate signals from OHLCV data."""
        from strategies.registry import get_production_scanner

        scanner = get_production_scanner()
        signals = scanner.scan_signals_over_time(sample_ohlcv_data)

        # Should return a DataFrame
        assert isinstance(signals, pd.DataFrame)

        # If signals exist, check required columns
        if len(signals) > 0:
            assert 'symbol' in signals.columns or 'side' in signals.columns


class TestBacktestWorkflow:
    """Test backtest workflow."""

    def test_backtest_produces_results(self, sample_ohlcv_data):
        """Test that backtest produces results."""
        from backtest.engine import Backtester, BacktestConfig
        from strategies.registry import get_production_scanner

        scanner = get_production_scanner()

        # Create backtester with strategy's signal function
        cfg = BacktestConfig(initial_cash=100000)

        def fetch_data(symbol):
            # Return our test data
            df = sample_ohlcv_data.copy()
            df['symbol'] = symbol
            return df

        bt = Backtester(cfg, scanner.scan_signals_over_time, fetch_data)

        # Run on a single test symbol
        result = bt.run(['TEST'])

        # Should return a result dict
        assert isinstance(result, dict)
        assert 'trades' in result
        assert 'pnl' in result


class TestRiskCheckWorkflow:
    """Test risk check workflow."""

    def test_order_passes_through_risk_gate(self):
        """Test that order flows through risk gate."""
        from risk.policy_gate import PolicyGate, RiskLimits

        limits = RiskLimits(max_notional_per_order=75, max_daily_notional=1000)
        gate = PolicyGate(limits)

        # Simulate order flow
        order = {
            "symbol": "AAPL",
            "side": "long",
            "qty": 1,
            "price": 50,
        }

        allowed, reason = gate.check(
            symbol=order["symbol"],
            side=order["side"],
            price=order["price"],
            qty=order["qty"]
        )

        assert allowed
        assert reason == "ok"

    def test_order_blocked_when_exceeds_budget(self):
        """Test that order is blocked when exceeding budget."""
        from risk.policy_gate import PolicyGate, RiskLimits

        limits = RiskLimits(max_notional_per_order=75, max_daily_notional=1000)
        gate = PolicyGate(limits)

        # Order that exceeds per-order limit
        allowed, reason = gate.check(
            symbol="AAPL",
            side="long",
            price=100,
            qty=1  # $100 > $75 limit
        )

        assert not allowed
        assert "per_order" in reason


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

    def test_order_history_tracking(self, temp_state_dir):
        """Test order history tracking."""
        import json
        from datetime import datetime

        orders_file = temp_state_dir / "order_history.json"

        # Load existing (empty)
        with open(orders_file) as f:
            orders = json.load(f)
        assert orders == []

        # Add an order
        new_order = {
            "order_id": "test_001",
            "symbol": "MSFT",
            "side": "BUY",
            "qty": 5,
            "price": 350.00,
            "timestamp": datetime.now().isoformat(),
        }
        orders.append(new_order)

        with open(orders_file, 'w') as f:
            json.dump(orders, f)

        # Verify
        with open(orders_file) as f:
            saved = json.load(f)

        assert len(saved) == 1
        assert saved[0]["order_id"] == "test_001"
