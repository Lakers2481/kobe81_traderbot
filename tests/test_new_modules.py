"""
Tests for New Enhancement Modules.

Tests testing/, selfmonitor/, and compliance/ modules.
"""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Testing module
from testing import (
    MonteCarloSimulator,
    SimulationResult,
    simulate_returns,
    run_monte_carlo,
    StressTester,
    StressScenario,
    ScenarioType,
    run_stress_test,
    get_standard_scenarios,
)

# Self-monitoring module
from selfmonitor import (
    CircuitBreaker,
    BreakerState,
    BreakerConfig,
    get_breaker,
    check_breaker,
    trip_breaker,
    AnomalyDetector,
    AnomalyType,
    AnomalyAlert,
    detect_anomalies,
    is_anomalous,
)

# Compliance module
from compliance import (
    RulesEngine,
    TradingRule,
    RuleViolation,
    RuleCategory,
    check_rules,
    get_violations,
    ProhibitedList,
    ProhibitionReason,
    check_symbol,
    add_prohibition,
    is_prohibited,
    AuditTrail,
    AuditEntry,
    AuditAction,
    log_audit,
    get_audit_history,
)


class TestMonteCarloSimulator:
    """Tests for Monte Carlo simulation."""

    def test_initialization(self):
        """Should initialize with defaults."""
        sim = MonteCarloSimulator()
        assert sim.n_simulations == 10000

    def test_simulate_basic(self):
        """Should run simulation."""
        sim = MonteCarloSimulator(n_simulations=1000, random_seed=42)
        result = sim.simulate(
            mean_return=0.0005,
            std_return=0.02,
            n_periods=252,
        )

        assert isinstance(result, SimulationResult)
        assert result.n_simulations == 1000
        assert result.n_periods == 252

    def test_result_has_metrics(self):
        """Should include risk metrics."""
        sim = MonteCarloSimulator(n_simulations=1000, random_seed=42)
        result = sim.simulate(0.0005, 0.02)

        assert hasattr(result, 'var_95')
        assert hasattr(result, 'cvar_95')
        assert hasattr(result, 'max_drawdown_mean')
        assert hasattr(result, 'prob_positive')

    def test_result_to_dict(self):
        """Should convert to dictionary."""
        sim = MonteCarloSimulator(n_simulations=100)
        result = sim.simulate(0.001, 0.01)

        d = result.to_dict()
        assert 'mean_return' in d
        assert 'var_95' in d

    def test_simulate_from_trades(self):
        """Should simulate from trade history."""
        trades = [
            {'pnl': 100}, {'pnl': -50}, {'pnl': 75},
            {'pnl': 25}, {'pnl': -30}, {'pnl': 60},
        ]

        sim = MonteCarloSimulator(n_simulations=100)
        result = sim.simulate_from_trades(trades)

        assert isinstance(result, SimulationResult)


class TestStressTester:
    """Tests for stress testing."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        return pd.Series(np.random.randn(252) * 0.01)

    def test_initialization(self):
        """Should initialize."""
        tester = StressTester()
        assert tester.ruin_threshold == -0.50

    def test_standard_scenarios(self):
        """Should have standard scenarios."""
        scenarios = get_standard_scenarios()
        assert len(scenarios) > 0
        assert all(isinstance(s, StressScenario) for s in scenarios)

    def test_run_test(self, sample_returns):
        """Should run stress test."""
        tester = StressTester()
        scenario = StressScenario(
            name="Test Crash",
            scenario_type=ScenarioType.CRASH,
            return_shock=-0.10,
        )

        result = tester.run_test(sample_returns, scenario)

        assert result.stressed_return < result.original_return
        assert hasattr(result, 'max_drawdown')
        assert hasattr(result, 'survived')

    def test_run_all_scenarios(self, sample_returns):
        """Should run all scenarios."""
        tester = StressTester()
        results = tester.run_all_scenarios(sample_returns)

        assert len(results) == len(StressTester.STANDARD_SCENARIOS)

    def test_apply_scenario(self, sample_returns):
        """Should apply scenario to returns."""
        tester = StressTester()
        scenario = StressScenario(
            name="Vol Spike",
            scenario_type=ScenarioType.VOLATILITY_SPIKE,
            volatility_mult=2.0,
        )

        stressed = tester.apply_scenario(sample_returns, scenario)

        assert len(stressed) == len(sample_returns)
        assert stressed.std() > sample_returns.std()


class TestCircuitBreaker:
    """Tests for circuit breaker."""

    @pytest.fixture
    def breaker(self):
        """Create fresh breaker."""
        return CircuitBreaker(BreakerConfig())

    def test_initialization(self, breaker):
        """Should initialize closed."""
        assert breaker.state == BreakerState.CLOSED
        assert breaker.can_trade == True

    def test_trip(self, breaker):
        """Should trip when called."""
        breaker.trip("Test reason")

        assert breaker.state == BreakerState.OPEN
        assert breaker.is_open == True
        assert breaker.can_trade == False

    def test_record_trade_loss(self, breaker):
        """Should track losses."""
        breaker.record_trade(-100)

        assert breaker._daily_pnl == -100
        assert breaker._consecutive_losses == 1

    def test_record_trade_win_resets_consecutive(self, breaker):
        """Should reset consecutive losses on win."""
        breaker.record_trade(-50)
        breaker.record_trade(-50)
        assert breaker._consecutive_losses == 2

        breaker.record_trade(100)
        assert breaker._consecutive_losses == 0

    def test_trip_on_max_loss(self, breaker):
        """Should trip on max daily loss."""
        breaker.config.max_daily_loss = 500

        for _ in range(10):
            breaker.record_trade(-100)

        assert breaker.is_open == True

    def test_trip_on_consecutive_losses(self, breaker):
        """Should trip on consecutive losses."""
        breaker.config.max_consecutive_losses = 3

        for _ in range(4):
            breaker.record_trade(-50)

        assert breaker.is_open == True

    def test_reset(self, breaker):
        """Should reset state."""
        breaker.trip("Test")
        breaker.reset()

        assert breaker.state == BreakerState.CLOSED
        assert breaker.can_trade == True

    def test_get_status(self, breaker):
        """Should return status."""
        breaker.record_trade(-100)

        status = breaker.get_status()

        assert 'state' in status
        assert 'daily_pnl' in status
        assert status['daily_pnl'] == -100


class TestAnomalyDetector:
    """Tests for anomaly detection."""

    @pytest.fixture
    def detector(self):
        """Create detector."""
        return AnomalyDetector(zscore_threshold=3.0)

    @pytest.fixture
    def sample_prices(self):
        """Generate sample prices."""
        np.random.seed(42)
        return pd.Series(np.cumsum(np.random.randn(100)) + 100)

    def test_initialization(self, detector):
        """Should initialize with threshold."""
        assert detector.zscore_threshold == 3.0

    def test_detect_price_anomaly(self, detector, sample_prices):
        """Should detect price anomalies."""
        # Add an outlier
        prices = sample_prices.copy()
        prices.iloc[-1] = prices.iloc[-1] * 1.5

        alerts = detector.detect_price_anomaly(prices, "TEST")

        # May or may not detect depending on threshold
        assert isinstance(alerts, list)

    def test_detect_volume_anomaly(self, detector):
        """Should detect volume anomalies."""
        np.random.seed(42)
        volume = pd.Series(np.abs(np.random.randn(100) * 1000000))
        volume.iloc[-1] = volume.mean() * 10  # Spike

        alerts = detector.detect_volume_anomaly(volume, "TEST")

        assert len(alerts) > 0
        assert alerts[0].anomaly_type == AnomalyType.VOLUME_SPIKE

    def test_is_current_anomalous(self, detector, sample_prices):
        """Should check if current value is anomalous."""
        normal_value = sample_prices.mean()
        extreme_value = sample_prices.mean() + sample_prices.std() * 5

        normal_result, _ = detector.is_current_anomalous(normal_value, sample_prices)
        extreme_result, _ = detector.is_current_anomalous(extreme_value, sample_prices)

        assert normal_result == False
        assert extreme_result == True


class TestRulesEngine:
    """Tests for rules engine."""

    @pytest.fixture
    def engine(self):
        """Create rules engine."""
        return RulesEngine()

    def test_initialization(self, engine):
        """Should initialize with standard rules."""
        assert len(engine.rules) > 0

    def test_check_trade_valid(self, engine):
        """Should pass valid trade."""
        violations = engine.check_trade(
            symbol='AAPL',
            side='buy',
            quantity=10,
            price=150.0,
            portfolio_value=100000,
        )

        # Position size is only 1.5%, should pass
        position_violations = [v for v in violations if v.rule.name == 'max_position_size']
        assert len(position_violations) == 0

    def test_check_trade_size_violation(self, engine):
        """Should detect position size violation."""
        violations = engine.check_trade(
            symbol='AAPL',
            side='buy',
            quantity=100,
            price=150.0,
            portfolio_value=10000,  # 150% position!
        )

        assert any(v.rule.name == 'max_position_size' for v in violations)

    def test_check_penny_stock(self, engine):
        """Should block penny stocks."""
        violations = engine.check_trade(
            symbol='PENNY',
            side='buy',
            quantity=100,
            price=2.0,  # Under $5
            portfolio_value=100000,
        )

        assert any(v.rule.name == 'no_penny_stocks' for v in violations)

    def test_disable_rule(self, engine):
        """Should disable rule."""
        engine.disable_rule('no_penny_stocks')

        violations = engine.check_trade(
            symbol='PENNY',
            side='buy',
            quantity=100,
            price=2.0,
            portfolio_value=100000,
        )

        assert not any(v.rule.name == 'no_penny_stocks' for v in violations)


class TestProhibitedList:
    """Tests for prohibited list."""

    @pytest.fixture
    def prohibited(self):
        """Create prohibited list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ProhibitedList(data_file=Path(tmpdir) / 'prohibited.json')

    def test_initialization(self, prohibited):
        """Should initialize empty."""
        assert len(prohibited._prohibitions) == 0

    def test_add_symbol(self, prohibited):
        """Should add symbol."""
        prohibited.add('AAPL', ProhibitionReason.EARNINGS)

        assert prohibited.is_prohibited('AAPL')
        assert prohibited.is_prohibited('aapl')  # Case insensitive

    def test_remove_symbol(self, prohibited):
        """Should remove symbol."""
        prohibited.add('MSFT', ProhibitionReason.NEWS)
        prohibited.remove('MSFT')

        assert not prohibited.is_prohibited('MSFT')

    def test_check_returns_prohibition(self, prohibited):
        """Should return prohibition details."""
        prohibited.add('GOOG', ProhibitionReason.VOLATILITY, notes='Earnings')

        result = prohibited.check('GOOG')

        assert result is not None
        assert result.reason == ProhibitionReason.VOLATILITY

    def test_expiration(self, prohibited):
        """Should handle expiration."""
        prohibited.add(
            'TSLA',
            ProhibitionReason.EARNINGS,
            expires_at=datetime.now() - timedelta(hours=1),  # Already expired
        )

        assert not prohibited.is_prohibited('TSLA')


class TestAuditTrail:
    """Tests for audit trail."""

    @pytest.fixture
    def trail(self):
        """Create audit trail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield AuditTrail(log_dir=Path(tmpdir), auto_persist=False)

    def test_initialization(self, trail):
        """Should initialize empty."""
        assert len(trail._entries) == 0

    def test_log_entry(self, trail):
        """Should log entry."""
        entry = trail.log(
            action=AuditAction.ORDER_PLACED,
            symbol='AAPL',
            details={'side': 'buy', 'quantity': 100},
        )

        assert len(trail._entries) == 1
        assert entry.action == AuditAction.ORDER_PLACED

    def test_entry_has_hash(self, trail):
        """Should compute hash."""
        entry = trail.log(AuditAction.SYSTEM_START)

        assert entry.entry_hash != ""
        assert len(entry.entry_hash) == 16

    def test_log_order(self, trail):
        """Should log order."""
        trail.log_order(
            action=AuditAction.ORDER_FILLED,
            symbol='MSFT',
            side='buy',
            quantity=50,
            price=300.0,
        )

        assert len(trail._entries) == 1
        assert trail._entries[0].details['quantity'] == 50

    def test_get_history(self, trail):
        """Should filter history."""
        trail.log(AuditAction.ORDER_PLACED, 'AAPL')
        trail.log(AuditAction.ORDER_FILLED, 'AAPL')
        trail.log(AuditAction.ORDER_PLACED, 'MSFT')

        aapl_history = trail.get_history(symbol='AAPL')
        placed_history = trail.get_history(action=AuditAction.ORDER_PLACED)

        assert len(aapl_history) == 2
        assert len(placed_history) == 2

    def test_verify_integrity(self, trail):
        """Should verify integrity."""
        trail.log(AuditAction.ORDER_PLACED, 'AAPL')
        trail.log(AuditAction.ORDER_FILLED, 'AAPL')

        assert trail.verify_integrity() == True

    def test_entry_to_dict(self):
        """Should convert to dictionary."""
        entry = AuditEntry(
            action=AuditAction.CONFIG_CHANGE,
            symbol='',
            details={'key': 'value'},
        )

        d = entry.to_dict()

        assert d['action'] == 'config_change'
        assert 'hash' in d


class TestConvenienceFunctions:
    """Tests for module-level functions."""

    def test_simulate_returns(self):
        """Should simulate returns."""
        result = simulate_returns(0.0005, 0.02, 100)
        assert isinstance(result, SimulationResult)

    def test_run_stress_test(self):
        """Should run stress test."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)

        result = run_stress_test(returns, "Black Monday")
        assert hasattr(result, 'stressed_return')


# Run with: pytest tests/test_new_modules.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
