"""
Unit tests for portfolio risk gate.
"""

from pathlib import Path
import tempfile

from risk.portfolio_risk import (
    PortfolioRiskGate,
    PortfolioRiskLimits,
    PortfolioRiskStatus,
    PortfolioState,
    PortfolioPosition,
    SectorMapper,
    CorrelationAnalyzer,
)

import pandas as pd
import numpy as np


class TestPortfolioRiskLimits:
    """Tests for risk limits configuration."""

    def test_default_limits(self):
        limits = PortfolioRiskLimits()
        assert limits.max_gross_exposure_pct == 100.0
        assert limits.max_single_name_pct == 10.0
        assert limits.max_sector_pct == 30.0
        assert limits.max_simultaneous_positions == 20

    def test_custom_limits(self):
        limits = PortfolioRiskLimits(
            max_gross_exposure_pct=50.0,
            max_single_name_pct=5.0,
        )
        assert limits.max_gross_exposure_pct == 50.0
        assert limits.max_single_name_pct == 5.0


class TestPortfolioState:
    """Tests for portfolio state."""

    def test_empty_portfolio(self):
        state = PortfolioState(nav=100000, cash=100000)
        assert state.long_exposure == 0
        assert state.short_exposure == 0
        assert state.gross_exposure == 0
        assert state.position_count == 0

    def test_with_positions(self):
        positions = [
            PortfolioPosition("AAPL", 100, 150.0, 155.0, "long"),
            PortfolioPosition("MSFT", 50, 300.0, 310.0, "long"),
        ]
        state = PortfolioState(nav=100000, cash=50000, positions=positions)

        # AAPL: 100 * 155 = 15500
        # MSFT: 50 * 310 = 15500
        assert state.long_exposure == 31000
        assert state.position_count == 2


class TestSectorMapper:
    """Tests for sector mapping."""

    def test_unknown_sector(self):
        mapper = SectorMapper()
        assert mapper.get_sector("UNKNOWN_SYMBOL") == "Unknown"

    def test_add_mapping(self):
        mapper = SectorMapper()
        mapper.add_mapping("AAPL", "Technology")
        assert mapper.get_sector("AAPL") == "Technology"
        assert mapper.get_sector("aapl") == "Technology"  # Case insensitive

    def test_load_from_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("symbol,sector\n")
            f.write("AAPL,Technology\n")
            f.write("JPM,Financials\n")
            f.flush()

            mapper = SectorMapper(Path(f.name))
            assert mapper.get_sector("AAPL") == "Technology"
            assert mapper.get_sector("JPM") == "Financials"


class TestCorrelationAnalyzer:
    """Tests for correlation analysis."""

    def test_compute_correlation(self):
        analyzer = CorrelationAnalyzer(window_days=20, threshold=0.70)

        # Create correlated price series
        np.random.seed(42)
        base = pd.Series(100 + np.cumsum(np.random.randn(30) * 2))
        correlated = base + np.random.randn(30) * 0.5  # Highly correlated
        uncorrelated = pd.Series(100 + np.cumsum(np.random.randn(30) * 2))

        analyzer.set_price_history("AAPL", base)
        analyzer.set_price_history("MSFT", correlated)
        analyzer.set_price_history("XOM", uncorrelated)

        # AAPL and MSFT should be correlated
        corr = analyzer.compute_correlation("AAPL", "MSFT")
        assert corr is not None
        assert corr > 0.5

    def test_identify_clusters(self):
        analyzer = CorrelationAnalyzer(window_days=20, threshold=0.70)

        # Create perfectly correlated series
        np.random.seed(42)
        base = pd.Series(np.arange(30, dtype=float))
        analyzer.set_price_history("A", base)
        analyzer.set_price_history("B", base + 0.1)
        analyzer.set_price_history("C", base * 2)

        clusters = analyzer.identify_correlation_clusters(["A", "B", "C"])
        # All should be in one cluster (perfectly correlated)
        assert len(clusters) >= 1


class TestPortfolioRiskGate:
    """Tests for portfolio risk gate."""

    def test_disabled_gate(self):
        gate = PortfolioRiskGate(enabled=False)
        state = PortfolioState(nav=100000, cash=100000)

        result = gate.check("AAPL", "buy", 150.0, 100, state)
        assert result.approved
        assert result.status == PortfolioRiskStatus.APPROVED

    def test_gross_exposure_limit(self):
        limits = PortfolioRiskLimits(max_gross_exposure_pct=50.0)
        gate = PortfolioRiskGate(limits=limits)

        # Already at 40% exposure
        positions = [
            PortfolioPosition("MSFT", 100, 300.0, 400.0, "long"),
        ]
        state = PortfolioState(nav=100000, cash=60000, positions=positions)

        # Try to add 15% more (would exceed 50% limit)
        result = gate.check("AAPL", "buy", 150.0, 100, state)  # 15000 notional
        assert not result.approved
        assert "Gross exposure" in result.rejection_reasons[0]

    def test_single_name_limit(self):
        limits = PortfolioRiskLimits(max_single_name_pct=10.0)
        gate = PortfolioRiskGate(limits=limits)

        state = PortfolioState(nav=100000, cash=100000)

        # Try to buy 15% in single name
        result = gate.check("AAPL", "buy", 150.0, 100, state)  # 15000 = 15%
        assert not result.approved
        assert "Single-name" in result.rejection_reasons[0]

    def test_sector_limit(self):
        limits = PortfolioRiskLimits(max_sector_pct=30.0)
        gate = PortfolioRiskGate(limits=limits)
        gate.sector_mapper.add_mapping("AAPL", "Technology")
        gate.sector_mapper.add_mapping("MSFT", "Technology")

        # Already have 25% in Technology
        positions = [
            PortfolioPosition("MSFT", 100, 250.0, 250.0, "long", "Technology"),
        ]
        state = PortfolioState(nav=100000, cash=75000, positions=positions)

        # Try to add 10% more Tech (would exceed 30%)
        result = gate.check("AAPL", "buy", 100.0, 100, state)
        assert not result.approved
        assert "Sector" in result.rejection_reasons[0]

    def test_position_count_limit(self):
        limits = PortfolioRiskLimits(max_simultaneous_positions=2)
        gate = PortfolioRiskGate(limits=limits)

        # Already have 2 positions
        positions = [
            PortfolioPosition("AAPL", 10, 150.0, 150.0, "long"),
            PortfolioPosition("MSFT", 10, 300.0, 300.0, "long"),
        ]
        state = PortfolioState(nav=100000, cash=90000, positions=positions)

        # Try to add a third
        result = gate.check("GOOG", "buy", 100.0, 10, state)
        assert not result.approved
        assert "Position count" in result.rejection_reasons[0]

    def test_approved_trade(self):
        limits = PortfolioRiskLimits()
        gate = PortfolioRiskGate(limits=limits)

        state = PortfolioState(nav=100000, cash=100000)

        # Small trade that passes all limits
        result = gate.check("AAPL", "buy", 150.0, 10, state)  # 1500 = 1.5%
        assert result.approved
        assert result.status == PortfolioRiskStatus.APPROVED

    def test_warning_status(self):
        # Disable sector limits to test only gross exposure warning
        limits = PortfolioRiskLimits(
            max_gross_exposure_pct=100.0,
            max_sector_pct=100.0,  # Disable sector limit
        )
        gate = PortfolioRiskGate(limits=limits)

        # Already at 85% exposure (close to limit)
        positions = [
            PortfolioPosition("MSFT", 100, 850.0, 850.0, "long"),
        ]
        state = PortfolioState(nav=100000, cash=15000, positions=positions)

        # Add 6% more (total 91%, above 90% warning threshold)
        result = gate.check("AAPL", "buy", 60.0, 100, state)
        assert result.approved  # Still allowed
        assert result.status == PortfolioRiskStatus.WARNING
        assert len(result.warnings) > 0

    def test_from_config(self):
        config = {
            "enabled": True,
            "limits": {
                "max_gross_exposure_pct": 80.0,
                "max_single_name_pct": 8.0,
            },
        }
        gate = PortfolioRiskGate.from_config(config)
        assert gate.enabled
        assert gate.limits.max_gross_exposure_pct == 80.0
        assert gate.limits.max_single_name_pct == 8.0
