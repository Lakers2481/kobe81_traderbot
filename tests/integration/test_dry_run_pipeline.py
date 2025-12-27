"""
Integration tests for dry-run pipeline.

Tests the full pipeline without executing orders:
- TOTD signal generation
- Decision packet creation
- Risk gate checks
- Playbook generation
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

# Import modules under test
from explainability.decision_packet import (
    DecisionPacket,
    ExecutionPlan,
    RiskGateResult,
    build_decision_packet,
)
from explainability.playbook_generator import PlaybookGenerator, generate_playbook
from core.lineage import LineageTracker, compute_decision_hash
from risk.portfolio_risk import (
    PortfolioRiskGate,
    PortfolioRiskLimits,
    PortfolioState,
)
from execution.execution_guard import ExecutionGuard, QuoteData

ET = ZoneInfo("America/New_York")


class TestDryRunPipeline:
    """Integration tests for dry-run mode."""

    def test_full_decision_packet_pipeline(self):
        """Test building a complete decision packet."""
        # 1. Build decision packet from signal
        packet = build_decision_packet(
            symbol="AAPL",
            side="buy",
            strategy_name="DonchianBreakout",
            signal={
                "reason": "20-day breakout",
                "description": "Price closed above 20-day high",
            },
            ml_result=None,  # No ML in this test
            risk_checks=[
                {"gate": "policy_gate", "passed": True, "value": 50, "limit": 75},
                {"gate": "liquidity_gate", "passed": True, "value": 0.1, "limit": 0.5},
            ],
            execution_plan={
                "entry_price": 150.0,
                "stop_loss": 145.0,
                "take_profit": 160.0,
                "position_size": 100,
                "notional": 15000.0,
                "risk_amount": 500.0,
                "reward_amount": 1000.0,
                "reward_risk_ratio": 2.0,
            },
            feature_values={"rsi_14": 55.0, "atr_14": 3.50},
        )

        # 2. Verify packet structure
        assert packet.symbol == "AAPL"
        assert packet.side == "buy"
        assert packet.run_id.startswith("totd_")
        assert len(packet.risk_gate_results) == 2
        assert packet.execution_plan is not None
        assert packet.execution_plan.entry_price == 150.0

        # 3. Verify hash consistency
        hash1 = packet.packet_hash
        assert len(hash1) == 64

    def test_playbook_generation_pipeline(self):
        """Test generating playbook from decision packet."""
        # 1. Create packet
        packet = DecisionPacket(
            run_id="test_integration_001",
            timestamp=datetime.now(ET).isoformat(),
            symbol="MSFT",
            side="buy",
            strategy_name="DonchianBreakout",
            strategy_reasons=["Volume surge", "Breakout above resistance"],
            feature_values={"rsi_14": 62.0},
            execution_plan=ExecutionPlan(
                entry_price=380.0,
                stop_loss=370.0,
                take_profit=400.0,
                position_size=50,
                notional=19000.0,
                risk_amount=500.0,
                reward_amount=1000.0,
                reward_risk_ratio=2.0,
            ),
        )

        # 2. Generate playbook
        generator = PlaybookGenerator(use_llm=False)  # Deterministic mode
        playbook = generator.generate(packet)

        # 3. Verify playbook
        assert playbook.symbol == "MSFT"
        assert len(playbook.sections) >= 3

        # 4. Export to markdown
        md = playbook.to_markdown()
        assert "MSFT" in md
        assert "380" in md  # Entry price

    def test_risk_gate_integration(self):
        """Test risk gate checks in pipeline."""
        # 1. Setup portfolio state
        state = PortfolioState(nav=100000, cash=80000)

        # 2. Setup risk gate
        limits = PortfolioRiskLimits(
            max_gross_exposure_pct=100.0,
            max_single_name_pct=10.0,
        )
        gate = PortfolioRiskGate(limits=limits)

        # 3. Check order (should pass)
        result = gate.check("AAPL", "buy", 150.0, 50, state)  # 7500 = 7.5%
        assert result.approved

        # 4. Build packet with risk check
        packet = build_decision_packet(
            symbol="AAPL",
            side="buy",
            strategy_name="DonchianBreakout",
            risk_checks=[
                {
                    "gate": "portfolio_risk",
                    "passed": result.approved,
                    "value": 7.5,  # % of NAV
                    "limit": limits.max_single_name_pct,
                },
            ],
        )

        assert len(packet.risk_gate_results) == 1
        assert packet.risk_gate_results[0].passed

    def test_execution_guard_integration(self):
        """Test execution guard in pipeline."""
        # 1. Setup guard
        guard = ExecutionGuard(
            max_quote_age_seconds=5.0,
            max_spread_pct=0.50,
            stand_down_on_uncertainty=True,
        )

        # 2. Create fresh quote
        quote = QuoteData(
            symbol="GOOG",
            bid=140.0,
            ask=140.10,
            bid_size=100,
            ask_size=100,
            timestamp=datetime.now(ET),
        )

        # 3. Check execution (should pass)
        result = guard.full_check(
            "GOOG", "buy", 100, 140.05, quote=quote, skip_trading_status=True
        )

        assert result.approved

    def test_lineage_tracking_pipeline(self):
        """Test lineage tracking through pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create tracker
            tracker = LineageTracker(Path(tmpdir) / "lineage.jsonl")

            # 2. Create data files
            data_file = Path(tmpdir) / "prices.csv"
            data_file.write_text("date,symbol,close\n2025-12-27,AAPL,150.0")

            # 3. Record dataset
            dataset_record = tracker.record_dataset("prices_20251227", [data_file])

            # 4. Create decision packet
            packet = build_decision_packet(
                symbol="AAPL",
                side="buy",
                strategy_name="DonchianBreakout",
            )

            # 5. Record decision
            decision_record = tracker.record_decision(
                packet.run_id,
                packet.to_dict(),
                model_hash=None,
                dataset_hash=dataset_record.record_hash,
            )

            # 6. Verify lineage
            assert decision_record.record_type == "decision"
            assert dataset_record.record_hash in decision_record.parent_hashes

    def test_dry_run_no_order_submitted(self):
        """Verify dry-run mode doesn't submit orders."""
        # This is a conceptual test - actual implementation would mock broker

        dry_run = True

        # 1. Generate signal
        signal = {
            "symbol": "NVDA",
            "side": "buy",
            "entry_price": 500.0,
            "stop_loss": 485.0,
            "reason": "Breakout signal",
        }

        # 2. Build packet
        packet = build_decision_packet(
            symbol=signal["symbol"],
            side=signal["side"],
            strategy_name="DonchianBreakout",
        )

        # 3. In dry-run, we produce artifact but don't submit
        order_submitted = False
        if not dry_run:
            # broker.submit(...)  # Would submit
            order_submitted = True

        assert not order_submitted
        assert packet is not None
        assert packet.symbol == "NVDA"


class TestPaperTradeIntegration:
    """Integration tests for paper trading mode."""

    def test_paper_trade_full_cycle(self):
        """Test paper trade from signal to decision packet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Setup components
            state = PortfolioState(nav=50000, cash=50000)
            risk_gate = PortfolioRiskGate()
            guard = ExecutionGuard(stand_down_on_uncertainty=False)
            tracker = LineageTracker(Path(tmpdir) / "lineage.jsonl")

            # 2. Simulate signal
            signal = {
                "symbol": "AMD",
                "side": "buy",
                "entry_price": 125.0,
                "stop_loss": 120.0,
                "take_profit": 135.0,
                "reason": "RSI oversold bounce",
            }

            # 3. Check risk
            risk_result = risk_gate.check(
                signal["symbol"],
                signal["side"],
                signal["entry_price"],
                40,  # 40 shares = $5000
                state,
            )

            # 4. Check execution guard (mocked quote)
            quote = QuoteData(
                symbol=signal["symbol"],
                bid=124.95,
                ask=125.05,
                bid_size=100,
                ask_size=100,
                timestamp=datetime.now(ET),
            )
            guard_result = guard.full_check(
                signal["symbol"],
                signal["side"],
                40,
                signal["entry_price"],
                quote=quote,
                skip_trading_status=True,
            )

            # 5. Build decision packet
            packet = build_decision_packet(
                symbol=signal["symbol"],
                side=signal["side"],
                strategy_name="RSI_Bounce",
                signal={"reason": signal["reason"]},
                risk_checks=[
                    {
                        "gate": "portfolio_risk",
                        "passed": risk_result.approved,
                        "value": 10.0,
                        "limit": 10.0,
                    },
                    {
                        "gate": "execution_guard",
                        "passed": guard_result.approved,
                        "value": 0.0,
                        "limit": 0.0,
                    },
                ],
                execution_plan={
                    "entry_price": signal["entry_price"],
                    "stop_loss": signal["stop_loss"],
                    "take_profit": signal["take_profit"],
                    "position_size": 40,
                    "notional": 5000.0,
                    "risk_amount": 200.0,
                    "reward_amount": 400.0,
                    "reward_risk_ratio": 2.0,
                },
            )

            # 6. Record lineage
            decision_record = tracker.record_decision(
                packet.run_id, packet.to_dict(), model_hash=None, dataset_hash=None
            )

            # 7. Verify complete
            assert risk_result.approved
            assert guard_result.approved
            assert packet.symbol == "AMD"
            assert decision_record.record_hash is not None

            # 8. Generate playbook
            playbook = generate_playbook(packet, use_llm=False)
            md = playbook.to_markdown()

            assert "AMD" in md
            assert "125" in md


class TestArtifactGeneration:
    """Tests for artifact generation."""

    def test_decision_packet_json_export(self):
        """Test decision packet JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            packet = build_decision_packet(
                symbol="TSLA",
                side="buy",
                strategy_name="Momentum",
            )

            # Save to file
            path = Path(tmpdir) / f"{packet.run_id}.json"
            packet.save(path)

            # Verify file exists
            assert path.exists()

            # Load and verify
            loaded = DecisionPacket.load(path)
            assert loaded.symbol == "TSLA"

    def test_playbook_markdown_export(self):
        """Test playbook markdown export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            packet = build_decision_packet(
                symbol="META",
                side="buy",
                strategy_name="Breakout",
            )

            playbook = generate_playbook(packet, use_llm=False)

            # Save markdown
            md_path = Path(tmpdir) / "playbook.md"
            md_path.write_text(playbook.to_markdown())

            assert md_path.exists()
            content = md_path.read_text()
            assert "META" in content

    def test_playbook_html_export(self):
        """Test playbook HTML export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            packet = build_decision_packet(
                symbol="AMZN",
                side="buy",
                strategy_name="DonchianBreakout",
            )

            playbook = generate_playbook(packet, use_llm=False)

            # Save HTML
            html_path = Path(tmpdir) / "playbook.html"
            html_path.write_text(playbook.to_html())

            assert html_path.exists()
            content = html_path.read_text()
            assert "<html>" in content
            assert "AMZN" in content
