"""
Unit tests for playbook generator.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from explainability.decision_packet import (
    DecisionPacket,
    ExecutionPlan,
    RiskGateResult,
    HistoricalAnalog,
)
from explainability.playbook_generator import (
    PlaybookGenerator,
    Playbook,
)


ET = ZoneInfo("America/New_York")


class TestPlaybook:
    """Tests for playbook dataclass."""

    def test_basic_playbook(self):
        playbook = Playbook(
            run_id="test_001",
            symbol="AAPL",
            timestamp=datetime.now(ET).isoformat(),
            executive_summary="Buy AAPL on breakout",
            full_playbook="Detailed playbook content...",
            risk_section="Stop loss at $145",
            confidence_section="65% win probability",
            checklist="- Verify entry\n- Check stops",
            generation_method="deterministic",
        )
        assert playbook.symbol == "AAPL"
        assert playbook.generation_method == "deterministic"


class TestPlaybookGenerator:
    """Tests for playbook generator."""

    def get_sample_packet(self) -> DecisionPacket:
        """Create a sample decision packet for testing."""
        return DecisionPacket(
            run_id="test_001",
            timestamp="2025-12-27T10:00:00Z",
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
            strategy_reasons=["20-day breakout", "Volume confirmation"],
            signal_description="Price closed above 20-day high with 1.5x average volume",
            feature_values={"rsi_14": 55.0, "atr_14": 3.50, "volume_ratio": 1.5},
            ml_outputs={"probability": 0.65, "calibrated": True},
            sentiment_score=0.3,
            sentiment_source="news_api",
            risk_gate_results=[
                RiskGateResult("policy_gate", True, 50.0, 75.0, "OK"),
                RiskGateResult("liquidity_gate", True, 0.1, 0.5, "OK"),
            ],
            execution_plan=ExecutionPlan(
                entry_price=150.0,
                stop_loss=145.0,
                take_profit=160.0,
                position_size=100,
                notional=15000.0,
                risk_amount=500.0,
                reward_amount=1000.0,
                reward_risk_ratio=2.0,
            ),
            historical_analogs=[
                HistoricalAnalog(
                    date="2024-01-15",
                    symbol="AAPL",
                    side="buy",
                    entry_price=145.0,
                    exit_price=155.0,
                    pnl_pct=6.9,
                    holding_days=5,
                    similarity_score=0.85,
                ),
            ],
        )

    def test_generate_deterministic(self):
        """Test deterministic playbook generation."""
        generator = PlaybookGenerator(use_claude=False)
        packet = self.get_sample_packet()

        playbook = generator.generate_from_packet(packet)

        assert playbook.symbol == "AAPL"
        assert playbook.generation_method == "deterministic"

    def test_executive_summary_content(self):
        """Test executive summary generation."""
        generator = PlaybookGenerator(use_claude=False)
        packet = self.get_sample_packet()

        playbook = generator.generate_from_packet(packet)

        assert "AAPL" in playbook.executive_summary
        assert "buy" in playbook.executive_summary.lower()

    def test_risk_section_content(self):
        """Test risk section generation."""
        generator = PlaybookGenerator(use_claude=False)
        packet = self.get_sample_packet()

        playbook = generator.generate_from_packet(packet)

        # Should contain stop loss info
        assert "145" in playbook.risk_section or "stop" in playbook.risk_section.lower()

    def test_execution_plan_in_playbook(self):
        """Test execution plan is included."""
        generator = PlaybookGenerator(use_claude=False)
        packet = self.get_sample_packet()

        playbook = generator.generate_from_packet(packet)

        # Should reference entry price somewhere
        full_content = playbook.full_playbook + playbook.executive_summary
        assert "150" in full_content or "entry" in full_content.lower()

    def test_unknowns_tracked(self):
        """Test that missing fields are tracked as unknowns."""
        generator = PlaybookGenerator(use_claude=False)
        packet = DecisionPacket(
            run_id="test_001",
            timestamp="2025-12-27T10:00:00Z",
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
            # Missing many optional fields
            unknowns=["ml_outputs: Not available", "sentiment_score: Not available"],
        )

        playbook = generator.generate_from_packet(packet)

        # Should still generate a valid playbook
        assert playbook.symbol == "AAPL"
        assert playbook.generation_method == "deterministic"

    def test_no_fabrication(self):
        """Test that generator doesn't fabricate data not in packet."""
        generator = PlaybookGenerator(use_claude=False)
        packet = DecisionPacket(
            run_id="test_001",
            timestamp="2025-12-27T10:00:00Z",
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
            # No ML outputs provided
        )

        playbook = generator.generate_from_packet(packet)

        # If probability is mentioned, it should acknowledge it's not available
        confidence_lower = playbook.confidence_section.lower()
        if "probability" in confidence_lower:
            assert "not available" in confidence_lower or "unknown" in confidence_lower or "n/a" in confidence_lower


class TestPlaybookWithHistoricalAnalogs:
    """Tests for playbook with historical analogs."""

    def test_analogs_included(self):
        generator = PlaybookGenerator(use_claude=False)
        packet = DecisionPacket(
            run_id="test_001",
            timestamp="2025-12-27T10:00:00Z",
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
            historical_analogs=[
                HistoricalAnalog(
                    date="2024-01-15",
                    symbol="AAPL",
                    side="buy",
                    entry_price=145.0,
                    exit_price=155.0,
                    pnl_pct=6.9,
                    holding_days=5,
                    similarity_score=0.85,
                ),
                HistoricalAnalog(
                    date="2024-03-20",
                    symbol="AAPL",
                    side="buy",
                    entry_price=170.0,
                    exit_price=165.0,
                    pnl_pct=-2.9,
                    holding_days=3,
                    similarity_score=0.78,
                ),
            ],
        )

        playbook = generator.generate_from_packet(packet)

        # Should reference historical analogs in full playbook
        assert playbook.full_playbook is not None
        assert len(playbook.full_playbook) > 0


class TestPlaybookChecklist:
    """Tests for playbook checklist."""

    def test_checklist_generation(self):
        generator = PlaybookGenerator(use_claude=False)
        packet = DecisionPacket(
            run_id="test_001",
            timestamp="2025-12-27T10:00:00Z",
            symbol="MSFT",
            side="buy",
            strategy_name="IBS_RSI",
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

        playbook = generator.generate_from_packet(packet)

        assert playbook.checklist is not None
        assert len(playbook.checklist) > 0
