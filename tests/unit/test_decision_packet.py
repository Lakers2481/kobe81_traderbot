"""
Unit tests for decision packet.
"""

import pytest
import json
import tempfile
from pathlib import Path

from explainability.decision_packet import (
    DecisionPacket,
    RiskGateResult,
    HistoricalAnalog,
    ExecutionPlan,
    build_decision_packet,
    load_latest_packet,
)


class TestExecutionPlan:
    """Tests for execution plan."""

    def test_basic_plan(self):
        plan = ExecutionPlan(
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            position_size=100,
            notional=15000.0,
            risk_amount=500.0,
            reward_amount=1000.0,
            reward_risk_ratio=2.0,
        )
        assert plan.entry_price == 150.0
        assert plan.reward_risk_ratio == 2.0


class TestRiskGateResult:
    """Tests for risk gate result."""

    def test_passed_gate(self):
        result = RiskGateResult(
            gate_name="policy_gate",
            passed=True,
            value=50.0,
            limit=75.0,
            message="Order within budget",
        )
        assert result.passed
        assert result.value < result.limit


class TestDecisionPacket:
    """Tests for decision packet."""

    def test_basic_packet(self):
        packet = DecisionPacket(
            run_id="test_001",
            timestamp="2025-12-27T10:00:00Z",
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
        )
        assert packet.symbol == "AAPL"
        assert packet.side == "buy"
        assert packet.packet_hash != ""

    def test_packet_hash_consistency(self):
        """Same inputs should produce same hash."""
        packet1 = DecisionPacket(
            run_id="test_001",
            timestamp="2025-12-27T10:00:00Z",
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
        )
        packet2 = DecisionPacket(
            run_id="test_001",
            timestamp="2025-12-27T10:00:00Z",
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
        )
        assert packet1.packet_hash == packet2.packet_hash

    def test_to_json(self):
        packet = DecisionPacket(
            run_id="test_001",
            timestamp="2025-12-27T10:00:00Z",
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
            strategy_reasons=["20-day breakout"],
        )
        json_str = packet.to_json()
        data = json.loads(json_str)
        assert data["symbol"] == "AAPL"
        assert "20-day breakout" in data["strategy_reasons"]

    def test_save_and_load(self):
        packet = DecisionPacket(
            run_id="test_001",
            timestamp="2025-12-27T10:00:00Z",
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
            strategy_reasons=["20-day breakout", "Volume confirmation"],
            feature_values={"rsi_14": 55.0, "atr_14": 3.50},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "packet.json"
            packet.save(path)

            loaded = DecisionPacket.load(path)
            assert loaded.run_id == packet.run_id
            assert loaded.symbol == packet.symbol
            assert loaded.strategy_reasons == packet.strategy_reasons
            assert loaded.feature_values == packet.feature_values

    def test_with_execution_plan(self):
        packet = DecisionPacket(
            run_id="test_001",
            timestamp="2025-12-27T10:00:00Z",
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
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
        )
        d = packet.to_dict()
        assert d["execution_plan"]["entry_price"] == 150.0

    def test_with_risk_gates(self):
        packet = DecisionPacket(
            run_id="test_001",
            timestamp="2025-12-27T10:00:00Z",
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
            risk_gate_results=[
                RiskGateResult("policy_gate", True, 50.0, 75.0, "OK"),
                RiskGateResult("liquidity_gate", True, 0.1, 0.5, "OK"),
            ],
        )
        d = packet.to_dict()
        assert len(d["risk_gate_results"]) == 2
        assert d["risk_gate_results"][0]["gate_name"] == "policy_gate"

    def test_field_presence(self):
        packet = DecisionPacket(
            run_id="test_001",
            timestamp="2025-12-27T10:00:00Z",
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
            ml_outputs={"probability": 0.65},
        )
        assert packet.is_field_present("ml_outputs")
        assert not packet.is_field_present("sentiment_score")

    def test_unknowns_tracking(self):
        packet = DecisionPacket(
            run_id="test_001",
            timestamp="2025-12-27T10:00:00Z",
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
            unknowns=["ml_outputs: No ML model used", "sentiment_score: Not available"],
        )
        assert len(packet.unknowns) == 2


class TestBuildDecisionPacket:
    """Tests for build_decision_packet function."""

    def test_minimal_build(self):
        packet = build_decision_packet(
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
        )
        assert packet.symbol == "AAPL"
        assert packet.run_id.startswith("totd_")
        # Should have unknowns since we didn't provide optional data
        assert len(packet.unknowns) > 0

    def test_full_build(self):
        packet = build_decision_packet(
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
            signal={"reason": "20-day breakout", "description": "Price above 20-day high"},
            ml_result={"probability": 0.65},
            risk_checks=[
                {"gate": "policy_gate", "passed": True, "value": 50, "limit": 75},
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
            feature_values={"rsi_14": 55.0},
            sentiment_score=0.3,
            sentiment_source="news_api",
        )
        assert packet.strategy_reasons == ["20-day breakout"]
        assert packet.signal_description == "Price above 20-day high"
        assert packet.ml_outputs["probability"] == 0.65
        assert len(packet.risk_gate_results) == 1
        assert packet.execution_plan is not None
        assert packet.sentiment_score == 0.3

    def test_historical_analogs(self):
        packet = build_decision_packet(
            symbol="AAPL",
            side="buy",
            strategy_name="IBS_RSI",
            historical_analogs=[
                {
                    "date": "2024-01-15",
                    "symbol": "AAPL",
                    "side": "buy",
                    "entry_price": 145.0,
                    "exit_price": 155.0,
                    "pnl_pct": 6.9,
                    "holding_days": 5,
                    "similarity_score": 0.85,
                },
            ],
        )
        assert len(packet.historical_analogs) == 1
        assert packet.historical_analogs[0].pnl_pct == 6.9

