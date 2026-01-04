"""
Decision Reproducibility Packets

Captures all inputs to a trading decision for:
- Post-trade analysis and debugging
- Regulatory compliance (full audit trail)
- Reproducible backtesting
- Machine learning training data

Based on: Codex & Gemini reliability recommendations (2026-01-04)

Every trade decision creates a packet containing:
- Market data snapshot at decision time
- All indicator values
- Strategy parameters used
- ML model inputs and outputs
- Risk gate results
- Final decision and reasoning

Usage:
    from core.decision_packet import DecisionPacket, create_decision_packet

    packet = create_decision_packet(
        symbol="AAPL",
        signal=signal_data,
        ohlcv=df,
        indicators=indicator_dict,
        ml_features=feature_vector,
        risk_checks=risk_results,
    )
    packet.save()

    # Later: reproduce the decision
    loaded = DecisionPacket.load("packets/2024-01-15_AAPL_abc123.json")
    assert loaded.reproduce() == loaded.decision
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """Snapshot of market data at decision time."""
    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    prev_close: Optional[float] = None
    vwap: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None


@dataclass
class IndicatorSnapshot:
    """Snapshot of all indicator values at decision time."""
    rsi_2: Optional[float] = None
    rsi_14: Optional[float] = None
    ibs: Optional[float] = None
    sma_10: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    atr_14: Optional[float] = None
    adx: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    donchian_high: Optional[float] = None
    donchian_low: Optional[float] = None
    sweep_strength: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLSnapshot:
    """Snapshot of ML model inputs and outputs."""
    model_name: str
    model_version: str
    feature_vector: List[float]
    feature_names: List[str]
    prediction: float
    confidence: float
    class_probabilities: Optional[Dict[str, float]] = None
    shap_values: Optional[List[float]] = None
    regime_state: Optional[str] = None


@dataclass
class RiskSnapshot:
    """Snapshot of risk check results."""
    policy_gate_passed: bool
    kill_zone_passed: bool
    exposure_limit_passed: bool
    correlation_limit_passed: bool
    current_exposure: float
    max_allowed_exposure: float
    position_size_calculated: float
    position_size_capped: float
    risk_per_trade: float
    notes: List[str] = field(default_factory=list)


@dataclass
class SignalSnapshot:
    """Snapshot of the generated signal."""
    side: str  # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy: str
    reason: str
    quality_score: Optional[float] = None
    confidence: Optional[float] = None


@dataclass
class DecisionOutcome:
    """Outcome of a trading decision (filled after trade closes)."""
    executed: bool
    fill_price: Optional[float] = None
    fill_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    holding_period_bars: Optional[int] = None


@dataclass
class DecisionPacket:
    """
    Complete reproducibility packet for a trading decision.

    Contains all inputs needed to reproduce the exact same decision.
    Can be saved to disk and loaded later for analysis.
    """

    # Identity
    packet_id: str
    created_at: str
    version: str = "1.0"

    # Market State
    market: Optional[MarketSnapshot] = None

    # Indicators
    indicators: Optional[IndicatorSnapshot] = None

    # ML Components
    ml_models: List[MLSnapshot] = field(default_factory=list)

    # Risk State
    risk: Optional[RiskSnapshot] = None

    # Signal Generated
    signal: Optional[SignalSnapshot] = None

    # Final Decision
    decision: str = "HOLD"  # "BUY", "SELL", "HOLD", "SKIP"
    decision_reason: str = ""

    # Outcome (filled after trade)
    outcome: Optional[DecisionOutcome] = None

    # Raw data hashes for verification
    data_hashes: Dict[str, str] = field(default_factory=dict)

    # Strategy parameters used
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    # Context
    context: Dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute SHA256 hash of packet for integrity verification."""
        # Create deterministic representation
        data = {
            "packet_id": self.packet_id,
            "created_at": self.created_at,
            "market": asdict(self.market) if self.market else None,
            "indicators": asdict(self.indicators) if self.indicators else None,
            "signal": asdict(self.signal) if self.signal else None,
            "decision": self.decision,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def save(self, directory: str = "packets") -> str:
        """
        Save packet to disk.

        Args:
            directory: Directory to save packets

        Returns:
            Path to saved file
        """
        os.makedirs(directory, exist_ok=True)

        # Generate filename
        symbol = self.market.symbol if self.market else "UNKNOWN"
        date = self.created_at[:10]
        filename = f"{date}_{symbol}_{self.packet_id[:8]}.json"
        filepath = os.path.join(directory, filename)

        # Convert to dict
        data = self.to_dict()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved decision packet: {filepath}")
        return filepath

    def to_dict(self) -> Dict[str, Any]:
        """Convert packet to dictionary."""
        return {
            "packet_id": self.packet_id,
            "created_at": self.created_at,
            "version": self.version,
            "market": asdict(self.market) if self.market else None,
            "indicators": asdict(self.indicators) if self.indicators else None,
            "ml_models": [asdict(m) for m in self.ml_models],
            "risk": asdict(self.risk) if self.risk else None,
            "signal": asdict(self.signal) if self.signal else None,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "outcome": asdict(self.outcome) if self.outcome else None,
            "data_hashes": self.data_hashes,
            "strategy_params": self.strategy_params,
            "context": self.context,
            "packet_hash": self.compute_hash(),
        }

    @classmethod
    def load(cls, filepath: str) -> "DecisionPacket":
        """Load packet from disk."""
        with open(filepath, "r") as f:
            data = json.load(f)

        packet = cls(
            packet_id=data["packet_id"],
            created_at=data["created_at"],
            version=data.get("version", "1.0"),
        )

        if data.get("market"):
            packet.market = MarketSnapshot(**data["market"])

        if data.get("indicators"):
            packet.indicators = IndicatorSnapshot(**data["indicators"])

        if data.get("ml_models"):
            packet.ml_models = [MLSnapshot(**m) for m in data["ml_models"]]

        if data.get("risk"):
            packet.risk = RiskSnapshot(**data["risk"])

        if data.get("signal"):
            packet.signal = SignalSnapshot(**data["signal"])

        if data.get("outcome"):
            packet.outcome = DecisionOutcome(**data["outcome"])

        packet.decision = data.get("decision", "HOLD")
        packet.decision_reason = data.get("decision_reason", "")
        packet.data_hashes = data.get("data_hashes", {})
        packet.strategy_params = data.get("strategy_params", {})
        packet.context = data.get("context", {})

        return packet

    def record_outcome(
        self,
        executed: bool,
        fill_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
        pnl: Optional[float] = None,
    ) -> None:
        """Record the outcome after trade closes."""
        self.outcome = DecisionOutcome(
            executed=executed,
            fill_price=fill_price,
            fill_time=datetime.utcnow().isoformat() if fill_price else None,
            exit_price=exit_price,
            exit_time=datetime.utcnow().isoformat() if exit_price else None,
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_pct=((exit_price - fill_price) / fill_price * 100) if exit_price and fill_price else None,
        )


# ============================================================================
# Factory Functions
# ============================================================================

def generate_packet_id() -> str:
    """Generate unique packet ID."""
    import uuid
    return uuid.uuid4().hex[:12]


def create_decision_packet(
    symbol: str,
    ohlcv: Optional[pd.DataFrame] = None,
    indicators: Optional[Dict[str, float]] = None,
    signal: Optional[Dict[str, Any]] = None,
    ml_predictions: Optional[List[Dict[str, Any]]] = None,
    risk_checks: Optional[Dict[str, Any]] = None,
    strategy_params: Optional[Dict[str, Any]] = None,
    decision: str = "HOLD",
    reason: str = "",
) -> DecisionPacket:
    """
    Create a decision packet from trading components.

    Args:
        symbol: Trading symbol
        ohlcv: DataFrame with OHLCV data (last row used)
        indicators: Dict of indicator name -> value
        signal: Signal dict from strategy
        ml_predictions: List of ML model predictions
        risk_checks: Results of risk checks
        strategy_params: Strategy parameters used
        decision: Final decision (BUY/SELL/HOLD/SKIP)
        reason: Reason for decision

    Returns:
        DecisionPacket ready for saving
    """
    packet = DecisionPacket(
        packet_id=generate_packet_id(),
        created_at=datetime.utcnow().isoformat(),
    )

    # Market snapshot from OHLCV
    if ohlcv is not None and not ohlcv.empty:
        row = ohlcv.iloc[-1]
        packet.market = MarketSnapshot(
            symbol=symbol,
            timestamp=str(row.name) if isinstance(row.name, pd.Timestamp) else str(row.get("timestamp", "")),
            open=float(row.get("open", row.get("Open", 0))),
            high=float(row.get("high", row.get("High", 0))),
            low=float(row.get("low", row.get("Low", 0))),
            close=float(row.get("close", row.get("Close", 0))),
            volume=int(row.get("volume", row.get("Volume", 0))),
            prev_close=float(ohlcv.iloc[-2]["close"]) if len(ohlcv) > 1 else None,
        )

        # Compute data hash
        data_str = ohlcv.tail(20).to_csv(index=True)
        packet.data_hashes["ohlcv_20bars"] = hashlib.sha256(data_str.encode()).hexdigest()[:16]

    # Indicator snapshot
    if indicators:
        packet.indicators = IndicatorSnapshot(
            rsi_2=indicators.get("rsi_2"),
            rsi_14=indicators.get("rsi_14"),
            ibs=indicators.get("ibs"),
            sma_10=indicators.get("sma_10"),
            sma_20=indicators.get("sma_20"),
            sma_50=indicators.get("sma_50"),
            sma_200=indicators.get("sma_200"),
            atr_14=indicators.get("atr_14"),
            adx=indicators.get("adx"),
            macd=indicators.get("macd"),
            macd_signal=indicators.get("macd_signal"),
            donchian_high=indicators.get("donchian_high"),
            donchian_low=indicators.get("donchian_low"),
            sweep_strength=indicators.get("sweep_strength"),
            extra={k: v for k, v in indicators.items() if k not in [
                "rsi_2", "rsi_14", "ibs", "sma_10", "sma_20", "sma_50", "sma_200",
                "atr_14", "adx", "macd", "macd_signal", "donchian_high", "donchian_low", "sweep_strength"
            ]},
        )

    # ML snapshots
    if ml_predictions:
        for ml in ml_predictions:
            packet.ml_models.append(MLSnapshot(
                model_name=ml.get("model", "unknown"),
                model_version=ml.get("version", "1.0"),
                feature_vector=ml.get("features", []),
                feature_names=ml.get("feature_names", []),
                prediction=ml.get("prediction", 0.0),
                confidence=ml.get("confidence", 0.0),
                class_probabilities=ml.get("probabilities"),
                regime_state=ml.get("regime"),
            ))

    # Risk snapshot
    if risk_checks:
        packet.risk = RiskSnapshot(
            policy_gate_passed=risk_checks.get("policy_gate", True),
            kill_zone_passed=risk_checks.get("kill_zone", True),
            exposure_limit_passed=risk_checks.get("exposure_limit", True),
            correlation_limit_passed=risk_checks.get("correlation_limit", True),
            current_exposure=risk_checks.get("current_exposure", 0.0),
            max_allowed_exposure=risk_checks.get("max_exposure", 0.4),
            position_size_calculated=risk_checks.get("size_calculated", 0),
            position_size_capped=risk_checks.get("size_capped", 0),
            risk_per_trade=risk_checks.get("risk_per_trade", 0.02),
            notes=risk_checks.get("notes", []),
        )

    # Signal snapshot
    if signal:
        packet.signal = SignalSnapshot(
            side=signal.get("side", "long"),
            entry_price=signal.get("entry_price", 0.0),
            stop_loss=signal.get("stop_loss", 0.0),
            take_profit=signal.get("take_profit", 0.0),
            strategy=signal.get("strategy", "unknown"),
            reason=signal.get("reason", ""),
            quality_score=signal.get("quality_score"),
            confidence=signal.get("confidence"),
        )

    # Strategy params
    if strategy_params:
        packet.strategy_params = strategy_params

    # Decision
    packet.decision = decision
    packet.decision_reason = reason

    return packet


def load_packets(
    directory: str = "packets",
    symbol: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[DecisionPacket]:
    """
    Load multiple packets from directory.

    Args:
        directory: Directory containing packets
        symbol: Filter by symbol
        start_date: Filter by start date (YYYY-MM-DD)
        end_date: Filter by end date (YYYY-MM-DD)

    Returns:
        List of DecisionPacket
    """
    packets = []
    path = Path(directory)

    if not path.exists():
        return packets

    for file in path.glob("*.json"):
        try:
            packet = DecisionPacket.load(str(file))

            # Apply filters
            if symbol and packet.market and packet.market.symbol != symbol:
                continue

            if start_date and packet.created_at[:10] < start_date:
                continue

            if end_date and packet.created_at[:10] > end_date:
                continue

            packets.append(packet)

        except Exception as e:
            logger.warning(f"Failed to load packet {file}: {e}")

    return sorted(packets, key=lambda p: p.created_at)


def analyze_packets(packets: List[DecisionPacket]) -> Dict[str, Any]:
    """
    Analyze a collection of decision packets.

    Returns summary statistics about decisions and outcomes.
    """
    if not packets:
        return {"count": 0}

    total = len(packets)
    executed = [p for p in packets if p.outcome and p.outcome.executed]
    winners = [p for p in executed if p.outcome and p.outcome.pnl and p.outcome.pnl > 0]

    # Decision distribution
    decisions = {}
    for p in packets:
        decisions[p.decision] = decisions.get(p.decision, 0) + 1

    # Strategy distribution
    strategies = {}
    for p in packets:
        if p.signal:
            strategies[p.signal.strategy] = strategies.get(p.signal.strategy, 0) + 1

    return {
        "total_packets": total,
        "executed_trades": len(executed),
        "win_count": len(winners),
        "win_rate": len(winners) / len(executed) if executed else 0,
        "total_pnl": sum(p.outcome.pnl for p in executed if p.outcome and p.outcome.pnl) if executed else 0,
        "decision_distribution": decisions,
        "strategy_distribution": strategies,
        "date_range": (packets[0].created_at[:10], packets[-1].created_at[:10]) if packets else None,
    }
