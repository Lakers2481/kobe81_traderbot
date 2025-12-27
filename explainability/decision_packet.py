"""
Decision Packet builder for TOTD explanations.

Creates JSON artifacts capturing all decision context for a trade.
CRITICAL: LLM playbooks may ONLY reference fields in this packet.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import json
import uuid


@dataclass
class RiskGateResult:
    """Result from a risk gate check."""
    gate_name: str
    passed: bool
    value: Optional[float] = None
    limit: Optional[float] = None
    message: str = ""


@dataclass
class HistoricalAnalog:
    """A similar historical trade for context."""
    date: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    holding_days: int
    similarity_score: float


@dataclass
class ExecutionPlan:
    """Planned execution details."""
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    notional: float
    risk_amount: float
    reward_amount: float
    reward_risk_ratio: float


@dataclass
class DecisionPacket:
    """
    Complete decision context for a trade.

    CRITICAL: This is the single source of truth for trade explanations.
    LLM playbooks MUST only reference fields present here.
    """
    # Identification (required fields first)
    run_id: str
    timestamp: str
    symbol: str
    side: str  # "buy" or "sell"
    strategy_name: str

    # Hash (computed on post_init, has default)
    packet_hash: str = ""

    # Why this trade?
    strategy_reasons: List[str] = field(default_factory=list)
    signal_description: str = ""

    # Feature values at signal time
    feature_values: Dict[str, float] = field(default_factory=dict)

    # ML model outputs (if applicable)
    ml_outputs: Dict[str, Any] = field(default_factory=dict)

    # Sentiment (if applicable)
    sentiment_score: Optional[float] = None
    sentiment_source: Optional[str] = None

    # Risk gate results
    risk_gate_results: List[RiskGateResult] = field(default_factory=list)

    # Historical analogs
    historical_analogs: List[HistoricalAnalog] = field(default_factory=list)

    # Execution plan
    execution_plan: Optional[ExecutionPlan] = None

    # Market context
    market_context: Dict[str, Any] = field(default_factory=dict)

    # What we don't know
    unknowns: List[str] = field(default_factory=list)

    # Lineage
    data_hash: str = ""
    model_hash: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.packet_hash:
            self.packet_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of packet contents."""
        content = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "side": self.side,
            "strategy_name": self.strategy_name,
            "strategy_reasons": self.strategy_reasons,
            "feature_values": self.feature_values,
            "execution_plan": asdict(self.execution_plan) if self.execution_plan else None,
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "packet_hash": self.packet_hash,
            "symbol": self.symbol,
            "side": self.side,
            "strategy_name": self.strategy_name,
            "strategy_reasons": self.strategy_reasons,
            "signal_description": self.signal_description,
            "feature_values": self.feature_values,
            "ml_outputs": self.ml_outputs,
            "sentiment_score": self.sentiment_score,
            "sentiment_source": self.sentiment_source,
            "risk_gate_results": [asdict(r) for r in self.risk_gate_results],
            "historical_analogs": [asdict(a) for a in self.historical_analogs],
            "execution_plan": asdict(self.execution_plan) if self.execution_plan else None,
            "market_context": self.market_context,
            "unknowns": self.unknowns,
            "data_hash": self.data_hash,
            "model_hash": self.model_hash,
            "metadata": self.metadata,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: Path) -> None:
        """Save packet to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DecisionPacket:
        """Load from dictionary."""
        # Convert nested objects
        risk_gates = [
            RiskGateResult(**r) for r in data.get("risk_gate_results", [])
        ]
        analogs = [
            HistoricalAnalog(**a) for a in data.get("historical_analogs", [])
        ]
        exec_plan = None
        if data.get("execution_plan"):
            exec_plan = ExecutionPlan(**data["execution_plan"])

        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            packet_hash=data.get("packet_hash", ""),
            symbol=data["symbol"],
            side=data["side"],
            strategy_name=data["strategy_name"],
            strategy_reasons=data.get("strategy_reasons", []),
            signal_description=data.get("signal_description", ""),
            feature_values=data.get("feature_values", {}),
            ml_outputs=data.get("ml_outputs", {}),
            sentiment_score=data.get("sentiment_score"),
            sentiment_source=data.get("sentiment_source"),
            risk_gate_results=risk_gates,
            historical_analogs=analogs,
            execution_plan=exec_plan,
            market_context=data.get("market_context", {}),
            unknowns=data.get("unknowns", []),
            data_hash=data.get("data_hash", ""),
            model_hash=data.get("model_hash", ""),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: Path) -> DecisionPacket:
        """Load packet from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_field(self, field_name: str, default: Any = None) -> Any:
        """Safely get a field value."""
        if hasattr(self, field_name):
            return getattr(self, field_name)
        return default

    def is_field_present(self, field_name: str) -> bool:
        """Check if a field is present and non-empty."""
        val = self.get_field(field_name)
        if val is None:
            return False
        if isinstance(val, (list, dict)) and len(val) == 0:
            return False
        if isinstance(val, str) and val.strip() == "":
            return False
        return True


def build_decision_packet(
    symbol: str,
    side: str,
    strategy_name: str,
    signal: Optional[Dict[str, Any]] = None,
    ml_result: Optional[Dict[str, Any]] = None,
    risk_checks: Optional[List[Dict[str, Any]]] = None,
    execution_plan: Optional[Dict[str, Any]] = None,
    feature_values: Optional[Dict[str, float]] = None,
    historical_analogs: Optional[List[Dict[str, Any]]] = None,
    market_context: Optional[Dict[str, Any]] = None,
    sentiment_score: Optional[float] = None,
    sentiment_source: Optional[str] = None,
    data_hash: str = "",
    model_hash: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> DecisionPacket:
    """
    Build a decision packet from trade components.

    This is the main entry point for creating packets.
    """
    run_id = f"totd_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Extract strategy reasons from signal
    strategy_reasons = []
    signal_description = ""
    if signal:
        if "reason" in signal:
            strategy_reasons.append(signal["reason"])
        if "reasons" in signal:
            strategy_reasons.extend(signal["reasons"])
        signal_description = signal.get("description", "")

    # Build risk gate results
    risk_gate_results = []
    unknowns = []

    if risk_checks:
        for check in risk_checks:
            result = RiskGateResult(
                gate_name=check.get("gate", "unknown"),
                passed=check.get("passed", False),
                value=check.get("value"),
                limit=check.get("limit"),
                message=check.get("message", ""),
            )
            risk_gate_results.append(result)
    else:
        unknowns.append("risk_checks: Not provided")

    # Build execution plan
    exec_plan = None
    if execution_plan:
        exec_plan = ExecutionPlan(
            entry_price=execution_plan.get("entry_price", 0),
            stop_loss=execution_plan.get("stop_loss", 0),
            take_profit=execution_plan.get("take_profit", 0),
            position_size=execution_plan.get("position_size", 0),
            notional=execution_plan.get("notional", 0),
            risk_amount=execution_plan.get("risk_amount", 0),
            reward_amount=execution_plan.get("reward_amount", 0),
            reward_risk_ratio=execution_plan.get("reward_risk_ratio", 0),
        )
    else:
        unknowns.append("execution_plan: Not provided")

    # Build historical analogs
    analogs = []
    if historical_analogs:
        for a in historical_analogs:
            analogs.append(HistoricalAnalog(
                date=a.get("date", ""),
                symbol=a.get("symbol", ""),
                side=a.get("side", ""),
                entry_price=a.get("entry_price", 0),
                exit_price=a.get("exit_price", 0),
                pnl_pct=a.get("pnl_pct", 0),
                holding_days=a.get("holding_days", 0),
                similarity_score=a.get("similarity_score", 0),
            ))

    # Check for unknowns
    if not feature_values:
        unknowns.append("feature_values: Not provided")
    if ml_result is None:
        unknowns.append("ml_outputs: No ML model used")
    if sentiment_score is None:
        unknowns.append("sentiment_score: Not available")
    if not market_context:
        unknowns.append("market_context: Not provided")

    return DecisionPacket(
        run_id=run_id,
        timestamp=timestamp,
        symbol=symbol,
        side=side,
        strategy_name=strategy_name,
        strategy_reasons=strategy_reasons,
        signal_description=signal_description,
        feature_values=feature_values or {},
        ml_outputs=ml_result or {},
        sentiment_score=sentiment_score,
        sentiment_source=sentiment_source,
        risk_gate_results=risk_gate_results,
        historical_analogs=analogs,
        execution_plan=exec_plan,
        market_context=market_context or {},
        unknowns=unknowns,
        data_hash=data_hash,
        model_hash=model_hash,
        metadata=metadata or {},
    )


def load_latest_packet(packets_dir: Path = Path("reports/totd")) -> Optional[DecisionPacket]:
    """Load the most recent decision packet."""
    if not packets_dir.exists():
        return None

    # Find most recent .json file
    json_files = sorted(packets_dir.glob("*.json"), reverse=True)
    if not json_files:
        return None

    return DecisionPacket.load(json_files[0])
