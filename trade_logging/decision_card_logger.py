"""
Decision Card Logger - Complete trade audit trail.

Creates structured JSON files for every trade decision with:
- WHAT: Trade plan (entry, target, stop, qty)
- WHY: Signal drivers and their values
- HOW: Execution context (limit price, spread, quote age)
- RISK: All constraint checks and their results
- MODEL: ML outputs and confidence scores
- TCA: Transaction cost analysis (filled post-trade)
- RESULT: Trade outcome (filled post-exit)

Storage: logs/decision_cards/YYYYMMDD/<card_id>.json

Usage:
    from trade_logging import get_card_logger, TradePlan, SignalDriver

    logger = get_card_logger()

    # Create card on entry decision
    card = logger.create_card(
        symbol="AAPL",
        side="long",
        strategy="ibs_rsi",
        plan=TradePlan(entry=185.50, target=188.20, stop=183.10, qty=1),
        signals=[
            SignalDriver(feature="ibs", value=0.05, contribution=0.35),
            SignalDriver(feature="rsi2", value=4.2, contribution=0.30),
        ],
        risk_checks=[...],
        model_info=ModelInfo(ml_confidence=0.72, regime="bull"),
    )
    logger.save_card(card)

    # Update after fill
    logger.update_tca(card.card_id, TCAResult(slippage_bps=5.2, fill_price=185.55))

    # Update after exit
    logger.update_result(card.card_id, TradeResult(pnl=27.50, bars_held=3))
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default storage location
DEFAULT_LOG_DIR = Path(__file__).parent.parent / "logs" / "decision_cards"


# ============================================================================
# Supporting Data Classes
# ============================================================================

@dataclass
class TradePlan:
    """Trade plan details: entry, target, stop, quantity."""
    entry_price: float
    target_price: float
    stop_loss: float
    qty: int = 1
    notional: float = 0.0
    rr_ratio: float = 0.0

    def __post_init__(self):
        if self.notional == 0.0:
            self.notional = self.entry_price * self.qty
        if self.rr_ratio == 0.0 and self.entry_price > 0:
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.target_price - self.entry_price)
            self.rr_ratio = reward / risk if risk > 0 else 0.0


@dataclass
class SignalDriver:
    """A feature that drove the signal with its contribution."""
    feature: str
    value: float
    contribution: float = 0.0  # Weight/importance (0-1)
    description: str = ""


@dataclass
class ExecutionContext:
    """Execution context at decision time."""
    limit_price: float = 0.0
    spread_pct: float = 0.0
    quote_age_ms: int = 0
    best_bid: float = 0.0
    best_ask: float = 0.0
    volume_at_decision: int = 0


@dataclass
class RiskCheck:
    """Result of a risk gate check."""
    gate_name: str
    passed: bool
    value: float = 0.0
    limit: float = 0.0
    message: str = ""


@dataclass
class ModelInfo:
    """ML/AI model outputs."""
    ml_confidence: float = 0.0
    calibrated_prob: float = 0.0
    regime: str = ""
    cognitive_confidence: float = 0.0
    adjudication_score: float = 0.0
    patterns_detected: List[str] = field(default_factory=list)


@dataclass
class TCAResult:
    """Transaction Cost Analysis results (post-trade)."""
    slippage_bps: float = 0.0
    fill_price: float = 0.0
    fill_time_ms: int = 0
    spread_capture_bps: float = 0.0
    market_impact_bps: float = 0.0


@dataclass
class TradeResult:
    """Trade outcome (post-exit)."""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    exit_reason: str = ""
    exit_price: float = 0.0
    exit_timestamp: str = ""
    win: bool = False

    def __post_init__(self):
        if self.pnl != 0.0:
            self.win = self.pnl > 0


# ============================================================================
# Decision Card
# ============================================================================

@dataclass
class DecisionCard:
    """
    Complete decision artifact for trade audit/review.

    Every trade decision (enter, skip, exit) generates a card.
    Cards are immutable once created, but TCA and Result can be added.
    """
    # Identity
    card_id: str
    timestamp: str
    environment: str  # "paper" | "live" | "backtest"

    # Universe/Action
    symbol: str
    side: str  # "long" | "short" | "skip"
    strategy: str

    # WHAT: Trade plan
    plan: TradePlan

    # WHY: Signal drivers (top k features)
    signals: List[SignalDriver]

    # HOW: Execution context
    execution: ExecutionContext

    # RISK: Constraint checks
    risk_checks: List[RiskCheck]

    # MODEL: ML/AI outputs
    model_info: ModelInfo

    # TCA: Transaction costs (filled post-trade)
    tca: Optional[TCAResult] = None

    # RESULT: Trade outcome (filled post-exit)
    result: Optional[TradeResult] = None

    # Lineage
    config_hash: str = ""
    data_hash: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "card_id": self.card_id,
            "timestamp": self.timestamp,
            "environment": self.environment,
            "symbol": self.symbol,
            "side": self.side,
            "strategy": self.strategy,
            "plan": asdict(self.plan),
            "signals": [asdict(s) for s in self.signals],
            "execution": asdict(self.execution),
            "risk_checks": [asdict(r) for r in self.risk_checks],
            "model_info": asdict(self.model_info),
            "tca": asdict(self.tca) if self.tca else None,
            "result": asdict(self.result) if self.result else None,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "metadata": self.metadata,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionCard":
        """Create DecisionCard from dictionary."""
        return cls(
            card_id=data["card_id"],
            timestamp=data["timestamp"],
            environment=data["environment"],
            symbol=data["symbol"],
            side=data["side"],
            strategy=data["strategy"],
            plan=TradePlan(**data["plan"]),
            signals=[SignalDriver(**s) for s in data["signals"]],
            execution=ExecutionContext(**data["execution"]),
            risk_checks=[RiskCheck(**r) for r in data["risk_checks"]],
            model_info=ModelInfo(**data["model_info"]),
            tca=TCAResult(**data["tca"]) if data.get("tca") else None,
            result=TradeResult(**data["result"]) if data.get("result") else None,
            config_hash=data.get("config_hash", ""),
            data_hash=data.get("data_hash", ""),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Decision Card Logger
# ============================================================================

class DecisionCardLogger:
    """
    Logs trade decisions as structured JSON files.

    Features:
    - Creates unique card IDs with timestamps
    - Stores cards organized by date
    - Supports updating TCA and Result after initial save
    - Provides retrieval by card ID or date
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        environment: str = "paper",
    ):
        """
        Initialize the decision card logger.

        Args:
            log_dir: Directory for card storage (default: logs/decision_cards)
            environment: Trading environment ("paper", "live", "backtest")
        """
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        self.environment = environment

        # Ensure directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DecisionCardLogger initialized: {self.log_dir}")

    def _generate_card_id(self, symbol: str) -> str:
        """Generate unique card ID."""
        now = datetime.utcnow()
        date_str = now.strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        return f"DC_{date_str}_{symbol}_{short_uuid}"

    def _get_date_dir(self, date_str: Optional[str] = None) -> Path:
        """Get directory for given date."""
        if date_str is None:
            date_str = datetime.utcnow().strftime("%Y%m%d")
        date_dir = self.log_dir / date_str
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir

    def _get_config_hash(self) -> str:
        """Get hash of frozen strategy parameters."""
        try:
            config_path = Path(__file__).parent.parent / "config" / "frozen_strategy_params_v2.2.json"
            if config_path.exists():
                with open(config_path, 'rb') as f:
                    return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            pass
        return ""

    def create_card(
        self,
        symbol: str,
        side: str,
        strategy: str,
        plan: TradePlan,
        signals: Optional[List[SignalDriver]] = None,
        execution: Optional[ExecutionContext] = None,
        risk_checks: Optional[List[RiskCheck]] = None,
        model_info: Optional[ModelInfo] = None,
        config_hash: str = "",
        data_hash: str = "",
        environment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DecisionCard:
        """
        Create a new decision card.

        Args:
            symbol: Stock symbol
            side: "long", "short", or "skip"
            strategy: Strategy name (e.g., "ibs_rsi", "turtle_soup")
            plan: Trade plan with entry/target/stop
            signals: List of signal drivers
            execution: Execution context
            risk_checks: List of risk check results
            model_info: ML/AI model outputs
            config_hash: Hash of strategy config (auto-generated if empty)
            data_hash: Hash of data used
            environment: Override default environment
            metadata: Additional metadata

        Returns:
            DecisionCard ready to be saved
        """
        card_id = self._generate_card_id(symbol)
        timestamp = datetime.utcnow().isoformat() + "Z"

        if not config_hash:
            config_hash = self._get_config_hash()

        card = DecisionCard(
            card_id=card_id,
            timestamp=timestamp,
            environment=environment or self.environment,
            symbol=symbol,
            side=side,
            strategy=strategy,
            plan=plan,
            signals=signals or [],
            execution=execution or ExecutionContext(),
            risk_checks=risk_checks or [],
            model_info=model_info or ModelInfo(),
            config_hash=config_hash,
            data_hash=data_hash,
            metadata=metadata or {},
        )

        logger.debug(f"Created decision card: {card_id}")
        return card

    def save_card(self, card: DecisionCard) -> Path:
        """
        Save decision card to disk.

        Args:
            card: DecisionCard to save

        Returns:
            Path to saved file
        """
        # Extract date from card_id (DC_YYYYMMDD_...)
        try:
            date_str = card.card_id.split("_")[1][:8]
        except (IndexError, ValueError):
            date_str = datetime.utcnow().strftime("%Y%m%d")

        date_dir = self._get_date_dir(date_str)
        file_path = date_dir / f"{card.card_id}.json"

        with open(file_path, 'w') as f:
            json.dump(card.to_dict(), f, indent=2)

        logger.info(f"Saved decision card: {file_path}")
        return file_path

    def load_card(self, card_id: str) -> Optional[DecisionCard]:
        """
        Load decision card by ID.

        Args:
            card_id: Card ID (e.g., "DC_20251231_093500_AAPL_abc12345")

        Returns:
            DecisionCard if found, None otherwise
        """
        try:
            date_str = card_id.split("_")[1][:8]
        except (IndexError, ValueError):
            logger.warning(f"Invalid card_id format: {card_id}")
            return None

        file_path = self._get_date_dir(date_str) / f"{card_id}.json"

        if not file_path.exists():
            logger.warning(f"Card not found: {file_path}")
            return None

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return DecisionCard.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load card {card_id}: {e}")
            return None

    def update_tca(self, card_id: str, tca: TCAResult) -> bool:
        """
        Update TCA results on existing card.

        Args:
            card_id: Card ID to update
            tca: TCA results from execution

        Returns:
            True if successful, False otherwise
        """
        card = self.load_card(card_id)
        if card is None:
            return False

        card.tca = tca
        self.save_card(card)
        logger.info(f"Updated TCA for card: {card_id}")
        return True

    def update_result(self, card_id: str, result: TradeResult) -> bool:
        """
        Update trade result on existing card.

        Args:
            card_id: Card ID to update
            result: Trade result from exit

        Returns:
            True if successful, False otherwise
        """
        card = self.load_card(card_id)
        if card is None:
            return False

        card.result = result
        self.save_card(card)
        logger.info(f"Updated result for card: {card_id}")
        return True

    def load_cards_for_date(self, date_str: str) -> List[DecisionCard]:
        """
        Load all cards for a given date.

        Args:
            date_str: Date string (YYYYMMDD)

        Returns:
            List of DecisionCards
        """
        date_dir = self.log_dir / date_str
        if not date_dir.exists():
            return []

        cards = []
        for file_path in date_dir.glob("DC_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                cards.append(DecisionCard.from_dict(data))
            except Exception as e:
                logger.warning(f"Failed to load card {file_path}: {e}")

        return cards

    def get_recent_cards(self, days: int = 7) -> List[DecisionCard]:
        """
        Get cards from the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of DecisionCards sorted by timestamp
        """
        from datetime import timedelta

        cards = []
        today = datetime.utcnow().date()

        for i in range(days):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y%m%d")
            cards.extend(self.load_cards_for_date(date_str))

        # Sort by timestamp descending
        cards.sort(key=lambda c: c.timestamp, reverse=True)
        return cards

    def get_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get statistics from recent cards.

        Args:
            days: Number of days to analyze

        Returns:
            Stats dictionary
        """
        cards = self.get_recent_cards(days)

        if not cards:
            return {"total": 0, "win_rate": 0.0, "avg_pnl": 0.0}

        completed = [c for c in cards if c.result is not None]
        wins = [c for c in completed if c.result.win]

        return {
            "total_cards": len(cards),
            "completed_trades": len(completed),
            "pending_trades": len(cards) - len(completed),
            "win_rate": len(wins) / len(completed) if completed else 0.0,
            "avg_pnl": sum(c.result.pnl for c in completed) / len(completed) if completed else 0.0,
            "total_pnl": sum(c.result.pnl for c in completed),
            "avg_bars_held": sum(c.result.bars_held for c in completed) / len(completed) if completed else 0.0,
            "by_strategy": self._group_by_strategy(completed),
        }

    def _group_by_strategy(self, cards: List[DecisionCard]) -> Dict[str, Dict]:
        """Group stats by strategy."""
        from collections import defaultdict

        by_strat = defaultdict(list)
        for c in cards:
            by_strat[c.strategy].append(c)

        result = {}
        for strat, strat_cards in by_strat.items():
            wins = [c for c in strat_cards if c.result and c.result.win]
            result[strat] = {
                "count": len(strat_cards),
                "win_rate": len(wins) / len(strat_cards) if strat_cards else 0.0,
                "avg_pnl": sum(c.result.pnl for c in strat_cards if c.result) / len(strat_cards) if strat_cards else 0.0,
            }
        return result


# ============================================================================
# Singleton Instance
# ============================================================================

_card_logger: Optional[DecisionCardLogger] = None


def get_card_logger(
    log_dir: Optional[Path] = None,
    environment: str = "paper",
) -> DecisionCardLogger:
    """Get or create singleton DecisionCardLogger."""
    global _card_logger
    if _card_logger is None or log_dir is not None:
        _card_logger = DecisionCardLogger(log_dir=log_dir, environment=environment)
    return _card_logger
