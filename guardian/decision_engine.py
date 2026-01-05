"""
Decision Engine - Autonomous Decision Making

Makes trading and system decisions autonomously, with escalation
for high-impact decisions.

Decision Types:
- Trade execution (auto)
- Position sizing (auto)
- Risk reduction (auto with alert)
- System restart (escalate)
- Kill switch (escalate)

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import json

from core.structured_log import get_logger

logger = get_logger(__name__)


class DecisionType(Enum):
    """Type of decision."""
    # Auto-execute
    EXECUTE_TRADE = "execute_trade"
    SKIP_TRADE = "skip_trade"
    REDUCE_SIZE = "reduce_size"
    CLOSE_POSITION = "close_position"
    REBALANCE = "rebalance"

    # Alert but execute
    HALT_NEW_TRADES = "halt_new_trades"
    REDUCE_ALL_POSITIONS = "reduce_all_positions"
    EXIT_ALL = "exit_all"

    # Require escalation
    ACTIVATE_KILL_SWITCH = "activate_kill_switch"
    OVERRIDE_RISK_LIMITS = "override_risk_limits"
    CHANGE_STRATEGY = "change_strategy"
    MANUAL_INTERVENTION = "manual_intervention"


class DecisionOutcome(Enum):
    """Outcome of decision."""
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    DEFERRED = "deferred"


class EscalationLevel(Enum):
    """Escalation level."""
    NONE = "none"           # Auto-execute
    NOTIFY = "notify"       # Execute and notify
    CONFIRM = "confirm"     # Wait for confirmation
    MANUAL = "manual"       # Require manual action


@dataclass
class Decision:
    """A decision made by the engine."""
    decision_type: DecisionType
    outcome: DecisionOutcome
    escalation: EscalationLevel
    reason: str
    context: Dict[str, Any]
    confidence: float           # 0-1
    impact: str                 # LOW, MEDIUM, HIGH, CRITICAL
    recommended_action: str
    alternative_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_type": self.decision_type.value,
            "outcome": self.outcome.value,
            "escalation": self.escalation.value,
            "reason": self.reason,
            "context": self.context,
            "confidence": self.confidence,
            "impact": self.impact,
            "recommended_action": self.recommended_action,
            "alternative_actions": self.alternative_actions,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DecisionRule:
    """Rule for making decisions."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    decision_type: DecisionType
    escalation: EscalationLevel
    impact: str
    action: str


class DecisionEngine:
    """
    Autonomous decision making engine.

    Features:
    - Rule-based decisions
    - Confidence scoring
    - Escalation handling
    - Audit trail
    """

    STATE_FILE = Path("state/guardian/decisions.json")
    HISTORY_FILE = Path("state/guardian/decision_history.jsonl")

    # Decisions that require escalation
    ESCALATION_REQUIRED = {
        DecisionType.ACTIVATE_KILL_SWITCH,
        DecisionType.OVERRIDE_RISK_LIMITS,
        DecisionType.CHANGE_STRATEGY,
        DecisionType.MANUAL_INTERVENTION,
    }

    # Decisions that notify but execute
    NOTIFY_ON_EXECUTE = {
        DecisionType.HALT_NEW_TRADES,
        DecisionType.REDUCE_ALL_POSITIONS,
        DecisionType.EXIT_ALL,
    }

    def __init__(self):
        """Initialize decision engine."""
        self._decision_history: List[Decision] = []
        self._pending_escalations: List[Decision] = []
        self._rules: List[DecisionRule] = []

        # Ensure directories
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Set up default decision rules."""
        # Example rules - these would be customized for each trader
        self._rules = [
            DecisionRule(
                name="high_drawdown_reduce",
                condition=lambda ctx: ctx.get("daily_pnl_pct", 0) < -0.02,
                decision_type=DecisionType.REDUCE_ALL_POSITIONS,
                escalation=EscalationLevel.NOTIFY,
                impact="HIGH",
                action="Reduce all positions by 50% due to 2%+ daily drawdown",
            ),
            DecisionRule(
                name="circuit_breaker_halt",
                condition=lambda ctx: ctx.get("circuit_breaker_tripped", False),
                decision_type=DecisionType.HALT_NEW_TRADES,
                escalation=EscalationLevel.NOTIFY,
                impact="HIGH",
                action="Halt new trades due to circuit breaker",
            ),
            DecisionRule(
                name="kill_switch_check",
                condition=lambda ctx: ctx.get("daily_pnl_pct", 0) < -0.05,
                decision_type=DecisionType.ACTIVATE_KILL_SWITCH,
                escalation=EscalationLevel.CONFIRM,
                impact="CRITICAL",
                action="Consider activating kill switch due to 5%+ drawdown",
            ),
        ]

    def evaluate(
        self,
        context: Dict[str, Any],
    ) -> List[Decision]:
        """
        Evaluate context against rules and make decisions.

        Args:
            context: Current system state

        Returns:
            List of decisions
        """
        decisions = []

        for rule in self._rules:
            try:
                if rule.condition(context):
                    decision = self._make_decision(rule, context)
                    decisions.append(decision)
            except Exception as e:
                logger.error(f"Rule {rule.name} failed: {e}")

        # Log decisions
        for decision in decisions:
            self._log_decision(decision)

        return decisions

    def _make_decision(
        self,
        rule: DecisionRule,
        context: Dict[str, Any],
    ) -> Decision:
        """Make a decision based on a rule."""
        # Determine if escalation is required
        if rule.decision_type in self.ESCALATION_REQUIRED:
            outcome = DecisionOutcome.ESCALATED
            escalation = EscalationLevel.CONFIRM
        elif rule.decision_type in self.NOTIFY_ON_EXECUTE:
            outcome = DecisionOutcome.APPROVED
            escalation = EscalationLevel.NOTIFY
        else:
            outcome = DecisionOutcome.APPROVED
            escalation = EscalationLevel.NONE

        decision = Decision(
            decision_type=rule.decision_type,
            outcome=outcome,
            escalation=escalation,
            reason=f"Rule triggered: {rule.name}",
            context=context,
            confidence=0.8,  # Would be calculated more sophisticatedly
            impact=rule.impact,
            recommended_action=rule.action,
            alternative_actions=[],
        )

        if outcome == DecisionOutcome.ESCALATED:
            self._pending_escalations.append(decision)

        return decision

    def decide_trade(
        self,
        signal: Dict[str, Any],
        system_state: Dict[str, Any],
    ) -> Decision:
        """
        Make a decision about a specific trade.

        Args:
            signal: Trading signal
            system_state: Current system state

        Returns:
            Decision about whether to execute
        """
        # Check various conditions
        reasons = []
        should_skip = False
        should_reduce = False
        reduce_factor = 1.0

        # Circuit breaker check
        if system_state.get("circuit_breaker_status") == "RED":
            should_skip = True
            reasons.append("Circuit breaker RED")

        # Kill zone check
        if not system_state.get("in_kill_zone", True):
            should_skip = True
            reasons.append("Outside kill zone")

        # Risk budget check
        if system_state.get("risk_budget_used", 0) > 0.9:
            should_skip = True
            reasons.append("Risk budget exhausted (>90%)")

        # Quality gate check
        signal_quality = signal.get("quality_score", 0)
        if signal_quality < 60:
            should_skip = True
            reasons.append(f"Quality below threshold ({signal_quality} < 60)")

        # Volatility adjustment
        if system_state.get("vix", 20) > 30:
            should_reduce = True
            reduce_factor = 0.5
            reasons.append("VIX > 30, reducing size 50%")

        # Make decision
        if should_skip:
            decision = Decision(
                decision_type=DecisionType.SKIP_TRADE,
                outcome=DecisionOutcome.APPROVED,
                escalation=EscalationLevel.NONE,
                reason="; ".join(reasons),
                context={"signal": signal, "system_state": system_state},
                confidence=0.9,
                impact="LOW",
                recommended_action="Skip this trade",
                alternative_actions=["Wait for better setup"],
            )
        elif should_reduce:
            decision = Decision(
                decision_type=DecisionType.REDUCE_SIZE,
                outcome=DecisionOutcome.APPROVED,
                escalation=EscalationLevel.NONE,
                reason="; ".join(reasons),
                context={
                    "signal": signal,
                    "reduce_factor": reduce_factor,
                },
                confidence=0.85,
                impact="MEDIUM",
                recommended_action=f"Execute at {reduce_factor:.0%} size",
                alternative_actions=["Skip trade entirely"],
            )
        else:
            decision = Decision(
                decision_type=DecisionType.EXECUTE_TRADE,
                outcome=DecisionOutcome.APPROVED,
                escalation=EscalationLevel.NONE,
                reason="All checks passed",
                context={"signal": signal},
                confidence=0.9,
                impact="LOW",
                recommended_action="Execute trade at full size",
                alternative_actions=[],
            )

        self._log_decision(decision)
        return decision

    def get_pending_escalations(self) -> List[Decision]:
        """Get decisions pending human approval."""
        return self._pending_escalations.copy()

    def resolve_escalation(
        self,
        decision: Decision,
        approved: bool,
        resolver: str = "human",
    ) -> None:
        """Resolve a pending escalation."""
        if decision in self._pending_escalations:
            self._pending_escalations.remove(decision)

            resolution = Decision(
                decision_type=decision.decision_type,
                outcome=DecisionOutcome.APPROVED if approved else DecisionOutcome.REJECTED,
                escalation=EscalationLevel.NONE,
                reason=f"Resolved by {resolver}: {'approved' if approved else 'rejected'}",
                context=decision.context,
                confidence=1.0,
                impact=decision.impact,
                recommended_action=decision.recommended_action if approved else "No action",
                alternative_actions=[],
            )

            self._log_decision(resolution)

    def _log_decision(self, decision: Decision) -> None:
        """Log a decision to history."""
        self._decision_history.append(decision)

        # Append to file
        try:
            with open(self.HISTORY_FILE, "a") as f:
                f.write(json.dumps(decision.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")

    def get_decision_history(
        self,
        hours: int = 24,
    ) -> List[Decision]:
        """Get recent decisions."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            d for d in self._decision_history
            if d.timestamp > cutoff
        ]


# Singleton
_engine: Optional[DecisionEngine] = None


def get_decision_engine() -> DecisionEngine:
    """Get or create singleton engine."""
    global _engine
    if _engine is None:
        _engine = DecisionEngine()
    return _engine


if __name__ == "__main__":
    # Demo
    engine = DecisionEngine()

    print("=== Decision Engine Demo ===\n")

    # Test trade decision
    signal = {
        "symbol": "AAPL",
        "side": "long",
        "entry_price": 175,
        "stop_loss": 170,
        "quality_score": 75,
    }

    state = {
        "circuit_breaker_status": "GREEN",
        "in_kill_zone": True,
        "risk_budget_used": 0.5,
        "vix": 25,
    }

    decision = engine.decide_trade(signal, state)
    print(f"Decision: {decision.decision_type.value}")
    print(f"Outcome: {decision.outcome.value}")
    print(f"Reason: {decision.reason}")
    print(f"Action: {decision.recommended_action}")

    # Test with high VIX
    print("\n--- High VIX Scenario ---")
    state["vix"] = 35
    decision = engine.decide_trade(signal, state)
    print(f"Decision: {decision.decision_type.value}")
    print(f"Action: {decision.recommended_action}")

    # Test with circuit breaker
    print("\n--- Circuit Breaker Scenario ---")
    state["circuit_breaker_status"] = "RED"
    decision = engine.decide_trade(signal, state)
    print(f"Decision: {decision.decision_type.value}")
    print(f"Reason: {decision.reason}")
