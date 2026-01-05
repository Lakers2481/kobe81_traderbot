"""
Emergency Protocol - Automatic Crisis Response

Handles emergency situations automatically with predefined responses.

Emergency Levels:
- LEVEL 1: Minor issue, continue trading with caution
- LEVEL 2: Significant issue, reduce exposure
- LEVEL 3: Major issue, halt new trades
- LEVEL 4: Critical, close all positions
- LEVEL 5: System failure, activate kill switch

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


class EmergencyLevel(Enum):
    """Emergency severity levels."""
    NONE = 0
    LEVEL_1 = 1     # Minor - continue with caution
    LEVEL_2 = 2     # Significant - reduce exposure
    LEVEL_3 = 3     # Major - halt new trades
    LEVEL_4 = 4     # Critical - close positions
    LEVEL_5 = 5     # System failure - kill switch


class EmergencyType(Enum):
    """Types of emergencies."""
    DRAWDOWN = "drawdown"
    CIRCUIT_BREAKER = "circuit_breaker"
    SYSTEM_FAILURE = "system_failure"
    DATA_FEED_FAILURE = "data_feed_failure"
    BROKER_DISCONNECTION = "broker_disconnection"
    MARKET_HALT = "market_halt"
    VIX_SPIKE = "vix_spike"
    POSITION_LIMIT = "position_limit"
    MANUAL = "manual"


class EmergencyAction(Enum):
    """Actions that can be taken."""
    CONTINUE = "continue"
    REDUCE_EXPOSURE = "reduce_exposure"
    HALT_NEW_TRADES = "halt_new_trades"
    CLOSE_ALL_POSITIONS = "close_all_positions"
    ACTIVATE_KILL_SWITCH = "activate_kill_switch"
    ALERT_ONLY = "alert_only"


@dataclass
class EmergencyEvent:
    """Record of an emergency event."""
    emergency_type: EmergencyType
    level: EmergencyLevel
    trigger_value: Any
    threshold: Any
    action_taken: EmergencyAction
    message: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emergency_type": self.emergency_type.value,
            "level": self.level.value,
            "trigger_value": self.trigger_value,
            "threshold": self.threshold,
            "action_taken": self.action_taken.value,
            "message": self.message,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EmergencyRule:
    """Rule for triggering emergency response."""
    name: str
    emergency_type: EmergencyType
    condition: Callable[[Dict[str, Any]], bool]
    get_value: Callable[[Dict[str, Any]], Any]
    threshold: Any
    level: EmergencyLevel
    action: EmergencyAction
    message_template: str


class EmergencyProtocol:
    """
    Automatic emergency response system.

    Features:
    - Rule-based triggers
    - Automatic responses
    - Escalation handling
    - Recovery procedures
    """

    STATE_FILE = Path("state/guardian/emergency.json")
    HISTORY_FILE = Path("state/guardian/emergency_history.jsonl")
    KILL_SWITCH_FILE = Path("state/KILL_SWITCH")

    def __init__(self):
        """Initialize emergency protocol."""
        self._rules: List[EmergencyRule] = []
        self._active_emergencies: List[EmergencyEvent] = []
        self._current_level: EmergencyLevel = EmergencyLevel.NONE
        self._action_handlers: Dict[EmergencyAction, Callable] = {}

        # Ensure directories
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        self._setup_default_rules()
        self._setup_action_handlers()
        self._load_state()

    def _setup_default_rules(self) -> None:
        """Set up default emergency rules."""
        self._rules = [
            # Drawdown rules
            EmergencyRule(
                name="minor_drawdown",
                emergency_type=EmergencyType.DRAWDOWN,
                condition=lambda ctx: ctx.get("daily_drawdown_pct", 0) > 0.01,
                get_value=lambda ctx: ctx.get("daily_drawdown_pct", 0),
                threshold=0.01,
                level=EmergencyLevel.LEVEL_1,
                action=EmergencyAction.CONTINUE,
                message_template="Daily drawdown at {value:.1%} (threshold: {threshold:.1%})",
            ),
            EmergencyRule(
                name="significant_drawdown",
                emergency_type=EmergencyType.DRAWDOWN,
                condition=lambda ctx: ctx.get("daily_drawdown_pct", 0) > 0.02,
                get_value=lambda ctx: ctx.get("daily_drawdown_pct", 0),
                threshold=0.02,
                level=EmergencyLevel.LEVEL_2,
                action=EmergencyAction.REDUCE_EXPOSURE,
                message_template="Daily drawdown at {value:.1%} - reducing exposure",
            ),
            EmergencyRule(
                name="major_drawdown",
                emergency_type=EmergencyType.DRAWDOWN,
                condition=lambda ctx: ctx.get("daily_drawdown_pct", 0) > 0.03,
                get_value=lambda ctx: ctx.get("daily_drawdown_pct", 0),
                threshold=0.03,
                level=EmergencyLevel.LEVEL_3,
                action=EmergencyAction.HALT_NEW_TRADES,
                message_template="Daily drawdown at {value:.1%} - halting new trades",
            ),
            EmergencyRule(
                name="critical_drawdown",
                emergency_type=EmergencyType.DRAWDOWN,
                condition=lambda ctx: ctx.get("daily_drawdown_pct", 0) > 0.05,
                get_value=lambda ctx: ctx.get("daily_drawdown_pct", 0),
                threshold=0.05,
                level=EmergencyLevel.LEVEL_4,
                action=EmergencyAction.CLOSE_ALL_POSITIONS,
                message_template="CRITICAL: Daily drawdown at {value:.1%} - closing positions",
            ),

            # VIX rules
            EmergencyRule(
                name="vix_elevated",
                emergency_type=EmergencyType.VIX_SPIKE,
                condition=lambda ctx: ctx.get("vix", 15) > 30,
                get_value=lambda ctx: ctx.get("vix", 15),
                threshold=30,
                level=EmergencyLevel.LEVEL_2,
                action=EmergencyAction.REDUCE_EXPOSURE,
                message_template="VIX at {value:.1f} - reducing exposure",
            ),
            EmergencyRule(
                name="vix_extreme",
                emergency_type=EmergencyType.VIX_SPIKE,
                condition=lambda ctx: ctx.get("vix", 15) > 40,
                get_value=lambda ctx: ctx.get("vix", 15),
                threshold=40,
                level=EmergencyLevel.LEVEL_3,
                action=EmergencyAction.HALT_NEW_TRADES,
                message_template="VIX at {value:.1f} - extreme volatility, halting trades",
            ),

            # System failures
            EmergencyRule(
                name="broker_disconnection",
                emergency_type=EmergencyType.BROKER_DISCONNECTION,
                condition=lambda ctx: not ctx.get("broker_connected", True),
                get_value=lambda ctx: "disconnected",
                threshold="connected",
                level=EmergencyLevel.LEVEL_3,
                action=EmergencyAction.HALT_NEW_TRADES,
                message_template="Broker disconnected - halting all trading",
            ),
            EmergencyRule(
                name="data_feed_failure",
                emergency_type=EmergencyType.DATA_FEED_FAILURE,
                condition=lambda ctx: not ctx.get("data_feed_healthy", True),
                get_value=lambda ctx: "failed",
                threshold="healthy",
                level=EmergencyLevel.LEVEL_3,
                action=EmergencyAction.HALT_NEW_TRADES,
                message_template="Data feed failure - halting all trading",
            ),
        ]

    def _setup_action_handlers(self) -> None:
        """Set up action handlers."""
        self._action_handlers = {
            EmergencyAction.CONTINUE: self._action_continue,
            EmergencyAction.REDUCE_EXPOSURE: self._action_reduce_exposure,
            EmergencyAction.HALT_NEW_TRADES: self._action_halt_trades,
            EmergencyAction.CLOSE_ALL_POSITIONS: self._action_close_all,
            EmergencyAction.ACTIVATE_KILL_SWITCH: self._action_kill_switch,
            EmergencyAction.ALERT_ONLY: self._action_alert_only,
        }

    def _action_continue(self, event: EmergencyEvent) -> None:
        """Continue trading with caution."""
        logger.warning(f"Emergency CONTINUE: {event.message}")

    def _action_reduce_exposure(self, event: EmergencyEvent) -> None:
        """
        Reduce position exposure by 50%.

        IMPLEMENTED (2026-01-04): Now actually reduces position sizing.
        """
        logger.warning(f"Emergency REDUCE_EXPOSURE: {event.message}")
        try:
            from risk.dynamic_position_sizer import set_size_multiplier
            from core.structured_log import jlog
            set_size_multiplier(0.5)  # Reduce all future positions by 50%
            jlog('emergency_action', {
                'action': 'reduce_exposure',
                'multiplier': 0.5,
                'event': event.message,
                'level': event.level.name
            })
            logger.warning("Position sizing reduced to 50%")
        except ImportError:
            logger.warning("dynamic_position_sizer not available, falling back to PolicyGate")
            try:
                from risk.policy_gate import PolicyGate
                gate = PolicyGate.from_config()
                gate.reduce_daily_budget(0.5)  # Reduce remaining daily budget by 50%
            except Exception as e:
                logger.error(f"Failed to reduce exposure: {e}")
        except Exception as e:
            logger.error(f"Failed to reduce exposure: {e}")

    def _action_halt_trades(self, event: EmergencyEvent) -> None:
        """
        Halt all new trades via PolicyGate.

        IMPLEMENTED (2026-01-04): Now actually halts trading.
        """
        logger.error(f"Emergency HALT_TRADES: {event.message}")
        try:
            from risk.policy_gate import PolicyGate
            from core.structured_log import jlog

            gate = PolicyGate.from_config()
            gate.set_trading_halted(True)
            jlog('emergency_action', {
                'action': 'halt_trades',
                'halted': True,
                'event': event.message,
                'level': event.level.name
            })
            logger.error("All new trades HALTED by emergency protocol")
        except Exception as e:
            logger.error(f"Failed to halt trades via PolicyGate: {e}")
            # Fallback: activate kill switch
            logger.error("Falling back to kill switch activation")
            self._action_kill_switch(event)

    def _action_close_all(self, event: EmergencyEvent) -> None:
        """
        Close all positions via broker.

        IMPLEMENTED (2026-01-04): Now actually closes all positions.
        """
        logger.critical(f"Emergency CLOSE_ALL: {event.message}")
        try:
            from execution.broker_alpaca import AlpacaBroker
            from core.structured_log import jlog

            broker = AlpacaBroker()
            result = broker.close_all_positions()
            jlog('emergency_action', {
                'action': 'close_all_positions',
                'result': str(result),
                'event': event.message,
                'level': event.level.name
            })
            logger.critical(f"All positions CLOSED: {result}")
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            # Fallback: activate kill switch to prevent further damage
            logger.error("Falling back to kill switch activation")
            self._action_kill_switch(event)

    def _action_kill_switch(self, event: EmergencyEvent) -> None:
        """Activate kill switch."""
        logger.critical(f"Emergency KILL_SWITCH: {event.message}")
        self._activate_kill_switch(event.message)

    def _action_alert_only(self, event: EmergencyEvent) -> None:
        """Alert only, no action."""
        logger.info(f"Emergency ALERT: {event.message}")

    def _activate_kill_switch(self, reason: str) -> None:
        """Activate the kill switch."""
        try:
            with open(self.KILL_SWITCH_FILE, "w") as f:
                json.dump({
                    "activated_at": datetime.now().isoformat(),
                    "reason": reason,
                    "activated_by": "emergency_protocol",
                }, f, indent=2)
            logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        except Exception as e:
            logger.error(f"Failed to activate kill switch: {e}")

    def _load_state(self) -> None:
        """Load emergency state."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    self._current_level = EmergencyLevel(data.get("current_level", 0))
            except Exception as e:
                logger.warning(f"Failed to load emergency state: {e}")

    def _save_state(self) -> None:
        """Save emergency state."""
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump({
                    "current_level": self._current_level.value,
                    "active_emergencies": [e.to_dict() for e in self._active_emergencies],
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save emergency state: {e}")

    def _log_event(self, event: EmergencyEvent) -> None:
        """Log emergency event to history."""
        try:
            with open(self.HISTORY_FILE, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to log emergency event: {e}")

    def evaluate(self, context: Dict[str, Any]) -> List[EmergencyEvent]:
        """
        Evaluate context against emergency rules.

        Args:
            context: Current system state

        Returns:
            List of emergency events triggered
        """
        events = []
        max_level = EmergencyLevel.NONE

        for rule in self._rules:
            try:
                if rule.condition(context):
                    value = rule.get_value(context)
                    message = rule.message_template.format(
                        value=value,
                        threshold=rule.threshold
                    )

                    event = EmergencyEvent(
                        emergency_type=rule.emergency_type,
                        level=rule.level,
                        trigger_value=value,
                        threshold=rule.threshold,
                        action_taken=rule.action,
                        message=message,
                    )

                    events.append(event)

                    # Track max level
                    if rule.level.value > max_level.value:
                        max_level = rule.level

                    # Execute action
                    handler = self._action_handlers.get(rule.action)
                    if handler:
                        handler(event)

                    # Log event
                    self._log_event(event)

            except Exception as e:
                logger.error(f"Rule {rule.name} failed: {e}")

        # Update current level
        self._current_level = max_level
        self._active_emergencies = events
        self._save_state()

        return events

    def get_current_level(self) -> EmergencyLevel:
        """Get current emergency level."""
        return self._current_level

    def get_active_emergencies(self) -> List[EmergencyEvent]:
        """Get currently active emergencies."""
        return self._active_emergencies.copy()

    def is_trading_allowed(self) -> tuple[bool, str]:
        """Check if trading is currently allowed."""
        if self.KILL_SWITCH_FILE.exists():
            return False, "Kill switch active"

        if self._current_level.value >= EmergencyLevel.LEVEL_3.value:
            return False, f"Emergency level {self._current_level.name}"

        return True, "Trading allowed"

    def resolve_emergency(self, emergency_type: EmergencyType) -> None:
        """Mark an emergency as resolved."""
        for event in self._active_emergencies:
            if event.emergency_type == emergency_type:
                event.resolved = True
                event.resolved_at = datetime.now()

        # Recalculate level
        active = [e for e in self._active_emergencies if not e.resolved]
        if active:
            self._current_level = max(e.level for e in active)
        else:
            self._current_level = EmergencyLevel.NONE

        self._save_state()

    def deactivate_kill_switch(self, reason: str) -> bool:
        """Deactivate the kill switch."""
        if self.KILL_SWITCH_FILE.exists():
            try:
                self.KILL_SWITCH_FILE.unlink()
                logger.info(f"Kill switch deactivated: {reason}")
                return True
            except Exception as e:
                logger.error(f"Failed to deactivate kill switch: {e}")
                return False
        return True


# Singleton
_protocol: Optional[EmergencyProtocol] = None


def get_emergency_protocol() -> EmergencyProtocol:
    """Get or create singleton protocol."""
    global _protocol
    if _protocol is None:
        _protocol = EmergencyProtocol()
    return _protocol


if __name__ == "__main__":
    # Demo
    protocol = EmergencyProtocol()

    print("=== Emergency Protocol Demo ===\n")

    # Normal conditions
    print("--- Normal Conditions ---")
    context = {
        "daily_drawdown_pct": 0.005,
        "vix": 18,
        "broker_connected": True,
        "data_feed_healthy": True,
    }
    events = protocol.evaluate(context)
    print(f"Level: {protocol.get_current_level().name}")
    print(f"Trading allowed: {protocol.is_trading_allowed()}")

    # Elevated drawdown
    print("\n--- Elevated Drawdown ---")
    context["daily_drawdown_pct"] = 0.025
    events = protocol.evaluate(context)
    print(f"Level: {protocol.get_current_level().name}")
    print(f"Events: {len(events)}")
    for e in events:
        print(f"  - {e.message}")

    # Critical VIX
    print("\n--- Critical VIX ---")
    context["vix"] = 45
    events = protocol.evaluate(context)
    print(f"Level: {protocol.get_current_level().name}")
    allowed, reason = protocol.is_trading_allowed()
    print(f"Trading allowed: {allowed} ({reason})")
