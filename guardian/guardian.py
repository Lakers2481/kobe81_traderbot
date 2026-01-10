"""
Guardian - Central Autonomous Trading Oversight

The Guardian is the main orchestrator for 24/7 autonomous trading.
It brings together all monitoring, decision-making, and alerting systems.

Responsibilities:
- Monitor all system components continuously
- Make autonomous trading decisions
- Escalate critical issues
- Generate reports and alerts
- Execute emergency protocols
- Ensure trading safety at all times

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
import json
import threading
import time as time_module

from core.structured_log import get_logger
from .system_monitor import ComponentStatus, get_system_monitor
from .decision_engine import Decision, DecisionType, get_decision_engine
from .alert_manager import Alert, AlertPriority, get_alert_manager
from .daily_digest import DailyDigest
from .emergency_protocol import EmergencyLevel, get_emergency_protocol

logger = get_logger(__name__)


class GuardianMode(Enum):
    """Guardian operating modes."""
    ACTIVE = "active"           # Full trading mode
    MONITORING = "monitoring"   # Watch only, no trades
    MAINTENANCE = "maintenance" # System updates
    EMERGENCY = "emergency"     # Emergency procedures active
    STOPPED = "stopped"         # Guardian stopped


@dataclass
class GuardianState:
    """Current Guardian state."""
    mode: GuardianMode
    is_running: bool
    system_health: ComponentStatus
    emergency_level: EmergencyLevel
    trading_allowed: bool
    last_check: datetime
    check_count: int
    alerts_today: int
    decisions_today: int
    uptime_hours: float
    as_of: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "is_running": self.is_running,
            "system_health": self.system_health.value,
            "emergency_level": self.emergency_level.value,
            "trading_allowed": self.trading_allowed,
            "last_check": self.last_check.isoformat(),
            "check_count": self.check_count,
            "alerts_today": self.alerts_today,
            "decisions_today": self.decisions_today,
            "uptime_hours": self.uptime_hours,
            "as_of": self.as_of.isoformat(),
        }

    def to_summary(self) -> str:
        """Generate status summary."""
        health_badge = {
            ComponentStatus.HEALTHY: "[OK]",
            ComponentStatus.DEGRADED: "[!!]",
            ComponentStatus.UNHEALTHY: "[XX]",
            ComponentStatus.UNKNOWN: "[??]",
        }

        lines = [
            "=" * 50,
            "GUARDIAN STATUS",
            "=" * 50,
            "",
            f"Mode: {self.mode.value.upper()}",
            f"Running: {'YES' if self.is_running else 'NO'}",
            f"Health: {health_badge.get(self.system_health, '[??]')} {self.system_health.value}",
            f"Emergency Level: {self.emergency_level.name}",
            f"Trading Allowed: {'YES' if self.trading_allowed else 'NO'}",
            "",
            f"Last Check: {self.last_check.strftime('%H:%M:%S')}",
            f"Check Count: {self.check_count}",
            f"Alerts Today: {self.alerts_today}",
            f"Decisions Today: {self.decisions_today}",
            f"Uptime: {self.uptime_hours:.1f} hours",
            "=" * 50,
        ]

        return "\n".join(lines)


class Guardian:
    """
    Central autonomous trading oversight system.

    The Guardian runs continuously to:
    1. Monitor all systems
    2. Make decisions
    3. Handle emergencies
    4. Generate reports
    5. Alert when needed
    """

    STATE_FILE = Path("state/guardian/guardian.json")
    HEARTBEAT_FILE = Path("state/guardian/heartbeat.json")

    # Check intervals (seconds)
    FULL_CHECK_INTERVAL = 60        # Full check every minute
    QUICK_CHECK_INTERVAL = 10       # Quick check every 10 seconds
    HEARTBEAT_INTERVAL = 30         # Heartbeat every 30 seconds

    def __init__(self):
        """Initialize Guardian."""
        self._mode = GuardianMode.STOPPED
        self._is_running = False
        self._start_time: Optional[datetime] = None
        self._check_count = 0
        self._stop_event = threading.Event()

        # Components
        self._system_monitor = get_system_monitor()
        self._decision_engine = get_decision_engine()
        self._alert_manager = get_alert_manager()
        self._emergency_protocol = get_emergency_protocol()
        self._daily_digest = DailyDigest()

        # Ensure directories
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        self._load_state()

    def _load_state(self) -> None:
        """Load Guardian state."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    # Could restore state here if needed
            except Exception as e:
                logger.warning(f"Failed to load Guardian state: {e}")

    def _save_state(self) -> None:
        """Save Guardian state."""
        try:
            state = self.get_state()
            with open(self.STATE_FILE, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Guardian state: {e}")

    def _update_heartbeat(self) -> None:
        """Update heartbeat file."""
        try:
            with open(self.HEARTBEAT_FILE, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "mode": self._mode.value,
                    "is_running": self._is_running,
                    "check_count": self._check_count,
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update heartbeat: {e}")

    def _build_context(self) -> Dict[str, Any]:
        """Build context for decision making."""
        # Get system health
        health = self._system_monitor.check_all()

        # Get emergency status
        trading_allowed, _ = self._emergency_protocol.is_trading_allowed()

        # Get alert summary
        alert_summary = self._alert_manager.get_alert_summary()

        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": health.overall_status.value,
            "unhealthy_components": health.critical_issues,
            "warnings": health.warnings,
            "emergency_level": self._emergency_protocol.get_current_level().value,
            "trading_allowed": trading_allowed,
            "alerts_1h": alert_summary.get("alerts_1h", 0),
            "alerts_24h": alert_summary.get("alerts_24h", 0),
            "critical_alerts": alert_summary.get("critical_24h", 0),
            # Would add more context in production:
            # - Current positions
            # - P&L
            # - VIX level
            # - Broker status
        }

    def _full_check(self) -> None:
        """Perform full system check."""
        self._check_count += 1

        # 1. Check system health
        health = self._system_monitor.check_all()

        # 2. Build context
        context = self._build_context()

        # 3. Evaluate emergency rules
        emergencies = self._emergency_protocol.evaluate(context)

        # 4. Handle any emergencies
        for emergency in emergencies:
            self._alert_manager.send(Alert(
                priority=AlertPriority.CRITICAL if emergency.level.value >= 3 else AlertPriority.HIGH,
                title=f"Emergency: {emergency.emergency_type.value}",
                message=emergency.message,
                category="emergency",
                source="guardian",
            ))

        # 5. Update mode based on status
        if health.overall_status == ComponentStatus.UNHEALTHY:
            if self._mode != GuardianMode.EMERGENCY:
                self._mode = GuardianMode.EMERGENCY
                self._alert_manager.send_quick(
                    AlertPriority.CRITICAL,
                    "Guardian Mode Changed",
                    "Switched to EMERGENCY mode due to unhealthy components",
                    "guardian",
                )
        elif self._emergency_protocol.get_current_level().value >= 3:
            self._mode = GuardianMode.EMERGENCY
        elif health.overall_status == ComponentStatus.DEGRADED:
            self._mode = GuardianMode.MONITORING
        else:
            if self._mode == GuardianMode.EMERGENCY:
                # Recovered from emergency
                self._alert_manager.send_quick(
                    AlertPriority.INFO,
                    "Guardian Recovered",
                    "Switched back to ACTIVE mode",
                    "guardian",
                )
            self._mode = GuardianMode.ACTIVE

        # 6. Save state
        self._save_state()

        logger.debug(f"Full check #{self._check_count}: {self._mode.value}")

    def _quick_check(self) -> None:
        """Perform quick health check."""
        # Just update heartbeat and check kill switch
        self._update_heartbeat()

        # Check if kill switch was activated externally
        if Path("state/KILL_SWITCH").exists():
            if self._mode != GuardianMode.EMERGENCY:
                self._mode = GuardianMode.EMERGENCY
                self._alert_manager.send_quick(
                    AlertPriority.EMERGENCY,
                    "Kill Switch Activated",
                    "Trading halted by kill switch",
                    "guardian",
                )

    def _check_schedule(self) -> None:
        """Check for scheduled actions."""
        now = datetime.now()
        current_time = now.time()

        # Generate daily digest at 4:30 PM ET
        if current_time >= time(16, 30) and current_time < time(16, 31):
            # Only once per day
            digest_file = self._daily_digest.REPORTS_DIR / f"daily_{now.date().isoformat()}.json"
            if not digest_file.exists():
                try:
                    report = self._daily_digest.generate()
                    self._alert_manager.send(Alert(
                        priority=AlertPriority.INFO,
                        title="Daily Digest Ready",
                        message=report.to_telegram(),
                        category="report",
                        source="guardian",
                    ))
                except Exception as e:
                    logger.error(f"Failed to generate daily digest: {e}")

    def start(self) -> None:
        """Start the Guardian."""
        if self._is_running:
            logger.warning("Guardian already running")
            return

        self._is_running = True
        self._start_time = datetime.now()
        self._mode = GuardianMode.ACTIVE
        self._stop_event.clear()

        logger.info("Guardian started")

        self._alert_manager.send_quick(
            AlertPriority.INFO,
            "Guardian Started",
            "Autonomous trading oversight is now active",
            "guardian",
        )

        # Initial full check
        self._full_check()

    def stop(self) -> None:
        """Stop the Guardian."""
        self._is_running = False
        self._mode = GuardianMode.STOPPED
        self._stop_event.set()

        logger.info("Guardian stopped")

        self._alert_manager.send_quick(
            AlertPriority.HIGH,
            "Guardian Stopped",
            "Autonomous trading oversight has been stopped",
            "guardian",
        )

        self._save_state()

    def run_once(self) -> None:
        """Run a single check cycle (for testing)."""
        if not self._is_running:
            self.start()

        self._full_check()
        self._check_schedule()

    def run(self) -> None:
        """Run the Guardian continuously."""
        self.start()

        last_full_check = datetime.now()
        last_quick_check = datetime.now()

        try:
            while self._is_running and not self._stop_event.is_set():
                now = datetime.now()

                # Full check
                if (now - last_full_check).total_seconds() >= self.FULL_CHECK_INTERVAL:
                    self._full_check()
                    self._check_schedule()
                    last_full_check = now

                # Quick check
                elif (now - last_quick_check).total_seconds() >= self.QUICK_CHECK_INTERVAL:
                    self._quick_check()
                    last_quick_check = now

                # Sleep briefly
                time_module.sleep(1)

        except KeyboardInterrupt:
            logger.info("Guardian interrupted")
        finally:
            self.stop()

    def get_state(self) -> GuardianState:
        """Get current Guardian state."""
        health = self._system_monitor.check_all()
        alert_summary = self._alert_manager.get_alert_summary()
        trading_allowed, _ = self._emergency_protocol.is_trading_allowed()

        uptime = 0.0
        if self._start_time:
            uptime = (datetime.now() - self._start_time).total_seconds() / 3600

        return GuardianState(
            mode=self._mode,
            is_running=self._is_running,
            system_health=health.overall_status,
            emergency_level=self._emergency_protocol.get_current_level(),
            trading_allowed=trading_allowed,
            last_check=datetime.now(),
            check_count=self._check_count,
            alerts_today=alert_summary.get("alerts_24h", 0),
            decisions_today=len(self._decision_engine.get_decision_history(24)),
            uptime_hours=uptime,
        )

    def decide_trade(self, signal: Dict[str, Any]) -> Decision:
        """Make a decision about a trade signal."""
        if not self._is_running:
            return Decision(
                decision_type=DecisionType.SKIP_TRADE,
                outcome="rejected",
                escalation="none",
                reason="Guardian not running",
                context={"signal": signal},
                confidence=1.0,
                impact="LOW",
                recommended_action="Wait for Guardian to start",
                alternative_actions=[],
            )

        # Build system state
        context = self._build_context()

        # Use decision engine
        return self._decision_engine.decide_trade(signal, context)

    def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status."""
        trading_allowed, reason = self._emergency_protocol.is_trading_allowed()

        return {
            "trading_allowed": trading_allowed,
            "reason": reason,
            "mode": self._mode.value,
            "emergency_level": self._emergency_protocol.get_current_level().value,
            "is_guardian_running": self._is_running,
        }


# Singleton
_guardian: Optional[Guardian] = None


def get_guardian() -> Guardian:
    """Get or create singleton Guardian."""
    global _guardian
    if _guardian is None:
        _guardian = Guardian()
    return _guardian


if __name__ == "__main__":
    # Demo
    guardian = Guardian()

    print("=== Guardian Demo ===\n")

    # Start and run once
    guardian.run_once()

    # Get state
    state = guardian.get_state()
    print(state.to_summary())

    # Check trading status
    print("\n--- Trading Status ---")
    status = guardian.get_trading_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Make a trade decision
    print("\n--- Trade Decision ---")
    signal = {
        "symbol": "AAPL",
        "side": "long",
        "entry_price": 175,
        "stop_loss": 170,
        "quality_score": 80,
    }
    decision = guardian.decide_trade(signal)
    print(f"  Decision: {decision.decision_type.value}")
    print(f"  Action: {decision.recommended_action}")

    # Stop
    guardian.stop()
