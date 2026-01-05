"""
Alert Manager - Smart Alerting System

Manages alerts with deduplication, prioritization, and delivery.

Features:
- Priority-based routing
- Deduplication (don't spam same alert)
- Cooldown periods
- Multiple channels (console, file, future: telegram)

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import json
import hashlib

from core.structured_log import get_logger

logger = get_logger(__name__)


class AlertPriority(Enum):
    """Alert priority levels."""
    DEBUG = 0           # Development only
    INFO = 1            # FYI - no action needed
    LOW = 2             # Can wait until EOD
    MEDIUM = 3          # Check within 1 hour
    HIGH = 4            # Check within 15 minutes
    CRITICAL = 5        # Immediate attention required
    EMERGENCY = 6       # Trading halted, check NOW


class AlertChannel(Enum):
    """Alert delivery channels."""
    CONSOLE = "console"
    FILE = "file"
    TELEGRAM = "telegram"
    EMAIL = "email"


@dataclass
class Alert:
    """Single alert."""
    priority: AlertPriority
    title: str
    message: str
    category: str
    source: str
    context: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority": self.priority.value,
            "priority_name": self.priority.name,
            "title": self.title,
            "message": self.message,
            "category": self.category,
            "source": self.source,
            "context": self.context,
            "actions": self.actions,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "timestamp": self.timestamp.isoformat(),
        }

    @property
    def hash(self) -> str:
        """Get unique hash for deduplication."""
        content = f"{self.title}:{self.message}:{self.category}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def format_console(self) -> str:
        """Format for console output."""
        priority_badges = {
            AlertPriority.DEBUG: "[DBG]",
            AlertPriority.INFO: "[INF]",
            AlertPriority.LOW: "[LOW]",
            AlertPriority.MEDIUM: "[MED]",
            AlertPriority.HIGH: "[HI!]",
            AlertPriority.CRITICAL: "[!!!]",
            AlertPriority.EMERGENCY: "[SOS]",
        }

        badge = priority_badges.get(self.priority, "[???]")
        time_str = self.timestamp.strftime("%H:%M:%S")

        lines = [
            f"{badge} [{time_str}] {self.title}",
            f"    {self.message}",
        ]

        if self.actions:
            lines.append("    Actions:")
            for action in self.actions:
                lines.append(f"      - {action}")

        return "\n".join(lines)

    def format_telegram(self) -> str:
        """Format for Telegram message."""
        priority_emoji = {
            AlertPriority.DEBUG: "bug",
            AlertPriority.INFO: "info",
            AlertPriority.LOW: "blue_circle",
            AlertPriority.MEDIUM: "yellow_circle",
            AlertPriority.HIGH: "orange_circle",
            AlertPriority.CRITICAL: "red_circle",
            AlertPriority.EMERGENCY: "rotating_light",
        }

        lines = [
            f"**{self.priority.name}: {self.title}**",
            "",
            self.message,
        ]

        if self.actions:
            lines.append("")
            lines.append("_Suggested Actions:_")
            for action in self.actions:
                lines.append(f"- {action}")

        return "\n".join(lines)


class AlertManager:
    """
    Manage system alerts.

    Features:
    - Priority routing
    - Deduplication
    - Rate limiting
    - Multi-channel delivery
    """

    STATE_FILE = Path("state/guardian/alerts.json")
    HISTORY_FILE = Path("state/guardian/alert_history.jsonl")

    # Cooldown periods by priority (seconds)
    COOLDOWN = {
        AlertPriority.DEBUG: 60,
        AlertPriority.INFO: 300,
        AlertPriority.LOW: 600,
        AlertPriority.MEDIUM: 300,
        AlertPriority.HIGH: 60,
        AlertPriority.CRITICAL: 30,
        AlertPriority.EMERGENCY: 0,  # No cooldown
    }

    # Default channels by priority
    DEFAULT_CHANNELS = {
        AlertPriority.DEBUG: [AlertChannel.FILE],
        AlertPriority.INFO: [AlertChannel.FILE, AlertChannel.CONSOLE],
        AlertPriority.LOW: [AlertChannel.FILE, AlertChannel.CONSOLE],
        AlertPriority.MEDIUM: [AlertChannel.FILE, AlertChannel.CONSOLE],
        AlertPriority.HIGH: [AlertChannel.FILE, AlertChannel.CONSOLE, AlertChannel.TELEGRAM],
        AlertPriority.CRITICAL: [AlertChannel.FILE, AlertChannel.CONSOLE, AlertChannel.TELEGRAM],
        AlertPriority.EMERGENCY: [AlertChannel.FILE, AlertChannel.CONSOLE, AlertChannel.TELEGRAM],
    }

    def __init__(self):
        """Initialize alert manager."""
        self._alert_history: List[Alert] = []
        self._alert_hashes: Dict[str, datetime] = {}  # hash -> last sent time
        self._channel_handlers: Dict[AlertChannel, Callable] = {}

        # Ensure directories
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Register default handlers
        self._channel_handlers[AlertChannel.CONSOLE] = self._send_console
        self._channel_handlers[AlertChannel.FILE] = self._send_file
        self._channel_handlers[AlertChannel.TELEGRAM] = self._send_telegram

        self._load_state()

    def _load_state(self) -> None:
        """Load alert state (for deduplication)."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    self._alert_hashes = {
                        h: datetime.fromisoformat(t)
                        for h, t in data.get("hashes", {}).items()
                    }
            except Exception as e:
                logger.warning(f"Failed to load alert state: {e}")

    def _save_state(self) -> None:
        """Save alert state."""
        try:
            # Clean old hashes (> 24h)
            cutoff = datetime.now() - timedelta(hours=24)
            self._alert_hashes = {
                h: t for h, t in self._alert_hashes.items()
                if t > cutoff
            }

            with open(self.STATE_FILE, "w") as f:
                json.dump({
                    "hashes": {h: t.isoformat() for h, t in self._alert_hashes.items()},
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alert state: {e}")

    def _should_send(self, alert: Alert) -> bool:
        """Check if alert should be sent (deduplication)."""
        last_sent = self._alert_hashes.get(alert.hash)

        if last_sent is None:
            return True

        cooldown = self.COOLDOWN.get(alert.priority, 300)
        elapsed = (datetime.now() - last_sent).total_seconds()

        return elapsed >= cooldown

    def _send_console(self, alert: Alert) -> bool:
        """Send to console."""
        try:
            print(alert.format_console())
            return True
        except Exception as e:
            logger.error(f"Console send failed: {e}")
            return False

    def _send_file(self, alert: Alert) -> bool:
        """Send to file."""
        try:
            with open(self.HISTORY_FILE, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
            return True
        except Exception as e:
            logger.error(f"File send failed: {e}")
            return False

    def _send_telegram(self, alert: Alert) -> bool:
        """Send to Telegram (placeholder - would integrate with telegram bot)."""
        try:
            # Would call telegram API here
            logger.info(f"Telegram alert: {alert.title}")
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def send(
        self,
        alert: Alert,
        force: bool = False,
    ) -> bool:
        """
        Send an alert through appropriate channels.

        Args:
            alert: Alert to send
            force: Bypass deduplication

        Returns:
            True if alert was sent
        """
        # Check deduplication
        if not force and not self._should_send(alert):
            logger.debug(f"Alert deduplicated: {alert.title}")
            return False

        # Get channels for priority
        channels = self.DEFAULT_CHANNELS.get(
            alert.priority,
            [AlertChannel.CONSOLE, AlertChannel.FILE]
        )

        # Send to each channel
        success = False
        for channel in channels:
            handler = self._channel_handlers.get(channel)
            if handler:
                try:
                    if handler(alert):
                        success = True
                except Exception as e:
                    logger.error(f"Channel {channel} failed: {e}")

        # Update tracking
        if success:
            self._alert_hashes[alert.hash] = datetime.now()
            self._alert_history.append(alert)
            self._save_state()

        return success

    def send_quick(
        self,
        priority: AlertPriority,
        title: str,
        message: str,
        category: str = "general",
    ) -> bool:
        """Quick helper to send an alert."""
        alert = Alert(
            priority=priority,
            title=title,
            message=message,
            category=category,
            source="quick_send",
        )
        return self.send(alert)

    def get_recent_alerts(
        self,
        hours: int = 24,
        priority: Optional[AlertPriority] = None,
    ) -> List[Alert]:
        """Get recent alerts."""
        cutoff = datetime.now() - timedelta(hours=hours)

        alerts = [a for a in self._alert_history if a.timestamp > cutoff]

        if priority:
            alerts = [a for a in alerts if a.priority.value >= priority.value]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert activity."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(hours=24)

        alerts_1h = [a for a in self._alert_history if a.timestamp > hour_ago]
        alerts_24h = [a for a in self._alert_history if a.timestamp > day_ago]

        by_priority_24h = {}
        for a in alerts_24h:
            by_priority_24h[a.priority.name] = by_priority_24h.get(a.priority.name, 0) + 1

        return {
            "alerts_1h": len(alerts_1h),
            "alerts_24h": len(alerts_24h),
            "by_priority_24h": by_priority_24h,
            "critical_24h": sum(1 for a in alerts_24h if a.priority.value >= AlertPriority.CRITICAL.value),
            "last_alert": alerts_24h[0].to_dict() if alerts_24h else None,
        }


# Singleton
_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create singleton manager."""
    global _manager
    if _manager is None:
        _manager = AlertManager()
    return _manager


if __name__ == "__main__":
    # Demo
    manager = AlertManager()

    print("=== Alert Manager Demo ===\n")

    # Send various alerts
    manager.send_quick(
        AlertPriority.INFO,
        "System Started",
        "Guardian system initialized successfully",
        "system",
    )

    manager.send_quick(
        AlertPriority.HIGH,
        "High VIX Detected",
        "VIX at 32 - position sizes reduced",
        "risk",
    )

    manager.send(Alert(
        priority=AlertPriority.CRITICAL,
        title="Circuit Breaker Tripped",
        message="Daily drawdown limit reached (-2.1%)",
        category="risk",
        source="circuit_breaker",
        actions=[
            "Review positions",
            "Consider manual intervention",
            "Check for news catalyst",
        ],
    ))

    # Summary
    print("\n--- Alert Summary ---")
    summary = manager.get_alert_summary()
    print(f"Alerts (1h): {summary['alerts_1h']}")
    print(f"Alerts (24h): {summary['alerts_24h']}")
    print(f"Critical (24h): {summary['critical_24h']}")
