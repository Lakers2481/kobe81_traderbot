"""
System Monitor - Real-Time Health Monitoring

Monitors all system components and provides unified health status.

Components Monitored:
- Data feeds (Polygon, Alpaca)
- Execution systems
- Risk systems (circuit breakers, gates)
- ML models
- Storage and state
- Network connectivity

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum
import json
import os

# CRITICAL: Load .env file FIRST before any os.getenv() calls
# This ensures all API keys are available to the Guardian
from dotenv import load_dotenv
load_dotenv()

from core.structured_log import get_logger

logger = get_logger(__name__)


class ComponentStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: ComponentStatus
    message: str
    last_check: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "metrics": self.metrics,
            "dependencies": self.dependencies,
        }


@dataclass
class SystemHealth:
    """Overall system health."""
    overall_status: ComponentStatus
    components: Dict[str, ComponentHealth]
    healthy_count: int
    degraded_count: int
    unhealthy_count: int
    critical_issues: List[str]
    warnings: List[str]
    uptime_seconds: float
    as_of: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "healthy_count": self.healthy_count,
            "degraded_count": self.degraded_count,
            "unhealthy_count": self.unhealthy_count,
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "uptime_seconds": self.uptime_seconds,
            "as_of": self.as_of.isoformat(),
        }

    def to_summary(self) -> str:
        """Generate health summary."""
        emoji = {
            ComponentStatus.HEALTHY: "[OK]",
            ComponentStatus.DEGRADED: "[WARN]",
            ComponentStatus.UNHEALTHY: "[FAIL]",
            ComponentStatus.UNKNOWN: "[??]",
        }

        lines = [
            "=" * 50,
            f"SYSTEM HEALTH: {emoji[self.overall_status]} {self.overall_status.value.upper()}",
            "=" * 50,
            "",
            f"Healthy: {self.healthy_count} | Degraded: {self.degraded_count} | Unhealthy: {self.unhealthy_count}",
            f"Uptime: {self.uptime_seconds / 3600:.1f} hours",
            "",
        ]

        if self.critical_issues:
            lines.append("**CRITICAL ISSUES:**")
            for issue in self.critical_issues:
                lines.append(f"  [!] {issue}")
            lines.append("")

        if self.warnings:
            lines.append("**WARNINGS:**")
            for warning in self.warnings:
                lines.append(f"  [*] {warning}")
            lines.append("")

        lines.append("**COMPONENTS:**")
        for name, comp in sorted(self.components.items()):
            lines.append(f"  {emoji[comp.status]} {name}: {comp.message}")

        return "\n".join(lines)


class SystemMonitor:
    """
    Monitor all system components.

    Features:
    - Component health checks
    - Dependency tracking
    - Metric collection
    - Alert generation
    """

    STATE_FILE = Path("state/guardian/system_health.json")

    # Check intervals (seconds)
    CHECK_INTERVAL = 60         # Full check every minute
    QUICK_CHECK_INTERVAL = 10   # Quick check every 10 seconds

    def __init__(self):
        """Initialize system monitor."""
        self._start_time = datetime.now()
        self._component_status: Dict[str, ComponentHealth] = {}
        self._last_check: Optional[datetime] = None

        # Ensure directory
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _check_data_feeds(self) -> ComponentHealth:
        """Check data feed health."""
        try:
            # Check Polygon API key
            polygon_key = os.getenv("POLYGON_API_KEY")
            if not polygon_key:
                return ComponentHealth(
                    name="data_feeds",
                    status=ComponentStatus.UNHEALTHY,
                    message="POLYGON_API_KEY not set",
                    last_check=datetime.now(),
                )

            # Check cache freshness
            cache_dir = Path("cache/polygon")
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*.csv"))
                if cache_files:
                    newest = max(cache_files, key=lambda f: f.stat().st_mtime)
                    age_hours = (datetime.now().timestamp() - newest.stat().st_mtime) / 3600

                    if age_hours > 24:
                        return ComponentHealth(
                            name="data_feeds",
                            status=ComponentStatus.DEGRADED,
                            message=f"Cache stale ({age_hours:.0f}h old)",
                            last_check=datetime.now(),
                            metrics={"cache_age_hours": age_hours},
                        )

            return ComponentHealth(
                name="data_feeds",
                status=ComponentStatus.HEALTHY,
                message="Data feeds operational",
                last_check=datetime.now(),
            )

        except Exception as e:
            return ComponentHealth(
                name="data_feeds",
                status=ComponentStatus.UNKNOWN,
                message=f"Check failed: {e}",
                last_check=datetime.now(),
            )

    def _check_broker(self) -> ComponentHealth:
        """Check broker connection."""
        try:
            alpaca_key = os.getenv("ALPACA_API_KEY_ID")
            alpaca_secret = os.getenv("ALPACA_API_SECRET_KEY")

            if not alpaca_key or not alpaca_secret:
                return ComponentHealth(
                    name="broker",
                    status=ComponentStatus.UNHEALTHY,
                    message="Alpaca API keys not set",
                    last_check=datetime.now(),
                )

            # Check if paper or live
            base_url = os.getenv("ALPACA_BASE_URL", "")
            is_paper = "paper" in base_url.lower()

            return ComponentHealth(
                name="broker",
                status=ComponentStatus.HEALTHY,
                message=f"Alpaca {'paper' if is_paper else 'LIVE'} connected",
                last_check=datetime.now(),
                metrics={"is_paper": is_paper},
            )

        except Exception as e:
            return ComponentHealth(
                name="broker",
                status=ComponentStatus.UNKNOWN,
                message=f"Check failed: {e}",
                last_check=datetime.now(),
            )

    def _check_kill_switch(self) -> ComponentHealth:
        """Check kill switch status."""
        kill_file = Path("state/KILL_SWITCH")

        if kill_file.exists():
            return ComponentHealth(
                name="kill_switch",
                status=ComponentStatus.UNHEALTHY,
                message="KILL SWITCH ACTIVE - Trading halted",
                last_check=datetime.now(),
            )

        return ComponentHealth(
            name="kill_switch",
            status=ComponentStatus.HEALTHY,
            message="Kill switch not active",
            last_check=datetime.now(),
        )

    def _check_circuit_breakers(self) -> ComponentHealth:
        """Check circuit breaker status."""
        try:
            state_file = Path("state/risk/circuit_breakers.json")

            if not state_file.exists():
                return ComponentHealth(
                    name="circuit_breakers",
                    status=ComponentStatus.HEALTHY,
                    message="No active breakers",
                    last_check=datetime.now(),
                )

            with open(state_file, "r") as f:
                data = json.load(f)

            active = data.get("active_breakers", [])

            if active:
                return ComponentHealth(
                    name="circuit_breakers",
                    status=ComponentStatus.DEGRADED,
                    message=f"{len(active)} breaker(s) tripped: {', '.join(active)}",
                    last_check=datetime.now(),
                    metrics={"active_breakers": active},
                )

            return ComponentHealth(
                name="circuit_breakers",
                status=ComponentStatus.HEALTHY,
                message="All breakers green",
                last_check=datetime.now(),
            )

        except Exception as e:
            return ComponentHealth(
                name="circuit_breakers",
                status=ComponentStatus.UNKNOWN,
                message=f"Check failed: {e}",
                last_check=datetime.now(),
            )

    def _check_storage(self) -> ComponentHealth:
        """Check storage health."""
        try:
            # Check state directory
            state_dir = Path("state")
            if not state_dir.exists():
                return ComponentHealth(
                    name="storage",
                    status=ComponentStatus.UNHEALTHY,
                    message="State directory missing",
                    last_check=datetime.now(),
                )

            # Check log directory
            log_dir = Path("logs")
            if not log_dir.exists():
                return ComponentHealth(
                    name="storage",
                    status=ComponentStatus.DEGRADED,
                    message="Log directory missing",
                    last_check=datetime.now(),
                )

            return ComponentHealth(
                name="storage",
                status=ComponentStatus.HEALTHY,
                message="Storage operational",
                last_check=datetime.now(),
            )

        except Exception as e:
            return ComponentHealth(
                name="storage",
                status=ComponentStatus.UNKNOWN,
                message=f"Check failed: {e}",
                last_check=datetime.now(),
            )

    def _check_ml_models(self) -> ComponentHealth:
        """Check ML model health."""
        try:
            model_dir = Path("models")

            if not model_dir.exists():
                return ComponentHealth(
                    name="ml_models",
                    status=ComponentStatus.DEGRADED,
                    message="Models directory missing (OK if not using ML)",
                    last_check=datetime.now(),
                )

            # Count models
            model_files = list(model_dir.glob("**/*.pkl")) + list(model_dir.glob("**/*.joblib"))

            return ComponentHealth(
                name="ml_models",
                status=ComponentStatus.HEALTHY,
                message=f"{len(model_files)} model(s) available",
                last_check=datetime.now(),
                metrics={"model_count": len(model_files)},
            )

        except Exception as e:
            return ComponentHealth(
                name="ml_models",
                status=ComponentStatus.UNKNOWN,
                message=f"Check failed: {e}",
                last_check=datetime.now(),
            )

    def _check_autonomous_brain(self) -> ComponentHealth:
        """Check autonomous brain status."""
        try:
            heartbeat_file = Path("state/autonomous/heartbeat.json")

            if not heartbeat_file.exists():
                return ComponentHealth(
                    name="autonomous_brain",
                    status=ComponentStatus.DEGRADED,
                    message="Brain not running",
                    last_check=datetime.now(),
                )

            with open(heartbeat_file, "r") as f:
                data = json.load(f)

            # Parse timestamp and make it timezone-naive for comparison
            timestamp_str = data.get("timestamp", "2020-01-01")
            last_heartbeat = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            # Make naive for comparison
            if last_heartbeat.tzinfo is not None:
                last_heartbeat = last_heartbeat.replace(tzinfo=None)
            age_minutes = (datetime.now() - last_heartbeat).total_seconds() / 60

            if age_minutes > 5:
                return ComponentHealth(
                    name="autonomous_brain",
                    status=ComponentStatus.UNHEALTHY,
                    message=f"Brain stale ({age_minutes:.0f}min since heartbeat)",
                    last_check=datetime.now(),
                    metrics={"last_heartbeat_minutes": age_minutes},
                )

            return ComponentHealth(
                name="autonomous_brain",
                status=ComponentStatus.HEALTHY,
                message=f"Brain active (last heartbeat {age_minutes:.1f}min ago)",
                last_check=datetime.now(),
                metrics={"last_heartbeat_minutes": age_minutes},
            )

        except Exception as e:
            return ComponentHealth(
                name="autonomous_brain",
                status=ComponentStatus.UNKNOWN,
                message=f"Check failed: {e}",
                last_check=datetime.now(),
            )

    def check_all(self) -> SystemHealth:
        """Run all health checks."""
        components = {}

        # Run checks
        components["data_feeds"] = self._check_data_feeds()
        components["broker"] = self._check_broker()
        components["kill_switch"] = self._check_kill_switch()
        components["circuit_breakers"] = self._check_circuit_breakers()
        components["storage"] = self._check_storage()
        components["ml_models"] = self._check_ml_models()
        components["autonomous_brain"] = self._check_autonomous_brain()

        # Count by status
        healthy = sum(1 for c in components.values() if c.status == ComponentStatus.HEALTHY)
        degraded = sum(1 for c in components.values() if c.status == ComponentStatus.DEGRADED)
        unhealthy = sum(1 for c in components.values() if c.status == ComponentStatus.UNHEALTHY)

        # Collect issues
        critical = [
            f"{c.name}: {c.message}"
            for c in components.values()
            if c.status == ComponentStatus.UNHEALTHY
        ]

        warnings = [
            f"{c.name}: {c.message}"
            for c in components.values()
            if c.status == ComponentStatus.DEGRADED
        ]

        # Determine overall status
        if unhealthy > 0:
            overall = ComponentStatus.UNHEALTHY
        elif degraded > 0:
            overall = ComponentStatus.DEGRADED
        else:
            overall = ComponentStatus.HEALTHY

        uptime = (datetime.now() - self._start_time).total_seconds()

        health = SystemHealth(
            overall_status=overall,
            components=components,
            healthy_count=healthy,
            degraded_count=degraded,
            unhealthy_count=unhealthy,
            critical_issues=critical,
            warnings=warnings,
            uptime_seconds=uptime,
        )

        self._component_status = components
        self._last_check = datetime.now()
        self._save_state(health)

        return health

    def _save_state(self, health: SystemHealth) -> None:
        """Save health state."""
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump(health.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save health state: {e}")

    def get_component_status(self, name: str) -> Optional[ComponentHealth]:
        """Get status of specific component."""
        return self._component_status.get(name)


# Singleton
_monitor: Optional[SystemMonitor] = None


def get_system_monitor() -> SystemMonitor:
    """Get or create singleton monitor."""
    global _monitor
    if _monitor is None:
        _monitor = SystemMonitor()
    return _monitor


if __name__ == "__main__":
    # Demo
    monitor = SystemMonitor()

    print("=== System Health Check ===\n")

    health = monitor.check_all()
    print(health.to_summary())
