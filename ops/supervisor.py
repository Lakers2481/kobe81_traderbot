"""
Self-healing supervisor for KOBE81.

Watches system health and automatically recovers from failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import threading
import time

logger = logging.getLogger(__name__)


class SupervisorMode(Enum):
    """Supervisor operating modes."""
    NORMAL = auto()      # All systems go
    DEGRADED = auto()    # Some issues, proceeding with caution
    SAFE_MODE = auto()   # Major issues, no trading


class ComponentStatus(Enum):
    """Status of a monitored component."""
    HEALTHY = auto()
    WARNING = auto()
    ERROR = auto()
    UNKNOWN = auto()


@dataclass
class HealthCheck:
    """Result of a health check."""
    component: str
    status: ComponentStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SupervisorState:
    """Current supervisor state."""
    mode: SupervisorMode
    last_check: Optional[datetime]
    components: Dict[str, ComponentStatus]
    error_counts: Dict[str, int]
    last_restart: Dict[str, datetime]
    safe_mode_reason: Optional[str]
    safe_mode_since: Optional[datetime]


class Supervisor:
    """
    Self-healing supervisor for the trading system.

    Monitors:
    - Component health (runner, broker, data)
    - Error rates
    - Reconciliation failures
    - System resources

    Actions:
    - Restart failed components
    - Activate safe mode on critical failures
    - Log and alert on issues
    """

    # Thresholds
    MAX_ERROR_COUNT = 5          # Errors before safe mode
    ERROR_WINDOW_MINUTES = 15    # Window for counting errors
    MAX_RESTART_ATTEMPTS = 3     # Max restarts per component per hour
    STALE_HEARTBEAT_SECONDS = 300  # 5 minutes
    CHECK_INTERVAL_SECONDS = 60  # How often to check

    def __init__(
        self,
        check_interval: int = 60,
        state_path: Optional[Path] = None,
    ):
        self.check_interval = check_interval
        self.state_path = state_path or Path("state/supervisor_state.json")

        # State
        self.mode = SupervisorMode.NORMAL
        self.error_counts: Dict[str, List[datetime]] = {}
        self.restart_counts: Dict[str, List[datetime]] = {}
        self.last_check: Optional[datetime] = None
        self.safe_mode_reason: Optional[str] = None
        self.safe_mode_since: Optional[datetime] = None

        # Component status
        self.component_status: Dict[str, ComponentStatus] = {}

        # Shutdown flag
        self._shutdown = False
        self._thread: Optional[threading.Thread] = None

        # Load persisted state
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.mode = SupervisorMode[data.get("mode", "NORMAL")]
                self.safe_mode_reason = data.get("safe_mode_reason")

                if data.get("safe_mode_since"):
                    self.safe_mode_since = datetime.fromisoformat(data["safe_mode_since"])

            except Exception as e:
                logger.warning(f"Could not load supervisor state: {e}")

    def _save_state(self) -> None:
        """Persist state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "mode": self.mode.name,
            "safe_mode_reason": self.safe_mode_reason,
            "safe_mode_since": self.safe_mode_since.isoformat() if self.safe_mode_since else None,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "component_status": {k: v.name for k, v in self.component_status.items()},
        }

        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def check_health(self) -> Dict[str, HealthCheck]:
        """Run all health checks."""
        checks = {}

        # Check runner heartbeat
        checks["runner"] = self._check_runner_heartbeat()

        # Check broker connection
        checks["broker"] = self._check_broker_connection()

        # Check data freshness
        checks["data"] = self._check_data_freshness()

        # Check reconciliation
        checks["reconcile"] = self._check_reconciliation()

        # Check disk space
        checks["disk"] = self._check_disk_space()

        # Check kill switch
        checks["kill_switch"] = self._check_kill_switch()

        # Update component status
        for name, check in checks.items():
            self.component_status[name] = check.status

        self.last_check = datetime.utcnow()
        return checks

    def _check_runner_heartbeat(self) -> HealthCheck:
        """Check if runner is alive via heartbeat."""
        try:
            from monitor.heartbeat import get_heartbeat_age, is_heartbeat_stale

            age = get_heartbeat_age()

            if age is None:
                return HealthCheck(
                    component="runner",
                    status=ComponentStatus.WARNING,
                    message="No heartbeat file found",
                )

            if is_heartbeat_stale():
                return HealthCheck(
                    component="runner",
                    status=ComponentStatus.ERROR,
                    message=f"Heartbeat stale ({age:.0f}s old)",
                    metadata={"age_seconds": age},
                )

            return HealthCheck(
                component="runner",
                status=ComponentStatus.HEALTHY,
                message=f"Heartbeat fresh ({age:.0f}s old)",
                metadata={"age_seconds": age},
            )

        except ImportError:
            return HealthCheck(
                component="runner",
                status=ComponentStatus.UNKNOWN,
                message="Heartbeat module not available",
            )
        except Exception as e:
            return HealthCheck(
                component="runner",
                status=ComponentStatus.ERROR,
                message=f"Heartbeat check failed: {e}",
            )

    def _check_broker_connection(self) -> HealthCheck:
        """Check broker API connection."""
        try:
            from execution.broker_alpaca import probe_broker

            success, message = probe_broker()

            if success:
                return HealthCheck(
                    component="broker",
                    status=ComponentStatus.HEALTHY,
                    message=message,
                )
            else:
                return HealthCheck(
                    component="broker",
                    status=ComponentStatus.ERROR,
                    message=message,
                )

        except ImportError:
            return HealthCheck(
                component="broker",
                status=ComponentStatus.UNKNOWN,
                message="Broker module not available",
            )
        except Exception as e:
            return HealthCheck(
                component="broker",
                status=ComponentStatus.ERROR,
                message=f"Broker check failed: {e}",
            )

    def _check_data_freshness(self) -> HealthCheck:
        """Check if market data is fresh."""
        try:
            # Check for recent data files
            data_dir = Path("data/prices")
            if not data_dir.exists():
                return HealthCheck(
                    component="data",
                    status=ComponentStatus.WARNING,
                    message="Data directory not found",
                )

            # Find most recent parquet file
            parquet_files = list(data_dir.glob("*.parquet"))
            if not parquet_files:
                return HealthCheck(
                    component="data",
                    status=ComponentStatus.WARNING,
                    message="No data files found",
                )

            newest = max(parquet_files, key=lambda p: p.stat().st_mtime)
            age_hours = (time.time() - newest.stat().st_mtime) / 3600

            if age_hours > 48:  # More than 2 days old
                return HealthCheck(
                    component="data",
                    status=ComponentStatus.WARNING,
                    message=f"Data is {age_hours:.1f} hours old",
                    metadata={"age_hours": age_hours},
                )

            return HealthCheck(
                component="data",
                status=ComponentStatus.HEALTHY,
                message=f"Data is {age_hours:.1f} hours old",
                metadata={"age_hours": age_hours},
            )

        except Exception as e:
            return HealthCheck(
                component="data",
                status=ComponentStatus.ERROR,
                message=f"Data check failed: {e}",
            )

    def _check_reconciliation(self) -> HealthCheck:
        """Check last reconciliation status."""
        try:
            reconcile_dir = Path("reports/reconcile")
            if not reconcile_dir.exists():
                return HealthCheck(
                    component="reconcile",
                    status=ComponentStatus.WARNING,
                    message="No reconciliation reports found",
                )

            # Find most recent report
            reports = sorted(reconcile_dir.glob("*.json"), reverse=True)
            if not reports:
                return HealthCheck(
                    component="reconcile",
                    status=ComponentStatus.WARNING,
                    message="No reconciliation reports found",
                )

            with open(reports[0], "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("is_clean", True):
                return HealthCheck(
                    component="reconcile",
                    status=ComponentStatus.HEALTHY,
                    message="Last reconciliation clean",
                )
            else:
                critical = data.get("summary", {}).get("critical", 0)
                return HealthCheck(
                    component="reconcile",
                    status=ComponentStatus.ERROR,
                    message=f"Reconciliation has {critical} critical issues",
                    metadata={"critical_issues": critical},
                )

        except Exception as e:
            return HealthCheck(
                component="reconcile",
                status=ComponentStatus.WARNING,
                message=f"Reconcile check failed: {e}",
            )

    def _check_disk_space(self) -> HealthCheck:
        """Check available disk space."""
        try:
            import shutil

            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024 ** 3)
            free_pct = (free / total) * 100

            if free_gb < 1:  # Less than 1 GB
                return HealthCheck(
                    component="disk",
                    status=ComponentStatus.ERROR,
                    message=f"Low disk space: {free_gb:.1f} GB free",
                    metadata={"free_gb": free_gb, "free_pct": free_pct},
                )
            elif free_pct < 10:  # Less than 10%
                return HealthCheck(
                    component="disk",
                    status=ComponentStatus.WARNING,
                    message=f"Disk space warning: {free_pct:.1f}% free",
                    metadata={"free_gb": free_gb, "free_pct": free_pct},
                )

            return HealthCheck(
                component="disk",
                status=ComponentStatus.HEALTHY,
                message=f"Disk space OK: {free_gb:.1f} GB free",
                metadata={"free_gb": free_gb, "free_pct": free_pct},
            )

        except Exception as e:
            return HealthCheck(
                component="disk",
                status=ComponentStatus.WARNING,
                message=f"Disk check failed: {e}",
            )

    def _check_kill_switch(self) -> HealthCheck:
        """Check if kill switch is active."""
        kill_switch_path = Path("state/KILL_SWITCH")

        if kill_switch_path.exists():
            return HealthCheck(
                component="kill_switch",
                status=ComponentStatus.WARNING,
                message="Kill switch is ACTIVE - trading halted",
            )

        return HealthCheck(
            component="kill_switch",
            status=ComponentStatus.HEALTHY,
            message="Kill switch not active",
        )

    def check_last_run_age(self) -> bool:
        """Check if last successful run was recent enough."""
        # Check runner log for last successful scan
        log_path = Path("logs/runner.log")
        if not log_path.exists():
            return True  # No log = assume OK

        try:
            age_hours = (time.time() - log_path.stat().st_mtime) / 3600
            return age_hours < 24  # Log should be touched at least daily
        except Exception:
            return True

    def check_error_rates(self) -> bool:
        """Check if error rates are acceptable."""
        now = datetime.utcnow()
        window = timedelta(minutes=self.ERROR_WINDOW_MINUTES)

        for component, errors in self.error_counts.items():
            # Filter to recent errors
            recent = [e for e in errors if now - e < window]
            self.error_counts[component] = recent

            if len(recent) >= self.MAX_ERROR_COUNT:
                return False

        return True

    def record_error(self, component: str) -> None:
        """Record an error for a component."""
        if component not in self.error_counts:
            self.error_counts[component] = []
        self.error_counts[component].append(datetime.utcnow())

        # Check if we should enter safe mode
        if not self.check_error_rates():
            self.activate_safe_mode(f"Too many errors in {component}")

    def maybe_restart_component(self, component: str) -> bool:
        """
        Attempt to restart a failed component.

        Returns True if restart was attempted.
        """
        now = datetime.utcnow()

        # Check restart limits
        if component not in self.restart_counts:
            self.restart_counts[component] = []

        # Filter to last hour
        hour_ago = now - timedelta(hours=1)
        recent = [r for r in self.restart_counts[component] if r > hour_ago]
        self.restart_counts[component] = recent

        if len(recent) >= self.MAX_RESTART_ATTEMPTS:
            logger.warning(f"Max restart attempts reached for {component}")
            self.activate_safe_mode(f"Max restart attempts for {component}")
            return False

        # Attempt restart based on component
        if component == "runner":
            return self._restart_runner()
        elif component == "data":
            return self._refresh_data()

        return False

    def _restart_runner(self) -> bool:
        """Attempt to restart the runner process."""
        logger.info("Attempting to restart runner...")

        try:
            # This is a placeholder - actual implementation depends on
            # how the runner is managed (systemd, supervisord, etc.)
            script_path = Path("scripts/runner.py")
            if not script_path.exists():
                return False

            self.restart_counts.setdefault("runner", []).append(datetime.utcnow())

            # For now, just log - actual restart should be done by
            # external process manager
            logger.info("Runner restart signaled")
            return True

        except Exception as e:
            logger.error(f"Runner restart failed: {e}")
            return False

    def _refresh_data(self) -> bool:
        """Attempt to refresh stale data."""
        logger.info("Attempting to refresh data...")

        try:
            # Trigger data refresh script
            script_path = Path("scripts/prefetch_polygon_universe.py")
            if not script_path.exists():
                return False

            self.restart_counts.setdefault("data", []).append(datetime.utcnow())

            # This would actually run the refresh
            # subprocess.Popen([sys.executable, str(script_path)])

            logger.info("Data refresh signaled")
            return True

        except Exception as e:
            logger.error(f"Data refresh failed: {e}")
            return False

    def activate_safe_mode(self, reason: str) -> None:
        """Activate safe mode - halt all trading."""
        if self.mode == SupervisorMode.SAFE_MODE:
            return  # Already in safe mode

        logger.critical(f"ACTIVATING SAFE MODE: {reason}")

        self.mode = SupervisorMode.SAFE_MODE
        self.safe_mode_reason = reason
        self.safe_mode_since = datetime.utcnow()

        # Create kill switch
        kill_switch_path = Path("state/KILL_SWITCH")
        kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
        with open(kill_switch_path, "w") as f:
            f.write(f"SAFE_MODE: {reason}\n{datetime.utcnow().isoformat()}\n")

        self._save_state()

        # Log to compliance audit
        try:
            from compliance.audit_trail import write_event
            write_event("safe_mode_activated", {"reason": reason})
        except ImportError:
            pass

    def deactivate_safe_mode(self) -> None:
        """Deactivate safe mode - allow trading to resume."""
        if self.mode != SupervisorMode.SAFE_MODE:
            return

        logger.info("Deactivating safe mode")

        self.mode = SupervisorMode.NORMAL
        self.safe_mode_reason = None
        self.safe_mode_since = None

        # Remove kill switch
        kill_switch_path = Path("state/KILL_SWITCH")
        if kill_switch_path.exists():
            kill_switch_path.unlink()

        self._save_state()

        try:
            from compliance.audit_trail import write_event
            write_event("safe_mode_deactivated", {})
        except ImportError:
            pass

    def determine_mode(self, checks: Dict[str, HealthCheck]) -> SupervisorMode:
        """Determine operating mode based on health checks."""
        error_count = sum(
            1 for c in checks.values()
            if c.status == ComponentStatus.ERROR
        )
        warning_count = sum(
            1 for c in checks.values()
            if c.status == ComponentStatus.WARNING
        )

        # Critical components
        critical = ["broker", "reconcile"]
        critical_errors = any(
            checks.get(c, HealthCheck(c, ComponentStatus.UNKNOWN, "")).status == ComponentStatus.ERROR
            for c in critical
        )

        if critical_errors or error_count >= 3:
            return SupervisorMode.SAFE_MODE
        elif error_count >= 1 or warning_count >= 2:
            return SupervisorMode.DEGRADED
        else:
            return SupervisorMode.NORMAL

    def run_once(self) -> None:
        """Run one iteration of supervisor checks."""
        logger.debug("Running supervisor check...")

        # Run health checks
        checks = self.check_health()

        # Determine mode
        new_mode = self.determine_mode(checks)

        # Handle mode changes
        if new_mode == SupervisorMode.SAFE_MODE and self.mode != SupervisorMode.SAFE_MODE:
            # Build reason from failing checks
            failing = [c.component for c in checks.values() if c.status == ComponentStatus.ERROR]
            self.activate_safe_mode(f"Health check failures: {', '.join(failing)}")
        elif new_mode == SupervisorMode.NORMAL and self.mode == SupervisorMode.DEGRADED:
            logger.info("System recovered to normal mode")
            self.mode = new_mode
        elif new_mode == SupervisorMode.DEGRADED:
            if self.mode != SupervisorMode.DEGRADED:
                logger.warning("Entering degraded mode")
            self.mode = new_mode

        # Attempt recovery for failed components (if not in safe mode)
        if self.mode != SupervisorMode.SAFE_MODE:
            for name, check in checks.items():
                if check.status == ComponentStatus.ERROR:
                    self.record_error(name)
                    self.maybe_restart_component(name)

        self._save_state()

    def run_loop(self) -> None:
        """Main supervisor loop."""
        logger.info("Supervisor starting...")

        while not self._shutdown:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Supervisor error: {e}")

            # Sleep until next check
            for _ in range(self.check_interval):
                if self._shutdown:
                    break
                time.sleep(1)

        logger.info("Supervisor stopped")

    def start(self) -> None:
        """Start supervisor in background thread."""
        if self._thread and self._thread.is_alive():
            return

        self._shutdown = False
        self._thread = threading.Thread(target=self.run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop supervisor."""
        self._shutdown = True
        if self._thread:
            self._thread.join(timeout=5)

    def get_status(self) -> Dict[str, Any]:
        """Get current supervisor status."""
        return {
            "mode": self.mode.name,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "components": {k: v.name for k, v in self.component_status.items()},
            "safe_mode_reason": self.safe_mode_reason,
            "safe_mode_since": self.safe_mode_since.isoformat() if self.safe_mode_since else None,
        }


# Global supervisor instance
_supervisor: Optional[Supervisor] = None


def get_supervisor() -> Supervisor:
    """Get global supervisor instance."""
    global _supervisor
    if _supervisor is None:
        _supervisor = Supervisor()
    return _supervisor


def start_supervisor() -> None:
    """Start the global supervisor."""
    get_supervisor().start()


def stop_supervisor() -> None:
    """Stop the global supervisor."""
    if _supervisor:
        _supervisor.stop()
