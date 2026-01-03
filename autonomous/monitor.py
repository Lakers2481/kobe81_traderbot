"""
Autonomous Brain Monitor.

Provides real-time monitoring of the 24/7 brain:
- Heartbeat monitoring
- Task execution history
- Research progress
- Learning metrics
- Alert generation
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class BrainMonitor:
    """Monitor the autonomous brain's health and activity."""

    def __init__(self, state_dir: Optional[Path] = None):
        if state_dir is None:
            state_dir = Path("state/autonomous")
        self.state_dir = state_dir

    def get_heartbeat(self) -> Dict[str, Any]:
        """Get latest heartbeat status."""
        heartbeat_file = self.state_dir / "heartbeat.json"

        if not heartbeat_file.exists():
            return {
                "status": "no_heartbeat",
                "message": "Brain has never run or heartbeat file missing",
            }

        try:
            data = json.loads(heartbeat_file.read_text())
            hb_time = datetime.fromisoformat(data["timestamp"])
            age_seconds = (datetime.now(ET) - hb_time).total_seconds()

            status = "healthy"
            if age_seconds > 300:  # 5 minutes
                status = "stale"
            elif age_seconds > 600:  # 10 minutes
                status = "dead"

            return {
                "status": status,
                "alive": data.get("alive", False),
                "last_heartbeat": data["timestamp"],
                "age_seconds": int(age_seconds),
                "phase": data.get("phase"),
                "work_mode": data.get("work_mode"),
                "cycles": data.get("cycles", 0),
                "uptime_hours": data.get("uptime_hours", 0),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_brain_state(self) -> Dict[str, Any]:
        """Get brain state."""
        state_file = self.state_dir / "brain_state.json"

        if not state_file.exists():
            return {"status": "no_state"}

        try:
            return json.loads(state_file.read_text())
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_research_progress(self) -> Dict[str, Any]:
        """Get research progress."""
        research_file = self.state_dir / "research/research_state.json"

        if not research_file.exists():
            return {"status": "no_research"}

        try:
            data = json.loads(research_file.read_text())
            experiments = data.get("experiments", [])
            discoveries = data.get("discoveries", [])

            # Calculate stats
            completed = [e for e in experiments if e.get("status") == "completed"]
            improvements = [
                e.get("improvement", 0) for e in completed
                if e.get("improvement") is not None
            ]

            return {
                "total_experiments": len(experiments),
                "completed_experiments": len(completed),
                "discoveries": len(discoveries),
                "avg_improvement": sum(improvements) / len(improvements) if improvements else 0,
                "best_improvement": max(improvements) if improvements else 0,
                "recent_experiments": experiments[-5:],
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_learning_progress(self) -> Dict[str, Any]:
        """Get learning progress."""
        learning_file = self.state_dir / "learning/learning_state.json"

        if not learning_file.exists():
            return {"status": "no_learning"}

        try:
            return json.loads(learning_file.read_text())
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_task_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent task execution history."""
        task_file = self.state_dir / "task_queue.json"

        if not task_file.exists():
            return []

        try:
            data = json.loads(task_file.read_text())
            tasks = data.get("tasks", [])

            # Get completed tasks with run history
            completed = [
                t for t in tasks
                if t.get("last_run") is not None
            ]

            # Sort by last run time
            completed.sort(key=lambda t: t.get("last_run", ""), reverse=True)

            return completed[:limit]

        except Exception as e:
            logger.error(f"Error getting task history: {e}")
            return []

    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for any alerts that need attention."""
        alerts = []

        # Check heartbeat
        hb = self.get_heartbeat()
        if hb["status"] == "stale":
            alerts.append({
                "level": "warning",
                "type": "heartbeat_stale",
                "message": f"Brain heartbeat is {hb['age_seconds']}s old",
            })
        elif hb["status"] == "dead":
            alerts.append({
                "level": "critical",
                "type": "heartbeat_dead",
                "message": "Brain appears to be dead",
            })
        elif hb["status"] == "no_heartbeat":
            alerts.append({
                "level": "info",
                "type": "no_heartbeat",
                "message": "Brain has not started yet",
            })

        # Check kill switch
        kill_switch = Path("state/KILL_SWITCH")
        if kill_switch.exists():
            alerts.append({
                "level": "critical",
                "type": "kill_switch",
                "message": "Kill switch is ACTIVE - trading halted",
            })

        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024 ** 3)
            if free_gb < 5:
                alerts.append({
                    "level": "warning",
                    "type": "low_disk",
                    "message": f"Low disk space: {free_gb:.1f}GB free",
                })
        except Exception:
            pass

        return alerts

    def get_full_status(self) -> Dict[str, Any]:
        """Get comprehensive monitor status."""
        return {
            "timestamp": datetime.now(ET).isoformat(),
            "heartbeat": self.get_heartbeat(),
            "brain_state": self.get_brain_state(),
            "research": self.get_research_progress(),
            "learning": self.get_learning_progress(),
            "recent_tasks": self.get_task_history(10),
            "alerts": self.check_alerts(),
        }


def print_status():
    """Print formatted status."""
    monitor = BrainMonitor()
    status = monitor.get_full_status()

    print(f"""
========================================================================
                    AUTONOMOUS BRAIN MONITOR
========================================================================
  Timestamp: {status['timestamp']}
""")

    # Heartbeat
    hb = status["heartbeat"]
    hb_icon = "[OK]" if hb["status"] == "healthy" else "[WARN]" if hb["status"] == "stale" else "[ERR]"
    print(f"""
Heartbeat Status: {hb_icon} {hb['status'].upper()}
--------------------------------------------------------
  Last Beat:    {hb.get('last_heartbeat', 'Never')}
  Age:          {hb.get('age_seconds', 'N/A')} seconds
  Cycles:       {hb.get('cycles', 0)}
  Uptime:       {hb.get('uptime_hours', 0):.1f} hours
  Phase:        {hb.get('phase', 'Unknown')}
  Work Mode:    {hb.get('work_mode', 'Unknown')}
""")

    # Research
    research = status["research"]
    if research.get("status") != "no_research":
        print(f"""
Research Progress
--------------------------------------------------------
  Experiments:  {research.get('total_experiments', 0)} total, {research.get('completed_experiments', 0)} completed
  Discoveries:  {research.get('discoveries', 0)}
  Avg Improve:  {research.get('avg_improvement', 0):+.1f}%
  Best Improve: {research.get('best_improvement', 0):+.1f}%
""")

    # Alerts
    alerts = status["alerts"]
    if alerts:
        print("""
Alerts
--------------------------------------------------------""")
        for alert in alerts:
            icon = "[CRIT]" if alert["level"] == "critical" else "[WARN]" if alert["level"] == "warning" else "[INFO]"
            print(f"  {icon} {alert['message']}")
    else:
        print("""
Alerts: None [OK]
--------------------------------------------------------""")

    # Recent tasks
    tasks = status["recent_tasks"]
    if tasks:
        print("""
Recent Tasks
--------------------------------------------------------""")
        for task in tasks[:5]:
            print(f"  - {task.get('name', 'Unknown')} - {task.get('last_run', 'Never')}")


if __name__ == "__main__":
    print_status()
