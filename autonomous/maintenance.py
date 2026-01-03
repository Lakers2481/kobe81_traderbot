"""
Autonomous Maintenance Tasks for Kobe.

Handles system upkeep:
- Data quality checks
- Log cleanup
- Health monitoring
- System optimization
"""

import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class MaintenanceEngine:
    """Handles all maintenance tasks."""

    def __init__(self):
        self.base_dir = Path(".")

    def check_data(self) -> Dict[str, Any]:
        """Check data quality and freshness."""
        logger.info("Checking data quality...")

        issues = []
        checks_passed = 0

        # Check 1: Universe file exists
        universe_file = self.base_dir / "data/universe/optionable_liquid_900.csv"
        if universe_file.exists():
            checks_passed += 1
        else:
            issues.append("Universe file missing")

        # Check 2: Cache directory
        cache_dir = self.base_dir / "data/cache"
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.csv"))
            if len(cache_files) > 0:
                checks_passed += 1

                # Check freshness
                newest = max(cache_files, key=lambda f: f.stat().st_mtime)
                age_days = (datetime.now() - datetime.fromtimestamp(newest.stat().st_mtime)).days
                if age_days > 7:
                    issues.append(f"Cache data is {age_days} days old")
            else:
                issues.append("No cached data files")
        else:
            issues.append("Cache directory missing")

        # Check 3: Models exist
        models_dir = self.base_dir / "models"
        required_models = ["hmm_regime_v1.pkl", "lstm_confidence_v1.h5", "ensemble_v1"]
        for model in required_models:
            if (models_dir / model).exists():
                checks_passed += 1
            else:
                issues.append(f"Model missing: {model}")

        # Check 4: State directory
        state_dir = self.base_dir / "state"
        if state_dir.exists():
            checks_passed += 1
        else:
            issues.append("State directory missing")

        return {
            "status": "healthy" if not issues else "issues_found",
            "checks_passed": checks_passed,
            "issues": issues,
            "timestamp": datetime.now(ET).isoformat(),
        }

    def cleanup(self) -> Dict[str, Any]:
        """Clean up old logs and temporary files."""
        logger.info("Running cleanup...")

        cleaned = {
            "logs_removed": 0,
            "cache_cleaned_mb": 0,
            "temp_removed": 0,
        }

        try:
            # Clean old logs (> 30 days)
            logs_dir = self.base_dir / "logs"
            if logs_dir.exists():
                cutoff = datetime.now() - timedelta(days=30)
                for log_file in logs_dir.glob("*.jsonl"):
                    if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff:
                        log_file.unlink()
                        cleaned["logs_removed"] += 1

                for log_file in logs_dir.glob("*.log"):
                    if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff:
                        log_file.unlink()
                        cleaned["logs_removed"] += 1

            # Clean old cache files (> 60 days)
            cache_dir = self.base_dir / "data/cache"
            if cache_dir.exists():
                cutoff = datetime.now() - timedelta(days=60)
                for cache_file in cache_dir.glob("*.csv"):
                    if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff:
                        size_mb = cache_file.stat().st_size / (1024 * 1024)
                        cache_file.unlink()
                        cleaned["cache_cleaned_mb"] += size_mb

            # Clean temp files
            for pattern in ["temp_*.csv", "*.tmp", "*.bak"]:
                for temp_file in self.base_dir.glob(pattern):
                    temp_file.unlink()
                    cleaned["temp_removed"] += 1

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return {"status": "error", "error": str(e)}

        cleaned["cache_cleaned_mb"] = round(cleaned["cache_cleaned_mb"], 2)
        return {
            "status": "success",
            **cleaned,
            "timestamp": datetime.now(ET).isoformat(),
        }

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        logger.info("Running health check...")

        health = {
            "overall": "healthy",
            "components": {},
            "warnings": [],
            "errors": [],
        }

        # Check 1: Kill switch
        kill_switch = self.base_dir / "state/KILL_SWITCH"
        health["components"]["kill_switch"] = "ACTIVE" if kill_switch.exists() else "inactive"
        if kill_switch.exists():
            health["warnings"].append("Kill switch is active!")
            health["overall"] = "degraded"

        # Check 2: Disk space
        try:
            total, used, free = shutil.disk_usage(self.base_dir)
            free_gb = free / (1024 ** 3)
            health["components"]["disk_free_gb"] = round(free_gb, 2)
            if free_gb < 5:
                health["warnings"].append(f"Low disk space: {free_gb:.1f}GB")
        except Exception:
            health["components"]["disk_free_gb"] = "unknown"

        # Check 3: Required directories
        required_dirs = ["state", "logs", "data", "models", "config"]
        for dir_name in required_dirs:
            dir_path = self.base_dir / dir_name
            health["components"][f"dir_{dir_name}"] = "exists" if dir_path.exists() else "missing"
            if not dir_path.exists():
                health["errors"].append(f"Missing directory: {dir_name}")
                health["overall"] = "unhealthy"

        # Check 4: Config files
        config_file = self.base_dir / "config/base.yaml"
        health["components"]["config"] = "exists" if config_file.exists() else "missing"

        # Check 5: Environment
        env_file = self.base_dir / ".env"
        health["components"]["env_file"] = "exists" if env_file.exists() else "missing"
        if not env_file.exists():
            health["warnings"].append("No .env file - API keys may be missing")

        # Check 6: Heartbeat
        heartbeat_file = self.base_dir / "state/autonomous/heartbeat.json"
        if heartbeat_file.exists():
            try:
                hb = json.loads(heartbeat_file.read_text())
                hb_time = datetime.fromisoformat(hb["timestamp"])
                age_minutes = (datetime.now(ET) - hb_time).total_seconds() / 60
                health["components"]["heartbeat_age_min"] = round(age_minutes, 1)
                if age_minutes > 5:
                    health["warnings"].append("Heartbeat is stale")
            except Exception:
                health["components"]["heartbeat_age_min"] = "error"
        else:
            health["components"]["heartbeat_age_min"] = "no_heartbeat"

        # Determine overall health
        if health["errors"]:
            health["overall"] = "unhealthy"
        elif health["warnings"]:
            health["overall"] = "degraded"

        health["timestamp"] = datetime.now(ET).isoformat()
        return health


# Task handlers
def check_data() -> Dict[str, Any]:
    """Task handler for data quality check."""
    engine = MaintenanceEngine()
    return engine.check_data()


def cleanup() -> Dict[str, Any]:
    """Task handler for cleanup."""
    engine = MaintenanceEngine()
    return engine.cleanup()


def health_check() -> Dict[str, Any]:
    """Task handler for health check."""
    engine = MaintenanceEngine()
    return engine.health_check()


if __name__ == "__main__":
    engine = MaintenanceEngine()

    print("Health Check:")
    print(json.dumps(engine.health_check(), indent=2))

    print("\nData Check:")
    print(json.dumps(engine.check_data(), indent=2))
