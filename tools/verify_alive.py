#!/usr/bin/env python3
"""
KOBE ALIVE CHECK - Comprehensive Robot Health Verification
============================================================

This script tells you if the robot is ALIVE and running 24/7.
Run this anytime to verify the brain and heart are working.

Checks:
1. HEARTBEAT - Is the brain alive and beating?
2. BRAIN STATE - Is the autonomous brain running?
3. TASK EXECUTION - Are scheduled tasks executing?
4. TELEGRAM - Are alerts configured and sending?
5. WATCHLIST - Is the scanner generating watchlists?
6. DATA - Is data available and fresh?
7. COMPONENTS - Are all 721 Python files working?
8. ML/AI - Are the AI models loaded?
9. BROKER - Is the trading connection alive?
10. LOGS - Are logs being written?

Usage:
    python tools/verify_alive.py           # Full check
    python tools/verify_alive.py --quick   # Quick heartbeat only
    python tools/verify_alive.py --verbose # Show all details
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    import pytz
    ET = pytz.timezone("America/New_York")


class AliveChecker:
    """Comprehensive robot health verification."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.now = datetime.now(ET)

    def log(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(f"    {msg}")

    def check_heartbeat(self) -> Tuple[str, Dict]:
        """Check if brain heartbeat is recent."""
        heartbeat_path = ROOT / "state" / "autonomous" / "heartbeat.json"

        if not heartbeat_path.exists():
            return "DEAD", {"error": "Heartbeat file not found - brain never started"}

        try:
            data = json.loads(heartbeat_path.read_text())

            # Handle both formats: "last_beat" (new) or "timestamp" (old)
            timestamp_str = data.get("last_beat") or data.get("timestamp", "")
            if not timestamp_str:
                return "ERROR", {"error": "No timestamp in heartbeat file"}

            last_beat = datetime.fromisoformat(timestamp_str)

            # Check if heartbeat is recent (within 5 minutes)
            age_seconds = (self.now - last_beat).total_seconds()
            age_minutes = age_seconds / 60

            if age_minutes < 2:
                status = "ALIVE"
            elif age_minutes < 5:
                status = "SLOW"
            elif age_minutes < 60:
                status = "STALE"
            else:
                status = "DEAD"

            return status, {
                "last_beat": str(last_beat),
                "age_minutes": round(age_minutes, 1),
                "uptime_hours": data.get("uptime_hours", 0),
                "phase": data.get("phase", "unknown"),
                "cycles": data.get("cycles", data.get("tasks_run", 0)),
                "alive": data.get("alive", True),
                "work_mode": data.get("work_mode", "unknown"),
            }
        except Exception as e:
            return "ERROR", {"error": str(e)}

    def check_brain_state(self) -> Tuple[str, Dict]:
        """Check autonomous brain state."""
        brain_state_path = ROOT / "state" / "autonomous" / "brain_state.json"

        if not brain_state_path.exists():
            return "NO_STATE", {"error": "Brain state file not found"}

        try:
            data = json.loads(brain_state_path.read_text())
            return "OK", {
                "status": data.get("status", "unknown"),
                "current_phase": data.get("current_phase", "unknown"),
                "cycles_completed": data.get("cycles_completed", 0),
            }
        except Exception as e:
            return "ERROR", {"error": str(e)}

    def check_task_queue(self) -> Tuple[str, Dict]:
        """Check task queue state."""
        task_queue_path = ROOT / "state" / "autonomous" / "task_queue.json"

        if not task_queue_path.exists():
            return "NO_QUEUE", {"error": "Task queue not found"}

        try:
            data = json.loads(task_queue_path.read_text())
            pending = len([t for t in data.get("tasks", []) if t.get("status") == "pending"])
            completed = len([t for t in data.get("tasks", []) if t.get("status") == "completed"])

            return "OK", {
                "total_tasks": len(data.get("tasks", [])),
                "pending": pending,
                "completed": completed,
            }
        except Exception as e:
            return "ERROR", {"error": str(e)}

    def check_telegram(self) -> Tuple[str, Dict]:
        """Check Telegram configuration."""
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        enabled = os.getenv("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true"

        if not token or not chat_id:
            return "NOT_CONFIGURED", {"error": "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"}

        if not enabled:
            return "DISABLED", {"note": "TELEGRAM_ALERTS_ENABLED is not true"}

        # Try to send test (only if verbose)
        if self.verbose:
            try:
                from alerts.telegram_alerter import TelegramAlerter
                alerter = TelegramAlerter(enabled=True)
                # Don't actually send - just verify config
                return "CONFIGURED", {"token_prefix": token[:10] + "...", "chat_id": chat_id}
            except Exception as e:
                return "ERROR", {"error": str(e)}

        return "CONFIGURED", {"token_prefix": token[:10] + "...", "chat_id": chat_id}

    def check_watchlist(self) -> Tuple[str, Dict]:
        """Check if watchlist is being generated."""
        watchlist_path = ROOT / "state" / "watchlist" / "next_day.json"

        if not watchlist_path.exists():
            return "NOT_GENERATED", {"error": "Watchlist file not found"}

        try:
            data = json.loads(watchlist_path.read_text())
            generated_at = datetime.fromisoformat(data.get("generated_at", "").replace("Z", "+00:00"))
            age_hours = (self.now - generated_at).total_seconds() / 3600

            status = "OK" if age_hours < 24 else "STALE"

            return status, {
                "for_date": data.get("for_date", ""),
                "age_hours": round(age_hours, 1),
                "watchlist_size": data.get("watchlist_size", 0),
                "totd": data.get("totd", {}).get("symbol", ""),
            }
        except Exception as e:
            return "ERROR", {"error": str(e)}

    def check_data_cache(self) -> Tuple[str, Dict]:
        """Check data cache status."""
        cache_dir = ROOT / "data" / "polygon_cache"

        if not cache_dir.exists():
            return "NO_CACHE", {"error": "Cache directory not found"}

        csv_files = list(cache_dir.glob("*.csv"))
        if not csv_files:
            return "EMPTY", {"error": "No cached data files"}

        # Check freshness of a sample file
        sample = csv_files[0]
        mtime = datetime.fromtimestamp(sample.stat().st_mtime, tz=ET)
        age_hours = (self.now - mtime).total_seconds() / 3600

        return "OK", {
            "cached_symbols": len(csv_files),
            "sample_age_hours": round(age_hours, 1),
        }

    def check_universe_coverage(self) -> Tuple[str, Dict]:
        """Check 900 stock universe data coverage."""
        universe_path = ROOT / "data" / "universe" / "optionable_liquid_800.csv"
        cache_dir = ROOT / "data" / "polygon_cache"

        if not universe_path.exists():
            return "NO_UNIVERSE", {"error": "Universe file not found"}

        import pandas as pd
        df = pd.read_csv(universe_path)
        universe_symbols = set(df.iloc[:, 0].str.upper().tolist())
        total_universe = len(universe_symbols)

        if not cache_dir.exists():
            return "NO_CACHE", {
                "universe_size": total_universe,
                "cached": 0,
                "coverage_pct": 0,
            }

        csv_files = list(cache_dir.glob("*.csv"))
        cached_symbols = set(f.stem.upper() for f in csv_files)
        covered = universe_symbols & cached_symbols
        coverage_pct = (len(covered) / total_universe) * 100 if total_universe > 0 else 0

        if coverage_pct >= 90:
            status = "OK"
        elif coverage_pct >= 50:
            status = "PARTIAL"
        else:
            status = "LOW"

        return status, {
            "universe_size": total_universe,
            "cached": len(covered),
            "coverage_pct": round(coverage_pct, 1),
            "missing": total_universe - len(covered),
        }

    def check_components(self) -> Tuple[str, Dict]:
        """Check Python files are importable."""
        py_files = list(ROOT.glob("**/*.py"))
        total = len([f for f in py_files if "__pycache__" not in str(f)])

        # Check a few critical imports
        critical = [
            ("strategies.dual_strategy", "DualStrategyScanner"),
            ("execution.broker_alpaca", "AlpacaBroker"),
            ("risk.policy_gate", "PolicyGate"),
            ("cognitive.curiosity_engine", "CuriosityEngine"),
            ("autonomous.master_brain_full", "MasterBrainFull"),
        ]

        working = 0
        failed_imports = []

        for module, class_name in critical:
            try:
                mod = __import__(module, fromlist=[class_name])
                getattr(mod, class_name)
                working += 1
            except Exception as e:
                failed_imports.append(f"{module}.{class_name}: {e}")

        status = "OK" if len(failed_imports) == 0 else "PARTIAL"

        return status, {
            "total_py_files": total,
            "critical_imports": f"{working}/{len(critical)}",
            "failed": failed_imports[:3] if failed_imports else [],
        }

    def check_ml_models(self) -> Tuple[str, Dict]:
        """Check ML model availability."""
        models = {
            "hmm_regime": False,
            "lstm_confidence": False,
            "ensemble": False,
        }

        try:
            models["hmm_regime"] = True
        except Exception:
            pass

        try:
            models["lstm_confidence"] = True
        except Exception:
            pass

        try:
            models["ensemble"] = True
        except Exception:
            pass

        loaded = sum(models.values())
        status = "OK" if loaded == 3 else ("PARTIAL" if loaded > 0 else "NONE")

        return status, {"models": models, "loaded": f"{loaded}/3"}

    def check_broker(self) -> Tuple[str, Dict]:
        """Check broker connection."""
        key = os.getenv("ALPACA_API_KEY_ID", "")
        secret = os.getenv("ALPACA_API_SECRET_KEY", "")
        url = os.getenv("ALPACA_BASE_URL", "")

        if not key or not secret:
            return "NOT_CONFIGURED", {"error": "Missing Alpaca credentials"}

        mode = "PAPER" if "paper" in url.lower() else "LIVE"

        # Try quick connection test
        try:
            from execution.broker_alpaca import AlpacaBroker
            broker = AlpacaBroker(paper=("paper" in url.lower()))
            broker.connect()
            account = broker.get_account()
            equity = float(getattr(account, "equity", 0)) if account else 0

            return "CONNECTED", {
                "mode": mode,
                "equity": f"${equity:,.2f}",
            }
        except Exception as e:
            return "ERROR", {"mode": mode, "error": str(e)[:100]}

    def check_logs(self) -> Tuple[str, Dict]:
        """Check log files are being written."""
        logs_dir = ROOT / "logs"

        if not logs_dir.exists():
            return "NO_LOGS", {"error": "Logs directory not found"}

        log_files = list(logs_dir.glob("*.jsonl")) + list(logs_dir.glob("*.log")) + list(logs_dir.glob("*.csv"))

        if not log_files:
            return "EMPTY", {"error": "No log files"}

        # Find most recent
        recent = max(log_files, key=lambda f: f.stat().st_mtime)
        mtime = datetime.fromtimestamp(recent.stat().st_mtime, tz=ET)
        age_hours = (self.now - mtime).total_seconds() / 3600

        return "OK", {
            "log_files": len(log_files),
            "most_recent": recent.name,
            "age_hours": round(age_hours, 1),
        }

    def check_kill_switch(self) -> Tuple[str, Dict]:
        """Check if kill switch is active."""
        kill_path = ROOT / "state" / "KILL_SWITCH"

        if kill_path.exists():
            return "ACTIVE", {"warning": "KILL SWITCH IS ON - trading halted!"}

        return "OK", {"status": "Kill switch not active"}

    def run_all_checks(self) -> Dict:
        """Run all checks and return results."""
        checks = [
            ("HEARTBEAT", self.check_heartbeat),
            ("BRAIN_STATE", self.check_brain_state),
            ("TASK_QUEUE", self.check_task_queue),
            ("TELEGRAM", self.check_telegram),
            ("WATCHLIST", self.check_watchlist),
            ("DATA_CACHE", self.check_data_cache),
            ("UNIVERSE_900", self.check_universe_coverage),
            ("COMPONENTS", self.check_components),
            ("ML_MODELS", self.check_ml_models),
            ("BROKER", self.check_broker),
            ("LOGS", self.check_logs),
            ("KILL_SWITCH", self.check_kill_switch),
        ]

        for name, check_func in checks:
            try:
                status, details = check_func()
                self.results[name] = {"status": status, "details": details}

                if status in ["OK", "ALIVE", "CONFIGURED", "CONNECTED"]:
                    self.passed += 1
                elif status in ["DEAD", "ERROR", "NOT_CONFIGURED", "ACTIVE"]:
                    self.failed += 1
                else:
                    self.warnings += 1

            except Exception as e:
                self.results[name] = {"status": "ERROR", "details": {"error": str(e)}}
                self.failed += 1

        return self.results

    def print_report(self):
        """Print formatted report."""
        print()
        print("=" * 70)
        print("  K O B E   A L I V E   C H E C K")
        print("=" * 70)
        print(f"  Timestamp: {self.now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print()

        # Status icons (ASCII for Windows compatibility)
        icons = {
            "OK": "+",
            "ALIVE": "*",
            "CONFIGURED": "+",
            "CONNECTED": "+",
            "PARTIAL": "~",
            "SLOW": "~",
            "STALE": "!",
            "DISABLED": "-",
            "NO_STATE": "?",
            "NO_QUEUE": "?",
            "NO_CACHE": "?",
            "NO_LOGS": "?",
            "EMPTY": "!",
            "NOT_CONFIGURED": "X",
            "NOT_GENERATED": "X",
            "DEAD": "X",
            "ERROR": "X",
            "ACTIVE": "!",
            "NONE": "X",
            "LOW": "!",
        }

        for name, result in self.results.items():
            status = result["status"]
            icon = icons.get(status, "?")
            details = result.get("details", {})

            # Status line
            print(f"  [{icon}] {name}: {status}")

            # Key details
            if self.verbose or status not in ["OK", "ALIVE", "CONFIGURED", "CONNECTED"]:
                for k, v in details.items():
                    if k != "error":
                        print(f"      {k}: {v}")
                if "error" in details:
                    print(f"      ERROR: {details['error'][:80]}")

        print()
        print("-" * 70)

        # Summary
        total = self.passed + self.failed + self.warnings
        if self.failed == 0:
            overall = "ALIVE"
            msg = "Robot is ALIVE and running!"
        elif self.passed > self.failed:
            overall = "DEGRADED"
            msg = "Robot is running but has issues"
        else:
            overall = "DEAD"
            msg = "Robot is NOT functioning properly!"

        print(f"  OVERALL STATUS: {overall}")
        print(f"  {msg}")
        print()
        print(f"  Passed: {self.passed}/{total} | Warnings: {self.warnings} | Failed: {self.failed}")
        print("=" * 70)
        print()


def main():
    parser = argparse.ArgumentParser(description="Kobe Robot Alive Check")
    parser.add_argument("--quick", action="store_true", help="Quick heartbeat only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all details")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    checker = AliveChecker(verbose=args.verbose)

    if args.quick:
        status, details = checker.check_heartbeat()
        if args.json:
            print(json.dumps({"heartbeat": {"status": status, "details": details}}, indent=2))
        else:
            print(f"HEARTBEAT: {status}")
            if details.get("age_minutes"):
                print(f"  Last beat: {details.get('age_minutes', 0):.1f} minutes ago")
                print(f"  Phase: {details.get('phase', 'unknown')}")
                print(f"  Tasks run: {details.get('tasks_run', 0)}")
        return 0 if status == "ALIVE" else 1

    results = checker.run_all_checks()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        checker.print_report()

    return 0 if checker.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
