#!/usr/bin/env python3
"""
Kobe Health Monitor - System Health Watchdog
=============================================

Monitors the health of the Kobe trading system:
1. Runner process status (is it running?)
2. Heartbeat freshness (has it updated recently?)
3. Broker connectivity (can we reach Alpaca?)
4. Data provider status (is Polygon working?)

Sends Telegram alerts when issues are detected.
Can optionally restart the runner if it dies.

Usage:
    python scripts/health_monitor.py --check-runner
    python scripts/health_monitor.py --check-runner --alert-on-failure
    python scripts/health_monitor.py --check-all
    python scripts/health_monitor.py --restart-runner  # Restart if dead
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from core.structured_log import jlog
from core.alerts import send_telegram
from core.clock.tz_utils import now_et, fmt_ct

# Health check thresholds
HEARTBEAT_STALE_MINUTES = 5  # Heartbeat older than this = stale
LOCK_STALE_MINUTES = 10      # Lock file older than this = possibly dead

# State files
HEARTBEAT_FILE = ROOT / 'state' / 'heartbeat.json'
LOCK_FILE = ROOT / 'state' / 'kobe_runner.lock'
HEALTH_STATE_FILE = ROOT / 'state' / 'health_monitor_state.json'


class HealthStatus:
    """Health check result."""
    def __init__(self, name: str, healthy: bool, message: str, details: Dict = None):
        self.name = name
        self.healthy = healthy
        self.message = message
        self.details = details or {}
        self.timestamp = now_et()

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'healthy': self.healthy,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
        }


def load_health_state() -> Dict:
    """Load health monitor state (for tracking alerts sent)."""
    if HEALTH_STATE_FILE.exists():
        try:
            return json.loads(HEALTH_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {'last_alert': {}, 'consecutive_failures': {}}


def save_health_state(state: Dict) -> None:
    """Save health monitor state."""
    HEALTH_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def check_heartbeat() -> HealthStatus:
    """Check if runner heartbeat is fresh."""
    if not HEARTBEAT_FILE.exists():
        return HealthStatus(
            name='heartbeat',
            healthy=False,
            message='Heartbeat file not found - runner may not be running',
        )

    try:
        heartbeat = json.loads(HEARTBEAT_FILE.read_text())
        last_update = heartbeat.get('last_update')

        if not last_update:
            return HealthStatus(
                name='heartbeat',
                healthy=False,
                message='Heartbeat file has no last_update field',
                details=heartbeat,
            )

        # Parse timestamp
        try:
            last_dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
        except ValueError:
            last_dt = datetime.fromisoformat(last_update)

        # Make timezone-aware if needed
        now = now_et()
        if last_dt.tzinfo is None:
            from core.clock.tz_utils import ET
            last_dt = last_dt.replace(tzinfo=ET)

        age_minutes = (now - last_dt).total_seconds() / 60

        if age_minutes > HEARTBEAT_STALE_MINUTES:
            return HealthStatus(
                name='heartbeat',
                healthy=False,
                message=f'Heartbeat stale ({age_minutes:.1f} min old)',
                details={'age_minutes': age_minutes, 'last_update': last_update},
            )

        return HealthStatus(
            name='heartbeat',
            healthy=True,
            message=f'Heartbeat fresh ({age_minutes:.1f} min old)',
            details={'age_minutes': age_minutes, 'status': heartbeat.get('status')},
        )

    except Exception as e:
        return HealthStatus(
            name='heartbeat',
            healthy=False,
            message=f'Error reading heartbeat: {e}',
        )


def check_lock_file() -> HealthStatus:
    """Check if runner lock file exists and is fresh."""
    if not LOCK_FILE.exists():
        return HealthStatus(
            name='lock_file',
            healthy=False,
            message='Lock file not found - runner not running',
        )

    try:
        # Check lock file modification time
        mtime = datetime.fromtimestamp(LOCK_FILE.stat().st_mtime)
        now = datetime.now()
        age_minutes = (now - mtime).total_seconds() / 60

        if age_minutes > LOCK_STALE_MINUTES:
            return HealthStatus(
                name='lock_file',
                healthy=False,
                message=f'Lock file stale ({age_minutes:.1f} min old) - runner may be frozen',
                details={'age_minutes': age_minutes},
            )

        return HealthStatus(
            name='lock_file',
            healthy=True,
            message=f'Lock file fresh ({age_minutes:.1f} min old)',
            details={'age_minutes': age_minutes},
        )

    except Exception as e:
        return HealthStatus(
            name='lock_file',
            healthy=False,
            message=f'Error checking lock file: {e}',
        )


def check_runner_process() -> HealthStatus:
    """Check if runner.py process is running."""
    try:
        # Windows: use tasklist
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
            capture_output=True, text=True, timeout=10
        )

        if 'python.exe' in result.stdout:
            # Check if it's specifically runner.py (approximate)
            # More accurate would be to check command line, but this is Windows...
            return HealthStatus(
                name='runner_process',
                healthy=True,
                message='Python process(es) found running',
                details={'output': result.stdout[:500]},
            )
        else:
            return HealthStatus(
                name='runner_process',
                healthy=False,
                message='No Python processes found',
            )

    except Exception as e:
        return HealthStatus(
            name='runner_process',
            healthy=False,
            message=f'Error checking process: {e}',
        )


def check_broker_connectivity() -> HealthStatus:
    """Check if we can connect to Alpaca."""
    import requests

    base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
    key = os.getenv('ALPACA_API_KEY_ID', '')
    sec = os.getenv('ALPACA_API_SECRET_KEY', '')

    if not key or not sec:
        return HealthStatus(
            name='broker',
            healthy=False,
            message='Alpaca credentials not configured',
        )

    headers = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': sec}

    try:
        resp = requests.get(f"{base}/v2/account", headers=headers, timeout=10)

        if resp.status_code == 200:
            account = resp.json()
            return HealthStatus(
                name='broker',
                healthy=True,
                message=f"Connected to Alpaca ({account.get('status', 'OK')})",
                details={
                    'equity': account.get('equity'),
                    'buying_power': account.get('buying_power'),
                    'status': account.get('status'),
                },
            )
        else:
            return HealthStatus(
                name='broker',
                healthy=False,
                message=f'Alpaca returned {resp.status_code}',
                details={'response': resp.text[:200]},
            )

    except requests.exceptions.Timeout:
        return HealthStatus(
            name='broker',
            healthy=False,
            message='Alpaca connection timeout',
        )
    except Exception as e:
        return HealthStatus(
            name='broker',
            healthy=False,
            message=f'Broker check error: {e}',
        )


def check_data_provider() -> HealthStatus:
    """Check if Polygon API is accessible."""
    import requests

    api_key = os.getenv('POLYGON_API_KEY', '')

    if not api_key:
        return HealthStatus(
            name='data_provider',
            healthy=False,
            message='Polygon API key not configured',
        )

    try:
        # Simple test: get market status
        resp = requests.get(
            f"https://api.polygon.io/v1/marketstatus/now?apiKey={api_key}",
            timeout=10
        )

        if resp.status_code == 200:
            data = resp.json()
            return HealthStatus(
                name='data_provider',
                healthy=True,
                message=f"Polygon OK (market: {data.get('market', 'unknown')})",
                details=data,
            )
        else:
            return HealthStatus(
                name='data_provider',
                healthy=False,
                message=f'Polygon returned {resp.status_code}',
            )

    except Exception as e:
        return HealthStatus(
            name='data_provider',
            healthy=False,
            message=f'Data provider check error: {e}',
        )


def restart_runner() -> bool:
    """Attempt to restart the runner via Task Scheduler."""
    try:
        # Start the scheduled task
        result = subprocess.run(
            ['schtasks', '/Run', '/TN', 'KobeRunner'],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            jlog('health_monitor_runner_restarted', level='INFO')
            return True
        else:
            jlog('health_monitor_restart_failed', error=result.stderr, level='ERROR')
            return False

    except Exception as e:
        jlog('health_monitor_restart_error', error=str(e), level='ERROR')
        return False


def send_health_alert(checks: List[HealthStatus], state: Dict) -> None:
    """Send Telegram alert for failed health checks."""
    failed = [c for c in checks if not c.healthy]

    if not failed:
        return

    # Rate limit: don't spam alerts
    last_alert_key = 'health_alert'
    last_alert = state.get('last_alert', {}).get(last_alert_key)

    if last_alert:
        try:
            last_dt = datetime.fromisoformat(last_alert)
            if (datetime.now() - last_dt).total_seconds() < 300:  # 5 min cooldown
                jlog('health_alert_rate_limited', level='DEBUG')
                return
        except ValueError:
            pass

    # Build alert message
    now = now_et()
    stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"

    lines = [f"KOBE HEALTH ALERT [{stamp}]"]
    for check in failed:
        lines.append(f"  {check.name}: {check.message}")

    msg = "\n".join(lines)

    try:
        send_telegram(msg)
        state.setdefault('last_alert', {})[last_alert_key] = datetime.now().isoformat()
        save_health_state(state)
        jlog('health_alert_sent', checks=[c.name for c in failed])
    except Exception as e:
        jlog('health_alert_failed', error=str(e), level='ERROR')


def main():
    ap = argparse.ArgumentParser(description='Kobe Health Monitor')
    ap.add_argument('--check-runner', action='store_true', help='Check if runner is healthy')
    ap.add_argument('--check-broker', action='store_true', help='Check broker connectivity')
    ap.add_argument('--check-data', action='store_true', help='Check data provider')
    ap.add_argument('--check-all', action='store_true', help='Run all health checks')
    ap.add_argument('--alert-on-failure', action='store_true', help='Send Telegram alert on failure')
    ap.add_argument('--restart-runner', action='store_true', help='Restart runner if dead')
    ap.add_argument('--json', action='store_true', help='Output as JSON')
    ap.add_argument('--dotenv', type=str, default='./.env', help='Path to .env file')
    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # Load state
    state = load_health_state()

    checks: List[HealthStatus] = []

    # Runner checks
    if args.check_runner or args.check_all:
        checks.append(check_heartbeat())
        checks.append(check_lock_file())
        checks.append(check_runner_process())

    # Broker check
    if args.check_broker or args.check_all:
        checks.append(check_broker_connectivity())

    # Data provider check
    if args.check_data or args.check_all:
        checks.append(check_data_provider())

    # Output results
    if args.json:
        print(json.dumps([c.to_dict() for c in checks], indent=2))
    else:
        print("=" * 60)
        print("KOBE HEALTH MONITOR")
        print("=" * 60)
        print(f"Time: {fmt_ct(now_et())} ET")
        print()

        all_healthy = True
        for check in checks:
            status = "OK" if check.healthy else "FAIL"
            icon = "[OK]" if check.healthy else "[!!]"
            print(f"{icon} {check.name}: {check.message}")
            if not check.healthy:
                all_healthy = False

        print()
        if all_healthy:
            print("STATUS: ALL SYSTEMS HEALTHY")
        else:
            print("STATUS: ISSUES DETECTED")

    # Send alert if requested
    if args.alert_on_failure:
        send_health_alert(checks, state)

    # Restart runner if requested and needed
    if args.restart_runner:
        runner_healthy = all(
            c.healthy for c in checks
            if c.name in ('heartbeat', 'lock_file')
        )

        if not runner_healthy:
            print("\nRunner appears down. Attempting restart...")
            if restart_runner():
                print("Restart command sent successfully.")
                try:
                    now = now_et()
                    stamp = f"{fmt_ct(now)} | {now.strftime('%I:%M %p').lstrip('0')} ET"
                    send_telegram(f"Kobe runner restarted by health monitor [{stamp}]")
                except Exception:
                    pass
            else:
                print("Restart failed. Check logs.")

    # Exit with appropriate code
    sys.exit(0 if all(c.healthy for c in checks) else 1)


if __name__ == '__main__':
    main()
