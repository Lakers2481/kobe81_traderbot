#!/usr/bin/env python3
"""
Alert management system for Kobe trading system.

Configures and monitors alert thresholds for drawdown, daily loss,
position size, and other risk metrics.

Usage:
    python alerts.py --check                     # Check current conditions
    python alerts.py --config                    # View/edit alert config
    python alerts.py --list                      # List active alerts
    python alerts.py --clear                     # Clear all active alerts
    python alerts.py --dotenv /path/to/.env      # Load env vars from file
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.env_loader import load_env

try:
    import requests
except ImportError:
    requests = None  # Will handle gracefully


# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ALERTS_CONFIG = CONFIGS_DIR / "alerts.json"
ALERTS_LOG = LOGS_DIR / "alerts.jsonl"
ACTIVE_ALERTS_FILE = LOGS_DIR / "active_alerts.json"
DEFAULT_DOTENV = Path("C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env")

# Default alert configuration
DEFAULT_CONFIG = {
    "thresholds": {
        "max_drawdown_pct": 5.0,          # Max account drawdown percentage
        "daily_loss_limit": 1000.0,        # Max daily loss in dollars
        "daily_loss_pct": 2.0,             # Max daily loss as percentage
        "position_size_limit": 10000.0,    # Max single position in dollars
        "position_count_limit": 10,        # Max number of open positions
        "trade_count_limit": 50,           # Max trades per day
        "consecutive_losses": 5,           # Max consecutive losing trades
        "profit_target_pct": 3.0,          # Daily profit target percentage
        "max_exposure_pct": 80.0           # Max portfolio exposure percentage
    },
    "notifications": {
        "telegram_enabled": True,
        "email_enabled": False,
        "sound_enabled": True
    },
    "severity_levels": {
        "warning": 0.7,    # Trigger warning at 70% of threshold
        "critical": 0.9,   # Trigger critical at 90% of threshold
        "breach": 1.0      # Full breach
    },
    "cooldown_minutes": 15,  # Minimum time between repeated alerts
    "auto_pause_on_breach": True,
    "updated_at": None
}


def ensure_dirs() -> None:
    """Create required directories if they don't exist."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Load alert configuration from file, creating default if needed."""
    ensure_dirs()

    if not ALERTS_CONFIG.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    try:
        with open(ALERTS_CONFIG, "r", encoding="utf-8") as f:
            config = json.load(f)
            # Merge with defaults for any missing keys
            merged = DEFAULT_CONFIG.copy()
            merged.update(config)
            if "thresholds" in config:
                merged["thresholds"] = {**DEFAULT_CONFIG["thresholds"], **config["thresholds"]}
            if "notifications" in config:
                merged["notifications"] = {**DEFAULT_CONFIG["notifications"], **config["notifications"]}
            return merged
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load config: {e}")
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]) -> None:
    """Save alert configuration to file."""
    ensure_dirs()
    config["updated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(ALERTS_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def log_alert(alert: Dict[str, Any]) -> None:
    """Append alert to JSONL log file."""
    ensure_dirs()
    alert["logged_at"] = datetime.utcnow().isoformat() + "Z"
    with open(ALERTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(alert) + "\n")


def load_active_alerts() -> List[Dict[str, Any]]:
    """Load currently active alerts."""
    if not ACTIVE_ALERTS_FILE.exists():
        return []
    try:
        with open(ACTIVE_ALERTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_active_alerts(alerts: List[Dict[str, Any]]) -> None:
    """Save active alerts to file."""
    ensure_dirs()
    with open(ACTIVE_ALERTS_FILE, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2)


def get_trading_state() -> Dict[str, Any]:
    """
    Get current trading state from outputs directory.
    Returns mock data if no real data available.
    """
    state = {
        "account_value": 100000.0,
        "starting_value": 100000.0,
        "daily_pnl": 0.0,
        "position_count": 0,
        "positions": [],
        "trade_count_today": 0,
        "consecutive_losses": 0,
        "max_position_value": 0.0,
        "total_exposure": 0.0
    }

    # Try to load account state
    account_file = OUTPUTS_DIR / "account_state.json"
    if account_file.exists():
        try:
            with open(account_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                state.update(data)
        except Exception:
            pass

    # Try to load positions
    positions_file = OUTPUTS_DIR / "positions.json"
    if positions_file.exists():
        try:
            with open(positions_file, "r", encoding="utf-8") as f:
                positions = json.load(f)
                state["positions"] = positions
                state["position_count"] = len(positions)
                if positions:
                    state["max_position_value"] = max(
                        abs(p.get("market_value", 0)) for p in positions
                    )
                    state["total_exposure"] = sum(
                        abs(p.get("market_value", 0)) for p in positions
                    )
        except Exception:
            pass

    # Try to load today's trades
    today = datetime.now().strftime("%Y-%m-%d")
    trades_file = OUTPUTS_DIR / f"trades_{today}.json"
    if trades_file.exists():
        try:
            with open(trades_file, "r", encoding="utf-8") as f:
                trades = json.load(f)
                state["trade_count_today"] = len(trades)

                # Calculate daily P&L and consecutive losses
                daily_pnl = 0.0
                consecutive_losses = 0
                for trade in trades:
                    pnl = trade.get("pnl", 0)
                    daily_pnl += pnl
                    if pnl < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

                state["daily_pnl"] = daily_pnl
                state["consecutive_losses"] = consecutive_losses
        except Exception:
            pass

    return state


def calculate_metrics(state: Dict[str, Any]) -> Dict[str, float]:
    """Calculate current risk metrics from trading state."""
    account_value = state.get("account_value", 100000)
    starting_value = state.get("starting_value", account_value)

    # Calculate drawdown
    peak_value = max(starting_value, account_value)  # Simplified
    drawdown = (peak_value - account_value) / peak_value * 100 if peak_value > 0 else 0

    # Calculate daily loss percentage
    daily_pnl = state.get("daily_pnl", 0)
    daily_loss_pct = (abs(daily_pnl) / starting_value * 100) if daily_pnl < 0 else 0

    # Calculate exposure percentage
    total_exposure = state.get("total_exposure", 0)
    exposure_pct = (total_exposure / account_value * 100) if account_value > 0 else 0

    return {
        "drawdown_pct": drawdown,
        "daily_pnl": daily_pnl,
        "daily_loss_pct": daily_loss_pct,
        "position_count": state.get("position_count", 0),
        "trade_count": state.get("trade_count_today", 0),
        "consecutive_losses": state.get("consecutive_losses", 0),
        "max_position_value": state.get("max_position_value", 0),
        "exposure_pct": exposure_pct
    }


def check_thresholds(config: Dict[str, Any], metrics: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Check current metrics against configured thresholds.
    Returns list of triggered alerts.
    """
    thresholds = config.get("thresholds", {})
    severity_levels = config.get("severity_levels", {})
    triggered = []

    # Define checks: (metric_key, threshold_key, comparison, message_template)
    checks = [
        ("drawdown_pct", "max_drawdown_pct", ">=",
         "Drawdown at {value:.2f}% (threshold: {threshold:.2f}%)"),

        ("daily_loss_pct", "daily_loss_pct", ">=",
         "Daily loss at {value:.2f}% (threshold: {threshold:.2f}%)"),

        ("position_count", "position_count_limit", ">=",
         "Position count at {value} (limit: {threshold})"),

        ("trade_count", "trade_count_limit", ">=",
         "Trade count at {value} (limit: {threshold})"),

        ("consecutive_losses", "consecutive_losses", ">=",
         "Consecutive losses at {value} (limit: {threshold})"),

        ("max_position_value", "position_size_limit", ">=",
         "Max position size ${value:,.2f} (limit: ${threshold:,.2f})"),

        ("exposure_pct", "max_exposure_pct", ">=",
         "Portfolio exposure at {value:.2f}% (limit: {threshold:.2f}%)"),
    ]

    # Check daily loss in dollars
    if metrics.get("daily_pnl", 0) < 0:
        daily_loss = abs(metrics["daily_pnl"])
        limit = thresholds.get("daily_loss_limit", 1000)
        if daily_loss >= limit:
            triggered.append({
                "type": "daily_loss_limit",
                "severity": "breach",
                "message": f"Daily loss ${daily_loss:,.2f} exceeds limit ${limit:,.2f}",
                "value": daily_loss,
                "threshold": limit
            })
        elif daily_loss >= limit * severity_levels.get("critical", 0.9):
            triggered.append({
                "type": "daily_loss_limit",
                "severity": "critical",
                "message": f"Daily loss ${daily_loss:,.2f} approaching limit ${limit:,.2f}",
                "value": daily_loss,
                "threshold": limit
            })
        elif daily_loss >= limit * severity_levels.get("warning", 0.7):
            triggered.append({
                "type": "daily_loss_limit",
                "severity": "warning",
                "message": f"Daily loss ${daily_loss:,.2f} nearing limit ${limit:,.2f}",
                "value": daily_loss,
                "threshold": limit
            })

    # Run standard checks
    for metric_key, threshold_key, comparison, msg_template in checks:
        metric_value = metrics.get(metric_key, 0)
        threshold_value = thresholds.get(threshold_key, float('inf'))

        if threshold_value <= 0:
            continue

        ratio = metric_value / threshold_value if threshold_value > 0 else 0

        if comparison == ">=" and metric_value >= threshold_value:
            triggered.append({
                "type": threshold_key,
                "severity": "breach",
                "message": msg_template.format(value=metric_value, threshold=threshold_value),
                "value": metric_value,
                "threshold": threshold_value
            })
        elif ratio >= severity_levels.get("critical", 0.9):
            triggered.append({
                "type": threshold_key,
                "severity": "critical",
                "message": "CRITICAL: " + msg_template.format(value=metric_value, threshold=threshold_value),
                "value": metric_value,
                "threshold": threshold_value
            })
        elif ratio >= severity_levels.get("warning", 0.7):
            triggered.append({
                "type": threshold_key,
                "severity": "warning",
                "message": "WARNING: " + msg_template.format(value=metric_value, threshold=threshold_value),
                "value": metric_value,
                "threshold": threshold_value
            })

    return triggered


def send_telegram_alert(message: str) -> bool:
    """Send alert via Telegram if configured."""
    if requests is None:
        print("Warning: requests library not available for Telegram")
        return False

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        return False

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        response = requests.post(url, json=payload, timeout=30)
        return response.json().get("ok", False)
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def trigger_alert(alert: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Process a triggered alert: log it and send notifications."""
    # Log the alert
    log_alert(alert)

    # Add to active alerts
    active = load_active_alerts()
    alert["triggered_at"] = datetime.utcnow().isoformat() + "Z"
    active.append(alert)
    save_active_alerts(active)

    # Print to console
    severity_emoji = {"warning": "[!]", "critical": "[!!]", "breach": "[!!!]"}
    emoji = severity_emoji.get(alert.get("severity", "warning"), "[!]")
    print(f"{emoji} ALERT: {alert.get('message', 'Unknown alert')}")

    # Send Telegram notification if enabled
    notifications = config.get("notifications", {})
    if notifications.get("telegram_enabled", True):
        severity = alert.get("severity", "warning").upper()
        telegram_msg = f"<b>{severity} ALERT</b>\n\n{alert.get('message', 'Unknown alert')}\n\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        if send_telegram_alert(telegram_msg):
            print("  -> Telegram notification sent")


def check_conditions(verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Check all conditions against thresholds and trigger alerts.
    Returns list of triggered alerts.
    """
    config = load_config()
    state = get_trading_state()
    metrics = calculate_metrics(state)

    if verbose:
        print("Current Metrics:")
        print(f"  Drawdown: {metrics['drawdown_pct']:.2f}%")
        print(f"  Daily P&L: ${metrics['daily_pnl']:,.2f}")
        print(f"  Daily Loss %: {metrics['daily_loss_pct']:.2f}%")
        print(f"  Position Count: {metrics['position_count']}")
        print(f"  Trade Count: {metrics['trade_count']}")
        print(f"  Consecutive Losses: {metrics['consecutive_losses']}")
        print(f"  Max Position: ${metrics['max_position_value']:,.2f}")
        print(f"  Exposure: {metrics['exposure_pct']:.2f}%")
        print()

    triggered = check_thresholds(config, metrics)

    if triggered:
        print(f"Found {len(triggered)} alert(s):")
        for alert in triggered:
            trigger_alert(alert, config)
    else:
        if verbose:
            print("All conditions within thresholds. No alerts triggered.")

    return triggered


def show_config() -> None:
    """Display current alert configuration."""
    config = load_config()

    print("Alert Configuration:")
    print("=" * 50)
    print("\nThresholds:")
    for key, value in config.get("thresholds", {}).items():
        formatted_key = key.replace("_", " ").title()
        if isinstance(value, float):
            if "pct" in key or "percent" in key:
                print(f"  {formatted_key}: {value:.1f}%")
            elif "limit" in key and value >= 100:
                print(f"  {formatted_key}: ${value:,.2f}")
            else:
                print(f"  {formatted_key}: {value}")
        else:
            print(f"  {formatted_key}: {value}")

    print("\nNotifications:")
    for key, value in config.get("notifications", {}).items():
        formatted_key = key.replace("_", " ").title()
        status = "Enabled" if value else "Disabled"
        print(f"  {formatted_key}: {status}")

    print("\nSeverity Levels:")
    for key, value in config.get("severity_levels", {}).items():
        print(f"  {key.title()}: {value * 100:.0f}%")

    print(f"\nCooldown: {config.get('cooldown_minutes', 15)} minutes")
    print(f"Auto-pause on breach: {config.get('auto_pause_on_breach', True)}")

    if config.get("updated_at"):
        print(f"\nLast updated: {config['updated_at']}")

    print(f"\nConfig file: {ALERTS_CONFIG}")


def configure_threshold(key: str, value: float) -> None:
    """Update a specific threshold value."""
    config = load_config()
    if "thresholds" not in config:
        config["thresholds"] = {}

    old_value = config["thresholds"].get(key, "not set")
    config["thresholds"][key] = value
    save_config(config)
    print(f"Updated {key}: {old_value} -> {value}")


def list_active_alerts() -> None:
    """List all currently active alerts."""
    alerts = load_active_alerts()

    if not alerts:
        print("No active alerts.")
        return

    print(f"Active Alerts ({len(alerts)}):")
    print("=" * 60)

    for i, alert in enumerate(alerts, 1):
        severity = alert.get("severity", "unknown").upper()
        alert_type = alert.get("type", "unknown")
        message = alert.get("message", "No message")
        triggered_at = alert.get("triggered_at", "Unknown time")

        print(f"\n{i}. [{severity}] {alert_type}")
        print(f"   Message: {message}")
        print(f"   Triggered: {triggered_at}")


def clear_alerts() -> None:
    """Clear all active alerts."""
    alerts = load_active_alerts()
    count = len(alerts)

    if count == 0:
        print("No active alerts to clear.")
        return

    # Log the clearing
    log_alert({
        "type": "alerts_cleared",
        "severity": "info",
        "message": f"Cleared {count} active alert(s)",
        "cleared_alerts": alerts
    })

    save_active_alerts([])
    print(f"Cleared {count} active alert(s).")


def interactive_config() -> None:
    """Interactive configuration mode."""
    config = load_config()

    print("\nInteractive Configuration")
    print("=" * 40)
    print("Enter new values or press Enter to keep current.\n")

    thresholds = config.get("thresholds", {})

    prompts = [
        ("max_drawdown_pct", "Max Drawdown %", "%"),
        ("daily_loss_limit", "Daily Loss Limit $", "$"),
        ("daily_loss_pct", "Daily Loss %", "%"),
        ("position_size_limit", "Position Size Limit $", "$"),
        ("position_count_limit", "Max Positions", ""),
        ("trade_count_limit", "Max Trades/Day", ""),
        ("consecutive_losses", "Max Consecutive Losses", ""),
        ("max_exposure_pct", "Max Exposure %", "%"),
    ]

    for key, label, unit in prompts:
        current = thresholds.get(key, DEFAULT_CONFIG["thresholds"].get(key, 0))
        if unit == "$":
            display = f"${current:,.2f}"
        elif unit == "%":
            display = f"{current:.1f}%"
        else:
            display = str(int(current))

        try:
            user_input = input(f"  {label} (current: {display}): ").strip()
            if user_input:
                new_value = float(user_input.replace("$", "").replace("%", "").replace(",", ""))
                thresholds[key] = new_value
                print(f"    -> Updated to {new_value}")
        except (ValueError, KeyboardInterrupt):
            print("    -> Keeping current value")

    config["thresholds"] = thresholds
    save_config(config)
    print("\nConfiguration saved.")


def main():
    ap = argparse.ArgumentParser(
        description="Alert management system for Kobe trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python alerts.py --check          # Check current conditions
  python alerts.py --config         # View configuration
  python alerts.py --config --edit  # Interactive configuration
  python alerts.py --list           # List active alerts
  python alerts.py --clear          # Clear all active alerts
  python alerts.py --set max_drawdown_pct=10.0  # Set specific threshold
        """
    )
    ap.add_argument("--check", action="store_true", help="Check current conditions against thresholds")
    ap.add_argument("--config", action="store_true", help="View alert configuration")
    ap.add_argument("--edit", action="store_true", help="Interactive configuration mode (use with --config)")
    ap.add_argument("--list", action="store_true", help="List active alerts")
    ap.add_argument("--clear", action="store_true", help="Clear all active alerts")
    ap.add_argument("--set", type=str, metavar="KEY=VALUE", help="Set a specific threshold (e.g., max_drawdown_pct=10.0)")
    ap.add_argument("--dotenv", type=str, default=str(DEFAULT_DOTENV),
                    help="Path to .env file (default: %(default)s)")
    ap.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (minimal output)")

    args = ap.parse_args()

    # Load environment variables
    dotenv_path = Path(args.dotenv)
    if dotenv_path.exists():
        loaded = load_env(dotenv_path)
        if not args.quiet:
            print(f"Loaded {len(loaded)} env vars from {dotenv_path}")
    elif not args.quiet:
        print(f"Note: .env file not found at {dotenv_path}")

    # Handle commands
    if args.set:
        if "=" not in args.set:
            print("ERROR: --set requires KEY=VALUE format")
            sys.exit(1)
        key, value = args.set.split("=", 1)
        try:
            configure_threshold(key.strip(), float(value.strip()))
        except ValueError:
            print(f"ERROR: Invalid value: {value}")
            sys.exit(1)

    elif args.config:
        if args.edit:
            interactive_config()
        else:
            show_config()

    elif args.list:
        list_active_alerts()

    elif args.clear:
        clear_alerts()

    elif args.check:
        triggered = check_conditions(verbose=not args.quiet)
        sys.exit(1 if triggered else 0)

    else:
        # Default: show status summary
        print("Kobe Alert System")
        print("=" * 40)
        load_config()
        alerts = load_active_alerts()

        print(f"Active alerts: {len(alerts)}")
        print(f"Config file: {ALERTS_CONFIG}")
        print(f"Log file: {ALERTS_LOG}")
        print()
        print("Commands:")
        print("  --check   Check current conditions")
        print("  --config  View configuration")
        print("  --list    List active alerts")
        print("  --clear   Clear active alerts")
        print()
        ap.print_usage()


if __name__ == "__main__":
    main()
