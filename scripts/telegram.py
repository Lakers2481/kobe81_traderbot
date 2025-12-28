#!/usr/bin/env python3
"""
Telegram notification service for Kobe trading system.

Sends trade alerts, daily P&L summaries, and error notifications via Telegram.

Usage:
    python telegram.py --test                    # Test Telegram connection
    python telegram.py --send "Your message"     # Send a custom message
    python telegram.py --daily-summary           # Send daily P&L summary
    python telegram.py --dotenv /path/to/.env    # Load env vars from file
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from core.clock.tz_utils import now_et, fmt_ct
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.env_loader import load_env

try:
    import requests
except ImportError:
    print("ERROR: requests library not installed. Run: pip install requests")
    sys.exit(1)


# Constants
TELEGRAM_API_BASE = "https://api.telegram.org/bot"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
ALERTS_LOG = LOGS_DIR / "alerts.jsonl"
DEFAULT_DOTENV = Path("C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env")


def ensure_logs_dir() -> None:
    """Create logs directory if it doesn't exist."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def log_alert(alert_type: str, message: str, success: bool, details: Optional[Dict[str, Any]] = None) -> None:
    """Log alert to JSONL file."""
    ensure_logs_dir()
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": alert_type,
        "message": message[:200],  # Truncate for log
        "success": success,
        "details": details or {}
    }
    with open(ALERTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def get_telegram_credentials() -> tuple[Optional[str], Optional[str]]:
    """Get Telegram credentials from environment."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    return bot_token, chat_id


def validate_credentials() -> tuple[str, str]:
    """Validate and return Telegram credentials, exit if missing."""
    bot_token, chat_id = get_telegram_credentials()

    missing = []
    if not bot_token:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not chat_id:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        print("Set these in your .env file or environment:")
        print("  TELEGRAM_BOT_TOKEN=your_bot_token")
        print("  TELEGRAM_CHAT_ID=your_chat_id")
        print("\nTo get these values:")
        print("  1. Create a bot via @BotFather on Telegram")
        print("  2. Get your chat ID by messaging @userinfobot")
        sys.exit(1)

    return bot_token, chat_id


def send_telegram_message(message: str, parse_mode: str = "HTML") -> Dict[str, Any]:
    """
    Send a message to Telegram.

    Args:
        message: The message text to send
        parse_mode: HTML or Markdown formatting

    Returns:
        Response dict with 'ok' boolean and 'result' or 'description'
    """
    bot_token, chat_id = validate_credentials()

    url = f"{TELEGRAM_API_BASE}{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        result = response.json()

        success = result.get("ok", False)
        log_alert("telegram_message", message, success, {"response_code": response.status_code})

        return result
    except requests.RequestException as e:
        error_msg = str(e)
        log_alert("telegram_message", message, False, {"error": error_msg})
        return {"ok": False, "description": f"Request failed: {error_msg}"}


def test_connection() -> bool:
    """Test Telegram bot connection by getting bot info."""
    bot_token, _ = validate_credentials()

    url = f"{TELEGRAM_API_BASE}{bot_token}/getMe"

    try:
        print("Testing Telegram connection...")
        response = requests.get(url, timeout=30)
        result = response.json()

        if result.get("ok"):
            bot_info = result.get("result", {})
            print(f"SUCCESS: Connected to bot @{bot_info.get('username', 'unknown')}")
            print(f"  Bot ID: {bot_info.get('id')}")
            print(f"  Name: {bot_info.get('first_name')}")

            # Also test sending a message
            now = now_et()
            test_msg = f"[TEST] Kobe Trading Bot connected at {now.strftime('%Y-%m-%d')} {fmt_ct(now)}"
            send_result = send_telegram_message(test_msg)

            if send_result.get("ok"):
                print("SUCCESS: Test message sent successfully")
                return True
            else:
                print(f"WARNING: Bot connected but message failed: {send_result.get('description')}")
                return False
        else:
            print(f"ERROR: Failed to connect: {result.get('description', 'Unknown error')}")
            log_alert("telegram_test", "Connection test", False, {"error": result.get("description")})
            return False

    except requests.RequestException as e:
        print(f"ERROR: Connection failed: {e}")
        log_alert("telegram_test", "Connection test", False, {"error": str(e)})
        return False


def send_trade_alert(
    symbol: str,
    side: str,
    action: str,
    price: float,
    quantity: int,
    reason: Optional[str] = None,
    pnl: Optional[float] = None
) -> bool:
    """
    Send a trade alert notification.

    Args:
        symbol: Ticker symbol
        side: 'long' or 'short'
        action: 'entry' or 'exit'
        price: Trade price
        quantity: Number of shares
        reason: Optional reason for trade
        pnl: Optional P&L for exit trades

    Returns:
        True if message sent successfully
    """
    emoji = "BUY" if side == "long" else "SELL"
    action_text = "ENTRY" if action == "entry" else "EXIT"

    lines = [
        f"<b>{emoji} {action_text}: {symbol}</b>",
        f"Side: {side.upper()}",
        f"Price: ${price:.2f}",
        f"Quantity: {quantity:,}",
    ]

    if reason:
        lines.append(f"Reason: {reason}")

    if pnl is not None:
        pnl_emoji = "+" if pnl >= 0 else ""
        lines.append(f"P&L: {pnl_emoji}${pnl:.2f}")

    lines.append(f"\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>")

    message = "\n".join(lines)
    result = send_telegram_message(message)

    return result.get("ok", False)


def send_error_alert(error_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> bool:
    """
    Send an error alert notification.

    Args:
        error_type: Type of error (e.g., 'API_ERROR', 'ORDER_FAILED')
        message: Error description
        details: Optional additional details

    Returns:
        True if message sent successfully
    """
    lines = [
        f"<b>ERROR: {error_type}</b>",
        f"{message}",
    ]

    if details:
        for key, value in details.items():
            lines.append(f"  {key}: {value}")

    lines.append(f"\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>")

    message_text = "\n".join(lines)
    result = send_telegram_message(message_text)

    log_alert("error_alert", f"{error_type}: {message}", result.get("ok", False), details)

    return result.get("ok", False)


def send_daily_summary(
    date: Optional[str] = None,
    total_pnl: float = 0.0,
    trade_count: int = 0,
    win_count: int = 0,
    loss_count: int = 0,
    positions: Optional[list] = None,
    account_value: Optional[float] = None
) -> bool:
    """
    Send daily P&L summary.

    Args:
        date: Date string (defaults to today)
        total_pnl: Total P&L for the day
        trade_count: Number of trades
        win_count: Number of winning trades
        loss_count: Number of losing trades
        positions: List of open positions
        account_value: Current account value

    Returns:
        True if message sent successfully
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    pnl_emoji = "+" if total_pnl >= 0 else ""
    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0

    lines = [
        f"<b>DAILY SUMMARY: {date}</b>",
        "",
        f"P&L: {pnl_emoji}${total_pnl:,.2f}",
        f"Trades: {trade_count} (W: {win_count}, L: {loss_count})",
        f"Win Rate: {win_rate:.1f}%",
    ]

    if account_value is not None:
        lines.append(f"Account Value: ${account_value:,.2f}")

    if positions:
        lines.append("")
        lines.append("<b>Open Positions:</b>")
        for pos in positions[:5]:  # Limit to 5 positions
            symbol = pos.get("symbol", "???")
            qty = pos.get("quantity", 0)
            unrealized = pos.get("unrealized_pnl", 0)
            pnl_str = f"+${unrealized:.2f}" if unrealized >= 0 else f"-${abs(unrealized):.2f}"
            lines.append(f"  {symbol}: {qty} shares ({pnl_str})")
        if len(positions) > 5:
            lines.append(f"  ... and {len(positions) - 5} more")

    now = now_et()
    lines.append(f"\n<i>Generated: {fmt_ct(now)}</i>")

    message = "\n".join(lines)
    result = send_telegram_message(message)

    log_alert("daily_summary", f"Summary for {date}", result.get("ok", False), {
        "pnl": total_pnl,
        "trades": trade_count
    })

    return result.get("ok", False)


def load_trading_data() -> Dict[str, Any]:
    """
    Load trading data from outputs directory for daily summary.
    Returns mock data if no real data available.
    """
    outputs_dir = PROJECT_ROOT / "outputs"

    # Try to find today's trades file
    today = now_et().strftime("%Y-%m-%d")
    trades_file = outputs_dir / f"trades_{today}.json"

    data = {
        "date": today,
        "total_pnl": 0.0,
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "positions": [],
        "account_value": None
    }

    if trades_file.exists():
        try:
            with open(trades_file, "r", encoding="utf-8") as f:
                trades = json.load(f)
                data["trade_count"] = len(trades)
                for t in trades:
                    pnl = t.get("pnl", 0)
                    data["total_pnl"] += pnl
                    if pnl >= 0:
                        data["win_count"] += 1
                    else:
                        data["loss_count"] += 1
        except Exception as e:
            print(f"Warning: Could not load trades file: {e}")

    # Try to load positions
    positions_file = outputs_dir / "positions.json"
    if positions_file.exists():
        try:
            with open(positions_file, "r", encoding="utf-8") as f:
                data["positions"] = json.load(f)
        except Exception:
            pass

    return data


def main():
    ap = argparse.ArgumentParser(
        description="Telegram notification service for Kobe trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python telegram.py --test
  python telegram.py --send "Trade executed: AAPL"
  python telegram.py --daily-summary
  python telegram.py --dotenv /path/to/.env --test
        """
    )
    ap.add_argument("--test", action="store_true", help="Test Telegram connection")
    ap.add_argument("--send", type=str, metavar="MESSAGE", help="Send a custom message")
    ap.add_argument("--daily-summary", action="store_true", help="Send daily P&L summary")
    ap.add_argument("--dotenv", type=str, default=str(DEFAULT_DOTENV),
                    help="Path to .env file (default: %(default)s)")

    args = ap.parse_args()

    # Load environment variables
    dotenv_path = Path(args.dotenv)
    if dotenv_path.exists():
        loaded = load_env(dotenv_path)
        print(f"Loaded {len(loaded)} env vars from {dotenv_path}")
    else:
        print(f"Note: .env file not found at {dotenv_path}")

    # Handle commands
    if args.test:
        success = test_connection()
        sys.exit(0 if success else 1)

    elif args.send:
        print(f"Sending message: {args.send[:50]}...")
        result = send_telegram_message(args.send)
        if result.get("ok"):
            print("SUCCESS: Message sent")
            sys.exit(0)
        else:
            print(f"ERROR: {result.get('description', 'Unknown error')}")
            sys.exit(1)

    elif args.daily_summary:
        print("Generating daily summary...")
        data = load_trading_data()
        success = send_daily_summary(
            date=data["date"],
            total_pnl=data["total_pnl"],
            trade_count=data["trade_count"],
            win_count=data["win_count"],
            loss_count=data["loss_count"],
            positions=data["positions"],
            account_value=data["account_value"]
        )
        if success:
            print("SUCCESS: Daily summary sent")
            sys.exit(0)
        else:
            print("ERROR: Failed to send daily summary")
            sys.exit(1)

    else:
        ap.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
