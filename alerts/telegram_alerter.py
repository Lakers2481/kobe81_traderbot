"""
Telegram Alert System for Kobe Trading Notifications.

Sends push notifications for trading events:
- Signals generated
- Orders submitted
- Fills received
- Gate rejections (summary)
- Kill switch triggered
- Daily summary

Setup:
1. Create bot via @BotFather on Telegram
2. Get bot token and your chat ID
3. Add to .env:
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   TELEGRAM_ALERTS_ENABLED=true
"""

import os
import urllib.request
import urllib.parse
import json
from datetime import datetime
from typing import Any, Optional, List, Dict


class TelegramAlerter:
    """
    Telegram notification system for trading alerts.

    Sends formatted messages to a Telegram chat via bot API.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize Telegram alerter.

        Args:
            bot_token: Telegram bot token (from @BotFather)
            chat_id: Target chat ID
            enabled: Whether alerts are enabled
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = enabled and os.getenv("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true"

        # Rate limiting
        self.last_message_time: Optional[datetime] = None
        self.min_interval_seconds = 1

        # Message queue for batching
        self.pending_messages: List[str] = []

        if self.enabled and (not self.bot_token or not self.chat_id):
            print("[WARN] Telegram alerts enabled but missing BOT_TOKEN or CHAT_ID")
            self.enabled = False

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send message via Telegram Bot API.

        Args:
            text: Message text (supports HTML formatting)
            parse_mode: "HTML" or "Markdown"

        Returns:
            True if sent successfully
        """
        return self._send_message_impl(text, parse_mode)

    def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Internal send (backward compatibility)."""
        return self._send_message_impl(text, parse_mode)

    def _send_message_impl(self, text: str, parse_mode: str = "HTML") -> bool:
        """Internal implementation of message sending."""
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }

            encoded_data = urllib.parse.urlencode(data).encode('utf-8')
            req = urllib.request.Request(url, data=encoded_data)

            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get("ok", False)

        except Exception as e:
            print(f"[WARN] Telegram send failed: {e}")
            return False

    def alert_signal(self, signal: Any) -> bool:
        """Alert on new trading signal."""
        direction_emoji = "+" if getattr(signal, 'direction', 'long') == "long" else "-"
        confidence_pct = getattr(signal, 'confidence', 0) * 100

        text = (
            f"{direction_emoji} <b>NEW SIGNAL</b>\n\n"
            f"<b>{signal.symbol}</b> {signal.direction.upper()}\n"
            f"Entry: ${signal.entry_price:.2f}\n"
            f"Stop: ${signal.stop_loss:.2f}\n"
            f"Target: ${signal.take_profit:.2f}\n"
            f"Confidence: {confidence_pct:.0f}%\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

        return self._send_message(text)

    def alert_order(self, symbol: str, shares: int, side: str, order_id: str) -> bool:
        """Alert on order submission."""
        text = (
            f"<b>ORDER SUBMITTED</b>\n\n"
            f"<b>{symbol}</b> {side} {shares} shares\n"
            f"Order ID: <code>{order_id[:12]}...</code>\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        return self._send_message(text)

    def alert_fill(self, symbol: str, shares: int, fill_price: float, side: str) -> bool:
        """Alert on order fill."""
        text = (
            f"<b>ORDER FILLED</b>\n\n"
            f"<b>{symbol}</b> {side} {shares} @ ${fill_price:.2f}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        return self._send_message(text)

    def alert_exit(self, symbol: str, pnl: float, exit_reason: str) -> bool:
        """Alert on position exit."""
        pnl_sign = "+" if pnl > 0 else ""

        text = (
            f"<b>POSITION CLOSED</b>\n\n"
            f"<b>{symbol}</b>\n"
            f"P&L: {pnl_sign}${pnl:.2f}\n"
            f"Reason: {exit_reason}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

        return self._send_message(text)

    def alert_kill_switch(self, reason: str, drawdown_pct: float) -> bool:
        """Alert on kill switch trigger."""
        text = (
            f"<b>KILL SWITCH TRIGGERED</b>\n\n"
            f"Reason: {reason}\n"
            f"Drawdown: {drawdown_pct:.1f}%\n"
            f"<b>Trading halted!</b>\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

        return self._send_message(text)

    def alert_gate_summary(self, rejections: Dict[str, int], total_signals: int) -> bool:
        """Alert on gate rejection summary."""
        total_rejections = sum(rejections.values())
        if total_rejections == 0:
            return False

        lines = ["<b>GATE REJECTIONS</b>\n"]
        lines.append(f"Signals: {total_signals}")
        lines.append(f"Rejected: {total_rejections}\n")

        for reason, count in rejections.items():
            if count > 0:
                lines.append(f"  - {reason}: {count}")

        return self._send_message("\n".join(lines))

    def alert_daily_summary(
        self,
        date: Any,
        signals: Optional[int] = None,
        orders: Optional[int] = None,
        fills: Optional[int] = None,
        daily_pnl: Optional[float] = None,
        equity: Optional[float] = None,
        total_pnl: Optional[float] = None,
        win_count: Optional[int] = None,
        loss_count: Optional[int] = None,
        signals_generated: Optional[int] = None,
        orders_submitted: Optional[int] = None,
        open_positions: Optional[int] = None,
    ) -> bool:
        """Send daily trading summary."""
        pnl = total_pnl if total_pnl is not None else (daily_pnl if daily_pnl is not None else 0.0)
        sig_count = signals_generated if signals_generated is not None else (signals if signals is not None else 0)
        ord_count = orders_submitted if orders_submitted is not None else (orders if orders is not None else 0)
        fill_count = fills if fills is not None else 0
        eq = equity if equity is not None else 0.0
        wins = win_count if win_count is not None else 0
        losses = loss_count if loss_count is not None else 0
        open_pos = open_positions if open_positions is not None else 0

        if hasattr(date, 'strftime'):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)

        pnl_sign = "+" if pnl >= 0 else ""

        text_lines = [
            "<b>DAILY SUMMARY</b>",
            f"Date: {date_str}",
            "",
            f"Signals: {sig_count}",
            f"Orders: {ord_count}",
        ]

        if fill_count > 0:
            text_lines.append(f"Fills: {fill_count}")

        if wins > 0 or losses > 0:
            text_lines.append(f"Wins: {wins} | Losses: {losses}")

        if open_pos > 0:
            text_lines.append(f"Open Positions: {open_pos}")

        text_lines.extend([
            "",
            f"Daily P&L: {pnl_sign}${pnl:.2f}",
            f"Equity: ${eq:,.2f}",
        ])

        return self._send_message("\n".join(text_lines))

    def alert_system_start(self, mode: str, symbols_count: int) -> bool:
        """Alert on system startup."""
        text = (
            f"<b>KOBE TRADING SYSTEM STARTED</b>\n\n"
            f"Mode: {mode}\n"
            f"Universe: {symbols_count} symbols\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return self._send_message(text)

    def alert_system_stop(self, reason: str = "User requested") -> bool:
        """Alert on system shutdown."""
        text = (
            f"<b>KOBE TRADING SYSTEM STOPPED</b>\n\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return self._send_message(text)

    def alert_error(self, error_type: str, message: str) -> bool:
        """Alert on critical error."""
        text = (
            f"<b>ERROR</b>\n\n"
            f"Type: {error_type}\n"
            f"Message: {message[:200]}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        return self._send_message(text)

    def send_alert(self, message: str, level: str = "INFO") -> bool:
        """Send a generic alert message."""
        level_prefix = {
            "INFO": "[INFO]",
            "WARNING": "[WARN]",
            "ERROR": "[ERROR]",
            "CRITICAL": "[CRITICAL]"
        }.get(level.upper(), "[ALERT]")

        text = (
            f"<b>{level_prefix}</b>\n\n"
            f"{message}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        return self._send_message(text)

    def alert_top3_ready(self, signals: List[Dict]) -> bool:
        """Alert on Top 3 Ready signals."""
        if not signals:
            return False

        text_lines = [
            "<b>TOP 3 READY SIGNALS</b>\n",
            "Signals meeting ALL entry conditions:\n"
        ]

        for idx, sig in enumerate(signals[:3], 1):
            symbol = sig.get("symbol", "???")
            score = sig.get("readiness_score", sig.get("confidence", 0))
            entry = sig.get("entry_price", 0)
            stop = sig.get("stop_loss", 0)
            target = sig.get("take_profit", 0)
            rr = sig.get("rr_ratio", 0)

            text_lines.append(f"<b>#{idx} {symbol}</b> (Score: {score})")
            text_lines.append(f"  Entry: ${entry:.2f}")
            text_lines.append(f"  Stop: ${stop:.2f} | Target: ${target:.2f}")
            text_lines.append(f"  R:R: {rr:.1f}:1")
            text_lines.append("")

        text_lines.append(f"Time: {datetime.now().strftime('%H:%M:%S')}")

        return self._send_message("\n".join(text_lines))

    def test_connection(self) -> bool:
        """Test Telegram connection."""
        if not self.enabled:
            print("[INFO] Telegram alerts disabled")
            return False

        return self._send_message("<b>Test Alert</b>\n\nKobe Telegram alerts working!")


# Module-level singleton
_alerter: Optional[TelegramAlerter] = None


def get_alerter() -> TelegramAlerter:
    """Get or create alerter singleton."""
    global _alerter
    if _alerter is None:
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass

        alerts_enabled = os.getenv("TELEGRAM_ALERTS_ENABLED", "").lower() == "true"
        telegram_enabled = os.getenv("TELEGRAM_ENABLED", "").lower() == "true"

        _alerter = TelegramAlerter(enabled=alerts_enabled or telegram_enabled)
    return _alerter
