"""
Telegram Trade Commander
========================

Interactive trade confirmation via Telegram.
Send TOTD cards with trade details, wait for user confirmation before execution.

Workflow:
1. Scanner generates TOTD signal
2. TelegramCommander sends trade card with details
3. User replies "YES" to confirm or "SKIP" to pass
4. On confirmation, order is submitted

Usage:
    from alerts.telegram_commander import TelegramCommander, get_commander

    commander = get_commander()

    # Send trade card and wait for confirmation
    confirm_id = commander.send_trade_card(signal)
    confirmed = commander.wait_for_confirmation(confirm_id, timeout_minutes=30)

    if confirmed:
        place_order(...)
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from uuid import uuid4
import logging

from alerts.telegram_alerter import TelegramAlerter, get_alerter
from core.clock.tz_utils import now_et, fmt_ct

logger = logging.getLogger(__name__)


@dataclass
class PendingTrade:
    """Represents a trade awaiting confirmation."""
    confirm_id: str
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    signal_data: Dict[str, Any]
    created_at: float
    message_id: Optional[int] = None


class TelegramCommander:
    """
    Trade confirmation handler for Telegram.

    Sends trade cards and tracks confirmation responses.
    """

    def __init__(
        self,
        alerter: Optional[TelegramAlerter] = None,
        timeout_minutes: int = 30,
    ):
        """
        Initialize TelegramCommander.

        Args:
            alerter: TelegramAlerter instance (uses singleton if None)
            timeout_minutes: Default timeout for confirmations
        """
        self.alerter = alerter or get_alerter()
        self.timeout_minutes = timeout_minutes
        self.pending_confirms: Dict[str, PendingTrade] = {}
        self.last_update_id = 0

    def send_trade_card(self, signal: Dict[str, Any]) -> str:
        """
        Send formatted trade card and return confirmation ID.

        Args:
            signal: Trade signal dict with symbol, side, entry_price, etc.

        Returns:
            confirm_id: Unique confirmation ID for tracking
        """
        confirm_id = uuid4().hex[:8].upper()

        symbol = signal.get("symbol", "???")
        side = signal.get("side", "long").upper()
        entry = signal.get("entry_price", 0)
        stop = signal.get("stop_loss", 0)
        target = signal.get("take_profit", 0)
        confidence = signal.get("confidence", signal.get("score", 0))
        if isinstance(confidence, (int, float)) and confidence > 1:
            confidence = confidence / 100.0  # Convert to decimal

        # Calculate R:R
        if side.upper() == "LONG":
            risk = entry - stop if stop < entry else 0
            reward = target - entry if target > entry else 0
        else:
            risk = stop - entry if stop > entry else 0
            reward = entry - target if target < entry else 0

        rr_ratio = reward / risk if risk > 0 else 0

        # Get additional context
        strategy = signal.get("strategy", "N/A")
        reason = signal.get("reason", "")
        spread_pct = signal.get("spread_pct", None)

        # Format message
        direction_emoji = "📈" if side.upper() == "LONG" else "📉"

        lines = [
            f"{direction_emoji} <b>TRADE CONFIRMATION</b> [{confirm_id}]",
            "",
            f"<b>{symbol}</b> {side}",
            f"Strategy: {strategy}",
            "",
            f"Entry: ${entry:.2f}",
            f"Stop: ${stop:.2f}",
            f"Target: ${target:.2f}",
            f"R:R: {rr_ratio:.1f}:1",
            f"Confidence: {confidence*100:.0f}%",
        ]

        if spread_pct is not None:
            lines.append(f"Spread: {spread_pct*100:.2f}%")

        if reason:
            lines.append(f"Reason: {reason[:50]}")

        lines.extend([
            "",
            f"⏰ Time: {fmt_ct(now_et())}",
            "",
            "Reply <b>YES</b> to execute or <b>SKIP</b> to pass",
            f"(Expires in {self.timeout_minutes} minutes)",
        ])

        text = "\n".join(lines)

        # Send message
        sent = self.alerter.send_message(text)

        if sent:
            # Track pending confirmation
            self.pending_confirms[confirm_id] = PendingTrade(
                confirm_id=confirm_id,
                symbol=symbol,
                side=side,
                entry_price=entry,
                stop_loss=stop,
                take_profit=target,
                confidence=confidence,
                signal_data=signal,
                created_at=time.time(),
            )
            logger.info(f"Trade card sent: {confirm_id} for {symbol}")
        else:
            logger.warning(f"Failed to send trade card for {symbol}")

        return confirm_id

    def get_updates(self, offset: int = 0) -> List[Dict]:
        """
        Get new messages from Telegram.

        Uses getUpdates API to poll for new messages.
        """
        if not self.alerter.enabled:
            return []

        try:
            url = f"https://api.telegram.org/bot{self.alerter.bot_token}/getUpdates"
            params = {
                "offset": offset,
                "timeout": 5,
                "allowed_updates": '["message"]',
            }
            query = urllib.parse.urlencode(params)
            full_url = f"{url}?{query}"

            req = urllib.request.Request(full_url)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))

            if data.get("ok"):
                return data.get("result", [])
            return []

        except Exception as e:
            logger.debug(f"Failed to get Telegram updates: {e}")
            return []

    def check_confirmations(self) -> Dict[str, Optional[bool]]:
        """
        Check for confirmation responses.

        Returns:
            Dict mapping confirm_id to:
                True - confirmed (YES)
                False - skipped (SKIP/NO)
                None - still pending
        """
        results = {}

        # Get new messages
        updates = self.get_updates(offset=self.last_update_id + 1)

        for update in updates:
            self.last_update_id = update.get("update_id", self.last_update_id)

            message = update.get("message", {})
            text = message.get("text", "").strip().upper()
            chat_id = str(message.get("chat", {}).get("id", ""))

            # Only process messages from our configured chat
            if chat_id != self.alerter.chat_id:
                continue

            # Check for confirmations
            if "YES" in text:
                # Find matching pending trade
                for cid, trade in list(self.pending_confirms.items()):
                    if cid in text or not results.get(cid):
                        results[cid] = True
                        del self.pending_confirms[cid]
                        logger.info(f"Trade {cid} CONFIRMED")
                        break

            elif "SKIP" in text or "NO" in text:
                for cid, trade in list(self.pending_confirms.items()):
                    if cid in text or not results.get(cid):
                        results[cid] = False
                        del self.pending_confirms[cid]
                        logger.info(f"Trade {cid} SKIPPED")
                        break

        # Check for expired trades
        now = time.time()
        for cid, trade in list(self.pending_confirms.items()):
            if now - trade.created_at > self.timeout_minutes * 60:
                results[cid] = False  # Expired = skip
                del self.pending_confirms[cid]
                logger.info(f"Trade {cid} EXPIRED")

        return results

    def wait_for_confirmation(
        self,
        confirm_id: str,
        timeout_minutes: Optional[int] = None,
        poll_interval_seconds: int = 5,
    ) -> bool:
        """
        Block and wait for confirmation of a specific trade.

        Args:
            confirm_id: The confirmation ID to wait for
            timeout_minutes: Override default timeout
            poll_interval_seconds: How often to poll for updates

        Returns:
            True if confirmed, False if skipped/expired
        """
        timeout = timeout_minutes or self.timeout_minutes
        deadline = time.time() + timeout * 60

        while time.time() < deadline:
            results = self.check_confirmations()

            if confirm_id in results:
                return results[confirm_id]

            # Also check if the confirm_id is no longer pending (expired elsewhere)
            if confirm_id not in self.pending_confirms:
                return False

            time.sleep(poll_interval_seconds)

        # Timeout
        if confirm_id in self.pending_confirms:
            del self.pending_confirms[confirm_id]

        return False

    def get_pending_count(self) -> int:
        """Get count of pending confirmations."""
        return len(self.pending_confirms)

    def cancel_pending(self, confirm_id: str) -> bool:
        """Cancel a pending confirmation."""
        if confirm_id in self.pending_confirms:
            del self.pending_confirms[confirm_id]
            return True
        return False

    def cancel_all_pending(self) -> int:
        """Cancel all pending confirmations."""
        count = len(self.pending_confirms)
        self.pending_confirms.clear()
        return count

    def send_confirmation_result(self, confirm_id: str, executed: bool, details: str = "") -> bool:
        """
        Send confirmation result back to user.

        Args:
            confirm_id: The confirmation ID
            executed: Whether trade was executed
            details: Additional details (order ID, fill price, etc.)
        """
        if executed:
            emoji = "✅"
            status = "EXECUTED"
        else:
            emoji = "❌"
            status = "NOT EXECUTED"

        lines = [
            f"{emoji} <b>Trade {confirm_id}: {status}</b>",
        ]

        if details:
            lines.append(f"{details}")

        lines.append(f"Time: {fmt_ct(now_et())}")

        return self.alerter.send_message("\n".join(lines))


# Global instance
_commander: Optional[TelegramCommander] = None


def get_commander(timeout_minutes: int = 30) -> TelegramCommander:
    """Get or create global TelegramCommander instance."""
    global _commander
    if _commander is None:
        _commander = TelegramCommander(timeout_minutes=timeout_minutes)
    return _commander


def send_trade_card_and_wait(
    signal: Dict[str, Any],
    timeout_minutes: int = 30,
) -> bool:
    """
    Convenience function: send trade card and wait for confirmation.

    Args:
        signal: Trade signal dict
        timeout_minutes: How long to wait for confirmation

    Returns:
        True if user confirmed, False otherwise
    """
    commander = get_commander(timeout_minutes)
    confirm_id = commander.send_trade_card(signal)
    return commander.wait_for_confirmation(confirm_id, timeout_minutes)
