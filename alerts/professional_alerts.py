"""
Professional Trading Alert System for Kobe.

Hedge-fund grade alerts for institutional presentation.
Sends comprehensive reports via Telegram including:
- EOD Summary (P&L, positions, hold times, expected moves)
- Position monitoring (hold time warnings, proximity alerts)
- Scan completion notifications
- System health reports
- 5W Analysis (Why, What, When, Where, Who)
"""

import os
from datetime import datetime
from typing import Any, Optional, Dict, List
from dataclasses import dataclass

from .telegram_alerter import get_alerter


@dataclass
class PositionReport:
    """Position data for reporting."""
    symbol: str
    side: str
    qty: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    days_held: int
    max_hold_days: int
    stop_loss: float
    take_profit: float
    entry_date: str
    atr: float = 0.0
    expected_move: float = 0.0


class ProfessionalAlerts:
    """
    Professional-grade trading alert system.

    Provides institutional-quality reporting for:
    - Daily P&L summaries
    - Position monitoring with hold time tracking
    - Risk alerts and warnings
    - Scan completion reports
    - System health monitoring
    """

    def __init__(self):
        """Initialize professional alerts."""
        self.alerter = get_alerter()
        self.max_hold_days = int(os.getenv("MAX_HOLD_DAYS", "5"))
        self.stop_loss_pct = float(os.getenv("STOP_LOSS_PCT", "2.0"))
        self.tp_r_multiple = float(os.getenv("TP_R_MULTIPLE", "2.0"))

    def _format_currency(self, amount: float) -> str:
        """Format currency with proper sign and commas."""
        sign = "+" if amount >= 0 else ""
        return f"{sign}${amount:,.2f}"

    def _format_pct(self, pct: float) -> str:
        """Format percentage with proper sign."""
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.2f}%"

    def _get_pnl_indicator(self, pnl: float) -> str:
        """Get indicator based on P&L."""
        if pnl > 0:
            return "[+]"
        elif pnl < 0:
            return "[-]"
        return "[=]"

    def send_eod_summary(
        self,
        date: str,
        daily_pnl: float,
        daily_pnl_pct: float,
        total_equity: float,
        starting_equity: float,
        positions: List[PositionReport],
        signals_today: int,
        orders_today: int,
        fills_today: int,
        scans_completed: int,
        drawdown_pct: float,
    ) -> bool:
        """Send comprehensive End-of-Day summary report."""
        pnl_indicator = self._get_pnl_indicator(daily_pnl)
        total_return = ((total_equity - starting_equity) / starting_equity) * 100

        lines = [
            "====================================",
            "<b>END OF DAY REPORT</b>",
            f"Date: {date}",
            "====================================",
            "",
            "<b>PERFORMANCE</b>",
            f"Daily P&L: {pnl_indicator} {self._format_currency(daily_pnl)} ({self._format_pct(daily_pnl_pct)})",
            f"Account Equity: ${total_equity:,.2f}",
            f"Total Return: {self._format_pct(total_return)}",
            f"Max Drawdown: {drawdown_pct:.2f}%",
            "",
        ]

        lines.extend([
            "<b>TODAY'S ACTIVITY</b>",
            f"Scans Completed: {scans_completed}",
            f"Signals Generated: {signals_today}",
            f"Orders Submitted: {orders_today}",
            f"Orders Filled: {fills_today}",
            "",
        ])

        if positions:
            lines.append(f"<b>OPEN POSITIONS ({len(positions)})</b>")
            lines.append("")

            for pos in positions:
                pos_indicator = self._get_pnl_indicator(pos.unrealized_pnl)
                hold_warning = ""

                if pos.days_held >= pos.max_hold_days:
                    hold_warning = " [!] MAX HOLD - EXIT NOW"
                elif pos.days_held >= pos.max_hold_days - 1:
                    hold_warning = " [!] Exit tomorrow"
                elif pos.days_held >= pos.max_hold_days - 3:
                    hold_warning = f" ({pos.max_hold_days - pos.days_held}d left)"

                sl_distance = ((pos.current_price - pos.stop_loss) / pos.current_price) * 100
                tp_distance = ((pos.take_profit - pos.current_price) / pos.current_price) * 100

                lines.extend([
                    f"<b>{pos.symbol}</b> {pos.side.upper()}",
                    f"  Entry: ${pos.entry_price:.2f} -> Now: ${pos.current_price:.2f}",
                    f"  P&L: {pos_indicator} {self._format_currency(pos.unrealized_pnl)} ({self._format_pct(pos.unrealized_pnl_pct)})",
                    f"  Hold: {pos.days_held}/{pos.max_hold_days} days{hold_warning}",
                    f"  SL: ${pos.stop_loss:.2f} ({sl_distance:.1f}% away)",
                    f"  TP: ${pos.take_profit:.2f} ({tp_distance:.1f}% to target)",
                ])

                if pos.expected_move > 0:
                    lines.append(f"  Expected Move: +/-${pos.expected_move:.2f} (1 ATR)")

                lines.append("")
        else:
            lines.extend([
                "<b>OPEN POSITIONS</b>",
                "No open positions",
                "",
            ])

        lines.extend([
            "====================================",
            f"Report generated: {datetime.now().strftime('%H:%M:%S')}",
            "====================================",
        ])

        return self.alerter._send_message("\n".join(lines))

    def send_hold_time_warning(
        self,
        symbol: str,
        days_held: int,
        max_hold: int,
        unrealized_pnl: float,
        current_price: float,
    ) -> bool:
        """Send hold time warning for position approaching max hold."""
        days_remaining = max_hold - days_held
        pnl_indicator = self._get_pnl_indicator(unrealized_pnl)

        if days_remaining <= 0:
            urgency = "[URGENT]"
            action = "EXIT IMMEDIATELY - Max hold reached"
        elif days_remaining == 1:
            urgency = "[WARNING]"
            action = "Exit by end of tomorrow"
        else:
            urgency = "[NOTICE]"
            action = f"Exit within {days_remaining} trading days"

        text = (
            f"{urgency} <b>HOLD TIME ALERT</b>\n\n"
            f"<b>{symbol}</b>\n"
            f"Days Held: {days_held}/{max_hold}\n"
            f"Current Price: ${current_price:.2f}\n"
            f"Unrealized P&L: {pnl_indicator} {self._format_currency(unrealized_pnl)}\n\n"
            f"<b>Action Required:</b> {action}"
        )

        return self.alerter._send_message(text)

    def send_stop_loss_proximity(
        self,
        symbol: str,
        current_price: float,
        stop_loss: float,
        distance_pct: float,
    ) -> bool:
        """Alert when price approaches stop loss."""
        text = (
            f"[!] <b>STOP LOSS PROXIMITY</b>\n\n"
            f"<b>{symbol}</b>\n"
            f"Current: ${current_price:.2f}\n"
            f"Stop Loss: ${stop_loss:.2f}\n"
            f"Distance: {distance_pct:.2f}%\n\n"
            f"<b>Monitor closely - stop may trigger soon</b>"
        )

        return self.alerter._send_message(text)

    def send_take_profit_proximity(
        self,
        symbol: str,
        current_price: float,
        take_profit: float,
        distance_pct: float,
    ) -> bool:
        """Alert when price approaches take profit."""
        text = (
            f"[TARGET] <b>APPROACHING TARGET</b>\n\n"
            f"<b>{symbol}</b>\n"
            f"Current: ${current_price:.2f}\n"
            f"Target: ${take_profit:.2f}\n"
            f"Distance: {distance_pct:.2f}%\n\n"
            f"<b>Position nearing profit target</b>"
        )

        return self.alerter._send_message(text)

    def send_scan_complete(
        self,
        scan_type: str,
        duration_seconds: float,
        universe_size: int,
        candidates_found: int,
        signals_found: int,
        top_signals: List[Dict[str, Any]],
    ) -> bool:
        """Send scan completion notification."""
        scan_names = {
            "weekly_prep": "Weekly Prep",
            "market_open": "Market Open",
            "mid_close": "Mid-Close",
            "post_close": "Post-Close",
        }
        scan_name = scan_names.get(scan_type, scan_type.title())

        lines = [
            f"<b>SCAN COMPLETE: {scan_name}</b>",
            "",
            f"Duration: {duration_seconds:.1f}s",
            f"Universe: {universe_size} stocks",
            f"Candidates: {candidates_found}",
            f"Signals: {signals_found}",
        ]

        if top_signals:
            lines.extend(["", "<b>Top Signals:</b>"])
            for i, sig in enumerate(top_signals[:5], 1):
                conf = sig.get('confidence', 0) * 100
                symbol = sig.get('symbol', 'N/A')
                entry = sig.get('entry_price', 0)
                lines.append(f"  {i}. {symbol} @ ${entry:.2f} ({conf:.0f}%)")
        elif signals_found == 0:
            lines.extend(["", "No actionable signals at this time."])

        lines.extend([
            "",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ])

        return self.alerter._send_message("\n".join(lines))

    def send_system_startup(
        self,
        mode: str,
        universe_size: int,
        equity: float,
        open_positions: int,
        data_freshness: str,
    ) -> bool:
        """Send system startup notification."""
        text = (
            "====================================\n"
            "<b>KOBE TRADING SYSTEM ONLINE</b>\n"
            "====================================\n\n"
            f"<b>Mode:</b> {mode.upper()}\n"
            f"<b>Universe:</b> {universe_size} stocks\n"
            f"<b>Equity:</b> ${equity:,.2f}\n"
            f"<b>Open Positions:</b> {open_positions}\n"
            f"<b>Data Freshness:</b> {data_freshness}\n\n"
            f"<b>Schedule:</b>\n"
            f"  - 09:35 ET - Market Open Scan\n"
            f"  - 10:30 ET - Mid-Morning Scan\n"
            f"  - 15:55 ET - Pre-Close Scan\n\n"
            f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "===================================="
        )

        return self.alerter._send_message(text)

    def send_system_shutdown(self, reason: str, equity: float, daily_pnl: float) -> bool:
        """Send system shutdown notification."""
        pnl_indicator = self._get_pnl_indicator(daily_pnl)

        text = (
            "====================================\n"
            "<b>KOBE TRADING SYSTEM OFFLINE</b>\n"
            "====================================\n\n"
            f"<b>Reason:</b> {reason}\n"
            f"<b>Final Equity:</b> ${equity:,.2f}\n"
            f"<b>Daily P&L:</b> {pnl_indicator} {self._format_currency(daily_pnl)}\n\n"
            f"Stopped: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "===================================="
        )

        return self.alerter._send_message(text)

    def send_critical_error(self, error_type: str, message: str, action: str) -> bool:
        """Send critical error alert."""
        text = (
            "[CRITICAL ERROR]\n\n"
            f"<b>Type:</b> {error_type}\n"
            f"<b>Message:</b> {message[:300]}\n\n"
            f"<b>Action Required:</b>\n{action}\n\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        return self.alerter._send_message(text)

    def send_signal_with_5w(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        why: str,
        what: str,
        when: str,
        where: str,
        who: str,
    ) -> bool:
        """Send signal with full 5W analysis."""
        direction_indicator = "[LONG]" if direction.upper() == "LONG" else "[SHORT]"
        conf_pct = confidence * 100
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        rr_ratio = reward / risk if risk > 0 else 0

        text = (
            "====================================\n"
            f"{direction_indicator} <b>NEW SIGNAL: {symbol}</b>\n"
            "====================================\n\n"
            f"<b>Direction:</b> {direction.upper()}\n"
            f"<b>Confidence:</b> {conf_pct:.0f}%\n\n"
            f"<b>LEVELS</b>\n"
            f"Entry: ${entry_price:.2f}\n"
            f"Stop Loss: ${stop_loss:.2f}\n"
            f"Take Profit: ${take_profit:.2f}\n"
            f"Risk/Reward: 1:{rr_ratio:.1f}\n\n"
            "====================================\n"
            "<b>5W ANALYSIS</b>\n"
            "====================================\n\n"
            f"<b>WHY:</b> {why}\n\n"
            f"<b>WHAT:</b> {what}\n\n"
            f"<b>WHEN:</b> {when}\n\n"
            f"<b>WHERE:</b> {where}\n\n"
            f"<b>WHO:</b> {who}\n\n"
            "====================================\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "===================================="
        )

        return self.alerter._send_message(text)

    def send_trade_entry(
        self,
        symbol: str,
        side: str,
        qty: int,
        fill_price: float,
        stop_loss: float,
        take_profit: float,
        position_size_usd: float,
        risk_amount: float,
    ) -> bool:
        """Send professional trade entry notification."""
        text = (
            "====================================\n"
            f"<b>TRADE EXECUTED</b>\n"
            "====================================\n\n"
            f"<b>{symbol}</b> {side} {qty} shares\n"
            f"Fill Price: ${fill_price:.2f}\n"
            f"Position Size: ${position_size_usd:,.2f}\n\n"
            f"<b>Risk Management:</b>\n"
            f"Stop Loss: ${stop_loss:.2f}\n"
            f"Take Profit: ${take_profit:.2f}\n"
            f"Risk Amount: ${risk_amount:.2f}\n\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "===================================="
        )

        return self.alerter._send_message(text)

    def send_trade_exit(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        exit_price: float,
        realized_pnl: float,
        realized_pnl_pct: float,
        exit_reason: str,
        hold_days: int,
    ) -> bool:
        """Send professional trade exit notification."""
        result = "WIN" if realized_pnl > 0 else "LOSS"

        text = (
            "====================================\n"
            f"<b>POSITION CLOSED - {result}</b>\n"
            "====================================\n\n"
            f"<b>{symbol}</b> {side} {qty} shares\n\n"
            f"<b>Trade Summary:</b>\n"
            f"Entry: ${entry_price:.2f}\n"
            f"Exit: ${exit_price:.2f}\n"
            f"P&L: {self._format_currency(realized_pnl)} ({self._format_pct(realized_pnl_pct)})\n"
            f"Hold Time: {hold_days} days\n"
            f"Exit Reason: {exit_reason}\n\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "===================================="
        )

        return self.alerter._send_message(text)


# Module-level singleton
_professional_alerter: Optional[ProfessionalAlerts] = None


def get_professional_alerter() -> ProfessionalAlerts:
    """Get or create professional alerter singleton."""
    global _professional_alerter
    if _professional_alerter is None:
        _professional_alerter = ProfessionalAlerts()
    return _professional_alerter
