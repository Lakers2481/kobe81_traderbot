"""
Daily Digest - Comprehensive Reports

Generates daily/weekly reports summarizing all trading activity
and system health.

Features:
- P&L summary
- Trade review
- System health
- Alerts summary
- Recommendations

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from core.structured_log import get_logger

logger = get_logger(__name__)


@dataclass
class DigestReport:
    """Daily/weekly digest report."""
    report_date: date
    report_type: str                # daily, weekly, monthly

    # P&L
    gross_pnl: float
    net_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    pnl_by_strategy: Dict[str, float]

    # Trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float

    # Positions
    open_positions: int
    position_value: float
    cash_available: float

    # Risk
    max_drawdown: float
    current_drawdown: float
    risk_budget_used: float
    circuit_breakers_tripped: List[str]

    # System
    system_health: str              # HEALTHY, DEGRADED, UNHEALTHY
    uptime_pct: float
    alerts_count: int
    critical_alerts: int

    # Recommendations
    recommendations: List[str]
    next_actions: List[str]

    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_date": self.report_date.isoformat(),
            "report_type": self.report_type,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "pnl_by_strategy": self.pnl_by_strategy,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "open_positions": self.open_positions,
            "position_value": self.position_value,
            "cash_available": self.cash_available,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "risk_budget_used": self.risk_budget_used,
            "circuit_breakers_tripped": self.circuit_breakers_tripped,
            "system_health": self.system_health,
            "uptime_pct": self.uptime_pct,
            "alerts_count": self.alerts_count,
            "critical_alerts": self.critical_alerts,
            "recommendations": self.recommendations,
            "next_actions": self.next_actions,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# {self.report_type.title()} Trading Report",
            f"**Date:** {self.report_date}",
            f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            "---",
            "",
            "## Performance Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Gross P&L | ${self.gross_pnl:+,.2f} |",
            f"| Net P&L | ${self.net_pnl:+,.2f} |",
            f"| Realized | ${self.realized_pnl:+,.2f} |",
            f"| Unrealized | ${self.unrealized_pnl:+,.2f} |",
            "",
            "## Trade Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Trades | {self.total_trades} |",
            f"| Winners | {self.winning_trades} |",
            f"| Losers | {self.losing_trades} |",
            f"| Win Rate | {self.win_rate:.1%} |",
            f"| Avg Win | ${self.average_win:,.2f} |",
            f"| Avg Loss | ${self.average_loss:,.2f} |",
            f"| Largest Win | ${self.largest_win:,.2f} |",
            f"| Largest Loss | ${self.largest_loss:,.2f} |",
            "",
        ]

        if self.pnl_by_strategy:
            lines.append("## P&L by Strategy")
            lines.append("")
            for strategy, pnl in sorted(self.pnl_by_strategy.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"- **{strategy}**: ${pnl:+,.2f}")
            lines.append("")

        lines.extend([
            "## Portfolio Status",
            "",
            f"- Open Positions: {self.open_positions}",
            f"- Position Value: ${self.position_value:,.2f}",
            f"- Cash Available: ${self.cash_available:,.2f}",
            "",
            "## Risk Metrics",
            "",
            f"- Max Drawdown: {self.max_drawdown:.1%}",
            f"- Current Drawdown: {self.current_drawdown:.1%}",
            f"- Risk Budget Used: {self.risk_budget_used:.0%}",
            "",
        ])

        if self.circuit_breakers_tripped:
            lines.append("**Circuit Breakers Tripped:**")
            for breaker in self.circuit_breakers_tripped:
                lines.append(f"- {breaker}")
            lines.append("")

        lines.extend([
            "## System Health",
            "",
            f"- Status: **{self.system_health}**",
            f"- Uptime: {self.uptime_pct:.1%}",
            f"- Alerts: {self.alerts_count} ({self.critical_alerts} critical)",
            "",
        ])

        if self.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        if self.next_actions:
            lines.append("## Next Actions")
            lines.append("")
            for action in self.next_actions:
                lines.append(f"- [ ] {action}")
            lines.append("")

        lines.append("---")
        lines.append("*Generated by Kobe Guardian System*")

        return "\n".join(lines)

    def to_telegram(self) -> str:
        """Generate Telegram summary."""
        emoji = {
            "HEALTHY": "green",
            "DEGRADED": "yellow",
            "UNHEALTHY": "red",
        }

        pnl_emoji = "green" if self.net_pnl >= 0 else "red"

        lines = [
            f"**{self.report_type.upper()} REPORT** - {self.report_date}",
            "",
            f"**P&L:** ${self.net_pnl:+,.2f}",
            f"**Trades:** {self.total_trades} ({self.win_rate:.0%} WR)",
            f"**Drawdown:** {self.current_drawdown:.1%}",
            f"**System:** {self.system_health}",
            "",
        ]

        if self.recommendations:
            lines.append(f"_{self.recommendations[0]}_")

        return "\n".join(lines)


class DailyDigest:
    """
    Generate comprehensive daily/weekly digests.

    Features:
    - Aggregates data from all systems
    - Generates recommendations
    - Saves to file
    - Sends alerts
    """

    REPORTS_DIR = Path("reports/digests")

    def __init__(self):
        """Initialize digest generator."""
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def _gather_pnl_data(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Gather P&L data for the period."""
        # In production, would query actual P&L sources
        return {
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "by_strategy": {},
        }

    def _gather_trade_data(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Gather trade data for the period."""
        # In production, would query trade log
        return {
            "total": 0,
            "winners": 0,
            "losers": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
        }

    def _gather_position_data(self) -> Dict[str, Any]:
        """Gather current position data."""
        # In production, would query position state
        return {
            "count": 0,
            "value": 0.0,
            "cash": 50000.0,
        }

    def _gather_risk_data(self) -> Dict[str, Any]:
        """Gather risk metrics."""
        # In production, would query risk systems
        return {
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "risk_budget_used": 0.0,
            "breakers_tripped": [],
        }

    def _gather_system_data(self) -> Dict[str, Any]:
        """Gather system health data."""
        # In production, would query system monitor
        return {
            "health": "HEALTHY",
            "uptime_pct": 99.9,
            "alerts_count": 5,
            "critical_alerts": 0,
        }

    def _generate_recommendations(
        self,
        pnl: Dict,
        trades: Dict,
        risk: Dict,
        system: Dict,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # P&L based
        if pnl.get("net_pnl", 0) < -500:
            recommendations.append("Review losing trades for pattern identification")

        # Trade based
        win_rate = trades.get("winners", 0) / max(trades.get("total", 1), 1)
        if win_rate < 0.5:
            recommendations.append("Win rate below 50% - consider tightening entry criteria")

        # Risk based
        if risk.get("current_drawdown", 0) > 0.03:
            recommendations.append("Drawdown elevated - consider reducing position sizes")

        if risk.get("breakers_tripped"):
            recommendations.append("Circuit breakers tripped - review risk parameters")

        # System based
        if system.get("health") != "HEALTHY":
            recommendations.append("System health degraded - check component status")

        if not recommendations:
            recommendations.append("System operating within normal parameters")

        return recommendations

    def _generate_next_actions(
        self,
        pnl: Dict,
        trades: Dict,
        positions: Dict,
    ) -> List[str]:
        """Generate next action items."""
        actions = []

        # Always suggest review
        actions.append("Review overnight watchlist for tomorrow")

        if positions.get("count", 0) > 0:
            actions.append("Check stop losses on open positions")

        if pnl.get("unrealized_pnl", 0) > 500:
            actions.append("Consider taking partial profits")

        actions.append("Run system health check before market open")

        return actions

    def generate(
        self,
        report_date: Optional[date] = None,
        report_type: str = "daily",
    ) -> DigestReport:
        """
        Generate a digest report.

        Args:
            report_date: Date for the report
            report_type: daily, weekly, or monthly

        Returns:
            DigestReport
        """
        if report_date is None:
            report_date = date.today()

        # Determine date range
        if report_type == "daily":
            start_date = report_date
            end_date = report_date
        elif report_type == "weekly":
            start_date = report_date - timedelta(days=6)
            end_date = report_date
        else:  # monthly
            start_date = report_date.replace(day=1)
            end_date = report_date

        # Gather data
        pnl = self._gather_pnl_data(start_date, end_date)
        trades = self._gather_trade_data(start_date, end_date)
        positions = self._gather_position_data()
        risk = self._gather_risk_data()
        system = self._gather_system_data()

        # Generate recommendations
        recommendations = self._generate_recommendations(pnl, trades, risk, system)
        next_actions = self._generate_next_actions(pnl, trades, positions)

        # Calculate win rate
        total_trades = trades.get("total", 0)
        win_rate = trades.get("winners", 0) / max(total_trades, 1)

        report = DigestReport(
            report_date=report_date,
            report_type=report_type,
            gross_pnl=pnl.get("gross_pnl", 0),
            net_pnl=pnl.get("net_pnl", 0),
            realized_pnl=pnl.get("realized_pnl", 0),
            unrealized_pnl=pnl.get("unrealized_pnl", 0),
            pnl_by_strategy=pnl.get("by_strategy", {}),
            total_trades=total_trades,
            winning_trades=trades.get("winners", 0),
            losing_trades=trades.get("losers", 0),
            win_rate=win_rate,
            average_win=trades.get("avg_win", 0),
            average_loss=trades.get("avg_loss", 0),
            largest_win=trades.get("largest_win", 0),
            largest_loss=trades.get("largest_loss", 0),
            open_positions=positions.get("count", 0),
            position_value=positions.get("value", 0),
            cash_available=positions.get("cash", 0),
            max_drawdown=risk.get("max_drawdown", 0),
            current_drawdown=risk.get("current_drawdown", 0),
            risk_budget_used=risk.get("risk_budget_used", 0),
            circuit_breakers_tripped=risk.get("breakers_tripped", []),
            system_health=system.get("health", "UNKNOWN"),
            uptime_pct=system.get("uptime_pct", 0),
            alerts_count=system.get("alerts_count", 0),
            critical_alerts=system.get("critical_alerts", 0),
            recommendations=recommendations,
            next_actions=next_actions,
        )

        # Save report
        self._save_report(report)

        return report

    def _save_report(self, report: DigestReport) -> None:
        """Save report to file."""
        try:
            # JSON
            json_file = self.REPORTS_DIR / f"{report.report_type}_{report.report_date.isoformat()}.json"
            with open(json_file, "w") as f:
                json.dump(report.to_dict(), f, indent=2)

            # Markdown
            md_file = self.REPORTS_DIR / f"{report.report_type}_{report.report_date.isoformat()}.md"
            with open(md_file, "w") as f:
                f.write(report.to_markdown())

            logger.info(f"Saved {report.report_type} report: {json_file}")

        except Exception as e:
            logger.error(f"Failed to save report: {e}")


# Convenience function
def generate_daily_digest() -> DigestReport:
    """Generate today's daily digest."""
    return DailyDigest().generate(report_type="daily")


if __name__ == "__main__":
    # Demo
    digest = DailyDigest()

    print("=== Daily Digest Demo ===\n")

    report = digest.generate()

    print(report.to_markdown())

    print("\n--- Telegram Format ---")
    print(report.to_telegram())
