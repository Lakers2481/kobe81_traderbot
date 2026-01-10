"""
Attribution Report Generator - Daily/Weekly P&L Attribution Reports

Generates comprehensive reports explaining WHERE your P&L came from:
- By strategy
- By factor
- By sector
- By position

In plain English that a solo trader can understand.

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from core.structured_log import get_logger
from .daily_pnl import get_daily_pnl_tracker
from .factor_attribution import FactorAttributor
from .strategy_attribution import StrategyAttributor

logger = get_logger(__name__)


@dataclass
class AttributionReport:
    """Complete P&L attribution report."""
    period: str                       # "daily" or "weekly"
    start_date: date
    end_date: date
    total_pnl: float
    by_strategy: Dict[str, float]
    by_factor: Dict[str, float]
    by_sector: Dict[str, float]
    costs: Dict[str, float]           # slippage, commissions
    alpha_pnl: float                  # Unexplained by factors
    highlights: List[str]             # Key takeaways
    concerns: List[str]               # Warnings/issues
    recommendations: List[str]
    generated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_pnl": self.total_pnl,
            "by_strategy": self.by_strategy,
            "by_factor": self.by_factor,
            "by_sector": self.by_sector,
            "costs": self.costs,
            "alpha_pnl": self.alpha_pnl,
            "highlights": self.highlights,
            "concerns": self.concerns,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        direction = "up" if self.total_pnl >= 0 else "down"
        emoji = "+" if self.total_pnl >= 0 else ""

        lines = [
            f"# P&L Attribution Report - {self.period.title()}",
            f"**Period:** {self.start_date} to {self.end_date}",
            f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            "---",
            "",
            f"## Total P&L: **{emoji}${self.total_pnl:,.2f}** ({direction})",
            "",
        ]

        # By Strategy
        if self.by_strategy:
            lines.extend([
                "### By Strategy",
                "| Strategy | P&L | Contribution |",
                "|----------|-----|--------------|",
            ])
            total_abs = sum(abs(v) for v in self.by_strategy.values()) or 1
            for strat, pnl in sorted(self.by_strategy.items(), key=lambda x: x[1], reverse=True):
                emoji = "+" if pnl >= 0 else ""
                contrib = abs(pnl) / total_abs * 100
                lines.append(f"| {strat} | {emoji}${pnl:,.2f} | {contrib:.1f}% |")
            lines.append("")

        # By Factor
        if self.by_factor:
            lines.extend([
                "### By Factor",
                "| Factor | P&L | Contribution |",
                "|--------|-----|--------------|",
            ])
            for factor, pnl in sorted(self.by_factor.items(), key=lambda x: abs(x[1]), reverse=True):
                emoji = "+" if pnl >= 0 else ""
                lines.append(f"| {factor.title()} | {emoji}${pnl:,.2f} | |")
            lines.append(f"| **Alpha (skill)** | ${self.alpha_pnl:,.2f} | |")
            lines.append("")

        # By Sector
        if self.by_sector:
            lines.extend([
                "### By Sector",
                "| Sector | P&L |",
                "|--------|-----|",
            ])
            for sector, pnl in sorted(self.by_sector.items(), key=lambda x: x[1], reverse=True):
                emoji = "+" if pnl >= 0 else ""
                lines.append(f"| {sector} | {emoji}${pnl:,.2f} |")
            lines.append("")

        # Costs
        if self.costs:
            lines.extend([
                "### Trading Costs",
                f"- Slippage: ${self.costs.get('slippage', 0):,.2f}",
                f"- Commissions: ${self.costs.get('commissions', 0):,.2f}",
                f"- **Total Costs:** ${sum(self.costs.values()):,.2f}",
                "",
            ])

        # Highlights
        if self.highlights:
            lines.extend([
                "### Highlights",
            ])
            for h in self.highlights:
                lines.append(f"- {h}")
            lines.append("")

        # Concerns
        if self.concerns:
            lines.extend([
                "### Concerns",
            ])
            for c in self.concerns:
                lines.append(f"- **{c}**")
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.extend([
                "### Recommendations",
            ])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)

    def to_telegram_summary(self) -> str:
        """Generate short Telegram summary."""
        direction = "up" if self.total_pnl >= 0 else "down"
        emoji = "+" if self.total_pnl >= 0 else ""

        lines = [
            f"**Daily P&L: {emoji}${self.total_pnl:,.2f}**",
            "",
        ]

        # Top strategy
        if self.by_strategy:
            best_strat = max(self.by_strategy.items(), key=lambda x: x[1])
            worst_strat = min(self.by_strategy.items(), key=lambda x: x[1])
            lines.append(f"Best: {best_strat[0]} (+${best_strat[1]:,.2f})")
            if worst_strat[1] < 0:
                lines.append(f"Worst: {worst_strat[0]} (${worst_strat[1]:,.2f})")

        lines.append("")

        # Key recommendation
        if self.recommendations:
            lines.append(f"_{self.recommendations[0]}_")

        return "\n".join(lines)


class AttributionReporter:
    """
    Generate comprehensive P&L attribution reports.

    Features:
    - Daily and weekly reports
    - Multi-dimensional attribution (strategy, factor, sector)
    - Plain English explanations
    - Telegram integration
    """

    REPORTS_DIR = Path("reports/attribution")

    def __init__(self):
        """Initialize reporter."""
        self._pnl_tracker = get_daily_pnl_tracker()
        self._factor_attributor = FactorAttributor()
        self._strategy_attributor = StrategyAttributor()

        # Ensure directory exists
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def _generate_highlights(
        self,
        total_pnl: float,
        by_strategy: Dict[str, float],
        by_sector: Dict[str, float],
    ) -> List[str]:
        """Generate key highlights."""
        highlights = []

        if total_pnl > 0:
            highlights.append(f"Profitable day: +${total_pnl:,.2f}")
        elif total_pnl < 0:
            highlights.append(f"Loss day: ${total_pnl:,.2f}")

        # Best strategy
        if by_strategy:
            best = max(by_strategy.items(), key=lambda x: x[1])
            if best[1] > 0:
                highlights.append(f"{best[0]} led with +${best[1]:,.2f}")

        # Best sector
        if by_sector:
            best = max(by_sector.items(), key=lambda x: x[1])
            if best[1] > 0:
                highlights.append(f"{best[0]} sector contributed +${best[1]:,.2f}")

        return highlights

    def _generate_concerns(
        self,
        total_pnl: float,
        by_strategy: Dict[str, float],
        costs: Dict[str, float],
    ) -> List[str]:
        """Generate warnings and concerns."""
        concerns = []

        # Big loss
        if total_pnl < -500:
            concerns.append(f"Significant loss of ${abs(total_pnl):,.2f}")

        # Strategy dragging
        if by_strategy:
            worst = min(by_strategy.items(), key=lambda x: x[1])
            if worst[1] < -200:
                concerns.append(f"{worst[0]} lost ${abs(worst[1]):,.2f}")

        # High costs
        total_costs = sum(costs.values())
        if total_costs > 100:
            concerns.append(f"High trading costs: ${total_costs:,.2f}")

        return concerns

    def _generate_recommendations(
        self,
        total_pnl: float,
        by_strategy: Dict[str, float],
        by_factor: Dict[str, float],
        alpha_pnl: float,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Strategy recommendations
        if by_strategy:
            worst = min(by_strategy.items(), key=lambda x: x[1])
            best = max(by_strategy.items(), key=lambda x: x[1])

            if worst[1] < 0 and best[1] > abs(worst[1]):
                recommendations.append(
                    f"Consider increasing allocation to {best[0]} "
                    f"(+${best[1]:,.2f}) vs {worst[0]} (${worst[1]:,.2f})"
                )

        # Alpha recommendation
        if alpha_pnl > 0:
            recommendations.append(
                f"Good alpha generation: ${alpha_pnl:,.2f} "
                "of P&L is from skill, not factors"
            )
        elif alpha_pnl < -100:
            recommendations.append(
                f"Negative alpha: ${alpha_pnl:,.2f} - "
                "review trade selection process"
            )

        # Factor recommendations
        if by_factor:
            market_contrib = by_factor.get("market", 0)
            if abs(market_contrib) > abs(total_pnl) * 0.7:
                recommendations.append(
                    "High market factor exposure - "
                    "consider hedging or reducing beta"
                )

        if not recommendations:
            recommendations.append("Continue current strategy - within normal parameters")

        return recommendations

    def generate_daily_report(
        self,
        report_date: Optional[date] = None,
    ) -> AttributionReport:
        """
        Generate daily attribution report.

        Args:
            report_date: Date to report on (defaults to today)

        Returns:
            AttributionReport
        """
        if report_date is None:
            report_date = date.today()

        # Get daily P&L
        daily_pnl = self._pnl_tracker.get_daily_pnl(report_date)

        # Get factor decomposition
        factor_decomp = self._factor_attributor.decompose(
            total_pnl=daily_pnl.net_pnl,
            positions=[p.to_dict() for p in daily_pnl.by_position],
            for_date=report_date,
        )

        # Get strategy comparison
        strategy_comp = self._strategy_attributor.compare_strategies(days=1)

        # Build report
        by_strategy = daily_pnl.by_strategy
        by_factor = {
            f: fp.attributed_pnl
            for f, fp in factor_decomp.factor_pnl.items()
        }
        by_sector = daily_pnl.by_sector
        costs = {
            "slippage": daily_pnl.slippage_cost,
            "commissions": daily_pnl.commission_cost,
        }

        highlights = self._generate_highlights(
            daily_pnl.net_pnl, by_strategy, by_sector
        )
        concerns = self._generate_concerns(
            daily_pnl.net_pnl, by_strategy, costs
        )
        recommendations = self._generate_recommendations(
            daily_pnl.net_pnl, by_strategy, by_factor, factor_decomp.alpha_pnl
        )

        report = AttributionReport(
            period="daily",
            start_date=report_date,
            end_date=report_date,
            total_pnl=daily_pnl.net_pnl,
            by_strategy=by_strategy,
            by_factor=by_factor,
            by_sector=by_sector,
            costs=costs,
            alpha_pnl=factor_decomp.alpha_pnl,
            highlights=highlights,
            concerns=concerns,
            recommendations=recommendations,
            generated_at=datetime.now(),
        )

        # Save report
        report_file = self.REPORTS_DIR / f"daily_{report_date.isoformat()}.json"
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        md_file = self.REPORTS_DIR / f"daily_{report_date.isoformat()}.md"
        with open(md_file, "w") as f:
            f.write(report.to_markdown())

        logger.info(f"Generated daily attribution report: {report_file}")

        return report

    def generate_weekly_report(
        self,
        week_end_date: Optional[date] = None,
    ) -> AttributionReport:
        """Generate weekly attribution report."""
        if week_end_date is None:
            week_end_date = date.today()

        week_start_date = week_end_date - timedelta(days=6)

        # Aggregate 7 days of data
        total_pnl = 0.0
        by_strategy: Dict[str, float] = {}
        by_sector: Dict[str, float] = {}
        costs = {"slippage": 0.0, "commissions": 0.0}

        for i in range(7):
            d = week_start_date + timedelta(days=i)
            daily = self._pnl_tracker.get_daily_pnl(d)

            total_pnl += daily.net_pnl
            costs["slippage"] += daily.slippage_cost
            costs["commissions"] += daily.commission_cost

            for strat, pnl in daily.by_strategy.items():
                by_strategy[strat] = by_strategy.get(strat, 0) + pnl

            for sector, pnl in daily.by_sector.items():
                by_sector[sector] = by_sector.get(sector, 0) + pnl

        # Get factor decomposition for week
        factor_decomp = self._factor_attributor.decompose(
            total_pnl=total_pnl,
            positions=[],  # Would need to aggregate
        )

        by_factor = {
            f: fp.attributed_pnl
            for f, fp in factor_decomp.factor_pnl.items()
        }

        highlights = self._generate_highlights(total_pnl, by_strategy, by_sector)
        concerns = self._generate_concerns(total_pnl, by_strategy, costs)
        recommendations = self._generate_recommendations(
            total_pnl, by_strategy, by_factor, factor_decomp.alpha_pnl
        )

        report = AttributionReport(
            period="weekly",
            start_date=week_start_date,
            end_date=week_end_date,
            total_pnl=total_pnl,
            by_strategy=by_strategy,
            by_factor=by_factor,
            by_sector=by_sector,
            costs=costs,
            alpha_pnl=factor_decomp.alpha_pnl,
            highlights=highlights,
            concerns=concerns,
            recommendations=recommendations,
            generated_at=datetime.now(),
        )

        # Save report
        report_file = self.REPORTS_DIR / f"weekly_{week_end_date.isoformat()}.json"
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        md_file = self.REPORTS_DIR / f"weekly_{week_end_date.isoformat()}.md"
        with open(md_file, "w") as f:
            f.write(report.to_markdown())

        logger.info(f"Generated weekly attribution report: {report_file}")

        return report

    def send_telegram_summary(self, report: AttributionReport) -> None:
        """Send report summary to Telegram."""
        try:
            from alerts.telegram_alerts import send_alert

            send_alert(report.to_telegram_summary(), level="info")

        except Exception as e:
            logger.warning(f"Failed to send Telegram summary: {e}")


# Convenience functions
def generate_daily_attribution() -> AttributionReport:
    """Generate today's attribution report."""
    return AttributionReporter().generate_daily_report()


def generate_weekly_attribution() -> AttributionReport:
    """Generate this week's attribution report."""
    return AttributionReporter().generate_weekly_report()


if __name__ == "__main__":
    # Demo
    reporter = AttributionReporter()

    print("=== Attribution Reporter Demo ===\n")

    report = reporter.generate_daily_report()

    print("Telegram Summary:")
    print(report.to_telegram_summary())

    print("\n--- Full Markdown Report (truncated) ---")
    print(report.to_markdown()[:1000] + "...")
