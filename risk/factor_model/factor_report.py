"""
Factor Risk Reporter - Comprehensive Factor Exposure Reports

Generates reports on portfolio factor exposures and concentration risks.

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from core.structured_log import get_logger
from .factor_calculator import FactorExposures, get_factor_calculator
from .sector_exposure import SectorAnalyzer, SectorExposures

logger = get_logger(__name__)


@dataclass
class FactorRiskReport:
    """Comprehensive factor risk report."""
    date: date
    factor_exposures: FactorExposures
    sector_exposures: SectorExposures
    overall_risk_level: str           # LOW, MODERATE, ELEVATED, HIGH, CRITICAL
    risk_flags: List[str]
    concentration_issues: List[str]
    factor_drift: Dict[str, float]
    recommendations: List[str]
    generated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "factor_exposures": self.factor_exposures.to_dict(),
            "sector_exposures": self.sector_exposures.to_dict(),
            "overall_risk_level": self.overall_risk_level,
            "risk_flags": self.risk_flags,
            "concentration_issues": self.concentration_issues,
            "factor_drift": self.factor_drift,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Factor Risk Report",
            f"**Date:** {self.date}",
            f"**Risk Level:** {self.overall_risk_level}",
            "",
            "---",
            "",
        ]

        # Factor exposures
        lines.extend([
            "## Factor Exposures",
            "",
            "| Factor | Exposure | Interpretation |",
            "|--------|----------|----------------|",
        ])

        fe = self.factor_exposures
        factors = [
            ("Market Beta", fe.market_beta, "1.0 = market neutral"),
            ("Size", fe.size_exposure, "- = Large, + = Small"),
            ("Value", fe.value_exposure, "- = Growth, + = Value"),
            ("Momentum", fe.momentum_exposure, "- = Reversal, + = Trend"),
            ("Volatility", fe.volatility_exposure, "- = High Vol, + = Low Vol"),
            ("Quality", fe.quality_exposure, "- = Junk, + = Quality"),
        ]

        for name, value, interp in factors:
            lines.append(f"| {name} | {value:+.2f} | {interp} |")

        lines.append("")

        # Concentration
        lines.extend([
            "## Concentration",
            "",
            f"- **Top 5 Weight:** {fe.top_5_weight:.1%}",
            f"- **Effective N:** {fe.effective_n:.1f}",
            f"- **Max Sector:** {self.sector_exposures.max_sector} ({self.sector_exposures.max_concentration:.1%})",
            "",
        ])

        # Risk flags
        if self.risk_flags:
            lines.extend([
                "## Risk Flags",
                "",
            ])
            for flag in self.risk_flags:
                lines.append(f"- **{flag}**")
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)

    def to_telegram_summary(self) -> str:
        """Generate Telegram summary."""
        emoji = {
            "LOW": "green",
            "MODERATE": "green",
            "ELEVATED": "yellow",
            "HIGH": "orange",
            "CRITICAL": "red",
        }.get(self.overall_risk_level, "white")

        lines = [
            f"**Factor Risk: {self.overall_risk_level}**",
            "",
            f"Beta: {self.factor_exposures.market_beta:.2f}",
            f"Top 5: {self.factor_exposures.top_5_weight:.0%}",
            f"Max Sector: {self.sector_exposures.max_concentration:.0%}",
        ]

        if self.risk_flags:
            lines.append(f"\nFlags: {len(self.risk_flags)}")

        if self.recommendations:
            lines.append(f"\n_{self.recommendations[0]}_")

        return "\n".join(lines)


class FactorRiskReporter:
    """
    Generate comprehensive factor risk reports.

    Features:
    - Factor exposure analysis
    - Sector concentration analysis
    - Risk level assessment
    - Actionable recommendations
    """

    REPORTS_DIR = Path("reports/factor_risk")

    def __init__(self):
        """Initialize reporter."""
        self._factor_calc = get_factor_calculator()
        self._sector_analyzer = SectorAnalyzer()

        # Ensure directory exists
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def _assess_risk_level(
        self,
        factor_exp: FactorExposures,
        sector_exp: SectorExposures,
    ) -> str:
        """Assess overall risk level."""
        risk_score = 0

        # Factor risks
        if abs(factor_exp.market_beta - 1.0) > 0.5:
            risk_score += 2
        if factor_exp.top_5_weight > 0.60:
            risk_score += 2
        if factor_exp.effective_n < 5:
            risk_score += 2
        if abs(factor_exp.momentum_exposure) > 0.5:
            risk_score += 1

        # Sector risks
        if sector_exp.max_concentration > 0.50:
            risk_score += 3
        elif sector_exp.max_concentration > 0.40:
            risk_score += 2
        elif sector_exp.max_concentration > 0.30:
            risk_score += 1

        # Convert to level
        if risk_score >= 6:
            return "CRITICAL"
        elif risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "ELEVATED"
        elif risk_score >= 1:
            return "MODERATE"
        else:
            return "LOW"

    def _generate_recommendations(
        self,
        factor_exp: FactorExposures,
        sector_exp: SectorExposures,
        risk_flags: List[str],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Beta recommendations
        if factor_exp.market_beta > 1.3:
            recommendations.append(
                f"High beta ({factor_exp.market_beta:.2f}) - consider hedging "
                "with SPY puts or reducing high-beta positions"
            )
        elif factor_exp.market_beta < 0.5:
            recommendations.append(
                f"Low beta ({factor_exp.market_beta:.2f}) - portfolio may "
                "underperform in strong bull markets"
            )

        # Concentration recommendations
        if factor_exp.top_5_weight > 0.60:
            recommendations.append(
                f"Top 5 positions at {factor_exp.top_5_weight:.0%} - "
                "consider spreading across more names"
            )

        if sector_exp.max_concentration > 0.40:
            recommendations.append(
                f"{sector_exp.max_sector} at {sector_exp.max_concentration:.0%} - "
                f"consider hedging with short sector ETF or reducing exposure"
            )

        # Factor tilt recommendations
        if abs(factor_exp.momentum_exposure) > 0.5:
            direction = "long" if factor_exp.momentum_exposure > 0 else "short"
            recommendations.append(
                f"Strong momentum tilt ({direction}) - "
                "be aware of reversal risk if market regime changes"
            )

        if not recommendations:
            recommendations.append(
                "Portfolio risk within normal parameters. Continue monitoring."
            )

        return recommendations

    def generate_report(
        self,
        positions: List[Dict],
        report_date: Optional[date] = None,
    ) -> FactorRiskReport:
        """
        Generate comprehensive factor risk report.

        Args:
            positions: List of position dicts
            report_date: Date for report

        Returns:
            FactorRiskReport
        """
        if report_date is None:
            report_date = date.today()

        # Calculate exposures
        factor_exp = self._factor_calc.calculate_exposures(positions)
        sector_exp = self._sector_analyzer.analyze(positions)

        # Get risk flags
        risk_flags = factor_exp.get_risk_flags()

        # Concentration issues
        concentration_issues = []
        if not sector_exp.is_balanced:
            concentration_issues.append(
                f"{sector_exp.max_sector}: {sector_exp.max_concentration:.1%}"
            )
        if factor_exp.top_5_weight > 0.50:
            concentration_issues.append(
                f"Top 5: {factor_exp.top_5_weight:.1%}"
            )

        # Factor drift
        factor_drift = self._factor_calc.get_exposure_drift()

        # Assess risk level
        risk_level = self._assess_risk_level(factor_exp, sector_exp)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            factor_exp, sector_exp, risk_flags
        )

        report = FactorRiskReport(
            date=report_date,
            factor_exposures=factor_exp,
            sector_exposures=sector_exp,
            overall_risk_level=risk_level,
            risk_flags=risk_flags,
            concentration_issues=concentration_issues,
            factor_drift=factor_drift,
            recommendations=recommendations,
            generated_at=datetime.now(),
        )

        # Save report
        report_file = self.REPORTS_DIR / f"factor_risk_{report_date.isoformat()}.json"
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        md_file = self.REPORTS_DIR / f"factor_risk_{report_date.isoformat()}.md"
        with open(md_file, "w") as f:
            f.write(report.to_markdown())

        logger.info(f"Generated factor risk report: {report_file}")

        return report


# Convenience function
def generate_factor_report(positions: List[Dict]) -> FactorRiskReport:
    """Generate factor risk report for positions."""
    return FactorRiskReporter().generate_report(positions)


if __name__ == "__main__":
    # Demo
    reporter = FactorRiskReporter()

    print("=== Factor Risk Report Demo ===\n")

    positions = [
        {"symbol": "AAPL", "shares": 100, "current_price": 175, "market_cap": "large", "momentum_score": 0.7},
        {"symbol": "MSFT", "shares": 50, "current_price": 380, "market_cap": "large", "momentum_score": 0.6},
        {"symbol": "NVDA", "shares": 30, "current_price": 480, "market_cap": "large", "momentum_score": 0.8},
        {"symbol": "AMD", "shares": 80, "current_price": 120, "market_cap": "large", "momentum_score": 0.75},
        {"symbol": "TSLA", "shares": 40, "current_price": 250, "market_cap": "large", "momentum_score": 0.65},
    ]

    report = reporter.generate_report(positions)

    print(f"Risk Level: {report.overall_risk_level}")
    print(f"Risk Flags: {report.risk_flags}")
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")

    print("\n--- Telegram Summary ---")
    print(report.to_telegram_summary())
