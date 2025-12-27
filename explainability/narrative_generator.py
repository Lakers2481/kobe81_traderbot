"""
Narrative Generator for Trading Reports
========================================

Generates human-readable narratives and reports from trading data.
Creates daily summaries, performance recaps, and strategy descriptions
in natural language.

Supports multiple styles from technical to casual.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from enum import Enum

logger = logging.getLogger(__name__)


class NarrativeStyle(Enum):
    """Style of narrative generation."""
    TECHNICAL = "technical"     # Professional, precise language
    CASUAL = "casual"           # Friendly, approachable language
    EXECUTIVE = "executive"     # High-level summary for executives
    DETAILED = "detailed"       # Comprehensive with all data


@dataclass
class Narrative:
    """A generated narrative text."""
    title: str
    content: str
    style: NarrativeStyle
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'content': self.content,
            'style': self.style.value,
            'generated_at': self.generated_at.isoformat(),
            'metadata': self.metadata,
        }


class NarrativeGenerator:
    """
    Generates natural language narratives from trading data.

    Creates various types of reports including daily summaries,
    trade explanations, and performance recaps.
    """

    # Performance thresholds for language
    PERFORMANCE_THRESHOLDS = {
        'excellent': 0.05,   # > 5% return
        'good': 0.02,        # > 2% return
        'moderate': 0.0,     # > 0% return
        'poor': -0.02,       # > -2% return
        'bad': -float('inf'),  # < -2% return
    }

    # Adjectives for different performance levels
    PERFORMANCE_ADJECTIVES = {
        'excellent': ['outstanding', 'exceptional', 'strong', 'impressive'],
        'good': ['solid', 'positive', 'favorable', 'good'],
        'moderate': ['modest', 'slight', 'marginal', 'small'],
        'poor': ['disappointing', 'weak', 'subpar', 'challenging'],
        'bad': ['difficult', 'tough', 'adverse', 'unfavorable'],
    }

    def __init__(
        self,
        default_style: NarrativeStyle = NarrativeStyle.TECHNICAL,
    ):
        """
        Initialize the narrative generator.

        Args:
            default_style: Default narrative style
        """
        self.default_style = default_style
        self._word_index = 0

        logger.info(f"NarrativeGenerator initialized with style={default_style.value}")

    def _get_performance_level(self, return_pct: float) -> str:
        """Determine performance level from return."""
        for level, threshold in self.PERFORMANCE_THRESHOLDS.items():
            if return_pct >= threshold:
                return level
        return 'bad'

    def _get_adjective(self, level: str) -> str:
        """Get an adjective for a performance level."""
        adjectives = self.PERFORMANCE_ADJECTIVES.get(level, [''])
        self._word_index = (self._word_index + 1) % len(adjectives)
        return adjectives[self._word_index]

    def _format_currency(self, value: float) -> str:
        """Format a currency value."""
        if abs(value) >= 1000:
            return f"${value:,.0f}"
        return f"${value:.2f}"

    def _format_percent(self, value: float) -> str:
        """Format a percentage."""
        return f"{value:+.2%}"

    def generate_daily_summary(
        self,
        date_str: str,
        trades: List[Dict[str, Any]],
        pnl: float,
        win_rate: Optional[float] = None,
        style: Optional[NarrativeStyle] = None,
    ) -> Narrative:
        """
        Generate a daily trading summary.

        Args:
            date_str: Date string (YYYY-MM-DD)
            trades: List of trades executed
            pnl: Total P&L for the day
            win_rate: Win rate if available
            style: Narrative style to use

        Returns:
            Generated Narrative
        """
        style = style or self.default_style
        n_trades = len(trades)

        # Determine performance level
        if n_trades > 0:
            avg_pnl = pnl / n_trades
            perf_level = self._get_performance_level(avg_pnl / 1000)  # Normalize
        else:
            perf_level = 'moderate'

        adjective = self._get_adjective(perf_level)

        # Build narrative based on style
        if style == NarrativeStyle.EXECUTIVE:
            if n_trades == 0:
                content = f"No trades executed on {date_str}."
            else:
                content = (
                    f"The trading system executed {n_trades} trade{'s' if n_trades > 1 else ''} "
                    f"on {date_str}, generating a {adjective} return of {self._format_currency(pnl)}."
                )
                if win_rate is not None:
                    content += f" Win rate: {win_rate:.0%}."

        elif style == NarrativeStyle.CASUAL:
            if n_trades == 0:
                content = f"Quiet day on {date_str} - no trades taken."
            else:
                if pnl >= 0:
                    emoji = "📈" if pnl > 100 else "✅"
                    content = (
                        f"{emoji} {adjective.capitalize()} day! Made {self._format_currency(pnl)} "
                        f"across {n_trades} trade{'s' if n_trades > 1 else ''}."
                    )
                else:
                    content = (
                        f"📉 Tough day with {self._format_currency(pnl)} across "
                        f"{n_trades} trade{'s' if n_trades > 1 else ''}."
                    )
                if win_rate is not None:
                    content += f" Hit {win_rate:.0%} of trades."

        elif style == NarrativeStyle.DETAILED:
            lines = [
                f"Daily Trading Summary - {date_str}",
                "=" * 40,
                f"Total Trades: {n_trades}",
                f"Net P&L: {self._format_currency(pnl)}",
            ]
            if win_rate is not None:
                lines.append(f"Win Rate: {win_rate:.1%}")
            if trades:
                lines.append("")
                lines.append("Trades:")
                for trade in trades[:5]:  # Show first 5
                    symbol = trade.get('symbol', '???')
                    side = trade.get('side', '?')
                    trade_pnl = trade.get('pnl', 0)
                    lines.append(f"  {symbol} ({side}): {self._format_currency(trade_pnl)}")
                if len(trades) > 5:
                    lines.append(f"  ... and {len(trades) - 5} more trades")
            content = "\n".join(lines)

        else:  # TECHNICAL
            if n_trades == 0:
                content = f"No trading activity recorded for {date_str}."
            else:
                content = (
                    f"Trading activity for {date_str}: {n_trades} execution{'s' if n_trades > 1 else ''} "
                    f"with net P&L of {self._format_currency(pnl)}."
                )
                if win_rate is not None:
                    content += f" Win rate: {win_rate:.1%}."

        return Narrative(
            title=f"Daily Summary - {date_str}",
            content=content,
            style=style,
            metadata={
                'date': date_str,
                'trades': n_trades,
                'pnl': pnl,
                'win_rate': win_rate,
            },
        )

    def generate_trade_narrative(
        self,
        trade: Dict[str, Any],
        style: Optional[NarrativeStyle] = None,
    ) -> Narrative:
        """
        Generate a narrative for a single trade.

        Args:
            trade: Trade dictionary
            style: Narrative style

        Returns:
            Generated Narrative
        """
        style = style or self.default_style

        symbol = trade.get('symbol', 'UNKNOWN')
        side = trade.get('side', 'unknown')
        entry = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price')
        pnl = trade.get('pnl', 0)
        reason = trade.get('reason', '')

        # Build narrative
        if style == NarrativeStyle.CASUAL:
            if pnl > 0:
                content = f"💰 Scored {self._format_currency(pnl)} on {symbol} {side}!"
            elif pnl < 0:
                content = f"😔 Lost {self._format_currency(abs(pnl))} on {symbol} {side}."
            else:
                content = f"Broke even on {symbol} {side}."
            if reason:
                content += f" Setup: {reason}"

        elif style == NarrativeStyle.EXECUTIVE:
            content = (
                f"{symbol}: {side.upper()} position "
                f"{'profitable' if pnl > 0 else 'closed at loss'} "
                f"({self._format_currency(pnl)})"
            )

        elif style == NarrativeStyle.DETAILED:
            lines = [
                f"Trade: {symbol} {side.upper()}",
                f"Entry: {self._format_currency(entry)}",
            ]
            if exit_price:
                lines.append(f"Exit: {self._format_currency(exit_price)}")
            lines.append(f"P&L: {self._format_currency(pnl)}")
            if reason:
                lines.append(f"Reason: {reason}")
            content = "\n".join(lines)

        else:  # TECHNICAL
            content = (
                f"Executed {side} on {symbol} at {self._format_currency(entry)}"
            )
            if exit_price:
                content += f", exited at {self._format_currency(exit_price)}"
            content += f". Net P&L: {self._format_currency(pnl)}."

        return Narrative(
            title=f"Trade: {symbol} {side.upper()}",
            content=content,
            style=style,
            metadata=trade,
        )

    def generate_performance_recap(
        self,
        period: str,
        total_pnl: float,
        total_trades: int,
        win_rate: float,
        sharpe: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        style: Optional[NarrativeStyle] = None,
    ) -> Narrative:
        """
        Generate a performance recap for a period.

        Args:
            period: Period description (e.g., "Week of Dec 23")
            total_pnl: Total P&L
            total_trades: Number of trades
            win_rate: Win rate
            sharpe: Sharpe ratio if available
            max_drawdown: Maximum drawdown if available
            style: Narrative style

        Returns:
            Generated Narrative
        """
        style = style or self.default_style

        perf_level = self._get_performance_level(total_pnl / max(1, total_trades) / 1000)
        adjective = self._get_adjective(perf_level)

        if style == NarrativeStyle.EXECUTIVE:
            content = (
                f"Performance for {period}: {adjective} results with "
                f"{self._format_currency(total_pnl)} net across {total_trades} trades. "
                f"Win rate of {win_rate:.0%}."
            )
            if sharpe is not None:
                content += f" Sharpe: {sharpe:.2f}."

        elif style == NarrativeStyle.CASUAL:
            if total_pnl >= 0:
                content = (
                    f"📊 {period} Recap: {adjective.capitalize()} week! "
                    f"Made {self._format_currency(total_pnl)} on {total_trades} trades. "
                    f"Batting {win_rate:.0%}."
                )
            else:
                content = (
                    f"📊 {period} Recap: {adjective.capitalize()} week. "
                    f"Down {self._format_currency(abs(total_pnl))} on {total_trades} trades."
                )

        elif style == NarrativeStyle.DETAILED:
            lines = [
                f"Performance Recap: {period}",
                "=" * 40,
                f"Net P&L: {self._format_currency(total_pnl)}",
                f"Total Trades: {total_trades}",
                f"Win Rate: {win_rate:.1%}",
            ]
            if sharpe is not None:
                lines.append(f"Sharpe Ratio: {sharpe:.2f}")
            if max_drawdown is not None:
                lines.append(f"Max Drawdown: {max_drawdown:.1%}")
            lines.append("")
            lines.append(f"Overall: {adjective.capitalize()} period.")
            content = "\n".join(lines)

        else:  # TECHNICAL
            content = (
                f"{period} performance: {total_trades} trades, "
                f"{self._format_currency(total_pnl)} net P&L, "
                f"{win_rate:.1%} win rate."
            )
            if sharpe is not None:
                content += f" Sharpe: {sharpe:.2f}."
            if max_drawdown is not None:
                content += f" Max DD: {max_drawdown:.1%}."

        return Narrative(
            title=f"Performance Recap - {period}",
            content=content,
            style=style,
            metadata={
                'period': period,
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
            },
        )


def generate_narrative(
    data: Dict[str, Any],
    narrative_type: str = "summary",
    style: NarrativeStyle = NarrativeStyle.TECHNICAL,
) -> Narrative:
    """Convenience function to generate a narrative."""
    generator = NarrativeGenerator(default_style=style)

    if narrative_type == "trade":
        return generator.generate_trade_narrative(data, style)
    elif narrative_type == "performance":
        return generator.generate_performance_recap(
            period=data.get('period', 'Period'),
            total_pnl=data.get('total_pnl', 0),
            total_trades=data.get('total_trades', 0),
            win_rate=data.get('win_rate', 0),
            sharpe=data.get('sharpe'),
            max_drawdown=data.get('max_drawdown'),
            style=style,
        )
    else:  # summary
        return generator.generate_daily_summary(
            date_str=data.get('date', datetime.now().strftime('%Y-%m-%d')),
            trades=data.get('trades', []),
            pnl=data.get('pnl', 0),
            win_rate=data.get('win_rate'),
            style=style,
        )


def generate_daily_summary(
    date_str: str,
    trades: List[Dict[str, Any]],
    pnl: float,
    **kwargs,
) -> Narrative:
    """Convenience function to generate daily summary."""
    generator = NarrativeGenerator()
    return generator.generate_daily_summary(date_str, trades, pnl, **kwargs)


# Module-level generator
_generator: Optional[NarrativeGenerator] = None


def get_generator() -> NarrativeGenerator:
    """Get or create the global generator instance."""
    global _generator
    if _generator is None:
        _generator = NarrativeGenerator()
    return _generator
