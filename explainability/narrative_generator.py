"""
Narrative Generator for Trading Reports
========================================

This module is responsible for generating human-readable narratives and reports
from structured trading data. It translates raw numbers (P&L, win rates, etc.)
into natural language summaries suitable for reports, logs, or user interfaces.

The generator supports multiple styles, allowing the output to be tailored to
different audiences, from a high-level executive summary to a casual,
emoji-filled update.

Usage:
    from explainability.narrative_generator import NarrativeGenerator, NarrativeStyle

    generator = NarrativeGenerator(default_style=NarrativeStyle.CASUAL)

    daily_trades = [{'symbol': 'AAPL', 'pnl': 150}, {'symbol': 'GOOG', 'pnl': -50}]
    pnl = 100.0
    win_rate = 0.5

    # Generate a casual daily summary
    narrative = generator.generate_daily_summary(
        date_str="2025-12-27",
        trades=daily_trades,
        pnl=pnl,
        win_rate=win_rate
    )
    print(narrative.content)
    # Output: "✅ Solid day! Made $100.00 across 2 trades. Hit 50% of trades."
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import random

logger = logging.getLogger(__name__)


class NarrativeStyle(Enum):
    """Enumerates the different stylistic options for the generated narrative."""
    TECHNICAL = "technical"     # Professional, precise, data-focused language.
    CASUAL = "casual"           # Friendly, approachable language, may include emojis.
    EXECUTIVE = "executive"     # High-level, concise summary for stakeholders.
    DETAILED = "detailed"       # Comprehensive, multi-line report with all available data.


@dataclass
class Narrative:
    """A structured object containing a piece of generated text."""
    title: str
    content: str
    style: NarrativeStyle
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict) # The data used to generate the narrative.

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the narrative to a dictionary."""
        return {
            'title': self.title,
            'content': self.content,
            'style': self.style.value,
            'generated_at': self.generated_at.isoformat(),
            'metadata': self.metadata,
        }


class NarrativeGenerator:
    """
    A class that generates natural language narratives from structured trading data.
    """

    # --- Templating Components ---
    PERFORMANCE_THRESHOLDS = {
        'excellent': 0.05,   # For returns > 5%
        'good': 0.02,        # For returns > 2%
        'moderate': 0.0,     # For returns > 0%
        'poor': -0.02,       # For returns > -2%
        'bad': -float('inf'),# For returns < -2%
    }
    PERFORMANCE_ADJECTIVES = {
        'excellent': ['outstanding', 'exceptional', 'strong', 'impressive'],
        'good': ['solid', 'positive', 'favorable', 'good'],
        'moderate': ['modest', 'slight', 'marginal', 'flat'],
        'poor': ['disappointing', 'weak', 'subpar', 'challenging'],
        'bad': ['difficult', 'tough', 'adverse', 'unfavorable'],
    }

    def __init__(self, default_style: NarrativeStyle = NarrativeStyle.TECHNICAL):
        self.default_style = default_style
        logger.info(f"NarrativeGenerator initialized with default style: {self.default_style.value}")

    def _get_performance_level(self, return_pct: float) -> str:
        """Determines a qualitative performance level based on a percentage return."""
        for level, threshold in self.PERFORMANCE_THRESHOLDS.items():
            if return_pct >= threshold:
                return level
        return 'bad'

    def _get_adjective(self, level: str) -> str:
        """Selects a random adjective for a given performance level to add variety."""
        adjectives = self.PERFORMANCE_ADJECTIVES.get(level, [''])
        return random.choice(adjectives)

    def _format_currency(self, value: float) -> str:
        """Formats a float as a currency string."""
        return f"${value:,.2f}"

    def _format_percent(self, value: float) -> str:
        """Formats a float as a percentage string."""
        return f"{value:+.2%}" if value != 0 else f"{value:.2%}"

    def generate_daily_summary(
        self,
        date_str: str,
        trades: List[Dict[str, Any]],
        pnl: float,
        win_rate: Optional[float] = None,
        style: Optional[NarrativeStyle] = None,
    ) -> Narrative:
        """
        Generates a narrative summary for a single day of trading.

        Args:
            date_str: The date for the summary (e.g., "2025-12-27").
            trades: A list of trade dictionaries executed on that day.
            pnl: The total net profit or loss for the day.
            win_rate: The win rate for the day, if available.
            style: The narrative style to use.

        Returns:
            A `Narrative` object containing the generated summary.
        """
        style = style or self.default_style
        n_trades = len(trades)
        
        # Determine performance level based on P&L.
        # Here we make a simple assumption that average P&L per trade is a proxy for return.
        avg_pnl_per_trade = pnl / n_trades if n_trades > 0 else 0
        # Normalize by an arbitrary trade size (e.g., $1000) to get a "return"
        pseudo_return_pct = avg_pnl_per_trade / 10000 
        perf_level = self._get_performance_level(pseudo_return_pct)
        adjective = self._get_adjective(perf_level)

        # --- Construct content based on the chosen style ---
        title = f"Daily Summary - {date_str}"
        content = ""

        if n_trades == 0:
            content = f"No trading activity was recorded on {date_str}."
        elif style == NarrativeStyle.EXECUTIVE:
            content = f"The system executed {n_trades} trade{'s' if n_trades != 1 else ''} on {date_str}, resulting in a {adjective} net P&L of {self._format_currency(pnl)}."
            if win_rate is not None: content += f" The win rate was {win_rate:.1%}."
        elif style == NarrativeStyle.CASUAL:
            if pnl > 0: content = f"✅ A {adjective} day! Banked {self._format_currency(pnl)} across {n_trades} trade{'s' if n_trades != 1 else ''}."
            else: content = f"📉 A {adjective} day. Net loss of {self._format_currency(abs(pnl))} across {n_trades} trade{'s' if n_trades != 1 else ''}."
            if win_rate is not None: content += f" Win rate was {win_rate:.1%}."
        elif style == NarrativeStyle.DETAILED:
            lines = [f"{title}\n" + "="*len(title), f"Total Trades: {n_trades}", f"Net P&L: {self._format_currency(pnl)}"]
            if win_rate is not None: lines.append(f"Win Rate: {win_rate:.1%}")
            if trades:
                lines.append("\nTop Trades:")
                for trade in sorted(trades, key=lambda t: t.get('pnl', 0), reverse=True)[:3]:
                    lines.append(f"  - {trade.get('symbol', '?')} ({trade.get('side', '?')}): {self._format_currency(trade.get('pnl',0))}")
            content = "\n".join(lines)
        else: # TECHNICAL (default)
            content = f"Trading activity for {date_str}: {n_trades} execution{'s' if n_trades != 1 else ''} resulted in a net P&L of {self._format_currency(pnl)}."
            if win_rate is not None: content += f" The session win rate was {win_rate:.1%}."

        return Narrative(title=title, content=content, style=style, metadata={'date': date_str, 'trades': n_trades, 'pnl': pnl})

    def generate_trade_narrative(self, trade: Dict[str, Any], style: Optional[NarrativeStyle] = None) -> Narrative:
        """Generates a narrative for a single trade."""
        style = style or self.default_style
        symbol, side, pnl = trade.get('symbol', '?'), trade.get('side', '?'), trade.get('pnl', 0)
        title = f"Trade Analysis: {symbol} {side.upper()}"
        content = ""

        if style == NarrativeStyle.CASUAL:
            if pnl > 0: content = f"💰 Nice win on {symbol}! Scored {self._format_currency(pnl)}."
            else: content = f"😔 Took a loss of {self._format_currency(abs(pnl))} on {symbol}."
        else: # TECHNICAL, EXECUTIVE, DETAILED are similar for a single trade.
            content = f"Executed {side.upper()} on {symbol} resulting in a P&L of {self._format_currency(pnl)}."
            if reason := trade.get('reason'): content += f" Reason: {reason}."
            
        return Narrative(title=title, content=content, style=style, metadata=trade)

    def generate_performance_recap(self, period: str, total_pnl: float, total_trades: int, win_rate: float, **kwargs) -> Narrative:
        """Generates a performance recap over a specified period."""
        style = kwargs.get('style', self.default_style)
        
        # Determine performance level based on P&L relative to trades.
        pseudo_return_pct = (total_pnl / total_trades) / 10000 if total_trades > 0 else 0
        perf_level = self._get_performance_level(pseudo_return_pct)
        adjective = self._get_adjective(perf_level)
        
        title = f"Performance Recap - {period}"
        content = ""

        if style == NarrativeStyle.EXECUTIVE:
            content = f"The {period} was {adjective}, with a net P&L of {self._format_currency(total_pnl)} from {total_trades} trades. Win rate stood at {win_rate:.1%}."
        elif style == NarrativeStyle.CASUAL:
            if total_pnl > 0: content = f"📊 Solid period! We're up {self._format_currency(total_pnl)} over the {period}."
            else: content = f"📊 A {adjective} period, down {self._format_currency(abs(total_pnl))}."
            content += f" Took {total_trades} trades with a {win_rate:.1%} win rate."
        else: # TECHNICAL / DETAILED
            lines = [f"{title}\n" + "="*len(title), f"Net P&L: {self._format_currency(total_pnl)}", f"Total Trades: {total_trades}", f"Win Rate: {win_rate:.1%}"]
            if sharpe := kwargs.get('sharpe'): lines.append(f"Sharpe Ratio: {sharpe:.2f}")
            if max_dd := kwargs.get('max_drawdown'): lines.append(f"Max Drawdown: {self._format_percent(max_dd)}")
            content = "\n".join(lines)
            
        return Narrative(title=title, content=content, style=style, metadata={'period': period, 'pnl': total_pnl})


# --- Convenience Functions ---
_generator_instance: Optional[NarrativeGenerator] = None

def get_generator() -> NarrativeGenerator:
    """Factory function to get the singleton instance of the NarrativeGenerator."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = NarrativeGenerator()
    return _generator_instance

def generate_narrative(data: Dict[str, Any], narrative_type: str, style: NarrativeStyle = NarrativeStyle.TECHNICAL) -> Narrative:
    """
    A high-level convenience function to generate any type of narrative.
    
    Args:
        data: The dictionary containing the source data.
        narrative_type: The type of narrative to generate ('summary', 'trade', 'performance').
        style: The desired narrative style.
    """
    generator = get_generator()
    generator.default_style = style
    
    if narrative_type == "trade":
        return generator.generate_trade_narrative(data)
    elif narrative_type == "performance":
        return generator.generate_performance_recap(**data)
    else: # Default to daily summary
        return generator.generate_daily_summary(**data)
