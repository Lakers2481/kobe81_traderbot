"""
Trade Explainer for Signal Transparency
========================================

Generates human-readable explanations for trade signals.
Analyzes indicator states, market context, and strategy logic
to explain why a specific trade was triggered.

Supports multiple explanation levels from brief summaries
to detailed technical breakdowns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ExplanationLevel(Enum):
    """Level of detail in explanation."""
    BRIEF = "brief"           # One-line summary
    STANDARD = "standard"     # Key factors only
    DETAILED = "detailed"     # Full technical breakdown
    EXPERT = "expert"         # Include confidence metrics


class FactorType(Enum):
    """Types of explanation factors."""
    INDICATOR = "indicator"       # Technical indicator triggered
    PATTERN = "pattern"           # Price pattern detected
    REGIME = "regime"             # Market regime state
    FILTER = "filter"             # Filter condition met
    RISK = "risk"                 # Risk consideration
    TIMING = "timing"             # Time-based factor
    CONFIDENCE = "confidence"     # Model confidence


@dataclass
class ExplanationFactor:
    """A single factor contributing to the trade decision."""
    factor_type: FactorType
    name: str
    value: Any
    threshold: Optional[Any] = None
    contribution: float = 0.0  # -1 to 1, how much this influenced decision
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'type': self.factor_type.value,
            'name': self.name,
            'value': self.value,
            'threshold': self.threshold,
            'contribution': self.contribution,
            'description': self.description,
        }

    def to_sentence(self) -> str:
        """Convert to human-readable sentence."""
        if self.threshold is not None:
            if isinstance(self.value, float):
                return f"{self.name} at {self.value:.2f} (threshold: {self.threshold})"
            return f"{self.name} at {self.value} (threshold: {self.threshold})"
        if self.description:
            return self.description
        return f"{self.name}: {self.value}"


@dataclass
class TradeExplanation:
    """Complete explanation for a trade signal."""
    symbol: str
    side: str
    entry_price: float
    signal_time: datetime = field(default_factory=datetime.now)

    # Factors that influenced the decision
    primary_factors: List[ExplanationFactor] = field(default_factory=list)
    secondary_factors: List[ExplanationFactor] = field(default_factory=list)
    risk_factors: List[ExplanationFactor] = field(default_factory=list)

    # Summary
    headline: str = ""
    summary: str = ""
    confidence_score: float = 0.0

    # Strategy info
    strategy_name: str = ""
    strategy_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'signal_time': self.signal_time.isoformat(),
            'primary_factors': [f.to_dict() for f in self.primary_factors],
            'secondary_factors': [f.to_dict() for f in self.secondary_factors],
            'risk_factors': [f.to_dict() for f in self.risk_factors],
            'headline': self.headline,
            'summary': self.summary,
            'confidence_score': self.confidence_score,
            'strategy_name': self.strategy_name,
        }

    def to_brief(self) -> str:
        """Generate brief one-line explanation."""
        if self.headline:
            return self.headline
        factors = ", ".join(f.name for f in self.primary_factors[:2])
        return f"{self.side.upper()} {self.symbol} @ ${self.entry_price:.2f} - {factors}"

    def to_detailed(self) -> str:
        """Generate detailed explanation."""
        lines = [
            f"=== Trade Signal: {self.side.upper()} {self.symbol} ===",
            f"Entry Price: ${self.entry_price:.2f}",
            f"Time: {self.signal_time.strftime('%Y-%m-%d %H:%M')}",
            f"Strategy: {self.strategy_name}",
            f"Confidence: {self.confidence_score:.1%}",
            "",
            "PRIMARY FACTORS:",
        ]

        for factor in self.primary_factors:
            lines.append(f"  - {factor.to_sentence()}")

        if self.secondary_factors:
            lines.append("")
            lines.append("SUPPORTING FACTORS:")
            for factor in self.secondary_factors:
                lines.append(f"  - {factor.to_sentence()}")

        if self.risk_factors:
            lines.append("")
            lines.append("RISK CONSIDERATIONS:")
            for factor in self.risk_factors:
                lines.append(f"  - {factor.to_sentence()}")

        if self.summary:
            lines.append("")
            lines.append(f"SUMMARY: {self.summary}")

        return "\n".join(lines)


class TradeExplainer:
    """
    Generates explanations for trade signals.

    Analyzes the state of indicators, filters, and market conditions
    to construct human-readable explanations for trading decisions.
    """

    # Standard indicator descriptions
    INDICATOR_DESCRIPTIONS = {
        'rsi': "RSI (Relative Strength Index) measures momentum",
        'rsi_2': "RSI(2) ultra-short-term momentum indicator",
        'ibs': "IBS (Internal Bar Strength) shows where price closed in range",
        'sma': "SMA (Simple Moving Average) trend filter",
        'ema': "EMA (Exponential Moving Average) trend filter",
        'atr': "ATR (Average True Range) volatility measure",
        'macd': "MACD momentum and trend indicator",
        'stoch': "Stochastic oscillator momentum indicator",
        'volume_ratio': "Volume relative to average",
    }

    # Threshold descriptions
    THRESHOLD_MEANINGS = {
        'rsi': {
            'low': (0, 30, "oversold territory"),
            'mid': (30, 70, "neutral zone"),
            'high': (70, 100, "overbought territory"),
        },
        'ibs': {
            'low': (0, 0.2, "closed near the low"),
            'mid': (0.2, 0.8, "closed mid-range"),
            'high': (0.8, 1.0, "closed near the high"),
        },
    }

    def __init__(
        self,
        include_confidence: bool = True,
        default_level: ExplanationLevel = ExplanationLevel.STANDARD,
    ):
        """
        Initialize the trade explainer.

        Args:
            include_confidence: Whether to include confidence metrics
            default_level: Default explanation detail level
        """
        self.include_confidence = include_confidence
        self.default_level = default_level

        logger.info(f"TradeExplainer initialized with level={default_level.value}")

    def _describe_indicator(
        self,
        name: str,
        value: float,
        threshold: Optional[float] = None,
    ) -> str:
        """Generate description for an indicator value."""
        base_desc = self.INDICATOR_DESCRIPTIONS.get(
            name.lower(),
            f"{name} indicator"
        )

        # Add threshold context
        if name.lower() in self.THRESHOLD_MEANINGS:
            zones = self.THRESHOLD_MEANINGS[name.lower()]
            for zone_name, (low, high, meaning) in zones.items():
                if low <= value <= high:
                    return f"{base_desc}, {meaning} at {value:.2f}"

        if threshold is not None:
            if value < threshold:
                return f"{base_desc} below threshold at {value:.2f}"
            return f"{base_desc} above threshold at {value:.2f}"

        return f"{base_desc} at {value:.2f}"

    def _extract_factors_from_signal(
        self,
        signal: Dict[str, Any],
        indicators: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[ExplanationFactor], List[ExplanationFactor], List[ExplanationFactor]]:
        """Extract explanation factors from signal and indicators."""
        primary = []
        secondary = []
        risk = []

        indicators = indicators or {}

        # Extract from signal reason if available
        reason = signal.get('reason', '')
        if reason:
            primary.append(ExplanationFactor(
                factor_type=FactorType.PATTERN,
                name="Signal Reason",
                value=reason,
                contribution=1.0,
                description=reason,
            ))

        # Process indicators
        for name, value in indicators.items():
            name_lower = name.lower()

            # Determine factor type and importance
            if 'rsi' in name_lower or 'ibs' in name_lower or 'stoch' in name_lower:
                factor_type = FactorType.INDICATOR
                is_primary = True
            elif 'sma' in name_lower or 'ema' in name_lower:
                factor_type = FactorType.FILTER
                is_primary = False
            elif 'atr' in name_lower:
                factor_type = FactorType.RISK
                is_primary = False
            else:
                factor_type = FactorType.INDICATOR
                is_primary = False

            description = self._describe_indicator(name, value)

            factor = ExplanationFactor(
                factor_type=factor_type,
                name=name,
                value=value,
                description=description,
                contribution=0.5 if is_primary else 0.2,
            )

            if factor_type == FactorType.RISK:
                risk.append(factor)
            elif is_primary:
                primary.append(factor)
            else:
                secondary.append(factor)

        # Add confidence if available
        confidence = signal.get('confidence', signal.get('probability'))
        if confidence is not None and self.include_confidence:
            secondary.append(ExplanationFactor(
                factor_type=FactorType.CONFIDENCE,
                name="Model Confidence",
                value=confidence,
                contribution=0.3,
                description=f"Model confidence: {confidence:.1%}",
            ))

        return primary, secondary, risk

    def _generate_headline(
        self,
        symbol: str,
        side: str,
        primary_factors: List[ExplanationFactor],
    ) -> str:
        """Generate a headline for the trade."""
        if not primary_factors:
            return f"{side.upper()} signal on {symbol}"

        key_factor = primary_factors[0]
        if key_factor.factor_type == FactorType.INDICATOR:
            return f"{symbol}: {key_factor.name} triggered {side} signal"
        return f"{symbol}: {side.upper()} on {key_factor.description}"

    def _generate_summary(
        self,
        side: str,
        primary_factors: List[ExplanationFactor],
        secondary_factors: List[ExplanationFactor],
    ) -> str:
        """Generate a summary paragraph."""
        parts = []

        if primary_factors:
            factor_names = [f.name for f in primary_factors[:3]]
            parts.append(f"Signal triggered by {', '.join(factor_names)}.")

        if secondary_factors:
            supporting = [f.name for f in secondary_factors[:2]]
            parts.append(f"Supported by {', '.join(supporting)}.")

        if side == "long":
            parts.append("Bullish setup identified.")
        else:
            parts.append("Bearish setup identified.")

        return " ".join(parts)

    def explain(
        self,
        signal: Dict[str, Any],
        indicators: Optional[Dict[str, float]] = None,
        level: Optional[ExplanationLevel] = None,
    ) -> TradeExplanation:
        """
        Generate explanation for a trade signal.

        Args:
            signal: Trade signal dictionary with symbol, side, entry_price, etc.
            indicators: Current indicator values
            level: Explanation detail level

        Returns:
            TradeExplanation with factors and summary
        """
        level = level or self.default_level

        symbol = signal.get('symbol', 'UNKNOWN')
        side = signal.get('side', 'unknown')
        entry_price = signal.get('entry_price', 0.0)

        # Extract factors
        primary, secondary, risk = self._extract_factors_from_signal(
            signal, indicators
        )

        # Generate headline and summary
        headline = self._generate_headline(symbol, side, primary)
        summary = self._generate_summary(side, primary, secondary)

        # Get confidence
        confidence = signal.get('confidence', 0.5)
        if isinstance(confidence, (int, float)):
            confidence_score = float(confidence)
        else:
            confidence_score = 0.5

        explanation = TradeExplanation(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            primary_factors=primary,
            secondary_factors=secondary,
            risk_factors=risk,
            headline=headline,
            summary=summary,
            confidence_score=confidence_score,
            strategy_name=signal.get('strategy', ''),
        )

        logger.debug(f"Generated explanation for {symbol}: {headline}")

        return explanation


def explain_trade(
    signal: Dict[str, Any],
    indicators: Optional[Dict[str, float]] = None,
) -> TradeExplanation:
    """Convenience function to explain a trade signal."""
    explainer = TradeExplainer()
    return explainer.explain(signal, indicators)


# Module-level explainer instance
_explainer: Optional[TradeExplainer] = None


def get_explainer() -> TradeExplainer:
    """Get or create the global explainer instance."""
    global _explainer
    if _explainer is None:
        _explainer = TradeExplainer()
    return _explainer
