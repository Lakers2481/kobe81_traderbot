"""
Rule Generator for Trading Strategies
======================================

Generates new trading rules from patterns discovered in data.
Uses template-based rule construction with indicator combinations.

Supports automatic discovery of entry/exit conditions that
show statistical significance in historical data.
"""

from __future__ import annotations

import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ConditionOperator(Enum):
    """Operators for rule conditions."""
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    EQUALS = "=="
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"


class IndicatorType(Enum):
    """Types of technical indicators."""
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PRICE = "price"


@dataclass
class RuleCondition:
    """A single condition in a trading rule."""
    indicator: str
    operator: ConditionOperator
    threshold: Any
    indicator_type: IndicatorType = IndicatorType.MOMENTUM
    lookback: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'indicator': self.indicator,
            'operator': self.operator.value,
            'threshold': self.threshold,
            'indicator_type': self.indicator_type.value,
            'lookback': self.lookback,
        }

    def to_expression(self) -> str:
        """Convert to human-readable expression."""
        if self.operator in [ConditionOperator.CROSSES_ABOVE, ConditionOperator.CROSSES_BELOW]:
            return f"{self.indicator} {self.operator.value} {self.threshold}"
        return f"{self.indicator} {self.operator.value} {self.threshold}"

    def evaluate(self, df: pd.DataFrame, row_idx: int = -1) -> bool:
        """
        Evaluate the condition on data.

        Args:
            df: DataFrame with indicator columns
            row_idx: Row index to evaluate (-1 for last row)

        Returns:
            True if condition is met
        """
        if self.indicator not in df.columns:
            return False

        try:
            current = df[self.indicator].iloc[row_idx]

            if self.operator == ConditionOperator.LESS_THAN:
                return current < self.threshold
            elif self.operator == ConditionOperator.LESS_EQUAL:
                return current <= self.threshold
            elif self.operator == ConditionOperator.GREATER_THAN:
                return current > self.threshold
            elif self.operator == ConditionOperator.GREATER_EQUAL:
                return current >= self.threshold
            elif self.operator == ConditionOperator.EQUALS:
                return current == self.threshold
            elif self.operator == ConditionOperator.CROSSES_ABOVE:
                if row_idx == 0 or abs(row_idx) >= len(df):
                    return False
                prev = df[self.indicator].iloc[row_idx - 1]
                return prev < self.threshold and current >= self.threshold
            elif self.operator == ConditionOperator.CROSSES_BELOW:
                if row_idx == 0 or abs(row_idx) >= len(df):
                    return False
                prev = df[self.indicator].iloc[row_idx - 1]
                return prev > self.threshold and current <= self.threshold
        except (IndexError, KeyError):
            return False

        return False


@dataclass
class TradingRule:
    """A complete trading rule with entry/exit conditions."""
    name: str
    entry_conditions: List[RuleCondition] = field(default_factory=list)
    exit_conditions: List[RuleCondition] = field(default_factory=list)
    side: str = "long"  # "long" or "short"
    confidence: float = 0.0
    historical_win_rate: Optional[float] = None
    historical_trades: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'entry_conditions': [c.to_dict() for c in self.entry_conditions],
            'exit_conditions': [c.to_dict() for c in self.exit_conditions],
            'side': self.side,
            'confidence': self.confidence,
            'historical_win_rate': self.historical_win_rate,
            'historical_trades': self.historical_trades,
        }

    def check_entry(self, df: pd.DataFrame, row_idx: int = -1) -> bool:
        """Check if all entry conditions are met."""
        if not self.entry_conditions:
            return False
        return all(c.evaluate(df, row_idx) for c in self.entry_conditions)

    def check_exit(self, df: pd.DataFrame, row_idx: int = -1) -> bool:
        """Check if any exit condition is met."""
        if not self.exit_conditions:
            return False
        return any(c.evaluate(df, row_idx) for c in self.exit_conditions)

    def describe(self) -> str:
        """Generate human-readable description."""
        entry_str = " AND ".join(c.to_expression() for c in self.entry_conditions)
        exit_str = " OR ".join(c.to_expression() for c in self.exit_conditions)
        return (
            f"Rule: {self.name}\n"
            f"Side: {self.side}\n"
            f"Entry: {entry_str}\n"
            f"Exit: {exit_str}\n"
            f"Confidence: {self.confidence:.2%}"
        )


class RuleGenerator:
    """
    Generates trading rules from templates and patterns.

    Combines indicators and thresholds to create testable
    trading rules with entry/exit conditions.
    """

    # Standard indicator templates
    INDICATOR_TEMPLATES = {
        'rsi': {
            'type': IndicatorType.MOMENTUM,
            'oversold': 30,
            'overbought': 70,
            'range': (0, 100),
        },
        'rsi_2': {
            'type': IndicatorType.MOMENTUM,
            'oversold': 10,
            'overbought': 90,
            'range': (0, 100),
        },
        'stoch': {
            'type': IndicatorType.MOMENTUM,
            'oversold': 20,
            'overbought': 80,
            'range': (0, 100),
        },
        'ibs': {
            'type': IndicatorType.PRICE,
            'oversold': 0.2,
            'overbought': 0.8,
            'range': (0, 1),
        },
        'atr_pct': {
            'type': IndicatorType.VOLATILITY,
            'low': 0.01,
            'high': 0.05,
            'range': (0, 0.2),
        },
        'volume_ratio': {
            'type': IndicatorType.VOLUME,
            'low': 0.5,
            'high': 2.0,
            'range': (0, 10),
        },
        'price_vs_sma': {
            'type': IndicatorType.TREND,
            'below': 0.98,
            'above': 1.02,
            'range': (0.8, 1.2),
        },
    }

    def __init__(
        self,
        min_conditions: int = 1,
        max_conditions: int = 3,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the rule generator.

        Args:
            min_conditions: Minimum conditions per rule
            max_conditions: Maximum conditions per rule
            random_seed: Random seed for reproducibility
        """
        self.min_conditions = min_conditions
        self.max_conditions = max_conditions

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self._rule_counter = 0

        logger.info(
            f"RuleGenerator initialized with {min_conditions}-{max_conditions} conditions"
        )

    def _generate_condition(
        self,
        indicator: str,
        for_entry: bool = True,
        side: str = "long",
    ) -> RuleCondition:
        """Generate a single condition for an indicator."""
        template = self.INDICATOR_TEMPLATES.get(indicator, {})
        ind_type = template.get('type', IndicatorType.MOMENTUM)

        # Choose threshold based on entry/exit and long/short
        if for_entry:
            if side == "long":
                # Long entry: look for oversold
                threshold = template.get('oversold', template.get('range', (0, 100))[0])
                operator = ConditionOperator.LESS_THAN
            else:
                # Short entry: look for overbought
                threshold = template.get('overbought', template.get('range', (0, 100))[1])
                operator = ConditionOperator.GREATER_THAN
        else:
            if side == "long":
                # Long exit: look for overbought
                threshold = template.get('overbought', template.get('range', (0, 100))[1])
                operator = ConditionOperator.GREATER_THAN
            else:
                # Short exit: look for oversold
                threshold = template.get('oversold', template.get('range', (0, 100))[0])
                operator = ConditionOperator.LESS_THAN

        # Add some randomness to threshold
        range_vals = template.get('range', (0, 100))
        range_size = range_vals[1] - range_vals[0]
        threshold += random.gauss(0, range_size * 0.1)
        threshold = max(range_vals[0], min(range_vals[1], threshold))

        return RuleCondition(
            indicator=indicator,
            operator=operator,
            threshold=round(threshold, 4),
            indicator_type=ind_type,
        )

    def generate_rule(
        self,
        available_indicators: Optional[List[str]] = None,
        side: str = "long",
        name: Optional[str] = None,
    ) -> TradingRule:
        """
        Generate a new trading rule.

        Args:
            available_indicators: List of indicator names to use
            side: "long" or "short"
            name: Rule name (auto-generated if None)

        Returns:
            Generated TradingRule
        """
        if available_indicators is None:
            available_indicators = list(self.INDICATOR_TEMPLATES.keys())

        # Filter to known indicators
        known = [i for i in available_indicators if i in self.INDICATOR_TEMPLATES]
        if not known:
            known = ['rsi', 'ibs']  # Defaults

        # Determine number of conditions
        n_entry = random.randint(self.min_conditions, self.max_conditions)
        n_exit = random.randint(1, max(1, self.max_conditions - 1))

        # Select indicators for entry
        entry_indicators = random.sample(
            known,
            min(n_entry, len(known))
        )

        # Select indicators for exit (can overlap)
        exit_indicators = random.sample(
            known,
            min(n_exit, len(known))
        )

        # Generate conditions
        entry_conditions = [
            self._generate_condition(ind, for_entry=True, side=side)
            for ind in entry_indicators
        ]

        exit_conditions = [
            self._generate_condition(ind, for_entry=False, side=side)
            for ind in exit_indicators
        ]

        # Generate name
        if name is None:
            self._rule_counter += 1
            name = f"rule_{side}_{self._rule_counter}"

        return TradingRule(
            name=name,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            side=side,
        )

    def generate_rules(
        self,
        count: int,
        available_indicators: Optional[List[str]] = None,
        sides: Optional[List[str]] = None,
    ) -> List[TradingRule]:
        """
        Generate multiple trading rules.

        Args:
            count: Number of rules to generate
            available_indicators: Indicators to use
            sides: List of sides to use (default: ["long"])

        Returns:
            List of generated rules
        """
        if sides is None:
            sides = ["long"]

        rules = []
        for i in range(count):
            side = random.choice(sides)
            rule = self.generate_rule(
                available_indicators=available_indicators,
                side=side,
            )
            rules.append(rule)

        logger.info(f"Generated {len(rules)} trading rules")
        return rules

    def evaluate_rule(
        self,
        rule: TradingRule,
        df: pd.DataFrame,
        forward_returns_col: str = 'forward_return',
        holding_period: int = 5,
    ) -> Dict[str, Any]:
        """
        Evaluate a rule's historical performance.

        Args:
            rule: Rule to evaluate
            df: Historical data with indicators
            forward_returns_col: Column with forward returns
            holding_period: Bars to hold position

        Returns:
            Performance metrics
        """
        signals = []

        for i in range(len(df) - holding_period):
            if rule.check_entry(df, i):
                signals.append(i)

        if not signals:
            return {
                'trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'sharpe': 0,
            }

        # Calculate returns for each signal
        returns = []
        for idx in signals:
            if forward_returns_col in df.columns:
                ret = df[forward_returns_col].iloc[idx]
            else:
                # Calculate simple return
                entry_price = df['close'].iloc[idx]
                exit_price = df['close'].iloc[min(idx + holding_period, len(df) - 1)]
                ret = (exit_price - entry_price) / entry_price
                if rule.side == "short":
                    ret = -ret
            returns.append(ret)

        returns = np.array(returns)
        wins = np.sum(returns > 0)

        return {
            'trades': len(returns),
            'win_rate': wins / len(returns) if len(returns) > 0 else 0,
            'avg_return': np.mean(returns),
            'total_return': np.sum(returns),
            'sharpe': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
        }


def discover_patterns(
    df: pd.DataFrame,
    indicators: List[str],
    target_col: str = 'forward_return',
    min_correlation: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Discover patterns between indicators and forward returns.

    Args:
        df: DataFrame with indicators and target
        indicators: List of indicator column names
        target_col: Target column for correlation
        min_correlation: Minimum correlation threshold

    Returns:
        List of discovered patterns
    """
    patterns = []

    if target_col not in df.columns:
        logger.warning(f"Target column {target_col} not found")
        return patterns

    for indicator in indicators:
        if indicator not in df.columns:
            continue

        # Calculate correlation
        valid_mask = df[indicator].notna() & df[target_col].notna()
        if valid_mask.sum() < 30:
            continue

        corr = df.loc[valid_mask, indicator].corr(df.loc[valid_mask, target_col])

        if abs(corr) >= min_correlation:
            patterns.append({
                'indicator': indicator,
                'correlation': corr,
                'direction': 'positive' if corr > 0 else 'negative',
                'samples': valid_mask.sum(),
            })

    # Sort by absolute correlation
    patterns.sort(key=lambda x: abs(x['correlation']), reverse=True)

    logger.info(f"Discovered {len(patterns)} patterns with correlation >= {min_correlation}")
    return patterns


def generate_rules(
    count: int = 10,
    indicators: Optional[List[str]] = None,
) -> List[TradingRule]:
    """Convenience function to generate trading rules."""
    generator = RuleGenerator()
    return generator.generate_rules(count, indicators)


# Module-level generator
_generator: Optional[RuleGenerator] = None


def get_generator() -> RuleGenerator:
    """Get or create the global generator instance."""
    global _generator
    if _generator is None:
        _generator = RuleGenerator()
    return _generator
