"""
Trading Rules Engine
=====================

Enforces trading rules and compliance requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, time
from enum import Enum

logger = logging.getLogger(__name__)


class RuleCategory(Enum):
    """Category of trading rule."""
    RISK = "risk"
    TIMING = "timing"
    SIZE = "size"
    SYMBOL = "symbol"
    REGULATORY = "regulatory"


@dataclass
class TradingRule:
    """Definition of a trading rule."""
    name: str
    category: RuleCategory
    description: str = ""
    enabled: bool = True
    severity: str = "warning"  # warning, error, critical

    # Configurable parameters
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'category': self.category.value,
            'description': self.description,
            'enabled': self.enabled,
            'severity': self.severity,
        }


@dataclass
class RuleViolation:
    """Record of a rule violation."""
    rule: TradingRule
    violated_at: datetime = field(default_factory=datetime.now)
    symbol: str = ""
    details: str = ""
    blocked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_name': self.rule.name,
            'category': self.rule.category.value,
            'severity': self.rule.severity,
            'violated_at': self.violated_at.isoformat(),
            'symbol': self.symbol,
            'details': self.details,
            'blocked': self.blocked,
        }


class RulesEngine:
    """
    Enforces trading rules and compliance requirements.
    """

    # Standard rules
    STANDARD_RULES = [
        TradingRule(
            name="max_position_size",
            category=RuleCategory.SIZE,
            description="Maximum position size as % of portfolio",
            params={'max_pct': 0.05},
        ),
        TradingRule(
            name="max_daily_trades",
            category=RuleCategory.RISK,
            description="Maximum trades per day",
            params={'max_trades': 50},
        ),
        TradingRule(
            name="trading_hours",
            category=RuleCategory.TIMING,
            description="Only trade during market hours",
            params={'start': '09:30', 'end': '16:00'},
        ),
        TradingRule(
            name="min_volume",
            category=RuleCategory.SYMBOL,
            description="Minimum average daily volume",
            params={'min_adv': 100000},
        ),
        TradingRule(
            name="no_penny_stocks",
            category=RuleCategory.SYMBOL,
            description="No stocks under $5",
            params={'min_price': 5.0},
        ),
        TradingRule(
            name="pattern_day_trader",
            category=RuleCategory.REGULATORY,
            description="PDT rule compliance",
            params={'min_equity': 25000},
        ),
    ]

    def __init__(self, rules: Optional[List[TradingRule]] = None):
        self.rules = {r.name: r for r in (rules or self.STANDARD_RULES)}
        self._violations: List[RuleViolation] = []
        self._daily_trades = 0
        logger.info(f"RulesEngine initialized with {len(self.rules)} rules")

    def check_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        portfolio_value: float,
        volume: Optional[float] = None,
    ) -> List[RuleViolation]:
        """Check if a trade violates any rules."""
        violations = []

        # Position size check
        rule = self.rules.get('max_position_size')
        if rule and rule.enabled:
            trade_value = quantity * price
            position_pct = trade_value / portfolio_value if portfolio_value > 0 else 1.0
            if position_pct > rule.params.get('max_pct', 0.05):
                violation = RuleViolation(
                    rule=rule,
                    symbol=symbol,
                    details=f"Position {position_pct:.1%} exceeds max {rule.params['max_pct']:.1%}",
                    blocked=rule.severity == 'critical',
                )
                violations.append(violation)

        # Daily trades check
        rule = self.rules.get('max_daily_trades')
        if rule and rule.enabled:
            if self._daily_trades >= rule.params.get('max_trades', 50):
                violation = RuleViolation(
                    rule=rule,
                    symbol=symbol,
                    details=f"Daily trade limit {rule.params['max_trades']} reached",
                    blocked=True,
                )
                violations.append(violation)

        # Trading hours check
        rule = self.rules.get('trading_hours')
        if rule and rule.enabled:
            now = datetime.now().time()
            start = time.fromisoformat(rule.params.get('start', '09:30'))
            end = time.fromisoformat(rule.params.get('end', '16:00'))
            if not (start <= now <= end):
                violation = RuleViolation(
                    rule=rule,
                    symbol=symbol,
                    details=f"Outside trading hours ({start}-{end})",
                    blocked=True,
                )
                violations.append(violation)

        # Penny stock check
        rule = self.rules.get('no_penny_stocks')
        if rule and rule.enabled:
            min_price = rule.params.get('min_price', 5.0)
            if price < min_price:
                violation = RuleViolation(
                    rule=rule,
                    symbol=symbol,
                    details=f"Price ${price:.2f} below minimum ${min_price}",
                    blocked=True,
                )
                violations.append(violation)

        # Volume check
        if volume is not None:
            rule = self.rules.get('min_volume')
            if rule and rule.enabled:
                min_adv = rule.params.get('min_adv', 100000)
                if volume < min_adv:
                    violation = RuleViolation(
                        rule=rule,
                        symbol=symbol,
                        details=f"Volume {volume:,.0f} below minimum {min_adv:,.0f}",
                        blocked=True,
                    )
                    violations.append(violation)

        self._violations.extend(violations)
        return violations

    def record_trade(self):
        """Record that a trade was executed."""
        self._daily_trades += 1

    def reset_daily(self):
        """Reset daily counters."""
        self._daily_trades = 0

    def get_violations(self, hours: int = 24) -> List[RuleViolation]:
        """Get recent violations."""
        cutoff = datetime.now().timestamp() - hours * 3600
        return [v for v in self._violations
                if v.violated_at.timestamp() > cutoff]

    def add_rule(self, rule: TradingRule):
        """Add a new rule."""
        self.rules[rule.name] = rule

    def disable_rule(self, name: str):
        """Disable a rule."""
        if name in self.rules:
            self.rules[name].enabled = False

    def enable_rule(self, name: str):
        """Enable a rule."""
        if name in self.rules:
            self.rules[name].enabled = True


# Global instance
_engine: Optional[RulesEngine] = None


def get_engine() -> RulesEngine:
    """Get global rules engine."""
    global _engine
    if _engine is None:
        _engine = RulesEngine()
    return _engine


def check_rules(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    portfolio_value: float,
) -> List[RuleViolation]:
    """Check trading rules."""
    return get_engine().check_trade(symbol, side, quantity, price, portfolio_value)


def get_violations() -> List[RuleViolation]:
    """Get recent violations."""
    return get_engine().get_violations()
