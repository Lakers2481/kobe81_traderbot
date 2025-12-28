"""
Symbolic Reasoner - Neuro-Symbolic Market Reasoning
====================================================

This module combines symbolic rule-based logic with neural network outputs
to make trading decisions. It provides:

1. **Rule Evaluation**: Evaluate symbolic rules against market context
2. **Verdict Generation**: Produce structured verdicts with confidence
3. **Override Detection**: Identify when symbolic rules should override ML signals
4. **Reasoning Chains**: Track the logical reasoning for explainability

The reasoner loads rules from `config/symbolic_rules.yaml` and evaluates them
against the current market context, signal data, and cognitive state.

Usage:
    from cognitive import SymbolicReasoner

    reasoner = SymbolicReasoner()
    verdict = reasoner.reason(
        market_context=context,
        signal_data=signal,
        cognitive_confidence=0.75,
        market_mood_score=0.3,
        self_model_status=status
    )

    if verdict.should_override:
        # Apply symbolic override
        adjusted_confidence = confidence * (1 - verdict.override_strength)
"""

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional YAML dependency
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None


class SymbolicVerdictType(Enum):
    """Types of verdicts the symbolic reasoner can produce."""
    PASS_THROUGH = "pass_through"
    OVERRIDE_LONG_DUE_TO_MACRO_RISK = "override_long_macro_risk"
    OVERRIDE_SHORT_DUE_TO_MACRO_RISK = "override_short_macro_risk"
    CONFIRM_LONG_DUE_TO_ALIGNMENT = "confirm_long_alignment"
    CONFIRM_SHORT_DUE_TO_ALIGNMENT = "confirm_short_alignment"
    SUGGEST_HEDGE = "suggest_hedge"
    COMPLIANCE_BLOCK = "compliance_block"
    REDUCE_SIZE = "reduce_size"
    REQUIRE_SLOW_PATH = "require_slow_path"


@dataclass
class SymbolicRule:
    """Represents a symbolic trading rule."""
    id: str
    name: str
    condition: str
    verdict: str
    confidence: float
    priority: int = 100
    rationale: str = ""
    description: str = ""
    # Optional modifiers
    confidence_boost: float = 0.0
    reduction_factor: float = 1.0
    category: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'condition': self.condition,
            'verdict': self.verdict,
            'confidence': self.confidence,
            'priority': self.priority,
            'rationale': self.rationale,
        }


@dataclass
class SymbolicVerdict:
    """The result of symbolic reasoning."""
    verdict_type: SymbolicVerdictType
    confidence: float
    reasoning_chain: List[str] = field(default_factory=list)
    triggered_rules: List[str] = field(default_factory=list)
    should_override: bool = False
    override_strength: float = 0.0
    # Additional action parameters
    confidence_boost: float = 0.0
    size_reduction: float = 1.0
    suggested_action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'verdict_type': self.verdict_type.value,
            'confidence': self.confidence,
            'reasoning_chain': self.reasoning_chain,
            'triggered_rules': self.triggered_rules,
            'should_override': self.should_override,
            'override_strength': self.override_strength,
            'confidence_boost': self.confidence_boost,
            'size_reduction': self.size_reduction,
            'suggested_action': self.suggested_action,
        }


class SymbolicReasoner:
    """
    Evaluates symbolic rules against market context to produce trading verdicts.

    The reasoner combines symbolic logic with neural network outputs by:
    1. Loading rules from YAML configuration
    2. Evaluating conditions against current market state
    3. Producing structured verdicts with confidence scores
    4. Tracking reasoning chains for explainability
    """

    # Verdict types that should override neural signals
    OVERRIDE_VERDICTS = {
        SymbolicVerdictType.OVERRIDE_LONG_DUE_TO_MACRO_RISK,
        SymbolicVerdictType.OVERRIDE_SHORT_DUE_TO_MACRO_RISK,
        SymbolicVerdictType.COMPLIANCE_BLOCK,
    }

    # Verdict types that boost confidence
    CONFIRMATION_VERDICTS = {
        SymbolicVerdictType.CONFIRM_LONG_DUE_TO_ALIGNMENT,
        SymbolicVerdictType.CONFIRM_SHORT_DUE_TO_ALIGNMENT,
    }

    def __init__(
        self,
        rules_file: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize the symbolic reasoner.

        Args:
            rules_file: Path to symbolic rules YAML. Defaults to config/symbolic_rules.yaml
            enabled: Whether symbolic reasoning is active
        """
        self.enabled = enabled
        self._rules: List[SymbolicRule] = []
        self._dynamic_rules: List[SymbolicRule] = []  # Rules added at runtime
        self._evaluation_cache: Dict[str, Tuple[bool, datetime]] = {}
        self._cache_ttl_seconds = 60

        # Determine rules file path
        if rules_file:
            self._rules_file = Path(rules_file)
        else:
            # Default to config/symbolic_rules.yaml relative to project root
            project_root = Path(__file__).parent.parent
            self._rules_file = project_root / "config" / "symbolic_rules.yaml"

        # Load rules
        self._load_rules()
        logger.info(f"SymbolicReasoner initialized with {len(self._rules)} rules")

    def _load_rules(self) -> None:
        """Load rules from YAML configuration file."""
        if not HAS_YAML:
            logger.warning("PyYAML not available. Using default rules.")
            self._load_default_rules()
            return

        if not self._rules_file.exists():
            logger.warning(f"Rules file not found: {self._rules_file}. Using defaults.")
            self._load_default_rules()
            return

        try:
            with open(self._rules_file, 'r') as f:
                config = yaml.safe_load(f)

            self._rules = []

            # Load each rule category
            for category in ['macro_risk_rules', 'alignment_rules', 'compliance_rules',
                             'sector_rules', 'self_model_rules']:
                if category in config:
                    for rule_dict in config[category]:
                        rule = self._parse_rule(rule_dict, category)
                        if rule:
                            self._rules.append(rule)

            # Sort by priority (lower = higher priority)
            self._rules.sort(key=lambda r: r.priority)
            logger.info(f"Loaded {len(self._rules)} rules from {self._rules_file}")

        except Exception as e:
            logger.error(f"Error loading rules: {e}. Using defaults.")
            self._load_default_rules()

    def _parse_rule(self, rule_dict: Dict[str, Any], category: str) -> Optional[SymbolicRule]:
        """Parse a rule dictionary into a SymbolicRule object."""
        try:
            return SymbolicRule(
                id=rule_dict.get('id', 'UNKNOWN'),
                name=rule_dict.get('name', ''),
                condition=rule_dict.get('condition', 'false'),
                verdict=rule_dict.get('verdict', 'PASS_THROUGH'),
                confidence=float(rule_dict.get('confidence', 0.5)),
                priority=int(rule_dict.get('priority', 100)),
                rationale=rule_dict.get('rationale', ''),
                description=rule_dict.get('description', ''),
                confidence_boost=float(rule_dict.get('confidence_boost', 0.0)),
                reduction_factor=float(rule_dict.get('reduction_factor', 1.0)),
                category=category,
            )
        except Exception as e:
            logger.error(f"Error parsing rule {rule_dict.get('id', 'UNKNOWN')}: {e}")
            return None

    def _load_default_rules(self) -> None:
        """Load minimal default rules when YAML is unavailable."""
        self._rules = [
            SymbolicRule(
                id="DEFAULT_001",
                name="Kill Switch Compliance",
                condition="kill_switch_active = true",
                verdict="COMPLIANCE_BLOCK",
                confidence=1.0,
                priority=0,
                rationale="Emergency halt",
                category="compliance_rules",
            ),
            SymbolicRule(
                id="DEFAULT_002",
                name="Prohibited Symbol Block",
                condition="is_prohibited = true",
                verdict="COMPLIANCE_BLOCK",
                confidence=1.0,
                priority=1,
                rationale="Regulatory compliance",
                category="compliance_rules",
            ),
            SymbolicRule(
                id="DEFAULT_003",
                name="Extreme VIX Override",
                condition="vix >= 40 AND side = LONG",
                verdict="OVERRIDE_LONG_DUE_TO_MACRO_RISK",
                confidence=0.90,
                priority=10,
                rationale="Extreme fear conditions",
                category="macro_risk_rules",
            ),
        ]
        logger.info(f"Loaded {len(self._rules)} default rules")

    def reason(
        self,
        market_context: Dict[str, Any],
        signal_data: Optional[Dict[str, Any]] = None,
        cognitive_confidence: float = 0.5,
        market_mood_score: float = 0.0,
        self_model_status: Optional[Dict[str, Any]] = None,
    ) -> SymbolicVerdict:
        """
        Evaluate symbolic rules and produce a verdict.

        Args:
            market_context: Current market state (vix, regime, etc.)
            signal_data: Signal being evaluated (side, symbol, etc.)
            cognitive_confidence: Neural network confidence score
            market_mood_score: Market emotional state (-1 to 1)
            self_model_status: Self-awareness status from SelfModel

        Returns:
            SymbolicVerdict with reasoning and any override actions
        """
        if not self.enabled:
            return SymbolicVerdict(
                verdict_type=SymbolicVerdictType.PASS_THROUGH,
                confidence=1.0,
                reasoning_chain=["Symbolic reasoning disabled"],
            )

        # Build evaluation context
        context = self._build_evaluation_context(
            market_context, signal_data, cognitive_confidence,
            market_mood_score, self_model_status
        )

        # Evaluate rules in priority order
        triggered_rules: List[SymbolicRule] = []
        reasoning_chain: List[str] = []

        for rule in self._rules + self._dynamic_rules:
            try:
                if self._evaluate_condition(rule.condition, context):
                    triggered_rules.append(rule)
                    reasoning_chain.append(
                        f"[{rule.id}] {rule.name}: {rule.rationale}"
                    )
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule.id}: {e}")

        # Determine final verdict
        if not triggered_rules:
            return SymbolicVerdict(
                verdict_type=SymbolicVerdictType.PASS_THROUGH,
                confidence=0.5,
                reasoning_chain=["No symbolic rules triggered"],
            )

        # Use highest priority (lowest number) triggered rule
        primary_rule = triggered_rules[0]

        # Parse verdict type
        try:
            verdict_type = SymbolicVerdictType(primary_rule.verdict.lower().replace(
                'override_long_due_to_macro_risk', 'override_long_macro_risk'
            ).replace(
                'override_short_due_to_macro_risk', 'override_short_macro_risk'
            ).replace(
                'confirm_long_due_to_alignment', 'confirm_long_alignment'
            ).replace(
                'confirm_short_due_to_alignment', 'confirm_short_alignment'
            ))
        except ValueError:
            verdict_type = SymbolicVerdictType.PASS_THROUGH

        # Calculate override strength
        should_override = verdict_type in self.OVERRIDE_VERDICTS
        override_strength = primary_rule.confidence if should_override else 0.0

        # Calculate confidence boost for confirmation verdicts
        confidence_boost = 0.0
        if verdict_type in self.CONFIRMATION_VERDICTS:
            confidence_boost = primary_rule.confidence_boost

        # Calculate size reduction
        size_reduction = 1.0
        if verdict_type == SymbolicVerdictType.REDUCE_SIZE:
            size_reduction = primary_rule.reduction_factor

        # Build suggested action
        suggested_action = None
        if verdict_type == SymbolicVerdictType.SUGGEST_HEDGE:
            suggested_action = "Consider protective puts or reducing position size"
        elif verdict_type == SymbolicVerdictType.REQUIRE_SLOW_PATH:
            suggested_action = "Route to deliberative (slow) processing"
        elif verdict_type == SymbolicVerdictType.COMPLIANCE_BLOCK:
            suggested_action = "Block trade - compliance requirement"

        return SymbolicVerdict(
            verdict_type=verdict_type,
            confidence=primary_rule.confidence,
            reasoning_chain=reasoning_chain,
            triggered_rules=[r.id for r in triggered_rules],
            should_override=should_override,
            override_strength=override_strength,
            confidence_boost=confidence_boost,
            size_reduction=size_reduction,
            suggested_action=suggested_action,
            metadata={
                'primary_rule': primary_rule.id,
                'rules_evaluated': len(self._rules) + len(self._dynamic_rules),
                'rules_triggered': len(triggered_rules),
            }
        )

    def _build_evaluation_context(
        self,
        market_context: Dict[str, Any],
        signal_data: Optional[Dict[str, Any]],
        cognitive_confidence: float,
        market_mood_score: float,
        self_model_status: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a unified context for rule evaluation."""
        context = {
            # Market context
            'vix': market_context.get('vix', 20.0),
            'regime': market_context.get('regime', 'NEUTRAL'),
            'has_positions': market_context.get('has_positions', False),
            'kill_switch_active': market_context.get('kill_switch_active', False),
            'rate_uncertainty': market_context.get('rate_uncertainty', 'low'),
            'earnings_season': market_context.get('earnings_season', False),

            # Cognitive state
            'cognitive_confidence': cognitive_confidence,
            'market_mood_score': market_mood_score,

            # Default signal data
            'side': 'NONE',
            'symbol': '',
            'sector': 'Unknown',
            'beta': 1.0,
            'is_prohibited': False,
            'has_earnings': False,
            'days_to_earnings': 999,
        }

        # Add signal data if provided
        if signal_data:
            context.update({
                'side': signal_data.get('side', 'NONE'),
                'symbol': signal_data.get('symbol', ''),
                'sector': signal_data.get('sector', 'Unknown'),
                'beta': signal_data.get('beta', 1.0),
                'is_prohibited': signal_data.get('is_prohibited', False),
                'has_earnings': signal_data.get('has_earnings', False),
                'days_to_earnings': signal_data.get('days_to_earnings', 999),
            })

        # Add self-model status if provided
        if self_model_status:
            context.update({
                'strategy_regime_weakness': self_model_status.get('strategy_regime_weakness', False),
                'calibration_drift': self_model_status.get('calibration_drift', 0.0),
                'situation_novelty': self_model_status.get('situation_novelty', 0.0),
            })

        return context

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate a symbolic condition string against a context.

        Supports:
        - Comparisons: =, !=, >, <, >=, <=
        - Logical: AND, OR, NOT
        - Variables from context

        Args:
            condition: Condition string (e.g., "vix >= 35 AND side = LONG")
            context: Variable values

        Returns:
            True if condition is satisfied
        """
        if not condition or condition.lower() == 'false':
            return False
        if condition.lower() == 'true':
            return True

        # Handle OR (lower precedence)
        if ' OR ' in condition:
            parts = condition.split(' OR ')
            return any(self._evaluate_condition(p.strip(), context) for p in parts)

        # Handle AND (higher precedence)
        if ' AND ' in condition:
            parts = condition.split(' AND ')
            return all(self._evaluate_condition(p.strip(), context) for p in parts)

        # Handle NOT
        if condition.startswith('NOT '):
            return not self._evaluate_condition(condition[4:], context)

        # Parse single comparison
        return self._evaluate_comparison(condition, context)

    def _evaluate_comparison(self, expr: str, context: Dict[str, Any]) -> bool:
        """Evaluate a single comparison expression."""
        # Match patterns like "var >= value", "var = value", etc.
        patterns = [
            (r'(\w+)\s*>=\s*(.+)', lambda a, b: a >= b),
            (r'(\w+)\s*<=\s*(.+)', lambda a, b: a <= b),
            (r'(\w+)\s*!=\s*(.+)', lambda a, b: a != b),
            (r'(\w+)\s*>\s*(.+)', lambda a, b: a > b),
            (r'(\w+)\s*<\s*(.+)', lambda a, b: a < b),
            (r'(\w+)\s*=\s*(.+)', lambda a, b: a == b),
        ]

        for pattern, comparator in patterns:
            match = re.match(pattern, expr.strip())
            if match:
                var_name = match.group(1).strip()
                raw_value = match.group(2).strip()

                # Get variable value from context
                var_value = context.get(var_name)
                if var_value is None:
                    return False

                # Parse comparison value
                try:
                    # Try numeric
                    comp_value = float(raw_value)
                    var_value = float(var_value) if not isinstance(var_value, bool) else var_value
                except ValueError:
                    # String comparison (remove quotes if present)
                    comp_value = raw_value.strip('"\'')
                    var_value = str(var_value)

                # Handle boolean strings
                if raw_value.lower() == 'true':
                    comp_value = True
                elif raw_value.lower() == 'false':
                    comp_value = False

                return comparator(var_value, comp_value)

        return False

    def add_dynamic_rule(self, rule: SymbolicRule) -> None:
        """Add a rule at runtime (e.g., from policy activation)."""
        self._dynamic_rules.append(rule)
        # Re-sort by priority
        self._dynamic_rules.sort(key=lambda r: r.priority)
        logger.info(f"Added dynamic rule: {rule.id}")

    def remove_dynamic_rule(self, rule_id: str) -> bool:
        """Remove a dynamic rule by ID."""
        original_count = len(self._dynamic_rules)
        self._dynamic_rules = [r for r in self._dynamic_rules if r.id != rule_id]
        removed = len(self._dynamic_rules) < original_count
        if removed:
            logger.info(f"Removed dynamic rule: {rule_id}")
        return removed

    def clear_dynamic_rules(self) -> int:
        """Clear all dynamic rules. Returns count of removed rules."""
        count = len(self._dynamic_rules)
        self._dynamic_rules = []
        logger.info(f"Cleared {count} dynamic rules")
        return count

    def get_rules(self, include_dynamic: bool = True) -> List[SymbolicRule]:
        """Get all loaded rules."""
        if include_dynamic:
            return self._rules + self._dynamic_rules
        return self._rules.copy()

    def reload_rules(self) -> int:
        """Reload rules from configuration file. Returns count of rules loaded."""
        self._load_rules()
        return len(self._rules)

    def introspect(self) -> str:
        """Generate human-readable description of the reasoner state."""
        lines = [
            "--- Symbolic Reasoner ---",
            f"Enabled: {self.enabled}",
            f"Rules file: {self._rules_file}",
            f"Static rules: {len(self._rules)}",
            f"Dynamic rules: {len(self._dynamic_rules)}",
            "",
            "Rule categories:",
        ]

        # Count by category
        categories: Dict[str, int] = {}
        for rule in self._rules:
            cat = rule.category
            categories[cat] = categories.get(cat, 0) + 1

        for cat, count in sorted(categories.items()):
            lines.append(f"  - {cat}: {count}")

        return "\n".join(lines)


# --- Singleton Implementation ---
_symbolic_reasoner: Optional[SymbolicReasoner] = None
_lock = threading.Lock()


def get_symbolic_reasoner() -> SymbolicReasoner:
    """Factory function to get the singleton SymbolicReasoner instance."""
    global _symbolic_reasoner
    if _symbolic_reasoner is None:
        with _lock:
            if _symbolic_reasoner is None:
                _symbolic_reasoner = SymbolicReasoner()
    return _symbolic_reasoner
