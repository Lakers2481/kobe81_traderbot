"""
Semantic Memory - Distilled Knowledge Base
============================================

Turns episodic lessons into reusable "rules of thumb".

Unlike episodic memory (specific events), semantic memory holds
generalized knowledge that applies across situations.

Features:
- Extracts patterns from episodes
- Maintains confidence-weighted rules
- Updates rules based on new evidence
- Provides fast lookup for applicable rules

Example Rules:
- "In high-vol + gap days, mean reversion fails unless volume confirms"
- "When VIX > 30, reduce position size by 50%"
- "Donchian works best in BULL regime with trend confirmation"

Usage:
    from cognitive.semantic_memory import get_semantic_memory

    memory = get_semantic_memory()

    # Add a rule
    memory.add_rule(
        condition="VIX > 30 AND regime = CHOPPY",
        action="reduce_position_size",
        parameters={'multiplier': 0.5},
        confidence=0.85,
        source="Extracted from 15 episodes"
    )

    # Query applicable rules
    rules = memory.get_applicable_rules(current_context)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SemanticRule:
    """A generalized rule extracted from experience."""
    rule_id: str
    condition: str          # Natural language or structured condition
    action: str             # What to do when condition matches
    parameters: Dict[str, Any]  # Action parameters
    confidence: float       # How confident we are (0-1)
    supporting_episodes: int  # Number of episodes supporting this rule
    contradicting_episodes: int  # Number of episodes contradicting
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None
    times_applied: int = 0
    times_successful: int = 0
    source: str = ""        # Where this rule came from
    tags: List[str] = field(default_factory=list)
    is_active: bool = True

    @property
    def success_rate(self) -> float:
        if self.times_applied == 0:
            return 0.0
        return self.times_successful / self.times_applied

    @property
    def evidence_ratio(self) -> float:
        """Ratio of supporting to total evidence."""
        total = self.supporting_episodes + self.contradicting_episodes
        if total == 0:
            return 0.5
        return self.supporting_episodes / total

    def to_dict(self) -> Dict:
        return {
            'rule_id': self.rule_id,
            'condition': self.condition,
            'action': self.action,
            'parameters': self.parameters,
            'confidence': self.confidence,
            'supporting_episodes': self.supporting_episodes,
            'contradicting_episodes': self.contradicting_episodes,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'last_applied': self.last_applied.isoformat() if self.last_applied else None,
            'times_applied': self.times_applied,
            'times_successful': self.times_successful,
            'success_rate': self.success_rate,
            'source': self.source,
            'tags': self.tags,
            'is_active': self.is_active,
        }

    @staticmethod
    def from_dict(d: Dict) -> 'SemanticRule':
        return SemanticRule(
            rule_id=d['rule_id'],
            condition=d['condition'],
            action=d['action'],
            parameters=d.get('parameters', {}),
            confidence=d.get('confidence', 0.5),
            supporting_episodes=d.get('supporting_episodes', 0),
            contradicting_episodes=d.get('contradicting_episodes', 0),
            created_at=datetime.fromisoformat(d['created_at']),
            last_updated=datetime.fromisoformat(d['last_updated']),
            last_applied=datetime.fromisoformat(d['last_applied']) if d.get('last_applied') else None,
            times_applied=d.get('times_applied', 0),
            times_successful=d.get('times_successful', 0),
            source=d.get('source', ''),
            tags=d.get('tags', []),
            is_active=d.get('is_active', True),
        )


class ConditionMatcher:
    """Evaluates whether a context matches a condition."""

    def __init__(self):
        self._operators = {
            '>': lambda a, b: a > b,
            '<': lambda a, b: a < b,
            '>=': lambda a, b: a >= b,
            '<=': lambda a, b: a <= b,
            '=': lambda a, b: str(a).lower() == str(b).lower(),
            '!=': lambda a, b: str(a).lower() != str(b).lower(),
            'contains': lambda a, b: str(b).lower() in str(a).lower(),
        }

    def matches(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Check if context matches condition.

        Supports conditions like:
        - "VIX > 30"
        - "regime = BULL"
        - "VIX > 25 AND regime = CHOPPY"
        - "strategy = donchian OR strategy = turtle_soup"
        """
        condition = condition.strip()

        # Handle AND
        if ' AND ' in condition:
            parts = condition.split(' AND ')
            return all(self.matches(p.strip(), context) for p in parts)

        # Handle OR
        if ' OR ' in condition:
            parts = condition.split(' OR ')
            return any(self.matches(p.strip(), context) for p in parts)

        # Parse single condition
        for op_str, op_func in sorted(self._operators.items(), key=lambda x: -len(x[0])):
            if op_str in condition:
                parts = condition.split(op_str, 1)
                if len(parts) == 2:
                    field = parts[0].strip().lower()
                    value_str = parts[1].strip()

                    # Get context value
                    ctx_value = context.get(field)
                    if ctx_value is None:
                        # Try nested access
                        for key in context:
                            if isinstance(context[key], dict):
                                ctx_value = context[key].get(field)
                                if ctx_value is not None:
                                    break

                    if ctx_value is None:
                        return False

                    # Try numeric comparison
                    try:
                        target_value = float(value_str)
                        ctx_value = float(ctx_value)
                    except (ValueError, TypeError):
                        target_value = value_str

                    return op_func(ctx_value, target_value)

        return False


class SemanticMemory:
    """
    Long-term storage for generalized rules and knowledge.

    Unlike episodic memory (specific events), semantic memory holds
    abstract knowledge extracted from patterns across episodes.
    """

    def __init__(
        self,
        storage_dir: str = "state/cognitive",
        auto_persist: bool = True,
        min_confidence: float = 0.5,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.auto_persist = auto_persist
        self.min_confidence = min_confidence

        self._rules: Dict[str, SemanticRule] = {}
        self._matcher = ConditionMatcher()

        self._load_rules()
        logger.info(f"SemanticMemory initialized with {len(self._rules)} rules")

    def add_rule(
        self,
        condition: str,
        action: str,
        parameters: Optional[Dict] = None,
        confidence: float = 0.7,
        supporting_episodes: int = 1,
        source: str = "",
        tags: Optional[List[str]] = None,
    ) -> SemanticRule:
        """
        Add a new rule to semantic memory.

        Args:
            condition: When this rule applies (e.g., "VIX > 30")
            action: What to do (e.g., "reduce_position_size")
            parameters: Action parameters
            confidence: Initial confidence (0-1)
            supporting_episodes: How many episodes support this
            source: Where this rule came from
            tags: Optional tags for categorization

        Returns:
            The created SemanticRule
        """
        rule_id = hashlib.md5(f"{condition}|{action}".encode()).hexdigest()[:8]

        # Check for existing similar rule
        if rule_id in self._rules:
            # Update existing rule
            existing = self._rules[rule_id]
            existing.supporting_episodes += supporting_episodes
            existing.confidence = min(1.0, existing.confidence + 0.05)
            existing.last_updated = datetime.now()
            if self.auto_persist:
                self._save_rules()
            return existing

        rule = SemanticRule(
            rule_id=rule_id,
            condition=condition,
            action=action,
            parameters=parameters or {},
            confidence=confidence,
            supporting_episodes=supporting_episodes,
            contradicting_episodes=0,
            source=source,
            tags=tags or [],
        )

        self._rules[rule_id] = rule

        if self.auto_persist:
            self._save_rules()

        logger.info(f"Added rule {rule_id}: {condition} -> {action}")
        return rule

    def get_applicable_rules(
        self,
        context: Dict[str, Any],
        min_confidence: Optional[float] = None,
    ) -> List[SemanticRule]:
        """
        Get all rules that apply to the current context.

        Args:
            context: Current context dict
            min_confidence: Minimum confidence threshold (uses default if not specified)

        Returns:
            List of applicable rules, sorted by confidence
        """
        threshold = min_confidence or self.min_confidence
        applicable = []

        for rule in self._rules.values():
            if not rule.is_active:
                continue
            if rule.confidence < threshold:
                continue

            try:
                if self._matcher.matches(rule.condition, context):
                    applicable.append(rule)
            except Exception as e:
                logger.warning(f"Error matching rule {rule.rule_id}: {e}")

        # Sort by confidence
        applicable.sort(key=lambda r: r.confidence, reverse=True)
        return applicable

    def get_rules_for_action(self, action: str) -> List[SemanticRule]:
        """Get all rules that produce a specific action."""
        return [r for r in self._rules.values()
                if r.action == action and r.is_active]

    def record_rule_outcome(
        self,
        rule_id: str,
        successful: bool,
    ) -> None:
        """
        Record the outcome of applying a rule.

        Updates success rate and adjusts confidence.
        """
        rule = self._rules.get(rule_id)
        if not rule:
            return

        rule.times_applied += 1
        rule.last_applied = datetime.now()

        if successful:
            rule.times_successful += 1
            # Boost confidence slightly
            rule.confidence = min(1.0, rule.confidence + 0.01)
        else:
            # Reduce confidence
            rule.confidence = max(0.1, rule.confidence - 0.02)

        # Deactivate if success rate is too low
        if rule.times_applied >= 10 and rule.success_rate < 0.3:
            rule.is_active = False
            logger.info(f"Deactivated rule {rule_id} due to low success rate")

        if self.auto_persist:
            self._save_rules()

    def add_evidence(
        self,
        rule_id: str,
        supporting: bool,
    ) -> None:
        """
        Add evidence for or against a rule.

        Args:
            rule_id: Rule to update
            supporting: True if evidence supports the rule
        """
        rule = self._rules.get(rule_id)
        if not rule:
            return

        if supporting:
            rule.supporting_episodes += 1
            rule.confidence = min(1.0, rule.confidence + 0.02)
        else:
            rule.contradicting_episodes += 1
            rule.confidence = max(0.1, rule.confidence - 0.03)

        rule.last_updated = datetime.now()

        # Deactivate if too much contradicting evidence
        if rule.evidence_ratio < 0.3:
            rule.is_active = False
            logger.info(f"Deactivated rule {rule_id} due to contradicting evidence")

        if self.auto_persist:
            self._save_rules()

    def extract_rules_from_episodes(self, episodes: List[Any]) -> List[SemanticRule]:
        """
        Extract new rules from a set of episodes.

        Looks for patterns in winning vs losing episodes.

        Args:
            episodes: List of Episode objects

        Returns:
            List of newly created rules
        """
        from cognitive.episodic_memory import EpisodeOutcome

        new_rules = []

        # Group by context signature
        by_context: Dict[str, List[Any]] = {}
        for ep in episodes:
            sig = ep.context_signature()
            if sig not in by_context:
                by_context[sig] = []
            by_context[sig].append(ep)

        # Look for patterns
        for sig, context_episodes in by_context.items():
            if len(context_episodes) < 5:
                continue

            wins = [e for e in context_episodes if e.outcome == EpisodeOutcome.WIN]
            losses = [e for e in context_episodes if e.outcome == EpisodeOutcome.LOSS]

            if len(wins) + len(losses) < 5:
                continue

            win_rate = len(wins) / (len(wins) + len(losses))

            # If strong pattern, create rule
            if win_rate > 0.7:
                # This context is good
                sample = context_episodes[0]
                condition = self._extract_condition(sample)
                rule = self.add_rule(
                    condition=condition,
                    action="increase_confidence",
                    parameters={'boost': 0.1},
                    confidence=win_rate,
                    supporting_episodes=len(wins),
                    source=f"Extracted from {len(context_episodes)} episodes",
                    tags=['auto_extracted'],
                )
                new_rules.append(rule)

            elif win_rate < 0.3:
                # This context is bad
                sample = context_episodes[0]
                condition = self._extract_condition(sample)
                rule = self.add_rule(
                    condition=condition,
                    action="reduce_confidence",
                    parameters={'reduction': 0.2},
                    confidence=1 - win_rate,
                    supporting_episodes=len(losses),
                    source=f"Extracted from {len(context_episodes)} episodes",
                    tags=['auto_extracted', 'warning'],
                )
                new_rules.append(rule)

        return new_rules

    def _extract_condition(self, episode: Any) -> str:
        """Extract a condition string from an episode."""
        parts = []

        if episode.market_context.get('regime'):
            parts.append(f"regime = {episode.market_context['regime']}")

        if episode.signal_context.get('strategy'):
            parts.append(f"strategy = {episode.signal_context['strategy']}")

        return " AND ".join(parts) if parts else "unknown"

    def get_rule(self, rule_id: str) -> Optional[SemanticRule]:
        """Get a specific rule by ID."""
        return self._rules.get(rule_id)

    def get_all_rules(self, active_only: bool = True) -> List[SemanticRule]:
        """Get all rules."""
        rules = list(self._rules.values())
        if active_only:
            rules = [r for r in rules if r.is_active]
        return sorted(rules, key=lambda r: r.confidence, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get semantic memory statistics."""
        rules = list(self._rules.values())
        active = [r for r in rules if r.is_active]

        return {
            'total_rules': len(rules),
            'active_rules': len(active),
            'avg_confidence': sum(r.confidence for r in active) / len(active) if active else 0,
            'total_applications': sum(r.times_applied for r in rules),
            'auto_extracted': len([r for r in rules if 'auto_extracted' in r.tags]),
        }

    def prune_low_confidence(self, threshold: float = 0.3) -> int:
        """Deactivate rules below confidence threshold."""
        count = 0
        for rule in self._rules.values():
            if rule.is_active and rule.confidence < threshold:
                rule.is_active = False
                count += 1

        if count > 0 and self.auto_persist:
            self._save_rules()

        return count

    def _save_rules(self) -> None:
        """Save rules to disk."""
        rules_file = self.storage_dir / "semantic_rules.json"
        data = {rid: rule.to_dict() for rid, rule in self._rules.items()}
        with open(rules_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_rules(self) -> None:
        """Load rules from disk."""
        rules_file = self.storage_dir / "semantic_rules.json"
        if not rules_file.exists():
            return

        try:
            with open(rules_file, 'r') as f:
                data = json.load(f)
            for rid, rule_data in data.items():
                self._rules[rid] = SemanticRule.from_dict(rule_data)
        except Exception as e:
            logger.warning(f"Failed to load semantic rules: {e}")


# Singleton
_semantic_memory: Optional[SemanticMemory] = None


def get_semantic_memory() -> SemanticMemory:
    """Get or create semantic memory singleton."""
    global _semantic_memory
    if _semantic_memory is None:
        _semantic_memory = SemanticMemory()
    return _semantic_memory
