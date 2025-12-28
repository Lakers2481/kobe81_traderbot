"""
Semantic Memory - The AI's Distilled Knowledge Base
=====================================================

This module provides the cognitive architecture with a long-term memory for
generalized knowledge, analogous to human semantic memory. While Episodic
Memory stores specific, detailed events ("I lost money on AAPL last Tuesday"),
Semantic Memory stores the abstract rules and "rules of thumb" extracted from
those events ("In a bearish market, momentum strategies are less reliable").

Core Functions:
- **Stores Generalized Rules:** Maintains a collection of `SemanticRule` objects,
  each representing a piece of learned knowledge.
- **Context Matching:** Uses a `ConditionMatcher` to find which rules are
  applicable to the current market context.
- **Confidence-Weighted Knowledge:** Each rule has a `confidence` score that
  is updated over time based on its predictive success.
- **Knowledge Extraction:** Includes methods to autonomously extract new rules
  by finding patterns in the data from Episodic Memory.
- **Forgetting Mechanism:** Can "forget" or deactivate rules that have proven to be
  unreliable, allowing the AI to adapt to changing market dynamics.

This component is the bridge between raw experience and actionable wisdom.

Usage:
    from cognitive.semantic_memory import get_semantic_memory

    memory = get_semantic_memory()

    # The ReflectionEngine might add a new rule after analyzing past trades.
    memory.add_rule(
        condition="vix > 30 AND regime = CHOPPY",
        action="reduce_position_size",
        parameters={'multiplier': 0.5},
        confidence=0.85
    )

    # The CognitiveBrain queries for applicable rules during deliberation.
    applicable_rules = memory.get_applicable_rules(current_market_context)
    for rule in applicable_rules:
        # Apply the rule's action...
        pass
"""

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import statistics

logger = logging.getLogger(__name__)


@dataclass
class SemanticRule:
    """
    A structured representation of a single piece of generalized knowledge, or a
    "rule of thumb," that the AI has learned.
    """
    rule_id: str
    condition: str          # A string describing the context when this rule applies (e.g., "regime = BULL AND vix < 20").
    action: str             # The suggested action to take (e.g., "increase_confidence", "reduce_position_size").
    parameters: Dict[str, Any] = field(default_factory=dict)  # Parameters for the action.
    confidence: float = 0.5       # The AI's confidence in this rule's validity (0.0 to 1.0).
    supporting_episodes: int = 0  # The number of past experiences that support this rule.
    contradicting_episodes: int = 0 # The number of past experiences that contradict this rule.
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None # The last time this rule was used in a decision.
    times_applied: int = 0
    times_successful: int = 0 # How many times applying this rule led to a good outcome.
    source: str = ""        # Where this rule originated from (e.g., "auto_extracted", "human_defined").
    tags: List[str] = field(default_factory=list)
    is_active: bool = True  # Inactive rules are "forgotten" and not used in deliberation.

    @property
    def success_rate(self) -> float:
        """The historical success rate of applying this rule."""
        return self.times_successful / self.times_applied if self.times_applied > 0 else 0.0

    @property
    def evidence_ratio(self) -> float:
        """The ratio of supporting evidence to total evidence."""
        total = self.supporting_episodes + self.contradicting_episodes
        return self.supporting_episodes / total if total > 0 else 0.5

    def to_dict(self) -> Dict:
        """Serializes the rule to a dictionary."""
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        d['last_updated'] = self.last_updated.isoformat()
        d['last_applied'] = self.last_applied.isoformat() if self.last_applied else None
        d['success_rate'] = self.success_rate # Add computed property
        return d

    @staticmethod
    def from_dict(d: Dict) -> 'SemanticRule':
        """Creates a SemanticRule object from a dictionary."""
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
    """A simple engine to evaluate if a context dictionary matches a rule's condition string."""

    def __init__(self):
        self._operators = {
            '>=': lambda a, b: float(a) >= float(b),
            '<=': lambda a, b: float(a) <= float(b),
            '>': lambda a, b: float(a) > float(b),
            '<': lambda a, b: float(a) < float(b),
            '=': lambda a, b: str(a).lower() == str(b).lower(),
            '!=': lambda a, b: str(a).lower() != str(b).lower(),
            'contains': lambda a, b: str(b).lower() in str(a).lower(),
        }

    def matches(self, condition_str: str, context: Dict[str, Any]) -> bool:
        """
        Checks if the given context matches the condition string.

        Supports simple logical combinations like:
        - "vix > 30"
        - "regime = BULL"
        - "vix > 25 AND regime = CHOPPY"
        """
        condition_str = condition_str.strip()

        # Recursively handle AND/OR logic.
        if ' AND ' in condition_str:
            return all(self.matches(p.strip(), context) for p in condition_str.split(' AND '))
        if ' OR ' in condition_str:
            return any(self.matches(p.strip(), context) for p in condition_str.split(' OR '))

        # Parse and evaluate a single condition (e.g., "vix > 30").
        for op_key, op_func in self._operators.items():
            if op_key in condition_str:
                parts = condition_str.split(op_key, 1)
                if len(parts) == 2:
                    field_name = parts[0].strip().lower()
                    target_value_str = parts[1].strip()

                    context_value = context.get(field_name)
                    if context_value is None:
                        return False # Field not in context, so condition cannot match.

                    # Attempt numeric comparison first, fall back to string comparison.
                    try:
                        return op_func(context_value, float(target_value_str))
                    except (ValueError, TypeError):
                        return op_func(context_value, target_value_str)
        return False


class SemanticMemory:
    """
    Manages the AI's long-term storage of generalized knowledge, handling the
    creation, retrieval, and maintenance of semantic rules.
    """

    def __init__(self, storage_dir: str = "state/cognitive", auto_persist: bool = True):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.auto_persist = auto_persist

        self._rules: Dict[str, SemanticRule] = {}
        self._matcher = ConditionMatcher()

        self._load_rules()
        logger.info(f"SemanticMemory initialized with {len(self._rules)} rules.")

    def add_rule(self, condition: str, action: str, **kwargs) -> SemanticRule:
        """
        Adds a new rule to the memory or updates an existing one if a similar
        rule is found.
        """
        rule_id = hashlib.md5(f"{condition}|{action}".encode()).hexdigest()[:12]

        if rule_id in self._rules:
            # If a rule with the same condition and action exists, strengthen it.
            existing_rule = self._rules[rule_id]
            existing_rule.supporting_episodes += kwargs.get('supporting_episodes', 1)
            existing_rule.confidence = min(1.0, existing_rule.confidence + 0.05)
            existing_rule.last_updated = datetime.now()
            logger.debug(f"Strengthened existing rule {rule_id} with new evidence.")
            if self.auto_persist: self._save_rules()
            return existing_rule

        # Otherwise, create a new rule.
        rule = SemanticRule(
            rule_id=rule_id,
            condition=condition,
            action=action,
            parameters=kwargs.get('parameters', {}),
            confidence=kwargs.get('confidence', 0.7),
            supporting_episodes=kwargs.get('supporting_episodes', 1),
            source=kwargs.get('source', 'unknown'),
            tags=kwargs.get('tags', []),
        )
        self._rules[rule_id] = rule

        if self.auto_persist: self._save_rules()
        logger.info(f"Added new semantic rule {rule_id}: IF {condition} THEN {action}.")
        return rule

    def get_applicable_rules(self, context: Dict[str, Any], min_confidence: float = 0.5) -> List[SemanticRule]:
        """
        Finds all active rules whose conditions match the given context.

        Returns:
            A list of matching `SemanticRule` objects, sorted by confidence.
        """
        applicable = []
        for rule in self._rules.values():
            if rule.is_active and rule.confidence >= min_confidence:
                try:
                    if self._matcher.matches(rule.condition, context):
                        applicable.append(rule)
                except Exception as e:
                    logger.warning(f"Error matching rule {rule.rule_id} ('{rule.condition}'): {e}")
        
        return sorted(applicable, key=lambda r: r.confidence, reverse=True)

    def record_rule_outcome(self, rule_id: str, successful: bool):
        """
        Updates a rule's statistics based on the outcome of its application.
        This is a key feedback mechanism for adjusting rule confidence.
        """
        if rule := self._rules.get(rule_id):
            rule.times_applied += 1
            rule.last_applied = datetime.now()
            if successful:
                rule.times_successful += 1
                rule.confidence = min(1.0, rule.confidence + 0.01) # Gently increase confidence.
            else:
                rule.confidence = max(0.1, rule.confidence - 0.05) # Penalize failure more harshly.
            
            # "Forget" rules that are consistently wrong.
            if rule.times_applied > 20 and rule.success_rate < 0.4:
                rule.is_active = False
                logger.info(f"Deactivated rule '{rule.condition}' due to poor performance ({rule.success_rate:.1%}).")
            
            if self.auto_persist: self._save_rules()

    def extract_rules_from_episodes(self, episodes: List[Any]) -> List[SemanticRule]:
        """
        The primary learning method. It analyzes a collection of `Episode`
        objects to find statistically significant patterns and create new rules.
        """
        from cognitive.episodic_memory import EpisodeOutcome
        new_rules = []
        
        # Group episodes by their context to find patterns.
        by_context: Dict[str, List[Any]] = {}
        for ep in episodes:
            context_str = self._extract_condition(ep)
            by_context.setdefault(context_str, []).append(ep)

        for context_str, context_episodes in by_context.items():
            if len(context_episodes) < 10: continue # Need minimum sample size.

            wins = [e for e in context_episodes if e.outcome == EpisodeOutcome.WIN]
            losses = [e for e in context_episodes if e.outcome == EpisodeOutcome.LOSS]
            if not wins and not losses: continue
            
            win_rate = len(wins) / (len(wins) + len(losses))

            # If a strong positive pattern is found, create a rule to increase confidence.
            if win_rate > 0.7:
                rule = self.add_rule(
                    condition=context_str,
                    action="increase_confidence",
                    confidence=win_rate,
                    supporting_episodes=len(wins),
                    source="Auto-extracted from winning episodes",
                    tags=['auto_extracted', 'positive']
                )
                new_rules.append(rule)
            # If a strong negative pattern is found, create a rule to decrease confidence.
            elif win_rate < 0.3:
                rule = self.add_rule(
                    condition=context_str,
                    action="reduce_confidence",
                    confidence=1.0 - win_rate,
                    supporting_episodes=len(losses),
                    source="Auto-extracted from losing episodes",
                    tags=['auto_extracted', 'caution']
                )
                new_rules.append(rule)
        
        if new_rules:
            logger.info(f"Extracted {len(new_rules)} new semantic rules from {len(episodes)} episodes.")
        return new_rules

    def _extract_condition(self, episode: Any) -> str:
        """Helper to create a condition string from an episode's context."""
        parts = []
        if regime := episode.market_context.get('regime'):
            parts.append(f"regime = {regime}")
        if strategy := episode.signal_context.get('strategy'):
            parts.append(f"strategy = {strategy}")
        return " AND ".join(parts) or "unknown"

    def get_all_rules(self, active_only: bool = True) -> List[SemanticRule]:
        """Returns all rules in memory, optionally filtering for active ones."""
        rules = self._rules.values()
        if active_only:
            rules = [r for r in rules if r.is_active]
        return sorted(rules, key=lambda r: r.confidence, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the contents of the semantic memory."""
        rules = list(self._rules.values())
        active = [r for r in rules if r.is_active]
        return {
            'total_rules': len(rules),
            'active_rules': len(active),
            'avg_confidence': statistics.mean(r.confidence for r in active) if active else 0,
        }

    def prune_low_confidence(self, threshold: float = 0.4) -> int:
        """
        Deactivates ("forgets") rules that have fallen below a confidence threshold,
        keeping the knowledge base relevant.
        """
        deactivated_count = 0
        for rule in self._rules.values():
            if rule.is_active and rule.confidence < threshold:
                rule.is_active = False
                deactivated_count += 1
        
        if deactivated_count > 0:
            logger.info(f"Pruned {deactivated_count} low-confidence rules from memory.")
            if self.auto_persist: self._save_rules()
        return deactivated_count

    def _save_rules(self) -> None:
        """Saves the entire rule base to a JSON file."""
        try:
            with open(self.storage_dir / "semantic_rules.json", 'w') as f:
                json.dump({rid: r.to_dict() for rid, r in self._rules.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save semantic rules: {e}")

    def _load_rules(self) -> None:
        """Loads the rule base from a JSON file on startup."""
        rules_file = self.storage_dir / "semantic_rules.json"
        if not rules_file.exists(): return

        try:
            with open(rules_file, 'r') as f:
                data = json.load(f)
            for rid, rule_data in data.items():
                self._rules[rid] = SemanticRule.from_dict(rule_data)
        except Exception as e:
            logger.warning(f"Failed to load semantic rules from {rules_file}: {e}. Starting fresh.")
            self._rules = {}

# --- Singleton Implementation ---
_semantic_memory: Optional[SemanticMemory] = None
_lock = threading.Lock()

def get_semantic_memory() -> SemanticMemory:
    """Factory function to get the singleton instance of SemanticMemory."""
    global _semantic_memory
    if _semantic_memory is None:
        with _lock:
            if _semantic_memory is None:
                _semantic_memory = SemanticMemory()
    return _semantic_memory
