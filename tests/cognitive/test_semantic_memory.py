"""
Unit tests for cognitive/semantic_memory.py

Tests the AI's knowledge base of general trading rules and patterns.
"""
import tempfile
from pathlib import Path


class TestSemanticRuleDataclass:
    """Tests for the SemanticRule dataclass."""

    def test_rule_creation(self):
        from cognitive.semantic_memory import SemanticRule

        rule = SemanticRule(
            rule_id="rule_001",
            condition="RSI < 10 AND regime = BULL",
            action="increase_confidence",
        )

        assert rule.rule_id == "rule_001"
        assert "RSI" in rule.condition
        assert rule.confidence == 0.5  # Default

    def test_rule_with_custom_confidence(self):
        from cognitive.semantic_memory import SemanticRule

        rule = SemanticRule(
            rule_id="rule_002",
            condition="IBS < 0.15",
            action="increase_confidence",
            confidence=0.85,
        )

        assert rule.confidence == 0.85

    def test_rule_to_dict(self):
        from cognitive.semantic_memory import SemanticRule

        rule = SemanticRule(
            rule_id="rule_003",
            condition="Price > SMA200",
            action="allow_long_trades",
        )
        d = rule.to_dict()

        assert d['rule_id'] == "rule_003"
        assert 'condition' in d
        assert 'action' in d

    def test_rule_from_dict(self):
        from cognitive.semantic_memory import SemanticRule

        original = SemanticRule(
            rule_id="rule_004",
            condition="vix > 30",
            action="reduce_position_size",
            confidence=0.75,
        )
        d = original.to_dict()
        restored = SemanticRule.from_dict(d)

        assert restored.rule_id == original.rule_id
        assert restored.condition == original.condition
        assert restored.confidence == original.confidence

    def test_rule_success_rate_property(self):
        from cognitive.semantic_memory import SemanticRule

        rule = SemanticRule(
            rule_id="rule_005",
            condition="regime = BULL",
            action="increase_confidence",
            times_applied=10,
            times_successful=7,
        )

        assert rule.success_rate == 0.7

    def test_rule_evidence_ratio_property(self):
        from cognitive.semantic_memory import SemanticRule

        rule = SemanticRule(
            rule_id="rule_006",
            condition="regime = BEAR",
            action="reduce_confidence",
            supporting_episodes=8,
            contradicting_episodes=2,
        )

        assert rule.evidence_ratio == 0.8


class TestConditionMatcher:
    """Tests for the ConditionMatcher class."""

    def test_simple_equality(self):
        from cognitive.semantic_memory import ConditionMatcher

        matcher = ConditionMatcher()

        assert matcher.matches("regime = BULL", {'regime': 'BULL'}) is True
        assert matcher.matches("regime = BULL", {'regime': 'BEAR'}) is False

    def test_numeric_comparison(self):
        from cognitive.semantic_memory import ConditionMatcher

        matcher = ConditionMatcher()

        assert matcher.matches("vix > 30", {'vix': 35}) is True
        assert matcher.matches("vix > 30", {'vix': 25}) is False
        assert matcher.matches("vix < 20", {'vix': 15}) is True

    def test_and_condition(self):
        from cognitive.semantic_memory import ConditionMatcher

        matcher = ConditionMatcher()

        context = {'regime': 'BULL', 'vix': 15}
        assert matcher.matches("regime = BULL AND vix < 20", context) is True
        assert matcher.matches("regime = BULL AND vix > 20", context) is False

    def test_or_condition(self):
        from cognitive.semantic_memory import ConditionMatcher

        matcher = ConditionMatcher()

        context = {'regime': 'BULL', 'vix': 15}
        assert matcher.matches("regime = BEAR OR vix < 20", context) is True
        assert matcher.matches("regime = BEAR OR vix > 20", context) is False

    def test_missing_field(self):
        from cognitive.semantic_memory import ConditionMatcher

        matcher = ConditionMatcher()

        # If field is missing from context, condition should not match
        assert matcher.matches("unknown_field = value", {'regime': 'BULL'}) is False


class TestSemanticMemoryInitialization:
    """Tests for SemanticMemory initialization."""

    def test_default_initialization(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            assert memory is not None
            assert len(memory._rules) == 0

    def test_initialization_creates_directory(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "semantic_state"
            SemanticMemory(storage_dir=str(storage_path), auto_persist=False)

            assert storage_path.exists()


class TestAddingRules:
    """Tests for adding rules to semantic memory."""

    def test_add_rule(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            rule = memory.add_rule(
                condition="VIX > 30",
                action="reduce_position_size",
                confidence=0.7,
            )

            assert rule is not None
            assert len(memory._rules) == 1

    def test_add_rule_with_parameters(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            rule = memory.add_rule(
                condition="RSI < 10",
                action="increase_confidence",
                parameters={'multiplier': 1.2},
            )

            assert rule.parameters['multiplier'] == 1.2

    def test_add_duplicate_rule_strengthens(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            rule1 = memory.add_rule("VIX > 30", "reduce_size", confidence=0.5)
            initial_confidence = rule1.confidence

            # Adding same condition+action should strengthen it
            rule2 = memory.add_rule("VIX > 30", "reduce_size")

            assert len(memory._rules) == 1  # Still just one rule
            assert rule2.confidence > initial_confidence


class TestGetApplicableRules:
    """Tests for getting applicable rules based on context."""

    def test_get_applicable_rules_empty(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            results = memory.get_applicable_rules({'regime': 'BULL'})
            assert results == []

    def test_get_applicable_rules_matching(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            memory.add_rule("regime = BULL", "increase_confidence", confidence=0.8)
            memory.add_rule("regime = BEAR", "reduce_confidence", confidence=0.8)
            memory.add_rule("vix > 30", "reduce_size", confidence=0.8)

            results = memory.get_applicable_rules({'regime': 'bull', 'vix': 40})

            # Should match "regime = BULL" and "vix > 30"
            assert len(results) == 2

    def test_get_applicable_rules_respects_min_confidence(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            memory.add_rule("regime = BULL", "action1", confidence=0.8)
            memory.add_rule("regime = BULL", "action2", confidence=0.4)

            results = memory.get_applicable_rules({'regime': 'BULL'}, min_confidence=0.6)

            assert len(results) == 1
            assert results[0].confidence >= 0.6


class TestRecordRuleOutcome:
    """Tests for recording rule outcomes."""

    def test_record_successful_outcome(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            rule = memory.add_rule("regime = BULL", "increase_conf", confidence=0.7)
            initial_conf = rule.confidence

            memory.record_rule_outcome(rule.rule_id, successful=True)

            # Confidence should increase
            assert memory._rules[rule.rule_id].confidence > initial_conf

    def test_record_failed_outcome(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            rule = memory.add_rule("regime = BEAR", "reduce_conf", confidence=0.7)

            memory.record_rule_outcome(rule.rule_id, successful=False)

            # Confidence should decrease
            assert memory._rules[rule.rule_id].confidence < 0.7

    def test_rule_deactivated_after_many_failures(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            rule = memory.add_rule("bad_condition", "bad_action", confidence=0.5)

            # Simulate many applications with low success
            memory._rules[rule.rule_id].times_applied = 25
            memory._rules[rule.rule_id].times_successful = 5

            memory.record_rule_outcome(rule.rule_id, successful=False)

            # Rule should be deactivated due to poor performance
            assert memory._rules[rule.rule_id].is_active is False


class TestRuleStatistics:
    """Tests for rule statistics and metrics."""

    def test_get_stats_empty(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            stats = memory.get_stats()
            assert stats['total_rules'] == 0

    def test_get_stats_with_rules(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            memory.add_rule("C1", "A1", confidence=0.8)
            memory.add_rule("C2", "A2", confidence=0.6)
            memory.add_rule("C3", "A3", confidence=0.4)

            stats = memory.get_stats()
            assert stats['total_rules'] == 3
            assert stats['active_rules'] == 3


class TestGetAllRules:
    """Tests for getting all rules."""

    def test_get_all_rules(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            memory.add_rule("C1", "A1", confidence=0.9)
            memory.add_rule("C2", "A2", confidence=0.7)
            memory.add_rule("C3", "A3", confidence=0.5)

            rules = memory.get_all_rules()

            assert len(rules) == 3
            # Should be sorted by confidence descending
            assert rules[0].confidence >= rules[1].confidence

    def test_get_all_rules_active_only(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            memory.add_rule("C1", "A1", confidence=0.9)
            rule2 = memory.add_rule("C2", "A2", confidence=0.7)

            # Deactivate one rule
            memory._rules[rule2.rule_id].is_active = False

            active_rules = memory.get_all_rules(active_only=True)
            all_rules = memory.get_all_rules(active_only=False)

            assert len(active_rules) == 1
            assert len(all_rules) == 2


class TestPruneLowConfidence:
    """Tests for pruning low confidence rules."""

    def test_prune_low_confidence(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            memory.add_rule("C1", "A1", confidence=0.8)
            memory.add_rule("C2", "A2", confidence=0.3)
            memory.add_rule("C3", "A3", confidence=0.2)

            deactivated = memory.prune_low_confidence(threshold=0.4)

            assert deactivated == 2
            active_rules = memory.get_all_rules(active_only=True)
            assert len(active_rules) == 1


class TestPersistence:
    """Tests for saving and loading semantic memory state."""

    def test_save_and_load_state(self):
        from cognitive.semantic_memory import SemanticMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate memory
            memory1 = SemanticMemory(storage_dir=tmpdir, auto_persist=True)
            memory1.add_rule("Saved condition", "Saved action", confidence=0.75)

            # Load in new instance
            memory2 = SemanticMemory(storage_dir=tmpdir, auto_persist=False)

            assert len(memory2._rules) == 1
            rule = list(memory2._rules.values())[0]
            assert rule.condition == "Saved condition"


class TestSingletonFactory:
    """Tests for the singleton factory function."""

    def test_get_semantic_memory(self):
        from cognitive.semantic_memory import get_semantic_memory

        memory1 = get_semantic_memory()
        memory2 = get_semantic_memory()

        assert memory1 is memory2
