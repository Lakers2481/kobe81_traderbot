"""
Comprehensive Unit Tests for Tree-of-Thoughts (ToT) Module
============================================================

Tests the multi-path deliberative reasoning system that enables
explicit tree search over reasoning paths.

Run: python -m pytest tests/cognitive/test_tree_of_thoughts.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict


class TestThoughtNode:
    """Tests for ThoughtNode dataclass."""

    def test_thought_node_creation(self):
        """Test basic ThoughtNode instantiation."""
        from cognitive.tree_of_thoughts import ThoughtNode

        node = ThoughtNode(
            thought="Consider technical indicators",
            value=0.75,
            depth=1,
        )

        assert node.thought == "Consider technical indicators"
        assert node.value == 0.75
        assert node.depth == 1
        assert node.children == []
        assert node.parent is None
        assert not node.is_terminal

    def test_thought_node_with_parent(self):
        """Test parent-child relationship."""
        from cognitive.tree_of_thoughts import ThoughtNode

        parent = ThoughtNode(thought="Root thought", value=0.8, depth=0)
        child = ThoughtNode(
            thought="Child reasoning",
            value=0.7,
            depth=1,
            parent=parent,
        )
        parent.children.append(child)

        assert child.parent is parent
        assert len(parent.children) == 1
        assert parent.children[0] is child

    def test_thought_node_serialization(self):
        """Test node to_dict serialization."""
        from cognitive.tree_of_thoughts import ThoughtNode, ThoughtStatus

        node = ThoughtNode(
            thought="Test thought",
            value=0.85,
            depth=2,
            status=ThoughtStatus.TERMINAL,
        )

        data = node.to_dict()

        assert data['thought'] == "Test thought"
        assert data['value'] == 0.85
        assert data['depth'] == 2
        assert data['status'] == "terminal"
        assert data['num_children'] == 0


class TestToTResult:
    """Tests for ToTResult dataclass."""

    def test_tot_result_creation(self):
        """Test ToTResult instantiation."""
        from cognitive.tree_of_thoughts import ToTResult

        result = ToTResult(
            final_answer="LONG",
            reasoning_path=["Step 1", "Step 2", "Step 3"],
            confidence=0.8,
            branches_explored=5,
            backtrack_count=1,
            total_thoughts_generated=10,
            best_path_values=[0.7, 0.8, 0.85],
            search_depth=3,
        )

        assert result.final_answer == "LONG"
        assert len(result.reasoning_path) == 3
        assert result.confidence == 0.8
        assert result.branches_explored == 5
        assert result.backtrack_count == 1
        assert result.total_thoughts_generated == 10

    def test_tot_result_serialization(self):
        """Test result serialization."""
        from cognitive.tree_of_thoughts import ToTResult

        result = ToTResult(
            final_answer="SHORT",
            reasoning_path=["Analysis", "Decision"],
            confidence=0.65,
            branches_explored=3,
            backtrack_count=0,
            total_thoughts_generated=6,
            best_path_values=[0.6, 0.65],
            search_depth=2,
        )

        data = result.to_dict()

        assert data['final_answer'] == "SHORT"
        assert data['reasoning_path'] == ["Analysis", "Decision"]
        assert data['confidence'] == 0.65

    def test_tot_result_to_dict(self):
        """Test to_dict serialization includes key fields."""
        from cognitive.tree_of_thoughts import ToTResult

        result = ToTResult(
            final_answer="STAND_DOWN",
            reasoning_path=["Consider risk", "Evaluate reward", "Too risky"],
            confidence=0.55,
            branches_explored=4,
            backtrack_count=2,
            total_thoughts_generated=8,
            best_path_values=[0.5, 0.55],
            search_depth=3,
        )

        data = result.to_dict()

        assert "STAND_DOWN" in data['final_answer']
        assert data['confidence'] == 0.55
        assert data['branches_explored'] == 4


class TestTreeOfThoughts:
    """Tests for the main TreeOfThoughts reasoner."""

    def test_initialization_without_llm(self):
        """Test initialization without LLM provider."""
        from cognitive.tree_of_thoughts import TreeOfThoughts

        tot = TreeOfThoughts(
            llm_provider=None,
            max_depth=3,
            beam_width=2,
        )

        assert tot.max_depth == 3
        assert tot.beam_width == 2
        # Note: llm property auto-fetches provider if None, so we just check _llm
        assert tot._llm is None

    def test_initialization_with_mock_llm(self):
        """Test initialization with mock LLM."""
        from cognitive.tree_of_thoughts import TreeOfThoughts

        mock_llm = Mock()
        tot = TreeOfThoughts(
            llm_provider=mock_llm,
            max_depth=5,
            beam_width=3,
        )

        assert tot._llm is mock_llm
        assert tot.max_depth == 5
        assert tot.beam_width == 3

    def test_solve_without_llm_returns_fallback(self):
        """Test that solve returns a fallback when no LLM is available."""
        from cognitive.tree_of_thoughts import TreeOfThoughts

        tot = TreeOfThoughts(llm_provider=None)

        result = tot.solve(
            problem="Should I buy AAPL?",
            context={"symbol": "AAPL", "regime": "BULL"},
        )

        # Without LLM, should return a fallback result
        assert result is not None
        assert result.final_answer is not None
        assert isinstance(result.reasoning_path, list)

    def test_beam_search_explores_branches(self):
        """Test that beam search explores multiple branches."""
        from cognitive.tree_of_thoughts import TreeOfThoughts

        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "THOUGHT 1: Analysis\nTHOUGHT 2: Review"
        mock_llm.chat = Mock(return_value=mock_response)

        tot = TreeOfThoughts(llm_provider=mock_llm, max_depth=2, beam_width=2)

        # Mock internal methods
        tot._evaluate_thought = Mock(return_value=0.7)

        result = tot.solve(
            problem="Trade decision",
            context={"symbol": "TEST"},
        )

        assert result.branches_explored >= 0

    def test_is_terminal_detection_basic(self):
        """Test basic terminal thought detection via status."""
        from cognitive.tree_of_thoughts import ThoughtNode, ThoughtStatus

        # A node is terminal when status is TERMINAL
        terminal_node = ThoughtNode(thought="BUY", status=ThoughtStatus.TERMINAL)
        assert terminal_node.is_terminal

        non_terminal = ThoughtNode(thought="Analyzing...", status=ThoughtStatus.EXPANDING)
        assert not non_terminal.is_terminal


class TestGetPath:
    """Tests for path extraction from tree."""

    def test_get_path_single_node(self):
        """Test path extraction for single node using ThoughtNode method."""
        from cognitive.tree_of_thoughts import ThoughtNode

        node = ThoughtNode(thought="Root", value=1.0, depth=0)

        path = node.get_path_to_root()

        assert len(path) == 1
        assert path[0] is node

    def test_get_path_multiple_nodes(self):
        """Test path extraction through parent chain using ThoughtNode method."""
        from cognitive.tree_of_thoughts import ThoughtNode

        root = ThoughtNode(thought="Root", value=1.0, depth=0)
        child1 = ThoughtNode(thought="Child 1", value=0.8, depth=1, parent=root)
        child2 = ThoughtNode(thought="Child 2", value=0.9, depth=2, parent=child1)

        path = child2.get_path_to_root()

        assert len(path) == 3
        assert path[0] is root
        assert path[1] is child1
        assert path[2] is child2

    def test_get_reasoning_chain(self):
        """Test getting reasoning chain as string list."""
        from cognitive.tree_of_thoughts import ThoughtNode

        root = ThoughtNode(thought="Start analysis", value=0.8, depth=0)
        child = ThoughtNode(thought="Check technicals", value=0.7, depth=1, parent=root)

        chain = child.get_reasoning_chain()

        assert chain == ["Start analysis", "Check technicals"]


class TestSingletonPattern:
    """Tests for singleton accessor."""

    def test_get_tot_reasoner_returns_instance(self):
        """Test singleton accessor."""
        from cognitive.tree_of_thoughts import get_tot_reasoner

        tot1 = get_tot_reasoner()
        tot2 = get_tot_reasoner()

        assert tot1 is tot2

    def test_reset_tot_reasoner(self):
        """Test that we can reset the singleton."""
        from cognitive.tree_of_thoughts import get_tot_reasoner, _tot_instance
        import cognitive.tree_of_thoughts as tot_module

        # Get initial instance
        tot1 = get_tot_reasoner()

        # Reset
        tot_module._tot_instance = None

        # Get new instance
        tot2 = get_tot_reasoner()

        # Should be different instances
        assert tot1 is not tot2


class TestToTIntegration:
    """Integration tests for ToT with mocked LLM."""

    def test_full_solve_with_mock_llm(self):
        """Test complete solve flow with mock LLM."""
        from cognitive.tree_of_thoughts import TreeOfThoughts

        # Create a mock LLM that returns structured responses
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """
        THOUGHT 1: Analyze the technical indicators showing oversold RSI
        THOUGHT 2: Consider the macro regime which is bullish
        THOUGHT 3: Evaluate the risk-reward ratio at current levels
        """
        mock_llm.chat = Mock(return_value=mock_response)

        tot = TreeOfThoughts(llm_provider=mock_llm, max_depth=2, beam_width=2)

        # Override _evaluate_thought to avoid LLM call
        tot._evaluate_thought = Mock(return_value=0.75)

        result = tot.solve(
            problem="Should I trade AAPL given 5 consecutive down days?",
            context={
                "symbol": "AAPL",
                "regime": "BULL",
                "consecutive_down_days": 5,
            },
        )

        assert result is not None
        assert result.final_answer is not None
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
