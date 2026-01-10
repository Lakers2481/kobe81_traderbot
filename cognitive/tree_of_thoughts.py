"""
Tree-of-Thoughts (ToT) Reasoning Framework
===========================================

Implements multi-path deliberative reasoning with tree search, evaluation,
and backtracking for complex trading decisions.

Based on: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
(Yao et al., 2023)

Key improvements over linear Chain-of-Thought:
- Explores multiple reasoning paths in parallel
- Uses heuristic evaluation to prune unpromising branches
- Backtracking when reaching dead ends
- Returns the best-evaluated path with confidence score

Usage:
    from cognitive.tree_of_thoughts import TreeOfThoughts, get_tot_reasoner
    from llm import get_provider

    tot = get_tot_reasoner()  # Singleton instance
    result = tot.solve(
        problem="Should I buy AAPL given bearish macro but bullish technicals?",
        context={
            "market_context": {...},
            "conflicting_signals": [...],
        }
    )

    print(result.final_answer)
    print(result.reasoning_path)
    print(f"Confidence: {result.confidence:.2%}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ThoughtStatus(Enum):
    """Status of a thought node in the tree."""
    EXPANDING = "expanding"      # Still generating children
    EVALUATED = "evaluated"      # Value assigned, ready for selection
    PRUNED = "pruned"           # Eliminated from consideration
    TERMINAL = "terminal"        # Reached a final answer
    BACKTRACKED = "backtracked"  # Node was abandoned due to dead end


@dataclass
class ThoughtNode:
    """
    A single node in the thought tree.

    Each node represents one step in a chain of reasoning.
    Nodes can have multiple children (branching thoughts) and
    track their own value/quality score.
    """
    thought: str                              # The reasoning step content
    value: float = 0.5                        # Heuristic evaluation (0-1)
    depth: int = 0                            # Tree depth (root = 0)
    status: ThoughtStatus = ThoughtStatus.EXPANDING
    children: List['ThoughtNode'] = field(default_factory=list)
    parent: Optional['ThoughtNode'] = None
    evaluation_reasoning: str = ""            # Why this value was assigned
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def is_terminal(self) -> bool:
        """Check if this node represents a final answer."""
        return self.status == ThoughtStatus.TERMINAL

    @property
    def is_pruned(self) -> bool:
        """Check if this node was pruned."""
        return self.status == ThoughtStatus.PRUNED

    def get_path_to_root(self) -> List['ThoughtNode']:
        """Get the path from this node back to the root."""
        path = [self]
        current = self
        while current.parent is not None:
            current = current.parent
            path.append(current)
        return list(reversed(path))

    def get_reasoning_chain(self) -> List[str]:
        """Get the reasoning chain as a list of thought strings."""
        return [node.thought for node in self.get_path_to_root()]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'thought': self.thought,
            'value': round(self.value, 3),
            'depth': self.depth,
            'status': self.status.value,
            'num_children': len(self.children),
            'evaluation_reasoning': self.evaluation_reasoning,
        }


@dataclass
class ToTResult:
    """
    Result of Tree-of-Thoughts reasoning.

    Contains the final answer, the reasoning path that led to it,
    and statistics about the search process.
    """
    final_answer: str                  # The concluded decision/answer
    reasoning_path: List[str]          # Step-by-step reasoning that led here
    confidence: float                  # Confidence in the answer (0-1)
    branches_explored: int             # Total branches considered
    backtrack_count: int               # Times we had to backtrack
    total_thoughts_generated: int      # Total thought nodes created
    best_path_values: List[float]      # Values along the winning path
    search_depth: int                  # Maximum depth reached
    processing_time_ms: int = 0        # Time taken for search
    alternative_answers: List[Dict] = field(default_factory=list)  # Other considered answers

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'final_answer': self.final_answer,
            'reasoning_path': self.reasoning_path,
            'confidence': round(self.confidence, 3),
            'branches_explored': self.branches_explored,
            'backtrack_count': self.backtrack_count,
            'total_thoughts_generated': self.total_thoughts_generated,
            'search_depth': self.search_depth,
            'processing_time_ms': self.processing_time_ms,
        }


class TreeOfThoughts:
    """
    Multi-path deliberative reasoning with tree search.

    This implements the Tree-of-Thoughts (ToT) framework which dramatically
    improves reasoning on complex problems by:

    1. Generating multiple possible reasoning paths (branching)
    2. Evaluating each path with a value function
    3. Using beam search to focus on promising paths
    4. Backtracking when reaching dead ends

    For trading decisions, this is particularly useful when:
    - Multiple conflicting signals exist (HMM says BULL, technicals say BEAR)
    - High uncertainty requires exploring alternatives
    - Complex multi-factor analysis is needed
    """

    # Prompts for thought generation and evaluation
    THOUGHT_GENERATION_PROMPT = """You are analyzing a trading decision problem.

Problem: {problem}

Current reasoning so far:
{current_reasoning}

Market Context:
{context_summary}

Generate {num_thoughts} different possible next reasoning steps.
Each should explore a DIFFERENT approach, consideration, or angle.

Think about:
- Different interpretations of the signals
- Alternative scenarios (bullish, bearish, neutral)
- Risk factors and edge cases
- Historical patterns and analogies

Format your response as:
THOUGHT 1: [Your first reasoning approach]
THOUGHT 2: [Your second, different reasoning approach]
THOUGHT 3: [Your third, different reasoning approach]

Make each thought substantive (2-4 sentences) and clearly distinct."""

    THOUGHT_EVALUATION_PROMPT = """You are evaluating a reasoning step in a trading analysis.

Original Problem: {problem}

Proposed Reasoning Step:
{thought}

Context:
{context_summary}

Evaluate this reasoning step on a scale from 0.0 to 1.0:
- 1.0 = Highly promising, logically sound, leads toward a clear answer
- 0.7 = Good reasoning, relevant considerations
- 0.5 = Neutral, uncertain value
- 0.3 = Weak reasoning, may be off-track
- 0.0 = Dead end, flawed logic, or irrelevant

Consider:
1. Logical soundness - Is the reasoning valid?
2. Relevance - Does it address the core problem?
3. Completeness - Does it consider key factors?
4. Progress - Does it move toward an actionable answer?

Respond with EXACTLY this format:
SCORE: [number between 0.0 and 1.0]
REASONING: [Brief explanation of why this score]"""

    TERMINAL_CHECK_PROMPT = """Determine if this reasoning chain reaches a clear, actionable conclusion.

Problem: {problem}

Reasoning Chain:
{reasoning_chain}

Does this chain provide a CLEAR, ACTIONABLE answer to the trading question?
A terminal answer should include:
- A clear decision (BUY, SELL, HOLD, or STAND_DOWN)
- Confidence level
- Key supporting reasons

Respond with EXACTLY:
IS_TERMINAL: [YES or NO]
ANSWER: [The clear decision if terminal, or "INCOMPLETE" if not]"""

    def __init__(
        self,
        llm_provider=None,
        max_depth: int = 5,
        beam_width: int = 3,
        min_value_threshold: float = 0.3,
        temperature: float = 0.7,
    ):
        """
        Initialize Tree-of-Thoughts reasoner.

        Args:
            llm_provider: LLM provider for generating/evaluating thoughts.
                         If None, will lazy-load from router.
            max_depth: Maximum tree depth (reasoning steps)
            beam_width: Number of top branches to keep (beam search)
            min_value_threshold: Minimum value to continue a branch
            temperature: LLM temperature for thought diversity
        """
        self._llm = llm_provider
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.min_value_threshold = min_value_threshold
        self.temperature = temperature

        # Statistics tracking
        self.branches_explored = 0
        self.backtrack_count = 0
        self.total_thoughts = 0

        logger.info(
            f"TreeOfThoughts initialized: depth={max_depth}, "
            f"beam_width={beam_width}, min_threshold={min_value_threshold}"
        )

    @property
    def llm(self):
        """Lazy-load LLM provider."""
        if self._llm is None:
            from llm.router import get_provider
            self._llm = get_provider(task_type="reasoning")
        return self._llm

    def solve(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToTResult:
        """
        Solve a problem using Tree-of-Thoughts reasoning.

        Args:
            problem: The problem/question to solve
            context: Additional context (market data, signals, etc.)

        Returns:
            ToTResult with final answer, reasoning path, and statistics
        """
        import time
        start_time = time.time()

        # Reset statistics
        self.branches_explored = 0
        self.backtrack_count = 0
        self.total_thoughts = 0

        context = context or {}
        context_summary = self._summarize_context(context)

        # Create root node
        root = ThoughtNode(
            thought=f"Analyzing: {problem}",
            value=1.0,
            depth=0,
            status=ThoughtStatus.EVALUATED,
        )
        self.total_thoughts = 1

        # Run beam search
        best_path = self._beam_search(root, problem, context_summary)

        # Extract final answer from best path
        final_node = best_path[-1] if best_path else root
        final_answer = self._extract_final_answer(final_node, problem, context_summary)

        # Calculate confidence from path values
        path_values = [node.value for node in best_path]
        confidence = self._calculate_confidence(path_values)

        processing_time = int((time.time() - start_time) * 1000)

        result = ToTResult(
            final_answer=final_answer,
            reasoning_path=[node.thought for node in best_path],
            confidence=confidence,
            branches_explored=self.branches_explored,
            backtrack_count=self.backtrack_count,
            total_thoughts_generated=self.total_thoughts,
            best_path_values=path_values,
            search_depth=max(node.depth for node in best_path) if best_path else 0,
            processing_time_ms=processing_time,
        )

        logger.info(
            f"ToT completed: confidence={confidence:.2%}, "
            f"branches={self.branches_explored}, backtracks={self.backtrack_count}, "
            f"time={processing_time}ms"
        )

        return result

    def _beam_search(
        self,
        root: ThoughtNode,
        problem: str,
        context_summary: str,
    ) -> List[ThoughtNode]:
        """
        Perform beam search through the thought tree.

        Maintains a "beam" of the top-k most promising partial solutions
        and expands them until reaching terminal states or max depth.
        """
        beam = [root]

        for depth in range(self.max_depth):
            candidates = []

            for node in beam:
                if node.is_terminal or node.is_pruned:
                    candidates.append(node)  # Keep terminal nodes
                    continue

                # Generate child thoughts
                children = self._expand_node(node, problem, context_summary)

                for child in children:
                    # Evaluate child
                    self._evaluate_node(child, problem, context_summary)

                    # Check if terminal
                    if self._check_terminal(child, problem, context_summary):
                        child.status = ThoughtStatus.TERMINAL

                    if child.value >= self.min_value_threshold:
                        candidates.append(child)
                        self.branches_explored += 1
                    else:
                        child.status = ThoughtStatus.PRUNED

            if not candidates:
                # All branches pruned - backtrack
                self.backtrack_count += 1
                logger.warning(f"All branches pruned at depth {depth}, backtracking")
                beam = self._backtrack(beam)
                if not beam:
                    break
                continue

            # Select top-k candidates for next iteration
            candidates.sort(key=lambda n: n.value, reverse=True)
            beam = candidates[:self.beam_width]

            # Check if we have a high-confidence terminal node
            terminal_nodes = [n for n in beam if n.is_terminal]
            if terminal_nodes:
                best_terminal = max(terminal_nodes, key=lambda n: n.value)
                if best_terminal.value >= 0.7:
                    # High confidence terminal - stop early
                    return best_terminal.get_path_to_root()

        # Return best path found
        if beam:
            best_node = max(beam, key=lambda n: n.value)
            return best_node.get_path_to_root()

        return [root]

    def _expand_node(
        self,
        node: ThoughtNode,
        problem: str,
        context_summary: str,
    ) -> List[ThoughtNode]:
        """Generate child thought nodes from a parent."""
        current_reasoning = "\n".join(node.get_reasoning_chain())

        prompt = self.THOUGHT_GENERATION_PROMPT.format(
            problem=problem,
            current_reasoning=current_reasoning,
            context_summary=context_summary,
            num_thoughts=self.beam_width,
        )

        try:
            from llm.provider_base import LLMMessage
            response = self.llm.chat(
                [LLMMessage(role="user", content=prompt)],
                temperature=self.temperature,
            )
            thoughts = self._parse_thoughts(response.content)
        except Exception as e:
            logger.error(f"Failed to generate thoughts: {e}")
            thoughts = [f"Continue analyzing {problem}"]

        children = []
        for thought in thoughts:
            child = ThoughtNode(
                thought=thought,
                depth=node.depth + 1,
                parent=node,
            )
            node.children.append(child)
            children.append(child)
            self.total_thoughts += 1

        return children

    def _evaluate_node(
        self,
        node: ThoughtNode,
        problem: str,
        context_summary: str,
    ) -> None:
        """Assign a value to a thought node."""
        prompt = self.THOUGHT_EVALUATION_PROMPT.format(
            problem=problem,
            thought=node.thought,
            context_summary=context_summary,
        )

        try:
            from llm.provider_base import LLMMessage
            response = self.llm.chat(
                [LLMMessage(role="user", content=prompt)],
                temperature=0.1,  # Low temperature for consistent evaluation
            )
            value, reasoning = self._parse_evaluation(response.content)
            node.value = value
            node.evaluation_reasoning = reasoning
        except Exception as e:
            logger.error(f"Failed to evaluate thought: {e}")
            node.value = 0.5  # Default neutral value

        node.status = ThoughtStatus.EVALUATED

    def _check_terminal(
        self,
        node: ThoughtNode,
        problem: str,
        context_summary: str,
    ) -> bool:
        """Check if a node represents a terminal (final) answer."""
        reasoning_chain = "\n".join(node.get_reasoning_chain())

        prompt = self.TERMINAL_CHECK_PROMPT.format(
            problem=problem,
            reasoning_chain=reasoning_chain,
        )

        try:
            from llm.provider_base import LLMMessage
            response = self.llm.chat(
                [LLMMessage(role="user", content=prompt)],
                temperature=0.1,
            )
            return self._parse_terminal_check(response.content)
        except Exception as e:
            logger.error(f"Failed to check terminal: {e}")
            return False

    def _backtrack(self, beam: List[ThoughtNode]) -> List[ThoughtNode]:
        """Backtrack to parent nodes when all branches are dead ends."""
        parents = set()
        for node in beam:
            if node.parent is not None:
                parents.add(node.parent)

        return list(parents) if parents else []

    def _parse_thoughts(self, response: str) -> List[str]:
        """Parse multiple thoughts from LLM response."""
        thoughts = []

        # Look for THOUGHT N: pattern
        pattern = r'THOUGHT\s*\d+:\s*(.+?)(?=THOUGHT\s*\d+:|$)'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            thought = match.strip()
            if thought and len(thought) > 10:  # Minimum length check
                thoughts.append(thought)

        # Fallback: split by newlines if pattern not found
        if not thoughts:
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            thoughts = [l for l in lines if len(l) > 20][:self.beam_width]

        return thoughts[:self.beam_width]

    def _parse_evaluation(self, response: str) -> Tuple[float, str]:
        """Parse value and reasoning from evaluation response."""
        value = 0.5
        reasoning = ""

        # Extract score
        score_match = re.search(r'SCORE:\s*([\d.]+)', response, re.IGNORECASE)
        if score_match:
            try:
                value = float(score_match.group(1))
                value = max(0.0, min(1.0, value))  # Clamp to [0, 1]
            except ValueError:
                pass

        # Extract reasoning
        reason_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL | re.IGNORECASE)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        return value, reasoning

    def _parse_terminal_check(self, response: str) -> bool:
        """Parse terminal check response."""
        match = re.search(r'IS_TERMINAL:\s*(YES|NO)', response, re.IGNORECASE)
        if match:
            return match.group(1).upper() == "YES"
        return False

    def _extract_final_answer(
        self,
        node: ThoughtNode,
        problem: str,
        context_summary: str,
    ) -> str:
        """Extract a clear final answer from the best node."""
        reasoning_chain = "\n".join(node.get_reasoning_chain())

        prompt = f"""Based on this reasoning chain, provide a clear, final answer.

Problem: {problem}

Reasoning:
{reasoning_chain}

Provide your final answer in this format:
DECISION: [BUY/SELL/HOLD/STAND_DOWN]
CONFIDENCE: [HIGH/MEDIUM/LOW]
SUMMARY: [1-2 sentence summary of the key reasoning]"""

        try:
            from llm.provider_base import LLMMessage
            response = self.llm.chat(
                [LLMMessage(role="user", content=prompt)],
                temperature=0.1,
            )
            return response.content
        except Exception as e:
            logger.error(f"Failed to extract final answer: {e}")
            return f"Unable to reach conclusion. Best reasoning: {node.thought}"

    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """Summarize context dict into a readable string."""
        if not context:
            return "No additional context provided."

        parts = []
        for key, value in context.items():
            if isinstance(value, dict):
                parts.append(f"{key}: {', '.join(f'{k}={v}' for k, v in value.items())}")
            elif isinstance(value, list):
                parts.append(f"{key}: {', '.join(str(v) for v in value[:5])}")
            else:
                parts.append(f"{key}: {value}")

        return "\n".join(parts)

    def _calculate_confidence(self, path_values: List[float]) -> float:
        """Calculate overall confidence from path values."""
        if not path_values:
            return 0.0

        # Use geometric mean to penalize any weak links in the chain
        import math
        product = 1.0
        for v in path_values:
            product *= max(v, 0.01)  # Avoid zero

        geometric_mean = product ** (1 / len(path_values))

        # Also consider minimum value (weakest link matters)
        min_value = min(path_values)

        # Weighted combination: 70% geometric mean, 30% min value
        return 0.7 * geometric_mean + 0.3 * min_value


# Singleton instance
_tot_instance: Optional[TreeOfThoughts] = None


def get_tot_reasoner(
    max_depth: int = 5,
    beam_width: int = 3,
) -> TreeOfThoughts:
    """
    Get the singleton Tree-of-Thoughts reasoner.

    Args:
        max_depth: Maximum reasoning depth
        beam_width: Number of branches to explore

    Returns:
        TreeOfThoughts instance
    """
    global _tot_instance
    if _tot_instance is None:
        _tot_instance = TreeOfThoughts(
            max_depth=max_depth,
            beam_width=beam_width,
        )
    return _tot_instance


def solve_with_tot(
    problem: str,
    context: Optional[Dict[str, Any]] = None,
) -> ToTResult:
    """
    Convenience function to solve a problem with Tree-of-Thoughts.

    Args:
        problem: The problem to solve
        context: Additional context

    Returns:
        ToTResult with answer and reasoning
    """
    tot = get_tot_reasoner()
    return tot.solve(problem, context)
