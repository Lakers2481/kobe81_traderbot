"""
Self-Consistency Decoding
=========================

Implements self-consistency decoding for improved LLM reasoning reliability.

Based on: "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
(Wang et al., 2022)

Key concept:
- Sample multiple reasoning chains with diverse outputs (using temperature)
- Each chain independently arrives at an answer
- The final answer is the majority vote across all chains
- Confidence = fraction of chains agreeing on the answer

This reduces variance in LLM outputs and improves accuracy on reasoning tasks.

Usage:
    from cognitive.self_consistency import SelfConsistencyDecoder, get_self_consistency

    decoder = get_self_consistency()
    result = decoder.decode(
        prompt="Should I buy AAPL given the current market conditions?",
        n_samples=5,
    )

    print(f"Answer: {result.final_answer}")
    print(f"Agreement: {result.agreement_ratio:.0%}")
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class AnswerType(Enum):
    """Types of answers we can extract."""
    TRADING_DECISION = "trading_decision"  # BUY, SELL, HOLD, STAND_DOWN
    BOOLEAN = "boolean"                     # YES, NO
    NUMERIC = "numeric"                     # Numbers/percentages
    CATEGORICAL = "categorical"             # Multiple choice
    FREEFORM = "freeform"                   # Open-ended


@dataclass
class ReasoningChain:
    """A single reasoning chain generated during self-consistency."""
    chain: str                    # The full reasoning text
    answer: str                   # The extracted final answer
    raw_response: str             # Original LLM response
    confidence: float = 0.0       # Chain-level confidence (if provided)
    extraction_method: str = ""   # How the answer was extracted


@dataclass
class SelfConsistencyResult:
    """
    Result of self-consistency decoding.

    Contains the majority answer, all sampled chains, and statistics
    about the agreement level.
    """
    final_answer: str                      # The majority-voted answer
    confidence: float                      # Agreement ratio (fraction agreeing)
    agreement_ratio: float                 # Same as confidence (explicit name)
    best_chain: str                        # The best reasoning chain for the answer
    all_chains: List[ReasoningChain]       # All sampled chains
    all_answers: List[str]                 # All extracted answers
    answer_distribution: Dict[str, int]    # Count per answer
    n_samples: int                         # Total samples taken
    unique_answers: int                    # Number of distinct answers
    processing_time_ms: int = 0
    answer_type: AnswerType = AnswerType.FREEFORM

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'final_answer': self.final_answer,
            'confidence': round(self.confidence, 3),
            'agreement_ratio': round(self.agreement_ratio, 3),
            'n_samples': self.n_samples,
            'unique_answers': self.unique_answers,
            'answer_distribution': self.answer_distribution,
            'processing_time_ms': self.processing_time_ms,
        }

    @property
    def is_high_confidence(self) -> bool:
        """Check if we have high agreement (>80%)."""
        return self.confidence >= 0.8

    @property
    def is_disputed(self) -> bool:
        """Check if there's significant disagreement."""
        return self.unique_answers >= 3 or self.confidence < 0.5


class SelfConsistencyDecoder:
    """
    Self-consistency decoding: Sample multiple chains, pick majority answer.

    This technique improves reasoning reliability by:
    1. Using temperature to get diverse reasoning paths
    2. Independently deriving answers from each path
    3. Voting to select the most common answer
    4. Using agreement ratio as a confidence measure

    Particularly useful for:
    - Trading decisions with uncertain outcomes
    - Complex multi-factor analysis
    - Questions where the LLM might be inconsistent
    """

    # Prompt suffix to ensure clear, extractable answers
    ANSWER_SUFFIX = """

At the end of your response, provide your final answer in this format:
FINAL ANSWER: [Your clear, concise answer]"""

    # Trading-specific answer extraction patterns
    TRADING_PATTERNS = [
        (r'\b(BUY|LONG)\b', 'BUY'),
        (r'\b(SELL|SHORT)\b', 'SELL'),
        (r'\b(HOLD|NEUTRAL)\b', 'HOLD'),
        (r'\bSTAND[_\s]?DOWN\b', 'STAND_DOWN'),
        (r'\bNO[_\s]?TRADE\b', 'STAND_DOWN'),
    ]

    def __init__(
        self,
        llm_provider=None,
        n_samples: int = 5,
        temperature: float = 0.7,
        answer_type: AnswerType = AnswerType.TRADING_DECISION,
    ):
        """
        Initialize Self-Consistency decoder.

        Args:
            llm_provider: LLM provider for generating chains.
                         If None, will lazy-load from router.
            n_samples: Default number of reasoning chains to sample
            temperature: Temperature for diversity (higher = more diverse)
            answer_type: Type of answers to extract
        """
        self._llm = llm_provider
        self.n_samples = n_samples
        self.temperature = temperature
        self.answer_type = answer_type

        logger.info(
            f"SelfConsistencyDecoder initialized: n_samples={n_samples}, "
            f"temperature={temperature}, answer_type={answer_type.value}"
        )

    @property
    def llm(self):
        """Lazy-load LLM provider."""
        if self._llm is None:
            from llm.router import get_provider
            self._llm = get_provider(task_type="reasoning")
        return self._llm

    def decode(
        self,
        prompt: str,
        n_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        answer_type: Optional[AnswerType] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelfConsistencyResult:
        """
        Generate multiple reasoning chains and return majority answer.

        Args:
            prompt: The question/problem to solve
            n_samples: Number of chains to sample (override default)
            temperature: Sampling temperature (override default)
            answer_type: Type of answer to extract
            context: Additional context to include in prompt

        Returns:
            SelfConsistencyResult with majority answer and statistics
        """
        import time
        start_time = time.time()

        n = n_samples or self.n_samples
        temp = temperature or self.temperature
        atype = answer_type or self.answer_type

        # Prepare prompt with context and answer suffix
        full_prompt = self._prepare_prompt(prompt, context)

        # Sample multiple chains
        chains = []
        answers = []

        for i in range(n):
            try:
                chain = self._sample_chain(full_prompt, temp)
                answer = self._extract_answer(chain.raw_response, atype)
                chain.answer = answer
                chains.append(chain)
                answers.append(answer)
            except Exception as e:
                logger.warning(f"Failed to sample chain {i+1}: {e}")
                continue

        if not answers:
            logger.error("Failed to generate any valid chains")
            return self._empty_result()

        # Find majority answer
        answer_counts = Counter(answers)
        majority_answer, count = answer_counts.most_common(1)[0]
        agreement_ratio = count / len(answers)

        # Find best chain for the majority answer
        best_chain = self._find_best_chain(chains, majority_answer)

        processing_time = int((time.time() - start_time) * 1000)

        result = SelfConsistencyResult(
            final_answer=majority_answer,
            confidence=agreement_ratio,
            agreement_ratio=agreement_ratio,
            best_chain=best_chain.chain if best_chain else "",
            all_chains=chains,
            all_answers=answers,
            answer_distribution=dict(answer_counts),
            n_samples=len(answers),
            unique_answers=len(answer_counts),
            processing_time_ms=processing_time,
            answer_type=atype,
        )

        logger.info(
            f"Self-consistency: answer='{majority_answer}', "
            f"agreement={agreement_ratio:.0%}, "
            f"unique={len(answer_counts)}/{len(answers)}, "
            f"time={processing_time}ms"
        )

        return result

    def _prepare_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Prepare the full prompt with context and answer formatting."""
        parts = []

        # Add context if provided
        if context:
            context_str = "\n".join(
                f"- {k}: {v}" for k, v in context.items()
            )
            parts.append(f"Context:\n{context_str}\n")

        # Add main prompt
        parts.append(prompt)

        # Add answer suffix for clear extraction
        parts.append(self.ANSWER_SUFFIX)

        return "\n".join(parts)

    def _sample_chain(self, prompt: str, temperature: float) -> ReasoningChain:
        """Sample a single reasoning chain from the LLM."""
        from llm.provider_base import LLMMessage

        response = self.llm.chat(
            [LLMMessage(role="user", content=prompt)],
            temperature=temperature,
        )

        return ReasoningChain(
            chain=response.content,
            answer="",  # Will be filled by extract_answer
            raw_response=response.content,
        )

    def _extract_answer(self, response: str, answer_type: AnswerType) -> str:
        """Extract the final answer from an LLM response."""

        # Method 1: Look for explicit FINAL ANSWER tag
        final_match = re.search(
            r'FINAL\s*ANSWER:\s*(.+?)(?:\n|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if final_match:
            answer = final_match.group(1).strip()
            return self._normalize_answer(answer, answer_type)

        # Method 2: Look for DECISION, ANSWER, or CONCLUSION tags
        for tag in ['DECISION', 'ANSWER', 'CONCLUSION', 'RECOMMENDATION']:
            tag_match = re.search(
                rf'{tag}:\s*(.+?)(?:\n|$)',
                response,
                re.IGNORECASE | re.DOTALL
            )
            if tag_match:
                answer = tag_match.group(1).strip()
                return self._normalize_answer(answer, answer_type)

        # Method 3: For trading decisions, look for keywords
        if answer_type == AnswerType.TRADING_DECISION:
            for pattern, normalized in self.TRADING_PATTERNS:
                if re.search(pattern, response, re.IGNORECASE):
                    return normalized

        # Method 4: Look for YES/NO for boolean
        if answer_type == AnswerType.BOOLEAN:
            if re.search(r'\bYES\b', response, re.IGNORECASE):
                return "YES"
            if re.search(r'\bNO\b', response, re.IGNORECASE):
                return "NO"

        # Method 5: Last resort - take last non-empty line
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        if lines:
            return self._normalize_answer(lines[-1][:100], answer_type)

        return "UNKNOWN"

    def _normalize_answer(self, answer: str, answer_type: AnswerType) -> str:
        """Normalize the answer to a canonical form."""
        answer = answer.strip().upper()

        if answer_type == AnswerType.TRADING_DECISION:
            # Map various phrasings to standard trading decisions
            for pattern, normalized in self.TRADING_PATTERNS:
                if re.search(pattern, answer, re.IGNORECASE):
                    return normalized

            # If it's a short answer, return as-is
            if len(answer) <= 20:
                return answer

        elif answer_type == AnswerType.BOOLEAN:
            if 'YES' in answer or 'TRUE' in answer or 'CORRECT' in answer:
                return "YES"
            if 'NO' in answer or 'FALSE' in answer or 'INCORRECT' in answer:
                return "NO"

        # For longer answers, truncate
        return answer[:50] if len(answer) > 50 else answer

    def _find_best_chain(
        self,
        chains: List[ReasoningChain],
        target_answer: str,
    ) -> Optional[ReasoningChain]:
        """Find the best reasoning chain for a given answer."""
        matching_chains = [c for c in chains if c.answer == target_answer]

        if not matching_chains:
            return None

        # Prefer longer, more detailed chains
        return max(matching_chains, key=lambda c: len(c.chain))

    def _empty_result(self) -> SelfConsistencyResult:
        """Return an empty result for error cases."""
        return SelfConsistencyResult(
            final_answer="UNKNOWN",
            confidence=0.0,
            agreement_ratio=0.0,
            best_chain="",
            all_chains=[],
            all_answers=[],
            answer_distribution={},
            n_samples=0,
            unique_answers=0,
        )


class EnhancedSelfConsistency(SelfConsistencyDecoder):
    """
    Enhanced self-consistency with additional features:

    1. Weighted voting based on chain quality
    2. Outlier detection and filtering
    3. Confidence calibration
    4. Reasoning quality scoring
    """

    def __init__(
        self,
        llm_provider=None,
        n_samples: int = 7,
        temperature: float = 0.7,
        use_weighted_voting: bool = True,
        filter_outliers: bool = True,
    ):
        super().__init__(llm_provider, n_samples, temperature)
        self.use_weighted_voting = use_weighted_voting
        self.filter_outliers = filter_outliers

    def decode(
        self,
        prompt: str,
        n_samples: Optional[int] = None,
        **kwargs,
    ) -> SelfConsistencyResult:
        """Enhanced decode with weighted voting."""
        # Get base result
        result = super().decode(prompt, n_samples, **kwargs)

        if self.use_weighted_voting and len(result.all_chains) > 2:
            # Re-weight based on chain quality
            result = self._apply_weighted_voting(result)

        if self.filter_outliers and result.unique_answers > 2:
            # Filter outlier answers
            result = self._filter_outliers(result)

        return result

    def _apply_weighted_voting(
        self,
        result: SelfConsistencyResult,
    ) -> SelfConsistencyResult:
        """Apply weighted voting based on reasoning quality."""
        weights = {}

        for chain in result.all_chains:
            # Score chain quality (simple heuristic)
            quality = self._score_chain_quality(chain)
            answer = chain.answer

            if answer not in weights:
                weights[answer] = 0.0
            weights[answer] += quality

        if weights:
            # Find weighted majority
            weighted_answer = max(weights.items(), key=lambda x: x[1])[0]

            if weighted_answer != result.final_answer:
                logger.info(
                    f"Weighted voting changed answer: "
                    f"{result.final_answer} -> {weighted_answer}"
                )
                # Update result
                result.final_answer = weighted_answer
                result.best_chain = self._find_best_chain(
                    result.all_chains, weighted_answer
                ).chain if self._find_best_chain(result.all_chains, weighted_answer) else ""

        return result

    def _score_chain_quality(self, chain: ReasoningChain) -> float:
        """Score the quality of a reasoning chain."""
        text = chain.chain
        score = 1.0

        # Longer reasoning is often better
        if len(text) > 500:
            score += 0.2
        elif len(text) < 100:
            score -= 0.3

        # Look for structured reasoning
        if any(word in text.lower() for word in ['because', 'therefore', 'however', 'consider']):
            score += 0.2

        # Look for specific data points
        if re.search(r'\d+\.?\d*%', text):  # Percentages
            score += 0.1
        if re.search(r'\$[\d,]+', text):  # Dollar amounts
            score += 0.1

        # Penalize vague language
        vague_words = ['maybe', 'perhaps', 'might', 'possibly', 'uncertain']
        vague_count = sum(1 for w in vague_words if w in text.lower())
        score -= vague_count * 0.1

        return max(0.1, min(2.0, score))

    def _filter_outliers(
        self,
        result: SelfConsistencyResult,
    ) -> SelfConsistencyResult:
        """Filter outlier answers that appear only once."""
        # Only filter if we have enough samples
        if result.n_samples < 5:
            return result

        # Filter answers that appear only once (likely noise)
        filtered_answers = [
            a for a in result.all_answers
            if result.answer_distribution.get(a, 0) > 1
        ]

        if filtered_answers:
            # Recalculate with filtered answers
            counts = Counter(filtered_answers)
            majority_answer, count = counts.most_common(1)[0]

            result.final_answer = majority_answer
            result.confidence = count / len(filtered_answers)
            result.agreement_ratio = result.confidence

        return result


# Singleton instance
_self_consistency_instance: Optional[SelfConsistencyDecoder] = None


def get_self_consistency(
    n_samples: int = 5,
    temperature: float = 0.7,
    enhanced: bool = False,
) -> SelfConsistencyDecoder:
    """
    Get the singleton Self-Consistency decoder.

    Args:
        n_samples: Number of chains to sample
        temperature: Sampling temperature
        enhanced: Use enhanced version with weighted voting

    Returns:
        SelfConsistencyDecoder instance
    """
    global _self_consistency_instance
    if _self_consistency_instance is None:
        if enhanced:
            _self_consistency_instance = EnhancedSelfConsistency(
                n_samples=n_samples,
                temperature=temperature,
            )
        else:
            _self_consistency_instance = SelfConsistencyDecoder(
                n_samples=n_samples,
                temperature=temperature,
            )
    return _self_consistency_instance


def decode_with_consistency(
    prompt: str,
    n_samples: int = 5,
    context: Optional[Dict[str, Any]] = None,
) -> SelfConsistencyResult:
    """
    Convenience function to decode with self-consistency.

    Args:
        prompt: The question to answer
        n_samples: Number of reasoning chains to sample
        context: Additional context

    Returns:
        SelfConsistencyResult with majority answer
    """
    decoder = get_self_consistency(n_samples=n_samples)
    return decoder.decode(prompt, context=context)
