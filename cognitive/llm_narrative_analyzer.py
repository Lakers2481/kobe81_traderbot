"""
LLM Narrative Analyzer - AI-Powered Meta-Reflection
=====================================================

This module provides a powerful enhancement to the `ReflectionEngine` by using
a Large Language Model (LLM) to perform a deeper, more abstract analysis of
the trading bot's performance. It takes the structured output from the
`ReflectionEngine` and asks an LLM, like Claude, to find meta-patterns,
critique reasoning, and suggest novel hypotheses.

Core Function:
- Takes a `Reflection` object as input.
- Constructs a detailed, sophisticated prompt for an LLM.
- Queries the LLM API (e.g., Anthropic's Claude).
- Parses the natural language response to extract actionable insights.
- Extracts structured hypotheses that can be tested by the CuriosityEngine.

This component allows the trading bot to move beyond simple statistical
learning and engage in a form of conceptual, abstract reasoning about its
own behavior.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# The dotenv library is already a project dependency, used for loading .env files.
from dotenv import load_dotenv

# The 'anthropic' library was added to requirements.txt and installed.
import anthropic

# Use TYPE_CHECKING to avoid circular import while still enabling type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cognitive.reflection_engine import Reflection

logger = logging.getLogger(__name__)


@dataclass
class LLMHypothesis:
    """
    Structured hypothesis extracted from LLM analysis.

    These hypotheses are passed to the CuriosityEngine for statistical testing.
    """
    description: str  # Human-readable description of the hypothesis
    condition: str    # Market condition when hypothesis applies (e.g., "regime = BEAR")
    prediction: str   # Expected outcome (e.g., "win_rate > 0.6")
    rationale: str    # Why the LLM suggested this hypothesis
    confidence: float = 0.5  # Initial confidence (0-1)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'description': self.description,
            'condition': self.condition,
            'prediction': self.prediction,
            'rationale': self.rationale,
            'confidence': self.confidence,
        }


@dataclass
class StrategyIdea:
    """
    Structured strategy idea extracted from LLM analysis.

    These strategy ideas represent novel trading concepts proposed by the LLM
    that could be developed into full strategies by the system or human developers.
    The CuriosityEngine can convert these into testable hypotheses.
    """
    name: str  # Short strategy name (e.g., "VIX_Mean_Reversion")
    concept: str  # Core trading concept in one sentence
    market_context: str  # When this strategy applies (e.g., "high volatility regimes")
    entry_conditions: List[str]  # List of entry conditions
    exit_conditions: List[str]  # List of exit conditions (stops, targets, time-based)
    risk_management: str  # Position sizing and risk rules
    rationale: str  # Why the LLM thinks this strategy might work
    confidence: float = 0.5  # LLM's confidence in the idea (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'concept': self.concept,
            'market_context': self.market_context,
            'entry_conditions': self.entry_conditions,
            'exit_conditions': self.exit_conditions,
            'risk_management': self.risk_management,
            'rationale': self.rationale,
            'confidence': self.confidence,
        }


@dataclass
class LLMAnalysisResult:
    """
    Complete result from LLM analysis, containing all extracted insights.

    This dataclass bundles together all the structured outputs from
    the LLM's analysis of a reflection, including critique text,
    testable hypotheses, and novel strategy ideas.
    """
    critique: Optional[str]  # The raw LLM critique text
    hypotheses: List[LLMHypothesis] = field(default_factory=list)
    strategy_ideas: List[StrategyIdea] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'critique': self.critique,
            'hypotheses': [h.to_dict() for h in self.hypotheses],
            'strategy_ideas': [s.to_dict() for s in self.strategy_ideas],
        }


class LLMNarrativeAnalyzer:
    """
    A class that uses an LLM to analyze performance reflections.
    """

    def __init__(self):
        """
        Initializes the analyzer and the Anthropic client.
        It loads the ANTHROPIC_API_KEY from the .env file in the project root.
        """
        load_dotenv()
        self._api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            logger.warning("ANTHROPIC_API_KEY not found in .env file. LLM analysis will be disabled.")
            self._client = None
        else:
            try:
                self._client = anthropic.Client(api_key=self._api_key)
                logger.info("Anthropic client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                self._client = None
    
    def analyze_reflection(self, reflection: "Reflection") -> LLMAnalysisResult:
        """
        Analyzes a reflection object using the configured LLM.

        Args:
            reflection: The Reflection object from the ReflectionEngine.

        Returns:
            An LLMAnalysisResult containing critique, hypotheses, and strategy ideas.
        """
        if not self._client:
            logger.debug("LLM client not available. Skipping analysis.")
            return LLMAnalysisResult(critique=None, hypotheses=[], strategy_ideas=[])

        # Construct a detailed prompt for the LLM.
        prompt = self._build_prompt(reflection)

        try:
            logger.debug("Sending reflection analysis request to LLM...")
            response = self._client.messages.create(
                model="claude-3-haiku-20240307",  # Using Haiku for speed and cost-effectiveness
                max_tokens=1200,  # Increased to accommodate strategy ideas
                temperature=0.7,
                system="You are a senior quantitative trading analyst and AI psychologist. Your task is to critique the performance and reasoning of a trading robot, generate testable hypotheses, and suggest novel trading strategies.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            )

            # Extract the content from the response object.
            analysis_text = response.content[0].text
            logger.info("Received LLM analysis for reflection.")

            # Parse structured hypotheses from the response
            hypotheses = self._parse_hypotheses(analysis_text)
            if hypotheses:
                logger.info(f"Extracted {len(hypotheses)} hypotheses from LLM response.")

            # Parse strategy ideas from the response
            strategy_ideas = self._parse_strategy_ideas(analysis_text)
            if strategy_ideas:
                logger.info(f"Extracted {len(strategy_ideas)} strategy ideas from LLM response.")

            return LLMAnalysisResult(
                critique=analysis_text,
                hypotheses=hypotheses,
                strategy_ideas=strategy_ideas,
            )

        except Exception as e:
            logger.error(f"Error during LLM API call: {e}", exc_info=True)
            return LLMAnalysisResult(critique=None, hypotheses=[], strategy_ideas=[])

    def analyze_reflection_legacy(self, reflection: "Reflection") -> Tuple[Optional[str], List[LLMHypothesis]]:
        """
        Legacy method that returns only the critique text and hypotheses.

        Maintained for backward compatibility with existing code.
        """
        result = self.analyze_reflection(reflection)
        return result.critique, result.hypotheses

    def _should_request_strategy_ideas(self, reflection: "Reflection") -> bool:
        """
        Determines if the LLM should be asked to generate strategy ideas.

        Strategy ideas are requested when:
        - The reflection scope is 'daily' or 'weekly' (periodic reviews are good times for innovation)
        - There are multiple things that went wrong (indicating need for new approach)
        - The summary indicates poor performance or uncharted territory
        """
        # Request for periodic reflections (daily/weekly are strategic review moments)
        if reflection.scope in ('daily', 'weekly'):
            return True

        # Request if many things went wrong (need for new approach)
        if len(reflection.what_went_wrong) >= 2:
            return True

        # Request if summary mentions key phrases indicating need for new approaches
        summary_lower = reflection.summary.lower() if reflection.summary else ""
        trigger_phrases = [
            'repeated loss', 'underperform', 'new regime', 'unfamiliar',
            'novel', 'struggling', 'consistently fail', 'pattern of loss',
            'need different', 'rethink', 'adapt'
        ]
        if any(phrase in summary_lower for phrase in trigger_phrases):
            return True

        return False

    def _build_prompt(self, reflection: "Reflection") -> str:
        """
        Constructs a sophisticated, multi-part prompt to guide the LLM's analysis.

        The prompt requests structured hypotheses and conditionally requests
        strategy ideas based on the reflection context.
        """
        prompt = (
            "Analyze the following trading performance reflection. Provide a concise, high-level critique.\n"
            "Focus on identifying potential meta-patterns, flawed reasoning, or biases that a simple statistical analysis might miss.\n"
            "Conclude with 1-2 novel, testable hypotheses for the bot's 'CuriosityEngine'.\n\n"
            "--- REFLECTION DATA ---\n"
            f"Scope: {reflection.scope}\n"
            f"Summary: {reflection.summary}\n"
        )

        if reflection.what_went_well:
            prompt += "\nWhat Went Well:\n"
            for item in reflection.what_went_well:
                prompt += f"- {item}\n"

        if reflection.what_went_wrong:
            prompt += "\nWhat Went Wrong:\n"
            for item in reflection.what_went_wrong:
                prompt += f"- {item}\n"

        if reflection.lessons:
            prompt += "\nLessons Identified by the System:\n"
            for lesson in reflection.lessons:
                prompt += f"- {lesson}\n"

        prompt += (
            "\n--- YOUR ANALYSIS ---\n"
            "**Meta-Critique:** (Your high-level analysis of the bot's performance and self-assessment)\n\n"
            "**Suggested Hypotheses:** (Provide 1-2 hypotheses in the EXACT format below)\n"
            "Each hypothesis MUST follow this structure:\n"
            "HYPOTHESIS: [one-line description]\n"
            "CONDITION: [market condition, e.g., 'regime = BEAR AND vix > 25']\n"
            "PREDICTION: [expected outcome, e.g., 'win_rate > 0.6']\n"
            "RATIONALE: [why you think this might be true]\n\n"
            "Example:\n"
            "HYPOTHESIS: Mean reversion strategies may outperform during high VIX periods\n"
            "CONDITION: vix > 30 AND strategy = ibs_rsi\n"
            "PREDICTION: win_rate > 0.65\n"
            "RATIONALE: High volatility creates larger price swings that mean reversion can capture.\n"
        )

        # Conditionally request strategy ideas
        if self._should_request_strategy_ideas(reflection):
            prompt += (
                "\n**Strategy Ideas:** (If you identify a significant opportunity or recurring flaw, "
                "propose ONE novel strategy idea using the format below)\n"
                "--- STRATEGY IDEA ---\n"
                "NAME: [short strategy name, e.g., VIX_Spike_Reversal]\n"
                "CONCEPT: [core trading concept in one sentence]\n"
                "MARKET_CONTEXT: [when this strategy applies, e.g., 'high volatility regimes']\n"
                "ENTRY_CONDITIONS:\n"
                "- [condition 1]\n"
                "- [condition 2]\n"
                "EXIT_CONDITIONS:\n"
                "- [stop condition]\n"
                "- [target condition]\n"
                "- [time-based exit if applicable]\n"
                "RISK_MANAGEMENT: [position sizing and risk rules]\n"
                "RATIONALE: [why this strategy might work]\n"
                "--- END STRATEGY IDEA ---\n"
            )

        return prompt

    def _parse_hypotheses(self, llm_response: str) -> List[LLMHypothesis]:
        """
        Parses structured hypotheses from the LLM's response.

        Looks for patterns like:
        HYPOTHESIS: [description]
        CONDITION: [condition]
        PREDICTION: [prediction]
        RATIONALE: [rationale]
        """
        hypotheses = []

        # Regular expression to match hypothesis blocks
        # Each block starts with HYPOTHESIS: and may contain CONDITION, PREDICTION, RATIONALE
        hypothesis_pattern = re.compile(
            r'HYPOTHESIS:\s*(.+?)(?:\n|$)'
            r'(?:CONDITION:\s*(.+?)(?:\n|$))?'
            r'(?:PREDICTION:\s*(.+?)(?:\n|$))?'
            r'(?:RATIONALE:\s*(.+?)(?:\n|$|(?=HYPOTHESIS:)))?',
            re.IGNORECASE | re.DOTALL
        )

        matches = hypothesis_pattern.findall(llm_response)

        for match in matches:
            description = match[0].strip() if match[0] else ""
            condition = match[1].strip() if len(match) > 1 and match[1] else "unknown"
            prediction = match[2].strip() if len(match) > 2 and match[2] else "win_rate > 0.55"
            rationale = match[3].strip() if len(match) > 3 and match[3] else "LLM-generated hypothesis"

            if description:  # Only add if we have a description
                hypotheses.append(LLMHypothesis(
                    description=description,
                    condition=condition,
                    prediction=prediction,
                    rationale=rationale,
                    confidence=0.5,  # Default confidence for LLM-generated hypotheses
                ))

        # Fallback: Try to parse simpler "Hypothesis: ..." format for backward compatibility
        if not hypotheses:
            simple_pattern = re.compile(r'Hypothesis:\s*(.+?)(?:\n|$)', re.IGNORECASE)
            simple_matches = simple_pattern.findall(llm_response)
            for desc in simple_matches:
                desc = desc.strip()
                if desc:
                    hypotheses.append(LLMHypothesis(
                        description=desc,
                        condition="unknown",
                        prediction="win_rate > 0.55",
                        rationale="Extracted from legacy LLM format",
                        confidence=0.4,  # Lower confidence for unstructured hypotheses
                    ))

        return hypotheses

    def _parse_strategy_ideas(self, llm_response: str) -> List[StrategyIdea]:
        """
        Parses structured strategy ideas from the LLM's response.

        Looks for sections delimited by "--- STRATEGY IDEA ---" and "--- END STRATEGY IDEA ---"
        and extracts the structured fields within.
        """
        strategy_ideas = []

        # Pattern to extract strategy idea blocks
        block_pattern = re.compile(
            r'---\s*STRATEGY\s*IDEA\s*---(.+?)---\s*END\s*STRATEGY\s*IDEA\s*---',
            re.IGNORECASE | re.DOTALL
        )

        blocks = block_pattern.findall(llm_response)

        for block in blocks:
            try:
                idea = self._parse_single_strategy_block(block)
                if idea:
                    strategy_ideas.append(idea)
            except Exception as e:
                logger.warning(f"Failed to parse strategy idea block: {e}")
                continue

        return strategy_ideas

    def _parse_single_strategy_block(self, block: str) -> Optional[StrategyIdea]:
        """
        Parses a single strategy idea block into a StrategyIdea object.
        """
        def extract_field(pattern: str, text: str, default: str = "") -> str:
            """Helper to extract a single field value."""
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else default

        def extract_list_field(pattern: str, text: str) -> List[str]:
            """Helper to extract a list field (bulleted items)."""
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if not match:
                return []
            content = match.group(1)
            # Parse bulleted items (lines starting with -)
            items = re.findall(r'-\s*(.+?)(?:\n|$)', content)
            return [item.strip() for item in items if item.strip()]

        # Extract each field using patterns that stop at the next field
        name = extract_field(r'NAME:\s*(.+?)(?:\n|CONCEPT:|$)', block)
        concept = extract_field(r'CONCEPT:\s*(.+?)(?:\n|MARKET_CONTEXT:|$)', block)
        market_context = extract_field(r'MARKET_CONTEXT:\s*(.+?)(?:\n|ENTRY_CONDITIONS:|$)', block)
        entry_conditions = extract_list_field(r'ENTRY_CONDITIONS:\s*(.+?)(?:EXIT_CONDITIONS:|RISK_MANAGEMENT:|$)', block)
        exit_conditions = extract_list_field(r'EXIT_CONDITIONS:\s*(.+?)(?:RISK_MANAGEMENT:|RATIONALE:|$)', block)
        risk_management = extract_field(r'RISK_MANAGEMENT:\s*(.+?)(?:\n|RATIONALE:|$)', block)
        rationale = extract_field(r'RATIONALE:\s*(.+?)$', block)

        # Validate required fields
        if not name or not concept:
            logger.debug("Strategy idea missing required fields (name or concept)")
            return None

        return StrategyIdea(
            name=name,
            concept=concept,
            market_context=market_context or "general",
            entry_conditions=entry_conditions or ["undefined"],
            exit_conditions=exit_conditions or ["undefined"],
            risk_management=risk_management or "standard position sizing",
            rationale=rationale or "LLM-generated strategy idea",
            confidence=0.5,  # Default confidence
        )


# Singleton instance for the analyzer
_llm_analyzer_instance: Optional[LLMNarrativeAnalyzer] = None

def get_llm_analyzer() -> LLMNarrativeAnalyzer:
    """Factory function to get the singleton instance of the LLMNarrativeAnalyzer."""
    global _llm_analyzer_instance
    if _llm_analyzer_instance is None:
        _llm_analyzer_instance = LLMNarrativeAnalyzer()
    return _llm_analyzer_instance
