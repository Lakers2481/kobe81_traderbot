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

This component allows the trading bot to move beyond simple statistical
learning and engage in a form of conceptual, abstract reasoning about its
own behavior.
"""

import os
import logging
from typing import Any, Optional

# The dotenv library is already a project dependency, used for loading .env files.
from dotenv import load_dotenv

# The 'anthropic' library was added to requirements.txt and installed.
import anthropic

# Use TYPE_CHECKING to avoid circular import while still enabling type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cognitive.reflection_engine import Reflection

logger = logging.getLogger(__name__)


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
    
    def analyze_reflection(self, reflection: "Reflection") -> Optional[str]:
        """
        Analyzes a reflection object using the configured LLM.

        Args:
            reflection: The Reflection object from the ReflectionEngine.

        Returns:
            A string containing the LLM's analysis, or None if the client
            is not available or an error occurs.
        """
        if not self._client:
            logger.debug("LLM client not available. Skipping analysis.")
            return None

        # Construct a detailed prompt for the LLM.
        prompt = self._build_prompt(reflection)
        
        try:
            logger.debug("Sending reflection analysis request to LLM...")
            response = self._client.messages.create(
                model="claude-3-haiku-20240307",  # Using Haiku for speed and cost-effectiveness
                max_tokens=500,
                temperature=0.7,
                system="You are a senior quantitative trading analyst and AI psychologist. Your task is to critique the performance and reasoning of a trading robot.",
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
            return analysis_text

        except Exception as e:
            logger.error(f"Error during LLM API call: {e}", exc_info=True)
            return None

    def _build_prompt(self, reflection: "Reflection") -> str:
        """
        Constructs a sophisticated, multi-part prompt to guide the LLM's analysis.
        """
        prompt = (
            "Analyze the following trading performance reflection. Provide a concise, high-level critique.\n"
            "Focus on identifying potential meta-patterns, flawed reasoning, or biases that a simple statistical analysis might miss.\n"
            "Conclude with one or two novel, testable hypotheses for the bot's 'CuriosityEngine'.\n\n"
            "--- REFLECTION DATA ---"
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
            "\n--- YOUR ANALYSIS ---" 
            "\n**Meta-Critique:** (Your high-level analysis of the bot's performance and self-assessment)\n\n"
            "**Suggested Hypotheses:** (Provide 1-2 new, creative hypotheses in the format 'Hypothesis: [description]')\n"
        )
        return prompt

# Singleton instance for the analyzer
_llm_analyzer_instance: Optional[LLMNarrativeAnalyzer] = None

def get_llm_analyzer() -> LLMNarrativeAnalyzer:
    """Factory function to get the singleton instance of the LLMNarrativeAnalyzer."""
    global _llm_analyzer_instance
    if _llm_analyzer_instance is None:
        _llm_analyzer_instance = LLMNarrativeAnalyzer()
    return _llm_analyzer_instance
