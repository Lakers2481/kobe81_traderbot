from __future__ import annotations

"""
Adjudicator
===========

This module acts as a final judgment layer, providing a verdict on a given
trading signal. It is designed to be a pluggable component within the cognitive
architecture, aggregating various checks to produce a final approval or rejection.

The primary function, `adjudicate`, can be extended to incorporate multiple
sources of judgment, such as different heuristic models or even calls to
external Large Language Models (LLMs) for a qualitative assessment.

For now, it defaults to a simple, configurable heuristic model.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Verdict:
    """
    A structured object representing the final verdict of the adjudication process.

    Attributes:
        approved: A boolean indicating whether the signal is approved for action.
        score: A numerical score (0.0 to 1.0) representing the confidence or
               quality of the signal, where 1.0 is highest quality.
        reasons: A list of strings detailing the reasons (especially negative ones)
                 that contributed to the final score.
    """
    approved: bool
    score: float
    reasons: list[str]


def heuristic(signal: Dict[str, Any]) -> Verdict:
    """
    A simple heuristic-based adjudication function.

    This function applies a set of common-sense trading rules to a signal to
    assess its quality. Each rule that flags a concern reduces the signal's
    final score.

    Args:
        signal: The trading signal dictionary to be evaluated.

    Returns:
        A Verdict object with the outcome of the heuristic checks.
    """
    reasons: list[str] = []
    score = 1.0  # Start with a perfect score

    # PENALTY 1: Avoid very low-priced stocks ("penny stocks").
    # These are often too volatile, illiquid, or subject to manipulation.
    if float(signal.get('entry_price', 0.0)) < 5.0:
        reasons.append('low_price')
        score -= 0.3
        
    # PENALTY 2: Penalize signals with a very wide stop-loss.
    # A stop-loss that is too far from the entry price (>5% of entry) can
    # indicate a poor risk/reward ratio for the trade.
    try:
        entry_price = float(signal.get('entry_price', 0.0))
        stop_loss = float(signal.get('stop_loss', 0.0))
        # Ensure prices are valid before calculation
        if entry_price > 0 and stop_loss > 0:
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share > entry_price * 0.05:
                reasons.append('wide_stop')
                score -= 0.2
    except (ValueError, TypeError):
        # If price data is missing or invalid, we cannot perform this check.
        pass

    # The signal is approved only if the final score meets a minimum threshold.
    is_approved = score >= 0.6
    
    # Clamp the score to ensure it's always within the [0.0, 1.0] range.
    final_score = max(0.0, min(score, 1.0))
    
    return Verdict(approved=is_approved, score=final_score, reasons=reasons)


def adjudicate(signal: Dict[str, Any]) -> Verdict:
    """
    The main entry point for the adjudication process.

    This function coordinates the adjudication of a signal. It can be extended
    to become a multi-agent system, for example, by collecting verdicts from
    several different sources (heuristics, LLMs, other models) and aggregating
    them into a final verdict.

    Args:
        signal: The trading signal to be adjudicated.

    Returns:
        The final Verdict on the signal.
    """
    # For now, the implementation is simple and relies solely on the heuristic model.
    # This could be extended in the future, e.g.:
    #   llm_verdict = llm_agent.adjudicate(signal)
    #   final_score = (heuristic_verdict.score + llm_verdict.score) / 2
    return heuristic(signal)

