"""
Explainability Module for Trading System
=========================================

Provides trade explanation, narrative generation, and decision
transparency for the trading system. Supports both rule-based
explanations and LLM-enhanced narratives.

Components:
- TradeExplainer: Generates explanations for individual trades
- NarrativeGenerator: Creates human-readable reports
- DecisionTracker: Records decision rationale for audit

Usage:
    from explainability import TradeExplainer, explain_trade

    explainer = TradeExplainer()
    explanation = explainer.explain(trade_signal)
"""

from .trade_explainer import (
    TradeExplainer,
    TradeExplanation,
    ExplanationFactor,
    explain_trade,
    get_explainer,
)

from .narrative_generator import (
    NarrativeGenerator,
    Narrative,
    NarrativeStyle,
    generate_narrative,
    generate_daily_summary,
)

from .decision_tracker import (
    DecisionTracker,
    DecisionRecord,
    DecisionContext,
    record_decision,
    get_decision_history,
)

__all__ = [
    # Trade Explainer
    'TradeExplainer',
    'TradeExplanation',
    'ExplanationFactor',
    'explain_trade',
    'get_explainer',
    # Narrative Generator
    'NarrativeGenerator',
    'Narrative',
    'NarrativeStyle',
    'generate_narrative',
    'generate_daily_summary',
    # Decision Tracker
    'DecisionTracker',
    'DecisionRecord',
    'DecisionContext',
    'record_decision',
    'get_decision_history',
]
