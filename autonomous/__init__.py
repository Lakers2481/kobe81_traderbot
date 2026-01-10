"""
Autonomous 24/7 Brain for Kobe Trading System.

This module provides always-on, self-improving capabilities:
- Time/day/season awareness
- Automatic task scheduling
- Self-improvement through research and backtesting
- Continuous learning and optimization

Enhanced (2026-01-07):
- Alpha mining integration (VectorBT, Alphalens)
- LangGraph formal state machine
- RAG evaluator for LLM quality
"""

from .awareness import TimeAwareness, MarketCalendarAwareness, SeasonalAwareness
from .scheduler import AutonomousScheduler, Task, TaskPriority
from .brain import AutonomousBrain, Discovery
from .handlers import register_all_handlers, HANDLERS
from .research import ResearchEngine
from .learning import LearningEngine

# Enhanced components (2026-01-07)
try:
    from .enhanced_research import EnhancedResearchEngine
    HAS_ENHANCED_RESEARCH = True
except ImportError:
    HAS_ENHANCED_RESEARCH = False
    EnhancedResearchEngine = None

try:
    from .enhanced_brain import EnhancedAutonomousBrain
    HAS_ENHANCED_BRAIN = True
except ImportError:
    HAS_ENHANCED_BRAIN = False
    EnhancedAutonomousBrain = None

__all__ = [
    # Base components
    'TimeAwareness',
    'MarketCalendarAwareness',
    'SeasonalAwareness',
    'AutonomousScheduler',
    'Task',
    'TaskPriority',
    'AutonomousBrain',
    'Discovery',
    'ResearchEngine',
    'LearningEngine',
    'register_all_handlers',
    'HANDLERS',

    # Enhanced components
    'EnhancedResearchEngine',
    'EnhancedAutonomousBrain',
    'HAS_ENHANCED_RESEARCH',
    'HAS_ENHANCED_BRAIN',
]
