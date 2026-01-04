"""
Integration Module - Kobe's Nervous System
==========================================

This module provides the central integration hub that wires together all of Kobe's
cognitive and learning components. It's the "nervous system" that connects:

- Trade outcomes → Episodic Memory
- Trade outcomes → Online Learning Manager
- Significant trades → Reflection Engine
- Insights → Semantic Memory rules

Without this integration, the components work in isolation. With it, Kobe
becomes a truly self-learning, continuously adapting trading system.

Author: Kobe Trading System
Created: 2026-01-04
"""

from integration.learning_hub import LearningHub, get_learning_hub

__all__ = ['LearningHub', 'get_learning_hub']
