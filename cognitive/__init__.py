"""
Cognitive Architecture Package
===============================

Brain-inspired AI system for intelligent trading decisions.

Based on:
- SOFAI dual-process architecture (System 1/System 2)
- Global Workspace Theory (shared blackboard)
- Metacognitive Control (executive function)
- Episodic & Semantic Memory (learning from experience)
- Recursive Self-Improvement (collapse-regeneration cycles)

Components:
- GlobalWorkspace: Shared bus where all modules publish/subscribe
- MetacognitiveGovernor: Executive function deciding when to think deeper
- SelfModel: Robot's self-awareness of capabilities, limits, recent errors
- EpisodicMemory: Trade episodes with full context for learning
- SemanticMemory: Distilled rules of thumb from experience
- ReflectionEngine: Self-critique that changes behavior
- KnowledgeBoundary: Uncertainty detection and stand-down policies
- CuriosityEngine: Pattern discovery and hypothesis testing

Usage:
    from cognitive import CognitiveBrain

    brain = CognitiveBrain()
    decision = brain.deliberate(signal, context)

    if decision.should_act:
        execute(decision.action)

    # After trade completes:
    brain.record_episode(context, reasoning, action, outcome)
    brain.reflect()  # Self-critique and learning
"""

from cognitive.global_workspace import GlobalWorkspace, get_workspace
from cognitive.metacognitive_governor import MetacognitiveGovernor
from cognitive.self_model import SelfModel, get_self_model
from cognitive.episodic_memory import EpisodicMemory, get_episodic_memory
from cognitive.semantic_memory import SemanticMemory, get_semantic_memory
from cognitive.reflection_engine import ReflectionEngine
from cognitive.knowledge_boundary import KnowledgeBoundary
from cognitive.curiosity_engine import CuriosityEngine
from cognitive.cognitive_brain import CognitiveBrain, get_cognitive_brain

__all__ = [
    'GlobalWorkspace',
    'MetacognitiveGovernor',
    'SelfModel',
    'EpisodicMemory',
    'SemanticMemory',
    'ReflectionEngine',
    'KnowledgeBoundary',
    'CuriosityEngine',
    'CognitiveBrain',
    'get_workspace',
    'get_self_model',
    'get_episodic_memory',
    'get_semantic_memory',
    'get_cognitive_brain',
]
