"""
Policy Generator Alias Module
==============================

This module provides a backwards-compatible alias for the DynamicPolicyGenerator.

The actual implementation is in `cognitive/dynamic_policy_generator.py`.
This alias exists for naming consistency and to match documentation references.

Usage:
    # Both of these work identically:
    from cognitive.policy_generator import get_policy_generator
    from cognitive.dynamic_policy_generator import get_policy_generator
"""

# Re-export everything from dynamic_policy_generator for backwards compatibility
from cognitive.dynamic_policy_generator import (
    PolicyType,
    TradingPolicy,
    DynamicPolicyGenerator,
    get_policy_generator,
)

__all__ = [
    'PolicyType',
    'TradingPolicy',
    'DynamicPolicyGenerator',
    'get_policy_generator',
]
