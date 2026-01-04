"""
Strategy Spec DSL
=================

Declarative strategy specification format.

A StrategySpec defines:
- Entry rules (conditions, indicators)
- Exit rules (stops, targets, time)
- Position sizing rules
- Risk limits
- Data requirements

StrategySpecs are:
- JSON serializable
- Version controlled
- Validated against schema
- Executable via backtest engine
"""

from .spec import StrategySpec, load_spec, save_spec, validate_spec

__all__ = [
    "StrategySpec",
    "load_spec",
    "save_spec",
    "validate_spec",
]
