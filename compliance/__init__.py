"""
Compliance Engine for Trading Systems
======================================

Enforces trading rules, manages prohibited symbols,
and maintains audit trails for regulatory compliance.

Components:
- RulesEngine: Enforce trading rules
- ProhibitedList: Manage restricted symbols
- AuditTrail: Compliance audit logging

SECURITY FIX (2026-01-04): Wired compliance engine into exports.
Previously all imports were commented out, leaving compliance unenforced.
"""

from .prohibited_list import (
    ProhibitedReason,
    is_prohibited,
    prohibited_reasons,
)

from .audit_trail import (
    write_event as log_compliance_event,
)

from .rules_engine import (
    RuleConfig,
    evaluate as evaluate_trade_rules,
)

__all__ = [
    # Prohibited list
    'ProhibitedReason',
    'is_prohibited',
    'prohibited_reasons',
    # Audit trail
    'log_compliance_event',
    # Rules engine
    'RuleConfig',
    'evaluate_trade_rules',
]
