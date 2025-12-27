"""
Compliance Engine for Trading Systems
======================================

Enforces trading rules, manages prohibited symbols,
and maintains audit trails for regulatory compliance.

Components:
- RulesEngine: Enforce trading rules
- ProhibitedList: Manage restricted symbols
- AuditTrail: Compliance audit logging
"""

from .rules_engine import (
    RulesEngine,
    TradingRule,
    RuleViolation,
    RuleCategory,
    check_rules,
    get_violations,
)

from .prohibited_list import (
    ProhibitedList,
    ProhibitionReason,
    check_symbol,
    add_prohibition,
    is_prohibited,
)

from .audit_trail import (
    AuditTrail,
    AuditEntry,
    AuditAction,
    log_audit,
    get_audit_history,
)

__all__ = [
    'RulesEngine',
    'TradingRule',
    'RuleViolation',
    'RuleCategory',
    'check_rules',
    'get_violations',
    'ProhibitedList',
    'ProhibitionReason',
    'check_symbol',
    'add_prohibition',
    'is_prohibited',
    'AuditTrail',
    'AuditEntry',
    'AuditAction',
    'log_audit',
    'get_audit_history',
]
