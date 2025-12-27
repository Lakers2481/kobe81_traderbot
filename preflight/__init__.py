"""
Kobe Trading System - Preflight & Evidence Gates
=================================================

Pre-trade validation and promotion gates to ensure:
- Strategies meet rigorous evidence standards before trading
- Data quality is validated before backtesting
- KnowledgeBoundary integration for uncertainty/stand-down decisions
"""

from .evidence_gate import (
    EvidenceGate,
    EvidenceRequirements,
    EvidenceReport,
    EvidenceLevel,
    check_promotion_gate,
)

from .data_quality import (
    DataQualityGate,
    DataQualityRequirements,
    DataQualityReport,
    DataQualityLevel,
    DataIssue,
    SymbolCoverage,
    validate_data_quality,
)

__all__ = [
    # Evidence Gate
    'EvidenceGate',
    'EvidenceRequirements',
    'EvidenceReport',
    'EvidenceLevel',
    'check_promotion_gate',
    # Data Quality Gate
    'DataQualityGate',
    'DataQualityRequirements',
    'DataQualityReport',
    'DataQualityLevel',
    'DataIssue',
    'SymbolCoverage',
    'validate_data_quality',
]
