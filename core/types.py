"""
Unified Type Exports for Kobe Trading System.

This module provides a single import location for all core types used
throughout the trading system. Instead of importing from multiple modules,
you can import commonly-used types from here.

Blueprint Alignment:
    Implements Section 2.1 requirement for unified type exports, enabling:
    - Single import location for common types
    - Consistent type usage across modules
    - Easier IDE autocomplete and type checking

Usage:
    from core.types import (
        # Execution types
        Order, OrderResult, Position, Quote, Account,
        OrderSide, OrderType, TimeInForce, BrokerOrderStatus,
        # OMS types
        OrderRecord, OrderStatus,
        # Decision types
        DecisionPacket,
        # Exception types
        QuantSystemError, SafetyError, KillSwitchActiveError,
        # Kill switch types
        KillSwitchTrigger,
        # Feature types
        FeatureMetadata, FeatureCategory,
    )

Note:
    This module only re-exports types. It does not move or duplicate
    the original definitions. All types remain in their original modules.
"""

from __future__ import annotations

# =============================================================================
# EXECUTION TYPES (from execution/broker_base.py)
# =============================================================================

from execution.broker_base import (
    # Enums
    BrokerType,
    OrderSide,
    OrderType,
    TimeInForce,
    BrokerOrderStatus,
    # Dataclasses
    Quote,
    Position,
    Account,
    Order,
    OrderResult,
)

# =============================================================================
# OMS TYPES (from oms/order_state.py)
# =============================================================================

from oms.order_state import (
    OrderStatus,
    OrderRecord,
)

# =============================================================================
# DECISION TYPES (from core/decision_packet.py)
# =============================================================================

from core.decision_packet import DecisionPacket

# =============================================================================
# EXCEPTION TYPES (from core/exceptions.py)
# =============================================================================

from core.exceptions import (
    # Base
    QuantSystemError,
    # Safety
    SafetyError,
    KillSwitchActiveError,
    SafetyViolationError,
    LiveTradingBlockedError,
    BypassAttemptError,
    KillZoneViolationError,
    # Execution
    ExecutionError,
    PolicyGateError,
    ComplianceError,
    PortfolioRiskError,
    CircuitBreakerError,
    InvalidTransitionError,
    LiquidityError,
    SlippageError,
    # Data
    DataError,
    DataFetchError,
    DataValidationError,
    FakeDataError,
    LookaheadBiasError,
    SurvivorshipBiasError,
    # Configuration
    ConfigurationError,
    SettingsValidationError,
    MissingConfigError,
    FrozenParamsError,
    # System
    UnsafePathError,
    LockError,
    ReconciliationError,
    StateCorruptionError,
    # Research
    ResearchError,
    ApprovalGateError,
    ExperimentError,
    ReproducibilityError,
    # Helpers
    is_safety_critical,
    get_error_code,
)

# =============================================================================
# KILL SWITCH TYPES (from core/kill_switch.py)
# =============================================================================

from core.kill_switch import KillSwitchTrigger

# =============================================================================
# FEATURE TYPES (from features/registry.py)
# =============================================================================

from features.registry import (
    FeatureMetadata,
    FeatureCategory,
)

# =============================================================================
# EVIDENCE TYPES (from research/evidence.py)
# =============================================================================

from research.evidence import (
    EvidencePack,
    EvidencePackBuilder,
)

# =============================================================================
# DATA PROVIDER TYPES (from data/providers/base.py)
# =============================================================================

from data.providers.base import (
    AssetClass,
    DataFrequency,
    ProviderCapabilities,
)

# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # Execution enums
    "BrokerType",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "BrokerOrderStatus",
    # Execution dataclasses
    "Quote",
    "Position",
    "Account",
    "Order",
    "OrderResult",
    # OMS
    "OrderStatus",
    "OrderRecord",
    # Decision
    "DecisionPacket",
    # Exceptions - Base
    "QuantSystemError",
    # Exceptions - Safety
    "SafetyError",
    "KillSwitchActiveError",
    "SafetyViolationError",
    "LiveTradingBlockedError",
    "BypassAttemptError",
    "KillZoneViolationError",
    # Exceptions - Execution
    "ExecutionError",
    "PolicyGateError",
    "ComplianceError",
    "PortfolioRiskError",
    "CircuitBreakerError",
    "InvalidTransitionError",
    "LiquidityError",
    "SlippageError",
    # Exceptions - Data
    "DataError",
    "DataFetchError",
    "DataValidationError",
    "FakeDataError",
    "LookaheadBiasError",
    "SurvivorshipBiasError",
    # Exceptions - Config
    "ConfigurationError",
    "SettingsValidationError",
    "MissingConfigError",
    "FrozenParamsError",
    # Exceptions - System
    "UnsafePathError",
    "LockError",
    "ReconciliationError",
    "StateCorruptionError",
    # Exceptions - Research
    "ResearchError",
    "ApprovalGateError",
    "ExperimentError",
    "ReproducibilityError",
    # Exception helpers
    "is_safety_critical",
    "get_error_code",
    # Kill switch
    "KillSwitchTrigger",
    # Features
    "FeatureMetadata",
    "FeatureCategory",
    # Evidence
    "EvidencePack",
    "EvidencePackBuilder",
    # Data providers
    "AssetClass",
    "DataFrequency",
    "ProviderCapabilities",
]
