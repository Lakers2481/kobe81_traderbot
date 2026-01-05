"""
Guardian System - Autonomous Trading Oversight

The Guardian is the central orchestrator for 24/7 autonomous trading.
It monitors all systems, makes decisions, and ensures safety.

Components:
- SystemMonitor: Real-time health monitoring of all components
- DecisionEngine: Autonomous decision making with escalation
- AlertManager: Smart alerting with priority and deduplication
- DailyDigest: Comprehensive daily/weekly reports
- EmergencyProtocol: Automatic response to critical events

Solo Trader Features:
- Fully autonomous operation (user doesn't need to babysit)
- Smart escalation (only alert for truly important events)
- Self-healing capabilities
- Comprehensive audit trail

Author: Kobe Trading System
Created: 2026-01-04
"""

# CRITICAL: Load .env file FIRST before any module imports
# This ensures all API keys are available to all Guardian components
from dotenv import load_dotenv
load_dotenv()

from .system_monitor import (
    SystemMonitor,
    SystemHealth,
    ComponentStatus,
    get_system_monitor,
)
from .decision_engine import (
    DecisionEngine,
    Decision,
    DecisionType,
    get_decision_engine,
)
from .alert_manager import (
    AlertManager,
    Alert,
    AlertPriority,
    get_alert_manager,
)
from .daily_digest import (
    DailyDigest,
    DigestReport,
    generate_daily_digest,
)
from .emergency_protocol import (
    EmergencyProtocol,
    EmergencyLevel,
    EmergencyAction,
    get_emergency_protocol,
)
from .guardian import (
    Guardian,
    get_guardian,
)
from .self_learner import (
    SelfLearner,
    get_self_learner,
    ChangeType,
    OutcomeType,
)

__all__ = [
    # System Monitor
    "SystemMonitor",
    "SystemHealth",
    "ComponentStatus",
    "get_system_monitor",
    # Decision Engine
    "DecisionEngine",
    "Decision",
    "DecisionType",
    "get_decision_engine",
    # Alert Manager
    "AlertManager",
    "Alert",
    "AlertPriority",
    "get_alert_manager",
    # Daily Digest
    "DailyDigest",
    "DigestReport",
    "generate_daily_digest",
    # Emergency Protocol
    "EmergencyProtocol",
    "EmergencyLevel",
    "EmergencyAction",
    "get_emergency_protocol",
    # Guardian
    "Guardian",
    "get_guardian",
    # Self-Learner
    "SelfLearner",
    "get_self_learner",
    "ChangeType",
    "OutcomeType",
]
