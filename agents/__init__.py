"""
Agent System - Multi-Agent Architecture
========================================

ReAct-pattern agents for autonomous research and validation:
- ScoutAgent: External source discovery
- AuditorAgent: Integrity/bias detection
- RiskAgent: Quant gate validation
- ReporterAgent: Report generation
- AgentOrchestrator: Multi-agent coordination

CRITICAL: Agents are advisory-only.
- NO live trading
- NO auto-merge
- Human approval required for all promotions
"""

from .base_agent import (
    BaseAgent,
    AgentThought,
    AgentAction,
    AgentResult,
    AgentConfig,
    ToolResult,
    AgentStatus,
    PAPER_ONLY,
)

from .agent_tools import (
    get_file_tools,
    get_data_tools,
    get_backtest_tools,
    get_registry_tools,
    get_all_tools,
    read_file,
    list_files,
    write_draft,
    get_universe_symbols,
    get_cached_data,
)

from .scout_agent import ScoutAgent, IdeaCard
from .auditor_agent import AuditorAgent, AuditFinding, BiasType, Severity
from .risk_agent import RiskAgent, GateResult, GateStatus
from .reporter_agent import ReporterAgent
from .orchestrator import (
    AgentOrchestrator,
    get_orchestrator,
    run_hourly_cycle,
    run_nightly_cycle,
    PipelineStage,
)

__all__ = [
    # Base agent
    "BaseAgent",
    "AgentThought",
    "AgentAction",
    "AgentResult",
    "AgentConfig",
    "ToolResult",
    "AgentStatus",
    "PAPER_ONLY",
    # Specialized agents
    "ScoutAgent",
    "IdeaCard",
    "AuditorAgent",
    "AuditFinding",
    "BiasType",
    "Severity",
    "RiskAgent",
    "GateResult",
    "GateStatus",
    "ReporterAgent",
    # Orchestrator
    "AgentOrchestrator",
    "get_orchestrator",
    "run_hourly_cycle",
    "run_nightly_cycle",
    "PipelineStage",
    # Tool getters
    "get_file_tools",
    "get_data_tools",
    "get_backtest_tools",
    "get_registry_tools",
    "get_all_tools",
    # Individual tools
    "read_file",
    "list_files",
    "write_draft",
    "get_universe_symbols",
    "get_cached_data",
]
