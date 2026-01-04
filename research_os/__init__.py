"""
KOBE Research Operating System
==============================

DISCOVER -> RESEARCH -> ENGINEER

A thin orchestration layer that integrates existing Kobe components
into a unified research workflow with human-gated engineering changes.

Design Philosophy:
    "Don't reinvent. Integrate."

Components:
    - Discovery Lane: CuriosityEngine + ResearchEngine + Scrapers
    - Research Lane: ExperimentRegistry + IntegrityGuardian
    - Engineering Lane: ApprovalGate + FrozenParams

Critical Constraints:
    - NEVER auto-merge
    - NEVER enable live trading without human approval
    - APPROVE_LIVE_ACTION = False by default

Usage:
    from research_os import ResearchOSOrchestrator

    os = ResearchOSOrchestrator()

    # Run discovery cycle
    cards = os.run_discovery_cycle()

    # Run research on hypothesis
    os.run_research_cycle(proposal)

    # Propose engineering change (requires human approval)
    os.propose_engineering_change(validated_card)

    # Human must approve separately
    # python scripts/research_os_cli.py approve --id <id> --approver "Name"
"""
from __future__ import annotations

from .knowledge_card import KnowledgeCard, DiscoveryType, CardStatus
from .proposal import ResearchProposal, ProposalStatus, ChangeType
from .approval_gate import ApprovalGate, APPROVE_LIVE_ACTION
from .orchestrator import ResearchOSOrchestrator

__all__ = [
    "ResearchOSOrchestrator",
    "KnowledgeCard",
    "DiscoveryType",
    "CardStatus",
    "ResearchProposal",
    "ProposalStatus",
    "ChangeType",
    "ApprovalGate",
    "APPROVE_LIVE_ACTION",
]

__version__ = "1.0.0"
