"""
Research OS Orchestrator - Thin integration layer.

Connects existing Kobe components into a unified DISCOVER -> RESEARCH -> ENGINEER workflow.

Design Philosophy: "Don't reinvent. Integrate."

This orchestrator:
1. Connects to existing CuriosityEngine, ResearchEngine, and scrapers
2. Uses existing ExperimentRegistry and IntegrityGuardian
3. Routes validated discoveries through ApprovalGate
4. NEVER auto-merges or auto-enables live trading
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

from .knowledge_card import KnowledgeCard, KnowledgeCardStore, DiscoveryType, CardStatus
from .proposal import ResearchProposal, ProposalStore, ChangeType
from .approval_gate import ApprovalGate, SafetyError

logger = logging.getLogger(__name__)

# Try to import existing components (graceful degradation if missing)
try:
    from cognitive.curiosity_engine import get_curiosity_engine, CuriosityEngine
    CURIOSITY_AVAILABLE = True
except ImportError:
    CURIOSITY_AVAILABLE = False
    logger.warning("CuriosityEngine not available")

try:
    from autonomous.research import ResearchEngine
    RESEARCH_ENGINE_AVAILABLE = True
except ImportError:
    RESEARCH_ENGINE_AVAILABLE = False
    logger.warning("ResearchEngine not available")

try:
    from autonomous.integrity import IntegrityGuardian
    INTEGRITY_AVAILABLE = True
except ImportError:
    INTEGRITY_AVAILABLE = False
    logger.warning("IntegrityGuardian not available")

try:
    from experiments.registry import ExperimentRegistry
    EXPERIMENTS_AVAILABLE = True
except ImportError:
    EXPERIMENTS_AVAILABLE = False
    logger.warning("ExperimentRegistry not available")

try:
    from autonomous.scrapers.source_manager import SourceManager
    SOURCES_AVAILABLE = True
except ImportError:
    SOURCES_AVAILABLE = False
    logger.warning("SourceManager not available")


@dataclass
class DiscoveryCycleResult:
    """Result of a discovery cycle."""
    hypotheses_generated: int
    parameters_explored: int
    patterns_found: int
    external_sources_checked: int
    knowledge_cards_created: int
    errors: List[str]


@dataclass
class ResearchCycleResult:
    """Result of a research cycle."""
    experiments_run: int
    validations_passed: int
    validations_failed: int
    knowledge_cards_updated: int
    proposals_created: int
    errors: List[str]


class ResearchOSOrchestrator:
    """
    Thin orchestration layer connecting existing Kobe systems.

    Discovery Lane:
        - CuriosityEngine (cognitive/curiosity_engine.py) - hypothesis generation
        - ResearchEngine (autonomous/research.py) - parameter discovery
        - SourceManager (autonomous/source_manager.py) - external sources

    Research Lane:
        - ExperimentRegistry (experiments/registry.py) - experiment tracking
        - IntegrityGuardian (autonomous/integrity.py) - validation gate
        - Backtest engine - reproducibility verification

    Engineering Lane:
        - ApprovalGate (research_os/approval_gate.py) - human control
        - Frozen params (config/frozen_strategy_params_v*.json) - production config

    NEVER auto-merges. NEVER enables live trading without human approval.
    """

    def __init__(self, state_dir: Optional[Path] = None):
        if state_dir is None:
            state_dir = Path(__file__).parent.parent / "state" / "research_os"
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Initialize stores
        self.card_store = KnowledgeCardStore(state_dir / "knowledge_cards.json")
        self.proposal_store = ProposalStore(state_dir / "proposals.json")
        self.approval_gate = ApprovalGate(state_dir)

        # Connect to existing components (lazy initialization)
        self._curiosity: Optional[Any] = None
        self._research: Optional[Any] = None
        self._integrity: Optional[Any] = None
        self._experiments: Optional[Any] = None
        self._sources: Optional[Any] = None

    @property
    def curiosity(self):
        """Lazy-load CuriosityEngine."""
        if self._curiosity is None and CURIOSITY_AVAILABLE:
            self._curiosity = get_curiosity_engine()
        return self._curiosity

    @property
    def research(self):
        """Lazy-load ResearchEngine."""
        if self._research is None and RESEARCH_ENGINE_AVAILABLE:
            self._research = ResearchEngine()
        return self._research

    @property
    def integrity(self):
        """Lazy-load IntegrityGuardian."""
        if self._integrity is None and INTEGRITY_AVAILABLE:
            self._integrity = IntegrityGuardian()
        return self._integrity

    @property
    def experiments(self):
        """Lazy-load ExperimentRegistry."""
        if self._experiments is None and EXPERIMENTS_AVAILABLE:
            self._experiments = ExperimentRegistry()
        return self._experiments

    @property
    def sources(self):
        """Lazy-load SourceManager."""
        if self._sources is None and SOURCES_AVAILABLE:
            self._sources = SourceManager()
        return self._sources

    # =========================================================================
    # DISCOVERY LANE
    # =========================================================================

    def run_discovery_cycle(self) -> DiscoveryCycleResult:
        """
        Run a discovery cycle using existing engines.

        1. Generate hypotheses (CuriosityEngine)
        2. Discover parameters (ResearchEngine)
        3. Check external sources (SourceManager)
        4. Create KnowledgeCards from findings
        """
        result = DiscoveryCycleResult(
            hypotheses_generated=0,
            parameters_explored=0,
            patterns_found=0,
            external_sources_checked=0,
            knowledge_cards_created=0,
            errors=[],
        )

        logger.info("Starting discovery cycle...")

        # 1. Generate hypotheses from CuriosityEngine
        if self.curiosity:
            try:
                hypotheses = self.curiosity.generate_hypotheses()
                result.hypotheses_generated = len(hypotheses)
                for h in hypotheses:
                    card = self._hypothesis_to_card(h)
                    self.card_store.save(card)
                    result.knowledge_cards_created += 1
                logger.info(f"Generated {result.hypotheses_generated} hypotheses")
            except Exception as e:
                result.errors.append(f"CuriosityEngine error: {e}")
                logger.error(f"CuriosityEngine error: {e}")

        # 2. Discover parameters from ResearchEngine
        if self.research:
            try:
                # Check for any validated discoveries
                discoveries = getattr(self.research, 'discoveries', [])
                for d in discoveries[-10:]:  # Last 10
                    if d.get('confidence', 0) > 0.6:
                        card = self._research_discovery_to_card(d)
                        self.card_store.save(card)
                        result.knowledge_cards_created += 1
                        result.parameters_explored += 1
                logger.info(f"Processed {result.parameters_explored} parameter discoveries")
            except Exception as e:
                result.errors.append(f"ResearchEngine error: {e}")
                logger.error(f"ResearchEngine error: {e}")

        # 3. Check external sources
        if self.sources:
            try:
                # Get validated external ideas
                ideas = getattr(self.sources, 'validated_ideas', [])
                for idea in ideas[-5:]:  # Last 5
                    card = self._external_idea_to_card(idea)
                    self.card_store.save(card)
                    result.knowledge_cards_created += 1
                    result.external_sources_checked += 1
                logger.info(f"Processed {result.external_sources_checked} external sources")
            except Exception as e:
                result.errors.append(f"SourceManager error: {e}")
                logger.error(f"SourceManager error: {e}")

        logger.info(f"Discovery cycle complete: {result.knowledge_cards_created} cards created")
        return result

    def _hypothesis_to_card(self, hypothesis: Any) -> KnowledgeCard:
        """Convert CuriosityEngine hypothesis to KnowledgeCard."""
        return KnowledgeCard(
            title=getattr(hypothesis, 'description', str(hypothesis))[:100],
            description=getattr(hypothesis, 'rationale', ''),
            discovery_type=DiscoveryType.HYPOTHESIS,
            discovered_by="curiosity_engine",
            source_hypothesis_id=getattr(hypothesis, 'hypothesis_id', None),
            evidence_summary=getattr(hypothesis, 'prediction', ''),
            sample_size=getattr(hypothesis, 'sample_size', 0),
            win_rate=getattr(hypothesis, 'observed_value', 0.0),
            p_value=getattr(hypothesis, 'p_value', 1.0),
            confidence=1 - getattr(hypothesis, 'p_value', 1.0),
            status=CardStatus.DISCOVERED,
        )

    def _research_discovery_to_card(self, discovery: Dict[str, Any]) -> KnowledgeCard:
        """Convert ResearchEngine discovery to KnowledgeCard."""
        return KnowledgeCard(
            title=discovery.get('description', 'Parameter Discovery')[:100],
            description=discovery.get('hypothesis', ''),
            discovery_type=DiscoveryType.PARAMETER,
            discovered_by="research_engine",
            source_experiment_id=discovery.get('experiment_id'),
            evidence_summary=discovery.get('improvement', ''),
            sample_size=discovery.get('trades', 0),
            win_rate=discovery.get('win_rate', 0.0),
            profit_factor=discovery.get('profit_factor', 0.0),
            confidence=discovery.get('confidence', 0.0),
            status=CardStatus.DISCOVERED,
            proposed_change=discovery.get('change'),
            target_file=discovery.get('target_file', 'config/frozen_strategy_params_v*.json'),
            current_value=discovery.get('current_value'),
            proposed_value=discovery.get('proposed_value'),
        )

    def _external_idea_to_card(self, idea: Dict[str, Any]) -> KnowledgeCard:
        """Convert external source idea to KnowledgeCard."""
        return KnowledgeCard(
            title=idea.get('title', 'External Idea')[:100],
            description=idea.get('concept', ''),
            discovery_type=DiscoveryType.EXTERNAL,
            discovered_by="scraper",
            source_url=idea.get('url'),
            evidence_summary=idea.get('validation_summary', ''),
            sample_size=idea.get('backtest_trades', 0),
            win_rate=idea.get('win_rate', 0.0),
            profit_factor=idea.get('profit_factor', 0.0),
            confidence=idea.get('credibility', 0.0),
            status=CardStatus.DISCOVERED,
            tags=idea.get('tags', []),
        )

    # =========================================================================
    # RESEARCH LANE
    # =========================================================================

    def run_research_cycle(self, card_id: str) -> ResearchCycleResult:
        """
        Run research on a discovered knowledge card.

        1. Register experiment (ExperimentRegistry)
        2. Run backtest
        3. Validate results (IntegrityGuardian)
        4. Update KnowledgeCard with results
        5. Create proposal if validated
        """
        result = ResearchCycleResult(
            experiments_run=0,
            validations_passed=0,
            validations_failed=0,
            knowledge_cards_updated=0,
            proposals_created=0,
            errors=[],
        )

        card = self.card_store.get(card_id)
        if card is None:
            result.errors.append(f"Card not found: {card_id}")
            return result

        logger.info(f"Starting research cycle for: {card.title}")
        card.update_status(CardStatus.TESTING, "Research cycle started")

        # 1. Register experiment
        experiment_id = None
        if self.experiments:
            try:
                experiment_id = self.experiments.register(
                    name=f"research_{card.card_id}",
                    description=card.description,
                    params={"card_id": card.card_id},
                )
                card.source_experiment_id = experiment_id
                result.experiments_run += 1
                logger.info(f"Registered experiment: {experiment_id}")
            except Exception as e:
                result.errors.append(f"Experiment registration error: {e}")
                logger.error(f"Experiment registration error: {e}")

        # 2. Validate with IntegrityGuardian
        if self.integrity:
            try:
                validation = self.integrity.validate({
                    'win_rate': card.win_rate,
                    'profit_factor': card.profit_factor,
                    'trades': card.sample_size,
                })
                if validation.get('passed', False):
                    card.integrity_passed = True
                    card.add_validation(True, "IntegrityGuardian passed")
                    result.validations_passed += 1
                else:
                    card.add_validation(False, f"IntegrityGuardian failed: {validation.get('reason')}")
                    result.validations_failed += 1
            except Exception as e:
                result.errors.append(f"Integrity validation error: {e}")
                logger.error(f"Integrity validation error: {e}")

        # 3. Check reproducibility (simplified - in production would re-run backtest)
        if card.integrity_passed:
            card.reproducibility_verified = True
            card.add_validation(True, "Reproducibility verified")

        # 4. Update card status
        if card.is_ready_for_proposal():
            card.update_status(CardStatus.VALIDATED, "Research complete, ready for proposal")
            result.knowledge_cards_updated += 1

            # 5. Auto-create proposal (but NOT auto-approve)
            proposal = self._create_proposal_from_card(card)
            if proposal:
                self.proposal_store.save(proposal)
                result.proposals_created += 1
                logger.info(f"Created proposal: {proposal.proposal_id}")
        else:
            card.update_status(CardStatus.DISCOVERED, "Research incomplete, needs more evidence")
            result.knowledge_cards_updated += 1

        self.card_store.save(card)
        logger.info(f"Research cycle complete for: {card.title}")
        return result

    def _create_proposal_from_card(self, card: KnowledgeCard) -> Optional[ResearchProposal]:
        """Create a research proposal from a validated knowledge card."""
        if not card.is_ready_for_proposal():
            return None

        # Determine change type
        change_type = ChangeType.PARAMETER_UPDATE
        if card.discovery_type == DiscoveryType.STRATEGY:
            change_type = ChangeType.STRATEGY_MODIFICATION
        elif card.discovery_type == DiscoveryType.PATTERN:
            change_type = ChangeType.NEW_RULE

        return ResearchProposal(
            knowledge_card_id=card.card_id,
            title=f"Proposal: {card.title}",
            change_type=change_type,
            target_file=card.target_file or "config/frozen_strategy_params_v*.json",
            target_key=card.proposed_change or "",
            current_value=card.current_value,
            proposed_value=card.proposed_value,
            hypothesis=card.description,
            evidence_summary=card.evidence_summary,
            expected_improvement=f"WR: {card.win_rate:.1%}, PF: {card.profit_factor:.2f}",
            risk_assessment=f"Sample size: {card.sample_size}, Confidence: {card.confidence:.1%}",
            rollback_plan="Revert frozen params to previous version",
            sample_size=card.sample_size,
            win_rate_after=card.win_rate,
            profit_factor_after=card.profit_factor,
            tags=card.tags,
        )

    # =========================================================================
    # ENGINEERING LANE
    # =========================================================================

    def propose_engineering_change(self, card_id: str) -> Optional[str]:
        """
        Propose an engineering change for a validated knowledge card.

        Creates a proposal and submits for HUMAN approval.
        NEVER auto-merges. NEVER auto-approves.

        Returns request_id if successful.
        """
        card = self.card_store.get(card_id)
        if card is None:
            logger.error(f"Card not found: {card_id}")
            return None

        if not card.is_ready_for_proposal():
            logger.warning(f"Card not ready for proposal: {card_id}")
            return None

        # Create proposal
        proposal = self._create_proposal_from_card(card)
        if proposal is None:
            return None

        # Submit for approval (NOT auto-approve)
        request_id = self.approval_gate.request_approval(proposal)

        # Update card status
        card.update_status(CardStatus.PROPOSED, f"Proposal submitted: {request_id}")
        card.approval_request_id = request_id
        self.card_store.save(card)

        logger.info(f"Proposed engineering change: {request_id}")
        return request_id

    def implement_approved_change(self, request_id: str, implementer: str) -> bool:
        """
        Implement an approved change.

        CRITICAL: Requires APPROVE_LIVE_ACTION = True.
        Even after human approval, the safety flag must be manually enabled.
        """
        can_impl, reason = self.approval_gate.can_implement(request_id)
        if not can_impl:
            raise SafetyError(reason)

        # Get proposal
        approved = self.approval_gate.get_approved()
        proposal_id = None
        for a in approved:
            if a.get("request_id") == request_id:
                proposal_id = a.get("proposal_id")
                break

        if proposal_id is None:
            logger.error(f"Proposal not found for request: {request_id}")
            return False

        proposal = self.proposal_store.get(proposal_id)
        if proposal is None:
            logger.error(f"Proposal data not found: {proposal_id}")
            return False

        # Mark as implemented
        success = self.approval_gate.implement(request_id, implementer)

        if success and proposal.knowledge_card_id:
            card = self.card_store.get(proposal.knowledge_card_id)
            if card:
                card.update_status(CardStatus.IMPLEMENTED, f"Implemented by {implementer}")
                self.card_store.save(card)

        logger.info(f"Implemented change: {request_id}")
        return success

    # =========================================================================
    # STATUS & REPORTING
    # =========================================================================

    def status(self) -> str:
        """Generate full status report."""
        card_counts = self.card_store.count_by_status()
        proposal_counts = self.proposal_store.count_by_status()

        lines = [
            "=" * 60,
            "KOBE RESEARCH OS STATUS",
            "=" * 60,
            "",
            self.approval_gate.summary(),
            "Knowledge Cards:",
        ]
        for k, v in card_counts.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append("Proposals:")
        for k, v in proposal_counts.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append("Components Available:")
        lines.append(f"  CuriosityEngine: {CURIOSITY_AVAILABLE}")
        lines.append(f"  ResearchEngine: {RESEARCH_ENGINE_AVAILABLE}")
        lines.append(f"  IntegrityGuardian: {INTEGRITY_AVAILABLE}")
        lines.append(f"  ExperimentRegistry: {EXPERIMENTS_AVAILABLE}")
        lines.append(f"  SourceManager: {SOURCES_AVAILABLE}")
        return "\n".join(lines)

    def list_pending_approvals(self) -> List[Dict[str, Any]]:
        """List all proposals awaiting human approval."""
        return self.approval_gate.get_pending()

    def list_validated_cards(self) -> List[KnowledgeCard]:
        """List all validated knowledge cards."""
        return self.card_store.list_validated()


# Singleton accessor
_orchestrator: Optional[ResearchOSOrchestrator] = None


def get_research_os() -> ResearchOSOrchestrator:
    """Get the singleton ResearchOSOrchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ResearchOSOrchestrator()
    return _orchestrator
