"""
Enhanced Autonomous Brain - Next-Generation Kobe Brain with Alpha Mining + LangGraph.

This extends the base AutonomousBrain with:
- EnhancedResearchEngine (VectorBT alpha mining, Alphalens validation)
- KobeBrainGraph (LangGraph formal state machine)
- RAGEvaluator (LLM reasoning quality tracking)
- Unified discovery alerting across all components

Created: 2026-01-07
Purpose: Integrate all new cognitive and alpha mining components into the autonomous brain
"""

import json
import logging
import signal
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from zoneinfo import ZoneInfo

# NOTE (2026-01-08): This brain EXTENDS autonomous.brain.AutonomousBrain
# It adds alpha mining features. The base brain is still canonical.
# Use EnhancedAutonomousBrain for advanced features, AutonomousBrain for standard operation.

from autonomous.brain import AutonomousBrain, Discovery
from autonomous.enhanced_research import EnhancedResearchEngine
from autonomous.learning import LearningEngine
from autonomous.awareness import ContextBuilder, MarketContext, WorkMode
from autonomous.scheduler import AutonomousScheduler
from autonomous.handlers import register_all_handlers

from core.structured_log import jlog

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class EnhancedAutonomousBrain(AutonomousBrain):
    """
    Enhanced Autonomous Brain with advanced capabilities.

    New Features:
    - EnhancedResearchEngine for alpha mining (VectorBT, Alphalens)
    - KobeBrainGraph for formal state machine decisions (LangGraph)
    - RAGEvaluator for LLM explanation quality tracking
    - Unified discovery system across all components
    """

    VERSION = "2.0.0"

    def __init__(self, state_dir: Optional[Path] = None, use_langgraph: bool = False):
        """
        Initialize enhanced brain.

        Args:
            state_dir: Directory for state persistence
            use_langgraph: Enable LangGraph formal state machine
        """
        # Initialize base brain (sets up context, scheduler, base research, learning)
        # We'll override research engine after
        super().__init__(state_dir)

        # Replace base ResearchEngine with EnhancedResearchEngine
        # Use self.state_dir which is set by parent __init__
        self.research = EnhancedResearchEngine(self.state_dir / "research")

        # Optional: LangGraph brain for formal state machine
        self.use_langgraph = use_langgraph
        self.brain_graph = None
        if use_langgraph:
            self._init_brain_graph()

        # RAG Evaluator for LLM quality tracking
        self.rag_evaluator = None
        self._init_rag_evaluator()

        # Enhanced discovery tracking
        self.alpha_discoveries_count = 0
        self.rag_evaluations_count = 0

        logger.info(f"EnhancedAutonomousBrain v{self.VERSION} initialized")
        logger.info(f"  LangGraph enabled: {use_langgraph}")
        logger.info(f"  RAGEvaluator enabled: {self.rag_evaluator is not None}")

    def _init_brain_graph(self):
        """Initialize LangGraph formal state machine."""
        try:
            from cognitive.brain_graph import get_brain_graph, HAS_LANGGRAPH

            if not HAS_LANGGRAPH:
                logger.warning("LangGraph not available. Run: pip install langgraph")
                return

            self.brain_graph = get_brain_graph()
            if self.brain_graph:
                logger.info("LangGraph brain initialized successfully")
            else:
                logger.warning("LangGraph brain initialization failed")

        except Exception as e:
            logger.warning(f"Could not initialize LangGraph brain: {e}")
            self.brain_graph = None

    def _init_rag_evaluator(self):
        """Initialize RAG evaluator for LLM quality tracking."""
        try:
            from cognitive.rag_evaluator import get_rag_evaluator

            self.rag_evaluator = get_rag_evaluator()
            logger.info("RAGEvaluator initialized successfully")

        except Exception as e:
            logger.warning(f"Could not initialize RAGEvaluator: {e}")
            self.rag_evaluator = None

    # =========================================================================
    # ENHANCED DISCOVERY CHECKING - All sources unified
    # =========================================================================

    def _check_for_discoveries(self) -> list:
        """
        Check all sources for new discoveries (BASE + ENHANCED).

        Sources:
        1. Base ResearchEngine parameter experiments (from parent)
        2. EnhancedResearchEngine alpha mining
        3. LangGraph state transitions (if enabled)
        4. RAGEvaluator explanation quality improvements
        5. External scrapers (from parent)

        Returns:
            List of Discovery objects to alert
        """
        discoveries = []

        # 1. Base research discoveries (from parent class)
        base_discoveries = super()._check_for_discoveries()
        discoveries.extend(base_discoveries)

        # 2. Alpha mining discoveries
        alpha_discoveries = self._check_alpha_discoveries()
        discoveries.extend(alpha_discoveries)

        # 3. LangGraph discoveries (state transition insights)
        if self.use_langgraph and self.brain_graph:
            graph_discoveries = self._check_langgraph_discoveries()
            discoveries.extend(graph_discoveries)

        # 4. RAG evaluator discoveries (LLM quality improvements)
        if self.rag_evaluator:
            rag_discoveries = self._check_rag_discoveries()
            discoveries.extend(rag_discoveries)

        return discoveries

    def _check_alpha_discoveries(self) -> List[Discovery]:
        """Check for new high-quality alpha discoveries."""
        discoveries = []

        # Get recent alpha discoveries from enhanced research engine
        for alpha_disc in self.research.alpha_discoveries:
            # Check if this is a new discovery worth alerting
            if getattr(alpha_disc, '_alerted', False):
                continue

            # Criteria for alerting:
            # 1. High Sharpe (> 1.0)
            # 2. Validated with Alphalens
            # 3. Statistically significant
            if (
                alpha_disc.sharpe_ratio > 1.0
                and alpha_disc.validated
                and alpha_disc.statistically_significant
            ):
                discovery = Discovery(
                    discovery_type="alpha_discovery",
                    description=f"High-quality alpha '{alpha_disc.alpha_name}' discovered and validated",
                    source="enhanced_research_engine",
                    improvement=alpha_disc.sharpe_ratio - 0.5,  # vs baseline Sharpe
                    confidence=min(0.95, 0.5 + alpha_disc.sharpe_ratio / 5),
                    data={
                        "alpha_id": alpha_disc.alpha_id,
                        "alpha_name": alpha_disc.alpha_name,
                        "category": alpha_disc.category,
                        "sharpe_ratio": alpha_disc.sharpe_ratio,
                        "win_rate": alpha_disc.win_rate,
                        "profit_factor": alpha_disc.profit_factor,
                        "ic_mean": alpha_disc.ic_mean,
                        "ic_sharpe": alpha_disc.ic_sharpe,
                        "statistically_significant": alpha_disc.statistically_significant,
                    },
                )
                discoveries.append(discovery)
                alpha_disc._alerted = True
                self.alpha_discoveries_count += 1

        return discoveries

    def _check_langgraph_discoveries(self) -> List[Discovery]:
        """Check for insights from LangGraph state transitions."""
        discoveries = []

        # This would track patterns in state transitions
        # For now, placeholder for future implementation
        # Example: "Notice that DECIDE->STANDBY happens frequently at 9:35 AM"

        return discoveries

    def _check_rag_discoveries(self) -> List[Discovery]:
        """Check for LLM explanation quality improvements."""
        discoveries = []

        try:
            # Get retriever performance stats
            stats = self.rag_evaluator.get_retriever_stats()

            # Check for high-performing retrievers
            from cognitive.rag_evaluator import RetrieverType

            for retriever_type, rt_stats in stats.items():
                if rt_stats['n_samples'] < 10:
                    continue

                # Alert if a retriever consistently scores > 85
                if rt_stats['avg_score'] > 85 and not hasattr(retriever_type, '_alerted'):
                    discovery = Discovery(
                        discovery_type="rag_quality",
                        description=f"Retriever '{retriever_type.value}' consistently produces high-quality explanations",
                        source="rag_evaluator",
                        improvement=(rt_stats['avg_score'] - 70) / 100,  # vs baseline 70
                        confidence=0.8,
                        data={
                            "retriever_type": retriever_type.value,
                            "avg_score": rt_stats['avg_score'],
                            "n_samples": rt_stats['n_samples'],
                            "avg_human_rating": rt_stats.get('avg_human_rating', 0),
                        },
                    )
                    discoveries.append(discovery)
                    retriever_type._alerted = True
                    self.rag_evaluations_count += 1

        except Exception as e:
            logger.debug(f"RAG discovery check failed: {e}")

        return discoveries

    # =========================================================================
    # ENHANCED BACKGROUND WORK - Alpha mining + RAG evaluation
    # =========================================================================

    def _do_background_work(self, context: MarketContext) -> Dict[str, Any]:
        """
        Enhanced background work that includes alpha mining.

        Extends parent's background work with:
        - Alpha mining sweeps during DEEP_RESEARCH
        - RAG evaluation during LEARNING
        - Hypothesis submission to CuriosityEngine
        """
        work_mode = context.work_mode

        if work_mode == WorkMode.DEEP_RESEARCH:
            # Weekend/holiday: Run alpha mining
            logger.info("Deep research mode: Running alpha mining sweep")
            return self._run_alpha_mining_sweep()

        elif work_mode == WorkMode.RESEARCH:
            # Regular research time: Mix of experiments and alpha validation
            if self.cycles_completed % 3 == 0:
                # Every 3rd cycle: alpha mining
                logger.info("Research mode: Running alpha mining")
                return self._run_alpha_mining_sweep()
            else:
                # Other cycles: parameter experiments (base behavior)
                return super()._do_background_work(context)

        elif work_mode == WorkMode.LEARNING:
            # Learning time: Analyze trades + evaluate RAG quality
            logger.info("Learning mode: Analyzing trades + RAG evaluation")
            trade_result = self.learning.analyze_trades()

            # Also evaluate RAG explanations if available
            if self.rag_evaluator:
                rag_result = self._evaluate_recent_explanations()
                return {
                    **trade_result,
                    "rag_evaluation": rag_result,
                }
            return trade_result

        elif work_mode == WorkMode.OPTIMIZATION:
            # Night: Run feature analysis + submit hypotheses
            logger.info("Optimization mode: Feature analysis + hypothesis submission")
            feature_result = self.research.analyze_features()

            # Submit alpha hypotheses to CuriosityEngine
            hypotheses_submitted = self.research.submit_alpha_hypotheses_to_curiosity_engine()

            return {
                **feature_result,
                "hypotheses_submitted": hypotheses_submitted,
            }

        else:
            # Monitoring or active trading - delegate to parent
            return super()._do_background_work(context)

    def _run_alpha_mining_sweep(self) -> Dict[str, Any]:
        """Run VectorBT alpha mining sweep."""
        try:
            result = self.research.run_vectorbt_alpha_sweep(
                min_sharpe=0.5,
                min_trades=30,
            )
            return result
        except Exception as e:
            logger.error(f"Alpha mining sweep failed: {e}")
            return {"status": "error", "error": str(e)}

    def _evaluate_recent_explanations(self) -> Dict[str, Any]:
        """Evaluate recent RAG explanations."""
        try:
            # Get recent evaluations
            stats = self.rag_evaluator.get_retriever_stats()
            return {
                "status": "success",
                "retriever_stats": {
                    rt.value: {
                        "n_samples": st['n_samples'],
                        "avg_score": st['avg_score'],
                    }
                    for rt, st in stats.items()
                },
            }
        except Exception as e:
            logger.error(f"RAG evaluation failed: {e}")
            return {"status": "error", "error": str(e)}

    # =========================================================================
    # LANGGRAPH INTEGRATION - Formal state machine execution
    # =========================================================================

    def think_with_langgraph(self) -> Dict[str, Any]:
        """
        Run thinking cycle using LangGraph formal state machine.

        This is an alternative to the base think() method that uses
        a formal StateGraph instead of ad-hoc logic.

        Returns:
            Result of the brain cycle
        """
        if not self.use_langgraph or not self.brain_graph:
            logger.warning("LangGraph not enabled, falling back to base think()")
            return self.think()

        try:
            # Run a brain cycle through the state graph
            from cognitive.states import create_initial_brain_state

            initial_state = create_initial_brain_state()
            result = self.brain_graph.run_cycle(initial_state)

            # Convert LangGraph result to standard format
            return {
                "timestamp": result.get("timestamp"),
                "trading_phase": result.get("trading_phase"),
                "decision": result.get("decision"),
                "decision_reason": result.get("decision_reason"),
                "confidence": result.get("confidence", 0.5),
                "orders_placed": result.get("orders_placed", []),
                "langgraph_enabled": True,
            }

        except Exception as e:
            logger.error(f"LangGraph think cycle failed: {e}")
            # Fallback to base think()
            return self.think()

    # =========================================================================
    # ENHANCED STATUS - All components
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive enhanced brain status."""
        # Get base status
        base_status = super().get_status()

        # Add alpha mining stats
        alpha_summary = self.research.get_enhanced_research_summary()

        # Add LangGraph stats
        langgraph_status = {
            "enabled": self.use_langgraph,
            "available": self.brain_graph is not None,
        }

        # Add RAG evaluator stats
        rag_status = {
            "enabled": self.rag_evaluator is not None,
            "evaluations_count": self.rag_evaluations_count,
        }
        if self.rag_evaluator:
            try:
                stats = self.rag_evaluator.get_retriever_stats()
                rag_status["retriever_count"] = len(stats)
                rag_status["total_evaluations"] = sum(
                    s['n_samples'] for s in stats.values()
                )
            except Exception:
                pass

        return {
            **base_status,
            "enhanced_version": self.VERSION,
            "alpha_mining": alpha_summary.get("alpha_mining", {}),
            "langgraph": langgraph_status,
            "rag_evaluator": rag_status,
            "alpha_discoveries_alerted": self.alpha_discoveries_count,
            "rag_discoveries_alerted": self.rag_evaluations_count,
        }

    # =========================================================================
    # RUN MODES
    # =========================================================================

    def run_forever(self, cycle_seconds: int = 60, use_langgraph: bool = False):
        """
        Run the enhanced brain forever.

        Args:
            cycle_seconds: Seconds between cycles
            use_langgraph: Use LangGraph for thinking (overrides init setting)
        """
        if use_langgraph:
            self.use_langgraph = True
            if not self.brain_graph:
                self._init_brain_graph()

        self.running = True
        self.started_at = datetime.now(ET)

        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Shutdown signal received")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("=" * 60)
        logger.info("ENHANCED AUTONOMOUS BRAIN STARTING")
        logger.info(f"Version: {self.VERSION}")
        logger.info(f"Cycle interval: {cycle_seconds}s")
        logger.info(f"LangGraph mode: {self.use_langgraph}")
        logger.info("=" * 60)

        # Initial status
        status = self.get_status()
        logger.info(f"Current phase: {status['awareness']['phase']}")
        logger.info(f"Work mode: {status['awareness']['work_mode']}")
        logger.info(f"Alpha discoveries: {status['alpha_mining']['total_discoveries']}")

        try:
            while self.running:
                try:
                    # Think and act (with optional LangGraph)
                    if self.use_langgraph and self.brain_graph:
                        result = self.think_with_langgraph()
                    else:
                        result = self.think()

                    if result.get("task"):
                        logger.info(f"Completed: {result['task']}")

                    # Save state periodically
                    if self.cycles_completed % 10 == 0:
                        self.save_state()

                    # Log status every hour
                    if self.cycles_completed % 60 == 0:
                        status = self.get_status()
                        logger.info(
                            f"Hourly status: {self.cycles_completed} cycles, "
                            f"uptime {status['uptime_hours']:.1f}h, "
                            f"mode: {status['awareness']['work_mode']}, "
                            f"alphas: {status['alpha_mining']['total_discoveries']}"
                        )

                except Exception as e:
                    logger.error(f"Error in think cycle: {e}")

                # Wait for next cycle
                time.sleep(cycle_seconds)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        finally:
            self.running = False
            self.save_state()
            logger.info("Enhanced autonomous brain stopped")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_enhanced_brain(
    cycle_seconds: int = 60,
    use_langgraph: bool = False,
    state_dir: Optional[Path] = None,
):
    """
    Run the enhanced autonomous brain.

    Args:
        cycle_seconds: Seconds between cycles
        use_langgraph: Enable LangGraph formal state machine
        state_dir: State persistence directory
    """
    brain = EnhancedAutonomousBrain(state_dir=state_dir, use_langgraph=use_langgraph)
    brain.run_forever(cycle_seconds=cycle_seconds)


def get_enhanced_brain_status(state_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Get enhanced brain status without starting it."""
    brain = EnhancedAutonomousBrain(state_dir=state_dir, use_langgraph=False)
    return brain.get_status()


if __name__ == "__main__":
    # Demo/CLI
    import argparse

    parser = argparse.ArgumentParser(description="Kobe Enhanced Autonomous Brain")
    parser.add_argument(
        "--cycle", type=int, default=60,
        help="Cycle interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--langgraph", action="store_true",
        help="Enable LangGraph formal state machine"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run single cycle and exit"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show status and exit"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    brain = EnhancedAutonomousBrain(use_langgraph=args.langgraph)

    if args.status:
        status = brain.get_status()
        print(json.dumps(status, indent=2))
        exit(0)

    if args.once:
        if args.langgraph:
            result = brain.think_with_langgraph()
        else:
            result = brain.run_single_cycle()
        print(json.dumps(result, indent=2))
        exit(0)

    # Run forever
    brain.run_forever(cycle_seconds=args.cycle, use_langgraph=args.langgraph)
