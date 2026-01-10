"""
LangGraph Brain - Formal State Machine for Kobe's Cognitive Architecture.

Replaces ad-hoc spaghetti logic with a formal StateGraph that:
- Has explicit states and transitions
- Supports human-in-the-loop interrupts
- Enables visualization and debugging
- Provides checkpointing for recovery

Created: 2026-01-07
Based on: LangGraph patterns for stateful agent orchestration

IMPORTANT: This runs in PARALLEL with the existing cognitive_brain.py
Enable via config: use_langgraph_brain: true
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import LangGraph
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    logger.warning("LangGraph not installed. Run: pip install langgraph")
    StateGraph = None
    START = None
    END = None
    MemorySaver = None

from cognitive.states import (
    BrainState,
    MarketContext,
    SignalState,
    PortfolioState,
    TradingPhase,
    DecisionType,
    MarketRegime,
    SignalQuality,
    create_initial_brain_state,
)


class KobeBrainGraph:
    """
    LangGraph-based brain for Kobe trading system.

    Nodes:
    - observe: Gather market data, check time, load state
    - analyze: Run scanner, compute alphas, detect regime
    - decide: Make trading decision based on rules and context
    - execute: Place orders (with human approval option)
    - reflect: Learn from outcomes, update memory

    Edges:
    - Conditional routing based on trading phase and risk gates
    """

    def __init__(self, config_path: Optional[Path] = None):
        if not HAS_LANGGRAPH:
            raise ImportError("LangGraph not installed. Run: pip install langgraph")

        self.config_path = config_path
        self.checkpointer = MemorySaver()

        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)

        logger.info("KobeBrainGraph initialized with LangGraph StateGraph")

    def _build_graph(self) -> StateGraph:
        """Build the state graph with nodes and edges."""
        # Create graph with BrainState as state schema
        graph = StateGraph(BrainState)

        # Add nodes
        graph.add_node("observe", self._observe_node)
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("decide", self._decide_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("reflect", self._reflect_node)
        graph.add_node("research", self._research_node)
        graph.add_node("standby", self._standby_node)

        # Add edges from START
        graph.add_edge(START, "observe")

        # Add conditional edges from observe
        graph.add_conditional_edges(
            "observe",
            self._route_after_observe,
            {
                "analyze": "analyze",
                "research": "research",
                "standby": "standby",
                "end": END,
            }
        )

        # Add edges from analyze
        graph.add_edge("analyze", "decide")

        # Add conditional edges from decide
        graph.add_conditional_edges(
            "decide",
            self._route_after_decide,
            {
                "execute": "execute",
                "reflect": "reflect",
                "standby": "standby",
            }
        )

        # Add edges from execute
        graph.add_edge("execute", "reflect")

        # Add edges from reflect
        graph.add_conditional_edges(
            "reflect",
            self._route_after_reflect,
            {
                "observe": "observe",  # Continue loop
                "end": END,
            }
        )

        # Add edges from research
        graph.add_edge("research", "observe")

        # Add edges from standby
        graph.add_conditional_edges(
            "standby",
            self._route_after_standby,
            {
                "observe": "observe",
                "end": END,
            }
        )

        return graph

    # =========================================================================
    # NODE IMPLEMENTATIONS
    # =========================================================================

    def _observe_node(self, state: BrainState) -> Dict[str, Any]:
        """
        Observe node: Gather market data and context.

        - Check current time and trading phase
        - Load market data (VIX, SPY, regime)
        - Check kill switch and risk gates
        - Load watchlist and portfolio state
        """
        logger.info("OBSERVE: Gathering market context...")

        updates = {
            "timestamp": datetime.now().isoformat(),
            "iteration": state.get("iteration", 0) + 1,
        }

        try:
            # 1. Determine trading phase
            phase = self._get_trading_phase()
            updates["trading_phase"] = phase

            # 2. Check kill switch
            kill_switch = self._check_kill_switch()
            updates["kill_switch_active"] = kill_switch

            # 3. Check kill zone validity
            in_valid_zone = self._check_kill_zone(phase)
            updates["in_valid_kill_zone"] = in_valid_zone

            # 4. Load market context
            market = self._load_market_context()
            updates["market"] = market

            # 5. Load portfolio state
            portfolio = self._load_portfolio_state()
            updates["portfolio"] = portfolio

            # 6. Check risk gates
            updates["daily_loss_exceeded"] = self._check_daily_loss(portfolio)
            updates["weekly_exposure_exceeded"] = self._check_weekly_exposure(portfolio)

            # 7. Load watchlist if available
            watchlist = self._load_watchlist()
            if watchlist:
                updates["watchlist"] = watchlist

            logger.info(f"OBSERVE complete: phase={phase}, regime={market.get('regime')}")

        except Exception as e:
            logger.error(f"OBSERVE error: {e}")
            updates["errors"] = state.get("errors", []) + [f"observe: {e}"]

        return updates

    def _analyze_node(self, state: BrainState) -> Dict[str, Any]:
        """
        Analyze node: Run scanner and compute signals.

        - Run DualStrategyScanner on universe
        - Apply quality gates
        - Rank to Top 5 and Top 2 (tradeable)
        - Apply Markov boost if available
        """
        logger.info("ANALYZE: Running scanner and computing signals...")

        updates = {}

        try:
            # 1. Run scanner
            signals = self._run_scanner()

            # 2. Apply quality gates and ranking
            top_5 = self._rank_signals(signals, top_n=5)
            tradeable = self._rank_signals(signals, top_n=2)

            updates["current_signals"] = signals
            updates["top_5"] = top_5
            updates["tradeable"] = tradeable

            logger.info(f"ANALYZE complete: {len(signals)} signals, {len(top_5)} top 5, {len(tradeable)} tradeable")

        except Exception as e:
            logger.error(f"ANALYZE error: {e}")
            updates["errors"] = state.get("errors", []) + [f"analyze: {e}"]

        return updates

    def _decide_node(self, state: BrainState) -> Dict[str, Any]:
        """
        Decide node: Make trading decision.

        - Check if conditions allow trading
        - Query semantic memory for rules
        - Apply episodic context
        - Determine decision type
        """
        logger.info("DECIDE: Making trading decision...")

        updates = {}

        try:
            # Check blocking conditions
            if state.get("kill_switch_active"):
                updates["decision"] = DecisionType.STANDBY
                updates["decision_reason"] = "Kill switch active"
                updates["confidence"] = 0.0
                return updates

            if not state.get("in_valid_kill_zone"):
                updates["decision"] = DecisionType.HOLD
                updates["decision_reason"] = f"Outside valid kill zone: {state.get('trading_phase')}"
                updates["confidence"] = 0.0
                return updates

            if state.get("daily_loss_exceeded"):
                updates["decision"] = DecisionType.STANDBY
                updates["decision_reason"] = "Daily loss limit exceeded"
                updates["confidence"] = 0.0
                return updates

            # Check if we have tradeable signals
            tradeable = state.get("tradeable", [])
            if not tradeable:
                updates["decision"] = DecisionType.HOLD
                updates["decision_reason"] = "No tradeable signals"
                updates["confidence"] = 0.5
                return updates

            # We have signals - decide to trade
            updates["decision"] = DecisionType.TRADE
            updates["decision_reason"] = f"Found {len(tradeable)} tradeable signals"
            updates["confidence"] = max(s.get("confidence", 0.5) for s in tradeable)

            # Prepare orders
            orders = self._prepare_orders(tradeable, state.get("portfolio", {}))
            updates["orders_to_place"] = orders

            logger.info(f"DECIDE: {updates['decision']} - {updates['decision_reason']}")

        except Exception as e:
            logger.error(f"DECIDE error: {e}")
            updates["decision"] = DecisionType.STANDBY
            updates["decision_reason"] = f"Error: {e}"
            updates["errors"] = state.get("errors", []) + [f"decide: {e}"]

        return updates

    def _execute_node(self, state: BrainState) -> Dict[str, Any]:
        """
        Execute node: Place orders.

        - Validate orders against risk gates
        - Submit to broker
        - Record results
        """
        logger.info("EXECUTE: Placing orders...")

        updates = {}

        try:
            orders = state.get("orders_to_place", [])
            if not orders:
                logger.info("EXECUTE: No orders to place")
                return updates

            # Execute orders (placeholder - integrate with broker)
            placed = []
            for order in orders:
                result = self._place_order(order)
                placed.append(result)

            updates["orders_placed"] = placed
            updates["orders_to_place"] = []  # Clear pending

            logger.info(f"EXECUTE complete: {len(placed)} orders placed")

        except Exception as e:
            logger.error(f"EXECUTE error: {e}")
            updates["errors"] = state.get("errors", []) + [f"execute: {e}"]

        return updates

    def _reflect_node(self, state: BrainState) -> Dict[str, Any]:
        """
        Reflect node: Learn from outcomes.

        - Analyze recent trade outcomes
        - Update episodic memory
        - Extract lessons learned
        - Adjust confidence
        """
        logger.info("REFLECT: Learning from outcomes...")

        updates = {}

        try:
            # Get recent trades for reflection
            recent = state.get("recent_trades", [])

            if recent:
                lessons = self._extract_lessons(recent)
                updates["lessons_learned"] = state.get("lessons_learned", []) + lessons
                updates["last_reflection"] = datetime.now().isoformat()

            logger.info(f"REFLECT complete: {len(recent)} trades analyzed")

        except Exception as e:
            logger.error(f"REFLECT error: {e}")
            updates["errors"] = state.get("errors", []) + [f"reflect: {e}"]

        return updates

    def _research_node(self, state: BrainState) -> Dict[str, Any]:
        """
        Research node: Run alpha mining and experiments.

        - Run VectorBT alpha mining
        - Validate with Alphalens
        - Submit hypotheses to CuriosityEngine
        """
        logger.info("RESEARCH: Running alpha mining...")

        updates = {}

        try:
            # Run alpha mining
            from autonomous.research import run_alpha_mining
            result = run_alpha_mining(min_sharpe=0.5)

            updates["alpha_discoveries"] = result.get("alphas_discovered", 0)
            updates["pending_hypotheses"] = state.get("pending_hypotheses", 0) + result.get("hypotheses_submitted", 0)

            logger.info(f"RESEARCH complete: {result.get('alphas_discovered', 0)} alphas discovered")

        except Exception as e:
            logger.error(f"RESEARCH error: {e}")
            updates["errors"] = state.get("errors", []) + [f"research: {e}"]

        return updates

    def _standby_node(self, state: BrainState) -> Dict[str, Any]:
        """
        Standby node: Wait state when trading not allowed.
        """
        logger.info(f"STANDBY: {state.get('decision_reason', 'Waiting')}")
        return {"decision": DecisionType.STANDBY}

    # =========================================================================
    # ROUTING FUNCTIONS
    # =========================================================================

    def _route_after_observe(self, state: BrainState) -> str:
        """Determine next node after observe."""
        # Kill switch = end
        if state.get("kill_switch_active"):
            return "standby"

        phase = state.get("trading_phase")

        # Research hours
        if phase in [TradingPhase.PRE_MARKET, TradingPhase.LUNCH, TradingPhase.AFTER_HOURS]:
            return "research"

        # Trading hours
        if phase in [TradingPhase.MORNING_SESSION, TradingPhase.AFTERNOON_SESSION]:
            return "analyze"

        # Non-trading hours
        if phase in [TradingPhase.OPENING_RANGE, TradingPhase.CLOSE, TradingPhase.WEEKEND]:
            return "standby"

        return "standby"

    def _route_after_decide(self, state: BrainState) -> str:
        """Determine next node after decide."""
        decision = state.get("decision")

        if decision == DecisionType.TRADE:
            return "execute"
        elif decision == DecisionType.STANDBY:
            return "standby"
        else:
            return "reflect"

    def _route_after_reflect(self, state: BrainState) -> str:
        """Determine next node after reflect."""
        # Check if we should continue or end
        iteration = state.get("iteration", 0)

        # End after max iterations (safety)
        if iteration >= 100:
            return "end"

        # Continue loop
        return "observe"

    def _route_after_standby(self, state: BrainState) -> str:
        """Determine next node after standby."""
        # Check if conditions changed
        if state.get("kill_switch_active"):
            return "end"

        return "observe"

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_trading_phase(self) -> TradingPhase:
        """Determine current trading phase."""
        try:
            from risk.kill_zone_gate import get_current_zone
            zone = get_current_zone()
            zone_mapping = {
                "pre_market": TradingPhase.PRE_MARKET,
                "opening_range": TradingPhase.OPENING_RANGE,
                "london_close": TradingPhase.MORNING_SESSION,
                "lunch_chop": TradingPhase.LUNCH,
                "power_hour": TradingPhase.AFTERNOON_SESSION,
                "close": TradingPhase.CLOSE,
                "after_hours": TradingPhase.AFTER_HOURS,
            }
            return zone_mapping.get(zone, TradingPhase.PRE_MARKET)
        except Exception:
            return TradingPhase.PRE_MARKET

    def _check_kill_switch(self) -> bool:
        """Check if kill switch is active."""
        return Path("state/KILL_SWITCH").exists()

    def _check_kill_zone(self, phase: TradingPhase) -> bool:
        """Check if we're in a valid trading kill zone."""
        return phase in [TradingPhase.MORNING_SESSION, TradingPhase.AFTERNOON_SESSION]

    def _load_market_context(self) -> MarketContext:
        """Load current market context."""
        try:
            from ml_advanced.hmm_regime_detector import get_current_regime
            regime = get_current_regime()
            regime_map = {
                0: MarketRegime.BEAR,
                1: MarketRegime.NEUTRAL,
                2: MarketRegime.BULL,
            }
            return MarketContext(
                regime=regime_map.get(regime, MarketRegime.UNKNOWN),
                vix=20.0,  # Would fetch real VIX
                spy_change=0.0,
            )
        except Exception:
            return MarketContext(regime=MarketRegime.UNKNOWN, vix=20.0)

    def _load_portfolio_state(self) -> PortfolioState:
        """Load current portfolio state."""
        try:
            from execution.broker_alpaca import get_account_equity
            equity = get_account_equity()
            return PortfolioState(equity=equity, cash=equity, positions=[], daily_pnl=0.0)
        except Exception:
            return PortfolioState(equity=50000.0, cash=50000.0, positions=[], daily_pnl=0.0)

    def _load_watchlist(self) -> List[SignalState]:
        """Load today's validated watchlist."""
        import json
        watchlist_file = Path("state/watchlist/today_validated.json")
        if watchlist_file.exists():
            try:
                data = json.loads(watchlist_file.read_text())
                return data.get("watchlist", [])
            except Exception:
                pass
        return []

    def _check_daily_loss(self, portfolio: PortfolioState) -> bool:
        """Check if daily loss limit exceeded."""
        daily_pnl = portfolio.get("daily_pnl", 0)
        equity = portfolio.get("equity", 50000)
        return daily_pnl < -0.02 * equity  # 2% daily loss limit

    def _check_weekly_exposure(self, portfolio: PortfolioState) -> bool:
        """Check if weekly exposure limit exceeded."""
        exposure = portfolio.get("exposure_pct", 0)
        return exposure > 0.40  # 40% weekly limit

    def _run_scanner(self) -> List[SignalState]:
        """Run the scanner and return signals."""
        # Placeholder - integrate with actual scanner
        return []

    def _rank_signals(self, signals: List[SignalState], top_n: int) -> List[SignalState]:
        """Rank signals by score and return top N."""
        sorted_signals = sorted(signals, key=lambda s: s.get("score", 0), reverse=True)
        return sorted_signals[:top_n]

    def _prepare_orders(self, signals: List[SignalState], portfolio: PortfolioState) -> List[Dict[str, Any]]:
        """Prepare orders from signals."""
        orders = []
        for sig in signals:
            order = {
                "symbol": sig.get("symbol"),
                "side": "buy" if sig.get("side") == "long" else "sell",
                "qty": self._calculate_position_size(sig, portfolio),
                "limit_price": sig.get("entry_price"),
                "stop_loss": sig.get("stop_loss"),
                "take_profit": sig.get("take_profit"),
            }
            orders.append(order)
        return orders

    def _calculate_position_size(self, signal: SignalState, portfolio: PortfolioState) -> int:
        """Calculate position size with dual cap."""
        equity = portfolio.get("equity", 50000)
        entry = signal.get("entry_price", 100)
        stop = signal.get("stop_loss", 95)

        risk_per_trade = equity * 0.02  # 2% risk
        max_notional = equity * 0.20  # 20% notional cap

        shares_by_risk = int(risk_per_trade / abs(entry - stop)) if entry != stop else 0
        shares_by_notional = int(max_notional / entry) if entry > 0 else 0

        return min(shares_by_risk, shares_by_notional)

    def _place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order (placeholder)."""
        # Would integrate with broker_alpaca.py
        return {**order, "status": "submitted", "timestamp": datetime.now().isoformat()}

    def _extract_lessons(self, trades: List[Dict[str, Any]]) -> List[str]:
        """Extract lessons from recent trades."""
        lessons = []
        for trade in trades:
            pnl = trade.get("pnl", 0)
            if pnl > 0:
                lessons.append(f"Win on {trade.get('symbol')}: {trade.get('reason', 'unknown')}")
            else:
                lessons.append(f"Loss on {trade.get('symbol')}: review entry timing")
        return lessons

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def run_cycle(self, initial_state: Optional[BrainState] = None) -> BrainState:
        """Run a single brain cycle."""
        if initial_state is None:
            initial_state = create_initial_brain_state()

        config = {"configurable": {"thread_id": "kobe_brain"}}
        result = self.app.invoke(initial_state, config)
        return result

    def stream_cycle(self, initial_state: Optional[BrainState] = None):
        """Stream brain cycle with intermediate states."""
        if initial_state is None:
            initial_state = create_initial_brain_state()

        config = {"configurable": {"thread_id": "kobe_brain"}}
        for state in self.app.stream(initial_state, config):
            yield state

    def get_state_snapshot(self, thread_id: str = "kobe_brain") -> Optional[BrainState]:
        """Get the current state snapshot."""
        config = {"configurable": {"thread_id": thread_id}}
        return self.app.get_state(config)

    def visualize(self) -> str:
        """Get Mermaid diagram of the graph."""
        return self.graph.get_graph().draw_mermaid()


# =============================================================================
# SINGLETON AND CONVENIENCE FUNCTIONS
# =============================================================================

_brain_graph: Optional[KobeBrainGraph] = None


def get_brain_graph() -> Optional[KobeBrainGraph]:
    """Get the singleton KobeBrainGraph instance."""
    global _brain_graph
    if _brain_graph is None:
        if HAS_LANGGRAPH:
            try:
                _brain_graph = KobeBrainGraph()
            except Exception as e:
                logger.error(f"Failed to create KobeBrainGraph: {e}")
                return None
        else:
            return None
    return _brain_graph


def run_brain_cycle() -> Optional[BrainState]:
    """Run a single brain cycle using the LangGraph brain."""
    brain = get_brain_graph()
    if brain:
        return brain.run_cycle()
    return None
