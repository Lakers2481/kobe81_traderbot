"""
LangGraph Coordinator for Trading Workflow

Implements a stateful workflow using LangGraph:
1. analyze_macro -> Check macro conditions
2. analyze_technical -> Get technical signals
3. check_memory -> Find similar past trades
4. validate_risk -> Enforce constraints
5. decide -> Make final decision
6. execute -> Execute trade (if approved)
7. reflect -> Learn from outcome

Author: Kobe Trading System
Created: 2026-01-04
"""

import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, TypedDict, Annotated, Literal
from enum import Enum

from core.structured_log import get_logger

logger = get_logger(__name__)

# Try to import LangGraph
_langgraph_available = False
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    _langgraph_available = True
except ImportError:
    logger.warning("LangGraph not installed. Install with: pip install langgraph")


class WorkflowState(Enum):
    """States in the trading workflow."""
    ANALYZE_MACRO = "analyze_macro"
    ANALYZE_TECHNICAL = "analyze_technical"
    CHECK_MEMORY = "check_memory"
    VALIDATE_RISK = "validate_risk"
    DECIDE = "decide"
    EXECUTE = "execute"
    REFLECT = "reflect"
    COMPLETE = "complete"
    BLOCKED = "blocked"


@dataclass
class TradingWorkflowState:
    """State object for the trading workflow."""

    # Input
    symbol: str = ""
    signal: Dict[str, Any] = field(default_factory=dict)
    positions: List[Dict[str, Any]] = field(default_factory=list)
    equity: float = 50000.0

    # Analysis results
    macro_analysis: Dict[str, Any] = field(default_factory=dict)
    technical_analysis: Dict[str, Any] = field(default_factory=dict)
    memory_analysis: Dict[str, Any] = field(default_factory=dict)
    risk_analysis: Dict[str, Any] = field(default_factory=dict)

    # Decision
    decision: str = "PENDING"
    confidence: float = 0.0
    reasoning: List[str] = field(default_factory=list)

    # Execution
    execution_result: Dict[str, Any] = field(default_factory=dict)
    trade_id: Optional[str] = None

    # State tracking
    current_state: str = "analyze_macro"
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal": self.signal,
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "current_state": self.current_state,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class TradingWorkflowCoordinator:
    """
    Coordinates the trading workflow using LangGraph-style state machine.

    Works with or without LangGraph installed - provides consistent interface.
    """

    def __init__(self, checkpoint_dir: str = "state/workflow"):
        """
        Initialize coordinator.

        Args:
            checkpoint_dir: Directory for workflow checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Import agents
        from agents.autogen_team import (
            MacroAnalystAgent,
            TechnicalAnalystAgent,
            RiskManagerAgent,
            MemoryKeeperAgent,
        )

        self.macro_agent = MacroAnalystAgent()
        self.technical_agent = TechnicalAnalystAgent()
        self.risk_agent = RiskManagerAgent()
        self.memory_agent = MemoryKeeperAgent()

        if _langgraph_available:
            self._build_graph()

    def _build_graph(self):
        """Build LangGraph state machine."""
        if not _langgraph_available:
            return

        # Define state type for LangGraph
        class GraphState(TypedDict):
            symbol: str
            signal: dict
            positions: list
            equity: float
            macro_analysis: dict
            technical_analysis: dict
            memory_analysis: dict
            risk_analysis: dict
            decision: str
            confidence: float
            reasoning: list
            current_state: str
            error: str

        # Create graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("analyze_macro", self._node_analyze_macro)
        workflow.add_node("analyze_technical", self._node_analyze_technical)
        workflow.add_node("check_memory", self._node_check_memory)
        workflow.add_node("validate_risk", self._node_validate_risk)
        workflow.add_node("decide", self._node_decide)
        workflow.add_node("execute", self._node_execute)
        workflow.add_node("reflect", self._node_reflect)

        # Add edges (linear flow with conditional branches)
        workflow.set_entry_point("analyze_macro")
        workflow.add_edge("analyze_macro", "analyze_technical")
        workflow.add_edge("analyze_technical", "check_memory")
        workflow.add_edge("check_memory", "validate_risk")

        # Conditional edge after risk validation
        workflow.add_conditional_edges(
            "validate_risk",
            self._should_proceed,
            {
                "proceed": "decide",
                "blocked": END,
            }
        )

        workflow.add_conditional_edges(
            "decide",
            self._should_execute,
            {
                "execute": "execute",
                "hold": END,
            }
        )

        workflow.add_edge("execute", "reflect")
        workflow.add_edge("reflect", END)

        # Compile with checkpoint
        self.graph = workflow.compile()

    def _node_analyze_macro(self, state: Dict) -> Dict:
        """Analyze macro conditions."""
        context = {"symbol": state.get("symbol"), "signal": state.get("signal")}
        result = self.macro_agent.analyze(context)
        return {"macro_analysis": result, "current_state": "analyze_macro"}

    def _node_analyze_technical(self, state: Dict) -> Dict:
        """Analyze technical conditions."""
        context = {"symbol": state.get("symbol"), "signal": state.get("signal")}
        result = self.technical_agent.analyze(context)
        return {"technical_analysis": result, "current_state": "analyze_technical"}

    def _node_check_memory(self, state: Dict) -> Dict:
        """Check similar past trades."""
        context = {"symbol": state.get("symbol"), "signal": state.get("signal")}
        result = self.memory_agent.analyze(context)
        return {"memory_analysis": result, "current_state": "check_memory"}

    def _node_validate_risk(self, state: Dict) -> Dict:
        """Validate against risk constraints."""
        context = {
            "symbol": state.get("symbol"),
            "signal": state.get("signal"),
            "positions": state.get("positions", []),
            "trade": {
                "symbol": state.get("symbol"),
                "weight": 0.10,  # Default position size
                "sector": state.get("signal", {}).get("sector", "Unknown"),
            }
        }
        result = self.risk_agent.analyze(context)
        return {"risk_analysis": result, "current_state": "validate_risk"}

    def _should_proceed(self, state: Dict) -> str:
        """Determine if we should proceed after risk check."""
        risk_analysis = state.get("risk_analysis", {})
        if risk_analysis.get("approved", False):
            return "proceed"
        return "blocked"

    def _node_decide(self, state: Dict) -> Dict:
        """Make final trading decision."""
        reasoning = []
        scores = {"bullish": 0, "bearish": 0, "neutral": 0}

        # Aggregate analysis
        macro = state.get("macro_analysis", {})
        if "BULLISH" in macro.get("recommendation", ""):
            scores["bullish"] += 1.5
            reasoning.append(f"Macro: {macro.get('reasoning', 'bullish environment')}")
        elif "BEARISH" in macro.get("recommendation", "") or "DEFENSIVE" in macro.get("recommendation", ""):
            scores["bearish"] += 1.5

        technical = state.get("technical_analysis", {})
        if "LONG" in technical.get("recommendation", "") or "BULLISH" in technical.get("recommendation", ""):
            scores["bullish"] += 2
            reasoning.append(f"Technical: {technical.get('reasoning', 'bullish signal')}")
        elif "SHORT" in technical.get("recommendation", "") or "BEARISH" in technical.get("recommendation", ""):
            scores["bearish"] += 2

        memory = state.get("memory_analysis", {})
        if memory.get("recommendation") == "BULLISH":
            scores["bullish"] += 1
            reasoning.append(f"Memory: {memory.get('reasoning', 'positive history')}")
        elif memory.get("recommendation") == "AVOID":
            scores["bearish"] += 2
            reasoning.append(f"Memory: {memory.get('reasoning', 'poor history')}")

        # Determine decision
        max_score = max(scores.values())
        decision = "HOLD"
        confidence = 0.5

        signal_side = state.get("signal", {}).get("side", "").upper()

        if scores["bullish"] >= 2.5 and signal_side == "LONG":
            decision = "BUY"
            confidence = min(0.9, 0.5 + scores["bullish"] * 0.1)
        elif scores["bearish"] >= 2.5:
            decision = "AVOID"
            confidence = min(0.9, 0.5 + scores["bearish"] * 0.1)
        elif scores["bullish"] >= 1.5 and signal_side == "LONG":
            decision = "BUY_SMALL"
            confidence = 0.6

        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "current_state": "decide"
        }

    def _should_execute(self, state: Dict) -> str:
        """Determine if we should execute the trade."""
        decision = state.get("decision", "HOLD")
        if decision in ["BUY", "SELL", "BUY_SMALL"]:
            return "execute"
        return "hold"

    def _node_execute(self, state: Dict) -> Dict:
        """Execute the trade (simulation only in this module)."""
        # In production, this would call the broker
        # Here we just record the intention
        execution_result = {
            "status": "SIMULATED",
            "symbol": state.get("symbol"),
            "side": "BUY" if state.get("decision") in ["BUY", "BUY_SMALL"] else "SELL",
            "timestamp": datetime.now().isoformat(),
            "note": "Execution handled by run_paper_trade.py",
        }

        trade_id = f"WF_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{state.get('symbol')}"

        return {
            "execution_result": execution_result,
            "trade_id": trade_id,
            "current_state": "execute"
        }

    def _node_reflect(self, state: Dict) -> Dict:
        """Learn from the trade (store in episodic memory)."""
        try:
            from cognitive.vector_memory import add_trade_episode

            episode = {
                "context": {
                    "symbol": state.get("symbol"),
                    "signal": state.get("signal"),
                    "macro_regime": state.get("macro_analysis", {}).get("analysis", {}).get("regime", {}),
                },
                "reasoning": state.get("reasoning", []),
                "outcome": {
                    "decision": state.get("decision"),
                    "confidence": state.get("confidence"),
                    # P&L will be updated later when trade closes
                }
            }

            episode_id = add_trade_episode(episode)
            logger.info(f"Stored trade episode: {episode_id}")

        except Exception as e:
            logger.warning(f"Failed to store episode: {e}")

        return {"current_state": "reflect"}

    def run_workflow(
        self,
        symbol: str,
        signal: Dict[str, Any],
        positions: Optional[List[Dict[str, Any]]] = None,
        equity: float = 50000.0
    ) -> TradingWorkflowState:
        """
        Run the complete trading workflow.

        Args:
            symbol: Stock symbol
            signal: Trading signal dict
            positions: Current positions
            equity: Account equity

        Returns:
            TradingWorkflowState with results
        """
        state = TradingWorkflowState(
            symbol=symbol,
            signal=signal,
            positions=positions or [],
            equity=equity,
        )

        try:
            # Run each step
            context = {"symbol": symbol, "signal": signal, "positions": positions or []}

            # Step 1: Macro analysis
            state.current_state = "analyze_macro"
            state.macro_analysis = self.macro_agent.analyze(context)

            # Step 2: Technical analysis
            state.current_state = "analyze_technical"
            state.technical_analysis = self.technical_agent.analyze(context)

            # Step 3: Memory check
            state.current_state = "check_memory"
            state.memory_analysis = self.memory_agent.analyze(context)

            # Step 4: Risk validation
            state.current_state = "validate_risk"
            risk_context = {
                **context,
                "trade": {
                    "symbol": symbol,
                    "weight": 0.10,
                    "sector": signal.get("sector", "Unknown"),
                }
            }
            state.risk_analysis = self.risk_agent.analyze(risk_context)

            # Check if blocked by risk
            if not state.risk_analysis.get("approved", False):
                state.decision = "BLOCKED_BY_RISK"
                state.confidence = 0.9
                state.reasoning = state.risk_analysis.get("violations", [])
                state.current_state = "blocked"
                return state

            # Step 5: Decide
            state.current_state = "decide"
            graph_state = {
                "symbol": symbol,
                "signal": signal,
                "macro_analysis": state.macro_analysis,
                "technical_analysis": state.technical_analysis,
                "memory_analysis": state.memory_analysis,
                "risk_analysis": state.risk_analysis,
            }
            decision_result = self._node_decide(graph_state)
            state.decision = decision_result["decision"]
            state.confidence = decision_result["confidence"]
            state.reasoning = decision_result["reasoning"]

            # Step 6: Execute (if applicable)
            if state.decision in ["BUY", "SELL", "BUY_SMALL"]:
                state.current_state = "execute"
                exec_result = self._node_execute({**graph_state, "decision": state.decision})
                state.execution_result = exec_result["execution_result"]
                state.trade_id = exec_result["trade_id"]

                # Step 7: Reflect
                state.current_state = "reflect"
                self._node_reflect({
                    "symbol": symbol,
                    "signal": signal,
                    "reasoning": state.reasoning,
                    "decision": state.decision,
                    "confidence": state.confidence,
                    "macro_analysis": state.macro_analysis,
                })

            state.current_state = "complete"

        except Exception as e:
            logger.error(f"Workflow error: {e}")
            state.error = str(e)
            state.decision = "ERROR"

        return state

    def save_checkpoint(self, state: TradingWorkflowState, checkpoint_id: str):
        """Save workflow checkpoint."""
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(state.to_dict(), f, indent=2, default=str)

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict]:
        """Load workflow checkpoint."""
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file) as f:
                return json.load(f)
        return None


# Singleton
_coordinator: Optional[TradingWorkflowCoordinator] = None


def get_workflow_coordinator() -> TradingWorkflowCoordinator:
    """Get or create workflow coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = TradingWorkflowCoordinator()
    return _coordinator


def run_trading_workflow(
    symbol: str,
    signal: Dict[str, Any],
    positions: Optional[List[Dict[str, Any]]] = None,
    equity: float = 50000.0
) -> TradingWorkflowState:
    """Run the trading workflow for a signal."""
    return get_workflow_coordinator().run_workflow(symbol, signal, positions, equity)


if __name__ == "__main__":
    # Demo
    print("=== LangGraph Trading Workflow Demo ===\n")

    coordinator = TradingWorkflowCoordinator()

    # Sample signal
    signal = {
        "side": "LONG",
        "entry_price": 178.50,
        "stop_loss": 175.00,
        "take_profit": 185.00,
        "strategy": "IBS_RSI",
        "score": 72,
        "reason": "IBS < 0.08, RSI(2) < 5",
        "sector": "Technology",
    }

    print("Running workflow for AAPL...")
    result = coordinator.run_workflow(
        symbol="AAPL",
        signal=signal,
        positions=[],
        equity=50000
    )

    print(f"\nWorkflow State: {result.current_state}")
    print(f"Decision: {result.decision}")
    print(f"Confidence: {result.confidence:.0%}")

    if result.reasoning:
        print("\nReasoning:")
        for r in result.reasoning:
            print(f"  - {r}")

    if result.error:
        print(f"\nError: {result.error}")

    if result.trade_id:
        print(f"\nTrade ID: {result.trade_id}")
