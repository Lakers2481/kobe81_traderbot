"""
AutoGen Multi-Agent Team for Trading Decisions

Specialized agents that collaborate on trading analysis:
- MacroAnalyst: Interprets FRED, Treasury, COT data
- TechnicalAnalyst: Price patterns, indicators
- RiskManager: Enforces constraints via OR-Tools
- MemoryKeeper: FAISS vector search for similar trades

Author: Kobe Trading System
Created: 2026-01-04
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

from core.structured_log import get_logger

logger = get_logger(__name__)

# Try to import AutoGen
_autogen_available = False
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    _autogen_available = True
except ImportError:
    logger.warning("AutoGen not installed. Install with: pip install pyautogen")


class BaseTradingAgent:
    """Base class for trading agents (fallback if AutoGen not available)."""

    def __init__(self, name: str, system_message: str):
        self.name = name
        self.system_message = system_message
        self.messages: List[Dict[str, str]] = []

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Override in subclass."""
        raise NotImplementedError

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})


class MacroAnalystAgent(BaseTradingAgent):
    """
    Analyzes macro economic conditions.

    Uses:
    - FRED data (rates, inflation, GDP)
    - Treasury yield curves
    - CFTC COT positioning
    """

    SYSTEM_MESSAGE = """You are a macro economist analyzing market conditions.
    Your job is to assess:
    1. Interest rate environment (Fed policy, yield curve)
    2. Inflation trends (CPI, PCE, breakeven rates)
    3. Positioning data (COT, speculator sentiment)
    4. Risk regime (risk-on vs risk-off)

    Provide clear, actionable insights for trading decisions.
    Always cite specific data points and their implications.
    """

    def __init__(self):
        super().__init__("MacroAnalyst", self.SYSTEM_MESSAGE)

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze macro conditions."""
        result = {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "analysis": {},
            "recommendation": "NEUTRAL",
            "confidence": 0.5,
        }

        try:
            from data.providers.fred_macro import get_macro_regime, get_yield_curve_slope
            from data.providers.cftc_cot import get_market_sentiment

            # Get macro regime
            regime = get_macro_regime()
            result["analysis"]["regime"] = regime

            # Yield curve
            slope = get_yield_curve_slope()
            result["analysis"]["yield_curve"] = slope

            # COT sentiment
            cot = get_market_sentiment()
            result["analysis"]["cot_sentiment"] = cot.get("overall", "NEUTRAL")

            # Determine recommendation
            signals = regime.get("signals", [])
            if "INVERTED_CURVE" in signals or slope.get("is_inverted"):
                result["recommendation"] = "DEFENSIVE"
                result["confidence"] = 0.7
                result["reasoning"] = "Yield curve inversion signals recession risk"
            elif regime.get("regime") == "RISK_ON":
                result["recommendation"] = "BULLISH"
                result["confidence"] = 0.6
            elif regime.get("regime") == "RISK_OFF":
                result["recommendation"] = "BEARISH"
                result["confidence"] = 0.6

        except Exception as e:
            logger.warning(f"MacroAnalyst error: {e}")
            result["error"] = str(e)

        return result


class TechnicalAnalystAgent(BaseTradingAgent):
    """
    Analyzes price patterns and technical indicators.

    Uses:
    - OHLCV data
    - Technical indicators (RSI, MACD, ATR, etc.)
    - Pattern recognition
    """

    SYSTEM_MESSAGE = """You are a technical analyst specializing in price action.
    Your job is to assess:
    1. Trend direction and strength
    2. Support and resistance levels
    3. Momentum indicators (RSI, MACD)
    4. Volume analysis
    5. Pattern recognition

    Be specific about price levels and provide clear trade setups.
    """

    def __init__(self):
        super().__init__("TechnicalAnalyst", self.SYSTEM_MESSAGE)

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze technical conditions for a symbol."""
        result = {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "analysis": {},
            "recommendation": "NEUTRAL",
            "confidence": 0.5,
        }

        symbol = context.get("symbol")
        signal = context.get("signal", {})

        if signal:
            result["analysis"]["signal"] = {
                "side": signal.get("side"),
                "entry_price": signal.get("entry_price"),
                "stop_loss": signal.get("stop_loss"),
                "take_profit": signal.get("take_profit"),
                "strategy": signal.get("strategy"),
            }

            # Assess signal quality
            if signal.get("score", 0) >= 70:
                result["recommendation"] = "STRONG_" + signal.get("side", "").upper()
                result["confidence"] = 0.7
            elif signal.get("score", 0) >= 50:
                result["recommendation"] = signal.get("side", "").upper()
                result["confidence"] = 0.6

            result["reasoning"] = signal.get("reason", "Technical signal generated")

        return result


class RiskManagerAgent(BaseTradingAgent):
    """
    Enforces risk constraints using OR-Tools.

    Uses:
    - Portfolio optimizer
    - Position sizing rules
    - Correlation limits
    """

    SYSTEM_MESSAGE = """You are a risk manager enforcing portfolio constraints.
    Your job is to:
    1. Validate position sizes (max 20% per position)
    2. Check sector exposure (max 30% per sector)
    3. Enforce correlation limits
    4. Verify capital allocation rules
    5. Block trades that violate risk policy

    Never approve trades that violate constraints. Capital preservation is priority.
    """

    def __init__(self):
        super().__init__("RiskManager", self.SYSTEM_MESSAGE)

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade against risk constraints."""
        result = {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "analysis": {},
            "approved": True,
            "violations": [],
            "warnings": [],
        }

        try:
            from risk.advanced.portfolio_optimizer import PortfolioOptimizer, PortfolioConstraints, Position

            optimizer = PortfolioOptimizer()

            # Check if this is a new position validation
            trade = context.get("trade", {})
            positions = context.get("positions", [])

            if trade:
                # Simulate adding this trade
                symbol = trade.get("symbol")
                proposed_weight = trade.get("weight", 0.1)

                # Check position size
                if proposed_weight > 0.20:
                    result["violations"].append(f"Position size {proposed_weight:.1%} exceeds 20% limit")
                    result["approved"] = False

                # Check sector exposure
                sector = trade.get("sector", "Unknown")
                sector_weight = sum(p.get("weight", 0) for p in positions if p.get("sector") == sector)
                if sector_weight + proposed_weight > 0.30:
                    result["violations"].append(f"Sector {sector} exposure would exceed 30%")
                    result["approved"] = False

            # Validate overall portfolio
            if positions:
                pos_objs = [
                    Position(p["symbol"], p.get("weight", 0), p.get("sector", "Unknown"), p.get("beta", 1.0))
                    for p in positions
                ]
                validation = optimizer.validate_portfolio(pos_objs)
                result["analysis"]["portfolio_validation"] = validation

                if not validation["valid"]:
                    result["violations"].extend(validation["violations"])
                    result["approved"] = False

                result["warnings"].extend(validation.get("warnings", []))

            result["recommendation"] = "APPROVED" if result["approved"] else "REJECTED"

        except Exception as e:
            logger.warning(f"RiskManager error: {e}")
            result["error"] = str(e)
            result["approved"] = False
            result["violations"].append(f"Risk check error: {e}")

        return result


class MemoryKeeperAgent(BaseTradingAgent):
    """
    Searches FAISS vector memory for similar past trades.

    Uses:
    - FAISS vector index
    - Episode embeddings
    - Win/loss tracking
    """

    SYSTEM_MESSAGE = """You are a memory keeper tracking trading history.
    Your job is to:
    1. Find similar past trades using vector similarity
    2. Report win rates for similar situations
    3. Identify patterns from historical trades
    4. Warn about past mistakes in similar setups

    Base recommendations on statistical evidence from past trades.
    """

    def __init__(self):
        super().__init__("MemoryKeeper", self.SYSTEM_MESSAGE)

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Find similar past trades."""
        result = {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "analysis": {},
            "similar_trades": [],
            "recommendation": "NEUTRAL",
        }

        try:
            from cognitive.vector_memory import get_vector_memory

            memory = get_vector_memory()

            # Build query from context
            signal = context.get("signal", {})
            symbol = context.get("symbol", "")

            query_parts = []
            if symbol:
                query_parts.append(f"Symbol: {symbol}")
            if signal.get("strategy"):
                query_parts.append(f"Strategy: {signal['strategy']}")
            if signal.get("side"):
                query_parts.append(f"Side: {signal['side']}")
            if signal.get("reason"):
                query_parts.append(f"Reason: {signal['reason']}")

            query = " | ".join(query_parts) if query_parts else "trading signal"

            # Search for similar trades
            stats = memory.get_win_rate_for_similar(query, k=10)

            result["analysis"]["memory_search"] = stats
            result["similar_trades"] = stats.get("similar_episodes", [])

            # Make recommendation based on history
            win_rate = stats.get("win_rate")
            sample_size = stats.get("sample_size", 0)

            if win_rate is not None and sample_size >= 5:
                if win_rate >= 0.70:
                    result["recommendation"] = "BULLISH"
                    result["reasoning"] = f"Similar trades had {win_rate:.0%} win rate (n={sample_size})"
                elif win_rate <= 0.30:
                    result["recommendation"] = "AVOID"
                    result["reasoning"] = f"Similar trades had only {win_rate:.0%} win rate (n={sample_size})"
                else:
                    result["recommendation"] = "NEUTRAL"
                    result["reasoning"] = f"Mixed results: {win_rate:.0%} win rate (n={sample_size})"
            else:
                result["recommendation"] = "INSUFFICIENT_DATA"
                result["reasoning"] = f"Only {sample_size} similar trades found"

        except Exception as e:
            logger.warning(f"MemoryKeeper error: {e}")
            result["error"] = str(e)

        return result


class TradingTeam:
    """
    Orchestrates the multi-agent trading team.

    Coordinates:
    - MacroAnalyst
    - TechnicalAnalyst
    - RiskManager
    - MemoryKeeper

    Makes final trading decisions based on agent consensus.
    """

    def __init__(self, use_autogen: bool = False):
        """
        Initialize trading team.

        Args:
            use_autogen: Use AutoGen framework (requires API keys)
        """
        self.use_autogen = use_autogen and _autogen_available

        # Initialize agents
        self.macro_analyst = MacroAnalystAgent()
        self.technical_analyst = TechnicalAnalystAgent()
        self.risk_manager = RiskManagerAgent()
        self.memory_keeper = MemoryKeeperAgent()

        if self.use_autogen:
            self._setup_autogen_team()

    def _setup_autogen_team(self):
        """Set up AutoGen group chat."""
        config_list = [
            {
                "model": os.getenv("AUTOGEN_MODEL", "gpt-4"),
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        ]

        llm_config = {"config_list": config_list, "temperature": 0}

        # Create AutoGen agents
        self.autogen_agents = {}

        if _autogen_available:
            self.autogen_agents["macro"] = AssistantAgent(
                name="MacroAnalyst",
                system_message=MacroAnalystAgent.SYSTEM_MESSAGE,
                llm_config=llm_config,
            )
            self.autogen_agents["technical"] = AssistantAgent(
                name="TechnicalAnalyst",
                system_message=TechnicalAnalystAgent.SYSTEM_MESSAGE,
                llm_config=llm_config,
            )
            self.autogen_agents["risk"] = AssistantAgent(
                name="RiskManager",
                system_message=RiskManagerAgent.SYSTEM_MESSAGE,
                llm_config=llm_config,
            )
            self.autogen_agents["memory"] = AssistantAgent(
                name="MemoryKeeper",
                system_message=MemoryKeeperAgent.SYSTEM_MESSAGE,
                llm_config=llm_config,
            )

            # Group chat
            self.group_chat = GroupChat(
                agents=list(self.autogen_agents.values()),
                messages=[],
                max_round=10,
            )
            self.manager = GroupChatManager(groupchat=self.group_chat)

    def analyze_trade(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full team analysis on a trade opportunity.

        Args:
            context: Trade context including signal, symbol, positions

        Returns:
            Dict with team decision and supporting analysis
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "agent_analyses": {},
            "final_decision": "HOLD",
            "confidence": 0.5,
            "reasoning": [],
        }

        # Run each agent
        result["agent_analyses"]["macro"] = self.macro_analyst.analyze(context)
        result["agent_analyses"]["technical"] = self.technical_analyst.analyze(context)
        result["agent_analyses"]["risk"] = self.risk_manager.analyze(context)
        result["agent_analyses"]["memory"] = self.memory_keeper.analyze(context)

        # Aggregate recommendations
        recommendations = {
            "BULLISH": 0,
            "BEARISH": 0,
            "NEUTRAL": 0,
            "AVOID": 0,
        }

        # Macro weight
        macro_rec = result["agent_analyses"]["macro"].get("recommendation", "NEUTRAL")
        if "BULLISH" in macro_rec:
            recommendations["BULLISH"] += 1.5
        elif "BEARISH" in macro_rec or "DEFENSIVE" in macro_rec:
            recommendations["BEARISH"] += 1.5

        # Technical weight
        tech_rec = result["agent_analyses"]["technical"].get("recommendation", "NEUTRAL")
        if "BULLISH" in tech_rec or "LONG" in tech_rec:
            recommendations["BULLISH"] += 2
        elif "BEARISH" in tech_rec or "SHORT" in tech_rec:
            recommendations["BEARISH"] += 2

        # Memory weight
        mem_rec = result["agent_analyses"]["memory"].get("recommendation", "NEUTRAL")
        if mem_rec == "BULLISH":
            recommendations["BULLISH"] += 1
        elif mem_rec == "AVOID":
            recommendations["AVOID"] += 2

        # Risk is a gate
        risk_approved = result["agent_analyses"]["risk"].get("approved", False)
        if not risk_approved:
            result["final_decision"] = "BLOCKED_BY_RISK"
            result["confidence"] = 0.9
            result["reasoning"].append("Risk manager blocked trade")
            result["reasoning"].extend(result["agent_analyses"]["risk"].get("violations", []))
            return result

        # Final decision
        max_rec = max(recommendations, key=recommendations.get)
        max_score = recommendations[max_rec]

        if max_rec == "AVOID" and max_score >= 2:
            result["final_decision"] = "AVOID"
            result["confidence"] = 0.7
        elif max_rec == "BULLISH" and max_score >= 2.5:
            result["final_decision"] = "BUY"
            result["confidence"] = min(0.9, 0.5 + max_score * 0.1)
        elif max_rec == "BEARISH" and max_score >= 2.5:
            result["final_decision"] = "SELL"
            result["confidence"] = min(0.9, 0.5 + max_score * 0.1)
        else:
            result["final_decision"] = "HOLD"
            result["confidence"] = 0.5

        # Build reasoning
        for agent_name, analysis in result["agent_analyses"].items():
            if analysis.get("reasoning"):
                result["reasoning"].append(f"{agent_name}: {analysis['reasoning']}")

        return result


# Singleton
_team: Optional[TradingTeam] = None


def get_trading_team(use_autogen: bool = False) -> TradingTeam:
    """Get or create trading team."""
    global _team
    if _team is None:
        _team = TradingTeam(use_autogen=use_autogen)
    return _team


def analyze_trade(context: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to analyze a trade."""
    return get_trading_team().analyze_trade(context)


if __name__ == "__main__":
    # Demo
    print("=== AutoGen Trading Team Demo ===\n")

    team = TradingTeam(use_autogen=False)  # Local mode

    # Sample context
    context = {
        "symbol": "AAPL",
        "signal": {
            "side": "LONG",
            "entry_price": 178.50,
            "stop_loss": 175.00,
            "take_profit": 185.00,
            "strategy": "IBS_RSI",
            "score": 72,
            "reason": "IBS < 0.08, RSI(2) < 5",
        },
        "positions": [],
        "trade": {
            "symbol": "AAPL",
            "weight": 0.15,
            "sector": "Technology",
        }
    }

    print("Analyzing trade...")
    result = team.analyze_trade(context)

    print(f"\nFinal Decision: {result['final_decision']}")
    print(f"Confidence: {result['confidence']:.0%}")

    print("\nReasoning:")
    for reason in result["reasoning"]:
        print(f"  - {reason}")

    print("\nAgent Analyses:")
    for agent, analysis in result["agent_analyses"].items():
        print(f"  {agent}: {analysis.get('recommendation', 'N/A')}")
