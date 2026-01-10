"""
LangGraph State Definitions for Kobe's Cognitive Brain.

Defines typed state objects for the formal state machine architecture.
Uses TypedDict for type safety and clear state transitions.

Created: 2026-01-07
Based on: LangGraph patterns for stateful agent graphs
"""

from __future__ import annotations

from typing import TypedDict, Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class MarketRegime(str, Enum):
    """Market regime states from HMM detector."""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class TradingPhase(str, Enum):
    """Time-based trading phases."""
    PRE_MARKET = "pre_market"
    OPENING_RANGE = "opening_range"  # 9:30-10:00 - NO TRADING
    MORNING_SESSION = "morning_session"  # 10:00-11:30 - PRIMARY
    LUNCH = "lunch"  # 11:30-14:00 - NO TRADING
    AFTERNOON_SESSION = "afternoon_session"  # 14:00-15:30 - SECONDARY
    CLOSE = "close"  # 15:30-16:00 - MANAGE ONLY
    AFTER_HOURS = "after_hours"
    WEEKEND = "weekend"


class DecisionType(str, Enum):
    """Types of decisions the brain can make."""
    SCAN = "scan"
    TRADE = "trade"
    HOLD = "hold"
    EXIT = "exit"
    RESEARCH = "research"
    MAINTENANCE = "maintenance"
    STANDBY = "standby"


class SignalQuality(str, Enum):
    """Signal quality tiers."""
    ELITE = "elite"  # Score >= 85, auto-pass criteria met
    HIGH = "high"  # Score >= 75
    MEDIUM = "medium"  # Score >= 65
    LOW = "low"  # Score < 65
    REJECTED = "rejected"  # Failed quality gate


# =============================================================================
# STATE DEFINITIONS
# =============================================================================

class MarketContext(TypedDict, total=False):
    """Current market context state."""
    timestamp: str
    regime: MarketRegime
    vix: float
    spy_change: float
    sector_rotation: Dict[str, float]
    news_sentiment: float
    is_fomc_day: bool
    is_opex: bool
    is_earnings_heavy: bool


class SignalState(TypedDict, total=False):
    """State for a trading signal."""
    symbol: str
    strategy: str
    side: Literal["long", "short"]
    entry_price: float
    stop_loss: float
    take_profit: float
    score: float
    confidence: float
    quality: SignalQuality
    reason: str
    historical_pattern: Optional[Dict[str, Any]]
    markov_boost: float
    regime_alignment: bool


class PortfolioState(TypedDict, total=False):
    """Current portfolio state."""
    equity: float
    cash: float
    positions: List[Dict[str, Any]]
    daily_pnl: float
    weekly_pnl: float
    open_risk: float
    exposure_pct: float
    sector_exposure: Dict[str, float]


class BrainState(TypedDict, total=False):
    """
    Main state object for the LangGraph brain.

    This is the central state that flows through all nodes in the graph.
    """
    # Timing
    timestamp: str
    trading_phase: TradingPhase

    # Market context
    market: MarketContext

    # Signals
    watchlist: List[SignalState]
    current_signals: List[SignalState]
    top_5: List[SignalState]
    tradeable: List[SignalState]  # Top 2 to actually trade

    # Portfolio
    portfolio: PortfolioState

    # Decision state
    decision: DecisionType
    decision_reason: str
    confidence: float

    # Risk gates
    kill_switch_active: bool
    daily_loss_exceeded: bool
    weekly_exposure_exceeded: bool
    in_valid_kill_zone: bool

    # Memory references
    recent_trades: List[Dict[str, Any]]
    episodic_context: List[str]  # IDs of relevant episodes
    semantic_rules: List[str]  # Active rules from semantic memory

    # Research state
    pending_hypotheses: int
    validated_edges: int
    alpha_discoveries: int

    # Execution
    orders_to_place: List[Dict[str, Any]]
    orders_placed: List[Dict[str, Any]]

    # Reflection
    last_reflection: str
    lessons_learned: List[str]

    # Meta
    iteration: int
    errors: List[str]


class ResearchState(TypedDict, total=False):
    """State for research/alpha mining operations."""
    phase: Literal["idle", "mining", "validating", "integrating"]
    alphas_tested: int
    alphas_validated: int
    top_performers: List[Dict[str, Any]]
    hypotheses_generated: int
    experiments_running: int
    last_mining_run: str


class ReflectionState(TypedDict, total=False):
    """State for reflection and learning."""
    trade_outcome: Literal["win", "loss", "scratch"]
    pnl_amount: float
    what_worked: List[str]
    what_failed: List[str]
    lessons: List[str]
    confidence_adjustment: float
    rule_updates: List[Dict[str, Any]]


# =============================================================================
# STATE FACTORIES
# =============================================================================

def create_initial_brain_state() -> BrainState:
    """Create a fresh brain state with defaults."""
    return BrainState(
        timestamp=datetime.now().isoformat(),
        trading_phase=TradingPhase.PRE_MARKET,
        market=MarketContext(
            regime=MarketRegime.UNKNOWN,
            vix=20.0,
            spy_change=0.0,
        ),
        watchlist=[],
        current_signals=[],
        top_5=[],
        tradeable=[],
        portfolio=PortfolioState(
            equity=0.0,
            cash=0.0,
            positions=[],
            daily_pnl=0.0,
        ),
        decision=DecisionType.STANDBY,
        decision_reason="Initializing",
        confidence=0.5,
        kill_switch_active=False,
        daily_loss_exceeded=False,
        weekly_exposure_exceeded=False,
        in_valid_kill_zone=False,
        recent_trades=[],
        episodic_context=[],
        semantic_rules=[],
        pending_hypotheses=0,
        validated_edges=0,
        alpha_discoveries=0,
        orders_to_place=[],
        orders_placed=[],
        last_reflection="",
        lessons_learned=[],
        iteration=0,
        errors=[],
    )


def create_research_state() -> ResearchState:
    """Create a fresh research state."""
    return ResearchState(
        phase="idle",
        alphas_tested=0,
        alphas_validated=0,
        top_performers=[],
        hypotheses_generated=0,
        experiments_running=0,
        last_mining_run="",
    )
