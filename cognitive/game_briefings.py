"""
Game Briefings Engine - PRE_GAME, HALF_TIME, POST_GAME Comprehensive Analysis
==============================================================================

Provides unified LLM/ML/AI-powered briefings for all phases of the trading day:

PRE_GAME (08:00 ET):
  - Morning game plan with regime analysis
  - Market mood interpretation
  - Top-3 picks with narratives
  - Trade of the Day deep dive
  - Step-by-step action plan
  - Position sizing recommendations

HALF_TIME (12:00 ET):
  - Position-by-position analysis
  - What's working / what's not
  - Regime change detection
  - Midday adjustments
  - Remaining opportunities

POST_GAME (16:00 ET):
  - Full day performance summary
  - Trade-by-trade analysis
  - Lessons learned
  - Hypothesis generation
  - Next day setup

Integration Points:
  - HMM Regime Detector (probabilistic regime)
  - Market Mood Analyzer (VIX + sentiment)
  - LLM Trade Analyzer (Claude narratives)
  - News Processor (sentiment data)
  - Portfolio Heat Monitor (risk exposure)
  - Reflection Engine (lesson extraction)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# Anthropic for LLM calls
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BriefingContext:
    """Unified context for all briefing phases."""
    timestamp: datetime
    date: str  # YYYY-MM-DD

    # Market Regime
    regime: str = "NEUTRAL"  # BULLISH, NEUTRAL, BEARISH
    regime_confidence: float = 0.0
    regime_probabilities: Dict[str, float] = field(default_factory=dict)
    days_in_regime: int = 0

    # Market Mood
    mood_state: str = "Neutral"
    mood_score: float = 0.0
    vix_level: float = 20.0
    is_mood_extreme: bool = False

    # News & Sentiment
    news_articles: List[Dict] = field(default_factory=list)
    sentiment_compound: float = 0.0
    news_interpretation: str = ""
    key_themes: List[str] = field(default_factory=list)

    # Portfolio State
    positions: List[Dict] = field(default_factory=list)
    total_positions: int = 0
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0

    # Heat Status
    heat_level: str = "NORMAL"
    heat_score: float = 0.0
    heat_alerts: List[str] = field(default_factory=list)

    # SPY Reference
    spy_price: float = 0.0
    spy_sma200: float = 0.0
    spy_position: str = "ABOVE"  # ABOVE or BELOW SMA200

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class PositionStatus:
    """Analysis of a single position."""
    symbol: str
    side: str
    entry_price: float
    current_price: float
    shares: int
    entry_date: str
    days_held: int

    # P&L
    unrealized_pnl: float = 0.0
    pnl_percent: float = 0.0

    # Stop/Target
    stop_loss: float = 0.0
    take_profit: Optional[float] = None
    distance_to_stop_pct: float = 0.0
    distance_to_target_pct: Optional[float] = None

    # Analysis
    analysis: str = ""  # LLM narrative
    recommendation: str = "HOLD"  # HOLD | ADJUST_STOP | TAKE_PROFIT | EXIT
    new_stop_level: Optional[float] = None
    key_level: float = 0.0  # Price that changes recommendation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreGameBriefing:
    """Morning game plan briefing."""
    context: BriefingContext

    # Analysis Sections
    regime_analysis: str = ""  # LLM: regime probabilities, outlook
    market_mood_description: str = ""  # LLM: emotional state interpretation
    overnight_news_summary: str = ""  # LLM: key themes, risks
    trading_bias: str = "NEUTRAL"  # AGGRESSIVE, NEUTRAL, DEFENSIVE

    # Top-3 Picks
    top3_picks: List[Dict] = field(default_factory=list)
    top3_narratives: List[Dict] = field(default_factory=list)

    # Trade of the Day
    totd: Optional[Dict] = None
    totd_deep_analysis: str = ""

    # Action Plan
    action_steps: List[str] = field(default_factory=list)
    pre_market_prep: List[str] = field(default_factory=list)
    entry_timing: List[str] = field(default_factory=list)
    execution_approach: List[str] = field(default_factory=list)

    # Position Sizing
    position_sizing_recs: Dict[str, Dict] = field(default_factory=dict)

    # Key Levels & Events
    key_levels: Dict[str, float] = field(default_factory=dict)
    macro_events: List[Dict] = field(default_factory=list)

    # Warnings
    risk_warnings: List[str] = field(default_factory=list)
    stand_down_conditions: List[str] = field(default_factory=list)

    # Metadata
    generated_at: str = ""
    llm_model: str = ""
    generation_method: str = "deterministic"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['context'] = self.context.to_dict()
        return d


@dataclass
class HalfTimeBriefing:
    """Midday status check briefing."""
    context: BriefingContext

    # Position Analysis
    position_analysis: List[PositionStatus] = field(default_factory=list)
    positions_summary: str = ""  # LLM overview

    # Performance Assessment
    whats_working: List[str] = field(default_factory=list)
    whats_not_working: List[str] = field(default_factory=list)

    # Regime Changes
    morning_regime: str = ""
    current_regime: str = ""
    regime_changed: bool = False
    regime_change_narrative: str = ""

    # Morning vs Now
    morning_vix: float = 0.0
    current_vix: float = 0.0
    morning_plan_picks: List[str] = field(default_factory=list)
    morning_totd: str = ""

    # News Updates
    midday_news: List[Dict] = field(default_factory=list)
    market_impact_analysis: str = ""

    # Adjustments
    adjustments_recommended: List[Dict] = field(default_factory=list)
    # Each: {'symbol': str, 'action': str, 'reason': str, 'new_level': float}

    # Afternoon Plan
    afternoon_game_plan: str = ""
    remaining_opportunities: List[Dict] = field(default_factory=list)
    priority_actions: List[str] = field(default_factory=list)
    exposure_check: str = ""

    # Metadata
    generated_at: str = ""
    llm_model: str = ""
    generation_method: str = "deterministic"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['context'] = self.context.to_dict()
        d['position_analysis'] = [p.to_dict() for p in self.position_analysis]
        return d


@dataclass
class TradeOutcome:
    """Single trade outcome for POST_GAME analysis."""
    symbol: str
    strategy: str
    side: str
    entry_price: float
    exit_price: float
    shares: int
    entry_time: str
    exit_time: str
    hold_bars: int
    pnl_dollars: float
    pnl_percent: float
    exit_reason: str  # STOP | TARGET | TIME | MANUAL
    was_winner: bool
    analysis: str = ""  # LLM analysis of this trade

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PostGameBriefing:
    """End of day full analysis briefing."""
    context: BriefingContext

    # Day Summary
    executive_summary: str = ""
    day_summary: Dict = field(default_factory=dict)
    # total_trades, wins, losses, win_rate, realized_pnl, best_trade, worst_trade

    # Trade-by-Trade
    trades_today: List[TradeOutcome] = field(default_factory=list)
    trade_narratives: List[str] = field(default_factory=list)

    # What Went Right/Wrong
    what_went_right: List[str] = field(default_factory=list)
    what_went_wrong: List[str] = field(default_factory=list)
    missed_opportunities: List[str] = field(default_factory=list)

    # Regime Analysis
    morning_regime: str = ""
    midday_regime: str = ""
    eod_regime: str = ""
    regime_behavior: str = ""  # LLM: How regime evolved
    regime_read_accuracy: str = ""

    # News Impact
    news_impact_analysis: str = ""
    major_news_events: List[Dict] = field(default_factory=list)

    # Strategy Breakdown
    strategy_breakdown: Dict[str, Dict] = field(default_factory=dict)
    # {strategy: {trades, wins, losses, win_rate, pnl}}
    best_performing_strategy: str = ""
    worst_performing_strategy: str = ""
    strategy_adjustments: List[str] = field(default_factory=list)

    # Reflection & Learning
    reflection_summary: str = ""
    lessons_learned: List[str] = field(default_factory=list)
    behavior_changes: List[str] = field(default_factory=list)

    # Hypotheses
    hypotheses_generated: List[Dict] = field(default_factory=list)
    # {description, condition, prediction, rationale, confidence}

    # Next Day Setup
    overnight_risks: List[str] = field(default_factory=list)
    gap_risk_assessment: str = ""
    next_day_bias: str = ""  # BULLISH, NEUTRAL, BEARISH
    next_day_setup: str = ""  # LLM: Tomorrow's outlook
    key_news_to_watch: List[str] = field(default_factory=list)

    # Confidence Tracking
    prediction_accuracy: Dict = field(default_factory=dict)
    # morning_predictions vs actual outcomes

    # Metadata
    generated_at: str = ""
    llm_model: str = ""
    generation_method: str = "deterministic"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['context'] = self.context.to_dict()
        d['trades_today'] = [t.to_dict() for t in self.trades_today]
        return d


# =============================================================================
# LLM Prompt Templates
# =============================================================================

PREGAME_SYSTEM_PROMPT = """You are Kobe, a senior quantitative trading analyst preparing the morning game plan.
Your briefing must be:
1. Data-backed - cite specific numbers (VIX, win rates, probabilities, prices)
2. Actionable - provide clear step-by-step guidance a trader can follow
3. Risk-aware - highlight what could go wrong and when to stand down
4. Confident but humble - show conviction with appropriate uncertainty
5. Professional - suitable for an institutional trading desk morning call

You speak in first person as the trading system's intelligence. Never fabricate data."""


PREGAME_REGIME_PROMPT = """Analyze today's market regime and create the morning game plan overview.

=== MARKET REGIME ===
Current Regime: {regime} ({regime_confidence:.0%} confidence)
Regime Probabilities: BULL={bull_prob:.0%}, NEUTRAL={neutral_prob:.0%}, BEAR={bear_prob:.0%}
Days in Current Regime: {days_in_regime}
SPY Position: {spy_position} 200-day SMA (${spy_sma200:.2f})

=== MARKET MOOD ===
Mood State: {mood_state} (score: {mood_score:.2f})
VIX Level: {vix:.1f}
Is Extreme: {is_extreme}

=== OVERNIGHT NEWS ===
Aggregate Sentiment: {sentiment_compound:.2f}
Key Themes: {key_themes}
Headlines Summary:
{headlines}

=== PORTFOLIO STATUS ===
Open Positions: {position_count}
Heat Level: {heat_level} ({heat_score}/100)
Unrealized P&L: ${unrealized_pnl:,.2f}

Provide a structured analysis with these exact sections:

**REGIME ANALYSIS**
[2-3 paragraphs analyzing regime probabilities and implications for today]

**MARKET MOOD INTERPRETATION**
[1-2 paragraphs on what the mood/VIX combination means for trading]

**NEWS IMPACT SUMMARY**
[1-2 paragraphs on key themes and how they might affect sectors/trades]

**TODAY'S TRADING BIAS**
[State: AGGRESSIVE | NEUTRAL | DEFENSIVE]
[1 paragraph explaining why and what this means for position sizing]"""


PREGAME_ACTION_PROMPT = """Create today's detailed action plan based on the picks.

=== TOP-3 PICKS ===
{top3_details}

=== TRADE OF THE DAY ===
{totd_details}

=== MARKET CONTEXT ===
Regime: {regime} ({regime_confidence:.0%})
VIX: {vix:.1f}
Mood: {mood_state}

Provide a structured action plan:

**STEP-BY-STEP ACTION PLAN**

1. PRE-MARKET PREPARATION (before 9:30 AM ET)
   - [specific tasks to complete]

2. ENTRY TIMING
   - [when to look for entries on each pick]
   - [avoid first 15 minutes? Wait for confirmation?]

3. ORDER EXECUTION APPROACH
   - [limit vs market orders]
   - [scaling in strategy]
   - [position monitoring checkpoints]

**POSITION SIZING** (for each pick)
[For each symbol: max shares at 2% risk, stop rationale, scaling approach]

**KEY LEVELS TO WATCH**
- SPY: [support/resistance levels]
- VIX: [warning levels]
- Each Pick: [key levels]

**RISK WARNINGS**
- [What invalidates the plan?]
- [When to stand down completely?]
- [Position correlation concerns?]

**MACRO EVENTS TODAY**
[Any economic releases, Fed speakers, earnings that matter]"""


HALFTIME_POSITION_PROMPT = """Analyze each open position and provide specific recommendations.

=== CURRENT POSITIONS ===
{positions_table}

=== MORNING PLAN ===
Top-3 Picks: {morning_picks}
TOTD: {totd_symbol}

=== MARKET CHANGES SINCE OPEN ===
Morning Regime: {morning_regime} -> Current: {current_regime}
Morning VIX: {morning_vix:.1f} -> Current: {current_vix:.1f}
Market Movement: {market_movement}

For EACH position, provide:

**{symbol}** ({side} @ ${entry_price:.2f})
- STATUS: [Performance vs expectations - cite specific P&L %]
- ANALYSIS: [Why working/not working - be specific]
- RECOMMENDATION: [HOLD | ADJUST_STOP (new level: $X.XX) | TAKE_PROFIT | EXIT]
- KEY LEVEL: [Price that would change your recommendation]
---"""


HALFTIME_SUMMARY_PROMPT = """Summarize the halftime assessment and afternoon plan.

=== POSITION PERFORMANCE ===
{position_summary}

=== REGIME STATUS ===
Changed: {regime_changed}
Current: {current_regime}

=== MIDDAY NEWS ===
{midday_news}

Provide:

**WHAT'S WORKING**
[Which setups are performing, why]

**WHAT'S NOT WORKING**
[Which trades need attention, why]

**REGIME CHANGE IMPACT**
[If regime changed, what adjustments needed. If not, note stability]

**AFTERNOON GAME PLAN**
1. Priority Adjustments: [specific actions]
2. New Opportunities: [any new setups worth watching]
3. Risk Focus Areas: [what to monitor closely]

**EXPOSURE CHECK**
[Heat level assessment, any concentration concerns]"""


POSTGAME_REFLECTION_PROMPT = """Complete end-of-day analysis and reflection.

=== DAY SUMMARY ===
Date: {date}
Total Trades: {trade_count}
Wins: {wins} | Losses: {losses}
Win Rate: {win_rate:.1%}
Realized P&L: ${realized_pnl:,.2f}
Best Trade: {best_trade}
Worst Trade: {worst_trade}

=== TRADES TODAY ===
{trades_table}

=== REGIME BEHAVIOR ===
Morning: {morning_regime} -> Midday: {midday_regime} -> EOD: {eod_regime}

=== STRATEGY BREAKDOWN ===
{strategy_stats}

=== MORNING PREDICTIONS ===
{morning_predictions}

Provide:

**EXECUTIVE SUMMARY**
[One paragraph: the key takeaway from today]

**WHAT WENT RIGHT**
[3-5 specific things that worked, cite trades]

**WHAT WENT WRONG**
[3-5 specific things that didn't work, cite trades]

**REGIME ANALYSIS**
[How accurate were our regime reads? What did we miss?]

**NEWS IMPACT**
[Did news events affect our trades? How?]"""


POSTGAME_LESSONS_PROMPT = """Extract lessons and generate hypotheses for future testing.

=== DAY PERFORMANCE ===
{performance_summary}

=== KEY OBSERVATIONS ===
{observations}

Provide:

**KEY LESSONS LEARNED** (3-5 actionable bullets)
- [Lesson 1 with specific evidence]
- [Lesson 2 with specific evidence]
- [etc.]

**HYPOTHESES TO TEST**
For each hypothesis:
- HYPOTHESIS: [description]
- CONDITION: [when it applies, e.g., "regime = BEAR and VIX > 25"]
- PREDICTION: [expected outcome, e.g., "IBS_RSI win rate > 65%"]
- RATIONALE: [why we think this based on today]
- CONFIDENCE: [HIGH/MEDIUM/LOW]

**STRATEGY EFFECTIVENESS**
[Which strategy performed best? Why? Any adjustments recommended?]

**NEXT DAY SETUP**
- Closing Regime: {eod_regime}
- Overnight Risks: [specific risks to monitor]
- Tomorrow's Bias: [BULLISH/NEUTRAL/BEARISH with rationale]
- Gap Risk: [assessment based on after-hours activity]

**KEY NEWS TO WATCH OVERNIGHT**
[Specific events, earnings, economic data]"""


# =============================================================================
# Game Briefing Engine
# =============================================================================

class GameBriefingEngine:
    """
    Unified engine for generating PRE_GAME, HALF_TIME, and POST_GAME briefings.

    Integrates all ML/AI/LLM components for comprehensive analysis.
    """

    def __init__(self, dotenv_path: str = "./.env"):
        """Initialize the briefing engine with all components."""
        load_dotenv(dotenv_path)

        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

        # Initialize components lazily
        self._trade_analyzer = None
        self._regime_detector = None
        self._mood_analyzer = None
        self._news_processor = None
        self._heat_monitor = None
        self._reflection_engine = None

        logger.info("GameBriefingEngine initialized")

    @property
    def trade_analyzer(self):
        if self._trade_analyzer is None:
            try:
                from cognitive.llm_trade_analyzer import get_trade_analyzer
                self._trade_analyzer = get_trade_analyzer()
            except Exception as e:
                logger.warning(f"Could not load trade analyzer: {e}")
        return self._trade_analyzer

    @property
    def regime_detector(self):
        if self._regime_detector is None:
            try:
                from ml_advanced.hmm_regime_detector import HMMRegimeDetector
                self._regime_detector = HMMRegimeDetector()
            except Exception as e:
                logger.warning(f"Could not load HMM regime detector: {e}")
                # Fall back to adaptive detector
                try:
                    from ml_advanced.adaptive_regime_detector import AdaptiveRegimeDetector
                    self._regime_detector = AdaptiveRegimeDetector()
                except Exception:
                    pass
        return self._regime_detector

    @property
    def mood_analyzer(self):
        if self._mood_analyzer is None:
            try:
                from altdata.market_mood_analyzer import get_market_mood_analyzer
                self._mood_analyzer = get_market_mood_analyzer()
            except Exception as e:
                logger.warning(f"Could not load mood analyzer: {e}")
        return self._mood_analyzer

    @property
    def news_processor(self):
        if self._news_processor is None:
            try:
                from altdata.news_processor import get_news_processor
                self._news_processor = get_news_processor()
            except Exception as e:
                logger.warning(f"Could not load news processor: {e}")
        return self._news_processor

    @property
    def heat_monitor(self):
        if self._heat_monitor is None:
            try:
                from portfolio.heat_monitor import PortfolioHeatMonitor
                self._heat_monitor = PortfolioHeatMonitor()
            except Exception as e:
                logger.warning(f"Could not load heat monitor: {e}")
        return self._heat_monitor

    @property
    def reflection_engine(self):
        if self._reflection_engine is None:
            try:
                from cognitive.reflection_engine import ReflectionEngine
                self._reflection_engine = ReflectionEngine()
            except Exception as e:
                logger.warning(f"Could not load reflection engine: {e}")
        return self._reflection_engine

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Make LLM API call with fallback."""
        if not ANTHROPIC_AVAILABLE or not self.api_key:
            logger.warning("Anthropic not available, using deterministic fallback")
            return "[LLM analysis not available - deterministic fallback]"

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=4000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[LLM analysis failed: {e}]"

    def gather_context(self, date: Optional[str] = None) -> BriefingContext:
        """Gather all context needed for briefings."""
        now = datetime.now()
        if date is None:
            date = now.date().isoformat()

        context = BriefingContext(timestamp=now, date=date)

        # Get regime - try to load from daily_insights.json first (most reliable)
        try:
            insights_file = ROOT / 'logs' / 'daily_insights.json'
            if insights_file.exists():
                with open(insights_file) as f:
                    insights = json.load(f)
                regime_str = insights.get('regime_assessment', '')
                if 'BULL' in regime_str.upper():
                    context.regime = 'BULLISH'
                    context.regime_confidence = 0.9
                elif 'BEAR' in regime_str.upper():
                    context.regime = 'BEARISH'
                    context.regime_confidence = 0.9
                else:
                    context.regime = 'NEUTRAL'
                    context.regime_confidence = 0.5
        except Exception as e:
            logger.debug(f"Could not get regime from insights: {e}")

        # Get market mood
        try:
            if self.mood_analyzer:
                mood = self.mood_analyzer.get_market_mood({})
                if mood:
                    # Handle both dataclass and dict returns
                    if hasattr(mood, 'mood_state'):
                        context.mood_state = mood.mood_state.value if hasattr(mood.mood_state, 'value') else str(mood.mood_state)
                        context.mood_score = getattr(mood, 'mood_score', 0.0)
                        context.vix_level = mood.components.get('vix', 20.0) if hasattr(mood, 'components') and mood.components else 20.0
                        context.is_mood_extreme = getattr(mood, 'is_extreme', False)
                    elif isinstance(mood, dict):
                        context.mood_state = mood.get('mood_state', 'Neutral')
                        context.mood_score = mood.get('mood_score', 0.0)
                        context.vix_level = mood.get('vix', 20.0)
                        context.is_mood_extreme = mood.get('is_extreme', False)
        except Exception as e:
            logger.warning(f"Could not get mood: {e}")

        # Get news/sentiment
        try:
            if self.news_processor:
                # Use fetch_news and get_aggregated_sentiment methods
                if hasattr(self.news_processor, 'fetch_news'):
                    articles = self.news_processor.fetch_news(symbols=['SPY'], limit=10)
                    if articles:
                        # Convert NewsArticle objects to dicts for JSON serialization
                        context.news_articles = [
                            a.to_dict() if hasattr(a, 'to_dict') else a
                            for a in articles
                        ]
                # get_aggregated_sentiment takes symbols list, not articles
                if hasattr(self.news_processor, 'get_aggregated_sentiment'):
                    sentiment = self.news_processor.get_aggregated_sentiment(symbols=['SPY'])
                    if sentiment and isinstance(sentiment, dict):
                        context.sentiment_compound = sentiment.get('compound', 0.0)
        except Exception as e:
            logger.debug(f"Could not get news: {e}")

        # Get portfolio heat - skip for now if no positions, calculate after getting positions
        # Will be calculated below after positions are loaded

        # Get positions
        try:
            positions_file = ROOT / 'state' / 'positions.json'
            if positions_file.exists():
                with open(positions_file) as f:
                    positions_data = json.load(f)
                # Handle both list and dict format
                if isinstance(positions_data, list):
                    context.positions = positions_data
                elif isinstance(positions_data, dict):
                    context.positions = positions_data.get('positions', [])
                else:
                    context.positions = []
                context.total_positions = len(context.positions)
                context.unrealized_pnl = sum(
                    p.get('unrealized_pnl', 0) for p in context.positions if isinstance(p, dict)
                )
        except Exception as e:
            logger.debug(f"Could not get positions: {e}")

        # Calculate portfolio heat now that we have positions
        try:
            if self.heat_monitor and context.positions:
                if hasattr(self.heat_monitor, 'calculate_heat'):
                    # Get equity from positions or use default
                    equity = sum(p.get('market_value', 0) for p in context.positions if isinstance(p, dict))
                    if equity > 0:
                        heat = self.heat_monitor.calculate_heat(context.positions, equity)
                        if heat:
                            context.heat_level = heat.heat_level.value if hasattr(heat.heat_level, 'value') else str(heat.heat_level)
                            context.heat_score = getattr(heat, 'heat_score', 0.0)
                            context.heat_alerts = getattr(heat, 'alerts', [])
        except Exception as e:
            logger.debug(f"Could not calculate heat: {e}")

        # Get SPY reference
        try:
            from data.providers.polygon_eod import fetch_daily_bars_polygon
            from datetime import timedelta
            end_date = datetime.now().date().isoformat()
            start_date = (datetime.now().date() - timedelta(days=300)).isoformat()
            spy_df = fetch_daily_bars_polygon('SPY', start_date, end_date)
            if spy_df is not None and not spy_df.empty:
                context.spy_price = float(spy_df['close'].iloc[-1])
                if len(spy_df) >= 200:
                    context.spy_sma200 = float(spy_df['close'].rolling(200).mean().iloc[-1])
                else:
                    context.spy_sma200 = float(spy_df['close'].mean())
                context.spy_position = "ABOVE" if context.spy_price > context.spy_sma200 else "BELOW"
        except Exception as e:
            logger.debug(f"Could not get SPY data: {e}")

        return context

    def _get_top3_and_totd(self, universe: str, cap: int, date: str) -> Tuple[List[Dict], Optional[Dict]]:
        """Get Top-3 picks and TOTD from existing scan outputs."""
        top3 = []
        totd = None

        # Try to load from daily_picks.csv
        picks_file = ROOT / 'logs' / 'daily_picks.csv'
        if picks_file.exists():
            try:
                df = pd.read_csv(picks_file)
                if not df.empty:
                    top3 = df.head(3).to_dict('records')
            except Exception as e:
                logger.warning(f"Could not load picks: {e}")

        # Try to load TOTD from comprehensive_totd.json
        totd_file = ROOT / 'logs' / 'comprehensive_totd.json'
        if totd_file.exists():
            try:
                with open(totd_file) as f:
                    totd = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load TOTD: {e}")

        return top3, totd

    def _get_daily_insights(self) -> Optional[Dict]:
        """Get daily insights from existing file."""
        insights_file = ROOT / 'logs' / 'daily_insights.json'
        if insights_file.exists():
            try:
                with open(insights_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def generate_pregame(
        self,
        universe: str = "data/universe/optionable_liquid_900.csv",
        cap: int = 300,
        date: Optional[str] = None
    ) -> PreGameBriefing:
        """Generate comprehensive PRE_GAME morning briefing."""
        logger.info("Generating PRE_GAME briefing...")

        # Gather context
        context = self.gather_context(date)

        # Get picks and TOTD
        top3, totd = self._get_top3_and_totd(universe, cap, context.date)

        # Get daily insights if available
        insights = self._get_daily_insights()

        # Create briefing
        briefing = PreGameBriefing(
            context=context,
            top3_picks=top3,
            totd=totd,
            generated_at=datetime.now().isoformat(),
            llm_model=self.model
        )

        # Generate regime analysis via LLM
        def get_headline(article):
            """Extract headline from article (dict or NewsArticle object)."""
            if hasattr(article, 'headline'):
                return article.headline
            elif isinstance(article, dict):
                return article.get('headline', article.get('title', 'No title'))
            return 'No title'

        headlines = "\n".join([
            f"- {get_headline(a)}"
            for a in context.news_articles[:5]
        ]) or "No headlines available"

        regime_prompt = PREGAME_REGIME_PROMPT.format(
            regime=context.regime,
            regime_confidence=context.regime_confidence,
            bull_prob=context.regime_probabilities.get('BULLISH', 0.33),
            neutral_prob=context.regime_probabilities.get('NEUTRAL', 0.34),
            bear_prob=context.regime_probabilities.get('BEARISH', 0.33),
            days_in_regime=context.days_in_regime,
            spy_position=context.spy_position,
            spy_sma200=context.spy_sma200,
            mood_state=context.mood_state,
            mood_score=context.mood_score,
            vix=context.vix_level,
            is_extreme=context.is_mood_extreme,
            sentiment_compound=context.sentiment_compound,
            key_themes=", ".join(context.key_themes) or "None identified",
            headlines=headlines,
            position_count=context.total_positions,
            heat_level=context.heat_level,
            heat_score=context.heat_score,
            unrealized_pnl=context.unrealized_pnl
        )

        regime_analysis = self._call_llm(PREGAME_SYSTEM_PROMPT, regime_prompt)

        # Parse regime analysis sections
        if "**REGIME ANALYSIS**" in regime_analysis:
            parts = regime_analysis.split("**")
            for i, part in enumerate(parts):
                if "REGIME ANALYSIS" in part and i + 1 < len(parts):
                    briefing.regime_analysis = parts[i + 1].strip()
                elif "MARKET MOOD" in part and i + 1 < len(parts):
                    briefing.market_mood_description = parts[i + 1].strip()
                elif "NEWS IMPACT" in part and i + 1 < len(parts):
                    briefing.overnight_news_summary = parts[i + 1].strip()
                elif "TRADING BIAS" in part and i + 1 < len(parts):
                    bias_text = parts[i + 1].strip()
                    if "AGGRESSIVE" in bias_text.upper():
                        briefing.trading_bias = "AGGRESSIVE"
                    elif "DEFENSIVE" in bias_text.upper():
                        briefing.trading_bias = "DEFENSIVE"
                    else:
                        briefing.trading_bias = "NEUTRAL"
        else:
            briefing.regime_analysis = regime_analysis

        # Generate action plan if we have picks
        if top3:
            top3_details = json.dumps(top3, indent=2, default=str)
            totd_details = json.dumps(totd, indent=2, default=str) if totd else "No TOTD available"

            action_prompt = PREGAME_ACTION_PROMPT.format(
                top3_details=top3_details,
                totd_details=totd_details,
                regime=context.regime,
                regime_confidence=context.regime_confidence,
                vix=context.vix_level,
                mood_state=context.mood_state
            )

            action_plan = self._call_llm(PREGAME_SYSTEM_PROMPT, action_prompt)

            # Parse action plan sections
            if "**STEP-BY-STEP" in action_plan:
                parts = action_plan.split("**")
                for i, part in enumerate(parts):
                    if "PRE-MARKET" in part and i + 1 < len(parts):
                        lines = [l.strip() for l in parts[i + 1].split("\n") if l.strip().startswith("-")]
                        briefing.pre_market_prep = [l.lstrip("- ") for l in lines]
                    elif "ENTRY TIMING" in part and i + 1 < len(parts):
                        lines = [l.strip() for l in parts[i + 1].split("\n") if l.strip().startswith("-")]
                        briefing.entry_timing = [l.lstrip("- ") for l in lines]
                    elif "KEY LEVELS" in part and i + 1 < len(parts):
                        # Parse key levels
                        pass
                    elif "RISK WARNINGS" in part and i + 1 < len(parts):
                        lines = [l.strip() for l in parts[i + 1].split("\n") if l.strip().startswith("-")]
                        briefing.risk_warnings = [l.lstrip("- ") for l in lines]

        # Use existing TOTD analysis if available
        if totd and totd.get('executive_summary'):
            briefing.totd_deep_analysis = totd.get('why_this_trade', '') or totd.get('executive_summary', '')

        # Use existing narratives if available
        if insights and insights.get('top3_narratives'):
            briefing.top3_narratives = insights['top3_narratives']

        briefing.generation_method = "claude" if self.api_key else "deterministic"

        return briefing

    def generate_halftime(self) -> HalfTimeBriefing:
        """Generate HALF_TIME midday status briefing."""
        logger.info("Generating HALF_TIME briefing...")

        context = self.gather_context()

        briefing = HalfTimeBriefing(
            context=context,
            current_regime=context.regime,
            current_vix=context.vix_level,
            generated_at=datetime.now().isoformat(),
            llm_model=self.model
        )

        # Load morning data for comparison
        pregame_file = ROOT / 'reports' / f'pregame_{context.date.replace("-", "")}.json'
        if pregame_file.exists():
            try:
                with open(pregame_file) as f:
                    morning_data = json.load(f)
                briefing.morning_regime = morning_data.get('context', {}).get('regime', '')
                briefing.morning_vix = morning_data.get('context', {}).get('vix_level', 0)
                briefing.morning_plan_picks = [
                    p.get('symbol', '') for p in morning_data.get('top3_picks', [])
                ]
                if morning_data.get('totd'):
                    briefing.morning_totd = morning_data['totd'].get('symbol', '')
            except Exception:
                pass

        briefing.regime_changed = briefing.morning_regime != briefing.current_regime and briefing.morning_regime != ""

        # Analyze positions
        position_statuses = []
        for pos in context.positions:
            status = PositionStatus(
                symbol=pos.get('symbol', ''),
                side=pos.get('side', 'long'),
                entry_price=pos.get('entry_price', 0),
                current_price=pos.get('current_price', pos.get('entry_price', 0)),
                shares=pos.get('shares', 0),
                entry_date=pos.get('entry_date', ''),
                days_held=pos.get('days_held', 0),
                unrealized_pnl=pos.get('unrealized_pnl', 0),
                pnl_percent=pos.get('pnl_percent', 0),
                stop_loss=pos.get('stop_loss', 0),
                take_profit=pos.get('take_profit')
            )
            position_statuses.append(status)

        briefing.position_analysis = position_statuses

        # Generate LLM analysis if we have positions
        if position_statuses:
            positions_table = "\n".join([
                f"- {p.symbol} ({p.side}): Entry ${p.entry_price:.2f}, Current ${p.current_price:.2f}, "
                f"P&L {p.pnl_percent:+.1f}%, Stop ${p.stop_loss:.2f}, Days Held: {p.days_held}"
                for p in position_statuses
            ])

            position_prompt = HALFTIME_POSITION_PROMPT.format(
                positions_table=positions_table,
                morning_picks=", ".join(briefing.morning_plan_picks) or "None recorded",
                totd_symbol=briefing.morning_totd or "None",
                morning_regime=briefing.morning_regime or "Unknown",
                current_regime=briefing.current_regime,
                morning_vix=briefing.morning_vix,
                current_vix=briefing.current_vix,
                market_movement="SPY " + ("up" if context.spy_price > context.spy_sma200 else "down")
            )

            position_analysis = self._call_llm(PREGAME_SYSTEM_PROMPT, position_prompt)
            briefing.positions_summary = position_analysis

        # Generate summary
        summary_prompt = HALFTIME_SUMMARY_PROMPT.format(
            position_summary=briefing.positions_summary[:1000] if briefing.positions_summary else "No positions",
            regime_changed=briefing.regime_changed,
            current_regime=briefing.current_regime,
            midday_news="\n".join([
                f"- {a.get('title', 'No title')}"
                for a in context.news_articles[:3]
            ]) or "No significant news"
        )

        summary = self._call_llm(PREGAME_SYSTEM_PROMPT, summary_prompt)

        # Parse summary sections
        if "**WHAT'S WORKING**" in summary:
            parts = summary.split("**")
            for i, part in enumerate(parts):
                if "WHAT'S WORKING" in part and i + 1 < len(parts):
                    lines = [l.strip() for l in parts[i + 1].split("\n") if l.strip().startswith("-") or l.strip().startswith("•")]
                    briefing.whats_working = [l.lstrip("-• ") for l in lines][:5]
                elif "WHAT'S NOT WORKING" in part and i + 1 < len(parts):
                    lines = [l.strip() for l in parts[i + 1].split("\n") if l.strip().startswith("-") or l.strip().startswith("•")]
                    briefing.whats_not_working = [l.lstrip("-• ") for l in lines][:5]
                elif "AFTERNOON GAME PLAN" in part and i + 1 < len(parts):
                    briefing.afternoon_game_plan = parts[i + 1].strip()
                elif "EXPOSURE CHECK" in part and i + 1 < len(parts):
                    briefing.exposure_check = parts[i + 1].strip()

        briefing.generation_method = "claude" if self.api_key else "deterministic"

        return briefing

    def generate_postgame(self) -> PostGameBriefing:
        """Generate POST_GAME end-of-day analysis briefing."""
        logger.info("Generating POST_GAME briefing...")

        context = self.gather_context()

        briefing = PostGameBriefing(
            context=context,
            eod_regime=context.regime,
            generated_at=datetime.now().isoformat(),
            llm_model=self.model
        )

        # Load morning and halftime data
        date_str = context.date.replace("-", "")
        pregame_file = ROOT / 'reports' / f'pregame_{date_str}.json'
        halftime_file = ROOT / 'reports' / f'halftime_{date_str}.json'

        if pregame_file.exists():
            try:
                with open(pregame_file) as f:
                    morning_data = json.load(f)
                briefing.morning_regime = morning_data.get('context', {}).get('regime', '')
            except Exception:
                pass

        if halftime_file.exists():
            try:
                with open(halftime_file) as f:
                    midday_data = json.load(f)
                briefing.midday_regime = midday_data.get('current_regime', '')
            except Exception:
                pass

        # Load today's trades from trade log
        # Only count real FILLED trades, not REJECTED/PENDING/TEST orders
        trades_file = ROOT / 'logs' / 'trades.jsonl'
        today_trades = []
        if trades_file.exists():
            try:
                with open(trades_file) as f:
                    for line in f:
                        if line.strip():
                            trade = json.loads(line)
                            trade_date = trade.get('timestamp', '')[:10]
                            trade_status = trade.get('status', '').upper()
                            decision_id = trade.get('decision_id', '')
                            strategy = trade.get('strategy_used', '') or ''

                            # Skip non-FILLED trades
                            if trade_status != 'FILLED':
                                continue
                            # Skip test trades (decision_id contains TEST or strategy is test)
                            if 'TEST' in decision_id.upper() or 'test' in strategy.lower():
                                continue
                            # Skip fake/mock broker order IDs (test harness artifacts)
                            broker_id = trade.get('broker_order_id', '') or ''
                            if 'test' in broker_id.lower() or broker_id == 'broker-order-id-123':
                                continue
                            # Only include today's real trades
                            if trade_date == context.date:
                                today_trades.append(trade)
            except Exception:
                pass

        # Convert to TradeOutcome objects
        trade_outcomes = []
        for t in today_trades:
            outcome = TradeOutcome(
                symbol=t.get('symbol', ''),
                strategy=t.get('strategy', ''),
                side=t.get('side', 'long'),
                entry_price=t.get('entry_price', 0),
                exit_price=t.get('exit_price', 0),
                shares=t.get('shares', 0),
                entry_time=t.get('entry_time', ''),
                exit_time=t.get('exit_time', ''),
                hold_bars=t.get('hold_bars', 0),
                pnl_dollars=t.get('pnl_dollars', 0),
                pnl_percent=t.get('pnl_percent', 0),
                exit_reason=t.get('exit_reason', 'UNKNOWN'),
                was_winner=t.get('pnl_dollars', 0) > 0
            )
            trade_outcomes.append(outcome)

        briefing.trades_today = trade_outcomes

        # Calculate day summary
        if trade_outcomes:
            wins = sum(1 for t in trade_outcomes if t.was_winner)
            losses = len(trade_outcomes) - wins
            total_pnl = sum(t.pnl_dollars for t in trade_outcomes)
            best = max(trade_outcomes, key=lambda t: t.pnl_dollars)
            worst = min(trade_outcomes, key=lambda t: t.pnl_dollars)

            briefing.day_summary = {
                'total_trades': len(trade_outcomes),
                'wins': wins,
                'losses': losses,
                'win_rate': wins / len(trade_outcomes) if trade_outcomes else 0,
                'realized_pnl': total_pnl,
                'best_trade': f"{best.symbol} +${best.pnl_dollars:.2f}",
                'worst_trade': f"{worst.symbol} ${worst.pnl_dollars:.2f}"
            }

        # Calculate strategy breakdown
        strategy_stats = {}
        for t in trade_outcomes:
            strat = t.strategy
            if strat not in strategy_stats:
                strategy_stats[strat] = {'trades': 0, 'wins': 0, 'pnl': 0}
            strategy_stats[strat]['trades'] += 1
            if t.was_winner:
                strategy_stats[strat]['wins'] += 1
            strategy_stats[strat]['pnl'] += t.pnl_dollars

        for strat, stats in strategy_stats.items():
            stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0

        briefing.strategy_breakdown = strategy_stats

        # Generate reflection via LLM
        if trade_outcomes:
            trades_table = "\n".join([
                f"- {t.symbol} ({t.strategy}): {t.side} @ ${t.entry_price:.2f} -> ${t.exit_price:.2f}, "
                f"P&L: ${t.pnl_dollars:+.2f} ({t.pnl_percent:+.1f}%), Exit: {t.exit_reason}"
                for t in trade_outcomes
            ])

            strategy_stats_str = "\n".join([
                f"- {strat}: {s['trades']} trades, {s['win_rate']:.0%} WR, ${s['pnl']:+.2f}"
                for strat, s in strategy_stats.items()
            ])

            reflection_prompt = POSTGAME_REFLECTION_PROMPT.format(
                date=context.date,
                trade_count=len(trade_outcomes),
                wins=briefing.day_summary.get('wins', 0),
                losses=briefing.day_summary.get('losses', 0),
                win_rate=briefing.day_summary.get('win_rate', 0),
                realized_pnl=briefing.day_summary.get('realized_pnl', 0),
                best_trade=briefing.day_summary.get('best_trade', 'N/A'),
                worst_trade=briefing.day_summary.get('worst_trade', 'N/A'),
                trades_table=trades_table,
                morning_regime=briefing.morning_regime or "Unknown",
                midday_regime=briefing.midday_regime or "Unknown",
                eod_regime=briefing.eod_regime,
                strategy_stats=strategy_stats_str,
                morning_predictions="See morning briefing for predictions"
            )

            reflection = self._call_llm(PREGAME_SYSTEM_PROMPT, reflection_prompt)
            briefing.reflection_summary = reflection

            # Parse reflection sections
            if "**EXECUTIVE SUMMARY**" in reflection:
                parts = reflection.split("**")
                for i, part in enumerate(parts):
                    if "EXECUTIVE SUMMARY" in part and i + 1 < len(parts):
                        briefing.executive_summary = parts[i + 1].strip()
                    elif "WHAT WENT RIGHT" in part and i + 1 < len(parts):
                        lines = [l.strip() for l in parts[i + 1].split("\n") if l.strip().startswith("-") or l.strip().startswith("•")]
                        briefing.what_went_right = [l.lstrip("-• ") for l in lines][:5]
                    elif "WHAT WENT WRONG" in part and i + 1 < len(parts):
                        lines = [l.strip() for l in parts[i + 1].split("\n") if l.strip().startswith("-") or l.strip().startswith("•")]
                        briefing.what_went_wrong = [l.lstrip("-• ") for l in lines][:5]
                    elif "REGIME ANALYSIS" in part and i + 1 < len(parts):
                        briefing.regime_behavior = parts[i + 1].strip()
                    elif "NEWS IMPACT" in part and i + 1 < len(parts):
                        briefing.news_impact_analysis = parts[i + 1].strip()

        # Generate lessons and hypotheses
        lessons_prompt = POSTGAME_LESSONS_PROMPT.format(
            performance_summary=json.dumps(briefing.day_summary, default=str),
            observations=briefing.reflection_summary[:1500] if briefing.reflection_summary else "No observations",
            eod_regime=briefing.eod_regime
        )

        lessons = self._call_llm(PREGAME_SYSTEM_PROMPT, lessons_prompt)

        # Parse lessons sections
        if "**KEY LESSONS" in lessons:
            parts = lessons.split("**")
            for i, part in enumerate(parts):
                if "KEY LESSONS" in part and i + 1 < len(parts):
                    lines = [l.strip() for l in parts[i + 1].split("\n") if l.strip().startswith("-") or l.strip().startswith("•")]
                    briefing.lessons_learned = [l.lstrip("-• ") for l in lines][:5]
                elif "NEXT DAY SETUP" in part and i + 1 < len(parts):
                    briefing.next_day_setup = parts[i + 1].strip()
                elif "OVERNIGHT RISKS" in part.upper() and i + 1 < len(parts):
                    lines = [l.strip() for l in parts[i + 1].split("\n") if l.strip().startswith("-") or l.strip().startswith("•")]
                    briefing.overnight_risks = [l.lstrip("-• ") for l in lines][:3]

        # Use reflection engine for deeper analysis if available
        if self.reflection_engine:
            try:
                reflection_result = self.reflection_engine.periodic_reflection(lookback_hours=8)
                if reflection_result:
                    if reflection_result.lessons:
                        briefing.lessons_learned.extend(reflection_result.lessons[:3])
                    if reflection_result.behavior_changes:
                        briefing.behavior_changes = [
                            str(bc) for bc in reflection_result.behavior_changes[:3]
                        ]
            except Exception as e:
                logger.warning(f"Reflection engine failed: {e}")

        briefing.generation_method = "claude" if self.api_key else "deterministic"

        return briefing

    def save_briefing(self, briefing: Any, phase: str) -> Tuple[Path, Path]:
        """Save briefing to JSON and Markdown files."""
        date_str = briefing.context.date.replace("-", "")

        reports_dir = ROOT / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON (UTF-8 for Unicode support)
        json_path = reports_dir / f'{phase}_{date_str}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(briefing.to_dict(), f, indent=2, default=str, ensure_ascii=False)

        # Generate Markdown (UTF-8 for Unicode support)
        md_path = reports_dir / f'{phase}_{date_str}.md'
        md_content = self._generate_markdown(briefing, phase)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Saved {phase} briefing to {json_path} and {md_path}")
        return json_path, md_path

    def _generate_markdown(self, briefing: Any, phase: str) -> str:
        """Generate readable Markdown from briefing."""
        lines = []

        if phase == "pregame":
            lines.extend([
                f"# PRE-GAME BRIEFING - {briefing.context.date}",
                f"*Generated: {briefing.generated_at}*",
                "",
                "## Market Context",
                f"- **Regime:** {briefing.context.regime} ({briefing.context.regime_confidence:.0%} confidence)",
                f"- **Mood:** {briefing.context.mood_state} (VIX: {briefing.context.vix_level:.1f})",
                f"- **Trading Bias:** {briefing.trading_bias}",
                "",
                "## Regime Analysis",
                briefing.regime_analysis or "No analysis available",
                "",
                "## Market Mood",
                briefing.market_mood_description or "No mood analysis available",
                "",
                "## News Summary",
                briefing.overnight_news_summary or "No news summary available",
                "",
                "## Top-3 Picks"
            ])

            for i, pick in enumerate(briefing.top3_picks, 1):
                lines.append(f"\n### {i}. {pick.get('symbol', 'Unknown')} ({pick.get('strategy', '')})")
                lines.append(f"- Side: {pick.get('side', '')}")
                lines.append(f"- Entry: ${pick.get('entry_price', 0):.2f}")
                lines.append(f"- Stop: ${pick.get('stop_loss', 0):.2f}")

            if briefing.totd:
                lines.extend([
                    "",
                    "## Trade of the Day",
                    f"**{briefing.totd.get('symbol', '')}** ({briefing.totd.get('strategy', '')})",
                    "",
                    briefing.totd_deep_analysis or "See TOTD report for details"
                ])

            if briefing.risk_warnings:
                lines.extend(["", "## Risk Warnings"])
                for w in briefing.risk_warnings:
                    lines.append(f"- {w}")

        elif phase == "halftime":
            lines.extend([
                f"# HALF-TIME BRIEFING - {briefing.context.date}",
                f"*Generated: {briefing.generated_at}*",
                "",
                "## Market Status",
                f"- **Current Regime:** {briefing.current_regime}",
                f"- **Regime Changed:** {'Yes' if briefing.regime_changed else 'No'}",
                f"- **VIX:** {briefing.current_vix:.1f} (Morning: {briefing.morning_vix:.1f})",
                "",
                "## Position Analysis"
            ])

            for p in briefing.position_analysis:
                lines.append(f"\n### {p.symbol} ({p.side})")
                lines.append(f"- Entry: ${p.entry_price:.2f} | Current: ${p.current_price:.2f}")
                lines.append(f"- P&L: ${p.unrealized_pnl:.2f} ({p.pnl_percent:+.1f}%)")
                lines.append(f"- Recommendation: **{p.recommendation}**")

            if briefing.whats_working:
                lines.extend(["", "## What's Working"])
                for w in briefing.whats_working:
                    lines.append(f"- {w}")

            if briefing.whats_not_working:
                lines.extend(["", "## What's Not Working"])
                for w in briefing.whats_not_working:
                    lines.append(f"- {w}")

            if briefing.afternoon_game_plan:
                lines.extend(["", "## Afternoon Game Plan", briefing.afternoon_game_plan])

        elif phase == "postgame":
            lines.extend([
                f"# POST-GAME BRIEFING - {briefing.context.date}",
                f"*Generated: {briefing.generated_at}*",
                "",
                "## Executive Summary",
                briefing.executive_summary or "No summary available",
                "",
                "## Day Summary",
                f"- **Trades:** {briefing.day_summary.get('total_trades', 0)}",
                f"- **Wins/Losses:** {briefing.day_summary.get('wins', 0)}/{briefing.day_summary.get('losses', 0)}",
                f"- **Win Rate:** {briefing.day_summary.get('win_rate', 0):.0%}",
                f"- **P&L:** ${briefing.day_summary.get('realized_pnl', 0):,.2f}",
                "",
                "## Regime Behavior",
                f"Morning: {briefing.morning_regime} -> Midday: {briefing.midday_regime} -> EOD: {briefing.eod_regime}",
                "",
                briefing.regime_behavior or ""
            ])

            if briefing.what_went_right:
                lines.extend(["", "## What Went Right"])
                for w in briefing.what_went_right:
                    lines.append(f"- {w}")

            if briefing.what_went_wrong:
                lines.extend(["", "## What Went Wrong"])
                for w in briefing.what_went_wrong:
                    lines.append(f"- {w}")

            if briefing.lessons_learned:
                lines.extend(["", "## Lessons Learned"])
                for l in briefing.lessons_learned:
                    lines.append(f"- {l}")

            if briefing.next_day_setup:
                lines.extend(["", "## Next Day Setup", briefing.next_day_setup])

        return "\n".join(lines)

    def send_telegram_summary(self, briefing: Any, phase: str) -> bool:
        """Send briefing summary to Telegram."""
        try:
            from alerts.telegram_alerter import send_telegram_message

            if phase == "pregame":
                msg = f"<b>PRE-GAME BRIEFING - {briefing.context.date}</b>\n\n"
                msg += f"<b>Regime:</b> {briefing.context.regime} ({briefing.context.regime_confidence:.0%})\n"
                msg += f"<b>VIX:</b> {briefing.context.vix_level:.1f}\n"
                msg += f"<b>Bias:</b> {briefing.trading_bias}\n\n"

                if briefing.top3_picks:
                    msg += "<b>Top-3:</b>\n"
                    for pick in briefing.top3_picks:
                        msg += f"• {pick.get('symbol')} ({pick.get('strategy')})\n"

                if briefing.totd:
                    msg += f"\n<b>TOTD:</b> {briefing.totd.get('symbol')}"

            elif phase == "halftime":
                msg = f"<b>HALF-TIME - {briefing.context.date}</b>\n\n"
                msg += f"<b>Regime:</b> {briefing.current_regime}"
                if briefing.regime_changed:
                    msg += " (CHANGED!)"
                msg += f"\n<b>VIX:</b> {briefing.current_vix:.1f}\n"
                msg += f"<b>Positions:</b> {len(briefing.position_analysis)}\n"

                if briefing.whats_working:
                    msg += f"\n<b>Working:</b> {briefing.whats_working[0][:50]}..."

            elif phase == "postgame":
                msg = f"<b>POST-GAME - {briefing.context.date}</b>\n\n"
                msg += f"<b>Trades:</b> {briefing.day_summary.get('total_trades', 0)}\n"
                msg += f"<b>W/L:</b> {briefing.day_summary.get('wins', 0)}/{briefing.day_summary.get('losses', 0)}\n"
                msg += f"<b>P&L:</b> ${briefing.day_summary.get('realized_pnl', 0):,.2f}\n"

                if briefing.lessons_learned:
                    msg += f"\n<b>Key Lesson:</b> {briefing.lessons_learned[0][:80]}..."

            send_telegram_message(msg)
            return True
        except Exception as e:
            logger.warning(f"Could not send Telegram: {e}")
            return False


# =============================================================================
# Module-level functions
# =============================================================================

_engine: Optional[GameBriefingEngine] = None


def get_briefing_engine(dotenv_path: str = "./.env") -> GameBriefingEngine:
    """Get or create singleton briefing engine."""
    global _engine
    if _engine is None:
        _engine = GameBriefingEngine(dotenv_path)
    return _engine


def generate_pregame_briefing(
    universe: str = "data/universe/optionable_liquid_900.csv",
    cap: int = 300,
    date: Optional[str] = None,
    dotenv_path: str = "./.env"
) -> PreGameBriefing:
    """Convenience function to generate PRE_GAME briefing."""
    engine = get_briefing_engine(dotenv_path)
    return engine.generate_pregame(universe, cap, date)


def generate_halftime_briefing(dotenv_path: str = "./.env") -> HalfTimeBriefing:
    """Convenience function to generate HALF_TIME briefing."""
    engine = get_briefing_engine(dotenv_path)
    return engine.generate_halftime()


def generate_postgame_briefing(dotenv_path: str = "./.env") -> PostGameBriefing:
    """Convenience function to generate POST_GAME briefing."""
    engine = get_briefing_engine(dotenv_path)
    return engine.generate_postgame()
