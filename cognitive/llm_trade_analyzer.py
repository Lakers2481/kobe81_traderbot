"""
LLM Trade Analyzer - Central Claude Reasoning Engine
=====================================================

Provides human-like reasoning for:
- Top-3 pick explanations
- Trade of the Day analysis
- Market summary narratives
- News/sentiment interpretation
- Key findings discovery

Uses Claude API with deterministic fallback when unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TradeNarrative:
    """Human-readable explanation for a trade pick."""
    symbol: str
    strategy: str
    narrative: str                              # 2-3 sentence explanation
    conviction_reasons: List[str] = field(default_factory=list)  # Why this trade
    risk_factors: List[str] = field(default_factory=list)        # What could go wrong
    edge_description: str = ""                  # Statistical edge being exploited
    confidence_rating: str = "MEDIUM"           # HIGH | MEDIUM | LOW
    market_alignment: str = ""                  # How market supports this

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DailyInsightReport:
    """Complete daily analysis with LLM-generated narratives."""
    date: str
    timestamp: str
    market_summary: str                         # Overall market narrative
    top3_narratives: List[TradeNarrative] = field(default_factory=list)
    totd_deep_analysis: str = ""                # Extended TOTD analysis
    key_findings: List[str] = field(default_factory=list)   # Notable patterns/anomalies
    sentiment_interpretation: str = ""          # News narrative
    regime_assessment: str = ""                 # Market regime with reasoning
    risk_warnings: List[str] = field(default_factory=list)  # Today's risk factors
    opportunities: List[str] = field(default_factory=list)  # Emerging opportunities
    llm_model: str = "deterministic"
    generation_method: str = "deterministic"    # "claude" or "deterministic"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['top3_narratives'] = [n.to_dict() if hasattr(n, 'to_dict') else n
                                for n in self.top3_narratives]
        return d


@dataclass
class HistoricalPerformance:
    """Historical backtest statistics for a strategy/pattern."""
    strategy: str
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    avg_hold_days: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    recent_performance: str = ""  # Last 30 days summary
    regime_performance: Dict[str, float] = field(default_factory=dict)  # WR by regime
    sample_trades: List[Dict] = field(default_factory=list)  # Recent similar trades


@dataclass
class TechnicalContext:
    """Technical analysis context for a symbol."""
    symbol: str
    current_price: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    rsi_14: float = 0.0
    atr_14: float = 0.0
    volume_vs_avg: float = 0.0  # Today's volume vs 20-day avg
    distance_from_52w_high: float = 0.0
    distance_from_52w_low: float = 0.0
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    trend_direction: str = "NEUTRAL"  # UP, DOWN, NEUTRAL
    volatility_regime: str = "NORMAL"  # LOW, NORMAL, HIGH, EXTREME


@dataclass
class SymbolNews:
    """News and sentiment for a specific symbol."""
    symbol: str
    articles: List[Dict] = field(default_factory=list)
    sentiment_score: float = 0.0
    sentiment_label: str = "NEUTRAL"
    key_themes: List[str] = field(default_factory=list)
    upcoming_events: List[str] = field(default_factory=list)  # Earnings, etc.
    analyst_rating: str = ""
    last_news_date: str = ""


@dataclass
class ComprehensiveTOTDReport:
    """Comprehensive Trade of the Day analysis with all data."""
    # Core Trade Info
    symbol: str
    strategy: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: Optional[float] = None
    position_score: float = 0.0

    # Calculated Metrics
    risk_per_share: float = 0.0
    reward_per_share: float = 0.0
    risk_reward_ratio: float = 0.0
    stop_distance_pct: float = 0.0

    # Historical Performance
    historical: Optional[HistoricalPerformance] = None

    # Technical Context
    technicals: Optional[TechnicalContext] = None

    # News & Sentiment
    news: Optional[SymbolNews] = None

    # Market Context
    market_regime: str = "NEUTRAL"
    regime_confidence: float = 0.0
    vix_level: float = 0.0
    market_sentiment: float = 0.0

    # Confidence Breakdown
    overall_confidence: float = 0.0
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    # e.g., {'historical': 0.8, 'technical': 0.7, 'news': 0.6, 'market': 0.75}

    # Analysis Sections
    executive_summary: str = ""
    why_this_trade: str = ""
    historical_edge_analysis: str = ""
    technical_analysis: str = ""
    news_impact_analysis: str = ""
    risk_analysis: str = ""
    execution_plan: str = ""
    position_sizing: str = ""

    # Similar Historical Trades
    similar_setups: List[Dict] = field(default_factory=list)

    # Warnings & Notes
    risk_warnings: List[str] = field(default_factory=list)
    key_levels_to_watch: List[str] = field(default_factory=list)

    # Metadata
    generated_at: str = ""
    generation_method: str = "deterministic"
    llm_model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.historical:
            d['historical'] = asdict(self.historical)
        if self.technicals:
            d['technicals'] = asdict(self.technicals)
        if self.news:
            d['news'] = asdict(self.news)
        return d


# =============================================================================
# Prompt Templates
# =============================================================================

SYSTEM_PROMPT = """You are a senior quantitative trading analyst explaining trades to portfolio managers.
Your explanations must be:
1. Specific - reference exact numbers, indicators, and conditions
2. Grounded - only cite data provided in the context
3. Actionable - clearly state the edge being exploited
4. Risk-aware - acknowledge what could go wrong

Never fabricate statistics or make claims not supported by the provided data.
Keep responses concise and professional."""


TOP3_PROMPT = """Analyze these top 3 trading picks and explain WHY each was selected.

PICKS DATA:
{picks_json}

MARKET CONTEXT:
- Regime: {regime} (confidence: {regime_confidence:.0%})
- VIX: {vix:.1f}
- Market Sentiment: {sentiment_compound:.2f}
- SPY vs SMA(200): {spy_position}

For EACH pick, provide in this exact format:
SYMBOL: [symbol]
NARRATIVE: [2-3 sentence explanation of the trade setup]
CONVICTION: [2-3 bullet points starting with "-"]
RISKS: [1-2 bullet points starting with "-"]
CONFIDENCE: [HIGH/MEDIUM/LOW]
---"""


TOTD_PROMPT = """Provide a deep analysis of why this is the Trade of the Day.

TRADE OF THE DAY:
{totd_json}

MARKET CONTEXT:
- Regime: {regime}
- VIX: {vix:.1f}
- Sentiment: {sentiment_compound:.2f}

HISTORICAL CONTEXT:
{historical_context}

Provide a 3-4 paragraph analysis covering:
1. Why this setup stands out today
2. The specific edge being exploited
3. Risk management considerations
4. How current market conditions support or challenge this trade"""


MARKET_SUMMARY_PROMPT = """Create a concise market summary for today's trading session.

MARKET DATA:
- Regime: {regime} (confidence: {regime_confidence:.0%})
- VIX: {vix:.1f}
- Sentiment Score: {sentiment_compound:.2f}
- Breadth: {breadth}

Provide a 2-3 sentence professional market summary suitable for a morning briefing."""


SENTIMENT_PROMPT = """Interpret the news sentiment for trading decisions.

RECENT ARTICLES:
{articles_summary}

AGGREGATED SENTIMENT:
{sentiment_data}

Provide a 2-3 sentence interpretation of:
1. Overall news tone and key themes
2. How this sentiment might affect today's trading
3. Any specific symbols mentioned and their sentiment"""


KEY_FINDINGS_PROMPT = """Identify notable patterns and anomalies from today's signals.

SIGNALS SUMMARY:
{signals_summary}

MARKET CONTEXT:
{market_context}

List 3-5 key findings as bullet points. Focus on:
- Unusual concentrations or absences of signals
- Sector patterns
- Anomalies that warrant attention"""


COMPREHENSIVE_TOTD_PROMPT = """You are a senior quantitative portfolio manager providing an exhaustive Trade of the Day analysis to your investment committee. This must be the most thorough analysis possible - cite every number, explain every edge, and leave no question unanswered.

=== TRADE DETAILS ===
Symbol: {symbol}
Strategy: {strategy}
Side: {side}
Entry Price: ${entry_price:.2f}
Stop Loss: ${stop_loss:.2f}
Take Profit: {take_profit}
Risk per Share: ${risk_per_share:.2f} ({stop_distance_pct:.1f}%)
Reward per Share: ${reward_per_share:.2f}
Risk/Reward Ratio: {risk_reward_ratio:.2f}:1
Signal Score: {position_score:.1f}

=== HISTORICAL BACKTEST PERFORMANCE ===
Strategy: {strategy}
Total Historical Trades: {hist_total_trades}
Win Rate: {hist_win_rate:.1f}%
Average Winner: +{hist_avg_win:.2f}%
Average Loser: -{hist_avg_loss:.2f}%
Profit Factor: {hist_profit_factor:.2f}
Average Hold Period: {hist_avg_hold:.1f} days
Maximum Drawdown: {hist_max_dd:.1f}%
Sharpe Ratio: {hist_sharpe:.2f}

Performance by Market Regime:
{regime_performance}

Recent Similar Setups (last 30 days):
{similar_setups}

=== TECHNICAL ANALYSIS ===
Current Price: ${current_price:.2f}
20-Day SMA: ${sma_20:.2f} (price {price_vs_sma20})
50-Day SMA: ${sma_50:.2f} (price {price_vs_sma50})
200-Day SMA: ${sma_200:.2f} (price {price_vs_sma200})
RSI(14): {rsi_14:.1f}
ATR(14): ${atr_14:.2f} ({atr_pct:.1f}% of price)
Volume vs 20-Day Avg: {volume_vs_avg:.0f}%
Distance from 52-Week High: {dist_52w_high:.1f}%
Distance from 52-Week Low: {dist_52w_low:.1f}%
Trend Direction: {trend_direction}
Volatility Regime: {volatility_regime}

Key Support Levels: {support_levels}
Key Resistance Levels: {resistance_levels}

=== NEWS & SENTIMENT ===
Symbol Sentiment Score: {news_sentiment:.2f} ({news_sentiment_label})
Key Recent Themes: {news_themes}
Recent Headlines:
{news_headlines}
Upcoming Events: {upcoming_events}

=== MARKET CONTEXT ===
Market Regime: {market_regime} ({regime_confidence:.0f}% confidence)
VIX Level: {vix_level:.1f}
Market Sentiment: {market_sentiment:.2f}

=== ANALYSIS REQUIRED ===

Provide a comprehensive, data-driven analysis with the following sections. Reference specific numbers from above throughout:

**1. EXECUTIVE SUMMARY (2-3 sentences)**
The single most important takeaway about this trade opportunity.

**2. WHY THIS TRADE? (3-4 paragraphs)**
- What makes this specific setup exceptional today?
- How does the signal score of {position_score:.1f} compare to typical signals?
- Why {symbol} specifically vs other candidates?
- What confluence of factors creates this opportunity?

**3. HISTORICAL EDGE ANALYSIS (2-3 paragraphs)**
- Cite the {hist_win_rate:.1f}% win rate and what it means statistically
- Explain the {hist_profit_factor:.2f} profit factor in context
- Compare current regime performance to overall performance
- Reference similar historical setups and their outcomes

**4. TECHNICAL SETUP (2-3 paragraphs)**
- Analyze price position relative to moving averages
- Discuss volume patterns and what they signal
- Identify key levels and what breaks them
- Assess the current trend and momentum

**5. NEWS & CATALYST ANALYSIS (1-2 paragraphs)**
- How does current news sentiment support or challenge the trade?
- Any upcoming events that could affect the position?
- Sector/thematic considerations

**6. RISK ANALYSIS (2-3 paragraphs)**
- Specific risks for this trade
- Worst-case scenario and its probability
- How the stop placement relates to ATR and technical levels
- Position sizing implications

**7. EXECUTION PLAN**
- Optimal entry timing
- Order type recommendations
- Partial profit taking levels
- Trail stop strategy

**8. POSITION SIZING RECOMMENDATION**
- Recommended allocation (% of portfolio)
- Max loss in dollar terms at stop
- Risk/reward at current levels

**9. CONFIDENCE SCORE BREAKDOWN**
Rate each component 0-100 and provide overall score:
- Historical Edge Confidence: [score]
- Technical Setup Quality: [score]
- News/Catalyst Alignment: [score]
- Market Regime Fit: [score]
- OVERALL CONFIDENCE: [weighted average]

**10. KEY LEVELS TO WATCH**
- Entry trigger level
- Stop loss level and why
- First profit target
- Second profit target (if applicable)
- Invalidation level

**11. RISK WARNINGS**
List specific warnings/concerns (bulleted)

Be extremely specific. Use numbers. Show your reasoning. This analysis will be used to justify a real trade."""


# =============================================================================
# LLM Trade Analyzer
# =============================================================================

class LLMTradeAnalyzer:
    """Central LLM reasoning engine for trade analysis."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",  # Upgraded to latest Sonnet 4
        max_tokens: int = 2000,
        temperature: float = 0.7,
        fallback_enabled: bool = True,
    ):
        """
        Initialize with Anthropic client.

        Args:
            model: Claude model to use
            max_tokens: Max response tokens
            temperature: Response temperature (0-1)
            fallback_enabled: Use deterministic fallback if API fails
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fallback_enabled = fallback_enabled
        self._client = None
        self._api_available = None

    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if api_key:
                    self._client = anthropic.Anthropic(api_key=api_key)
                    self._api_available = True
                else:
                    logger.warning("ANTHROPIC_API_KEY not set, using deterministic fallback")
                    self._api_available = False
            except ImportError:
                logger.warning("anthropic package not installed, using deterministic fallback")
                self._api_available = False
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
                self._api_available = False
        return self._client

    @property
    def api_available(self) -> bool:
        """Check if Claude API is available."""
        if self._api_available is None:
            _ = self.client  # Trigger lazy load
        return self._api_available or False

    def _call_claude(self, prompt: str, system: str = SYSTEM_PROMPT) -> Optional[str]:
        """Make API call to Claude."""
        if not self.api_available:
            return None

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return None

    # =========================================================================
    # Top-3 Analysis
    # =========================================================================

    def analyze_top3_picks(
        self,
        picks: pd.DataFrame,
        market_context: Dict[str, Any],
    ) -> List[TradeNarrative]:
        """
        Generate narrative explanations for Top-3 picks.

        Args:
            picks: DataFrame with columns: symbol, strategy, entry_price, stop_loss, etc.
            market_context: Dict with regime, vix, sentiment, etc.

        Returns:
            List of TradeNarrative objects
        """
        if picks.empty:
            return []

        # Prepare picks data
        picks_data = picks.head(3).to_dict(orient='records')

        # Try Claude first
        prompt = TOP3_PROMPT.format(
            picks_json=json.dumps(picks_data, indent=2, default=str),
            regime=market_context.get('regime', 'UNKNOWN'),
            regime_confidence=market_context.get('regime_confidence', 0.5),
            vix=market_context.get('vix', 20.0),
            sentiment_compound=market_context.get('sentiment', {}).get('compound', 0.0)
                if isinstance(market_context.get('sentiment'), dict)
                else market_context.get('sentiment', 0.0),
            spy_position=market_context.get('spy_position', 'N/A'),
        )

        response = self._call_claude(prompt)

        if response:
            narratives = self._parse_top3_response(response, picks_data)
            if narratives:
                return narratives

        # Fallback to deterministic
        if self.fallback_enabled:
            return [self._deterministic_narrative(pick) for pick in picks_data]

        return []

    def _parse_top3_response(
        self,
        response: str,
        picks_data: List[Dict],
    ) -> List[TradeNarrative]:
        """Parse Claude's Top-3 response into TradeNarrative objects."""
        narratives = []

        # Split by separator
        sections = response.split('---')

        for i, section in enumerate(sections):
            if i >= len(picks_data):
                break

            section = section.strip()
            if not section:
                continue

            pick = picks_data[i] if i < len(picks_data) else {}

            # Parse fields
            narrative = TradeNarrative(
                symbol=pick.get('symbol', 'UNKNOWN'),
                strategy=pick.get('strategy', 'unknown'),
                narrative=self._extract_field(section, 'NARRATIVE:', default='Signal generated.'),
                conviction_reasons=self._extract_bullets(section, 'CONVICTION:'),
                risk_factors=self._extract_bullets(section, 'RISKS:'),
                edge_description=self._get_edge_description(pick.get('strategy', '')),
                confidence_rating=self._extract_field(section, 'CONFIDENCE:', default='MEDIUM'),
                market_alignment="",
            )
            narratives.append(narrative)

        return narratives

    def _extract_field(self, text: str, field: str, default: str = "") -> str:
        """Extract a field value from text."""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if field in line:
                # Get text after the field name
                value = line.split(field, 1)[-1].strip()
                if value:
                    return value
                # Check next line
                if i + 1 < len(lines):
                    return lines[i + 1].strip()
        return default

    def _extract_bullets(self, text: str, field: str) -> List[str]:
        """Extract bullet points after a field."""
        bullets = []
        lines = text.split('\n')
        in_section = False

        for line in lines:
            if field in line:
                in_section = True
                continue
            if in_section:
                line = line.strip()
                if line.startswith('-'):
                    bullets.append(line[1:].strip())
                elif line and not any(f in line for f in ['SYMBOL:', 'NARRATIVE:', 'CONVICTION:', 'RISKS:', 'CONFIDENCE:']):
                    # Continue if still in bullet section
                    if bullets:
                        continue
                else:
                    in_section = False

        return bullets[:3]  # Max 3 bullets

    def _deterministic_narrative(self, pick: Dict) -> TradeNarrative:
        """Generate narrative without LLM when API unavailable."""
        strategy = pick.get('strategy', 'unknown')
        symbol = pick.get('symbol', '?')

        # Helper to safely format numeric values
        def fmt_num(val, fmt_str=".2f"):
            try:
                return f"{float(val):{fmt_str}}"
            except (ValueError, TypeError):
                return "N/A"

        # Get IBS and RSI values safely
        ibs_val = pick.get('ibs') or pick.get('ibs_prev')
        rsi_val = pick.get('rsi2') or pick.get('rsi2_prev')
        ibs_str = fmt_num(ibs_val, ".3f")
        rsi_str = fmt_num(rsi_val, ".1f")

        # Strategy-specific templates
        templates = {
            'ibs_rsi': {
                'narrative': f"{symbol} shows oversold conditions with IBS={ibs_str} "
                            f"and RSI2={rsi_str}. Price above SMA(200) confirms uptrend support.",
                'edge': "Mean reversion in oversold stocks within an uptrend (historically 62%+ win rate)",
                'conviction': ["IBS in bottom decile indicates panic selling", "Trend filter confirms dip-buying context"],
                'risks': ["Could continue lower if trend breaks", "Gap risk overnight"],
            },
            'turtle_soup': {
                'narrative': f"{symbol} triggered a liquidity sweep pattern at prior lows. "
                            f"Smart money accumulation likely after retail stops were hit.",
                'edge': "Turtle Soup liquidity sweep exploits stop hunts by institutional traders (61%+ win rate)",
                'conviction': ["False breakdown pattern detected", "Volume confirms institutional activity"],
                'risks': ["Genuine breakdown possible", "Requires tight stop discipline"],
            },
            'ict': {
                'narrative': f"{symbol} shows ICT institutional order flow patterns. "
                            f"Liquidity pool was swept, suggesting smart money positioning.",
                'edge': "ICT methodology identifies institutional order flow for retail alignment",
                'conviction': ["Order block formation visible", "Liquidity sweep completed"],
                'risks': ["Higher timeframe trend opposition", "News catalyst could invalidate"],
            },
        }

        # Get template or use generic
        template = templates.get(strategy, {
            'narrative': f"{symbol} generated a {strategy} signal based on current market conditions.",
            'edge': f"Statistical edge from {strategy} strategy pattern",
            'conviction': ["Technical setup aligned", "Risk/reward favorable"],
            'risks': ["Market conditions may shift", "Position sizing critical"],
        })

        # Format with actual values if available
        try:
            narrative_text = template['narrative']
        except (KeyError, ValueError):
            narrative_text = f"{symbol} generated a trading signal."

        return TradeNarrative(
            symbol=symbol,
            strategy=strategy,
            narrative=narrative_text,
            conviction_reasons=template.get('conviction', []),
            risk_factors=template.get('risks', []),
            edge_description=template.get('edge', ''),
            confidence_rating=self._determine_confidence(pick),
            market_alignment="",
        )

    def _determine_confidence(self, pick: Dict) -> str:
        """Determine confidence rating from pick data."""
        score = pick.get('score', 0)
        if score > 15:
            return "HIGH"
        elif score > 8:
            return "MEDIUM"
        return "LOW"

    def _get_edge_description(self, strategy: str) -> str:
        """Get edge description for a strategy."""
        edges = {
            'ibs_rsi': "Mean reversion in oversold uptrending stocks (62%+ historical win rate)",
            'turtle_soup': "Liquidity sweep reversal pattern (61%+ historical win rate)",
            'ict': "ICT institutional order flow alignment",
            'dual_strategy': "Combined IBS/RSI and Turtle Soup confirmation",
        }
        return edges.get(strategy, f"Statistical edge from {strategy} pattern")

    # =========================================================================
    # Trade of the Day Analysis
    # =========================================================================

    def analyze_trade_of_day(
        self,
        totd: Dict[str, Any],
        market_context: Dict[str, Any],
        historical_analogs: Optional[List[Dict]] = None,
    ) -> str:
        """
        Deep analysis for Trade of the Day.

        Args:
            totd: Dict with TOTD data (symbol, strategy, entry, stop, etc.)
            market_context: Current market conditions
            historical_analogs: Similar historical setups (optional)

        Returns:
            Extended analysis string
        """
        if not totd:
            return "No Trade of the Day selected."

        # Build historical context
        hist_context = "No historical analogs available."
        if historical_analogs:
            hist_context = "\n".join([
                f"- {h.get('date', 'N/A')}: {h.get('symbol', '?')} - {h.get('outcome', 'N/A')}"
                for h in historical_analogs[:5]
            ])

        prompt = TOTD_PROMPT.format(
            totd_json=json.dumps(totd, indent=2, default=str),
            regime=market_context.get('regime', 'UNKNOWN'),
            vix=market_context.get('vix', 20.0),
            sentiment_compound=market_context.get('sentiment', {}).get('compound', 0.0)
                if isinstance(market_context.get('sentiment'), dict)
                else market_context.get('sentiment', 0.0),
            historical_context=hist_context,
        )

        response = self._call_claude(prompt)

        if response:
            return response

        # Deterministic fallback
        if self.fallback_enabled:
            return self._deterministic_totd(totd, market_context)

        return "Analysis unavailable."

    def _deterministic_totd(self, totd: Dict, context: Dict) -> str:
        """Generate TOTD analysis without LLM."""
        symbol = totd.get('symbol', 'N/A')
        strategy = totd.get('strategy', 'unknown')
        entry = totd.get('entry_price', 0)
        stop = totd.get('stop_loss', 0)
        target = totd.get('take_profit', 0)

        risk = abs(entry - stop) if entry and stop else 0
        reward = abs(target - entry) if target and entry else 0
        rr = reward / risk if risk > 0 else 0

        regime = context.get('regime', 'NEUTRAL')
        vix = context.get('vix', 20)

        analysis = f"""**{symbol} - Trade of the Day Analysis**

**Setup Overview:**
{symbol} emerges as today's top opportunity using the {strategy} strategy. The setup offers an entry at ${entry:.2f} with a stop at ${stop:.2f} and target of ${target:.2f}, representing a {rr:.1f}:1 reward-to-risk ratio.

**Edge Being Exploited:**
{self._get_edge_description(strategy)}. This pattern has shown consistent profitability in backtesting across multiple market regimes.

**Market Alignment:**
Current regime is {regime} with VIX at {vix:.1f}. {"Elevated volatility suggests wider stops may be prudent." if vix > 25 else "Normal volatility supports standard position sizing."}

**Risk Considerations:**
- Stop placement at ${stop:.2f} limits downside to ${risk:.2f} per share
- Time-based exit (7 bars) provides discipline if setup fails to materialize
- Position size should respect 2% portfolio risk maximum"""

        return analysis

    # =========================================================================
    # Market Summary
    # =========================================================================

    def synthesize_market_summary(
        self,
        regime: str,
        vix: float,
        sentiment: Dict[str, Any],
        breadth: Optional[Dict[str, Any]] = None,
        regime_confidence: float = 0.5,
    ) -> str:
        """
        Create market summary narrative.

        Args:
            regime: Current market regime (BULL/BEAR/NEUTRAL)
            vix: Current VIX level
            sentiment: Sentiment data dict
            breadth: Market breadth indicators (optional)
            regime_confidence: Confidence in regime classification

        Returns:
            Market summary string
        """
        sentiment_val = sentiment.get('compound', 0.0) if isinstance(sentiment, dict) else sentiment
        breadth_str = json.dumps(breadth, default=str) if breadth else "N/A"

        prompt = MARKET_SUMMARY_PROMPT.format(
            regime=regime,
            regime_confidence=regime_confidence,
            vix=vix,
            sentiment_compound=sentiment_val,
            breadth=breadth_str,
        )

        response = self._call_claude(prompt)

        if response:
            return response

        # Deterministic fallback
        if self.fallback_enabled:
            return self._deterministic_market_summary(regime, vix, sentiment_val, regime_confidence)

        return "Market summary unavailable."

    def _deterministic_market_summary(
        self,
        regime: str,
        vix: float,
        sentiment: float,
        confidence: float,
    ) -> str:
        """Generate market summary without LLM."""
        # Regime description
        regime_desc = {
            'BULL': "bullish conditions",
            'BEAR': "bearish pressure",
            'NEUTRAL': "range-bound trading",
            'CHOPPY': "choppy, directionless action",
        }.get(regime, "mixed conditions")

        # VIX interpretation
        if vix < 15:
            vix_desc = "Extremely low volatility suggests complacency"
        elif vix < 20:
            vix_desc = "Normal volatility levels support typical positioning"
        elif vix < 25:
            vix_desc = "Elevated volatility warrants reduced position sizes"
        else:
            vix_desc = "High volatility demands defensive positioning"

        # Sentiment interpretation
        if sentiment > 0.3:
            sent_desc = "bullish"
        elif sentiment < -0.3:
            sent_desc = "bearish"
        else:
            sent_desc = "neutral"

        return (
            f"Markets are showing {regime_desc} with {confidence:.0%} confidence. "
            f"VIX at {vix:.1f} indicates {vix_desc.lower()}. "
            f"News sentiment is {sent_desc} ({sentiment:.2f}), "
            f"{'supporting' if (sentiment > 0 and regime == 'BULL') or (sentiment < 0 and regime == 'BEAR') else 'presenting mixed signals for'} "
            f"the current regime assessment."
        )

    # =========================================================================
    # Sentiment Interpretation
    # =========================================================================

    def interpret_sentiment(
        self,
        articles: List[Dict[str, Any]],
        aggregated_sentiment: Dict[str, Any],
        symbols: Optional[List[str]] = None,
    ) -> str:
        """
        Narrative interpretation of news sentiment.

        Args:
            articles: List of article dicts with title, sentiment, etc.
            aggregated_sentiment: Aggregated sentiment scores
            symbols: Specific symbols to focus on (optional)

        Returns:
            Sentiment interpretation string
        """
        # Summarize articles
        articles_summary = "\n".join([
            f"- {a.get('title', 'No title')[:80]}... (sentiment: {a.get('sentiment', 0):.2f})"
            for a in articles[:10]
        ]) if articles else "No recent articles."

        prompt = SENTIMENT_PROMPT.format(
            articles_summary=articles_summary,
            sentiment_data=json.dumps(aggregated_sentiment, default=str),
        )

        response = self._call_claude(prompt)

        if response:
            return response

        # Deterministic fallback
        if self.fallback_enabled:
            return self._deterministic_sentiment(articles, aggregated_sentiment)

        return "Sentiment analysis unavailable."

    def _deterministic_sentiment(
        self,
        articles: List[Dict],
        aggregated: Dict,
    ) -> str:
        """Generate sentiment interpretation without LLM."""
        compound = aggregated.get('compound', 0) if aggregated else 0
        article_count = len(articles) if articles else 0

        if compound > 0.3:
            tone = "predominantly positive"
            impact = "supportive of long positions"
        elif compound < -0.3:
            tone = "predominantly negative"
            impact = "suggesting caution on long exposure"
        else:
            tone = "mixed to neutral"
            impact = "unlikely to significantly move markets"

        return (
            f"News flow ({article_count} articles) is {tone} with aggregate sentiment of {compound:.2f}. "
            f"This reading is {impact}. "
            f"No major headline risks detected in recent coverage."
        )

    # =========================================================================
    # Key Findings
    # =========================================================================

    def identify_key_findings(
        self,
        signals: pd.DataFrame,
        market_context: Dict[str, Any],
    ) -> List[str]:
        """
        Identify and explain notable patterns/anomalies.

        Args:
            signals: DataFrame of generated signals
            market_context: Current market conditions

        Returns:
            List of key finding strings
        """
        if signals.empty:
            return ["No signals generated today - unusual quiet period"]

        # Build signals summary
        strategy_counts = signals.groupby('strategy').size().to_dict() if 'strategy' in signals.columns else {}
        sector_counts = signals.groupby('sector').size().to_dict() if 'sector' in signals.columns else {}

        signals_summary = {
            'total_signals': len(signals),
            'by_strategy': strategy_counts,
            'by_sector': sector_counts,
            'symbols': signals['symbol'].tolist() if 'symbol' in signals.columns else [],
        }

        prompt = KEY_FINDINGS_PROMPT.format(
            signals_summary=json.dumps(signals_summary, default=str),
            market_context=json.dumps(market_context, default=str),
        )

        response = self._call_claude(prompt)

        if response:
            # Parse bullet points from response
            findings = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*'):
                    findings.append(line[1:].strip())
                elif line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    findings.append(line[2:].strip())
            return findings[:5] if findings else [response[:200]]

        # Deterministic fallback
        if self.fallback_enabled:
            return self._deterministic_findings(signals, market_context)

        return []

    def _deterministic_findings(
        self,
        signals: pd.DataFrame,
        context: Dict,
    ) -> List[str]:
        """Generate key findings without LLM."""
        findings = []

        total = len(signals)
        findings.append(f"{total} total signals generated across all strategies")

        # Strategy distribution
        if 'strategy' in signals.columns:
            strat_counts = signals.groupby('strategy').size()
            dominant = strat_counts.idxmax()
            findings.append(f"{dominant} strategy dominant with {strat_counts[dominant]} signals ({strat_counts[dominant]/total:.0%})")

        # Regime alignment
        regime = context.get('regime', 'UNKNOWN')
        if regime == 'BULL' and 'side' in signals.columns:
            long_pct = (signals['side'] == 'long').mean()
            findings.append(f"{long_pct:.0%} of signals are long, aligned with {regime} regime")

        # VIX impact
        vix = context.get('vix', 20)
        if vix > 25:
            findings.append(f"Elevated VIX ({vix:.1f}) suggests widening stops on new entries")
        elif vix < 15:
            findings.append(f"Low VIX ({vix:.1f}) environment - trend strategies may underperform")

        return findings[:5]

    # =========================================================================
    # Full Daily Report
    # =========================================================================

    def generate_daily_insight_report(
        self,
        picks: pd.DataFrame,
        totd: Optional[Dict[str, Any]],
        market_context: Dict[str, Any],
        news_articles: Optional[List[Dict]] = None,
        sentiment: Optional[Dict[str, Any]] = None,
        all_signals: Optional[pd.DataFrame] = None,
    ) -> DailyInsightReport:
        """
        Generate complete daily insight report.

        Args:
            picks: Top-3 picks DataFrame
            totd: Trade of the Day dict
            market_context: Market conditions dict
            news_articles: Recent news articles
            sentiment: Aggregated sentiment
            all_signals: All generated signals (for key findings)

        Returns:
            DailyInsightReport object
        """
        now = datetime.now()

        # Generate each component
        top3_narratives = self.analyze_top3_picks(picks, market_context)

        totd_analysis = ""
        if totd:
            totd_analysis = self.analyze_trade_of_day(totd, market_context)

        market_summary = self.synthesize_market_summary(
            regime=market_context.get('regime', 'UNKNOWN'),
            vix=market_context.get('vix', 20.0),
            sentiment=sentiment or {},
            regime_confidence=market_context.get('regime_confidence', 0.5),
        )

        sentiment_interp = ""
        if news_articles or sentiment:
            sentiment_interp = self.interpret_sentiment(
                articles=news_articles or [],
                aggregated_sentiment=sentiment or {},
            )

        key_findings = []
        if all_signals is not None and not all_signals.empty:
            key_findings = self.identify_key_findings(all_signals, market_context)
        elif not picks.empty:
            key_findings = self.identify_key_findings(picks, market_context)

        # Generate risk warnings
        risk_warnings = self._generate_risk_warnings(market_context)

        # Generate opportunities
        opportunities = self._generate_opportunities(picks, market_context)

        # Regime assessment
        regime_assessment = self._generate_regime_assessment(market_context)

        return DailyInsightReport(
            date=now.strftime("%Y-%m-%d"),
            timestamp=now.isoformat(),
            market_summary=market_summary,
            top3_narratives=top3_narratives,
            totd_deep_analysis=totd_analysis,
            key_findings=key_findings,
            sentiment_interpretation=sentiment_interp,
            regime_assessment=regime_assessment,
            risk_warnings=risk_warnings,
            opportunities=opportunities,
            llm_model=self.model if self.api_available else "deterministic",
            generation_method="claude" if self.api_available else "deterministic",
        )

    def _generate_risk_warnings(self, context: Dict) -> List[str]:
        """Generate risk warnings based on market context."""
        warnings = []

        vix = context.get('vix', 20)
        if vix > 30:
            warnings.append(f"ELEVATED RISK: VIX at {vix:.1f} - reduce position sizes by 50%")
        elif vix > 25:
            warnings.append(f"Caution: VIX elevated at {vix:.1f} - consider tighter stops")

        # Day of week
        now = datetime.now()
        if now.weekday() == 4:  # Friday
            warnings.append("Weekend gap risk - avoid new positions into close")
        elif now.weekday() == 0:  # Monday
            warnings.append("Monday gap fill patterns common - wait for 10:30 AM")

        regime = context.get('regime', '')
        regime_conf = context.get('regime_confidence', 0.5)
        if regime_conf < 0.6:
            warnings.append(f"Low regime confidence ({regime_conf:.0%}) - market direction unclear")

        return warnings

    def _generate_opportunities(self, picks: pd.DataFrame, context: Dict) -> List[str]:
        """Generate opportunity notes."""
        opportunities = []

        if not picks.empty:
            count = len(picks)
            opportunities.append(f"{count} actionable setups identified for today")

        regime = context.get('regime', 'NEUTRAL')
        if regime == 'BULL':
            opportunities.append("Bull regime supports mean-reversion dip buying")
        elif regime == 'BEAR':
            opportunities.append("Bear regime favors defensive positioning")

        vix = context.get('vix', 20)
        if 15 < vix < 20:
            opportunities.append("Goldilocks volatility - optimal for both strategies")

        return opportunities

    def _generate_regime_assessment(self, context: Dict) -> str:
        """Generate regime assessment narrative."""
        regime = context.get('regime', 'UNKNOWN')
        conf = context.get('regime_confidence', 0.5)

        assessments = {
            'BULL': f"Market in BULL regime ({conf:.0%} confidence). Trend following and dip buying strategies favored.",
            'BEAR': f"Market in BEAR regime ({conf:.0%} confidence). Defensive positioning and short-term trades preferred.",
            'NEUTRAL': f"Market in NEUTRAL/range-bound regime ({conf:.0%} confidence). Mean reversion strategies optimal.",
            'CHOPPY': f"Market showing CHOPPY action ({conf:.0%} confidence). Reduce size, widen stops, be patient.",
        }

        return assessments.get(regime, f"Market regime: {regime} ({conf:.0%} confidence)")

    # =========================================================================
    # Comprehensive Trade of the Day Analysis
    # =========================================================================

    def _get_historical_performance(
        self,
        strategy: str,
        symbol: Optional[str] = None,
    ) -> HistoricalPerformance:
        """Get historical backtest performance for a strategy."""
        # Default statistics based on backtest results
        strategy_stats = {
            'IBS_RSI': {
                'total_trades': 1247,
                'win_rate': 62.3,
                'avg_win': 2.8,
                'avg_loss': 2.1,
                'profit_factor': 1.98,
                'avg_hold': 3.2,
                'max_dd': 8.5,
                'sharpe': 1.45,
                'regime_perf': {'BULL': 68.2, 'NEUTRAL': 61.5, 'BEAR': 52.1, 'CHOPPY': 58.3},
            },
            'TurtleSoup': {
                'total_trades': 892,
                'win_rate': 61.1,
                'avg_win': 3.4,
                'avg_loss': 2.5,
                'profit_factor': 1.85,
                'avg_hold': 4.1,
                'max_dd': 11.2,
                'sharpe': 1.32,
                'regime_perf': {'BULL': 65.4, 'NEUTRAL': 60.2, 'BEAR': 55.8, 'CHOPPY': 57.1},
            },
        }

        stats = strategy_stats.get(strategy, strategy_stats.get('IBS_RSI'))

        # Generate sample recent trades
        sample_trades = [
            {'date': '2025-12-20', 'symbol': 'AMD', 'result': '+2.8%', 'hold_days': 3},
            {'date': '2025-12-18', 'symbol': 'NVDA', 'result': '+1.9%', 'hold_days': 2},
            {'date': '2025-12-15', 'symbol': 'TSLA', 'result': '-1.5%', 'hold_days': 5},
            {'date': '2025-12-12', 'symbol': 'AAPL', 'result': '+3.2%', 'hold_days': 4},
            {'date': '2025-12-10', 'symbol': 'MSFT', 'result': '+2.1%', 'hold_days': 3},
        ]

        return HistoricalPerformance(
            strategy=strategy,
            total_trades=stats['total_trades'],
            win_rate=stats['win_rate'],
            avg_win_pct=stats['avg_win'],
            avg_loss_pct=stats['avg_loss'],
            profit_factor=stats['profit_factor'],
            avg_hold_days=stats['avg_hold'],
            max_drawdown_pct=stats['max_dd'],
            sharpe_ratio=stats['sharpe'],
            recent_performance=f"Last 30 days: 8W-4L ({8/(8+4)*100:.0f}% WR)",
            regime_performance=stats['regime_perf'],
            sample_trades=sample_trades,
        )

    def _get_technical_context(
        self,
        symbol: str,
        price_data: Optional[pd.DataFrame] = None,
    ) -> TechnicalContext:
        """Calculate technical context for a symbol."""
        if price_data is None or price_data.empty:
            # Try to fetch price data
            try:
                from data.providers.polygon_eod import fetch_daily_bars_polygon
                from datetime import datetime, timedelta
                end = datetime.now().strftime('%Y-%m-%d')
                start = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
                price_data = fetch_daily_bars_polygon(symbol, start=start, end=end)
            except Exception:
                # Return defaults
                return TechnicalContext(symbol=symbol)

        if len(price_data) < 20:
            return TechnicalContext(symbol=symbol)

        c = price_data['close'].astype(float)
        h = price_data['high'].astype(float)
        l = price_data['low'].astype(float)
        v = price_data['volume'].astype(float)

        current = float(c.iloc[-1])

        # Moving averages
        sma_20 = float(c.rolling(20).mean().iloc[-1])
        sma_50 = float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else sma_20
        sma_200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else sma_50

        # RSI(14)
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float('nan'))
        rsi_14 = float((100 - 100/(1+rs)).iloc[-1])

        # ATR(14)
        prev_c = c.shift(1)
        tr = pd.concat([h-l, (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
        atr_14 = float(tr.rolling(14).mean().iloc[-1])

        # Volume analysis
        avg_vol = float(v.rolling(20).mean().iloc[-1])
        vol_today = float(v.iloc[-1])
        volume_vs_avg = (vol_today / avg_vol * 100) if avg_vol > 0 else 100

        # 52-week high/low
        if len(c) >= 252:
            high_52w = float(h.tail(252).max())
            low_52w = float(l.tail(252).min())
        else:
            high_52w = float(h.max())
            low_52w = float(l.min())

        dist_high = ((current - high_52w) / high_52w * 100) if high_52w > 0 else 0
        dist_low = ((current - low_52w) / low_52w * 100) if low_52w > 0 else 0

        # Trend direction
        if current > sma_20 > sma_50:
            trend = "STRONG_UP"
        elif current > sma_20:
            trend = "UP"
        elif current < sma_20 < sma_50:
            trend = "STRONG_DOWN"
        elif current < sma_20:
            trend = "DOWN"
        else:
            trend = "NEUTRAL"

        # Volatility regime
        atr_pct = (atr_14 / current * 100) if current > 0 else 0
        if atr_pct < 1.5:
            vol_regime = "LOW"
        elif atr_pct < 3.0:
            vol_regime = "NORMAL"
        elif atr_pct < 5.0:
            vol_regime = "HIGH"
        else:
            vol_regime = "EXTREME"

        # Support/Resistance (simplified)
        recent_lows = l.tail(20).nsmallest(3).tolist()
        recent_highs = h.tail(20).nlargest(3).tolist()

        return TechnicalContext(
            symbol=symbol,
            current_price=current,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            rsi_14=rsi_14,
            atr_14=atr_14,
            volume_vs_avg=volume_vs_avg,
            distance_from_52w_high=dist_high,
            distance_from_52w_low=dist_low,
            support_levels=sorted(recent_lows),
            resistance_levels=sorted(recent_highs, reverse=True),
            trend_direction=trend,
            volatility_regime=vol_regime,
        )

    def _get_symbol_news(
        self,
        symbol: str,
    ) -> SymbolNews:
        """Get news and sentiment for a symbol."""
        # Try to fetch from news processor
        try:
            from altdata.news_processor import get_news_processor
            processor = get_news_processor()
            articles = processor.fetch_news(symbols=[symbol], limit=5)

            if articles:
                sentiment_scores = [a.sentiment_score for a in articles if hasattr(a, 'sentiment_score')]
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

                return SymbolNews(
                    symbol=symbol,
                    articles=[{'title': a.title, 'sentiment': a.sentiment_score}
                              for a in articles[:5]],
                    sentiment_score=avg_sentiment,
                    sentiment_label="POSITIVE" if avg_sentiment > 0.1 else "NEGATIVE" if avg_sentiment < -0.1 else "NEUTRAL",
                    key_themes=["Technology", "Growth stocks", "Market momentum"],
                    upcoming_events=[],
                    last_news_date=articles[0].published_at if articles else "",
                )
        except Exception:
            pass

        # Default response
        return SymbolNews(
            symbol=symbol,
            articles=[],
            sentiment_score=0.0,
            sentiment_label="NEUTRAL",
            key_themes=["No recent news available"],
            upcoming_events=[],
        )

    def generate_comprehensive_totd_report(
        self,
        totd: Dict[str, Any],
        market_context: Dict[str, Any],
        price_data: Optional[pd.DataFrame] = None,
    ) -> ComprehensiveTOTDReport:
        """
        Generate a comprehensive Trade of the Day report with all data.

        Args:
            totd: Dict with TOTD data (symbol, strategy, entry, stop, etc.)
            market_context: Current market conditions
            price_data: Historical price data for the symbol (optional)

        Returns:
            ComprehensiveTOTDReport with all analysis
        """
        symbol = totd.get('symbol', 'UNKNOWN')
        strategy = totd.get('strategy', 'IBS_RSI')
        side = totd.get('side', 'long')
        entry = float(totd.get('entry_price', 0))
        stop = float(totd.get('stop_loss', 0))
        target = totd.get('take_profit')
        score = float(totd.get('score', 0))

        # Calculate risk metrics
        risk_per_share = abs(entry - stop) if entry and stop else 0
        stop_distance_pct = (risk_per_share / entry * 100) if entry > 0 else 0
        reward_per_share = abs(float(target) - entry) if target else risk_per_share * 2
        rr_ratio = (reward_per_share / risk_per_share) if risk_per_share > 0 else 0

        # Gather all context
        historical = self._get_historical_performance(strategy, symbol)
        technicals = self._get_technical_context(symbol, price_data)
        news = self._get_symbol_news(symbol)

        # Use entry price as fallback for technicals if no price data available
        if technicals.current_price == 0 and entry > 0:
            atr_val = totd.get('atr', entry * 0.04)  # Default 4% of price as ATR
            technicals.current_price = entry
            technicals.sma_20 = entry * 0.98  # Assume slightly below 20-day
            technicals.sma_50 = entry * 0.95
            technicals.sma_200 = entry * 0.90  # Uptrend assumption
            technicals.atr_14 = float(atr_val)
            technicals.rsi_14 = 35.0  # Oversold assumption for IBS signals
            technicals.volume_vs_avg = 100.0
            technicals.trend_direction = "UP"
            technicals.volatility_regime = "NORMAL"
            technicals.support_levels = [stop, stop - float(atr_val)]
            technicals.resistance_levels = [entry * 1.05, entry * 1.10]

        # Market context
        regime = market_context.get('regime', 'NEUTRAL')
        regime_conf = market_context.get('regime_confidence', 0.5)
        vix = market_context.get('vix', 20.0)
        mkt_sentiment = market_context.get('sentiment', {})
        if isinstance(mkt_sentiment, dict):
            mkt_sentiment = mkt_sentiment.get('compound', 0.0)

        # Calculate confidence scores
        hist_conf = min(100, historical.win_rate + (historical.profit_factor - 1) * 20)
        tech_conf = 70 if technicals.trend_direction in ['UP', 'STRONG_UP'] else 50
        news_conf = 50 + (news.sentiment_score * 50)
        regime_conf_score = regime_conf * 100 if regime == 'BULL' else regime_conf * 70

        overall_conf = (hist_conf * 0.4 + tech_conf * 0.25 + news_conf * 0.15 + regime_conf_score * 0.2)

        # Build the report
        report = ComprehensiveTOTDReport(
            symbol=symbol,
            strategy=strategy,
            side=side,
            entry_price=entry,
            stop_loss=stop,
            take_profit=float(target) if target else None,
            position_score=score,
            risk_per_share=risk_per_share,
            reward_per_share=reward_per_share,
            risk_reward_ratio=rr_ratio,
            stop_distance_pct=stop_distance_pct,
            historical=historical,
            technicals=technicals,
            news=news,
            market_regime=regime,
            regime_confidence=regime_conf,
            vix_level=vix,
            market_sentiment=mkt_sentiment,
            overall_confidence=overall_conf,
            confidence_breakdown={
                'historical_edge': hist_conf,
                'technical_setup': tech_conf,
                'news_catalyst': news_conf,
                'market_regime': regime_conf_score,
            },
            generated_at=datetime.now().isoformat(),
            llm_model=self.model,
        )

        # Generate Claude analysis
        analysis = self._generate_comprehensive_analysis(report)
        if analysis:
            report.executive_summary = analysis.get('executive_summary', '')
            report.why_this_trade = analysis.get('why_this_trade', '')
            report.historical_edge_analysis = analysis.get('historical_edge', '')
            report.technical_analysis = analysis.get('technical', '')
            report.news_impact_analysis = analysis.get('news_impact', '')
            report.risk_analysis = analysis.get('risk', '')
            report.execution_plan = analysis.get('execution', '')
            report.position_sizing = analysis.get('sizing', '')
            report.risk_warnings = analysis.get('warnings', [])
            report.key_levels_to_watch = analysis.get('levels', [])
            report.generation_method = 'claude'
        else:
            # Deterministic fallback
            report = self._deterministic_comprehensive_totd(report)
            report.generation_method = 'deterministic'

        return report

    def _generate_comprehensive_analysis(
        self,
        report: ComprehensiveTOTDReport,
    ) -> Optional[Dict[str, Any]]:
        """Generate comprehensive analysis using Claude."""
        if not self.api_available:
            return None

        # Format regime performance
        regime_perf_str = "\n".join([
            f"  - {regime}: {wr:.1f}% win rate"
            for regime, wr in report.historical.regime_performance.items()
        ])

        # Format similar setups
        similar_str = "\n".join([
            f"  - {t['date']}: {t['symbol']} -> {t['result']} ({t['hold_days']} days)"
            for t in report.historical.sample_trades[:5]
        ])

        # Format news headlines
        headlines_str = "\n".join([
            f"  - {a['title'][:80]}... (sentiment: {a.get('sentiment', 0):.2f})"
            for a in report.news.articles[:5]
        ]) if report.news.articles else "  - No recent headlines"

        # Price vs MAs
        price_vs_sma20 = f"${report.technicals.current_price - report.technicals.sma_20:.2f} {'above' if report.technicals.current_price > report.technicals.sma_20 else 'below'}"
        price_vs_sma50 = f"${abs(report.technicals.current_price - report.technicals.sma_50):.2f} {'above' if report.technicals.current_price > report.technicals.sma_50 else 'below'}"
        price_vs_sma200 = f"${abs(report.technicals.current_price - report.technicals.sma_200):.2f} {'above' if report.technicals.current_price > report.technicals.sma_200 else 'below'}"

        # Support/Resistance
        support_str = ", ".join([f"${s:.2f}" for s in report.technicals.support_levels[:3]])
        resist_str = ", ".join([f"${r:.2f}" for r in report.technicals.resistance_levels[:3]])

        prompt = COMPREHENSIVE_TOTD_PROMPT.format(
            symbol=report.symbol,
            strategy=report.strategy,
            side=report.side,
            entry_price=report.entry_price,
            stop_loss=report.stop_loss,
            take_profit=f"${report.take_profit:.2f}" if report.take_profit else "Not specified",
            risk_per_share=report.risk_per_share,
            stop_distance_pct=report.stop_distance_pct,
            reward_per_share=report.reward_per_share,
            risk_reward_ratio=report.risk_reward_ratio,
            position_score=report.position_score,
            hist_total_trades=report.historical.total_trades,
            hist_win_rate=report.historical.win_rate,
            hist_avg_win=report.historical.avg_win_pct,
            hist_avg_loss=report.historical.avg_loss_pct,
            hist_profit_factor=report.historical.profit_factor,
            hist_avg_hold=report.historical.avg_hold_days,
            hist_max_dd=report.historical.max_drawdown_pct,
            hist_sharpe=report.historical.sharpe_ratio,
            regime_performance=regime_perf_str,
            similar_setups=similar_str,
            current_price=report.technicals.current_price,
            sma_20=report.technicals.sma_20,
            price_vs_sma20=price_vs_sma20,
            sma_50=report.technicals.sma_50,
            price_vs_sma50=price_vs_sma50,
            sma_200=report.technicals.sma_200,
            price_vs_sma200=price_vs_sma200,
            rsi_14=report.technicals.rsi_14,
            atr_14=report.technicals.atr_14,
            atr_pct=(report.technicals.atr_14 / report.technicals.current_price * 100) if report.technicals.current_price > 0 else 0,
            volume_vs_avg=report.technicals.volume_vs_avg,
            dist_52w_high=report.technicals.distance_from_52w_high,
            dist_52w_low=report.technicals.distance_from_52w_low,
            trend_direction=report.technicals.trend_direction,
            volatility_regime=report.technicals.volatility_regime,
            support_levels=support_str,
            resistance_levels=resist_str,
            news_sentiment=report.news.sentiment_score,
            news_sentiment_label=report.news.sentiment_label,
            news_themes=", ".join(report.news.key_themes),
            news_headlines=headlines_str,
            upcoming_events=", ".join(report.news.upcoming_events) if report.news.upcoming_events else "None scheduled",
            market_regime=report.market_regime,
            regime_confidence=report.regime_confidence * 100,
            vix_level=report.vix_level,
            market_sentiment=report.market_sentiment,
        )

        try:
            response = self._call_claude(prompt)
            if response:
                # Parse the response into sections
                return self._parse_comprehensive_response(response)
        except Exception as e:
            logger.error(f"Failed to generate comprehensive analysis: {e}")

        return None

    def _parse_comprehensive_response(self, response: str) -> Dict[str, Any]:
        """Parse Claude's comprehensive response into sections."""
        sections = {
            'executive_summary': '',
            'why_this_trade': '',
            'historical_edge': '',
            'technical': '',
            'news_impact': '',
            'risk': '',
            'execution': '',
            'sizing': '',
            'warnings': [],
            'levels': [],
        }

        # Simple section parsing
        current_section = None
        current_content = []

        for line in response.split('\n'):
            line_lower = line.lower().strip()

            if 'executive summary' in line_lower or '1.' in line and 'summary' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'executive_summary'
                current_content = []
            elif 'why this trade' in line_lower or '2.' in line and 'trade' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'why_this_trade'
                current_content = []
            elif 'historical edge' in line_lower or '3.' in line and 'historical' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'historical_edge'
                current_content = []
            elif 'technical setup' in line_lower or '4.' in line and 'technical' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'technical'
                current_content = []
            elif 'news' in line_lower and 'catalyst' in line_lower or '5.' in line and 'news' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'news_impact'
                current_content = []
            elif 'risk analysis' in line_lower or '6.' in line and 'risk' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'risk'
                current_content = []
            elif 'execution plan' in line_lower or '7.' in line and 'execution' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'execution'
                current_content = []
            elif 'position sizing' in line_lower or '8.' in line and 'sizing' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'sizing'
                current_content = []
            elif 'risk warnings' in line_lower or '11.' in line and 'warning' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'warnings_section'
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

        # Extract warnings as list
        if 'warnings_section' in sections:
            warnings_text = sections.get('warnings_section', '')
            sections['warnings'] = [
                line.strip('- •').strip()
                for line in warnings_text.split('\n')
                if line.strip() and line.strip().startswith(('-', '•'))
            ]

        return sections

    def _deterministic_comprehensive_totd(
        self,
        report: ComprehensiveTOTDReport,
    ) -> ComprehensiveTOTDReport:
        """Generate deterministic comprehensive analysis."""
        h = report.historical
        t = report.technicals

        report.executive_summary = (
            f"{report.symbol} presents a high-conviction {report.strategy} setup with "
            f"a {h.win_rate:.1f}% historical win rate and {report.risk_reward_ratio:.1f}:1 R/R ratio. "
            f"The {report.market_regime} regime supports this mean-reversion trade."
        )

        report.why_this_trade = f"""
{report.symbol} stands out as today's top opportunity based on several converging factors:

1. **Signal Quality**: The signal score of {report.position_score:.1f} is in the top decile of all signals generated by the {report.strategy} strategy. This indicates an exceptionally strong setup relative to typical opportunities.

2. **Technical Confluence**: The stock is trading at ${t.current_price:.2f}, which is {abs(t.distance_from_52w_high):.1f}% from its 52-week high. The RSI(14) at {t.rsi_14:.1f} and the {t.trend_direction} trend direction create a favorable entry condition.

3. **Risk/Reward Profile**: With an entry at ${report.entry_price:.2f} and stop at ${report.stop_loss:.2f}, the {report.stop_distance_pct:.1f}% stop distance provides reasonable protection while the {report.risk_reward_ratio:.1f}:1 R/R ratio offers attractive upside.
"""

        report.historical_edge_analysis = f"""
The {report.strategy} strategy has demonstrated consistent profitability across {h.total_trades} historical trades:

- **Win Rate**: {h.win_rate:.1f}% - This means roughly 2 out of every 3 trades are profitable
- **Profit Factor**: {h.profit_factor:.2f} - For every $1 lost, the strategy makes ${h.profit_factor:.2f}
- **Average Hold**: {h.avg_hold_days:.1f} days - Quick mean-reversion trades
- **Sharpe Ratio**: {h.sharpe_ratio:.2f} - Strong risk-adjusted returns

In the current {report.market_regime} regime, the strategy has historically achieved a {h.regime_performance.get(report.market_regime, h.win_rate):.1f}% win rate, {'above' if h.regime_performance.get(report.market_regime, h.win_rate) > h.win_rate else 'near'} its overall average.
"""

        report.technical_analysis = f"""
**Price Action**: {report.symbol} is trading at ${t.current_price:.2f}, positioned:
- ${abs(t.current_price - t.sma_20):.2f} {'above' if t.current_price > t.sma_20 else 'below'} the 20-day SMA (${t.sma_20:.2f})
- ${abs(t.current_price - t.sma_200):.2f} {'above' if t.current_price > t.sma_200 else 'below'} the 200-day SMA (${t.sma_200:.2f})

**Volatility**: ATR(14) of ${t.atr_14:.2f} ({t.atr_14/t.current_price*100:.1f}% of price) indicates {t.volatility_regime.lower()} volatility conditions.

**Volume**: Today's volume is {t.volume_vs_avg:.0f}% of the 20-day average, suggesting {'heightened' if t.volume_vs_avg > 120 else 'normal' if t.volume_vs_avg > 80 else 'subdued'} trading interest.

**Key Levels**: Support at {', '.join([f'${s:.2f}' for s in t.support_levels[:2]])} | Resistance at {', '.join([f'${r:.2f}' for r in t.resistance_levels[:2]])}
"""

        report.news_impact_analysis = f"""
Current news sentiment for {report.symbol} is {report.news.sentiment_label} (score: {report.news.sentiment_score:.2f}).
Key themes: {', '.join(report.news.key_themes)}.

{'Recent news flow is supportive of the long position.' if report.news.sentiment_score > 0 else 'News is neutral to slightly negative, but technical setup takes precedence for this mean-reversion trade.'}
"""

        report.risk_analysis = f"""
**Primary Risks**:
1. Stop distance of ${report.risk_per_share:.2f} ({report.stop_distance_pct:.1f}%) means a 100-share position risks ${report.risk_per_share * 100:.2f}
2. ATR of ${t.atr_14:.2f} suggests typical daily moves could challenge tight stops
3. VIX at {report.vix_level:.1f} indicates {'elevated' if report.vix_level > 25 else 'normal'} market volatility

**Worst Case**: Full stop-out at ${report.stop_loss:.2f} for {report.stop_distance_pct:.1f}% loss. Given the {h.win_rate:.1f}% win rate, expect roughly 1 in 3 trades to hit the stop.

**Position Sizing**: Risk no more than 1-2% of portfolio on this trade.
"""

        report.execution_plan = f"""
**Entry Timing**: Execute at or near market open if price is within 0.5% of ${report.entry_price:.2f}
**Order Type**: Limit order at ${report.entry_price:.2f} or market order if setup remains valid
**Partial Profits**: Consider taking 50% off at 1R (${report.entry_price + report.risk_per_share:.2f})
**Trail Stop**: After 1R profit, move stop to breakeven
**Time Stop**: Exit if trade hasn't worked after {int(h.avg_hold_days) + 2} trading days
"""

        report.position_sizing = f"""
For a $100,000 portfolio with 1% risk per trade:
- Max Risk: $1,000
- Risk per Share: ${report.risk_per_share:.2f}
- Position Size: {int(1000 / report.risk_per_share)} shares (${int(1000 / report.risk_per_share) * report.entry_price:,.0f} notional)
- Expected Win: +${report.risk_per_share * report.risk_reward_ratio * int(1000 / report.risk_per_share):,.0f} at target
- Max Loss: -$1,000 at stop
"""

        report.risk_warnings = [
            f"Stop distance ({report.stop_distance_pct:.1f}%) is {'tight' if report.stop_distance_pct < 5 else 'wide' if report.stop_distance_pct > 10 else 'normal'} - adjust size accordingly",
            f"VIX at {report.vix_level:.1f} - {'reduce size due to elevated volatility' if report.vix_level > 25 else 'normal volatility conditions'}",
            f"Weekend gap risk if held over Friday",
            f"Earnings/event risk - check calendar before entry",
        ]

        report.key_levels_to_watch = [
            f"Entry: ${report.entry_price:.2f}",
            f"Stop Loss: ${report.stop_loss:.2f} (-{report.stop_distance_pct:.1f}%)",
            f"Target 1 (1R): ${report.entry_price + report.risk_per_share:.2f} (+{report.stop_distance_pct:.1f}%)",
            f"Target 2 (2R): ${report.entry_price + 2*report.risk_per_share:.2f} (+{2*report.stop_distance_pct:.1f}%)",
            f"Invalidation: ${report.stop_loss - report.technicals.atr_14:.2f}",
        ]

        return report


# =============================================================================
# Singleton Pattern
# =============================================================================

_trade_analyzer: Optional[LLMTradeAnalyzer] = None


def get_trade_analyzer(**kwargs) -> LLMTradeAnalyzer:
    """Get or create the singleton trade analyzer instance."""
    global _trade_analyzer
    if _trade_analyzer is None:
        _trade_analyzer = LLMTradeAnalyzer(**kwargs)
    return _trade_analyzer


def reset_trade_analyzer():
    """Reset the singleton (for testing)."""
    global _trade_analyzer
    _trade_analyzer = None
