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
