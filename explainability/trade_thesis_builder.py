"""
Trade Thesis Builder - Comprehensive Evidence Aggregation
==========================================================

Builds a complete evidence package for each trade, including:

1. Historical pattern analysis (consecutive days, reversal rates)
2. News and sentiment analysis
3. Options expected move
4. Support/resistance levels
5. Volume profile
6. Sector relative strength
7. AI confidence scoring with breakdown
8. Bull/bear cases
9. Price level justifications

This is the core module that powers the Pre-Game Blueprint's trade theses.

Usage:
    from explainability.trade_thesis_builder import TradeThesisBuilder

    builder = TradeThesisBuilder()
    thesis = builder.build_thesis(signal, context)
    print(thesis.executive_summary)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TradeThesis:
    """Complete evidence package for a trade."""
    # Core Trade Info
    symbol: str
    strategy: str
    side: str  # "long" | "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float

    # Historical Evidence
    consecutive_pattern: Dict = field(default_factory=dict)
    support_resistance_levels: List[Dict] = field(default_factory=list)
    volume_analysis: Dict = field(default_factory=dict)
    sector_relative_strength: Dict = field(default_factory=dict)

    # NEW: Entry Timing
    entry_timing: Dict = field(default_factory=dict)
    is_a_plus_pattern: bool = False  # True if 20+ samples, 90%+ win rate
    pattern_grade: str = "D"  # "A+" | "A" | "B" | "C" | "D"

    # News & Sentiment
    news_headlines: List[Dict] = field(default_factory=list)
    aggregated_sentiment: Dict = field(default_factory=dict)
    sentiment_interpretation: str = ""

    # NEW: Alternative Data Sources
    congressional_trades: Dict = field(default_factory=dict)
    insider_activity: Dict = field(default_factory=dict)
    options_flow: Dict = field(default_factory=dict)

    # Options Data
    expected_move: Dict = field(default_factory=dict)
    remaining_room: Dict = field(default_factory=dict)

    # AI Analysis
    ai_confidence: float = 0.0  # 0-100
    ai_confidence_breakdown: Dict = field(default_factory=dict)
    bull_case: str = ""
    bear_case: str = ""
    what_could_go_wrong: List[str] = field(default_factory=list)

    # Price Level Justification
    entry_justification: str = ""
    stop_justification: str = ""
    target_justification: str = ""

    # Decision Summary
    trade_grade: str = "C"  # "A+" | "A" | "B" | "C"
    executive_summary: str = ""
    recommendation: str = "SKIP"  # "EXECUTE" | "WATCHLIST" | "SKIP"

    # Metadata
    generated_at: str = ""
    generation_method: str = "deterministic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TradeThesisBuilder:
    """
    Builds comprehensive trade theses with evidence aggregation.

    Integrates:
    - HistoricalPatternAnalyzer
    - ExpectedMoveCalculator
    - NewsProcessor
    - LLM for narrative generation (optional)
    """

    def __init__(self, dotenv_path: str = "./.env"):
        """Initialize the builder with all components."""
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)

        self.api_key = os.getenv("ANTHROPIC_API_KEY")

        # Lazy load components
        self._pattern_analyzer = None
        self._em_calculator = None
        self._news_processor = None
        self._congressional_client = None
        self._insider_client = None
        self._options_flow_client = None

    @property
    def pattern_analyzer(self):
        if self._pattern_analyzer is None:
            try:
                from analysis.historical_patterns import HistoricalPatternAnalyzer
                self._pattern_analyzer = HistoricalPatternAnalyzer(lookback_years=5)
            except Exception as e:
                logger.warning(f"Could not load pattern analyzer: {e}")
        return self._pattern_analyzer

    @property
    def em_calculator(self):
        if self._em_calculator is None:
            try:
                from analysis.options_expected_move import ExpectedMoveCalculator
                self._em_calculator = ExpectedMoveCalculator()
            except Exception as e:
                logger.warning(f"Could not load expected move calculator: {e}")
        return self._em_calculator

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
    def congressional_client(self):
        """Lazy load congressional trades client."""
        if self._congressional_client is None:
            try:
                from altdata.congressional_trades import get_congressional_client
                self._congressional_client = get_congressional_client()
            except Exception as e:
                logger.warning(f"Could not load congressional client: {e}")
        return self._congressional_client

    @property
    def insider_client(self):
        """Lazy load insider activity client."""
        if self._insider_client is None:
            try:
                from altdata.insider_activity import get_insider_client
                self._insider_client = get_insider_client()
            except Exception as e:
                logger.warning(f"Could not load insider client: {e}")
        return self._insider_client

    @property
    def options_flow_client(self):
        """Lazy load options flow client."""
        if self._options_flow_client is None:
            try:
                from altdata.options_flow import get_options_flow_client
                self._options_flow_client = get_options_flow_client()
            except Exception as e:
                logger.warning(f"Could not load options flow client: {e}")
        return self._options_flow_client

    def build_thesis(
        self,
        signal: Dict,
        context: Optional[Dict] = None,
    ) -> TradeThesis:
        """
        Build a complete trade thesis for a signal.

        Args:
            signal: Signal dict with symbol, strategy, entry, stop, target
            context: Optional market context (regime, VIX, etc.)

        Returns:
            TradeThesis with all evidence fields populated
        """
        symbol = signal.get('symbol', '')
        strategy = signal.get('strategy', 'unknown')
        side = signal.get('side', 'long')
        entry = float(signal.get('entry_price', 0))
        stop = float(signal.get('stop_loss', 0))
        target = float(signal.get('take_profit', 0))

        # Calculate R:R
        if entry > 0 and stop > 0 and entry != stop:
            risk = abs(entry - stop)
            reward = abs(target - entry) if target > 0 else risk * 2
            rr_ratio = reward / risk
        else:
            rr_ratio = 0.0

        thesis = TradeThesis(
            symbol=symbol,
            strategy=strategy,
            side=side,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            risk_reward_ratio=round(rr_ratio, 2),
            generated_at=datetime.now().isoformat(),
        )

        # Gather evidence
        self._gather_historical_evidence(thesis)
        self._gather_news_evidence(thesis)
        self._gather_options_evidence(thesis)
        self._gather_congressional_evidence(thesis)
        self._gather_insider_evidence(thesis)
        self._gather_options_flow_evidence(thesis)

        # Calculate AI confidence (boosted for A+ patterns)
        self._calculate_ai_confidence(thesis)

        # Generate narratives
        self._generate_price_justifications(thesis, signal)
        self._generate_bull_bear_cases(thesis, context)
        self._generate_executive_summary(thesis)

        # Determine grade and recommendation
        self._determine_grade_and_recommendation(thesis)

        return thesis

    def _gather_historical_evidence(self, thesis: TradeThesis) -> None:
        """Gather historical pattern analysis."""
        if not self.pattern_analyzer:
            return

        try:
            # Consecutive day pattern
            pattern = self.pattern_analyzer.analyze_consecutive_days(symbol=thesis.symbol)
            thesis.consecutive_pattern = pattern.to_dict()

            # Entry timing recommendation
            entry_timing = self.pattern_analyzer.get_entry_timing_recommendation(
                thesis.symbol, pattern=pattern
            )
            thesis.entry_timing = entry_timing.to_dict()

            # Determine pattern grade (A+, A, B, C, D)
            from analysis.historical_patterns import get_pattern_grade, qualifies_for_auto_pass
            thesis.pattern_grade = get_pattern_grade(pattern)
            thesis.is_a_plus_pattern = qualifies_for_auto_pass(pattern)

            if thesis.is_a_plus_pattern:
                logger.info(
                    f"A+ PATTERN: {thesis.symbol} - {pattern.sample_size} samples, "
                    f"{pattern.historical_reversal_rate:.0%} win rate"
                )

            # Support/resistance
            levels = self.pattern_analyzer.analyze_support_resistance(symbol=thesis.symbol)
            thesis.support_resistance_levels = [l.to_dict() for l in levels]

            # Volume profile
            vol = self.pattern_analyzer.analyze_volume_profile(symbol=thesis.symbol)
            thesis.volume_analysis = vol.to_dict()

            # Sector relative strength
            sr = self.pattern_analyzer.get_sector_relative_strength(thesis.symbol)
            thesis.sector_relative_strength = sr.to_dict()

        except Exception as e:
            logger.warning(f"Error gathering historical evidence for {thesis.symbol}: {e}")

    def _gather_news_evidence(self, thesis: TradeThesis) -> None:
        """Gather news and sentiment data."""
        if not self.news_processor:
            return

        try:
            # Get recent news
            articles = self.news_processor.fetch_news(
                symbols=[thesis.symbol],
                limit=10,
            )

            thesis.news_headlines = [
                {
                    'headline': a.headline,
                    'sentiment': a.sentiment_score.get('compound', 0),
                    'source': a.source or 'Unknown',
                    'time': a.created_at.isoformat() if hasattr(a.created_at, 'isoformat') else str(a.created_at),
                }
                for a in articles[:5]
            ]

            # Get aggregated sentiment
            sentiment = self.news_processor.get_aggregated_sentiment(
                symbols=[thesis.symbol],
                lookback_minutes=240,  # 4 hours
            )
            thesis.aggregated_sentiment = sentiment

            # Interpret sentiment
            compound = sentiment.get('compound', 0)
            if compound > 0.2:
                thesis.sentiment_interpretation = "Strongly positive sentiment supporting long thesis"
            elif compound > 0.05:
                thesis.sentiment_interpretation = "Mildly positive sentiment, neutral to slightly supportive"
            elif compound < -0.2:
                thesis.sentiment_interpretation = "Strongly negative sentiment, caution warranted"
            elif compound < -0.05:
                thesis.sentiment_interpretation = "Mildly negative sentiment, be cautious"
            else:
                thesis.sentiment_interpretation = "Neutral sentiment, no strong directional bias"

        except Exception as e:
            logger.warning(f"Error gathering news evidence for {thesis.symbol}: {e}")

    def _gather_options_evidence(self, thesis: TradeThesis) -> None:
        """Gather options expected move data."""
        if not self.em_calculator:
            return

        try:
            em = self.em_calculator.calculate_weekly_expected_move(
                thesis.symbol,
                thesis.entry_price,
            )
            thesis.expected_move = em.to_dict()
            thesis.remaining_room = {
                'up_pct': em.remaining_room_up_pct,
                'down_pct': em.remaining_room_down_pct,
                'direction': em.remaining_room_direction,
                'interpretation': em.interpretation,
            }

        except Exception as e:
            logger.warning(f"Error gathering options evidence for {thesis.symbol}: {e}")

    def _gather_congressional_evidence(self, thesis: TradeThesis) -> None:
        """Gather congressional trading data."""
        if not self.congressional_client:
            return

        try:
            summary = self.congressional_client.get_trade_summary(thesis.symbol, days_back=90)
            thesis.congressional_trades = {
                'total_trades': summary.total_trades,
                'buy_count': summary.buy_count,
                'sell_count': summary.sell_count,
                'net_activity': summary.net_activity,
                'total_buy_value': summary.total_buy_value,
                'total_sell_value': summary.total_sell_value,
                'unique_representatives': summary.unique_representatives,
                'notable_officials': summary.notable_officials,
                'party_breakdown': summary.party_breakdown,
                'recent_trades': [t.to_dict() for t in summary.recent_trades[:5]],
                'data_source': summary.data_source,
            }

            if summary.net_activity == 'NET_BUY':
                logger.info(f"Congressional NET_BUY for {thesis.symbol}: {summary.buy_count} buys, ${summary.total_buy_value:,.0f}")

        except Exception as e:
            logger.warning(f"Error gathering congressional evidence for {thesis.symbol}: {e}")

    def _gather_insider_evidence(self, thesis: TradeThesis) -> None:
        """Gather insider trading data."""
        if not self.insider_client:
            return

        try:
            summary = self.insider_client.get_activity_summary(thesis.symbol, days_back=30)
            thesis.insider_activity = {
                'total_filings': summary.total_filings,
                'buy_count': summary.buy_count,
                'sell_count': summary.sell_count,
                'net_activity': summary.net_activity,
                'total_buy_value': summary.total_buy_value,
                'total_sell_value': summary.total_sell_value,
                'total_buy_shares': summary.total_buy_shares,
                'total_sell_shares': summary.total_sell_shares,
                'unique_insiders': summary.unique_insiders,
                'notable_insiders': summary.notable_insiders,
                'recent_trades': [t.to_dict() for t in summary.recent_trades[:5]],
                'data_source': summary.data_source,
            }

            if summary.net_activity == 'NET_BUY':
                logger.info(f"Insider NET_BUY for {thesis.symbol}: {summary.buy_count} buys, ${summary.total_buy_value:,.0f}")

        except Exception as e:
            logger.warning(f"Error gathering insider evidence for {thesis.symbol}: {e}")

    def _gather_options_flow_evidence(self, thesis: TradeThesis) -> None:
        """Gather options flow and unusual activity data."""
        if not self.options_flow_client:
            return

        try:
            flow = self.options_flow_client.get_flow_summary(thesis.symbol, days_back=7)
            thesis.options_flow = {
                'total_call_volume': flow.total_call_volume,
                'total_put_volume': flow.total_put_volume,
                'put_call_ratio': flow.put_call_ratio,
                'pcr_signal': flow.pcr_signal,
                'current_iv': flow.current_iv,
                'iv_percentile': flow.iv_percentile,
                'iv_rank': flow.iv_rank,
                'iv_signal': flow.iv_signal,
                'unusual_activity_count': flow.unusual_activity_count,
                'unusual_activities': [a.to_dict() for a in flow.unusual_activities[:3]],
                'bullish_flow_count': flow.bullish_flow_count,
                'bearish_flow_count': flow.bearish_flow_count,
                'net_flow_bias': flow.net_flow_bias,
                'data_source': flow.data_source,
            }

            if flow.unusual_activity_count > 0:
                logger.info(f"Options flow for {thesis.symbol}: {flow.unusual_activity_count} unusual, bias={flow.net_flow_bias}")

        except Exception as e:
            logger.warning(f"Error gathering options flow evidence for {thesis.symbol}: {e}")

    def _calculate_ai_confidence(self, thesis: TradeThesis) -> None:
        """Calculate AI confidence score with breakdown.

        For A+ patterns (20+ samples, 90%+ win rate), historical weight
        is boosted to 40% (from 20%) and score is maxed at 100%.
        """
        scores = {}

        # Technical score (R:R and setup quality)
        if thesis.risk_reward_ratio >= 2.0:
            scores['technical'] = 90
        elif thesis.risk_reward_ratio >= 1.5:
            scores['technical'] = 75
        elif thesis.risk_reward_ratio >= 1.0:
            scores['technical'] = 60
        else:
            scores['technical'] = 40

        # Historical pattern score - BOOSTED for A+ patterns
        pattern = thesis.consecutive_pattern
        if thesis.is_a_plus_pattern:
            # A+ pattern: 20+ samples, 90%+ win rate = MAX score
            scores['historical'] = 100
            logger.info(f"A+ PATTERN BOOST: {thesis.symbol} historical score = 100")
        elif thesis.pattern_grade == 'A':
            scores['historical'] = 95
        elif thesis.pattern_grade == 'B':
            scores['historical'] = 85
        elif pattern.get('confidence') == 'HIGH':
            scores['historical'] = 90
        elif pattern.get('confidence') == 'MEDIUM-HIGH':
            scores['historical'] = 80
        elif pattern.get('confidence') == 'MEDIUM':
            scores['historical'] = 75
        elif pattern.get('historical_reversal_rate', 0) > 0.6:
            scores['historical'] = 70
        else:
            scores['historical'] = 50

        # Sentiment score
        compound = thesis.aggregated_sentiment.get('compound', 0)
        if thesis.side == 'long':
            if compound > 0.2:
                scores['sentiment'] = 85
            elif compound > 0:
                scores['sentiment'] = 70
            elif compound > -0.1:
                scores['sentiment'] = 60
            else:
                scores['sentiment'] = 40
        else:  # short
            if compound < -0.2:
                scores['sentiment'] = 85
            elif compound < 0:
                scores['sentiment'] = 70
            elif compound < 0.1:
                scores['sentiment'] = 60
            else:
                scores['sentiment'] = 40

        # Options/Volatility score
        em = thesis.expected_move
        if em.get('remaining_room_direction') == 'BOTH':
            scores['options'] = 80
        elif em.get('remaining_room_direction') == thesis.side.upper():
            scores['options'] = 75
        elif em.get('remaining_room_direction') == 'EXHAUSTED':
            scores['options'] = 40
        else:
            scores['options'] = 60

        # Sector strength score
        sr = thesis.sector_relative_strength
        if sr.get('outperforming', False):
            scores['sector'] = 80
        else:
            rs = sr.get('relative_strength', 0)
            if rs > -0.02:
                scores['sector'] = 65
            else:
                scores['sector'] = 50

        # Volume score
        vol = thesis.volume_analysis
        bp = vol.get('buying_pressure', 0.5)
        if thesis.side == 'long' and bp > 0.55:
            scores['volume'] = 80
        elif thesis.side == 'short' and bp < 0.45:
            scores['volume'] = 80
        else:
            scores['volume'] = 60

        # Alt-data scores (NEW)
        # Congressional trades
        cong = thesis.congressional_trades
        if cong.get('net_activity') == 'NET_BUY' and thesis.side == 'long':
            scores['congressional'] = 85
        elif cong.get('net_activity') == 'NET_SELL' and thesis.side == 'short':
            scores['congressional'] = 85
        elif cong.get('total_trades', 0) > 0:
            scores['congressional'] = 60
        else:
            scores['congressional'] = 50

        # Insider activity
        insider = thesis.insider_activity
        if insider.get('net_activity') == 'NET_BUY' and thesis.side == 'long':
            scores['insider'] = 85
        elif insider.get('net_activity') == 'NET_SELL' and thesis.side == 'short':
            scores['insider'] = 85
        elif insider.get('total_filings', 0) > 0:
            scores['insider'] = 60
        else:
            scores['insider'] = 50

        # Options flow
        flow = thesis.options_flow
        if flow.get('net_flow_bias') == 'BULLISH' and thesis.side == 'long':
            scores['options_flow'] = 85
        elif flow.get('net_flow_bias') == 'BEARISH' and thesis.side == 'short':
            scores['options_flow'] = 85
        elif flow.get('unusual_activity_count', 0) > 0:
            scores['options_flow'] = 70
        else:
            scores['options_flow'] = 50

        # Calculate weighted average
        # For A+ patterns, boost historical weight to 40% (from 20%)
        if thesis.is_a_plus_pattern:
            weights = {
                'technical': 0.15,
                'historical': 0.40,  # BOOSTED for A+ patterns
                'sentiment': 0.10,
                'options': 0.10,
                'sector': 0.05,
                'volume': 0.05,
                'congressional': 0.05,
                'insider': 0.05,
                'options_flow': 0.05,
            }
        else:
            weights = {
                'technical': 0.20,
                'historical': 0.20,
                'sentiment': 0.12,
                'options': 0.12,
                'sector': 0.08,
                'volume': 0.08,
                'congressional': 0.07,
                'insider': 0.07,
                'options_flow': 0.06,
            }

        total_score = sum(
            scores.get(k, 50) * v for k, v in weights.items()
        )

        thesis.ai_confidence = round(total_score, 1)
        thesis.ai_confidence_breakdown = scores

    def _generate_price_justifications(self, thesis: TradeThesis, signal: Dict) -> None:
        """Generate justifications for entry, stop, and target prices."""
        # Entry justification
        pattern = thesis.consecutive_pattern
        if pattern.get('pattern_type') == 'consecutive_down' and thesis.side == 'long':
            thesis.entry_justification = (
                f"Entry at ${thesis.entry_price:.2f} after {pattern.get('current_streak', 0)} consecutive down days. "
                f"Historical data shows {pattern.get('historical_reversal_rate', 0):.0%} reversal probability at this streak length."
            )
        elif pattern.get('pattern_type') == 'consecutive_up' and thesis.side == 'short':
            thesis.entry_justification = (
                f"Entry at ${thesis.entry_price:.2f} after {pattern.get('current_streak', 0)} consecutive up days. "
                f"Historical data shows reversal probability at extended streaks."
            )
        else:
            thesis.entry_justification = (
                f"Entry at ${thesis.entry_price:.2f} based on {thesis.strategy} signal. "
                "Signal generated from strategy rules."
            )

        # Stop justification
        risk_pct = abs(thesis.entry_price - thesis.stop_loss) / thesis.entry_price * 100
        support_levels = [l for l in thesis.support_resistance_levels if l.get('level_type') == 'support']

        if support_levels and thesis.side == 'long':
            nearest_support = min(support_levels, key=lambda x: abs(x['price'] - thesis.stop_loss))
            thesis.stop_justification = (
                f"Stop at ${thesis.stop_loss:.2f} ({risk_pct:.1f}% risk) placed below support at ${nearest_support['price']:.2f}. "
                f"{nearest_support.get('justification', '')}."
            )
        else:
            thesis.stop_justification = (
                f"Stop at ${thesis.stop_loss:.2f} ({risk_pct:.1f}% risk) based on ATR-derived volatility buffer. "
                "Provides room for normal price fluctuation while protecting capital."
            )

        # Target justification
        reward_pct = abs(thesis.take_profit - thesis.entry_price) / thesis.entry_price * 100
        resistance_levels = [l for l in thesis.support_resistance_levels if l.get('level_type') == 'resistance']

        if resistance_levels and thesis.side == 'long':
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x['price'] - thesis.take_profit))
            thesis.target_justification = (
                f"Target at ${thesis.take_profit:.2f} ({reward_pct:.1f}% reward) near resistance at ${nearest_resistance['price']:.2f}. "
                f"{nearest_resistance.get('justification', '')}. R:R = {thesis.risk_reward_ratio:.2f}:1."
            )
        else:
            thesis.target_justification = (
                f"Target at ${thesis.take_profit:.2f} ({reward_pct:.1f}% reward) based on historical reversal magnitude. "
                f"Average reversal: {pattern.get('avg_reversal_magnitude', 0):+.1%}. R:R = {thesis.risk_reward_ratio:.2f}:1."
            )

    def _generate_bull_bear_cases(self, thesis: TradeThesis, context: Optional[Dict]) -> None:
        """Generate bull and bear case narratives."""
        pattern = thesis.consecutive_pattern
        em = thesis.expected_move
        sentiment = thesis.aggregated_sentiment

        # Bull case (for long trades)
        if thesis.side == 'long':
            bull_parts = []

            if pattern.get('pattern_type') == 'consecutive_down':
                bull_parts.append(
                    f"{thesis.symbol} has been down {pattern.get('current_streak', 0)} consecutive days, "
                    f"historically reversing {pattern.get('historical_reversal_rate', 0):.0%} of the time."
                )

            if em.get('remaining_room_direction') == 'UP':
                bull_parts.append(
                    f"Price has already moved significantly, leaving {em.get('remaining_room_up_pct', 0):.1%} room to the upside within expected range."
                )

            if sentiment.get('compound', 0) > 0:
                bull_parts.append(
                    f"News sentiment is positive ({sentiment.get('compound', 0):+.2f}), supporting the long thesis."
                )

            bull_parts.append(
                f"If entry triggers and holds above ${thesis.stop_loss:.2f}, target ${thesis.take_profit:.2f} "
                f"within {pattern.get('avg_reversal_magnitude', 0):+.1%} historical average."
            )

            thesis.bull_case = " ".join(bull_parts)

            # Bear case
            bear_parts = [
                f"The trade fails if {thesis.symbol} breaks below ${thesis.stop_loss:.2f}, "
                f"indicating the pullback has become a trend change."
            ]

            if sentiment.get('compound', 0) < 0:
                bear_parts.append(
                    f"Negative news sentiment ({sentiment.get('compound', 0):+.2f}) could accelerate selling."
                )

            sr = thesis.sector_relative_strength
            if not sr.get('outperforming', False):
                bear_parts.append(
                    f"{thesis.symbol} is underperforming its sector ({sr.get('sector_etf', 'SPY')}), "
                    "suggesting relative weakness."
                )

            thesis.bear_case = " ".join(bear_parts)

        else:  # short
            thesis.bull_case = f"Bearish thesis requires {thesis.symbol} to continue lower from entry."
            thesis.bear_case = f"Trade fails if {thesis.symbol} breaks above ${thesis.stop_loss:.2f}."

        # What could go wrong
        thesis.what_could_go_wrong = [
            f"Broader market sell-off could extend the decline beyond ${thesis.stop_loss:.2f}",
            f"Unexpected news (earnings, analyst action) could gap the stock significantly",
            f"Sector rotation could override individual stock patterns",
            f"VIX spike could increase volatility and trigger stop prematurely",
        ]

    def _generate_executive_summary(self, thesis: TradeThesis) -> None:
        """Generate executive summary for the trade."""
        pattern = thesis.consecutive_pattern
        em = thesis.expected_move

        parts = []

        # Lead with the setup
        if pattern.get('pattern_type') == 'consecutive_down' and thesis.side == 'long':
            parts.append(
                f"{thesis.symbol} presents a mean-reversion opportunity after {pattern.get('current_streak', 0)} consecutive down days."
            )
        elif pattern.get('pattern_type') == 'consecutive_up' and thesis.side == 'short':
            parts.append(
                f"{thesis.symbol} shows potential reversal after {pattern.get('current_streak', 0)} consecutive up days."
            )
        else:
            parts.append(
                f"{thesis.symbol} triggered a {thesis.strategy} signal for a {thesis.side} trade."
            )

        # Add historical context
        if pattern.get('sample_size', 0) > 10:
            parts.append(
                f"Historical analysis across {pattern.get('sample_size', 0)} instances shows "
                f"{pattern.get('historical_reversal_rate', 0):.0%} reversal rate with "
                f"{pattern.get('avg_reversal_magnitude', 0):+.1%} average move."
            )

        # Add expected move context
        if em.get('move_from_week_open_pct', 0):
            parts.append(
                f"Already moved {em.get('move_from_week_open_pct', 0):+.1%} from week open, "
                f"suggesting {em.get('remaining_room_direction', 'room')} for mean reversion."
            )

        # Add trade parameters
        parts.append(
            f"Entry ${thesis.entry_price:.2f}, Stop ${thesis.stop_loss:.2f}, Target ${thesis.take_profit:.2f} "
            f"({thesis.risk_reward_ratio:.2f}:1 R:R)."
        )

        # Add confidence
        parts.append(
            f"AI Confidence: {thesis.ai_confidence:.0f}% ({thesis.trade_grade} grade)."
        )

        thesis.executive_summary = " ".join(parts)

    def _determine_grade_and_recommendation(self, thesis: TradeThesis) -> None:
        """Determine trade grade and recommendation.

        A+ patterns (20+ samples, 90%+ win rate) automatically get A+ grade
        and EXECUTE recommendation, overriding confidence score.
        """
        confidence = thesis.ai_confidence

        # A+ PATTERN OVERRIDE: If pattern qualifies, auto-pass to A+ grade
        if thesis.is_a_plus_pattern:
            thesis.trade_grade = "A+"
            thesis.recommendation = "EXECUTE"
            logger.info(
                f"A+ PATTERN AUTO-PASS: {thesis.symbol} -> Grade=A+, Recommendation=EXECUTE "
                f"(pattern: {thesis.pattern_grade}, confidence: {confidence:.1f}%)"
            )
        elif confidence >= 80:
            thesis.trade_grade = "A+"
            thesis.recommendation = "EXECUTE"
        elif confidence >= 70:
            thesis.trade_grade = "A"
            thesis.recommendation = "EXECUTE"
        elif confidence >= 60:
            thesis.trade_grade = "B"
            thesis.recommendation = "WATCHLIST"
        else:
            thesis.trade_grade = "C"
            thesis.recommendation = "SKIP"

        # Override based on R:R (but NOT for A+ patterns)
        if thesis.risk_reward_ratio < 1.0 and not thesis.is_a_plus_pattern:
            thesis.recommendation = "SKIP"
            thesis.trade_grade = min(thesis.trade_grade, "C")

        # Override based on sentiment if extreme (but NOT for A+ patterns)
        if thesis.side == 'long' and thesis.aggregated_sentiment.get('compound', 0) < -0.3:
            if not thesis.is_a_plus_pattern:
                thesis.recommendation = "WATCHLIST"
                if thesis.trade_grade > "B":
                    thesis.trade_grade = "B"


def get_trade_thesis_builder(dotenv_path: str = "./.env") -> TradeThesisBuilder:
    """Factory function to get builder instance."""
    return TradeThesisBuilder(dotenv_path)
