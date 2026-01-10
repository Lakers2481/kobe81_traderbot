"""
Tests for LLM Trade Analyzer
============================

Tests for the central LLM reasoning engine that provides human-like
trade explanations using Claude.

Run: pytest tests/cognitive/test_llm_trade_analyzer.py -v
"""

import pytest
import pandas as pd
from datetime import datetime


class TestTradeNarrative:
    """Tests for TradeNarrative dataclass."""

    def test_import(self):
        from cognitive.llm_trade_analyzer import TradeNarrative
        assert TradeNarrative is not None

    def test_create_narrative(self):
        from cognitive.llm_trade_analyzer import TradeNarrative

        narrative = TradeNarrative(
            symbol="AAPL",
            strategy="ibs_rsi",
            narrative="AAPL shows oversold conditions.",
            conviction_reasons=["IBS in bottom decile", "Above SMA(200)"],
            risk_factors=["Earnings in 2 weeks"],
            edge_description="Mean reversion edge",
            confidence_rating="HIGH",
            market_alignment="Bull regime supports longs",
        )

        assert narrative.symbol == "AAPL"
        assert narrative.strategy == "ibs_rsi"
        assert len(narrative.conviction_reasons) == 2
        assert narrative.confidence_rating == "HIGH"

    def test_narrative_to_dict(self):
        from cognitive.llm_trade_analyzer import TradeNarrative

        narrative = TradeNarrative(
            symbol="TSLA",
            strategy="turtle_soup",
            narrative="Liquidity sweep detected.",
        )

        d = narrative.to_dict()
        assert isinstance(d, dict)
        assert d["symbol"] == "TSLA"
        assert d["strategy"] == "turtle_soup"
        assert "narrative" in d


class TestDailyInsightReport:
    """Tests for DailyInsightReport dataclass."""

    def test_import(self):
        from cognitive.llm_trade_analyzer import DailyInsightReport
        assert DailyInsightReport is not None

    def test_create_report(self):
        from cognitive.llm_trade_analyzer import DailyInsightReport, TradeNarrative

        report = DailyInsightReport(
            date="2025-12-28",
            timestamp=datetime.now().isoformat(),
            market_summary="Markets bullish today.",
            top3_narratives=[
                TradeNarrative(symbol="AAPL", strategy="ibs_rsi", narrative="Test"),
            ],
            totd_deep_analysis="Deep analysis here.",
            key_findings=["Finding 1", "Finding 2"],
            sentiment_interpretation="Sentiment is positive.",
            regime_assessment="BULL regime confirmed.",
            risk_warnings=["VIX elevated"],
            opportunities=["Mean reversion setups"],
            llm_model="claude-3-haiku",
            generation_method="deterministic",
        )

        assert report.date == "2025-12-28"
        assert len(report.top3_narratives) == 1
        assert len(report.key_findings) == 2

    def test_report_to_dict(self):
        from cognitive.llm_trade_analyzer import DailyInsightReport

        report = DailyInsightReport(
            date="2025-12-28",
            timestamp=datetime.now().isoformat(),
            market_summary="Test summary",
        )

        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["date"] == "2025-12-28"
        assert "market_summary" in d


class TestLLMTradeAnalyzer:
    """Tests for LLMTradeAnalyzer class."""

    def test_import(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer
        assert LLMTradeAnalyzer is not None

    def test_init_defaults(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer()
        assert analyzer.model == "claude-sonnet-4-20250514"
        assert analyzer.max_tokens == 2000
        assert analyzer.temperature == 0.7
        assert analyzer.fallback_enabled is True

    def test_init_custom_params(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer(
            model="claude-3-sonnet",
            max_tokens=2000,
            temperature=0.5,
            fallback_enabled=False,
        )

        assert analyzer.model == "claude-3-sonnet"
        assert analyzer.max_tokens == 2000
        assert analyzer.temperature == 0.5
        assert analyzer.fallback_enabled is False


class TestDeterministicFallback:
    """Tests for deterministic fallback when Claude API unavailable."""

    def test_deterministic_narrative_ibs_rsi(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer(fallback_enabled=True)

        pick = {
            "symbol": "AAPL",
            "strategy": "ibs_rsi",
            "ibs": 0.08,
            "rsi2": 5.0,
            "entry_price": 175.0,
            "score": 12.0,
        }

        narrative = analyzer._deterministic_narrative(pick)

        assert narrative.symbol == "AAPL"
        assert narrative.strategy == "ibs_rsi"
        assert len(narrative.narrative) > 0
        assert len(narrative.conviction_reasons) > 0
        assert len(narrative.risk_factors) > 0
        assert narrative.edge_description != ""

    def test_deterministic_narrative_turtle_soup(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer()

        pick = {
            "symbol": "TSLA",
            "strategy": "turtle_soup",
            "sweep_strength": 1.5,
            "entry_price": 250.0,
            "score": 150.0,
        }

        narrative = analyzer._deterministic_narrative(pick)

        assert narrative.symbol == "TSLA"
        assert narrative.strategy == "turtle_soup"
        assert "liquidity" in narrative.narrative.lower() or "turtle" in narrative.narrative.lower()

    def test_deterministic_narrative_unknown_strategy(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer()

        pick = {
            "symbol": "MSFT",
            "strategy": "unknown_strategy",
            "entry_price": 400.0,
            "score": 10.0,
        }

        narrative = analyzer._deterministic_narrative(pick)

        assert narrative.symbol == "MSFT"
        assert "MSFT" in narrative.narrative


class TestTop3Analysis:
    """Tests for Top-3 picks analysis."""

    def test_analyze_empty_picks(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer()
        picks = pd.DataFrame()
        market_context = {"regime": "BULL", "vix": 18.0}

        narratives = analyzer.analyze_top3_picks(picks, market_context)

        assert narratives == []

    def test_analyze_top3_deterministic(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer(fallback_enabled=True)

        picks = pd.DataFrame([
            {"symbol": "AAPL", "strategy": "ibs_rsi", "entry_price": 175.0, "score": 12.0},
            {"symbol": "TSLA", "strategy": "turtle_soup", "entry_price": 250.0, "score": 150.0},
            {"symbol": "MSFT", "strategy": "ibs_rsi", "entry_price": 400.0, "score": 10.0},
        ])

        market_context = {
            "regime": "BULL",
            "regime_confidence": 0.8,
            "vix": 18.0,
            "sentiment": {"compound": 0.3},
            "spy_position": "above SMA(200)",
        }

        narratives = analyzer.analyze_top3_picks(picks, market_context)

        assert len(narratives) == 3
        assert narratives[0].symbol == "AAPL"
        assert narratives[1].symbol == "TSLA"
        assert narratives[2].symbol == "MSFT"


class TestTOTDAnalysis:
    """Tests for Trade of the Day analysis."""

    def test_analyze_empty_totd(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer()
        market_context = {"regime": "BULL", "vix": 18.0}

        analysis = analyzer.analyze_trade_of_day(None, market_context)

        assert "No Trade of the Day" in analysis

    def test_analyze_totd_deterministic(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer(fallback_enabled=True)

        totd = {
            "symbol": "NVDA",
            "strategy": "turtle_soup",
            "entry_price": 500.0,
            "stop_loss": 480.0,
            "take_profit": 540.0,
        }

        market_context = {
            "regime": "BULL",
            "vix": 18.0,
            "sentiment": {"compound": 0.2},
        }

        analysis = analyzer.analyze_trade_of_day(totd, market_context)

        assert "NVDA" in analysis
        assert len(analysis) > 100  # Should be substantial analysis


class TestMarketSummary:
    """Tests for market summary generation."""

    def test_market_summary_bull(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer(fallback_enabled=True)

        summary = analyzer.synthesize_market_summary(
            regime="BULL",
            vix=16.0,
            sentiment={"compound": 0.4},
            regime_confidence=0.85,
        )

        assert len(summary) > 0
        assert "BULL" in summary or "bullish" in summary.lower()

    def test_market_summary_bear(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer(fallback_enabled=True)

        summary = analyzer.synthesize_market_summary(
            regime="BEAR",
            vix=30.0,
            sentiment={"compound": -0.3},
            regime_confidence=0.75,
        )

        assert len(summary) > 0
        assert "BEAR" in summary or "bearish" in summary.lower()


class TestSentimentInterpretation:
    """Tests for sentiment interpretation."""

    def test_interpret_no_articles(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer(fallback_enabled=True)

        interp = analyzer.interpret_sentiment(
            articles=[],
            aggregated_sentiment={"compound": 0.0},
        )

        assert len(interp) > 0

    def test_interpret_with_articles(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer(fallback_enabled=True)

        articles = [
            {"title": "AAPL beats earnings", "sentiment": 0.6},
            {"title": "Market volatility rises", "sentiment": -0.2},
        ]

        interp = analyzer.interpret_sentiment(
            articles=articles,
            aggregated_sentiment={"compound": 0.2},
        )

        assert len(interp) > 0


class TestKeyFindings:
    """Tests for key findings identification."""

    def test_findings_empty_signals(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer(fallback_enabled=True)

        findings = analyzer.identify_key_findings(
            signals=pd.DataFrame(),
            market_context={"regime": "NEUTRAL"},
        )

        assert len(findings) >= 1
        assert "No signals" in findings[0] or "unusual" in findings[0].lower()

    def test_findings_with_signals(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer(fallback_enabled=True)

        signals = pd.DataFrame([
            {"symbol": "AAPL", "strategy": "ibs_rsi", "side": "long"},
            {"symbol": "MSFT", "strategy": "ibs_rsi", "side": "long"},
            {"symbol": "TSLA", "strategy": "turtle_soup", "side": "long"},
        ])

        findings = analyzer.identify_key_findings(
            signals=signals,
            market_context={"regime": "BULL", "vix": 18.0},
        )

        assert len(findings) >= 1


class TestDailyInsightReportGeneration:
    """Tests for full daily insight report generation."""

    def test_generate_full_report(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer(fallback_enabled=True)

        picks = pd.DataFrame([
            {"symbol": "AAPL", "strategy": "ibs_rsi", "entry_price": 175.0, "score": 12.0},
            {"symbol": "TSLA", "strategy": "turtle_soup", "entry_price": 250.0, "score": 150.0},
        ])

        totd = {"symbol": "AAPL", "strategy": "ibs_rsi", "entry_price": 175.0}

        market_context = {
            "regime": "BULL",
            "regime_confidence": 0.8,
            "vix": 18.0,
        }

        report = analyzer.generate_daily_insight_report(
            picks=picks,
            totd=totd,
            market_context=market_context,
            news_articles=[],
            sentiment={"compound": 0.1},
            all_signals=picks,
        )

        assert report.date is not None
        assert len(report.market_summary) > 0
        assert len(report.top3_narratives) >= 1
        assert len(report.totd_deep_analysis) > 0
        assert report.generation_method in ["claude", "deterministic"]


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def test_get_trade_analyzer(self):
        from cognitive.llm_trade_analyzer import get_trade_analyzer, reset_trade_analyzer

        # Reset first to ensure clean state
        reset_trade_analyzer()

        analyzer1 = get_trade_analyzer()
        analyzer2 = get_trade_analyzer()

        assert analyzer1 is analyzer2

    def test_reset_trade_analyzer(self):
        from cognitive.llm_trade_analyzer import get_trade_analyzer, reset_trade_analyzer

        analyzer1 = get_trade_analyzer()
        reset_trade_analyzer()
        analyzer2 = get_trade_analyzer()

        # After reset, should be a new instance
        assert analyzer1 is not analyzer2


class TestEdgeDescriptions:
    """Tests for edge description generation."""

    def test_edge_description_ibs_rsi(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer()
        edge = analyzer._get_edge_description("ibs_rsi")

        assert "Mean reversion" in edge or "oversold" in edge.lower()
        assert "62%" in edge or "win rate" in edge.lower()

    def test_edge_description_turtle_soup(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer()
        edge = analyzer._get_edge_description("turtle_soup")

        assert "Liquidity" in edge or "sweep" in edge.lower()

    def test_edge_description_unknown(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer()
        edge = analyzer._get_edge_description("unknown_strat")

        assert "unknown_strat" in edge


class TestConfidenceRating:
    """Tests for confidence rating determination."""

    def test_high_confidence(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer()
        rating = analyzer._determine_confidence({"score": 20})

        assert rating == "HIGH"

    def test_medium_confidence(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer()
        rating = analyzer._determine_confidence({"score": 10})

        assert rating == "MEDIUM"

    def test_low_confidence(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer()
        rating = analyzer._determine_confidence({"score": 5})

        assert rating == "LOW"


class TestRiskWarnings:
    """Tests for risk warning generation."""

    def test_high_vix_warning(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer()
        warnings = analyzer._generate_risk_warnings({"vix": 35.0})

        assert len(warnings) >= 1
        assert any("VIX" in w for w in warnings)

    def test_friday_warning(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer()

        # The method uses datetime.now().weekday() internally
        # We just test that warnings can be generated without error
        warnings = analyzer._generate_risk_warnings({"vix": 18.0, "regime": "BULL"})

        # Should return a list
        assert isinstance(warnings, list)


class TestOpportunities:
    """Tests for opportunity generation."""

    def test_opportunities_with_picks(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer()

        picks = pd.DataFrame([
            {"symbol": "AAPL", "strategy": "ibs_rsi"},
            {"symbol": "MSFT", "strategy": "turtle_soup"},
        ])

        opportunities = analyzer._generate_opportunities(picks, {"regime": "BULL"})

        assert len(opportunities) >= 1

    def test_opportunities_bull_regime(self):
        from cognitive.llm_trade_analyzer import LLMTradeAnalyzer

        analyzer = LLMTradeAnalyzer()

        opportunities = analyzer._generate_opportunities(pd.DataFrame(), {"regime": "BULL"})

        assert any("Bull" in opp or "dip" in opp.lower() for opp in opportunities)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
