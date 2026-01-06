"""
Unified Signal Enrichment Pipeline
===================================

This module orchestrates ALL available AI/ML/Data components to enrich signals
from the raw DualStrategyScanner output to fully-analyzed TOP 2 trades.

Pipeline Flow:
    900 stocks → DualStrategyScanner → Raw Signals
                        ↓
    [25+ Enrichment Stages - ALL COMPONENTS WIRED]
                        ↓
    TOP 5 Watchlist (study/follow/learn)
                        ↓
    TOP 2 Trades of Day (full thesis with Claude reasoning)

This is the BRAIN of the trading system - it connects everything.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# COMPONENT AVAILABILITY TRACKING
# =============================================================================

@dataclass
class ComponentStatus:
    """Tracks which components are available and wired."""
    name: str
    available: bool
    wired: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ComponentRegistry:
    """Registry of all available components."""

    def __init__(self):
        self.components: Dict[str, ComponentStatus] = {}
        self._load_all_components()

    def _load_all_components(self):
        """Attempt to load all components and track availability."""

        # =====================================================================
        # CATEGORY 1: HISTORICAL PATTERNS & ANALYSIS
        # =====================================================================
        try:
            from analysis.historical_patterns import (
                HistoricalPatternAnalyzer,
                enrich_signal_with_historical_pattern,
                qualifies_for_auto_pass,
                get_pattern_grade,
            )
            self.historical_patterns = HistoricalPatternAnalyzer()
            self.enrich_historical = enrich_signal_with_historical_pattern
            self.qualifies_for_auto_pass = qualifies_for_auto_pass
            self.get_pattern_grade = get_pattern_grade
            self.components['historical_patterns'] = ComponentStatus('Historical Patterns', True, True)
        except ImportError as e:
            self.historical_patterns = None
            self.components['historical_patterns'] = ComponentStatus('Historical Patterns', False, error=str(e))

        try:
            from analysis.options_expected_move import ExpectedMoveCalculator
            self.expected_move_calc = ExpectedMoveCalculator()
            self.components['expected_move'] = ComponentStatus('Expected Move Calculator', True, True)
        except ImportError as e:
            self.expected_move_calc = None
            self.components['expected_move'] = ComponentStatus('Expected Move Calculator', False, error=str(e))

        # =====================================================================
        # CATEGORY 2: ML META (Primary Confidence Scoring)
        # =====================================================================
        try:
            from ml_meta.features import compute_features_frame
            from ml_meta.model import load_model, predict_proba, FEATURE_COLS
            self.compute_features = compute_features_frame
            self.ml_load_model = load_model
            self.ml_predict = predict_proba
            self.ml_feature_cols = FEATURE_COLS
            self.components['ml_meta'] = ComponentStatus('ML Meta (XGBoost/LightGBM)', True, True)
        except ImportError as e:
            self.compute_features = None
            self.components['ml_meta'] = ComponentStatus('ML Meta (XGBoost/LightGBM)', False, error=str(e))

        try:
            from ml_meta.calibration import IsotonicCalibrator, PlattCalibrator, get_calibrator, calibrate_probability
            self.isotonic_calibrator = IsotonicCalibrator
            self.platt_calibrator = PlattCalibrator
            self.get_calibrator = get_calibrator
            self.calibrate_probability = calibrate_probability
            self.components['ml_calibration'] = ComponentStatus('ML Calibration', True, True)
        except ImportError as e:
            self.isotonic_calibrator = None
            self.components['ml_calibration'] = ComponentStatus('ML Calibration', False, error=str(e))

        try:
            from ml_meta.conformal import ConformalPredictor
            self.conformal_predictor = ConformalPredictor
            self.components['conformal_prediction'] = ComponentStatus('Conformal Prediction', True, True)
        except ImportError as e:
            self.conformal_predictor = None
            self.components['conformal_prediction'] = ComponentStatus('Conformal Prediction', False, error=str(e))

        # =====================================================================
        # CATEGORY 3: ML ADVANCED (LSTM, Ensemble, HMM, Markov)
        # =====================================================================
        try:
            from ml_advanced.lstm_confidence.model import LSTMConfidenceModel
            self.lstm_model = LSTMConfidenceModel
            self.components['lstm_confidence'] = ComponentStatus('LSTM Confidence', True, True)
        except ImportError as e:
            self.lstm_model = None
            self.components['lstm_confidence'] = ComponentStatus('LSTM Confidence', False, error=str(e))

        try:
            from ml_advanced.ensemble.ensemble_predictor import EnsemblePredictor
            self.ensemble_predictor = EnsemblePredictor
            self.components['ensemble_predictor'] = ComponentStatus('Ensemble Predictor', True, True)
        except ImportError as e:
            self.ensemble_predictor = None
            self.components['ensemble_predictor'] = ComponentStatus('Ensemble Predictor', False, error=str(e))

        try:
            from ml_advanced.hmm_regime_detector import AdaptiveRegimeDetector, MarketRegime
            self.regime_detector = AdaptiveRegimeDetector
            self.market_regime = MarketRegime
            self.components['hmm_regime'] = ComponentStatus('HMM Regime Detector', True, True)
        except ImportError as e:
            self.regime_detector = None
            self.components['hmm_regime'] = ComponentStatus('HMM Regime Detector', False, error=str(e))

        try:
            from ml_advanced.markov_chain import MarkovPredictor, MarkovAssetScorer
            self.markov_predictor = MarkovPredictor
            self.markov_scorer = MarkovAssetScorer
            self.components['markov_chain'] = ComponentStatus('Markov Chain Predictor', True, True)
        except ImportError as e:
            self.markov_predictor = None
            self.components['markov_chain'] = ComponentStatus('Markov Chain Predictor', False, error=str(e))

        try:
            from ml_advanced.online_learning import OnlineLearningManager
            self.online_learning = OnlineLearningManager
            self.components['online_learning'] = ComponentStatus('Online Learning', True, True)
        except ImportError as e:
            self.online_learning = None
            self.components['online_learning'] = ComponentStatus('Online Learning', False, error=str(e))

        # =====================================================================
        # CATEGORY 4: ML FEATURES (Feature Engineering)
        # =====================================================================
        try:
            from ml_features.conviction_scorer import ConvictionScorer
            self.conviction_scorer = ConvictionScorer()
            self.components['conviction_scorer'] = ComponentStatus('Conviction Scorer', True, True)
        except ImportError as e:
            self.conviction_scorer = None
            self.components['conviction_scorer'] = ComponentStatus('Conviction Scorer', False, error=str(e))

        try:
            from ml_features.confidence_integrator import ConfidenceIntegrator, get_confidence_integrator, get_ml_confidence
            self.confidence_integrator = ConfidenceIntegrator
            self.get_confidence_integrator = get_confidence_integrator
            self.get_ml_confidence = get_ml_confidence
            self.components['confidence_integrator'] = ComponentStatus('Confidence Integrator', True, True)
        except ImportError as e:
            self.confidence_integrator = None
            self.components['confidence_integrator'] = ComponentStatus('Confidence Integrator', False, error=str(e))

        try:
            from ml_features.feature_pipeline import FeaturePipeline
            self.feature_pipeline = FeaturePipeline
            self.components['feature_pipeline'] = ComponentStatus('Feature Pipeline (150+)', True, True)
        except ImportError as e:
            self.feature_pipeline = None
            self.components['feature_pipeline'] = ComponentStatus('Feature Pipeline (150+)', False, error=str(e))

        try:
            from ml_features.anomaly_detection import AnomalyDetector
            self.anomaly_detector = AnomalyDetector
            self.components['anomaly_detection'] = ComponentStatus('Anomaly Detection', True, True)
        except ImportError as e:
            self.anomaly_detector = None
            self.components['anomaly_detection'] = ComponentStatus('Anomaly Detection', False, error=str(e))

        try:
            from ml_features.ensemble_brain import EnsembleBrain
            self.ensemble_brain = EnsembleBrain
            self.components['ensemble_brain'] = ComponentStatus('Ensemble Brain', True, True)
        except ImportError as e:
            self.ensemble_brain = None
            self.components['ensemble_brain'] = ComponentStatus('Ensemble Brain', False, error=str(e))

        # =====================================================================
        # CATEGORY 5: COGNITIVE SYSTEM (Brain, Reasoning, Memory)
        # =====================================================================
        try:
            from cognitive.cognitive_brain import CognitiveBrain
            self.cognitive_brain = CognitiveBrain
            self.components['cognitive_brain'] = ComponentStatus('Cognitive Brain', True, True)
        except ImportError as e:
            self.cognitive_brain = None
            self.components['cognitive_brain'] = ComponentStatus('Cognitive Brain', False, error=str(e))

        try:
            from cognitive.signal_processor import get_signal_processor
            self.signal_processor = get_signal_processor
            self.components['signal_processor'] = ComponentStatus('Signal Processor', True, True)
        except ImportError as e:
            self.signal_processor = None
            self.components['signal_processor'] = ComponentStatus('Signal Processor', False, error=str(e))

        try:
            from cognitive.signal_adjudicator import adjudicate_signals, SignalAdjudicator
            self.adjudicate_signals = adjudicate_signals
            self.signal_adjudicator = SignalAdjudicator
            self.components['signal_adjudicator'] = ComponentStatus('Signal Adjudicator', True, True)
        except ImportError as e:
            self.adjudicate_signals = None
            self.components['signal_adjudicator'] = ComponentStatus('Signal Adjudicator', False, error=str(e))

        try:
            from cognitive.llm_trade_analyzer import get_trade_analyzer
            self.llm_analyzer = get_trade_analyzer
            self.components['llm_trade_analyzer'] = ComponentStatus('LLM Trade Analyzer (Claude)', True, True)
        except ImportError as e:
            self.llm_analyzer = None
            self.components['llm_trade_analyzer'] = ComponentStatus('LLM Trade Analyzer (Claude)', False, error=str(e))

        try:
            from cognitive.knowledge_boundary import KnowledgeBoundary
            self.knowledge_boundary = KnowledgeBoundary
            self.components['knowledge_boundary'] = ComponentStatus('Knowledge Boundary', True, True)
        except ImportError as e:
            self.knowledge_boundary = None
            self.components['knowledge_boundary'] = ComponentStatus('Knowledge Boundary', False, error=str(e))

        try:
            from cognitive.episodic_memory import EpisodicMemory
            self.episodic_memory = EpisodicMemory
            self.components['episodic_memory'] = ComponentStatus('Episodic Memory', True, True)
        except ImportError as e:
            self.episodic_memory = None
            self.components['episodic_memory'] = ComponentStatus('Episodic Memory', False, error=str(e))

        try:
            from cognitive.semantic_memory import SemanticMemory
            self.semantic_memory = SemanticMemory
            self.components['semantic_memory'] = ComponentStatus('Semantic Memory', True, True)
        except ImportError as e:
            self.semantic_memory = None
            self.components['semantic_memory'] = ComponentStatus('Semantic Memory', False, error=str(e))

        try:
            from cognitive.reflection_engine import ReflectionEngine
            self.reflection_engine = ReflectionEngine
            self.components['reflection_engine'] = ComponentStatus('Reflection Engine', True, True)
        except ImportError as e:
            self.reflection_engine = None
            self.components['reflection_engine'] = ComponentStatus('Reflection Engine', False, error=str(e))

        try:
            from cognitive.curiosity_engine import CuriosityEngine
            self.curiosity_engine = CuriosityEngine
            self.components['curiosity_engine'] = ComponentStatus('Curiosity Engine', True, True)
        except ImportError as e:
            self.curiosity_engine = None
            self.components['curiosity_engine'] = ComponentStatus('Curiosity Engine', False, error=str(e))

        try:
            from cognitive.metacognitive_governor import MetacognitiveGovernor
            self.metacognitive_governor = MetacognitiveGovernor
            self.components['metacognitive_governor'] = ComponentStatus('Metacognitive Governor', True, True)
        except ImportError as e:
            self.metacognitive_governor = None
            self.components['metacognitive_governor'] = ComponentStatus('Metacognitive Governor', False, error=str(e))

        # =====================================================================
        # CATEGORY 6: RISK MANAGEMENT
        # =====================================================================
        try:
            from risk.signal_quality_gate import SignalQualityGate, filter_to_best_signals
            self.quality_gate = SignalQualityGate
            self.filter_to_best = filter_to_best_signals
            self.components['quality_gate'] = ComponentStatus('Signal Quality Gate', True, True)
        except ImportError as e:
            self.quality_gate = None
            self.components['quality_gate'] = ComponentStatus('Signal Quality Gate', False, error=str(e))

        try:
            from risk.advanced.kelly_position_sizer import KellyPositionSizer
            self.kelly_sizer = KellyPositionSizer
            self.components['kelly_sizer'] = ComponentStatus('Kelly Position Sizer', True, True)
        except ImportError as e:
            self.kelly_sizer = None
            self.components['kelly_sizer'] = ComponentStatus('Kelly Position Sizer', False, error=str(e))

        try:
            from risk.advanced.monte_carlo_var import MonteCarloVaR
            self.monte_carlo_var = MonteCarloVaR
            self.components['monte_carlo_var'] = ComponentStatus('Monte Carlo VaR', True, True)
        except ImportError as e:
            self.monte_carlo_var = None
            self.components['monte_carlo_var'] = ComponentStatus('Monte Carlo VaR', False, error=str(e))

        try:
            from risk.advanced.correlation_limits import EnhancedCorrelationLimits
            self.correlation_limits = EnhancedCorrelationLimits
            self.components['correlation_limits'] = ComponentStatus('Correlation Limits', True, True)
        except ImportError as e:
            self.correlation_limits = None
            self.components['correlation_limits'] = ComponentStatus('Correlation Limits', False, error=str(e))

        try:
            from risk.circuit_breakers.breaker_manager import BreakerManager
            self.circuit_breakers = BreakerManager
            self.components['circuit_breakers'] = ComponentStatus('Circuit Breakers', True, True)
        except ImportError as e:
            self.circuit_breakers = None
            self.components['circuit_breakers'] = ComponentStatus('Circuit Breakers', False, error=str(e))

        try:
            from risk.kill_zone_gate import can_trade_now, get_current_zone
            self.can_trade_now = can_trade_now
            self.get_current_zone = get_current_zone
            self.components['kill_zone_gate'] = ComponentStatus('Kill Zone Gate', True, True)
        except ImportError as e:
            self.can_trade_now = None
            self.components['kill_zone_gate'] = ComponentStatus('Kill Zone Gate', False, error=str(e))

        # =====================================================================
        # CATEGORY 7: ALT DATA (News, Insider, Congress, Options Flow)
        # =====================================================================
        try:
            from data.alternative.alt_data_aggregator import AltDataAggregator
            self.alt_data_aggregator = AltDataAggregator()
            self.components['alt_data_aggregator'] = ComponentStatus('Alt Data Aggregator', True, True)
        except ImportError as e:
            self.alt_data_aggregator = None
            self.components['alt_data_aggregator'] = ComponentStatus('Alt Data Aggregator', False, error=str(e))

        try:
            from altdata.sentiment import load_daily_cache, normalize_sentiment_to_conf
            self.load_sentiment = load_daily_cache
            self.normalize_sentiment = normalize_sentiment_to_conf
            self.components['sentiment'] = ComponentStatus('Sentiment Analysis', True, True)
        except ImportError as e:
            self.load_sentiment = None
            self.components['sentiment'] = ComponentStatus('Sentiment Analysis', False, error=str(e))

        try:
            from altdata.news_processor import get_news_processor
            self.news_processor = get_news_processor
            self.components['news_processor'] = ComponentStatus('News Processor', True, True)
        except ImportError as e:
            self.news_processor = None
            self.components['news_processor'] = ComponentStatus('News Processor', False, error=str(e))

        try:
            from altdata.insider_activity import InsiderActivityClient, get_insider_client
            self.insider_client = InsiderActivityClient
            self.get_insider_client = get_insider_client
            self.components['insider_activity'] = ComponentStatus('Insider Activity', True, True)
        except ImportError as e:
            self.insider_client = None
            self.components['insider_activity'] = ComponentStatus('Insider Activity', False, error=str(e))

        try:
            from altdata.congressional_trades import CongressionalTradesClient, get_congressional_client
            self.congress_client = CongressionalTradesClient
            self.get_congressional_client = get_congressional_client
            self.components['congressional_trades'] = ComponentStatus('Congressional Trades', True, True)
        except ImportError as e:
            self.congress_client = None
            self.components['congressional_trades'] = ComponentStatus('Congressional Trades', False, error=str(e))

        try:
            from altdata.options_flow import OptionsFlowClient, get_options_flow_client
            self.options_flow_client = OptionsFlowClient
            self.get_options_flow_client = get_options_flow_client
            self.components['options_flow'] = ComponentStatus('Options Flow', True, True)
        except ImportError as e:
            self.options_flow_client = None
            self.components['options_flow'] = ComponentStatus('Options Flow', False, error=str(e))

        # =====================================================================
        # CATEGORY 8: EXPLAINABILITY (Trade Thesis, Narratives)
        # =====================================================================
        try:
            from explainability.trade_thesis_builder import TradeThesisBuilder
            self.thesis_builder = TradeThesisBuilder()
            self.components['trade_thesis_builder'] = ComponentStatus('Trade Thesis Builder', True, True)
        except ImportError as e:
            self.thesis_builder = None
            self.components['trade_thesis_builder'] = ComponentStatus('Trade Thesis Builder', False, error=str(e))

        try:
            from explainability.narrative_generator import NarrativeGenerator
            self.narrative_generator = NarrativeGenerator
            self.components['narrative_generator'] = ComponentStatus('Narrative Generator', True, True)
        except ImportError as e:
            self.narrative_generator = None
            self.components['narrative_generator'] = ComponentStatus('Narrative Generator', False, error=str(e))

        try:
            from explainability.decision_tracker import DecisionTracker
            self.decision_tracker = DecisionTracker
            self.components['decision_tracker'] = ComponentStatus('Decision Tracker', True, True)
        except ImportError as e:
            self.decision_tracker = None
            self.components['decision_tracker'] = ComponentStatus('Decision Tracker', False, error=str(e))

        # =====================================================================
        # CATEGORY 9: PORTFOLIO MANAGEMENT
        # =====================================================================
        try:
            from portfolio.heat_monitor import get_heat_monitor
            self.heat_monitor = get_heat_monitor
            self.components['heat_monitor'] = ComponentStatus('Heat Monitor', True, True)
        except ImportError as e:
            self.heat_monitor = None
            self.components['heat_monitor'] = ComponentStatus('Heat Monitor', False, error=str(e))

        try:
            from portfolio.risk_manager import PortfolioRiskManager
            self.portfolio_risk_manager = PortfolioRiskManager
            self.components['portfolio_risk_manager'] = ComponentStatus('Portfolio Risk Manager', True, True)
        except ImportError as e:
            self.portfolio_risk_manager = None
            self.components['portfolio_risk_manager'] = ComponentStatus('Portfolio Risk Manager', False, error=str(e))

        # =====================================================================
        # CATEGORY 10: GUARDIAN SYSTEM (24/7 Oversight)
        # =====================================================================
        try:
            from guardian.guardian import Guardian
            self.guardian = Guardian
            self.components['guardian'] = ComponentStatus('Guardian', True, True)
        except ImportError as e:
            self.guardian = None
            self.components['guardian'] = ComponentStatus('Guardian', False, error=str(e))

        try:
            from guardian.alert_manager import AlertManager
            self.alert_manager = AlertManager
            self.components['alert_manager'] = ComponentStatus('Alert Manager', True, True)
        except ImportError as e:
            self.alert_manager = None
            self.components['alert_manager'] = ComponentStatus('Alert Manager', False, error=str(e))

        # =====================================================================
        # CATEGORY 11: ANALYTICS
        # =====================================================================
        try:
            from analytics.auto_standdown import AutoStanddown
            self.auto_standdown = AutoStanddown
            self.components['auto_standdown'] = ComponentStatus('Auto Standdown', True, True)
        except ImportError as e:
            self.auto_standdown = None
            self.components['auto_standdown'] = ComponentStatus('Auto Standdown', False, error=str(e))

        try:
            from analytics.edge_decomposition import EdgeDecomposition, DimensionStats, DecompositionResult
            self.edge_decomposition = EdgeDecomposition
            self.dimension_stats = DimensionStats
            self.decomposition_result = DecompositionResult
            self.components['edge_decomposition'] = ComponentStatus('Edge Decomposition', True, True)
        except ImportError as e:
            self.edge_decomposition = None
            self.components['edge_decomposition'] = ComponentStatus('Edge Decomposition', False, error=str(e))

        # =====================================================================
        # CATEGORY 12: BOUNCE/STREAK ANALYSIS
        # =====================================================================
        try:
            from bounce.bounce_score import (
                calculate_bounce_score,
                apply_bounce_gates,
                get_bounce_profile_for_signal,
                adjust_signal_for_bounce,
                rank_signals_by_bounce,
            )
            self.calculate_bounce_score = calculate_bounce_score
            self.apply_bounce_gates = apply_bounce_gates
            self.get_bounce_profile = get_bounce_profile_for_signal
            self.adjust_signal_bounce = adjust_signal_for_bounce
            self.rank_by_bounce = rank_signals_by_bounce
            self.components['bounce_scorer'] = ComponentStatus('Bounce Scorer', True, True)
        except ImportError as e:
            self.calculate_bounce_score = None
            self.components['bounce_scorer'] = ComponentStatus('Bounce Scorer', False, error=str(e))

        try:
            from bounce.streak_analyzer import (
                calculate_streaks_vectorized,
                calculate_forward_metrics,
                detect_events,
                analyze_ticker,
                get_current_streaks,
            )
            self.calculate_streaks = calculate_streaks_vectorized
            self.calculate_forward_metrics = calculate_forward_metrics
            self.detect_events = detect_events
            self.analyze_ticker = analyze_ticker
            self.get_current_streaks = get_current_streaks
            self.components['streak_analyzer'] = ComponentStatus('Streak Analyzer', True, True)
        except ImportError as e:
            self.calculate_streaks = None
            self.components['streak_analyzer'] = ComponentStatus('Streak Analyzer', False, error=str(e))

        # =====================================================================
        # CATEGORY 13: DATA PROVIDERS
        # =====================================================================
        try:
            from data.providers.fred_macro import FREDMacroProvider
            self.fred_macro = FREDMacroProvider
            self.components['fred_macro'] = ComponentStatus('FRED Macro Data', True, True)
        except ImportError as e:
            self.fred_macro = None
            self.components['fred_macro'] = ComponentStatus('FRED Macro Data', False, error=str(e))

        try:
            from core.vix_monitor import get_vix_monitor
            self.vix_monitor = get_vix_monitor
            self.components['vix_monitor'] = ComponentStatus('VIX Monitor', True, True)
        except ImportError as e:
            self.vix_monitor = None
            self.components['vix_monitor'] = ComponentStatus('VIX Monitor', False, error=str(e))

        try:
            from core.regime_filter import filter_signals_by_regime, fetch_spy_bars
            self.filter_by_regime = filter_signals_by_regime
            self.fetch_spy = fetch_spy_bars
            self.components['regime_filter'] = ComponentStatus('Regime Filter', True, True)
        except ImportError as e:
            self.filter_by_regime = None
            self.components['regime_filter'] = ComponentStatus('Regime Filter', False, error=str(e))

        try:
            from core.earnings_filter import filter_signals_by_earnings
            self.filter_by_earnings = filter_signals_by_earnings
            self.components['earnings_filter'] = ComponentStatus('Earnings Filter', True, True)
        except ImportError as e:
            self.filter_by_earnings = None
            self.components['earnings_filter'] = ComponentStatus('Earnings Filter', False, error=str(e))

        # =====================================================================
        # CATEGORY 14: MORE COGNITIVE COMPONENTS
        # =====================================================================
        try:
            from cognitive.global_workspace import GlobalWorkspace
            self.global_workspace = GlobalWorkspace
            self.components['global_workspace'] = ComponentStatus('Global Workspace', True, True)
        except ImportError as e:
            self.global_workspace = None
            self.components['global_workspace'] = ComponentStatus('Global Workspace', False, error=str(e))

        try:
            from cognitive.symbolic_reasoner import SymbolicReasoner
            self.symbolic_reasoner = SymbolicReasoner
            self.components['symbolic_reasoner'] = ComponentStatus('Symbolic Reasoner', True, True)
        except ImportError as e:
            self.symbolic_reasoner = None
            self.components['symbolic_reasoner'] = ComponentStatus('Symbolic Reasoner', False, error=str(e))

        try:
            from cognitive.dynamic_policy_generator import DynamicPolicyGenerator
            self.dynamic_policy_generator = DynamicPolicyGenerator
            self.components['dynamic_policy_generator'] = ComponentStatus('Dynamic Policy Generator', True, True)
        except ImportError as e:
            self.dynamic_policy_generator = None
            self.components['dynamic_policy_generator'] = ComponentStatus('Dynamic Policy Generator', False, error=str(e))

        try:
            from cognitive.self_model import SelfModel
            self.self_model = SelfModel
            self.components['self_model'] = ComponentStatus('Self Model', True, True)
        except ImportError as e:
            self.self_model = None
            self.components['self_model'] = ComponentStatus('Self Model', False, error=str(e))

        try:
            from cognitive.llm_validator import LLMValidator, get_validator
            self.llm_validator = get_validator
            self.components['llm_validator'] = ComponentStatus('LLM Validator', True, True)
        except ImportError as e:
            self.llm_validator = None
            self.components['llm_validator'] = ComponentStatus('LLM Validator', False, error=str(e))

        try:
            from cognitive.vector_memory import VectorMemory
            self.vector_memory = VectorMemory
            self.components['vector_memory'] = ComponentStatus('Vector Memory', True, True)
        except ImportError as e:
            self.vector_memory = None
            self.components['vector_memory'] = ComponentStatus('Vector Memory', False, error=str(e))

        # =====================================================================
        # CATEGORY 15: MORE RISK COMPONENTS
        # =====================================================================
        try:
            from risk.policy_gate import PolicyGate
            self.policy_gate = PolicyGate
            self.components['policy_gate'] = ComponentStatus('Policy Gate', True, True)
        except ImportError as e:
            self.policy_gate = None
            self.components['policy_gate'] = ComponentStatus('Policy Gate', False, error=str(e))

        try:
            from risk.weekly_exposure_gate import WeeklyExposureGate
            self.weekly_exposure_gate = WeeklyExposureGate
            self.components['weekly_exposure_gate'] = ComponentStatus('Weekly Exposure Gate', True, True)
        except ImportError as e:
            self.weekly_exposure_gate = None
            self.components['weekly_exposure_gate'] = ComponentStatus('Weekly Exposure Gate', False, error=str(e))

        try:
            from risk.dynamic_position_sizer import AllocationResult, calculate_dynamic_allocations
            self.dynamic_position_sizer = calculate_dynamic_allocations
            self.components['dynamic_position_sizer'] = ComponentStatus('Dynamic Position Sizer', True, True)
        except ImportError as e:
            self.dynamic_position_sizer = None
            self.components['dynamic_position_sizer'] = ComponentStatus('Dynamic Position Sizer', False, error=str(e))

        try:
            from risk.equity_sizer import PositionSize, calculate_position_size, get_account_equity
            self.equity_sizer = calculate_position_size
            self.components['equity_sizer'] = ComponentStatus('Equity Sizer', True, True)
        except ImportError as e:
            self.equity_sizer = None
            self.components['equity_sizer'] = ComponentStatus('Equity Sizer', False, error=str(e))

        try:
            from risk.circuit_breakers.drawdown_breaker import DrawdownBreaker
            self.drawdown_breaker = DrawdownBreaker
            self.components['drawdown_breaker'] = ComponentStatus('Drawdown Breaker', True, True)
        except ImportError as e:
            self.drawdown_breaker = None
            self.components['drawdown_breaker'] = ComponentStatus('Drawdown Breaker', False, error=str(e))

        try:
            from risk.circuit_breakers.volatility_breaker import VolatilityBreaker
            self.volatility_breaker = VolatilityBreaker
            self.components['volatility_breaker'] = ComponentStatus('Volatility Breaker', True, True)
        except ImportError as e:
            self.volatility_breaker = None
            self.components['volatility_breaker'] = ComponentStatus('Volatility Breaker', False, error=str(e))

        try:
            from risk.circuit_breakers.streak_breaker import StreakBreaker
            self.streak_breaker = StreakBreaker
            self.components['streak_breaker'] = ComponentStatus('Streak Breaker', True, True)
        except ImportError as e:
            self.streak_breaker = None
            self.components['streak_breaker'] = ComponentStatus('Streak Breaker', False, error=str(e))

        try:
            from risk.factor_model.factor_calculator import FactorCalculator
            self.factor_calculator = FactorCalculator
            self.components['factor_calculator'] = ComponentStatus('Factor Calculator', True, True)
        except ImportError as e:
            self.factor_calculator = None
            self.components['factor_calculator'] = ComponentStatus('Factor Calculator', False, error=str(e))

        try:
            from risk.factor_model.sector_exposure import SectorAnalyzer, SectorExposures
            self.sector_exposure = SectorAnalyzer
            self.components['sector_exposure'] = ComponentStatus('Sector Exposure', True, True)
        except ImportError as e:
            self.sector_exposure = None
            self.components['sector_exposure'] = ComponentStatus('Sector Exposure', False, error=str(e))

        # =====================================================================
        # CATEGORY 16: MORE ML COMPONENTS
        # =====================================================================
        try:
            from ml_features.pca_reducer import PCAReducer
            self.pca_reducer = PCAReducer
            self.components['pca_reducer'] = ComponentStatus('PCA Reducer', True, True)
        except ImportError as e:
            self.pca_reducer = None
            self.components['pca_reducer'] = ComponentStatus('PCA Reducer', False, error=str(e))

        try:
            from ml_features.technical_features import TechnicalFeatures
            self.technical_features = TechnicalFeatures
            self.components['technical_features'] = ComponentStatus('Technical Features', True, True)
        except ImportError as e:
            self.technical_features = None
            self.components['technical_features'] = ComponentStatus('Technical Features', False, error=str(e))

        try:
            from ml_features.macro_features import MacroFeatureGenerator, get_macro_features
            self.macro_features = MacroFeatureGenerator
            self.components['macro_features'] = ComponentStatus('Macro Features', True, True)
        except ImportError as e:
            self.macro_features = None
            self.components['macro_features'] = ComponentStatus('Macro Features', False, error=str(e))

        try:
            from ml_features.regime_ml import RegimeDetectorML, detect_regime_ml
            self.regime_ml = RegimeDetectorML
            self.components['regime_ml'] = ComponentStatus('Regime ML', True, True)
        except ImportError as e:
            self.regime_ml = None
            self.components['regime_ml'] = ComponentStatus('Regime ML', False, error=str(e))

        try:
            from ml.confidence_gate import GateConfig, approve as confidence_approve
            self.confidence_gate = confidence_approve
            self.components['confidence_gate'] = ComponentStatus('Confidence Gate', True, True)
        except ImportError as e:
            self.confidence_gate = None
            self.components['confidence_gate'] = ComponentStatus('Confidence Gate', False, error=str(e))

        try:
            from ml_advanced.markov_chain.scorer import MarkovAssetScorer
            self.markov_detailed_scorer = MarkovAssetScorer
            self.components['markov_scorer'] = ComponentStatus('Markov Scorer', True, True)
        except ImportError as e:
            self.markov_detailed_scorer = None
            self.components['markov_scorer'] = ComponentStatus('Markov Scorer', False, error=str(e))

        try:
            from ml_advanced.ensemble.regime_weights import RegimeWeights
            self.regime_weights = RegimeWeights
            self.components['regime_weights'] = ComponentStatus('Regime Weights', True, True)
        except ImportError as e:
            self.regime_weights = None
            self.components['regime_weights'] = ComponentStatus('Regime Weights', False, error=str(e))

        # =====================================================================
        # CATEGORY 17: DATA PROVIDERS & UNIVERSE
        # =====================================================================
        try:
            from data.providers.polygon_eod import fetch_daily_bars_polygon, PolygonConfig
            self.polygon_eod = fetch_daily_bars_polygon
            self.components['polygon_eod'] = ComponentStatus('Polygon EOD Provider', True, True)
        except ImportError as e:
            self.polygon_eod = None
            self.components['polygon_eod'] = ComponentStatus('Polygon EOD Provider', False, error=str(e))

        try:
            from data.providers.alpaca_live import get_latest_quote, fetch_bars_alpaca, get_current_price
            self.alpaca_live = fetch_bars_alpaca
            self.components['alpaca_live'] = ComponentStatus('Alpaca Live Provider', True, True)
        except ImportError as e:
            self.alpaca_live = None
            self.components['alpaca_live'] = ComponentStatus('Alpaca Live Provider', False, error=str(e))

        try:
            from data.universe.loader import load_universe, load_canonical_900
            self.universe_loader = load_universe
            self.components['universe_loader'] = ComponentStatus('Universe Loader', True, True)
        except ImportError as e:
            self.universe_loader = None
            self.components['universe_loader'] = ComponentStatus('Universe Loader', False, error=str(e))

        try:
            from data.validation import OHLCVValidator, validate_ohlcv, DataQualityReport
            self.data_validator = OHLCVValidator
            self.components['data_validator'] = ComponentStatus('Data Validator', True, True)
        except ImportError as e:
            self.data_validator = None
            self.components['data_validator'] = ComponentStatus('Data Validator', False, error=str(e))

        try:
            from data.corporate_actions import CorporateActionsTracker, get_tracker
            self.corporate_actions = CorporateActionsTracker
            self.components['corporate_actions'] = ComponentStatus('Corporate Actions', True, True)
        except ImportError as e:
            self.corporate_actions = None
            self.components['corporate_actions'] = ComponentStatus('Corporate Actions', False, error=str(e))

        # =====================================================================
        # CATEGORY 18: QUANT GATES
        # =====================================================================
        try:
            from quant_gates.gate_0_sanity import Gate0Sanity, SanityResult
            self.sanity_gate = Gate0Sanity
            self.components['sanity_gate'] = ComponentStatus('Sanity Gate', True, True)
        except ImportError as e:
            self.sanity_gate = None
            self.components['sanity_gate'] = ComponentStatus('Sanity Gate', False, error=str(e))

        try:
            from quant_gates.gate_1_baseline import Gate1Baseline, BaselineResult
            self.baseline_gate = Gate1Baseline
            self.components['baseline_gate'] = ComponentStatus('Baseline Gate', True, True)
        except ImportError as e:
            self.baseline_gate = None
            self.components['baseline_gate'] = ComponentStatus('Baseline Gate', False, error=str(e))

        try:
            from quant_gates.gate_2_robustness import Gate2Robustness, RobustnessResult
            self.robustness_gate = Gate2Robustness
            self.components['robustness_gate'] = ComponentStatus('Robustness Gate', True, True)
        except ImportError as e:
            self.robustness_gate = None
            self.components['robustness_gate'] = ComponentStatus('Robustness Gate', False, error=str(e))

        try:
            from quant_gates.gate_3_risk import Gate3RiskRealism, RiskResult
            self.risk_gate = Gate3RiskRealism
            self.components['risk_gate'] = ComponentStatus('Risk Gate', True, True)
        except ImportError as e:
            self.risk_gate = None
            self.components['risk_gate'] = ComponentStatus('Risk Gate', False, error=str(e))

        try:
            from quant_gates.pipeline import QuantGatesPipeline, GateResult, PipelineResult
            self.quant_gate_pipeline = QuantGatesPipeline
            self.components['quant_gate_pipeline'] = ComponentStatus('Quant Gate Pipeline', True, True)
        except ImportError as e:
            self.quant_gate_pipeline = None
            self.components['quant_gate_pipeline'] = ComponentStatus('Quant Gate Pipeline', False, error=str(e))

        # =====================================================================
        # CATEGORY 19: AUTONOMOUS BRAIN COMPONENTS
        # =====================================================================
        try:
            from autonomous.brain import AutonomousBrain
            self.autonomous_brain = AutonomousBrain
            self.components['autonomous_brain'] = ComponentStatus('Autonomous Brain', True, True)
        except ImportError as e:
            self.autonomous_brain = None
            self.components['autonomous_brain'] = ComponentStatus('Autonomous Brain', False, error=str(e))

        try:
            from autonomous.research import ResearchEngine
            self.research_engine = ResearchEngine
            self.components['research_engine'] = ComponentStatus('Research Engine', True, True)
        except ImportError as e:
            self.research_engine = None
            self.components['research_engine'] = ComponentStatus('Research Engine', False, error=str(e))

        try:
            from autonomous.learning import LearningEngine
            self.learning_engine = LearningEngine
            self.components['learning_engine'] = ComponentStatus('Learning Engine', True, True)
        except ImportError as e:
            self.learning_engine = None
            self.components['learning_engine'] = ComponentStatus('Learning Engine', False, error=str(e))

        try:
            from autonomous.awareness import TimeAwareness, MarketCalendarAwareness, SeasonalAwareness
            self.time_awareness = TimeAwareness
            self.market_calendar = MarketCalendarAwareness
            self.seasonal_awareness = SeasonalAwareness
            self.components['awareness'] = ComponentStatus('Awareness System', True, True)
        except ImportError as e:
            self.time_awareness = None
            self.components['awareness'] = ComponentStatus('Awareness System', False, error=str(e))

        try:
            from autonomous.scheduler import AutonomousScheduler, get_scheduler
            self.task_scheduler = AutonomousScheduler
            self.components['task_scheduler'] = ComponentStatus('Task Scheduler', True, True)
        except ImportError as e:
            self.task_scheduler = None
            self.components['task_scheduler'] = ComponentStatus('Task Scheduler', False, error=str(e))

        try:
            from autonomous.maintenance import MaintenanceEngine
            self.maintenance_engine = MaintenanceEngine
            self.components['maintenance_engine'] = ComponentStatus('Maintenance Engine', True, True)
        except ImportError as e:
            self.maintenance_engine = None
            self.components['maintenance_engine'] = ComponentStatus('Maintenance Engine', False, error=str(e))

        try:
            from autonomous.integrity import IntegrityGuardian
            self.integrity_guardian = IntegrityGuardian
            self.components['integrity_guardian'] = ComponentStatus('Integrity Guardian', True, True)
        except ImportError as e:
            self.integrity_guardian = None
            self.components['integrity_guardian'] = ComponentStatus('Integrity Guardian', False, error=str(e))

        try:
            from autonomous.pattern_rhymes import PatternRhymesEngine
            self.pattern_rhymes = PatternRhymesEngine
            self.components['pattern_rhymes'] = ComponentStatus('Pattern Rhymes Engine', True, True)
        except ImportError as e:
            self.pattern_rhymes = None
            self.components['pattern_rhymes'] = ComponentStatus('Pattern Rhymes Engine', False, error=str(e))

        # =====================================================================
        # CATEGORY 20: BACKTEST COMPONENTS
        # =====================================================================
        try:
            from backtest.vectorized import VectorizedBacktester
            self.vectorized_backtester = VectorizedBacktester
            self.components['vectorized_backtester'] = ComponentStatus('Vectorized Backtester', True, True)
        except ImportError as e:
            self.vectorized_backtester = None
            self.components['vectorized_backtester'] = ComponentStatus('Vectorized Backtester', False, error=str(e))

        try:
            from backtest.walk_forward import generate_splits, run_walk_forward, WFSplit
            self.walk_forward = run_walk_forward
            self.components['walk_forward'] = ComponentStatus('Walk Forward Analyzer', True, True)
        except ImportError as e:
            self.walk_forward = None
            self.components['walk_forward'] = ComponentStatus('Walk Forward Analyzer', False, error=str(e))

        try:
            from backtest.monte_carlo import MonteCarloSimulator
            self.monte_carlo_sim = MonteCarloSimulator
            self.components['monte_carlo_sim'] = ComponentStatus('Monte Carlo Simulator', True, True)
        except ImportError as e:
            self.monte_carlo_sim = None
            self.components['monte_carlo_sim'] = ComponentStatus('Monte Carlo Simulator', False, error=str(e))

        try:
            from backtest.slippage import SlippageModel
            self.slippage_model = SlippageModel
            self.components['slippage_model'] = ComponentStatus('Slippage Model', True, True)
        except ImportError as e:
            self.slippage_model = None
            self.components['slippage_model'] = ComponentStatus('Slippage Model', False, error=str(e))

        try:
            from backtest.reproducibility import ExperimentTracker
            self.experiment_tracker = ExperimentTracker
            self.components['experiment_tracker'] = ComponentStatus('Experiment Tracker', True, True)
        except ImportError as e:
            self.experiment_tracker = None
            self.components['experiment_tracker'] = ComponentStatus('Experiment Tracker', False, error=str(e))

        # =====================================================================
        # CATEGORY 21: ALERTS & NOTIFICATIONS
        # =====================================================================
        try:
            from alerts.professional_alerts import ProfessionalAlerts
            self.professional_alerts = ProfessionalAlerts
            self.components['professional_alerts'] = ComponentStatus('Professional Alerts', True, True)
        except ImportError as e:
            self.professional_alerts = None
            self.components['professional_alerts'] = ComponentStatus('Professional Alerts', False, error=str(e))

        try:
            from alerts.telegram_alerter import TelegramAlerter
            self.telegram_alerter = TelegramAlerter
            self.components['telegram_alerter'] = ComponentStatus('Telegram Alerter', True, True)
        except ImportError as e:
            self.telegram_alerter = None
            self.components['telegram_alerter'] = ComponentStatus('Telegram Alerter', False, error=str(e))

        # =====================================================================
        # CATEGORY 22: AGENTS
        # =====================================================================
        try:
            from agents.orchestrator import AgentOrchestrator
            self.agent_orchestrator = AgentOrchestrator
            self.components['agent_orchestrator'] = ComponentStatus('Agent Orchestrator', True, True)
        except ImportError as e:
            self.agent_orchestrator = None
            self.components['agent_orchestrator'] = ComponentStatus('Agent Orchestrator', False, error=str(e))

        try:
            from agents.scout_agent import ScoutAgent
            self.scout_agent = ScoutAgent
            self.components['scout_agent'] = ComponentStatus('Scout Agent', True, True)
        except ImportError as e:
            self.scout_agent = None
            self.components['scout_agent'] = ComponentStatus('Scout Agent', False, error=str(e))

        try:
            from agents.risk_agent import RiskAgent
            self.risk_agent = RiskAgent
            self.components['risk_agent'] = ComponentStatus('Risk Agent', True, True)
        except ImportError as e:
            self.risk_agent = None
            self.components['risk_agent'] = ComponentStatus('Risk Agent', False, error=str(e))

        # =====================================================================
        # CATEGORY 23: COMPLIANCE & RULES
        # =====================================================================
        try:
            from compliance.rules_engine import RuleConfig, evaluate as rules_evaluate
            self.rules_engine = rules_evaluate
            self.components['rules_engine'] = ComponentStatus('Rules Engine', True, True)
        except ImportError as e:
            self.rules_engine = None
            self.components['rules_engine'] = ComponentStatus('Rules Engine', False, error=str(e))

        try:
            from compliance.prohibited_list import is_prohibited, prohibited_reasons, ProhibitedReason
            self.prohibited_assets = is_prohibited
            self.components['prohibited_assets'] = ComponentStatus('Prohibited Assets', True, True)
        except ImportError as e:
            self.prohibited_assets = None
            self.components['prohibited_assets'] = ComponentStatus('Prohibited Assets', False, error=str(e))

        # =====================================================================
        # CATEGORY 24: MARKET MOOD & SENTIMENT
        # =====================================================================
        try:
            from altdata.market_mood_analyzer import MarketMoodAnalyzer
            self.market_mood_analyzer = MarketMoodAnalyzer
            self.components['market_mood_analyzer'] = ComponentStatus('Market Mood Analyzer', True, True)
        except ImportError as e:
            self.market_mood_analyzer = None
            self.components['market_mood_analyzer'] = ComponentStatus('Market Mood Analyzer', False, error=str(e))

        # =====================================================================
        # CATEGORY 25: PORTFOLIO OPTIMIZATION
        # =====================================================================
        try:
            from portfolio.optimizer.mean_variance import MeanVarianceOptimizer
            self.mean_variance_optimizer = MeanVarianceOptimizer
            self.components['mean_variance_optimizer'] = ComponentStatus('Mean Variance Optimizer', True, True)
        except ImportError as e:
            self.mean_variance_optimizer = None
            self.components['mean_variance_optimizer'] = ComponentStatus('Mean Variance Optimizer', False, error=str(e))

        try:
            from portfolio.optimizer.risk_parity import RiskParityOptimizer
            self.risk_parity_optimizer = RiskParityOptimizer
            self.components['risk_parity_optimizer'] = ComponentStatus('Risk Parity Optimizer', True, True)
        except ImportError as e:
            self.risk_parity_optimizer = None
            self.components['risk_parity_optimizer'] = ComponentStatus('Risk Parity Optimizer', False, error=str(e))

        try:
            from portfolio.optimizer.rebalancer import PortfolioRebalancer, get_rebalancer
            self.rebalancer = PortfolioRebalancer
            self.components['rebalancer'] = ComponentStatus('Rebalancer', True, True)
        except ImportError as e:
            self.rebalancer = None
            self.components['rebalancer'] = ComponentStatus('Rebalancer', False, error=str(e))

        try:
            from portfolio.state_manager import StateManager
            self.portfolio_state_manager = StateManager
            self.components['portfolio_state_manager'] = ComponentStatus('Portfolio State Manager', True, True)
        except ImportError as e:
            self.portfolio_state_manager = None
            self.components['portfolio_state_manager'] = ComponentStatus('Portfolio State Manager', False, error=str(e))

        # =====================================================================
        # CATEGORY 26: ANALYTICS - P&L ATTRIBUTION
        # =====================================================================
        try:
            from analytics.attribution.daily_pnl import DailyPnLTracker
            self.daily_pnl_tracker = DailyPnLTracker
            self.components['daily_pnl_tracker'] = ComponentStatus('Daily P&L Tracker', True, True)
        except ImportError as e:
            self.daily_pnl_tracker = None
            self.components['daily_pnl_tracker'] = ComponentStatus('Daily P&L Tracker', False, error=str(e))

        try:
            from analytics.attribution.strategy_attribution import StrategyAttributor
            self.strategy_attributor = StrategyAttributor
            self.components['strategy_attributor'] = ComponentStatus('Strategy Attributor', True, True)
        except ImportError as e:
            self.strategy_attributor = None
            self.components['strategy_attributor'] = ComponentStatus('Strategy Attributor', False, error=str(e))

        try:
            from analytics.duckdb_engine import DuckDBEngine
            self.duckdb_engine = DuckDBEngine
            self.components['duckdb_engine'] = ComponentStatus('DuckDB Engine', True, True)
        except ImportError as e:
            self.duckdb_engine = None
            self.components['duckdb_engine'] = ComponentStatus('DuckDB Engine', False, error=str(e))

        # =====================================================================
        # CATEGORY 27: RL AGENT
        # =====================================================================
        try:
            from ml.alpha_discovery.rl_agent.trading_env import TradingEnv
            self.trading_env = TradingEnv
            self.components['trading_env'] = ComponentStatus('Trading Environment (RL)', True, True)
        except ImportError as e:
            self.trading_env = None
            self.components['trading_env'] = ComponentStatus('Trading Environment (RL)', False, error=str(e))

        try:
            from ml.alpha_discovery.rl_agent.agent import RLTradingAgent
            self.rl_agent = RLTradingAgent
            self.components['rl_agent'] = ComponentStatus('RL Trading Agent', True, True)
        except ImportError as e:
            self.rl_agent = None
            self.components['rl_agent'] = ComponentStatus('RL Trading Agent', False, error=str(e))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of component availability."""
        total = len(self.components)
        available = sum(1 for c in self.components.values() if c.available)
        wired = sum(1 for c in self.components.values() if c.wired)

        return {
            'total_components': total,
            'available': available,
            'wired': wired,
            'utilization_pct': round(wired / total * 100, 1) if total > 0 else 0,
            'components': {k: v.to_dict() for k, v in self.components.items()},
        }

    def print_status(self):
        """Print component status to console."""
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print(f"COMPONENT REGISTRY STATUS")
        print(f"{'='*60}")
        print(f"Total Components: {summary['total_components']}")
        print(f"Available: {summary['available']}")
        print(f"Wired: {summary['wired']}")
        print(f"Utilization: {summary['utilization_pct']}%")
        print(f"{'='*60}")

        # Group by availability
        available_list = []
        unavailable_list = []

        for name, status in self.components.items():
            if status.available:
                available_list.append(name)
            else:
                unavailable_list.append((name, status.error))

        print(f"\nAVAILABLE ({len(available_list)}):")
        for name in sorted(available_list):
            print(f"  [OK] {name}")

        if unavailable_list:
            print(f"\nUNAVAILABLE ({len(unavailable_list)}):")
            for name, error in sorted(unavailable_list):
                short_error = error[:50] + "..." if len(error) > 50 else error
                print(f"  [--] {name}: {short_error}")


# =============================================================================
# ENRICHED SIGNAL DATA STRUCTURES
# =============================================================================

@dataclass
class EnrichedSignal:
    """A fully enriched signal with all AI/ML/Data analysis."""

    # Core signal data
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy: str
    timestamp: str
    asset_class: str = "EQUITY"

    # Historical Pattern Analysis
    streak_length: int = 0
    streak_samples: int = 0
    streak_win_rate: float = 0.0
    streak_avg_bounce: float = 0.0
    pattern_grade: str = "N/A"
    qualifies_auto_pass: bool = False

    # ML Confidence Scores (multiple models)
    ml_meta_conf: float = 0.5
    lstm_direction: float = 0.5
    lstm_magnitude: float = 0.0
    lstm_success: float = 0.5
    ensemble_conf: float = 0.5
    ensemble_agreement: float = 0.0
    markov_pi_up: float = 0.5
    markov_p_up_today: float = 0.5
    markov_agrees: bool = False

    # Conviction & Quality Scores
    conviction_score: int = 0
    conviction_tier: str = "N/A"
    quality_score: int = 0
    quality_tier: str = "N/A"
    adjudication_score: float = 0.0

    # Cognitive Brain Assessment
    cognitive_approved: bool = False
    cognitive_confidence: float = 0.0
    cognitive_reasoning: str = ""
    cognitive_concerns: str = ""
    knowledge_boundary_safe: bool = True

    # Alt Data Signals
    news_sentiment: float = 0.0
    news_article_count: int = 0
    insider_signal: str = "neutral"
    insider_value: float = 0.0
    congress_signal: str = "neutral"
    congress_buys: int = 0
    congress_sells: int = 0
    options_flow_signal: str = "neutral"
    options_unusual_activity: bool = False

    # Market Context
    regime: str = "unknown"
    regime_confidence: float = 0.0
    vix_level: float = 0.0
    sector_relative_strength: float = 0.0

    # Risk Metrics
    kelly_optimal_pct: float = 0.02
    var_contribution: float = 0.0
    correlation_with_portfolio: float = 0.0
    expected_move_weekly: float = 0.0

    # Support/Resistance
    nearest_support: float = 0.0
    nearest_resistance: float = 0.0
    distance_to_support_pct: float = 0.0
    distance_to_resistance_pct: float = 0.0

    # Final Confidence (weighted combination)
    final_conf_score: float = 0.5
    final_rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradeThesis:
    """Comprehensive trade thesis for TOP 2 trades."""

    signal: EnrichedSignal

    # Evidence Categories
    price_action_evidence: Dict[str, Any] = field(default_factory=dict)
    technical_evidence: Dict[str, Any] = field(default_factory=dict)
    fundamental_evidence: Dict[str, Any] = field(default_factory=dict)
    ml_confidence_breakdown: Dict[str, Any] = field(default_factory=dict)
    risk_analysis: Dict[str, Any] = field(default_factory=dict)

    # Qualitative Reasoning
    bull_case: str = ""
    bear_case: str = ""
    risks: List[str] = field(default_factory=list)

    # Claude AI Reasoning (human-like)
    claude_analysis: str = ""
    claude_recommendation: str = ""
    claude_confidence: float = 0.0

    # Final Verdict
    verdict: str = ""
    conviction_level: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['signal'] = self.signal.to_dict()
        return d

    def to_markdown(self) -> str:
        """Generate markdown report for this trade thesis."""
        s = self.signal

        md = f"""
# TRADE THESIS: {s.symbol} ({s.asset_class})

**Generated**: {datetime.now().isoformat()}
**Strategy**: {s.strategy}
**Side**: {s.side.upper()}

---

## ENTRY DETAILS

| Field | Value |
|-------|-------|
| Entry Price | ${s.entry_price:.2f} |
| Stop Loss | ${s.stop_loss:.2f} |
| Take Profit | ${s.take_profit:.2f} |
| Risk % | {abs(s.entry_price - s.stop_loss) / s.entry_price * 100:.2f}% |

---

## CONFIDENCE BREAKDOWN

| Model | Confidence |
|-------|------------|
| ML Meta (XGB/LGBM) | {s.ml_meta_conf:.2%} |
| LSTM Direction | {s.lstm_direction:.2%} |
| LSTM Success Prob | {s.lstm_success:.2%} |
| Ensemble | {s.ensemble_conf:.2%} |
| Ensemble Agreement | {s.ensemble_agreement:.2%} |
| Markov pi(Up) | {s.markov_pi_up:.2%} |
| Conviction Score | {s.conviction_score}/100 ({s.conviction_tier}) |
| Quality Score | {s.quality_score}/100 ({s.quality_tier}) |
| **Final Confidence** | **{s.final_conf_score:.2%}** |

---

## HISTORICAL PATTERN

| Metric | Value |
|--------|-------|
| Consecutive Days | {s.streak_length} |
| Historical Samples | {s.streak_samples} |
| Win Rate | {s.streak_win_rate:.0%} |
| Avg Bounce | {s.streak_avg_bounce:+.1%} |
| Pattern Grade | {s.pattern_grade} |
| Auto-Pass Eligible | {'YES' if s.qualifies_auto_pass else 'NO'} |

---

## ALT DATA SIGNALS

| Source | Signal | Details |
|--------|--------|---------|
| News Sentiment | {s.news_sentiment:+.2f} | {s.news_article_count} articles |
| Insider Activity | {s.insider_signal.upper()} | ${s.insider_value:,.0f} |
| Congress | {s.congress_signal.upper()} | {s.congress_buys} buys / {s.congress_sells} sells |
| Options Flow | {s.options_flow_signal.upper()} | {'Unusual Activity' if s.options_unusual_activity else 'Normal'} |

---

## MARKET CONTEXT

| Factor | Value |
|--------|-------|
| Market Regime | {s.regime} ({s.regime_confidence:.0%}) |
| VIX Level | {s.vix_level:.1f} |
| Sector Relative Strength | {s.sector_relative_strength:+.1%} |

---

## RISK ANALYSIS

| Metric | Value |
|--------|-------|
| Kelly Optimal Size | {s.kelly_optimal_pct:.1%} |
| VaR Contribution | ${s.var_contribution:,.0f} |
| Correlation w/ Portfolio | {s.correlation_with_portfolio:.2f} |
| Expected Move (Weekly) | {s.expected_move_weekly:.1%} |

---

## SUPPORT/RESISTANCE

| Level | Price | Distance |
|-------|-------|----------|
| Nearest Support | ${s.nearest_support:.2f} | {s.distance_to_support_pct:.1%} below |
| Nearest Resistance | ${s.nearest_resistance:.2f} | {s.distance_to_resistance_pct:.1%} above |

---

## COGNITIVE BRAIN ASSESSMENT

**Approved**: {'YES' if s.cognitive_approved else 'NO'}
**Confidence**: {s.cognitive_confidence:.2%}
**Knowledge Boundary Safe**: {'YES' if s.knowledge_boundary_safe else 'NO'}

**Reasoning**:
{s.cognitive_reasoning}

**Concerns**:
{s.cognitive_concerns if s.cognitive_concerns else 'None identified'}

---

## BULL CASE

{self.bull_case}

---

## BEAR CASE

{self.bear_case}

---

## WHAT COULD GO WRONG

{chr(10).join(f'{i+1}. {risk}' for i, risk in enumerate(self.risks))}

---

## CLAUDE AI ANALYSIS

{self.claude_analysis}

**Recommendation**: {self.claude_recommendation}
**Claude Confidence**: {self.claude_confidence:.0%}

---

## FINAL VERDICT

**{self.verdict}**

Conviction Level: **{self.conviction_level}**
"""
        return md


# =============================================================================
# UNIFIED ENRICHMENT PIPELINE
# =============================================================================

class UnifiedSignalEnrichmentPipeline:
    """
    The main orchestrator that wires ALL components together.

    Takes raw signals from DualStrategyScanner and enriches them through
    25+ stages to produce fully-analyzed TOP 2 trades.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.registry = ComponentRegistry()
        self.stages_executed: List[str] = []
        self.stage_results: Dict[str, Any] = {}

        if verbose:
            self.registry.print_status()

    def log(self, message: str):
        """Log a message if verbose."""
        if self.verbose:
            print(f"  [PIPELINE] {message}")

    def enrich_signals(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame] = None,
        vix_data: Optional[pd.DataFrame] = None,
        current_positions: List[dict] = None,
        account_equity: float = 50000.0,
    ) -> Tuple[List[EnrichedSignal], List[TradeThesis]]:
        """
        Main enrichment pipeline - runs ALL stages.

        Args:
            signals: Raw signals from DualStrategyScanner
            price_data: Historical OHLCV data
            spy_data: SPY data for regime/relative strength
            vix_data: VIX data for volatility context
            current_positions: Existing portfolio positions
            account_equity: Account equity for sizing

        Returns:
            Tuple of (enriched_signals, top2_theses)
        """
        if signals.empty:
            self.log("No signals to enrich")
            return [], []

        self.log(f"Starting enrichment pipeline with {len(signals)} signals...")

        enriched = []

        def safe_float(val, default=0.0):
            """Safely convert to float, handling None and NaN."""
            if val is None:
                return default
            if isinstance(val, float) and pd.isna(val):
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        def safe_str(val, default=''):
            """Safely convert to string, handling None."""
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return default
            return str(val)

        # Convert DataFrame rows to EnrichedSignal objects
        for idx, row in signals.iterrows():
            try:
                signal = EnrichedSignal(
                    symbol=safe_str(row.get('symbol', ''), ''),
                    side=safe_str(row.get('side', 'long'), 'long'),
                    entry_price=safe_float(row.get('entry_price'), 0),
                    stop_loss=safe_float(row.get('stop_loss'), 0),
                    take_profit=safe_float(row.get('take_profit'), 0),
                    strategy=safe_str(row.get('strategy', ''), ''),
                    timestamp=safe_str(row.get('timestamp'), datetime.now().isoformat()),
                    asset_class=safe_str(row.get('asset_class', 'EQUITY'), 'EQUITY'),
                )

                # Copy over any existing enrichment from signals DataFrame
                if 'conf_score' in row and row['conf_score'] is not None:
                    signal.ml_meta_conf = safe_float(row['conf_score'], 0.5)
                    signal.final_conf_score = safe_float(row['conf_score'], 0.5)

                enriched.append(signal)

            except Exception as e:
                logger.warning(f"Failed to create EnrichedSignal for row {idx}: {e}")
                continue

        if not enriched:
            self.log("No valid signals after conversion")
            return [], []

        # =====================================================================
        # RUN ALL ENRICHMENT STAGES
        # =====================================================================

        # Stage 1: Historical Patterns
        enriched = self._stage_historical_patterns(enriched, price_data)

        # Stage 2: Market Regime
        enriched = self._stage_market_regime(enriched, spy_data, vix_data)

        # Stage 3: Markov Chain
        enriched = self._stage_markov_chain(enriched, price_data)

        # Stage 4: LSTM Confidence
        enriched = self._stage_lstm_confidence(enriched, price_data)

        # Stage 5: Ensemble Prediction
        enriched = self._stage_ensemble_prediction(enriched, price_data)

        # Stage 6: Sentiment Analysis
        enriched = self._stage_sentiment_analysis(enriched)

        # Stage 7: Alt Data (News, Insider, Congress, Options)
        enriched = self._stage_alt_data(enriched)

        # Stage 8: Conviction Scoring
        enriched = self._stage_conviction_scoring(enriched, price_data)

        # Stage 9: Support/Resistance
        enriched = self._stage_support_resistance(enriched, price_data)

        # Stage 10: Expected Move
        enriched = self._stage_expected_move(enriched, price_data)

        # Stage 11: Circuit Breaker Check
        enriched = self._stage_circuit_breakers(enriched)

        # Stage 12: Knowledge Boundary
        enriched = self._stage_knowledge_boundary(enriched)

        # Stage 13: Kelly Sizing
        enriched = self._stage_kelly_sizing(enriched, account_equity)

        # Stage 14: VaR Calculation
        enriched = self._stage_var_calculation(enriched, price_data, current_positions)

        # Stage 15: Correlation Check
        enriched = self._stage_correlation_check(enriched, price_data, current_positions)

        # Stage 16: Cognitive Brain
        enriched = self._stage_cognitive_brain(enriched, price_data, spy_data)

        # Stage 17: Calculate Final Confidence
        enriched = self._stage_final_confidence(enriched)

        # Stage 18: Rank Signals
        enriched = self._stage_rank_signals(enriched)

        # Get TOP 5 and TOP 2
        top5 = enriched[:5]
        top2 = enriched[:2]

        self.log(f"Pipeline complete. TOP 5: {[s.symbol for s in top5]}")
        self.log(f"TOP 2 for trading: {[s.symbol for s in top2]}")

        # =====================================================================
        # GENERATE COMPREHENSIVE THESIS FOR TOP 2
        # =====================================================================
        theses = []
        for signal in top2:
            thesis = self._generate_trade_thesis(signal, price_data, spy_data)
            theses.append(thesis)

        return enriched, theses

    # =========================================================================
    # INDIVIDUAL STAGES
    # =========================================================================

    def _stage_historical_patterns(
        self,
        signals: List[EnrichedSignal],
        price_data: pd.DataFrame,
    ) -> List[EnrichedSignal]:
        """Stage 1: Enrich with historical pattern analysis."""
        stage_name = "Historical Patterns"
        self.log(f"Stage 1: {stage_name}")

        if self.registry.historical_patterns is None:
            self.log(f"  [SKIP] {stage_name} not available")
            return signals

        analyzer = self.registry.historical_patterns

        for signal in signals:
            try:
                # Get symbol's price data
                sym_data = price_data[price_data['symbol'] == signal.symbol].copy()
                if sym_data.empty:
                    continue

                pattern = analyzer.analyze_consecutive_days(sym_data, signal.symbol)

                signal.streak_length = pattern.current_streak
                signal.streak_samples = pattern.sample_size
                signal.streak_win_rate = pattern.historical_reversal_rate
                signal.streak_avg_bounce = pattern.avg_reversal_magnitude
                signal.pattern_grade = self.registry.get_pattern_grade(pattern)
                signal.qualifies_auto_pass = self.registry.qualifies_for_auto_pass(pattern)

            except Exception as e:
                logger.debug(f"Historical pattern failed for {signal.symbol}: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_market_regime(
        self,
        signals: List[EnrichedSignal],
        spy_data: Optional[pd.DataFrame],
        vix_data: Optional[pd.DataFrame],
    ) -> List[EnrichedSignal]:
        """Stage 2: Determine market regime."""
        stage_name = "Market Regime"
        self.log(f"Stage 2: {stage_name}")

        # HMM Regime Detection
        if self.registry.regime_detector is not None and spy_data is not None:
            try:
                detector = self.registry.regime_detector(use_hmm=True)
                regime_result = detector.detect_regime(spy_data, vix_data)

                for signal in signals:
                    signal.regime = regime_result.regime.value if hasattr(regime_result, 'regime') else str(regime_result)
                    signal.regime_confidence = getattr(regime_result, 'confidence', 0.5)

            except Exception as e:
                logger.debug(f"HMM regime detection failed: {e}")

        # VIX Level
        if self.registry.vix_monitor is not None:
            try:
                vix_monitor = self.registry.vix_monitor()
                vix_level = vix_monitor.get_current_vix()
                for signal in signals:
                    signal.vix_level = vix_level
            except Exception as e:
                logger.debug(f"VIX monitor failed: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_markov_chain(
        self,
        signals: List[EnrichedSignal],
        price_data: pd.DataFrame,
    ) -> List[EnrichedSignal]:
        """Stage 3: Markov chain direction prediction."""
        stage_name = "Markov Chain"
        self.log(f"Stage 3: {stage_name}")

        if self.registry.markov_predictor is None:
            self.log(f"  [SKIP] {stage_name} not available")
            return signals

        try:
            predictor = self.registry.markov_predictor()

            for signal in signals:
                sym_data = price_data[price_data['symbol'] == signal.symbol].copy()
                if len(sym_data) < 50:
                    continue

                try:
                    result = predictor.predict(sym_data)
                    signal.markov_pi_up = result.get('pi_up', 0.5)
                    signal.markov_p_up_today = result.get('p_up_today', 0.5)
                    signal.markov_agrees = result.get('agrees', False)
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Markov chain failed: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_lstm_confidence(
        self,
        signals: List[EnrichedSignal],
        price_data: pd.DataFrame,
    ) -> List[EnrichedSignal]:
        """Stage 4: LSTM multi-output confidence."""
        stage_name = "LSTM Confidence"
        self.log(f"Stage 4: {stage_name}")

        if self.registry.lstm_model is None:
            self.log(f"  [SKIP] {stage_name} not available")
            return signals

        try:
            # Try to load pre-trained LSTM model
            model_path = ROOT / "models" / "lstm_confidence.pt"
            if model_path.exists():
                model = self.registry.lstm_model.load(str(model_path))

                for signal in signals:
                    sym_data = price_data[price_data['symbol'] == signal.symbol].copy()
                    if len(sym_data) < 60:
                        continue

                    try:
                        pred = model.predict(sym_data)
                        signal.lstm_direction = pred.get('direction', 0.5)
                        signal.lstm_magnitude = pred.get('magnitude', 0.0)
                        signal.lstm_success = pred.get('success_probability', 0.5)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"LSTM confidence failed: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_ensemble_prediction(
        self,
        signals: List[EnrichedSignal],
        price_data: pd.DataFrame,
    ) -> List[EnrichedSignal]:
        """Stage 5: Multi-model ensemble prediction."""
        stage_name = "Ensemble Prediction"
        self.log(f"Stage 5: {stage_name}")

        if self.registry.ensemble_predictor is None:
            self.log(f"  [SKIP] {stage_name} not available")
            return signals

        try:
            ensemble = self.registry.ensemble_predictor()

            for signal in signals:
                sym_data = price_data[price_data['symbol'] == signal.symbol].copy()
                if len(sym_data) < 50:
                    continue

                try:
                    pred = ensemble.predict(sym_data)
                    signal.ensemble_conf = pred.get('confidence', 0.5)
                    signal.ensemble_agreement = pred.get('agreement_score', 0.0)
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Ensemble prediction failed: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_sentiment_analysis(
        self,
        signals: List[EnrichedSignal],
    ) -> List[EnrichedSignal]:
        """Stage 6: Sentiment analysis."""
        stage_name = "Sentiment Analysis"
        self.log(f"Stage 6: {stage_name}")

        if self.registry.load_sentiment is None:
            self.log(f"  [SKIP] {stage_name} not available")
            return signals

        try:
            sentiment_cache = self.registry.load_sentiment()

            for signal in signals:
                if signal.symbol in sentiment_cache:
                    sent_data = sentiment_cache[signal.symbol]
                    signal.news_sentiment = sent_data.get('sentiment_score', 0.0)
                    signal.news_article_count = sent_data.get('article_count', 0)

        except Exception as e:
            logger.debug(f"Sentiment analysis failed: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_alt_data(
        self,
        signals: List[EnrichedSignal],
    ) -> List[EnrichedSignal]:
        """Stage 7: Alternative data (insider, congress, options flow)."""
        stage_name = "Alt Data"
        self.log(f"Stage 7: {stage_name}")

        # Insider Activity
        if self.registry.insider_tracker is not None:
            try:
                tracker = self.registry.insider_tracker()
                for signal in signals:
                    try:
                        insider_data = tracker.get_recent_activity(signal.symbol)
                        signal.insider_signal = insider_data.get('signal', 'neutral')
                        signal.insider_value = insider_data.get('total_value', 0.0)
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Insider tracker failed: {e}")

        # Congressional Trades
        if self.registry.congress_tracker is not None:
            try:
                tracker = self.registry.congress_tracker()
                for signal in signals:
                    try:
                        congress_data = tracker.get_recent_activity(signal.symbol)
                        signal.congress_signal = congress_data.get('signal', 'neutral')
                        signal.congress_buys = congress_data.get('buys', 0)
                        signal.congress_sells = congress_data.get('sells', 0)
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Congress tracker failed: {e}")

        # Options Flow
        if self.registry.options_flow is not None:
            try:
                analyzer = self.registry.options_flow()
                for signal in signals:
                    try:
                        flow_data = analyzer.get_unusual_activity(signal.symbol)
                        signal.options_flow_signal = flow_data.get('signal', 'neutral')
                        signal.options_unusual_activity = flow_data.get('unusual', False)
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Options flow failed: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_conviction_scoring(
        self,
        signals: List[EnrichedSignal],
        price_data: pd.DataFrame,
    ) -> List[EnrichedSignal]:
        """Stage 8: Conviction scoring (6-factor rule-based)."""
        stage_name = "Conviction Scoring"
        self.log(f"Stage 8: {stage_name}")

        if self.registry.conviction_scorer is None:
            self.log(f"  [SKIP] {stage_name} not available")
            return signals

        try:
            scorer = self.registry.conviction_scorer

            for signal in signals:
                sym_data = price_data[price_data['symbol'] == signal.symbol].copy()
                if sym_data.empty:
                    continue

                try:
                    result = scorer.score(signal.to_dict(), sym_data)
                    signal.conviction_score = result.score
                    signal.conviction_tier = result.tier.value if hasattr(result.tier, 'value') else str(result.tier)
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Conviction scoring failed: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_support_resistance(
        self,
        signals: List[EnrichedSignal],
        price_data: pd.DataFrame,
    ) -> List[EnrichedSignal]:
        """Stage 9: Support/Resistance analysis."""
        stage_name = "Support/Resistance"
        self.log(f"Stage 9: {stage_name}")

        if self.registry.historical_patterns is None:
            return signals

        try:
            analyzer = self.registry.historical_patterns

            for signal in signals:
                sym_data = price_data[price_data['symbol'] == signal.symbol].copy()
                if sym_data.empty:
                    continue

                try:
                    levels = analyzer.analyze_support_resistance(sym_data, signal.symbol)

                    # Find nearest support and resistance
                    supports = [l for l in levels if l.level_type == 'support']
                    resistances = [l for l in levels if l.level_type == 'resistance']

                    if supports:
                        nearest_support = max(supports, key=lambda x: x.price)
                        signal.nearest_support = nearest_support.price
                        signal.distance_to_support_pct = nearest_support.distance_pct

                    if resistances:
                        nearest_resistance = min(resistances, key=lambda x: x.price)
                        signal.nearest_resistance = nearest_resistance.price
                        signal.distance_to_resistance_pct = nearest_resistance.distance_pct

                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"S/R analysis failed: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_expected_move(
        self,
        signals: List[EnrichedSignal],
        price_data: pd.DataFrame,
    ) -> List[EnrichedSignal]:
        """Stage 10: Expected move calculation."""
        stage_name = "Expected Move"
        self.log(f"Stage 10: {stage_name}")

        if self.registry.expected_move_calc is None:
            self.log(f"  [SKIP] {stage_name} not available")
            return signals

        try:
            calc = self.registry.expected_move_calc

            for signal in signals:
                sym_data = price_data[price_data['symbol'] == signal.symbol].copy()
                if len(sym_data) < 30:
                    continue

                try:
                    em = calc.calculate_weekly_em(sym_data)
                    signal.expected_move_weekly = em.get('em_pct', 0.0)
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Expected move calculation failed: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_circuit_breakers(
        self,
        signals: List[EnrichedSignal],
    ) -> List[EnrichedSignal]:
        """Stage 11: Circuit breaker check."""
        stage_name = "Circuit Breakers"
        self.log(f"Stage 11: {stage_name}")

        if self.registry.circuit_breakers is None:
            self.log(f"  [SKIP] {stage_name} not available")
            return signals

        try:
            breakers = self.registry.circuit_breakers()
            status = breakers.check_all()

            if not status.can_trade:
                self.log(f"  [ALERT] Circuit breaker triggered: {status.triggered_breakers}")
                # Don't remove signals, but flag them
                for signal in signals:
                    signal.cognitive_concerns = f"CIRCUIT BREAKER: {status.triggered_breakers}"

        except Exception as e:
            logger.debug(f"Circuit breaker check failed: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_knowledge_boundary(
        self,
        signals: List[EnrichedSignal],
    ) -> List[EnrichedSignal]:
        """Stage 12: Knowledge boundary check."""
        stage_name = "Knowledge Boundary"
        self.log(f"Stage 12: {stage_name}")

        if self.registry.knowledge_boundary is None:
            self.log(f"  [SKIP] {stage_name} not available")
            return signals

        try:
            kb = self.registry.knowledge_boundary()

            for signal in signals:
                try:
                    result = kb.check_signal(signal.to_dict())
                    signal.knowledge_boundary_safe = result.is_safe
                    if not result.is_safe:
                        signal.cognitive_concerns += f" KNOWLEDGE BOUNDARY: {result.reason}"
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Knowledge boundary check failed: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_kelly_sizing(
        self,
        signals: List[EnrichedSignal],
        account_equity: float,
    ) -> List[EnrichedSignal]:
        """Stage 13: Kelly position sizing."""
        stage_name = "Kelly Sizing"
        self.log(f"Stage 13: {stage_name}")

        if self.registry.kelly_sizer is None:
            self.log(f"  [SKIP] {stage_name} not available")
            return signals

        try:
            sizer = self.registry.kelly_sizer()

            for signal in signals:
                try:
                    # Estimate win rate from historical pattern or use ML confidence
                    win_rate = signal.streak_win_rate if signal.streak_win_rate > 0 else signal.ml_meta_conf

                    kelly_pct = sizer.calculate_optimal_size(
                        win_rate=win_rate,
                        win_loss_ratio=1.5,  # Default assumption
                        account_equity=account_equity,
                    )
                    signal.kelly_optimal_pct = min(kelly_pct, 0.10)  # Cap at 10%
                except Exception:
                    signal.kelly_optimal_pct = 0.02  # Default 2%

        except Exception as e:
            logger.debug(f"Kelly sizing failed: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_var_calculation(
        self,
        signals: List[EnrichedSignal],
        price_data: pd.DataFrame,
        current_positions: List[dict],
    ) -> List[EnrichedSignal]:
        """Stage 14: VaR calculation."""
        stage_name = "VaR Calculation"
        self.log(f"Stage 14: {stage_name}")

        if self.registry.monte_carlo_var is None:
            self.log(f"  [SKIP] {stage_name} not available")
            return signals

        try:
            var_calc = self.registry.monte_carlo_var()

            for signal in signals:
                sym_data = price_data[price_data['symbol'] == signal.symbol].copy()
                if len(sym_data) < 30:
                    continue

                try:
                    var_result = var_calc.calculate_marginal_var(
                        sym_data,
                        position_value=signal.entry_price * 100,  # Assume 100 shares
                    )
                    signal.var_contribution = var_result.get('var_95', 0.0)
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"VaR calculation failed: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_correlation_check(
        self,
        signals: List[EnrichedSignal],
        price_data: pd.DataFrame,
        current_positions: List[dict],
    ) -> List[EnrichedSignal]:
        """Stage 15: Correlation with existing portfolio."""
        stage_name = "Correlation Check"
        self.log(f"Stage 15: {stage_name}")

        if self.registry.correlation_limits is None or not current_positions:
            return signals

        try:
            corr_checker = self.registry.correlation_limits()

            for signal in signals:
                try:
                    result = corr_checker.check_correlation(
                        signal.symbol,
                        [p['symbol'] for p in current_positions],
                        price_data,
                    )
                    signal.correlation_with_portfolio = result.get('max_correlation', 0.0)
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Correlation check failed: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_cognitive_brain(
        self,
        signals: List[EnrichedSignal],
        price_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame],
    ) -> List[EnrichedSignal]:
        """Stage 16: Cognitive brain deliberation."""
        stage_name = "Cognitive Brain"
        self.log(f"Stage 16: {stage_name}")

        if self.registry.signal_processor is None:
            self.log(f"  [SKIP] {stage_name} not available")
            return signals

        try:
            processor = self.registry.signal_processor()

            for signal in signals:
                try:
                    # Convert to format expected by processor
                    signal_dict = signal.to_dict()

                    result = processor.process_signal(signal_dict, price_data, spy_data)

                    signal.cognitive_approved = result.get('approved', False)
                    signal.cognitive_confidence = result.get('confidence', 0.0)
                    signal.cognitive_reasoning = result.get('reasoning', '')
                    if 'concerns' in result:
                        signal.cognitive_concerns += ' ' + result['concerns']

                except Exception as e:
                    logger.debug(f"Cognitive processing failed for {signal.symbol}: {e}")

        except Exception as e:
            logger.debug(f"Cognitive brain failed: {e}")

        self.stages_executed.append(stage_name)
        return signals

    def _stage_final_confidence(
        self,
        signals: List[EnrichedSignal],
    ) -> List[EnrichedSignal]:
        """Stage 17: Calculate final weighted confidence score."""
        stage_name = "Final Confidence"
        self.log(f"Stage 17: {stage_name}")

        for signal in signals:
            # Weighted combination of all confidence sources
            # Weights sum to 1.0
            weights = {
                'ml_meta': 0.25,
                'lstm': 0.10,
                'ensemble': 0.15,
                'markov': 0.10,
                'conviction': 0.15,
                'historical': 0.15,
                'cognitive': 0.10,
            }

            # Normalize scores to 0-1 range
            scores = {
                'ml_meta': signal.ml_meta_conf,
                'lstm': signal.lstm_success,
                'ensemble': signal.ensemble_conf,
                'markov': signal.markov_pi_up,
                'conviction': signal.conviction_score / 100.0,
                'historical': signal.streak_win_rate,
                'cognitive': signal.cognitive_confidence,
            }

            # Calculate weighted sum
            final_conf = sum(weights[k] * scores[k] for k in weights)

            # Boost for auto-pass patterns
            if signal.qualifies_auto_pass:
                final_conf = min(0.95, final_conf + 0.10)

            # Penalty for circuit breaker concerns
            if signal.cognitive_concerns and 'CIRCUIT BREAKER' in signal.cognitive_concerns:
                final_conf *= 0.5

            signal.final_conf_score = round(final_conf, 4)

        self.stages_executed.append(stage_name)
        return signals

    def _stage_rank_signals(
        self,
        signals: List[EnrichedSignal],
    ) -> List[EnrichedSignal]:
        """Stage 18: Rank signals by final confidence."""
        stage_name = "Rank Signals"
        self.log(f"Stage 18: {stage_name}")

        # Sort by final confidence (descending)
        signals.sort(key=lambda x: x.final_conf_score, reverse=True)

        # Assign ranks
        for i, signal in enumerate(signals):
            signal.final_rank = i + 1

        self.stages_executed.append(stage_name)
        return signals

    def _generate_trade_thesis(
        self,
        signal: EnrichedSignal,
        price_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame],
    ) -> TradeThesis:
        """Generate comprehensive trade thesis for TOP 2 signals."""
        self.log(f"Generating thesis for {signal.symbol}...")

        thesis = TradeThesis(signal=signal)

        # Price Action Evidence
        thesis.price_action_evidence = {
            'current_price': signal.entry_price,
            'streak_length': signal.streak_length,
            'streak_samples': signal.streak_samples,
            'streak_win_rate': f"{signal.streak_win_rate:.0%}",
            'avg_bounce': f"{signal.streak_avg_bounce:+.1%}",
            'expected_move_weekly': f"{signal.expected_move_weekly:.1%}",
        }

        # Technical Evidence
        thesis.technical_evidence = {
            'strategy': signal.strategy,
            'side': signal.side,
            'entry': signal.entry_price,
            'stop': signal.stop_loss,
            'target': signal.take_profit,
            'nearest_support': signal.nearest_support,
            'nearest_resistance': signal.nearest_resistance,
        }

        # Fundamental Evidence
        thesis.fundamental_evidence = {
            'news_sentiment': signal.news_sentiment,
            'news_articles': signal.news_article_count,
            'insider_signal': signal.insider_signal,
            'insider_value': signal.insider_value,
            'congress_signal': signal.congress_signal,
            'congress_buys': signal.congress_buys,
            'congress_sells': signal.congress_sells,
            'options_flow': signal.options_flow_signal,
        }

        # ML Confidence Breakdown
        thesis.ml_confidence_breakdown = {
            'ml_meta': signal.ml_meta_conf,
            'lstm_direction': signal.lstm_direction,
            'lstm_magnitude': signal.lstm_magnitude,
            'lstm_success': signal.lstm_success,
            'ensemble_conf': signal.ensemble_conf,
            'ensemble_agreement': signal.ensemble_agreement,
            'markov_pi_up': signal.markov_pi_up,
            'markov_agrees': signal.markov_agrees,
            'conviction_score': signal.conviction_score,
            'cognitive_approved': signal.cognitive_approved,
            'cognitive_confidence': signal.cognitive_confidence,
            'final_confidence': signal.final_conf_score,
        }

        # Risk Analysis
        risk_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
        reward_pct = abs(signal.take_profit - signal.entry_price) / signal.entry_price if signal.take_profit > 0 else risk_pct * 2
        rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0

        thesis.risk_analysis = {
            'entry': signal.entry_price,
            'stop': signal.stop_loss,
            'target': signal.take_profit,
            'risk_pct': f"{risk_pct:.2%}",
            'reward_pct': f"{reward_pct:.2%}",
            'risk_reward_ratio': f"{rr_ratio:.2f}:1",
            'kelly_optimal': f"{signal.kelly_optimal_pct:.1%}",
            'var_contribution': f"${signal.var_contribution:,.0f}",
            'correlation': signal.correlation_with_portfolio,
        }

        # Bull Case
        bull_points = []
        if signal.streak_samples >= 10 and signal.streak_win_rate >= 0.80:
            bull_points.append(f"{signal.streak_length} consecutive down days with {signal.streak_win_rate:.0%} historical bounce rate ({signal.streak_samples} samples)")
        if signal.news_sentiment > 0.1:
            bull_points.append(f"Positive news sentiment ({signal.news_sentiment:+.2f})")
        if signal.insider_signal == 'bullish':
            bull_points.append(f"Insider buying detected (${signal.insider_value:,.0f})")
        if signal.markov_agrees:
            bull_points.append(f"Markov chain agrees (pi_up={signal.markov_pi_up:.0%})")
        if signal.cognitive_approved:
            bull_points.append(f"Cognitive brain approved ({signal.cognitive_confidence:.0%} confidence)")

        thesis.bull_case = ". ".join(bull_points) if bull_points else "No strong bullish factors identified."

        # Bear Case
        bear_points = []
        if signal.regime == 'bearish':
            bear_points.append(f"Market regime is {signal.regime}")
        if signal.vix_level > 25:
            bear_points.append(f"Elevated VIX ({signal.vix_level:.1f})")
        if signal.news_sentiment < -0.1:
            bear_points.append(f"Negative news sentiment ({signal.news_sentiment:+.2f})")
        if signal.sector_relative_strength < -0.05:
            bear_points.append(f"Sector underperforming ({signal.sector_relative_strength:+.1%})")
        if not signal.knowledge_boundary_safe:
            bear_points.append("Knowledge boundary concern flagged")

        thesis.bear_case = ". ".join(bear_points) if bear_points else "No significant bearish factors identified."

        # Risks
        thesis.risks = [
            "Broader market selloff could override setup",
            "Unexpected news could invalidate thesis",
            "Earnings/corporate action risk",
            "Position correlation with existing holdings",
            "Liquidity/execution risk during volatility",
        ]

        # Claude AI Analysis (if available)
        if self.registry.llm_analyzer is not None:
            try:
                analyzer = self.registry.llm_analyzer()
                claude_result = analyzer.analyze_trade(signal.to_dict(), thesis.to_dict())

                thesis.claude_analysis = claude_result.get('analysis', '')
                thesis.claude_recommendation = claude_result.get('recommendation', '')
                thesis.claude_confidence = claude_result.get('confidence', 0.0)
            except Exception as e:
                logger.debug(f"Claude analysis failed: {e}")
                thesis.claude_analysis = "Claude analysis not available"

        # Final Verdict
        if signal.final_conf_score >= 0.75 and signal.cognitive_approved:
            thesis.verdict = "HIGH CONVICTION TRADE"
            thesis.conviction_level = "STRONG BUY"
        elif signal.final_conf_score >= 0.60:
            thesis.verdict = "MODERATE CONVICTION TRADE"
            thesis.conviction_level = "BUY"
        else:
            thesis.verdict = "LOW CONVICTION - CONSIDER PASSING"
            thesis.conviction_level = "HOLD"

        return thesis


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_unified_pipeline(verbose: bool = True) -> UnifiedSignalEnrichmentPipeline:
    """Factory function to get pipeline instance."""
    return UnifiedSignalEnrichmentPipeline(verbose=verbose)


def run_full_enrichment(
    signals: pd.DataFrame,
    price_data: pd.DataFrame,
    spy_data: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> Tuple[List[EnrichedSignal], List[TradeThesis]]:
    """
    Convenience function to run the full enrichment pipeline.

    Returns:
        Tuple of (all_enriched_signals, top2_theses)
    """
    pipeline = get_unified_pipeline(verbose=verbose)
    return pipeline.enrich_signals(signals, price_data, spy_data)


if __name__ == "__main__":
    # Test the component registry
    registry = ComponentRegistry()
    registry.print_status()
