"""
Cognitive Signal Processor - The Bridge to the Brain
======================================================

This module serves as the primary integration layer, connecting the standard
trading signal generation workflow with the advanced cognitive architecture.
It acts as a "gatekeeper," taking raw signals, preparing them for the brain,
managing the deliberation process, and then returning the final, adjudicated
decisions.

Core Workflow:
1.  **Receive Signals:** Takes a DataFrame of raw trading signals from the
    strategy layer.
2.  **Build Context:** Enriches these signals by gathering and structuring a
    comprehensive market context (e.g., regime, volatility, market breadth, **news sentiment**).
3.  **Deliberate:** For each signal, it calls the `CognitiveBrain` to perform
    a deep, reasoned evaluation.
4.  **Return Approved Signals:** It returns a new DataFrame containing only the
    signals that have been approved by the cognitive system, augmented with
    additional data like the confidence score and position size multiplier.
5.  **Manage Learning Loop:** It provides a `record_outcome` method to ensure
    that the results of trades are fed back to the `CognitiveBrain`, closing
    the learning loop.

Usage:
    from cognitive.signal_processor import get_signal_processor

    # In the main trading script:
    processor = get_signal_processor()

    # Instead of trading raw signals, evaluate them first.
    raw_signals_df = generate_raw_signals()
    approved_signals_df, _ = processor.evaluate_signals(raw_signals_df, market_data)

    # Trade the approved signals...
    # ... then after a trade is closed:
    processor.record_outcome_by_symbol(
        symbol='AAPL',
        strategy='ibs_rsi',
        won=True,
        pnl=150.0
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import the NewsProcessor for sentiment analysis
from altdata.news_processor import get_news_processor
# Import the MarketMoodAnalyzer for holistic market emotional state
from altdata.market_mood_analyzer import get_market_mood_analyzer
# Import OnlineLearningManager for incremental model updates
from ml_advanced.online_learning import OnlineLearningManager, create_online_learning_manager
# Import structured logging for audit trail
from core.structured_log import jlog

logger = logging.getLogger(__name__)


@dataclass
class EvaluatedSignal:
    """
    A data class that enriches a raw signal with the output of the cognitive
    deliberation process. It represents a signal that has been "thought about."
    """
    original_signal: Dict[str, Any]
    approved: bool # Did the brain approve this signal for action?
    cognitive_confidence: float # The brain's final confidence score for this signal.
    reasoning_trace: List[str] # The step-by-step reasoning process.
    concerns: List[str] # Any negative factors the brain considered.
    knowledge_gaps: List[str] # Information the brain identified as missing.
    invalidators: List[str] # Conditions that would invalidate this decision.
    episode_id: str # The unique ID for the memory of this decision.
    decision_mode: str # The cognitive mode used ('fast', 'slow', etc.).
    size_multiplier: float = 1.0 # A multiplier (0.0-1.0) for position size based on confidence.

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the evaluated signal into a dictionary, suitable for adding
        to a pandas DataFrame. It prefixes cognitive fields to avoid name clashes.
        """
        return {
            **self.original_signal,
            'cognitive_approved': self.approved,
            'cognitive_confidence': self.cognitive_confidence,
            'cognitive_reasoning': '; '.join(self.reasoning_trace[:3]),
            'cognitive_concerns': '; '.join(self.concerns),
            'cognitive_episode_id': self.episode_id,
            'cognitive_decision_mode': self.decision_mode,
            'cognitive_size_multiplier': self.size_multiplier,
        }


class CognitiveSignalProcessor:
    """
    Manages the interaction between the trading system's signal generation and
    the `CognitiveBrain`, orchestrating evaluation and learning.
    """

    def __init__(self, min_confidence: float = 0.45):
        """
        Initializes the processor.

        Args:
            min_confidence: The minimum cognitive confidence required for a
                            signal to be considered "approved."

        NOTE: Lowered from 0.51 to 0.45 (2025-12-31) because ML ensemble models
        predict honestly around 45-50% for mean-reversion signals, which is
        appropriate given the ~48% baseline win rate in training data.
        """
        self.min_confidence = min_confidence
        self._brain = None  # Lazy-loaded to avoid circular imports.
        self._knowledge_boundary = None  # Lazy-loaded for acceptance decisions.
        self._news_processor = None # Lazy-loaded for news and sentiment.
        self._market_mood_analyzer = None # Lazy-loaded for market mood analysis.
        self._online_learning_manager = None  # Lazy-loaded for incremental learning.

        # A dictionary to track the link between a unique decision identifier
        # (like symbol + strategy) and the brain's internal episode_id. This
        # is crucial for recording outcomes later.
        self._active_episodes: Dict[str, str] = {}

        # Store signal features for online learning when outcomes are recorded.
        # Maps decision_id -> (features_array, prediction_confidence, entry_timestamp)
        self._signal_features: Dict[str, Tuple[np.ndarray, float, datetime]] = {}

        logger.info("CognitiveSignalProcessor initialized.")

    @property
    def brain(self):
        """Lazy-loads the CognitiveBrain singleton instance."""
        if self._brain is None:
            from cognitive.cognitive_brain import get_cognitive_brain
            self._brain = get_cognitive_brain()
        return self._brain

    @property
    def knowledge_boundary(self):
        """Lazy-loads the KnowledgeBoundary singleton instance."""
        if self._knowledge_boundary is None:
            from cognitive.knowledge_boundary import KnowledgeBoundary
            self._knowledge_boundary = KnowledgeBoundary()
        return self._knowledge_boundary

    @property
    def news_processor(self):
        """Lazy-loads the NewsProcessor singleton instance."""
        if self._news_processor is None:
            self._news_processor = get_news_processor()
        return self._news_processor

    @property
    def market_mood_analyzer(self):
        """Lazy-loads the MarketMoodAnalyzer singleton instance."""
        if self._market_mood_analyzer is None:
            self._market_mood_analyzer = get_market_mood_analyzer()
        return self._market_mood_analyzer

    @property
    def online_learning_manager(self) -> OnlineLearningManager:
        """Lazy-loads the OnlineLearningManager for incremental model updates."""
        if self._online_learning_manager is None:
            self._online_learning_manager = create_online_learning_manager(
                update_frequency='daily',
                auto_update=True
            )
        return self._online_learning_manager

    def build_market_context(
        self,
        market_data: Optional[pd.DataFrame] = None,
        spy_data: Optional[pd.DataFrame] = None,
        vix_value: Optional[float] = None,
        current_positions: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Gathers data from various sources and assembles a comprehensive
        market context dictionary for the brain to use in its deliberations.
        """
        context = {'timestamp': datetime.now().isoformat(), 'data_timestamp': datetime.now()}

        # 1. Determine Market Regime (e.g., from SPY data).
        if spy_data is not None and not spy_data.empty:
            context.update(self._detect_regime(spy_data))
        else:
            context['regime'], context['regime_confidence'] = 'unknown', 0.3

        # 2. Get Volatility (VIX).
        context['vix'] = vix_value if vix_value is not None else self._fetch_vix()

        # 3. Get Portfolio Context.
        context['current_positions'] = len(current_positions) if current_positions else 0
        context['position_symbols'] = [p.get('symbol') for p in current_positions] if current_positions else []

        # 4. Calculate Market Breadth.
        if market_data is not None and not market_data.empty:
            context.update(self._calculate_breadth(market_data))

        # 5. Add Market News Sentiment
        market_sentiment = self.news_processor.get_aggregated_sentiment()
        context['market_sentiment'] = market_sentiment

        # 6. Add Market Mood (combines VIX + Sentiment for holistic emotional state)
        # Extract compound sentiment for mood calculation
        sentiment_score = market_sentiment.get('compound', 0.0)
        mood_context = {
            'vix': context['vix'],
            'sentiment': sentiment_score,
        }
        mood_result = self.market_mood_analyzer.get_market_mood(mood_context)
        context.update(mood_result)  # Adds market_mood, market_mood_score, market_mood_state, is_extreme_mood

        logger.debug(
            f"Built market context: Regime={context['regime']}, VIX={context['vix']:.2f}, "
            f"Market Sentiment (compound): {sentiment_score:.2f}, "
            f"Market Mood: {context.get('market_mood_state', 'unknown')} ({context.get('market_mood_score', 0):.2f})"
        )
        return context

    def _detect_regime(self, spy_data: pd.DataFrame) -> Dict[str, Any]:
        """A simple market regime detection based on SPY moving averages."""
        try:
            df = spy_data.sort_values('timestamp').tail(200)
            if len(df) < 50: return {'regime': 'unknown', 'regime_confidence': 0.3}

            close = df['close'].values
            sma20, sma50 = close[-20:].mean(), close[-50:].mean()
            sma200 = close.mean()
            current = close[-1]
            
            # Simple trend detection
            if current > sma20 > sma50 > sma200: regime, confidence = 'BULL', 0.85
            elif current > sma20 > sma50: regime, confidence = 'BULL', 0.70
            elif current < sma20 < sma50 < sma200: regime, confidence = 'BEAR', 0.85
            elif current < sma20 < sma50: regime, confidence = 'BEAR', 0.70
            else: regime, confidence = 'CHOPPY', 0.60
            
            return {'regime': regime, 'regime_confidence': confidence}
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return {'regime': 'unknown', 'regime_confidence': 0.3}

    def _fetch_vix(self) -> float:
        """Placeholder for fetching the current VIX value."""
        # In a real system, this would call a data provider.
        return 20.0

    def _calculate_breadth(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Placeholder for calculating market breadth indicators."""
        return {}

    def _extract_signal_features(self, signal: Dict[str, Any], context: Dict[str, Any]) -> np.ndarray:
        """
        Extracts a feature vector from a signal for online learning.

        The feature vector captures the key characteristics of the trade setup
        that can be used to train incremental ML models.

        Args:
            signal: The trading signal dictionary.
            context: The market context dictionary.

        Returns:
            A numpy array of features.
        """
        # Extract numeric features from signal
        entry_price = float(signal.get('entry_price', 0) or 0)
        stop_loss = float(signal.get('stop_loss', 0) or 0)
        take_profit = float(signal.get('take_profit', 0) or 0)
        conf_score = float(signal.get('conf_score', 0.5) or 0.5)

        # Calculate risk/reward ratio
        risk = abs(entry_price - stop_loss) if entry_price and stop_loss else 1.0
        reward = abs(take_profit - entry_price) if take_profit and entry_price else 1.0
        rr_ratio = reward / risk if risk > 0 else 0.0

        # Extract context features
        vix = float(context.get('vix', 20.0) or 20.0)
        regime_confidence = float(context.get('regime_confidence', 0.5) or 0.5)

        # Encode regime as numeric
        regime_map = {'BULL': 1.0, 'BEAR': -1.0, 'CHOPPY': 0.0, 'unknown': 0.0}
        regime_val = regime_map.get(context.get('regime', 'unknown'), 0.0)

        # Get market sentiment if available
        sentiment = context.get('market_sentiment', {})
        sentiment_compound = float(sentiment.get('compound', 0.0) if isinstance(sentiment, dict) else 0.0)

        # Get market mood score if available
        mood_score = float(context.get('market_mood_score', 0.0) or 0.0)

        # Strategy encoding (simple one-hot for common strategies)
        strategy = signal.get('strategy', '').lower()
        is_ibs_rsi = 1.0 if 'ibs' in strategy or 'rsi' in strategy else 0.0
        is_turtle_soup = 1.0 if 'turtle' in strategy or 'ict' in strategy else 0.0

        # Build feature vector
        features = np.array([
            conf_score,           # 0: Signal confidence
            rr_ratio,             # 1: Risk/reward ratio
            vix / 100.0,          # 2: VIX normalized (0-1 range roughly)
            regime_val,           # 3: Regime (-1 to 1)
            regime_confidence,    # 4: Regime confidence
            sentiment_compound,   # 5: Market sentiment (-1 to 1)
            mood_score,           # 6: Market mood score
            is_ibs_rsi,           # 7: IBS/RSI strategy flag
            is_turtle_soup,       # 8: Turtle Soup strategy flag
        ], dtype=np.float32)

        return features

    def evaluate_signals(
        self,
        signals: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        spy_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, List[EvaluatedSignal]]:
        """
        The main method of this class. It orchestrates the evaluation of a
        batch of trading signals through the cognitive system.

        Args:
            signals: A DataFrame of raw trading signals.
            market_data: Supporting market data for context building.
            spy_data: SPY data for regime detection.

        Returns:
            A tuple containing:
            - A DataFrame of signals approved by the cognitive system.
            - A list of all `EvaluatedSignal` objects.
        """
        if signals.empty:
            return pd.DataFrame(), []

        # 1. Build a single market context for this batch of signals.
        context = self.build_market_context(market_data=market_data, spy_data=spy_data)
        evaluated_signals: List[EvaluatedSignal] = []

        # 2. Iterate through each signal and send it to the brain for deliberation.
        for _, row in signals.iterrows():
            signal_dict = row.to_dict()
            symbol = signal_dict.get('symbol', 'UNKNOWN')
            
            try:
                # Get symbol-specific news sentiment
                symbol_sentiment = self.news_processor.get_aggregated_sentiment(symbols=[symbol])

                # 3. The `deliberate` call is the core "thinking" step.
                full_context = {
                    **context, # Include overall market context
                    'symbol': symbol,
                    'price': signal_dict.get('entry_price', 0),
                    'volume': signal_dict.get('volume', 0),
                    'symbol_sentiment': symbol_sentiment, # Add symbol-specific sentiment
                }
                ensemble_confidence = float(signal_dict.get('conf_score', 0.5))

                decision = self.brain.deliberate(
                    signal=signal_dict,
                    context=full_context,
                    fast_confidence=ensemble_confidence
                )

                # 4. Check if brain rejected but episodic/ensemble evidence supports acceptance.
                #    This allows signals with strong historical support to pass even if brain
                #    has limited live trading experience.
                final_approved = decision.should_act
                final_confidence = decision.confidence
                final_size_multiplier = decision.action.get('size_multiplier', 1.0) if decision.action else 0.5
                override_reasoning = []

                if not decision.should_act:
                    # Try knowledge_boundary.should_accept() as fallback
                    accept_decision = self.knowledge_boundary.should_accept(
                        signal=signal_dict,
                        context=full_context,
                        ensemble_confidence=ensemble_confidence
                    )

                    # Log the cognitive decision with full evidence for audit
                    metadata = accept_decision.get('metadata', {})
                    jlog('cognitive_decision',
                         symbol=symbol,
                         decision=accept_decision['decision'],
                         accept=accept_decision['accept'],
                         reason=accept_decision['reason'],
                         size_multiplier=accept_decision['size_multiplier'],
                         signature=metadata.get('signature'),
                         ensemble_confidence=metadata.get('ensemble_confidence'),
                         episodic_n=metadata.get('episodic_n'),
                         episodic_wr=metadata.get('episodic_wr'),
                         self_model_n=metadata.get('self_model_n'),
                         self_model_wr=metadata.get('self_model_wr'),
                         vix=metadata.get('vix'),
                         regime=metadata.get('regime'),
                         strategy=metadata.get('strategy'))

                    if accept_decision['accept']:
                        # Override brain's rejection with episodic evidence
                        final_approved = True
                        # Use ensemble confidence or a reasonable minimum
                        final_confidence = max(ensemble_confidence, 0.45)
                        final_size_multiplier = accept_decision['size_multiplier']
                        override_reasoning = [
                            f"OVERRIDE: {accept_decision['decision']} via episodic evidence",
                            accept_decision['reason']
                        ]
                        logger.info(
                            f"  {symbol}: Brain rejected -> KnowledgeBoundary override: "
                            f"{accept_decision['decision']} ({accept_decision['reason']})"
                        )

                # 5. Wrap the result in our structured `EvaluatedSignal` object.
                eval_signal = EvaluatedSignal(
                    original_signal=signal_dict,
                    approved=final_approved,
                    cognitive_confidence=final_confidence,
                    reasoning_trace=decision.reasoning_trace + override_reasoning,
                    concerns=decision.concerns,
                    knowledge_gaps=decision.knowledge_gaps,
                    invalidators=decision.invalidators,
                    episode_id=decision.episode_id,
                    decision_mode=decision.decision_mode if not override_reasoning else 'episodic_override',
                    size_multiplier=final_size_multiplier,
                )
                evaluated_signals.append(eval_signal)

                # 6. Store the episode_id so we can link the trade outcome back later.
                decision_id = f"{symbol}|{signal_dict.get('strategy', 'unknown')}"
                self._active_episodes[decision_id] = decision.episode_id

                # 6b. Store signal features for online learning (for approved signals).
                if final_approved:
                    features = self._extract_signal_features(signal_dict, context)
                    self._signal_features[decision_id] = (
                        features,
                        decision.confidence,
                        datetime.now()
                    )

                logger.info(f"Cognitive evaluation for {symbol}: Approved={eval_signal.approved}, Confidence={eval_signal.cognitive_confidence:.2f}")

            except Exception as e:
                logger.error(f"Cognitive deliberation failed for {symbol}: {e}", exc_info=True)
        
        # 6. Filter for signals that were approved and met the confidence threshold.
        approved_rows = [ev.to_dict() for ev in evaluated_signals if ev.approved and ev.cognitive_confidence >= self.min_confidence]
        approved_df = pd.DataFrame(approved_rows) if approved_rows else pd.DataFrame()

        # 7. Sort the final approved signals by confidence.
        if not approved_df.empty:
            approved_df = approved_df.sort_values('cognitive_confidence', ascending=False)

        logger.info(f"Cognitive evaluation complete: {len(signals)} signals in, {len(approved_df)} signals approved.")
        return approved_df, evaluated_signals

    def record_outcome(self, decision_id: str, won: bool, pnl: float, r_multiple: Optional[float] = None) -> bool:
        """
        Records the final outcome of a trade, closing the learning loop.

        This method now also feeds the outcome to the OnlineLearningManager
        for incremental model updates and concept drift detection.

        Args:
            decision_id: A unique identifier for the decision (e.g., "AAPL|ibs_rsi").
            won: Whether the trade was profitable.
            pnl: The profit or loss amount.
            r_multiple: The P&L as a multiple of the initial risk.

        Returns:
            True if the outcome was successfully recorded.
        """
        # Find the corresponding brain episode_id for this decision.
        episode_id = self._active_episodes.pop(decision_id, None)
        if not episode_id:
            logger.warning(f"No active cognitive episode found for decision '{decision_id}'. Cannot record outcome.")
            return False

        # Retrieve stored signal features for online learning.
        signal_data = self._signal_features.pop(decision_id, None)

        try:
            # Tell the brain to learn from what happened.
            outcome_data = {'won': won, 'pnl': pnl, 'r_multiple': r_multiple}
            self.brain.learn_from_outcome(episode_id, outcome_data)
            logger.info(f"Successfully recorded outcome for episode {episode_id} linked to decision '{decision_id}'.")

            # Feed outcome to online learning manager for incremental model updates.
            if signal_data is not None:
                features, prediction_confidence, entry_time = signal_data
                holding_period = (datetime.now() - entry_time).days

                # Extract symbol from decision_id
                symbol = decision_id.split('|')[0] if '|' in decision_id else 'UNKNOWN'

                try:
                    self.online_learning_manager.record_trade_outcome(
                        symbol=symbol,
                        features=features,
                        prediction=prediction_confidence,
                        actual_pnl=pnl,
                        holding_period=max(1, holding_period)  # At least 1 day
                    )

                    # Log drift detection status
                    status = self.online_learning_manager.get_status()
                    logger.info(
                        f"Online learning updated: buffer={status['buffer_size']}, "
                        f"ready={status['buffer_ready']}, "
                        f"accuracy={status['drift_current_accuracy']:.2%}"
                    )
                except Exception as e:
                    logger.warning(f"Online learning update failed: {e}")
            else:
                logger.debug(f"No stored features for decision '{decision_id}', skipping online learning.")

            return True
        except Exception as e:
            logger.error(f"Failed to record outcome for episode {episode_id}: {e}", exc_info=True)
            # Put it back in case it was a transient error.
            self._active_episodes[decision_id] = episode_id
            if signal_data is not None:
                self._signal_features[decision_id] = signal_data
            return False

    def record_outcome_by_symbol(self, symbol: str, strategy: str, **kwargs) -> bool:
        """A convenience method to record an outcome using symbol and strategy."""
        decision_id = f"{symbol}|{strategy.lower()}"
        # Find the most recent decision for this combination.
        matching_ids = [k for k in self._active_episodes if k.startswith(decision_id)]
        if not matching_ids:
            logger.warning(f"No active episode found for {symbol}/{strategy} to record outcome.")
            return False
        
        # In a real system, you might need more sophisticated logic to find the
        # correct ID if multiple are active. Here, we assume the last one.
        return self.record_outcome(decision_id=matching_ids[-1], **kwargs)

    def daily_maintenance(self) -> Dict[str, Any]:
        """Triggers the brain's daily consolidation and learning tasks."""
        try:
            return self.brain.daily_consolidation()
        except Exception as e:
            logger.error(f"Cognitive daily maintenance failed: {e}", exc_info=True)
            return {'error': str(e)}

    def introspect(self) -> str:
        """Returns a human-readable introspection report from the brain."""
        try:
            return self.brain.introspect()
        except Exception as e:
            return f"Brain introspection failed: {e}"

    def get_cognitive_status(self) -> Dict[str, Any]:
        """Returns a status dictionary for the cognitive signal processor."""
        try:
            brain_status = self.brain.get_status()
        except Exception as e:
            brain_status = {'error': str(e)}

        try:
            online_learning_status = self.online_learning_manager.get_status()
        except Exception as e:
            online_learning_status = {'error': str(e)}

        return {
            'processor_active': True,
            'brain_status': brain_status,
            'active_episodes': len(self._active_episodes),
            'pending_features': len(self._signal_features),
            'online_learning': online_learning_status,
        }

# --- Singleton Implementation ---
_signal_processor: Optional[CognitiveSignalProcessor] = None

def get_signal_processor() -> CognitiveSignalProcessor:
    """Factory function to get the singleton instance of the CognitiveSignalProcessor."""
    global _signal_processor
    if _signal_processor is None:
        _signal_processor = CognitiveSignalProcessor()
    return _signal_processor
