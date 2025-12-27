"""
Cognitive Signal Processor
===========================

Integration layer between the Cognitive Brain and the trading workflow.

This module provides:
- Signal evaluation through the cognitive system
- Pre-trade approval with reasoning
- Post-trade outcome learning
- Market context building from current data

Usage:
    from cognitive.signal_processor import CognitiveSignalProcessor

    processor = CognitiveSignalProcessor()

    # Evaluate signals before trading
    approved_signals = processor.evaluate_signals(signals_df, market_data)

    # After trade completes
    processor.record_outcome(decision_id, won=True, pnl=150.0)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EvaluatedSignal:
    """A signal that has been evaluated by the cognitive system."""
    original_signal: Dict[str, Any]
    approved: bool
    cognitive_confidence: float
    reasoning_trace: List[str]
    concerns: List[str]
    knowledge_gaps: List[str]
    invalidators: List[str]
    episode_id: str
    decision_mode: str
    size_multiplier: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
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
    Integrates CognitiveBrain with the trading signal workflow.

    Key responsibilities:
    - Build market context from available data
    - Evaluate each signal through cognitive deliberation
    - Filter/rank signals by cognitive confidence
    - Record outcomes for learning
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        max_concurrent_positions: int = 5,
        vix_threshold: float = 35.0,
    ):
        self.min_confidence = min_confidence
        self.max_concurrent_positions = max_concurrent_positions
        self.vix_threshold = vix_threshold

        # Lazy load brain to avoid circular imports
        self._brain = None
        self._active_episodes: Dict[str, str] = {}  # decision_id -> episode_id

        logger.info("CognitiveSignalProcessor initialized")

    @property
    def brain(self):
        """Lazy load cognitive brain."""
        if self._brain is None:
            from cognitive.cognitive_brain import get_cognitive_brain
            self._brain = get_cognitive_brain()
        return self._brain

    def build_market_context(
        self,
        market_data: Optional[pd.DataFrame] = None,
        spy_data: Optional[pd.DataFrame] = None,
        vix_value: Optional[float] = None,
        current_positions: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Build market context from available data sources.

        Args:
            market_data: Combined OHLCV data for universe
            spy_data: SPY data for regime detection
            vix_value: Current VIX value (fetched if not provided)
            current_positions: List of current positions from broker

        Returns:
            Context dictionary for cognitive evaluation
        """
        context = {
            'timestamp': datetime.now().isoformat(),
            'data_timestamp': datetime.now(),
        }

        # Regime detection from SPY
        if spy_data is not None and not spy_data.empty:
            context.update(self._detect_regime(spy_data))
        else:
            context['regime'] = 'unknown'
            context['regime_confidence'] = 0.5

        # VIX for volatility context
        if vix_value is not None:
            context['vix'] = vix_value
        else:
            context['vix'] = self._fetch_vix()

        # Position context
        if current_positions:
            context['current_positions'] = len(current_positions)
            context['position_symbols'] = [p.get('symbol') for p in current_positions]
        else:
            context['current_positions'] = 0
            context['position_symbols'] = []

        # Market breadth from universe data
        if market_data is not None and not market_data.empty:
            context.update(self._calculate_breadth(market_data))

        return context

    def _detect_regime(self, spy_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect market regime from SPY data."""
        try:
            df = spy_data.sort_values('timestamp').tail(200)
            if len(df) < 50:
                return {'regime': 'unknown', 'regime_confidence': 0.3}

            close = df['close'].values

            # Simple regime detection
            sma20 = close[-20:].mean()
            sma50 = close[-50:].mean()
            sma200 = close[-200:].mean() if len(close) >= 200 else close.mean()
            current = close[-1]

            # Calculate trend strength
            if current > sma20 > sma50 > sma200:
                regime = 'BULL'
                confidence = 0.85
            elif current > sma20 > sma50:
                regime = 'BULL'
                confidence = 0.70
            elif current < sma20 < sma50 < sma200:
                regime = 'BEAR'
                confidence = 0.85
            elif current < sma20 < sma50:
                regime = 'BEAR'
                confidence = 0.70
            else:
                regime = 'CHOPPY'
                confidence = 0.60

            # Volatility adjustment
            returns = pd.Series(close).pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5)  # Annualized

            if volatility > 0.30:
                regime = 'HIGH_VOL'
                confidence = min(confidence, 0.65)

            return {
                'regime': regime,
                'regime_confidence': confidence,
                'spy_sma20': sma20,
                'spy_sma50': sma50,
                'spy_sma200': sma200,
                'spy_volatility': volatility,
            }

        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return {'regime': 'unknown', 'regime_confidence': 0.3}

    def _fetch_vix(self) -> float:
        """Fetch current VIX value."""
        try:
            from data.providers.multi_source import fetch_daily_bars_multi
            from datetime import timedelta

            end = datetime.now().date().isoformat()
            start = (datetime.now().date() - timedelta(days=5)).isoformat()

            vix_data = fetch_daily_bars_multi('VIX', start, end)
            if not vix_data.empty:
                return float(vix_data['close'].iloc[-1])
        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}")

        return 20.0  # Default neutral VIX

    def _calculate_breadth(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market breadth indicators."""
        try:
            # Get latest bar for each symbol
            latest = market_data.sort_values('timestamp').groupby('symbol').tail(1)

            if len(latest) < 10:
                return {}

            # Calculate advances/declines
            if 'open' in latest.columns and 'close' in latest.columns:
                advances = (latest['close'] > latest['open']).sum()
                declines = (latest['close'] < latest['open']).sum()
                total = len(latest)

                return {
                    'advances': advances,
                    'declines': declines,
                    'advance_decline_ratio': advances / max(1, declines),
                    'breadth_pct': advances / total if total > 0 else 0.5,
                }
        except Exception as e:
            logger.warning(f"Breadth calculation failed: {e}")

        return {}

    def evaluate_signals(
        self,
        signals: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        spy_data: Optional[pd.DataFrame] = None,
        vix_value: Optional[float] = None,
        current_positions: Optional[List[Dict]] = None,
        fast_confidences: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, List[EvaluatedSignal]]:
        """
        Evaluate trading signals through the cognitive system.

        Args:
            signals: DataFrame of trading signals
            market_data: Universe OHLCV data
            spy_data: SPY data for regime
            vix_value: Current VIX
            current_positions: Broker positions
            fast_confidences: Pre-computed ML confidences by symbol

        Returns:
            Tuple of (approved_signals_df, all_evaluated_signals)
        """
        if signals.empty:
            return pd.DataFrame(), []

        # Build market context
        context = self.build_market_context(
            market_data=market_data,
            spy_data=spy_data,
            vix_value=vix_value,
            current_positions=current_positions,
        )

        evaluated: List[EvaluatedSignal] = []

        for idx, row in signals.iterrows():
            signal_dict = row.to_dict()
            symbol = signal_dict.get('symbol', 'UNKNOWN')
            strategy = signal_dict.get('strategy', 'unknown')

            # Get fast confidence if available
            fast_conf = None
            if fast_confidences and symbol in fast_confidences:
                fast_conf = fast_confidences[symbol]
            elif 'conf_score' in signal_dict:
                fast_conf = float(signal_dict.get('conf_score', 0.5))

            # Build signal context
            signal_context = {
                **context,
                'strategy': strategy,
                'symbol': symbol,
                'price': signal_dict.get('entry_price', 0),
                'volume': signal_dict.get('volume', 0),
            }

            # Deliberate through cognitive brain
            try:
                decision = self.brain.deliberate(
                    signal=signal_dict,
                    context=signal_context,
                    fast_confidence=fast_conf,
                )

                eval_signal = EvaluatedSignal(
                    original_signal=signal_dict,
                    approved=decision.should_act,
                    cognitive_confidence=decision.confidence,
                    reasoning_trace=decision.reasoning_trace,
                    concerns=decision.concerns,
                    knowledge_gaps=decision.knowledge_gaps,
                    invalidators=decision.invalidators,
                    episode_id=decision.episode_id,
                    decision_mode=decision.decision_mode,
                    size_multiplier=decision.action.get('size_multiplier', 1.0) if decision.action else 0.5,
                )

                evaluated.append(eval_signal)

                # Track episode for later outcome recording
                decision_id = f"{symbol}_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self._active_episodes[decision_id] = decision.episode_id

                logger.info(
                    f"Evaluated {symbol}: approved={decision.should_act}, "
                    f"conf={decision.confidence:.2f}, mode={decision.decision_mode}"
                )

            except Exception as e:
                logger.error(f"Cognitive evaluation failed for {symbol}: {e}")
                # Create a pass-through evaluation on error
                eval_signal = EvaluatedSignal(
                    original_signal=signal_dict,
                    approved=True,  # Don't block on cognitive errors
                    cognitive_confidence=fast_conf or 0.5,
                    reasoning_trace=["Cognitive evaluation error - using fallback"],
                    concerns=[str(e)],
                    knowledge_gaps=[],
                    invalidators=[],
                    episode_id="",
                    decision_mode="fallback",
                    size_multiplier=0.5,
                )
                evaluated.append(eval_signal)

        # Build approved signals DataFrame
        approved_rows = []
        for ev in evaluated:
            if ev.approved and ev.cognitive_confidence >= self.min_confidence:
                approved_rows.append(ev.to_dict())

        approved_df = pd.DataFrame(approved_rows) if approved_rows else pd.DataFrame()

        # Sort by cognitive confidence
        if not approved_df.empty and 'cognitive_confidence' in approved_df.columns:
            approved_df = approved_df.sort_values('cognitive_confidence', ascending=False)

        logger.info(
            f"Cognitive evaluation: {len(evaluated)} signals -> "
            f"{len(approved_df)} approved (min_conf={self.min_confidence})"
        )

        return approved_df, evaluated

    def record_outcome(
        self,
        decision_id: str,
        won: bool,
        pnl: float,
        r_multiple: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Record trade outcome for cognitive learning.

        Args:
            decision_id: Identifier linking to original signal
            won: Whether the trade was profitable
            pnl: Profit/loss amount
            r_multiple: P&L as multiple of risk (optional)
            notes: Additional notes

        Returns:
            True if outcome was recorded successfully
        """
        episode_id = self._active_episodes.get(decision_id)

        if not episode_id:
            logger.warning(f"No episode found for decision {decision_id}")
            return False

        try:
            outcome = {
                'won': won,
                'pnl': pnl,
                'r_multiple': r_multiple,
                'notes': notes,
                'recorded_at': datetime.now().isoformat(),
            }

            self.brain.learn_from_outcome(episode_id, outcome)

            # Remove from active episodes
            del self._active_episodes[decision_id]

            logger.info(
                f"Recorded outcome for {decision_id}: "
                f"won={won}, pnl={pnl:.2f}, r={r_multiple}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")
            return False

    def record_outcome_by_symbol(
        self,
        symbol: str,
        strategy: str,
        won: bool,
        pnl: float,
        r_multiple: Optional[float] = None,
    ) -> bool:
        """
        Record outcome by symbol (finds most recent matching episode).

        Useful when you don't have the exact decision_id.
        """
        # Find matching active episode
        for decision_id, episode_id in list(self._active_episodes.items()):
            if symbol in decision_id and strategy.lower() in decision_id.lower():
                return self.record_outcome(
                    decision_id=decision_id,
                    won=won,
                    pnl=pnl,
                    r_multiple=r_multiple,
                )

        logger.warning(f"No active episode found for {symbol}/{strategy}")
        return False

    def get_cognitive_status(self) -> Dict[str, Any]:
        """Get status of the cognitive system."""
        try:
            brain_status = self.brain.get_status()
            return {
                'processor_active': True,
                'min_confidence': self.min_confidence,
                'active_episodes': len(self._active_episodes),
                'brain_status': brain_status,
            }
        except Exception as e:
            return {
                'processor_active': False,
                'error': str(e),
            }

    def daily_maintenance(self) -> Dict[str, Any]:
        """Run daily cognitive maintenance tasks."""
        try:
            result = self.brain.daily_consolidation()

            # Clean up stale active episodes (older than 24 hours)
            # In production, you'd check timestamps
            stale_count = 0

            return {
                'consolidation': result,
                'stale_episodes_cleaned': stale_count,
            }
        except Exception as e:
            logger.error(f"Daily maintenance failed: {e}")
            return {'error': str(e)}

    def introspect(self) -> str:
        """Generate introspection report."""
        lines = [
            "=== Cognitive Signal Processor ===",
            "",
            f"Min confidence threshold: {self.min_confidence}",
            f"Max concurrent positions: {self.max_concurrent_positions}",
            f"VIX threshold: {self.vix_threshold}",
            f"Active episodes tracking: {len(self._active_episodes)}",
            "",
            "--- Brain Status ---",
        ]

        try:
            lines.append(self.brain.introspect())
        except Exception as e:
            lines.append(f"Brain introspection error: {e}")

        return "\n".join(lines)


# Convenience functions
def get_signal_processor() -> CognitiveSignalProcessor:
    """Get or create cognitive signal processor."""
    global _signal_processor
    if '_signal_processor' not in globals() or _signal_processor is None:
        _signal_processor = CognitiveSignalProcessor()
    return _signal_processor

_signal_processor: Optional[CognitiveSignalProcessor] = None
