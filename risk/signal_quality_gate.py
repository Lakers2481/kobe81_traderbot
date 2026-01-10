"""
Signal Quality Gate - Central filtering for high-quality signals only.

Reduces ~50 signals/week to ~5 by applying multi-factor quality scoring.

Components:
1. Hard Gates (binary pass/fail): ADV, earnings, portfolio heat, spread
2. Soft Scoring (0-100 points): Conviction, ML confidence, strategy, regime, liquidity
3. Penalties (subtracted): Correlation, timing, volatility

Integration Points:
- scripts/scan.py: Filter signals before output
- portfolio/risk_manager.py: Pre-trade validation

Usage:
    from risk.signal_quality_gate import get_quality_gate, filter_to_best_signals

    gate = get_quality_gate()

    # Filter signals
    quality_signals = filter_to_best_signals(
        signals=raw_signals_df,
        price_data=combined_df,
        spy_data=spy_df,
        max_signals=1,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Quality Tiers
# ============================================================================

class QualityTier(Enum):
    """Signal quality tiers."""
    ELITE = "ELITE"           # 90-100: Trade immediately
    EXCELLENT = "EXCELLENT"   # 80-89: Strong candidate
    GOOD = "GOOD"             # 70-79: Passes gate
    MARGINAL = "MARGINAL"     # 60-69: Below threshold
    REJECT = "REJECT"         # 0-59: Do not trade


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class QualityScore:
    """Comprehensive signal quality assessment."""
    # Core scores
    raw_score: float                    # 0-100
    normalized_score: float             # 0-1.0
    tier: QualityTier

    # Component breakdown (max 100 total)
    conviction_component: float = 0.0   # 0-30
    ml_confidence_component: float = 0.0 # 0-25
    strategy_component: float = 0.0     # 0-15
    regime_component: float = 0.0       # 0-15
    liquidity_component: float = 0.0    # 0-15

    # Penalties (subtracted from score)
    correlation_penalty: float = 0.0    # 0-10
    timing_penalty: float = 0.0         # 0-10
    volatility_penalty: float = 0.0     # 0-5

    # Decision
    passes_gate: bool = False
    rejection_reasons: List[str] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = ""
    strategy: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/storage."""
        return {
            "raw_score": round(self.raw_score, 2),
            "normalized_score": round(self.normalized_score, 4),
            "tier": self.tier.value,
            "passes_gate": self.passes_gate,
            "components": {
                "conviction": round(self.conviction_component, 2),
                "ml_confidence": round(self.ml_confidence_component, 2),
                "strategy": round(self.strategy_component, 2),
                "regime": round(self.regime_component, 2),
                "liquidity": round(self.liquidity_component, 2),
            },
            "penalties": {
                "correlation": round(self.correlation_penalty, 2),
                "timing": round(self.timing_penalty, 2),
                "volatility": round(self.volatility_penalty, 2),
            },
            "rejection_reasons": self.rejection_reasons,
            "symbol": self.symbol,
            "strategy": self.strategy,
        }


@dataclass
class QualityGateConfig:
    """Configuration for the quality gate."""
    # Score thresholds
    # ML ensemble models (HMM, LSTM, XGBoost, LightGBM) are now trained.
    min_score_to_pass: float = 70.0
    elite_threshold: float = 90.0
    excellent_threshold: float = 80.0
    good_threshold: float = 70.0
    marginal_threshold: float = 60.0

    # Component weights (total = 100)
    conviction_max: float = 30.0
    ml_confidence_max: float = 25.0
    strategy_max: float = 15.0
    regime_max: float = 15.0
    liquidity_max: float = 15.0

    # Liquidity settings
    min_adv_usd: float = 5_000_000.0       # $5M minimum
    preferred_adv_usd: float = 20_000_000.0  # $20M for full score
    max_spread_pct: float = 0.02           # 2% max spread
    max_order_vs_adv_pct: float = 0.01     # 1% max order impact

    # Timing settings
    avoid_first_minutes: int = 15
    avoid_last_minutes: int = 15
    monday_penalty: float = 1.0            # Penalty points
    friday_penalty: float = 1.5            # Higher weekend risk

    # Volatility settings
    vix_high_threshold: float = 35.0       # Full penalty above this
    vix_elevated_threshold: float = 25.0   # Partial penalty above this

    # Max daily signals
    max_signals_per_day: int = 1


# ============================================================================
# Signal Quality Gate
# ============================================================================

class SignalQualityGate:
    """
    Central quality gate for signal filtering.

    Reduces signal volume by 10x through comprehensive scoring:
    1. Parameter tightening (already done at strategy level)
    2. Quality scoring (this module)
    3. Top-N selection (this module)
    """

    def __init__(self, config: Optional[QualityGateConfig] = None):
        self.config = config or QualityGateConfig()

        # Lazy-loaded components
        self._conviction_scorer = None
        self._confidence_integrator = None

        logger.info("SignalQualityGate initialized with min_score=%s",
                    self.config.min_score_to_pass)

    @property
    def conviction_scorer(self):
        """Lazy load conviction scorer."""
        if self._conviction_scorer is None:
            try:
                from ml_features.conviction_scorer import get_conviction_scorer
                self._conviction_scorer = get_conviction_scorer()
            except ImportError:
                logger.warning("ConvictionScorer not available")
        return self._conviction_scorer

    @property
    def confidence_integrator(self):
        """Lazy load confidence integrator."""
        if self._confidence_integrator is None:
            try:
                from ml_features.confidence_integrator import get_confidence_integrator
                self._confidence_integrator = get_confidence_integrator()
            except ImportError:
                logger.warning("ConfidenceIntegrator not available")
        return self._confidence_integrator

    def _get_calibration_enabled(self) -> bool:
        """Check if calibration is enabled in config."""
        try:
            from config.settings_loader import load_settings
            config = load_settings()
            return config.get('ml', {}).get('calibration', {}).get('enabled', False)
        except Exception:
            return False

    def _get_conformal_enabled(self) -> bool:
        """Check if conformal prediction is enabled in config."""
        try:
            from config.settings_loader import load_settings
            config = load_settings()
            return config.get('ml', {}).get('conformal', {}).get('enabled', False)
        except Exception:
            return False

    def _apply_calibration(self, raw_probability: float) -> float:
        """Apply probability calibration if enabled and fitted."""
        if not self._get_calibration_enabled():
            return raw_probability
        try:
            from ml_meta.calibration import calibrate_probability
            calibrated = calibrate_probability(raw_probability)
            logger.debug(f"Calibrated {raw_probability:.3f} -> {calibrated:.3f}")
            return calibrated
        except Exception as e:
            logger.debug(f"Calibration failed, using raw: {e}")
            return raw_probability

    def _get_uncertainty_adjustment(self, prediction: float) -> float:
        """Get position size multiplier from conformal uncertainty."""
        if not self._get_conformal_enabled():
            return 1.0
        try:
            from ml_meta.conformal import get_position_multiplier
            multiplier = get_position_multiplier(prediction)
            logger.debug(f"Conformal multiplier for {prediction:.3f}: {multiplier:.3f}")
            return multiplier
        except Exception as e:
            logger.debug(f"Conformal prediction failed: {e}")
            return 1.0

    def evaluate_signal(
        self,
        signal: Dict[str, Any],
        price_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame] = None,
        vix_level: Optional[float] = None,
        current_positions: Optional[List[Dict]] = None,
    ) -> QualityScore:
        """
        Evaluate a single signal and return quality score.

        Args:
            signal: Signal dict with symbol, strategy, entry_price, etc.
            price_data: Price data for the signal's symbol
            spy_data: SPY data for regime detection (optional)
            vix_level: Current VIX level (optional)
            current_positions: List of current positions (optional)

        Returns:
            QualityScore with full breakdown
        """
        rejection_reasons = []
        current_positions = current_positions or []

        symbol = signal.get('symbol', '')
        strategy = signal.get('strategy', '')

        # === HISTORICAL PATTERN AUTO-PASS ===
        # If 20+ samples with 90%+ win rate, this is a statistically significant edge
        # AUTO-PASS the quality gate with ELITE tier
        historical_pattern = signal.get('historical_pattern', {})
        if historical_pattern:
            sample_size = historical_pattern.get('sample_size', 0)
            win_rate = historical_pattern.get('historical_reversal_rate', 0)

            if sample_size >= 20 and win_rate >= 0.90:
                logger.info(
                    f"HISTORICAL PATTERN AUTO-PASS: {symbol} has {sample_size} samples "
                    f"with {win_rate:.0%} win rate - bypassing quality gate"
                )
                return QualityScore(
                    raw_score=95.0,  # ELITE tier
                    normalized_score=0.95,
                    tier=QualityTier.ELITE,
                    conviction_component=30.0,  # Max conviction
                    ml_confidence_component=25.0,  # Max ML
                    strategy_component=15.0,
                    regime_component=15.0,
                    liquidity_component=10.0,
                    correlation_penalty=0.0,
                    timing_penalty=0.0,
                    volatility_penalty=0.0,
                    passes_gate=True,
                    rejection_reasons=[],
                    symbol=symbol,
                    strategy=strategy,
                )

        # === HARD GATES (Binary) ===

        # 1. Liquidity hard gate
        passes_liquidity, liq_reason = self._check_liquidity_hard_gate(
            signal, price_data
        )
        if not passes_liquidity:
            rejection_reasons.append(liq_reason)

        # 2. Earnings hard gate
        passes_earnings, earn_reason = self._check_earnings_gate(signal)
        if not passes_earnings:
            rejection_reasons.append(earn_reason)

        # If any hard gate failed, return early rejection
        if rejection_reasons:
            return QualityScore(
                raw_score=0.0,
                normalized_score=0.0,
                tier=QualityTier.REJECT,
                passes_gate=False,
                rejection_reasons=rejection_reasons,
                symbol=symbol,
                strategy=strategy,
            )

        # === SOFT SCORING ===

        # Component 1: Conviction (30 points max)
        conviction_component = self._calculate_conviction_component(
            signal, price_data, spy_data, vix_level
        )

        # Component 2: ML Confidence (25 points max)
        ml_confidence_component = self._calculate_ml_confidence_component(
            signal, price_data, spy_data, vix_level
        )

        # Component 3: Strategy Score (15 points max)
        strategy_component = self._calculate_strategy_component(signal)

        # Component 4: Regime Alignment (15 points max)
        regime_component = self._calculate_regime_component(signal, spy_data)

        # Component 5: Liquidity Score (15 points max)
        liquidity_component = self._calculate_liquidity_component(signal, price_data)

        # === PENALTIES ===

        correlation_penalty = self._calculate_correlation_penalty(
            signal, current_positions, price_data
        )
        timing_penalty = self._calculate_timing_penalty(signal)
        volatility_penalty = self._calculate_volatility_penalty(vix_level)

        # === FINAL CALCULATION ===

        raw_score = (
            conviction_component +
            ml_confidence_component +
            strategy_component +
            regime_component +
            liquidity_component -
            correlation_penalty -
            timing_penalty -
            volatility_penalty
        )
        raw_score = max(0, min(100, raw_score))
        normalized_score = raw_score / 100

        tier = self._get_tier(raw_score)
        passes_gate = raw_score >= self.config.min_score_to_pass

        if not passes_gate:
            rejection_reasons.append(
                f"Score {raw_score:.1f} below threshold {self.config.min_score_to_pass}"
            )

        return QualityScore(
            raw_score=raw_score,
            normalized_score=normalized_score,
            tier=tier,
            conviction_component=conviction_component,
            ml_confidence_component=ml_confidence_component,
            strategy_component=strategy_component,
            regime_component=regime_component,
            liquidity_component=liquidity_component,
            correlation_penalty=correlation_penalty,
            timing_penalty=timing_penalty,
            volatility_penalty=volatility_penalty,
            passes_gate=passes_gate,
            rejection_reasons=rejection_reasons,
            symbol=symbol,
            strategy=strategy,
        )

    def filter_signals(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame] = None,
        vix_level: Optional[float] = None,
        current_positions: Optional[List[Dict]] = None,
        max_signals: int = 1,
    ) -> pd.DataFrame:
        """
        Filter signals to only the highest quality.

        Args:
            signals: DataFrame of raw signals
            price_data: Combined price data for all symbols
            spy_data: SPY data for regime detection
            vix_level: Current VIX level
            current_positions: Current portfolio positions
            max_signals: Maximum signals to return (default: 1)

        Returns:
            DataFrame of filtered signals with quality scores
        """
        if signals.empty:
            return signals

        # Evaluate each signal
        quality_results = []
        auto_pass_count = 0
        for _, row in signals.iterrows():
            signal = row.to_dict()
            symbol = signal.get('symbol', '')

            # ================================================================
            # AUTO-PASS LOGIC: Elite historical patterns bypass quality gate
            # ================================================================
            qualifies_auto_pass = signal.get('qualifies_auto_pass', False)
            if qualifies_auto_pass:
                # AUTO-PASS: Bypass evaluation, force pass with ELITE score
                quality_results.append({
                    **signal,
                    'quality_score': 95.0,  # ELITE tier (guaranteed top 2)
                    'quality_tier': QualityTier.ELITE.value,
                    'passes_gate': True,
                    'rejection_reasons': '',
                    'auto_pass_applied': True,  # Flag for tracking
                })
                auto_pass_count += 1
                logger.info(
                    "AUTO-PASS: %s (streak=%s, samples=%s, WR=%.0f%%)",
                    symbol,
                    signal.get('streak_length', 0),
                    signal.get('streak_samples', 0),
                    signal.get('streak_win_rate', 0) * 100
                )
                continue  # Skip normal evaluation

            # Normal evaluation for non-auto-pass signals
            # Get symbol-specific price data
            if 'symbol' in price_data.columns:
                symbol_prices = price_data[price_data['symbol'] == symbol]
            else:
                symbol_prices = price_data

            quality = self.evaluate_signal(
                signal=signal,
                price_data=symbol_prices,
                spy_data=spy_data,
                vix_level=vix_level,
                current_positions=current_positions,
            )

            quality_results.append({
                **signal,
                'quality_score': quality.raw_score,
                'quality_tier': quality.tier.value,
                'passes_gate': quality.passes_gate,
                'rejection_reasons': '; '.join(quality.rejection_reasons) if quality.rejection_reasons else '',
                'auto_pass_applied': False,
            })

        result_df = pd.DataFrame(quality_results)

        # Filter to only passing signals
        passing = result_df[result_df['passes_gate']].copy()

        # Sort by quality score and take top N
        if not passing.empty:
            passing = passing.sort_values('quality_score', ascending=False)
            passing = passing.head(max_signals)

        logger.info(
            "Quality gate: %d -> %d signals (filter ratio: %.1fx, auto-pass: %d)",
            len(signals), len(passing),
            len(signals) / max(len(passing), 1),
            auto_pass_count
        )

        return passing

    # === Hard Gate Methods ===

    def _check_liquidity_hard_gate(
        self,
        signal: Dict,
        price_data: pd.DataFrame
    ) -> Tuple[bool, str]:
        """Check hard liquidity requirements."""
        adv_usd = self._calculate_adv_usd(price_data)
        if adv_usd < self.config.min_adv_usd:
            return False, f"ADV ${adv_usd:,.0f} below min ${self.config.min_adv_usd:,.0f}"
        return True, ""

    def _check_earnings_gate(self, signal: Dict) -> Tuple[bool, str]:
        """Check earnings blackout period."""
        try:
            from core.earnings_filter import is_near_earnings
            symbol = signal.get('symbol', '')
            signal_date = signal.get('timestamp', datetime.now())
            if is_near_earnings(symbol, signal_date):
                return False, "Within earnings blackout period"
        except ImportError:
            pass  # Earnings filter not available, skip check
        return True, ""

    # === Scoring Component Methods ===

    def _calculate_conviction_component(
        self,
        signal: Dict,
        price_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame],
        vix_level: Optional[float],
    ) -> float:
        """Calculate conviction component (0-30 points)."""
        if self.conviction_scorer is None:
            # Fallback: use technical score
            return self._calculate_strategy_component(signal) * 2

        try:
            breakdown = self.conviction_scorer.calculate_conviction(
                signal, price_data, spy_data, vix_level
            )
            return (breakdown.total_score / 100) * self.config.conviction_max
        except Exception as e:
            logger.debug(f"Conviction scoring failed: {e}")
            return self._calculate_strategy_component(signal) * 2

    def _calculate_ml_confidence_component(
        self,
        signal: Dict,
        price_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame],
        vix_level: Optional[float],
    ) -> float:
        """
        Calculate ML confidence component (0-25 points).

        Optionally applies:
        - Probability calibration (isotonic/platt) if ml.calibration.enabled
        - Uncertainty adjustment (conformal) if ml.conformal.enabled
        """
        if self.confidence_integrator is None:
            # Fallback: use technical score
            return self._calculate_strategy_component(signal) * 1.67

        try:
            # Get raw ML confidence
            ml_conf = self.confidence_integrator.get_simple_confidence(
                signal, price_data, spy_data, vix_level
            )

            # Apply probability calibration if enabled
            ml_conf = self._apply_calibration(ml_conf)

            # Apply uncertainty adjustment if enabled (scales down when uncertain)
            uncertainty_mult = self._get_uncertainty_adjustment(ml_conf)
            ml_conf = ml_conf * uncertainty_mult

            return ml_conf * self.config.ml_confidence_max
        except Exception as e:
            logger.debug(f"ML confidence scoring failed: {e}")
            return self._calculate_strategy_component(signal) * 1.67

    def _calculate_strategy_component(self, signal: Dict) -> float:
        """Calculate strategy score component (0-15 points)."""
        raw_score = signal.get('score', 0)
        strategy = signal.get('strategy', '')

        # Normalize based on strategy type
        if strategy == 'TurtleSoup':
            # Turtle Soup scores: sweep_strength * 100, typically 100-300
            normalized = min(raw_score / 300, 1.0)
        else:  # IBS_RSI
            # IBS+RSI scores: (0.08 - ibs) * 100 + (5.0 - rsi), typically 5-20
            normalized = min(raw_score / 25, 1.0)

        return normalized * self.config.strategy_max

    def _calculate_regime_component(
        self,
        signal: Dict,
        spy_data: Optional[pd.DataFrame]
    ) -> float:
        """Calculate regime alignment component (0-15 points)."""
        if spy_data is None or spy_data.empty:
            return self.config.regime_max * 0.5  # Neutral

        try:
            # Simple regime detection: price vs SMA(50)
            close = spy_data['close'].iloc[-1] if 'close' in spy_data.columns else None
            sma50 = spy_data['close'].rolling(50).mean().iloc[-1] if close else None

            if close is None or sma50 is None:
                return self.config.regime_max * 0.5

            is_bullish = close > sma50
            strategy = signal.get('strategy', '')

            # Strategy-regime alignment
            if is_bullish:
                # Bullish: IBS+RSI preferred (mean reversion works in uptrends)
                alignment = 0.8 if strategy == 'IBS_RSI' else 0.5
            else:
                # Bearish: Turtle Soup preferred (liquidity sweeps)
                alignment = 0.7 if strategy == 'TurtleSoup' else 0.4

            return alignment * self.config.regime_max

        except Exception as e:
            logger.debug(f"Regime calculation failed: {e}")
            return self.config.regime_max * 0.5

    def _calculate_liquidity_component(
        self,
        signal: Dict,
        price_data: pd.DataFrame
    ) -> float:
        """Calculate liquidity score component (0-15 points)."""
        adv_usd = self._calculate_adv_usd(price_data)

        if adv_usd >= self.config.preferred_adv_usd:
            return self.config.liquidity_max
        elif adv_usd >= self.config.min_adv_usd:
            ratio = (adv_usd - self.config.min_adv_usd) / (
                self.config.preferred_adv_usd - self.config.min_adv_usd
            )
            return (0.5 + 0.5 * ratio) * self.config.liquidity_max
        else:
            return 0.0

    # === Penalty Methods ===

    def _calculate_correlation_penalty(
        self,
        signal: Dict,
        current_positions: List[Dict],
        price_data: pd.DataFrame,
    ) -> float:
        """Calculate correlation penalty (0-10 points) based on returns correlation.

        Penalizes signals that are highly correlated with existing positions,
        which would increase portfolio concentration risk.

        Args:
            signal: Signal dict with 'symbol' key.
            current_positions: List of position dicts with 'symbol' keys.
            price_data: DataFrame with 'symbol', 'timestamp', and 'close' columns.

        Returns:
            Penalty from 0-10 based on maximum correlation with positions.
        """
        if not current_positions:
            return 0.0

        symbol = signal.get('symbol', '')

        # If same symbol already in positions, maximum penalty
        for pos in current_positions:
            if pos.get('symbol') == symbol:
                return 10.0

        # Calculate returns-based correlation with existing positions
        try:
            if price_data.empty or 'symbol' not in price_data.columns:
                return 0.0

            # Get signal symbol's returns (last 60 days)
            signal_data = price_data[price_data['symbol'] == symbol].copy()
            if len(signal_data) < 20:
                return 0.0

            close_col = 'close' if 'close' in signal_data.columns else 'Close'
            signal_data = signal_data.sort_values('timestamp').tail(60)
            signal_returns = signal_data[close_col].pct_change().dropna()

            if len(signal_returns) < 15:
                return 0.0

            # Calculate correlation with each position
            max_corr = 0.0
            for pos in current_positions:
                pos_symbol = pos.get('symbol', '')
                if pos_symbol == symbol:
                    continue

                pos_data = price_data[price_data['symbol'] == pos_symbol].copy()
                if len(pos_data) < 20:
                    continue

                pos_data = pos_data.sort_values('timestamp').tail(60)
                pos_returns = pos_data[close_col].pct_change().dropna()

                if len(pos_returns) < 15:
                    continue

                # Align by index length (simple approach)
                min_len = min(len(signal_returns), len(pos_returns))
                if min_len < 15:
                    continue

                corr = np.corrcoef(
                    signal_returns.values[-min_len:],
                    pos_returns.values[-min_len:]
                )[0, 1]

                if not np.isnan(corr):
                    max_corr = max(max_corr, abs(corr))

            # Apply penalty based on correlation level
            # corr >= 0.8: full penalty (10)
            # corr >= 0.6: moderate penalty (5)
            # corr >= 0.4: light penalty (2)
            # corr < 0.4: no penalty
            if max_corr >= 0.8:
                return 10.0
            elif max_corr >= 0.6:
                return 5.0
            elif max_corr >= 0.4:
                return 2.0
            else:
                return 0.0

        except Exception as e:
            logger.debug(f"Correlation calculation failed: {e}")
            return 0.0

    def _calculate_timing_penalty(self, signal: Dict) -> float:
        """Calculate timing penalty (0-10 points)."""
        penalty = 0.0

        # Check earnings proximity
        try:
            from core.earnings_filter import is_near_earnings
            symbol = signal.get('symbol', '')
            signal_date = signal.get('timestamp', datetime.now())
            if is_near_earnings(symbol, signal_date):
                penalty += 5.0
        except ImportError:
            pass

        # Day of week penalty
        signal_date = signal.get('timestamp', datetime.now())
        if isinstance(signal_date, datetime):
            dow = signal_date.weekday()
            if dow == 0:  # Monday
                penalty += self.config.monday_penalty
            elif dow == 4:  # Friday
                penalty += self.config.friday_penalty

        return min(penalty, 10.0)

    def _calculate_volatility_penalty(self, vix_level: Optional[float]) -> float:
        """Calculate volatility penalty (0-5 points)."""
        if vix_level is None:
            return 0.0

        if vix_level > self.config.vix_high_threshold:
            return 5.0
        elif vix_level > self.config.vix_elevated_threshold:
            return 2.5

        return 0.0

    # === Helper Methods ===

    def _calculate_adv_usd(self, price_data: pd.DataFrame) -> float:
        """Calculate 60-day average daily volume in USD."""
        if price_data.empty or len(price_data) < 20:
            return 0.0
        try:
            close_col = 'close' if 'close' in price_data.columns else 'Close'
            vol_col = 'volume' if 'volume' in price_data.columns else 'Volume'
            if close_col not in price_data.columns or vol_col not in price_data.columns:
                return 0.0
            usd_volume = price_data[close_col].astype(float) * price_data[vol_col].astype(float)
            return float(usd_volume.tail(60).mean())
        except Exception:
            return 0.0

    def _get_tier(self, score: float) -> QualityTier:
        """Determine quality tier from score."""
        if score >= self.config.elite_threshold:
            return QualityTier.ELITE
        elif score >= self.config.excellent_threshold:
            return QualityTier.EXCELLENT
        elif score >= self.config.good_threshold:
            return QualityTier.GOOD
        elif score >= self.config.marginal_threshold:
            return QualityTier.MARGINAL
        else:
            return QualityTier.REJECT


# ============================================================================
# Singleton Instance
# ============================================================================

_quality_gate: Optional[SignalQualityGate] = None


def get_quality_gate(config: Optional[QualityGateConfig] = None) -> SignalQualityGate:
    """Get or create singleton SignalQualityGate."""
    global _quality_gate
    if _quality_gate is None or config is not None:
        _quality_gate = SignalQualityGate(config)
    return _quality_gate


def filter_to_best_signals(
    signals: pd.DataFrame,
    price_data: pd.DataFrame,
    spy_data: Optional[pd.DataFrame] = None,
    vix_level: Optional[float] = None,
    current_positions: Optional[List[Dict]] = None,
    max_signals: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience function to filter signals.

    Example:
        best = filter_to_best_signals(
            signals=raw_signals,
            price_data=combined,
            spy_data=spy,
            max_signals=1,
        )
    """
    gate = get_quality_gate()
    return gate.filter_signals(
        signals=signals,
        price_data=price_data,
        spy_data=spy_data,
        vix_level=vix_level,
        current_positions=current_positions,
        max_signals=max_signals,
    )
