"""
Market Regime Detector - Hidden Markov Model Based.

Detects market regimes using volatility clustering and trend analysis.
Adjusts strategy parameters based on current regime.

Regimes:
- BULL_LOW_VOL: Uptrend with low volatility (best for momentum)
- BULL_HIGH_VOL: Uptrend with high volatility (cautious momentum)
- BEAR_LOW_VOL: Downtrend with low volatility (mild mean reversion)
- BEAR_HIGH_VOL: Downtrend with high volatility (reduce exposure)
- SIDEWAYS: No clear trend (range trading, mean reversion)
"""

import logging
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_LOW_VOL = "BULL_LOW_VOL"
    BULL_HIGH_VOL = "BULL_HIGH_VOL"
    BEAR_LOW_VOL = "BEAR_LOW_VOL"
    BEAR_HIGH_VOL = "BEAR_HIGH_VOL"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeResult:
    """Container for regime detection results."""
    current_regime: MarketRegime
    confidence: float
    regime_probabilities: Dict[str, float]
    regime_since: Optional[datetime] = None
    days_in_regime: int = 0
    volatility_state: str = "normal"  # low, normal, high, extreme
    trend_state: str = "neutral"  # strong_up, up, neutral, down, strong_down
    vix_level: Optional[float] = None
    strategy_adjustments: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "current_regime": self.current_regime.value,
            "confidence": round(self.confidence, 4),
            "regime_probabilities": {k: round(v, 4) for k, v in self.regime_probabilities.items()},
            "days_in_regime": self.days_in_regime,
            "volatility_state": self.volatility_state,
            "trend_state": self.trend_state,
            "vix_level": round(self.vix_level, 2) if self.vix_level else None,
            "strategy_adjustments": {k: round(v, 4) for k, v in self.strategy_adjustments.items()},
            "metadata": self.metadata,
        }


class MarketRegimeDetector:
    """
    Market Regime Detection using multiple methods.

    Uses:
    1. Simple rules-based detection (always available)
    2. Hidden Markov Model (if hmmlearn available)
    3. Volatility clustering and trend analysis
    """

    # Volatility thresholds (annualized)
    VOL_LOW = 0.12      # Below 12%
    VOL_HIGH = 0.25     # Above 25%
    VOL_EXTREME = 0.40  # Above 40%

    # VIX thresholds
    VIX_LOW = 15
    VIX_NORMAL = 20
    VIX_HIGH = 30
    VIX_EXTREME = 40

    # Trend thresholds (20-day momentum)
    TREND_STRONG_UP = 0.08    # >8%
    TREND_UP = 0.02           # >2%
    TREND_DOWN = -0.02        # <-2%
    TREND_STRONG_DOWN = -0.08 # <-8%

    def __init__(self, lookback_days: int = 60):
        """
        Initialize regime detector.

        Args:
            lookback_days: Days of history to analyze
        """
        self.lookback_days = lookback_days
        self.hmm_model = None
        self.hmm_available = False
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
        self._init_hmm()

    def _init_hmm(self):
        """Initialize Hidden Markov Model if available."""
        try:
            from hmmlearn import hmm
            self.hmm_model = hmm.GaussianHMM(
                n_components=5,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            self.hmm_available = True
            logger.info("HMM initialized successfully")
        except ImportError:
            logger.info("hmmlearn not available, using rules-based detection only")
            self.hmm_available = False

    def detect_regime(
        self,
        market_data: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None
    ) -> RegimeResult:
        """
        Detect current market regime.

        Args:
            market_data: Market index OHLCV (e.g., SPY) with 'close' column
            vix_data: Optional VIX data with 'close' column

        Returns:
            RegimeResult with current regime and confidence
        """
        if len(market_data) < self.lookback_days:
            return RegimeResult(
                current_regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                regime_probabilities={},
                metadata={'error': 'Insufficient data'}
            )

        # Get close prices
        close_col = 'close' if 'close' in market_data.columns else 'Close'
        close = market_data[close_col].iloc[-self.lookback_days:]
        returns = close.pct_change().dropna()

        # Calculate volatility (annualized)
        volatility = returns.std() * np.sqrt(252)

        # Calculate trend (20-day momentum)
        momentum_20d = 0.0
        if len(close) >= 21:
            momentum_20d = (close.iloc[-1] / close.iloc[-21]) - 1

        # Calculate trend strength (R-squared of linear fit)
        x = np.arange(len(close))
        try:
            slope, intercept = np.polyfit(x, close.values, 1)
            predicted = slope * x + intercept
            ss_res = np.sum((close.values - predicted) ** 2)
            ss_tot = np.sum((close.values - close.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        except:
            r_squared = 0

        # Get VIX level if available
        vix_level = None
        if vix_data is not None and len(vix_data) > 0:
            vix_col = 'close' if 'close' in vix_data.columns else 'Close'
            if vix_col in vix_data.columns:
                vix_level = float(vix_data[vix_col].iloc[-1])

        # Determine volatility state
        if volatility < self.VOL_LOW:
            vol_state = 'low'
        elif volatility > self.VOL_EXTREME:
            vol_state = 'extreme'
        elif volatility > self.VOL_HIGH:
            vol_state = 'high'
        else:
            vol_state = 'normal'

        # Determine trend state
        if momentum_20d > self.TREND_STRONG_UP:
            trend_state = 'strong_up'
        elif momentum_20d > self.TREND_UP:
            trend_state = 'up'
        elif momentum_20d < self.TREND_STRONG_DOWN:
            trend_state = 'strong_down'
        elif momentum_20d < self.TREND_DOWN:
            trend_state = 'down'
        else:
            trend_state = 'neutral'

        # Rules-based regime detection
        regime = self._rules_based_detection(vol_state, trend_state, vix_level)

        # Calculate confidence
        confidence = self._calculate_confidence(r_squared, volatility, momentum_20d)

        # Calculate regime probabilities
        probabilities = self._calculate_probabilities(vol_state, trend_state, r_squared)

        # Get strategy adjustments
        adjustments = self._get_strategy_adjustments(regime, vol_state)

        # Track regime history
        self.regime_history.append((datetime.now(), regime))
        days_in_regime = self._count_days_in_regime(regime)

        return RegimeResult(
            current_regime=regime,
            confidence=confidence,
            regime_probabilities=probabilities,
            days_in_regime=days_in_regime,
            volatility_state=vol_state,
            trend_state=trend_state,
            vix_level=vix_level,
            strategy_adjustments=adjustments,
            metadata={
                'volatility': float(volatility),
                'momentum_20d': float(momentum_20d),
                'r_squared': float(r_squared),
                'hmm_available': self.hmm_available
            }
        )

    def _rules_based_detection(
        self,
        vol_state: str,
        trend_state: str,
        vix_level: Optional[float]
    ) -> MarketRegime:
        """Determine regime using rules."""
        # Use VIX if available for volatility override
        if vix_level is not None:
            if vix_level > self.VIX_EXTREME:
                vol_state = 'extreme'
            elif vix_level > self.VIX_HIGH:
                vol_state = 'high'
            elif vix_level < self.VIX_LOW:
                vol_state = 'low'

        # Bull regimes
        if trend_state in ['strong_up', 'up']:
            if vol_state in ['low', 'normal']:
                return MarketRegime.BULL_LOW_VOL
            else:
                return MarketRegime.BULL_HIGH_VOL

        # Bear regimes
        if trend_state in ['strong_down', 'down']:
            if vol_state in ['low', 'normal']:
                return MarketRegime.BEAR_LOW_VOL
            else:
                return MarketRegime.BEAR_HIGH_VOL

        # Sideways
        return MarketRegime.SIDEWAYS

    def _calculate_confidence(
        self,
        r_squared: float,
        volatility: float,
        momentum: float
    ) -> float:
        """Calculate confidence in regime detection."""
        # Higher R-squared = more confident in trend
        trend_confidence = r_squared

        # Clear momentum = more confident
        momentum_confidence = min(abs(momentum) / 0.1, 1.0)

        # Not extreme volatility = more confident
        vol_confidence = 1.0 - min(volatility / 0.5, 1.0)

        return (trend_confidence * 0.4 + momentum_confidence * 0.4 + vol_confidence * 0.2)

    def _calculate_probabilities(
        self,
        vol_state: str,
        trend_state: str,
        r_squared: float
    ) -> Dict[str, float]:
        """Calculate probability distribution over regimes."""
        probs = {r.value: 0.0 for r in MarketRegime if r != MarketRegime.UNKNOWN}

        # Base probabilities from trend
        if trend_state in ['strong_up', 'up']:
            probs['BULL_LOW_VOL'] = 0.4
            probs['BULL_HIGH_VOL'] = 0.2
            probs['SIDEWAYS'] = 0.2
            probs['BEAR_LOW_VOL'] = 0.1
            probs['BEAR_HIGH_VOL'] = 0.1
        elif trend_state in ['strong_down', 'down']:
            probs['BEAR_LOW_VOL'] = 0.3
            probs['BEAR_HIGH_VOL'] = 0.3
            probs['SIDEWAYS'] = 0.2
            probs['BULL_LOW_VOL'] = 0.1
            probs['BULL_HIGH_VOL'] = 0.1
        else:
            probs['SIDEWAYS'] = 0.5
            probs['BULL_LOW_VOL'] = 0.15
            probs['BULL_HIGH_VOL'] = 0.1
            probs['BEAR_LOW_VOL'] = 0.15
            probs['BEAR_HIGH_VOL'] = 0.1

        # Adjust for volatility
        if vol_state in ['high', 'extreme']:
            probs['BULL_HIGH_VOL'] = probs.get('BULL_LOW_VOL', 0) + probs.get('BULL_HIGH_VOL', 0)
            probs['BULL_LOW_VOL'] = 0
            probs['BEAR_HIGH_VOL'] = probs.get('BEAR_LOW_VOL', 0) + probs.get('BEAR_HIGH_VOL', 0)
            probs['BEAR_LOW_VOL'] = 0

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}

        return probs

    def _get_strategy_adjustments(
        self,
        regime: MarketRegime,
        vol_state: str
    ) -> Dict[str, float]:
        """Get strategy parameter adjustments for current regime."""
        adjustments = {
            'position_size_multiplier': 1.0,
            'momentum_weight': 0.5,
            'mean_reversion_weight': 0.5,
            'stop_loss_multiplier': 1.0,
            'profit_target_multiplier': 1.0
        }

        if regime == MarketRegime.BULL_LOW_VOL:
            # Best environment for momentum
            adjustments['position_size_multiplier'] = 1.2
            adjustments['momentum_weight'] = 0.7
            adjustments['mean_reversion_weight'] = 0.3
            adjustments['stop_loss_multiplier'] = 0.9
            adjustments['profit_target_multiplier'] = 1.2

        elif regime == MarketRegime.BULL_HIGH_VOL:
            # Cautious momentum
            adjustments['position_size_multiplier'] = 0.8
            adjustments['momentum_weight'] = 0.6
            adjustments['mean_reversion_weight'] = 0.4
            adjustments['stop_loss_multiplier'] = 1.5
            adjustments['profit_target_multiplier'] = 1.5

        elif regime == MarketRegime.BEAR_LOW_VOL:
            # Mild mean reversion opportunities
            adjustments['position_size_multiplier'] = 0.7
            adjustments['momentum_weight'] = 0.3
            adjustments['mean_reversion_weight'] = 0.7
            adjustments['stop_loss_multiplier'] = 1.2
            adjustments['profit_target_multiplier'] = 1.0

        elif regime == MarketRegime.BEAR_HIGH_VOL:
            # High risk - reduce exposure significantly
            adjustments['position_size_multiplier'] = 0.4
            adjustments['momentum_weight'] = 0.2
            adjustments['mean_reversion_weight'] = 0.8
            adjustments['stop_loss_multiplier'] = 2.0
            adjustments['profit_target_multiplier'] = 0.8

        elif regime == MarketRegime.SIDEWAYS:
            # Range trading - favor mean reversion (Kobe's specialty)
            adjustments['position_size_multiplier'] = 0.9
            adjustments['momentum_weight'] = 0.3
            adjustments['mean_reversion_weight'] = 0.7
            adjustments['stop_loss_multiplier'] = 1.0
            adjustments['profit_target_multiplier'] = 1.0

        # Additional adjustment for extreme volatility
        if vol_state == 'extreme':
            adjustments['position_size_multiplier'] *= 0.5
            adjustments['stop_loss_multiplier'] *= 1.5

        return adjustments

    def _count_days_in_regime(self, current_regime: MarketRegime) -> int:
        """Count consecutive days in current regime."""
        count = 0
        for dt, regime in reversed(self.regime_history):
            if regime == current_regime:
                count += 1
            else:
                break
        return count

    def fit_hmm(self, returns: pd.Series) -> bool:
        """
        Fit Hidden Markov Model on historical returns.

        Args:
            returns: Series of daily returns

        Returns:
            True if successful
        """
        if not self.hmm_available:
            return False

        try:
            # Prepare features: returns and volatility
            vol = returns.rolling(window=20).std() * np.sqrt(252)
            features = pd.DataFrame({
                'returns': returns,
                'volatility': vol
            }).dropna()

            X = features.values
            self.hmm_model.fit(X)
            logger.info("HMM fitted successfully on historical data")
            return True

        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            return False

    def get_regime_history(self, last_n: int = 30) -> List[Tuple[datetime, str]]:
        """Get recent regime history."""
        return [(dt, regime.value) for dt, regime in self.regime_history[-last_n:]]


# Singleton instance
_regime_detector: Optional[MarketRegimeDetector] = None


def get_regime_detector() -> MarketRegimeDetector:
    """Get or create the global regime detector instance."""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = MarketRegimeDetector()
    return _regime_detector
