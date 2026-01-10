"""
Adaptive Strategy Selector
==========================

Uses HMM regime detection to automatically select the best strategy
for current market conditions.

Logic:
- BULL regime -> IBS+RSI Mean Reversion (trend-following)
- BEAR regime -> TurtleSoup (mean-reversion on failed breakdowns)
- NEUTRAL/CHOPPY -> Reduced size or skip trading

Usage:
    from strategies.adaptive_selector import AdaptiveStrategySelector

    selector = AdaptiveStrategySelector()
    strategy, config = selector.get_strategy_for_regime(spy_data)

    signals = strategy.scan_signals_over_time(stock_data)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    CHOPPY = "choppy"
    UNKNOWN = "unknown"


@dataclass
class StrategyConfig:
    """Configuration for strategy based on regime."""
    strategy_name: str
    position_size_multiplier: float  # 1.0 = normal, 0.5 = half size
    take_profit_multiplier: float    # Adjust targets
    stop_loss_multiplier: float      # Adjust stops
    max_positions: int               # Limit concurrent positions
    skip_trading: bool = False       # True if regime is too risky
    notes: str = ""


# Regime-specific configurations
REGIME_CONFIGS = {
    MarketRegime.BULL: StrategyConfig(
        strategy_name="ibs_rsi",
        position_size_multiplier=1.0,
        take_profit_multiplier=1.2,   # Extend targets in trends
        stop_loss_multiplier=1.0,
        max_positions=5,
        notes="Trend-following works best in bull markets"
    ),
    MarketRegime.BEAR: StrategyConfig(
        strategy_name="turtle_soup",
        position_size_multiplier=0.75,  # Slightly reduced
        take_profit_multiplier=0.8,     # Tighter targets
        stop_loss_multiplier=1.2,       # Wider stops for volatility
        max_positions=3,
        notes="Mean reversion on failed breakdowns"
    ),
    MarketRegime.NEUTRAL: StrategyConfig(
        strategy_name="turtle_soup",
        position_size_multiplier=0.5,   # Half size
        take_profit_multiplier=0.7,     # Quick profits
        stop_loss_multiplier=1.0,
        max_positions=2,
        notes="Reduced exposure in directionless market"
    ),
    MarketRegime.CHOPPY: StrategyConfig(
        strategy_name="none",
        position_size_multiplier=0.0,
        take_profit_multiplier=1.0,
        stop_loss_multiplier=1.0,
        max_positions=0,
        skip_trading=True,
        notes="Skip trading in choppy/whipsaw conditions"
    ),
    MarketRegime.UNKNOWN: StrategyConfig(
        strategy_name="turtle_soup",
        position_size_multiplier=0.5,
        take_profit_multiplier=0.8,
        stop_loss_multiplier=1.2,
        max_positions=2,
        notes="Conservative approach when regime unclear"
    ),
}


class AdaptiveStrategySelector:
    """
    Selects and configures strategies based on market regime.

    Uses HMM regime detection to identify market state, then
    returns the appropriate strategy with adjusted parameters.
    """

    def __init__(
        self,
        regime_lookback: int = 60,
        confidence_threshold: float = 0.6,
        use_hmm: bool = True,
        use_simple_regime: bool = True,
    ):
        self.regime_lookback = regime_lookback
        self.confidence_threshold = confidence_threshold
        self.use_hmm = use_hmm
        self.use_simple_regime = use_simple_regime

        # Cache strategies
        self._strategies = {}
        self._current_regime = MarketRegime.UNKNOWN
        self._regime_confidence = 0.0

        logger.info("AdaptiveStrategySelector initialized")

    def _get_strategy_instance(self, name: str):
        """Get or create strategy instance."""
        if name not in self._strategies:
            if name == "ibs_rsi":
                from strategies.ibs_rsi.strategy import IbsRsiStrategy
                self._strategies[name] = IbsRsiStrategy()
            elif name == "turtle_soup":
                from strategies.ict.turtle_soup import TurtleSoupStrategy
                self._strategies[name] = TurtleSoupStrategy()
            else:
                return None
        return self._strategies.get(name)

    def detect_regime_hmm(self, price_data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Detect regime using HMM model.

        Args:
            price_data: DataFrame with OHLCV data (needs 'close' column)

        Returns:
            Tuple of (MarketRegime, confidence)
        """
        try:
            from ml_advanced.hmm_regime_detector import HMMRegimeDetector, HMM_AVAILABLE

            if not HMM_AVAILABLE:
                logger.warning("HMM not available, falling back to simple regime")
                return self.detect_regime_simple(price_data)

            detector = HMMRegimeDetector()

            # Fit on recent data
            if len(price_data) < self.regime_lookback:
                return MarketRegime.UNKNOWN, 0.0

            detector.fit(price_data.tail(self.regime_lookback * 2))
            regime_state = detector.predict_regime(price_data.tail(self.regime_lookback))

            # Map HMM state to our regime enum
            regime_map = {
                'bull': MarketRegime.BULL,
                'bear': MarketRegime.BEAR,
                'neutral': MarketRegime.NEUTRAL,
                'high_volatility': MarketRegime.CHOPPY,
            }

            regime = regime_map.get(regime_state.regime.lower(), MarketRegime.UNKNOWN)
            confidence = regime_state.confidence if hasattr(regime_state, 'confidence') else 0.7

            return regime, confidence

        except Exception as e:
            logger.warning(f"HMM regime detection failed: {e}")
            return self.detect_regime_simple(price_data)

    def detect_regime_simple(self, price_data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Simple regime detection using SMA and volatility.

        Logic:
        - Price > SMA(50) & SMA(50) > SMA(200) -> BULL
        - Price < SMA(50) & SMA(50) < SMA(200) -> BEAR
        - Otherwise -> NEUTRAL
        - High volatility -> CHOPPY
        """
        if len(price_data) < 200:
            return MarketRegime.UNKNOWN, 0.0

        close = price_data['close']

        # Calculate indicators
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        current_price = close.iloc[-1]
        current_sma50 = sma_50.iloc[-1]
        current_sma200 = sma_200.iloc[-1]

        # Volatility check
        returns = close.pct_change().dropna()
        volatility = returns.tail(20).std() * np.sqrt(252)

        # Regime detection
        if volatility > 0.40:  # >40% annualized vol
            return MarketRegime.CHOPPY, 0.8

        if current_price > current_sma50 and current_sma50 > current_sma200:
            # Check trend strength
            trend_strength = (current_price - current_sma200) / current_sma200
            confidence = min(0.5 + trend_strength * 2, 0.95)
            return MarketRegime.BULL, confidence

        elif current_price < current_sma50 and current_sma50 < current_sma200:
            trend_strength = (current_sma200 - current_price) / current_sma200
            confidence = min(0.5 + trend_strength * 2, 0.95)
            return MarketRegime.BEAR, confidence

        else:
            return MarketRegime.NEUTRAL, 0.6

    def detect_regime(self, price_data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime.

        Uses HMM if available and enabled, otherwise falls back to simple method.
        """
        if self.use_hmm:
            regime, confidence = self.detect_regime_hmm(price_data)
        elif self.use_simple_regime:
            regime, confidence = self.detect_regime_simple(price_data)
        else:
            regime, confidence = MarketRegime.UNKNOWN, 0.0

        self._current_regime = regime
        self._regime_confidence = confidence

        logger.info(f"Detected regime: {regime.value} (confidence: {confidence:.2f})")

        return regime, confidence

    def get_strategy_for_regime(
        self,
        market_data: pd.DataFrame,
    ) -> Tuple[Optional[Any], StrategyConfig]:
        """
        Get the appropriate strategy for current market regime.

        Args:
            market_data: DataFrame with market index data (SPY/QQQ)

        Returns:
            Tuple of (strategy_instance, config)
        """
        regime, confidence = self.detect_regime(market_data)

        # Get config for this regime
        config = REGIME_CONFIGS.get(regime, REGIME_CONFIGS[MarketRegime.UNKNOWN])

        # If confidence is low, be more conservative
        if confidence < self.confidence_threshold:
            config = StrategyConfig(
                strategy_name=config.strategy_name,
                position_size_multiplier=config.position_size_multiplier * 0.5,
                take_profit_multiplier=config.take_profit_multiplier,
                stop_loss_multiplier=config.stop_loss_multiplier * 1.2,
                max_positions=max(1, config.max_positions // 2),
                skip_trading=config.skip_trading,
                notes=f"Low confidence ({confidence:.2f}), reduced exposure"
            )

        # Skip trading if configured
        if config.skip_trading:
            logger.info(f"Skipping trading: {config.notes}")
            return None, config

        # Get strategy instance
        strategy = self._get_strategy_instance(config.strategy_name)

        if strategy is None:
            logger.warning(f"Strategy '{config.strategy_name}' not found")
            return None, config

        return strategy, config

    def get_current_regime(self) -> Tuple[MarketRegime, float]:
        """Get the last detected regime."""
        return self._current_regime, self._regime_confidence

    def get_regime_summary(self, market_data: pd.DataFrame) -> Dict:
        """Get detailed regime analysis."""
        regime, confidence = self.detect_regime(market_data)
        config = REGIME_CONFIGS.get(regime, REGIME_CONFIGS[MarketRegime.UNKNOWN])

        return {
            'regime': regime.value,
            'confidence': confidence,
            'strategy': config.strategy_name,
            'position_multiplier': config.position_size_multiplier,
            'max_positions': config.max_positions,
            'skip_trading': config.skip_trading,
            'notes': config.notes,
        }


# Convenience function
def get_adaptive_strategy(market_data: pd.DataFrame) -> Tuple[Optional[Any], StrategyConfig]:
    """
    Quick function to get strategy for current regime.

    Example:
        import yfinance as yf
        spy = yf.download('SPY', period='1y')
        strategy, config = get_adaptive_strategy(spy)

        if strategy:
            signals = strategy.scan_signals_over_time(stock_data)
    """
    selector = AdaptiveStrategySelector()
    return selector.get_strategy_for_regime(market_data)
