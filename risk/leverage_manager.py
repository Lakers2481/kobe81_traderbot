"""
Leverage Manager - Renaissance-Inspired Dynamic Leverage

Adjusts leverage based on market regime and confidence, inspired by
Renaissance Technologies' approach of using leverage on diversified portfolios.

Key Principles:
1. Leverage ONLY with diversification (20+ positions)
2. Reduce leverage in BEARISH regimes
3. Scale with confidence
4. Never exceed safety limits

Renaissance Context:
- Medallion used 12.5-20x leverage
- BUT with 3,500+ positions for diversification
- Kobe uses 2-3x leverage with 20 positions (scaled down proportionally)

SAFETY: This module DOES NOT automatically enable leverage.
        User must explicitly enable via config or CLI flag.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LeverageMode(Enum):
    """Leverage operation modes."""
    DISABLED = "disabled"      # 1x only (default, safe)
    CONSERVATIVE = "conservative"  # Up to 2x
    MODERATE = "moderate"      # Up to 3x
    AGGRESSIVE = "aggressive"  # Up to 5x (Renaissance-lite)


@dataclass
class LeverageConfig:
    """Configuration for leverage management."""
    # Mode
    mode: LeverageMode = LeverageMode.DISABLED

    # Leverage limits by mode
    max_leverage_by_mode: Dict[LeverageMode, float] = None

    # Regime multipliers (reduce leverage in bad regimes)
    regime_multipliers: Dict[str, float] = None

    # Minimum positions required to use leverage
    min_positions_for_leverage: int = 10

    # Minimum diversification score (ENP)
    min_diversification: float = 5.0

    # VIX thresholds (reduce leverage when fear is high)
    vix_reduce_threshold: float = 25.0
    vix_halt_threshold: float = 35.0

    # Drawdown limits (reduce leverage during drawdown)
    drawdown_reduce_threshold: float = 0.05  # 5% drawdown
    drawdown_halt_threshold: float = 0.10    # 10% drawdown

    def __post_init__(self):
        if self.max_leverage_by_mode is None:
            self.max_leverage_by_mode = {
                LeverageMode.DISABLED: 1.0,
                LeverageMode.CONSERVATIVE: 2.0,
                LeverageMode.MODERATE: 3.0,
                LeverageMode.AGGRESSIVE: 5.0,
            }

        if self.regime_multipliers is None:
            self.regime_multipliers = {
                'BULLISH': 1.2,   # Can use more leverage in bull
                'BULL': 1.2,
                'NEUTRAL': 1.0,  # Normal leverage
                'BEARISH': 0.5,  # Reduce leverage in bear
                'BEAR': 0.5,
            }


@dataclass
class LeverageResult:
    """Result of leverage calculation."""
    leverage: float
    mode: LeverageMode
    regime_adjustment: float
    vix_adjustment: float
    drawdown_adjustment: float
    confidence_adjustment: float
    diversification_ok: bool
    positions_ok: bool
    final_multiplier: float
    reason: str

    def to_dict(self) -> Dict:
        return {
            'leverage': round(self.leverage, 2),
            'mode': self.mode.value,
            'regime_adjustment': round(self.regime_adjustment, 2),
            'vix_adjustment': round(self.vix_adjustment, 2),
            'drawdown_adjustment': round(self.drawdown_adjustment, 2),
            'confidence_adjustment': round(self.confidence_adjustment, 2),
            'diversification_ok': self.diversification_ok,
            'positions_ok': self.positions_ok,
            'final_multiplier': round(self.final_multiplier, 2),
            'reason': self.reason,
        }


class LeverageManager:
    """
    Dynamic leverage manager with safety controls.

    Calculates appropriate leverage based on:
    - Market regime (bull/neutral/bear)
    - VIX level (fear index)
    - Portfolio diversification
    - Current drawdown
    - Signal confidence
    """

    def __init__(self, config: Optional[LeverageConfig] = None):
        self.config = config or LeverageConfig()
        self._current_drawdown = 0.0
        self._peak_equity = 0.0

        logger.info(f"LeverageManager initialized: mode={self.config.mode.value}")

    def calculate_leverage(
        self,
        regime: str = 'NEUTRAL',
        regime_confidence: float = 0.5,
        vix_level: float = 20.0,
        current_positions: int = 0,
        effective_positions: float = 0.0,  # ENP
        current_equity: float = 0.0,
        signal_confidence: float = 0.5,
    ) -> LeverageResult:
        """
        Calculate appropriate leverage for current conditions.

        Args:
            regime: Market regime (BULLISH, NEUTRAL, BEARISH)
            regime_confidence: Confidence in regime detection (0-1)
            vix_level: Current VIX level
            current_positions: Number of open positions
            effective_positions: Effective number of positions (ENP)
            current_equity: Current account equity
            signal_confidence: Confidence in signals (0-1)

        Returns:
            LeverageResult with recommended leverage and reasoning
        """
        # Start with mode max
        max_leverage = self.config.max_leverage_by_mode.get(
            self.config.mode, 1.0
        )

        # If disabled, always return 1.0
        if self.config.mode == LeverageMode.DISABLED:
            return LeverageResult(
                leverage=1.0,
                mode=self.config.mode,
                regime_adjustment=1.0,
                vix_adjustment=1.0,
                drawdown_adjustment=1.0,
                confidence_adjustment=1.0,
                diversification_ok=True,
                positions_ok=True,
                final_multiplier=1.0,
                reason="Leverage disabled"
            )

        # Check position count
        positions_ok = current_positions >= self.config.min_positions_for_leverage
        if not positions_ok and max_leverage > 1.0:
            return LeverageResult(
                leverage=1.0,
                mode=self.config.mode,
                regime_adjustment=1.0,
                vix_adjustment=1.0,
                drawdown_adjustment=1.0,
                confidence_adjustment=1.0,
                diversification_ok=False,
                positions_ok=False,
                final_multiplier=1.0,
                reason=f"Need {self.config.min_positions_for_leverage}+ positions for leverage, have {current_positions}"
            )

        # Check diversification (ENP)
        diversification_ok = effective_positions >= self.config.min_diversification
        if not diversification_ok and max_leverage > 1.0:
            return LeverageResult(
                leverage=1.0,
                mode=self.config.mode,
                regime_adjustment=1.0,
                vix_adjustment=1.0,
                drawdown_adjustment=1.0,
                confidence_adjustment=1.0,
                diversification_ok=False,
                positions_ok=positions_ok,
                final_multiplier=1.0,
                reason=f"Need ENP >= {self.config.min_diversification}, have {effective_positions:.1f}"
            )

        # === ADJUSTMENT 1: Regime ===
        regime_upper = str(regime).upper()
        regime_mult = self.config.regime_multipliers.get(regime_upper, 1.0)
        # Scale by regime confidence
        regime_adjustment = 1.0 + (regime_mult - 1.0) * regime_confidence

        # === ADJUSTMENT 2: VIX ===
        if vix_level >= self.config.vix_halt_threshold:
            vix_adjustment = 0.0  # No leverage in extreme fear
        elif vix_level >= self.config.vix_reduce_threshold:
            # Linear reduction from 1.0 to 0.5 as VIX goes from threshold to halt
            reduction_range = self.config.vix_halt_threshold - self.config.vix_reduce_threshold
            reduction = (vix_level - self.config.vix_reduce_threshold) / reduction_range
            vix_adjustment = 1.0 - (reduction * 0.5)
        else:
            vix_adjustment = 1.0

        # === ADJUSTMENT 3: Drawdown ===
        drawdown = self._calculate_drawdown(current_equity)
        if drawdown >= self.config.drawdown_halt_threshold:
            drawdown_adjustment = 0.0  # No leverage during significant drawdown
        elif drawdown >= self.config.drawdown_reduce_threshold:
            # Linear reduction
            reduction_range = self.config.drawdown_halt_threshold - self.config.drawdown_reduce_threshold
            reduction = (drawdown - self.config.drawdown_reduce_threshold) / reduction_range
            drawdown_adjustment = 1.0 - (reduction * 0.5)
        else:
            drawdown_adjustment = 1.0

        # === ADJUSTMENT 4: Confidence ===
        # Scale leverage with signal confidence (0.5 to 1.0 range)
        confidence_adjustment = 0.5 + (signal_confidence * 0.5)

        # === FINAL CALCULATION ===
        final_multiplier = (
            regime_adjustment *
            vix_adjustment *
            drawdown_adjustment *
            confidence_adjustment
        )

        # Calculate final leverage (capped at max)
        leverage = min(max_leverage * final_multiplier, max_leverage)
        leverage = max(1.0, leverage)  # Never below 1.0

        # Build reason
        reasons = []
        if regime_adjustment != 1.0:
            reasons.append(f"regime:{regime_upper}={regime_adjustment:.2f}")
        if vix_adjustment != 1.0:
            reasons.append(f"vix:{vix_level:.0f}={vix_adjustment:.2f}")
        if drawdown_adjustment != 1.0:
            reasons.append(f"dd:{drawdown:.1%}={drawdown_adjustment:.2f}")
        if confidence_adjustment != 1.0:
            reasons.append(f"conf:{signal_confidence:.2f}={confidence_adjustment:.2f}")

        reason = f"Leverage {leverage:.1f}x" + (f" ({', '.join(reasons)})" if reasons else "")

        return LeverageResult(
            leverage=leverage,
            mode=self.config.mode,
            regime_adjustment=regime_adjustment,
            vix_adjustment=vix_adjustment,
            drawdown_adjustment=drawdown_adjustment,
            confidence_adjustment=confidence_adjustment,
            diversification_ok=diversification_ok,
            positions_ok=positions_ok,
            final_multiplier=final_multiplier,
            reason=reason
        )

    def _calculate_drawdown(self, current_equity: float) -> float:
        """Calculate current drawdown from peak."""
        if current_equity <= 0:
            return 0.0

        # Update peak
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        if self._peak_equity <= 0:
            return 0.0

        self._current_drawdown = (self._peak_equity - current_equity) / self._peak_equity
        return self._current_drawdown

    def apply_leverage(
        self,
        base_position_value: float,
        leverage_result: LeverageResult,
    ) -> Tuple[float, str]:
        """
        Apply leverage to position value.

        Args:
            base_position_value: Base position value (without leverage)
            leverage_result: Result from calculate_leverage()

        Returns:
            Tuple of (leveraged_value, reason_string)
        """
        leveraged_value = base_position_value * leverage_result.leverage
        return leveraged_value, leverage_result.reason

    def get_status(self) -> Dict:
        """Get current leverage manager status."""
        return {
            'mode': self.config.mode.value,
            'max_leverage': self.config.max_leverage_by_mode.get(self.config.mode, 1.0),
            'min_positions': self.config.min_positions_for_leverage,
            'min_diversification': self.config.min_diversification,
            'vix_thresholds': {
                'reduce': self.config.vix_reduce_threshold,
                'halt': self.config.vix_halt_threshold,
            },
            'drawdown_thresholds': {
                'reduce': self.config.drawdown_reduce_threshold,
                'halt': self.config.drawdown_halt_threshold,
            },
            'current_drawdown': self._current_drawdown,
            'peak_equity': self._peak_equity,
        }


# Factory function for easy access
_leverage_manager: Optional[LeverageManager] = None


def get_leverage_manager(mode: str = 'disabled') -> LeverageManager:
    """
    Get or create leverage manager singleton.

    Args:
        mode: One of 'disabled', 'conservative', 'moderate', 'aggressive'

    Returns:
        LeverageManager instance
    """
    global _leverage_manager

    mode_enum = {
        'disabled': LeverageMode.DISABLED,
        'conservative': LeverageMode.CONSERVATIVE,
        'moderate': LeverageMode.MODERATE,
        'aggressive': LeverageMode.AGGRESSIVE,
    }.get(mode.lower(), LeverageMode.DISABLED)

    if _leverage_manager is None or _leverage_manager.config.mode != mode_enum:
        config = LeverageConfig(mode=mode_enum)
        _leverage_manager = LeverageManager(config)

    return _leverage_manager


# Quick test
if __name__ == '__main__':
    # Test leverage calculations
    print("=" * 60)
    print("LEVERAGE MANAGER TEST")
    print("=" * 60)

    # Test conservative mode
    mgr = get_leverage_manager('conservative')

    scenarios = [
        {'regime': 'BULLISH', 'vix': 15, 'positions': 20, 'enp': 15, 'equity': 100000, 'conf': 0.8},
        {'regime': 'NEUTRAL', 'vix': 20, 'positions': 20, 'enp': 15, 'equity': 100000, 'conf': 0.6},
        {'regime': 'BEARISH', 'vix': 30, 'positions': 20, 'enp': 15, 'equity': 100000, 'conf': 0.5},
        {'regime': 'NEUTRAL', 'vix': 40, 'positions': 20, 'enp': 15, 'equity': 100000, 'conf': 0.6},
        {'regime': 'BULLISH', 'vix': 15, 'positions': 5, 'enp': 4, 'equity': 100000, 'conf': 0.8},  # Too few positions
    ]

    for s in scenarios:
        result = mgr.calculate_leverage(
            regime=s['regime'],
            regime_confidence=0.8,
            vix_level=s['vix'],
            current_positions=s['positions'],
            effective_positions=s['enp'],
            current_equity=s['equity'],
            signal_confidence=s['conf'],
        )
        print(f"\n{s['regime']} | VIX={s['vix']} | Pos={s['positions']} | ENP={s['enp']} | Conf={s['conf']}")
        print(f"  -> {result.reason}")
        print(f"  -> Leverage: {result.leverage:.2f}x")
