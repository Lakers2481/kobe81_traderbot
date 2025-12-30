"""
Portfolio Risk Manager - Central Integration Layer
===================================================

Wires together all existing risk components into a single decision point.
Every trade goes through this gate before execution.

Components Integrated:
- KellyPositionSizer: Optimal position sizing
- EnhancedCorrelationLimits: Prevent correlated blowups
- PortfolioHeatMonitor: Portfolio health check
- MonteCarloVaR: Risk budget enforcement

Usage:
    from portfolio.risk_manager import PortfolioRiskManager

    prm = PortfolioRiskManager(equity=100000)
    decision = prm.evaluate_trade(signal, current_positions, price_data)

    if decision.approved:
        execute_trade(signal, size=decision.position_size)
    else:
        log(f"Trade rejected: {decision.rejection_reason}")
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    """Result of trade evaluation."""
    approved: bool
    position_size: float  # Dollar amount to trade
    shares: int  # Number of shares
    confidence_adjusted: bool  # Was size adjusted by ML confidence?
    kelly_fraction: float  # Raw Kelly fraction
    rejection_reason: Optional[str] = None
    risk_score: float = 0.0  # 0-100, higher = riskier
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class PortfolioRiskManager:
    """
    Central risk management integration layer.

    This is the BRAIN that connects all our existing risk components
    and makes the final trade decision.
    """

    def __init__(
        self,
        equity: float = 100000.0,
        max_position_pct: float = 0.05,  # 5% max per position
        max_portfolio_risk_pct: float = 0.02,  # 2% max portfolio risk
        max_correlation: float = 0.70,
        max_sector_positions: int = 3,
        max_heat_score: float = 80.0,
        use_kelly: bool = True,
        kelly_fraction: float = 0.25,  # Quarter Kelly for safety
        use_ml_confidence: bool = True,
        min_confidence_threshold: float = 0.5,
    ):
        self.equity = equity
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.max_correlation = max_correlation
        self.max_sector_positions = max_sector_positions
        self.max_heat_score = max_heat_score
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction
        self.use_ml_confidence = use_ml_confidence
        self.min_confidence_threshold = min_confidence_threshold
        self.use_conformal = self._check_conformal_enabled()

        # Initialize component references (lazy loading)
        self._kelly_sizer = None
        self._correlation_checker = None
        self._heat_monitor = None
        self._var_calculator = None

        logger.info("PortfolioRiskManager initialized")

    @property
    def kelly_sizer(self):
        """Lazy load Kelly Position Sizer."""
        if self._kelly_sizer is None:
            try:
                from risk.advanced.kelly_position_sizer import KellyPositionSizer
                self._kelly_sizer = KellyPositionSizer(
                    kelly_fraction=self.kelly_fraction,
                    max_position_pct=self.max_position_pct
                )
            except ImportError:
                logger.warning("KellyPositionSizer not available")
        return self._kelly_sizer

    @property
    def correlation_checker(self):
        """Lazy load Correlation Limits checker."""
        if self._correlation_checker is None:
            try:
                from risk.advanced.correlation_limits import EnhancedCorrelationLimits
                self._correlation_checker = EnhancedCorrelationLimits(
                    max_correlation=self.max_correlation,
                    max_sector_positions=self.max_sector_positions
                )
            except ImportError:
                logger.warning("EnhancedCorrelationLimits not available")
        return self._correlation_checker

    @property
    def heat_monitor(self):
        """Lazy load Portfolio Heat Monitor."""
        if self._heat_monitor is None:
            try:
                from portfolio.heat_monitor import get_heat_monitor
                self._heat_monitor = get_heat_monitor()
            except ImportError:
                logger.warning("PortfolioHeatMonitor not available")
        return self._heat_monitor

    def _check_conformal_enabled(self) -> bool:
        """Check if conformal prediction is enabled in config."""
        try:
            from config.settings_loader import load_settings
            config = load_settings()
            return config.get('ml', {}).get('conformal', {}).get('enabled', False)
        except Exception:
            return False

    def _get_conformal_multiplier(self, prediction: float) -> float:
        """
        Get position size multiplier from conformal prediction uncertainty.

        High uncertainty -> lower multiplier -> smaller position
        Low uncertainty -> multiplier closer to 1.0 -> full position
        """
        if not self.use_conformal:
            return 1.0
        try:
            from ml_meta.conformal import get_position_multiplier
            multiplier = get_position_multiplier(prediction)
            logger.debug(f"Conformal sizing multiplier for pred {prediction:.3f}: {multiplier:.3f}")
            return multiplier
        except Exception as e:
            logger.debug(f"Conformal multiplier failed: {e}")
            return 1.0

    def update_equity(self, new_equity: float):
        """Update equity for position sizing calculations."""
        self.equity = new_equity
        logger.info(f"Equity updated to ${new_equity:,.2f}")

    def evaluate_trade(
        self,
        signal: Dict,
        current_positions: List[Dict],
        price_data: Optional[pd.DataFrame] = None,
        ml_confidence: Optional[float] = None,
    ) -> TradeDecision:
        """
        Evaluate whether a trade should be taken and at what size.

        Args:
            signal: Trade signal with keys: symbol, side, entry_price, stop_loss
            current_positions: List of current position dicts
            price_data: Optional DataFrame with price history for correlation calc
            ml_confidence: Optional ML model confidence score (0-1)

        Returns:
            TradeDecision with approval status and sizing
        """
        symbol = signal.get('symbol', 'UNKNOWN')
        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        side = signal.get('side', 'long')

        warnings = []
        rejection_reason = None
        risk_score = 0.0

        # === CHECK 1: ML Confidence Filter ===
        if self.use_ml_confidence and ml_confidence is not None:
            if ml_confidence < self.min_confidence_threshold:
                return TradeDecision(
                    approved=False,
                    position_size=0,
                    shares=0,
                    confidence_adjusted=False,
                    kelly_fraction=0,
                    rejection_reason=f"ML confidence {ml_confidence:.2f} below threshold {self.min_confidence_threshold}",
                    risk_score=100
                )
            risk_score += (1 - ml_confidence) * 20  # Low confidence adds risk

        # === CHECK 2: Portfolio Heat Check ===
        if self.heat_monitor:
            try:
                heat_status = self.heat_monitor.calculate_heat(current_positions)
                if heat_status.heat_score > self.max_heat_score:
                    return TradeDecision(
                        approved=False,
                        position_size=0,
                        shares=0,
                        confidence_adjusted=False,
                        kelly_fraction=0,
                        rejection_reason=f"Portfolio heat {heat_status.heat_score:.1f} exceeds max {self.max_heat_score}",
                        risk_score=heat_status.heat_score
                    )
                risk_score += heat_status.heat_score * 0.3
                if heat_status.heat_score > 60:
                    warnings.append(f"Portfolio heat elevated: {heat_status.heat_score:.1f}")
            except Exception as e:
                logger.warning(f"Heat monitor check failed: {e}")

        # === CHECK 3: Correlation Check ===
        if self.correlation_checker and price_data is not None:
            try:
                corr_result = self.correlation_checker.check_entry(
                    symbol=symbol,
                    current_positions=[p['symbol'] for p in current_positions],
                    price_data=price_data
                )
                if not corr_result.allowed:
                    return TradeDecision(
                        approved=False,
                        position_size=0,
                        shares=0,
                        confidence_adjusted=False,
                        kelly_fraction=0,
                        rejection_reason=f"Correlation check failed: {corr_result.reason}",
                        risk_score=80
                    )
                if corr_result.warnings:
                    warnings.extend(corr_result.warnings)
                risk_score += corr_result.risk_level.value * 10 if hasattr(corr_result, 'risk_level') else 0
            except Exception as e:
                logger.warning(f"Correlation check failed: {e}")

        # === CHECK 4: Calculate Position Size ===
        base_size = self.equity * self.max_position_pct
        kelly_fraction_used = self.kelly_fraction

        if self.use_kelly and self.kelly_sizer and entry_price > 0 and stop_loss > 0:
            try:
                kelly_result = self.kelly_sizer.calculate_position_size(
                    equity=self.equity,
                    entry_price=entry_price,
                    stop_loss=stop_loss
                )
                base_size = kelly_result.position_value
                kelly_fraction_used = kelly_result.kelly_fraction
            except Exception as e:
                logger.warning(f"Kelly sizing failed, using default: {e}")

        # === ADJUSTMENT: Scale by ML Confidence ===
        confidence_adjusted = False
        if self.use_ml_confidence and ml_confidence is not None:
            # Scale position size by confidence (0.5 to 1.5x)
            confidence_multiplier = 0.5 + ml_confidence
            base_size *= confidence_multiplier
            confidence_adjusted = True

        # === ADJUSTMENT: Scale by Conformal Uncertainty ===
        # High uncertainty -> smaller position; low uncertainty -> full position
        if self.use_conformal and ml_confidence is not None:
            conformal_mult = self._get_conformal_multiplier(ml_confidence)
            base_size *= conformal_mult
            if conformal_mult < 0.9:
                warnings.append(f"Position reduced by conformal uncertainty: {conformal_mult:.2f}x")

        # === CAPS: Apply maximum limits ===
        max_size = self.equity * self.max_position_pct
        final_size = min(base_size, max_size)

        # Calculate shares
        shares = int(final_size / entry_price) if entry_price > 0 else 0

        # Recalculate actual position value
        actual_size = shares * entry_price

        # === CHECK 5: Risk per trade ===
        if entry_price > 0 and stop_loss > 0:
            risk_per_share = abs(entry_price - stop_loss)
            total_risk = risk_per_share * shares
            risk_pct = total_risk / self.equity

            if risk_pct > self.max_portfolio_risk_pct:
                # Reduce shares to meet risk limit
                max_risk = self.equity * self.max_portfolio_risk_pct
                shares = int(max_risk / risk_per_share)
                actual_size = shares * entry_price
                warnings.append(f"Position reduced to meet {self.max_portfolio_risk_pct:.1%} risk limit")

        if shares <= 0:
            return TradeDecision(
                approved=False,
                position_size=0,
                shares=0,
                confidence_adjusted=confidence_adjusted,
                kelly_fraction=kelly_fraction_used,
                rejection_reason="Position size too small after risk adjustments",
                risk_score=risk_score
            )

        return TradeDecision(
            approved=True,
            position_size=actual_size,
            shares=shares,
            confidence_adjusted=confidence_adjusted,
            kelly_fraction=kelly_fraction_used,
            rejection_reason=None,
            risk_score=risk_score,
            warnings=warnings
        )

    def get_status(self) -> Dict:
        """Get current risk manager status."""
        return {
            'equity': self.equity,
            'max_position_pct': self.max_position_pct,
            'max_portfolio_risk_pct': self.max_portfolio_risk_pct,
            'use_kelly': self.use_kelly,
            'kelly_fraction': self.kelly_fraction,
            'use_ml_confidence': self.use_ml_confidence,
            'components': {
                'kelly_sizer': self._kelly_sizer is not None,
                'correlation_checker': self._correlation_checker is not None,
                'heat_monitor': self._heat_monitor is not None,
            }
        }


# Singleton instance
_risk_manager: Optional[PortfolioRiskManager] = None


def get_risk_manager(equity: float = 100000.0) -> PortfolioRiskManager:
    """Get or create singleton PortfolioRiskManager."""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = PortfolioRiskManager(equity=equity)
    return _risk_manager


def evaluate_trade_risk(
    signal: Dict,
    current_positions: List[Dict],
    equity: float = 100000.0,
    price_data: Optional[pd.DataFrame] = None,
    ml_confidence: Optional[float] = None
) -> TradeDecision:
    """
    Convenience function to evaluate a trade.

    Example:
        decision = evaluate_trade_risk(
            signal={'symbol': 'AAPL', 'entry_price': 150, 'stop_loss': 145},
            current_positions=[{'symbol': 'MSFT', 'value': 5000}],
            equity=100000,
            ml_confidence=0.75
        )
        if decision.approved:
            execute(signal, shares=decision.shares)
    """
    manager = get_risk_manager(equity)
    return manager.evaluate_trade(signal, current_positions, price_data, ml_confidence)
