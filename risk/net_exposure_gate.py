"""
Net Exposure Gate for Kobe Trading System.

Enforces net exposure limits (long - short) to prevent directional overexposure.
This is a SAFETY-CRITICAL component for real money trading.

The PortfolioState already tracks net_exposure, but this gate ENFORCES limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging
import yaml

if TYPE_CHECKING:
    from risk.portfolio_risk import PortfolioState

logger = logging.getLogger(__name__)


@dataclass
class NetExposureLimits:
    """Configuration for net exposure limits."""
    max_net_exposure_pct: float = 80.0    # Max |long - short| as % of NAV
    max_long_exposure_pct: float = 100.0   # Max long as % of NAV
    max_short_exposure_pct: float = 50.0   # Max short as % of NAV
    warning_threshold_pct: float = 70.0    # Warning when approaching limit


@dataclass
class NetExposureCheck:
    """Result of net exposure check."""
    passed: bool
    current_net_pct: float
    current_long_pct: float
    current_short_pct: float
    proposed_net_pct: float
    proposed_long_pct: float
    proposed_short_pct: float
    headroom_pct: float                    # How much more exposure allowed
    rejection_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "current_net_pct": round(self.current_net_pct, 2),
            "current_long_pct": round(self.current_long_pct, 2),
            "current_short_pct": round(self.current_short_pct, 2),
            "proposed_net_pct": round(self.proposed_net_pct, 2),
            "proposed_long_pct": round(self.proposed_long_pct, 2),
            "proposed_short_pct": round(self.proposed_short_pct, 2),
            "headroom_pct": round(self.headroom_pct, 2),
            "rejection_reason": self.rejection_reason,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
        }


class NetExposureGate:
    """
    Enforces net exposure limits for the portfolio.

    This gate BLOCKS trades that would push net exposure beyond limits.
    All gates are fail-safe: if we can't compute exposure, we BLOCK.

    Usage:
        gate = NetExposureGate.from_config()
        check = gate.check("AAPL", "long", 5000.0, portfolio_state)
        if not check.passed:
            logger.warning(f"Trade blocked: {check.rejection_reason}")
    """

    def __init__(
        self,
        limits: Optional[NetExposureLimits] = None,
        enabled: bool = True,
    ):
        self.limits = limits or NetExposureLimits()
        self.enabled = enabled
        self._last_check: Optional[NetExposureCheck] = None

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> 'NetExposureGate':
        """Create NetExposureGate from config file."""
        if config_path is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "base.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Config not found at {config_path}, using defaults")
            return cls()

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            net_exp_config = config.get('net_exposure', {})
            enabled = net_exp_config.get('enabled', True)

            limits = NetExposureLimits(
                max_net_exposure_pct=net_exp_config.get('max_net_exposure_pct', 80.0),
                max_long_exposure_pct=net_exp_config.get('max_long_exposure_pct', 100.0),
                max_short_exposure_pct=net_exp_config.get('max_short_exposure_pct', 50.0),
                warning_threshold_pct=net_exp_config.get('warning_threshold_pct', 70.0),
            )

            return cls(limits=limits, enabled=enabled)

        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            return cls()

    def check(
        self,
        symbol: str,
        side: str,  # "long" or "short"
        proposed_notional: float,
        portfolio_state: 'PortfolioState',
    ) -> NetExposureCheck:
        """
        Check if proposed trade would violate net exposure limits.

        Args:
            symbol: Stock symbol
            side: "long" or "short"
            proposed_notional: Dollar value of proposed position
            portfolio_state: Current portfolio state with NAV and positions

        Returns:
            NetExposureCheck with pass/fail status and details
        """
        if not self.enabled:
            return NetExposureCheck(
                passed=True,
                current_net_pct=0.0,
                current_long_pct=0.0,
                current_short_pct=0.0,
                proposed_net_pct=0.0,
                proposed_long_pct=0.0,
                proposed_short_pct=0.0,
                headroom_pct=100.0,
                warnings=["net_exposure_gate_disabled"],
            )

        nav = portfolio_state.nav

        # FAIL-SAFE: Block if NAV is invalid
        if nav <= 0:
            check = NetExposureCheck(
                passed=False,
                current_net_pct=0.0,
                current_long_pct=0.0,
                current_short_pct=0.0,
                proposed_net_pct=0.0,
                proposed_long_pct=0.0,
                proposed_short_pct=0.0,
                headroom_pct=0.0,
                rejection_reason="nav_invalid_or_zero",
            )
            self._last_check = check
            return check

        # Current exposures
        current_long = portfolio_state.long_exposure
        current_short = portfolio_state.short_exposure
        current_net = portfolio_state.net_exposure

        current_long_pct = (current_long / nav) * 100
        current_short_pct = (current_short / nav) * 100
        current_net_pct = (abs(current_net) / nav) * 100

        # Calculate proposed exposures
        if side.lower() == "long":
            proposed_long = current_long + proposed_notional
            proposed_short = current_short
        else:  # short
            proposed_long = current_long
            proposed_short = current_short + proposed_notional

        proposed_net = proposed_long - proposed_short
        proposed_long_pct = (proposed_long / nav) * 100
        proposed_short_pct = (proposed_short / nav) * 100
        proposed_net_pct = (abs(proposed_net) / nav) * 100

        # Check all limits
        warnings = []
        rejection_reason = None

        # Check net exposure limit
        if proposed_net_pct > self.limits.max_net_exposure_pct:
            rejection_reason = (
                f"net_exposure_{proposed_net_pct:.1f}pct_exceeds_limit_"
                f"{self.limits.max_net_exposure_pct}pct"
            )

        # Check long exposure limit
        elif proposed_long_pct > self.limits.max_long_exposure_pct:
            rejection_reason = (
                f"long_exposure_{proposed_long_pct:.1f}pct_exceeds_limit_"
                f"{self.limits.max_long_exposure_pct}pct"
            )

        # Check short exposure limit
        elif proposed_short_pct > self.limits.max_short_exposure_pct:
            rejection_reason = (
                f"short_exposure_{proposed_short_pct:.1f}pct_exceeds_limit_"
                f"{self.limits.max_short_exposure_pct}pct"
            )

        # Add warnings for approaching limits
        if rejection_reason is None:
            if proposed_net_pct > self.limits.warning_threshold_pct:
                warnings.append(
                    f"net_exposure_approaching_limit_{proposed_net_pct:.1f}pct"
                )
            if proposed_long_pct > self.limits.max_long_exposure_pct * 0.8:
                warnings.append(
                    f"long_exposure_approaching_limit_{proposed_long_pct:.1f}pct"
                )
            if proposed_short_pct > self.limits.max_short_exposure_pct * 0.8:
                warnings.append(
                    f"short_exposure_approaching_limit_{proposed_short_pct:.1f}pct"
                )

        # Calculate headroom
        net_headroom = self.limits.max_net_exposure_pct - proposed_net_pct
        headroom_pct = max(0.0, net_headroom)

        check = NetExposureCheck(
            passed=rejection_reason is None,
            current_net_pct=current_net_pct,
            current_long_pct=current_long_pct,
            current_short_pct=current_short_pct,
            proposed_net_pct=proposed_net_pct,
            proposed_long_pct=proposed_long_pct,
            proposed_short_pct=proposed_short_pct,
            headroom_pct=headroom_pct,
            rejection_reason=rejection_reason,
            warnings=warnings,
        )

        self._last_check = check

        if rejection_reason:
            logger.warning(
                f"NetExposureGate BLOCKED {symbol} {side}: {rejection_reason}"
            )
        elif warnings:
            logger.info(
                f"NetExposureGate APPROVED {symbol} {side} with warnings: {warnings}"
            )

        return check

    def get_max_allowable_notional(
        self,
        side: str,
        portfolio_state: 'PortfolioState',
    ) -> float:
        """
        Calculate maximum notional for given side that stays within limits.

        Args:
            side: "long" or "short"
            portfolio_state: Current portfolio state

        Returns:
            Maximum allowable notional in dollars
        """
        if not self.enabled:
            return float('inf')

        nav = portfolio_state.nav
        if nav <= 0:
            return 0.0

        current_long = portfolio_state.long_exposure
        current_short = portfolio_state.short_exposure

        if side.lower() == "long":
            # Max long is limited by:
            # 1. Max long exposure limit
            # 2. Max net exposure limit
            max_by_long_limit = (
                self.limits.max_long_exposure_pct / 100 * nav - current_long
            )

            # Net exposure = long - short
            # Max net = max_net_pct * nav
            # long - short <= max_net_pct * nav
            # long <= max_net_pct * nav + short
            max_by_net_limit = (
                self.limits.max_net_exposure_pct / 100 * nav + current_short - current_long
            )

            return max(0.0, min(max_by_long_limit, max_by_net_limit))

        else:  # short
            # Max short is limited by:
            # 1. Max short exposure limit
            # 2. Max net exposure limit (negative direction)
            max_by_short_limit = (
                self.limits.max_short_exposure_pct / 100 * nav - current_short
            )

            # Net exposure = long - short
            # -Max net = -max_net_pct * nav (if short dominant)
            # long - short >= -max_net_pct * nav
            # short <= long + max_net_pct * nav
            max_by_net_limit = (
                current_long + self.limits.max_net_exposure_pct / 100 * nav - current_short
            )

            return max(0.0, min(max_by_short_limit, max_by_net_limit))

    def get_status(self) -> Dict[str, Any]:
        """Get current gate status."""
        return {
            "enabled": self.enabled,
            "limits": {
                "max_net_exposure_pct": self.limits.max_net_exposure_pct,
                "max_long_exposure_pct": self.limits.max_long_exposure_pct,
                "max_short_exposure_pct": self.limits.max_short_exposure_pct,
                "warning_threshold_pct": self.limits.warning_threshold_pct,
            },
            "last_check": self._last_check.to_dict() if self._last_check else None,
        }
