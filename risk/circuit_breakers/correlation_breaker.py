"""
Correlation Circuit Breaker - Regime Change Detection

Protects capital by detecting when market correlations break down,
indicating a regime change that invalidates historical relationships.

During correlation breakdowns:
- Historical patterns stop working
- Diversification fails (everything moves together)
- Risk models underestimate true risk

Thresholds:
- Correlation spike to 0.9+: PAUSE_NEW (everything correlated)
- Correlation breakdown: REDUCE_SIZE (relationships changing)
- Beta instability: ALERT_ONLY (market sensitivity changing)

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

import numpy as np
import pandas as pd

from core.structured_log import get_logger
from .breaker_manager import BreakerAction, BreakerStatus

logger = get_logger(__name__)


@dataclass
class CorrelationThresholds:
    """Correlation thresholds for circuit breaker."""
    # High correlation (everything moves together = no diversification)
    correlation_spike: float = 0.85     # Pause if avg correlation > 0.85
    correlation_high: float = 0.70      # Reduce if avg correlation > 0.70

    # Correlation breakdown (relationships changing)
    correlation_change: float = 0.30    # Alert if correlation changed > 0.30

    # Beta instability
    beta_spike: float = 1.5             # Alert if portfolio beta > 1.5
    beta_change: float = 0.3            # Alert if beta changed > 0.3 in short period

    # Lookback periods
    short_window: int = 5               # Short-term correlation window (days)
    long_window: int = 20               # Long-term correlation window (days)


class CorrelationBreaker:
    """
    Circuit breaker that monitors portfolio correlations.

    Solo Trader Features:
    - Tracks position correlations
    - Detects regime changes via correlation breakdown
    - Warns of excessive portfolio correlation
    - Beta monitoring for market sensitivity
    """

    STATE_FILE = Path("state/circuit_breakers/correlation_history.json")

    def __init__(self, thresholds: Optional[CorrelationThresholds] = None):
        """
        Initialize correlation breaker.

        Args:
            thresholds: Custom thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or CorrelationThresholds()
        self._correlation_history: List[Dict] = []
        self._last_check: Optional[datetime] = None
        self._cached_correlation: Optional[float] = None
        self._cached_beta: Optional[float] = None

        # Ensure state directory exists
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load history
        self._load_history()

    def _load_history(self) -> None:
        """Load correlation history from state file."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    self._correlation_history = data.get("history", [])[-100:]
            except Exception as e:
                logger.warning(f"Failed to load correlation history: {e}")

    def _save_history(self) -> None:
        """Save correlation history to state file."""
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump({
                    "history": self._correlation_history[-100:],
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save correlation history: {e}")

    def _calculate_portfolio_correlation(
        self,
        returns_df: Optional[pd.DataFrame] = None,
    ) -> Optional[float]:
        """
        Calculate average pairwise correlation of portfolio.

        Args:
            returns_df: DataFrame with symbol returns as columns

        Returns:
            Average pairwise correlation or None
        """
        if returns_df is None or returns_df.empty or len(returns_df.columns) < 2:
            return None

        try:
            corr_matrix = returns_df.corr()

            # Get upper triangle (exclude diagonal)
            upper_tri = np.triu(corr_matrix.values, k=1)
            n_pairs = np.sum(upper_tri != 0)

            if n_pairs == 0:
                return None

            avg_corr = np.sum(np.abs(upper_tri)) / n_pairs
            return float(avg_corr)

        except Exception as e:
            logger.warning(f"Failed to calculate correlation: {e}")
            return None

    def _calculate_portfolio_beta(
        self,
        portfolio_returns: Optional[pd.Series] = None,
        market_returns: Optional[pd.Series] = None,
    ) -> Optional[float]:
        """
        Calculate portfolio beta relative to market.

        Args:
            portfolio_returns: Series of portfolio returns
            market_returns: Series of market (SPY) returns

        Returns:
            Portfolio beta or None
        """
        if portfolio_returns is None or market_returns is None:
            return None

        try:
            # Align series
            aligned = pd.concat([portfolio_returns, market_returns], axis=1).dropna()

            if len(aligned) < 5:
                return None

            port = aligned.iloc[:, 0]
            mkt = aligned.iloc[:, 1]

            covariance = port.cov(mkt)
            variance = mkt.var()

            if variance == 0:
                return None

            beta = covariance / variance
            return float(beta)

        except Exception as e:
            logger.warning(f"Failed to calculate beta: {e}")
            return None

    def _get_correlation_change(self) -> float:
        """Calculate how much correlation has changed recently."""
        if len(self._correlation_history) < 2:
            return 0.0

        # Compare recent to historical
        recent = [h["avg_correlation"] for h in self._correlation_history[-5:] if h.get("avg_correlation")]
        older = [h["avg_correlation"] for h in self._correlation_history[-20:-5] if h.get("avg_correlation")]

        if not recent or not older:
            return 0.0

        return abs(np.mean(recent) - np.mean(older))

    def _get_spy_returns(self, days: int = 30) -> Optional[pd.Series]:
        """Fetch SPY returns for beta calculation."""
        try:
            from data.providers.polygon_eod import PolygonEODProvider

            provider = PolygonEODProvider()
            df = provider.get("SPY", limit=days + 10)

            if df is not None and not df.empty:
                returns = df["close"].pct_change().dropna()
                return returns.tail(days)

        except Exception as e:
            logger.debug(f"Failed to fetch SPY returns: {e}")

        return None

    def update_correlation(
        self,
        avg_correlation: float,
        portfolio_beta: Optional[float] = None,
    ) -> None:
        """
        Record correlation data point.

        Args:
            avg_correlation: Average pairwise correlation
            portfolio_beta: Portfolio beta vs SPY
        """
        self._correlation_history.append({
            "timestamp": datetime.now().isoformat(),
            "avg_correlation": avg_correlation,
            "portfolio_beta": portfolio_beta,
        })

        self._cached_correlation = avg_correlation
        self._cached_beta = portfolio_beta
        self._save_history()

    def check(
        self,
        returns_df: Optional[pd.DataFrame] = None,
        portfolio_returns: Optional[pd.Series] = None,
        avg_correlation: Optional[float] = None,
        portfolio_beta: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Check correlation levels against thresholds.

        Args:
            returns_df: DataFrame with symbol returns (for correlation calc)
            portfolio_returns: Portfolio returns series (for beta calc)
            avg_correlation: Pre-calculated average correlation
            portfolio_beta: Pre-calculated portfolio beta
            **kwargs: Ignored (for compatibility with BreakerManager)

        Returns:
            Dict with status, action, message, and details
        """
        # Calculate or use provided correlation
        if avg_correlation is None and returns_df is not None:
            avg_correlation = self._calculate_portfolio_correlation(returns_df)

        if avg_correlation is None:
            avg_correlation = self._cached_correlation

        # Calculate or use provided beta
        if portfolio_beta is None and portfolio_returns is not None:
            market_returns = self._get_spy_returns()
            portfolio_beta = self._calculate_portfolio_beta(portfolio_returns, market_returns)

        if portfolio_beta is None:
            portfolio_beta = self._cached_beta

        # Record if we have new data
        if avg_correlation is not None:
            self.update_correlation(avg_correlation, portfolio_beta)

        # Can't check without data
        if avg_correlation is None:
            return {
                "status": BreakerStatus.GREEN,
                "action": BreakerAction.CONTINUE,
                "message": "Insufficient data for correlation analysis",
                "threshold": 0,
                "current_value": 0,
                "details": {"data_available": False},
            }

        # Get correlation change
        corr_change = self._get_correlation_change()

        # Determine status and action
        status = BreakerStatus.GREEN
        action = BreakerAction.CONTINUE
        triggered_by = None
        threshold_hit = 0

        # Check correlation spike (everything moves together)
        if avg_correlation >= self.thresholds.correlation_spike:
            status = BreakerStatus.RED
            action = BreakerAction.PAUSE_NEW
            triggered_by = "correlation_spike"
            threshold_hit = self.thresholds.correlation_spike
            logger.warning(
                f"PAUSE: Avg correlation {avg_correlation:.2f} >= {self.thresholds.correlation_spike}"
            )

        elif avg_correlation >= self.thresholds.correlation_high:
            status = BreakerStatus.YELLOW
            action = BreakerAction.REDUCE_SIZE
            triggered_by = "correlation_high"
            threshold_hit = self.thresholds.correlation_high
            logger.info(
                f"REDUCE: Avg correlation {avg_correlation:.2f} >= {self.thresholds.correlation_high}"
            )

        # Check correlation breakdown (relationships changing)
        elif corr_change >= self.thresholds.correlation_change:
            status = BreakerStatus.YELLOW
            action = BreakerAction.ALERT_ONLY
            triggered_by = "correlation_breakdown"
            threshold_hit = self.thresholds.correlation_change
            logger.info(f"ALERT: Correlation change {corr_change:.2f} (regime shift?)")

        # Check beta levels
        if portfolio_beta is not None:
            if portfolio_beta >= self.thresholds.beta_spike:
                if status == BreakerStatus.GREEN:
                    status = BreakerStatus.YELLOW
                    action = BreakerAction.ALERT_ONLY
                    triggered_by = "beta_spike"
                    threshold_hit = self.thresholds.beta_spike
                logger.info(f"ALERT: Portfolio beta {portfolio_beta:.2f} is high")

        # Build message
        if triggered_by:
            if triggered_by == "correlation_spike":
                message = f"High correlation {avg_correlation:.2f} - diversification failing"
            elif triggered_by == "correlation_high":
                message = f"Elevated correlation {avg_correlation:.2f} - reduce concentration"
            elif triggered_by == "correlation_breakdown":
                message = f"Correlation changed {corr_change:.2f} - possible regime shift"
            elif triggered_by == "beta_spike":
                message = f"High portfolio beta {portfolio_beta:.2f} - excess market exposure"
            else:
                message = f"Correlation anomaly detected"
        else:
            message = f"Correlation normal at {avg_correlation:.2f}"

        self._last_check = datetime.now()

        return {
            "status": status,
            "action": action,
            "message": message,
            "triggered_by": triggered_by,
            "threshold": threshold_hit,
            "current_value": avg_correlation,
            "details": {
                "avg_correlation": avg_correlation,
                "portfolio_beta": portfolio_beta,
                "correlation_change": corr_change,
                "thresholds": {
                    "correlation_spike": self.thresholds.correlation_spike,
                    "correlation_high": self.thresholds.correlation_high,
                    "correlation_change": self.thresholds.correlation_change,
                    "beta_spike": self.thresholds.beta_spike,
                },
            },
        }

    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get summary of correlation status."""
        recent = self._correlation_history[-5:] if self._correlation_history else []

        return {
            "current_correlation": self._cached_correlation,
            "current_beta": self._cached_beta,
            "recent_avg": np.mean([h["avg_correlation"] for h in recent if h.get("avg_correlation")]) if recent else None,
            "data_points": len(self._correlation_history),
        }


if __name__ == "__main__":
    # Demo
    breaker = CorrelationBreaker()

    print("=== Correlation Breaker Demo ===\n")

    # Test scenarios
    scenarios = [
        {"corr": 0.30, "beta": 0.8, "desc": "Normal - low correlation"},
        {"corr": 0.50, "beta": 1.0, "desc": "Normal - moderate correlation"},
        {"corr": 0.75, "beta": 1.2, "desc": "Warning - high correlation"},
        {"corr": 0.90, "beta": 1.6, "desc": "Critical - correlation spike"},
    ]

    for s in scenarios:
        result = breaker.check(avg_correlation=s["corr"], portfolio_beta=s["beta"])
        print(f"{s['desc']}:")
        print(f"  Correlation: {s['corr']:.2f}, Beta: {s['beta']:.1f}")
        print(f"  Status: {result['status'].value}")
        print(f"  Action: {result['action'].value}")
        print(f"  Message: {result['message']}")
        print()

    print(f"Summary: {breaker.get_correlation_summary()}")
