"""
Generative Market Model - Synthetic Scenario & Counterfactual Generation
=========================================================================

This module provides the capability to generate synthetic market scenarios
and counterfactual simulations for "what-if" analysis. This enables the
cognitive architecture to:

1. **Test Hypotheses**: Generate synthetic data to test proposed trading rules
   without waiting for real market conditions.
2. **Counterfactual Analysis**: Answer "what would have happened if..." questions
   by generating alternative market trajectories.
3. **Stress Testing**: Create extreme but plausible scenarios to test system
   robustness.

The model uses a graceful degradation approach:
- Primary: GARCH volatility modeling (if scipy available)
- Fallback 1: Bootstrap resampling (numpy only)
- Fallback 2: Parametric generation (numpy only)

Usage:
    from data.ml import GenerativeMarketModel, ScenarioParams

    model = GenerativeMarketModel()

    # Generate a future scenario
    params = ScenarioParams(regime_type="bear", volatility_level="high")
    synthetic_data = model.generate_future_scenario(seed_data, params)

    # Generate a counterfactual
    cf_params = CounterfactualParams(
        deviation_point=datetime(2024, 1, 15),
        deviation_type="vix_spike",
        magnitude=1.5
    )
    counterfactual = model.generate_counterfactual_series(actual_data, cf_params)
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Optional dependencies with graceful degradation
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class ScenarioParams:
    """Parameters defining a synthetic market scenario."""
    regime_type: str = "neutral"  # bull, bear, neutral, choppy
    volatility_level: str = "normal"  # low, normal, high, extreme
    duration_days: int = 20
    trend_direction: float = 0.0  # -1 to 1, annualized drift adjustment
    vix_level: float = 20.0
    include_gaps: bool = True
    gap_frequency: float = 0.05  # Probability of gap on any given day
    gap_magnitude_pct: float = 0.02  # Average gap size

    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime_type': self.regime_type,
            'volatility_level': self.volatility_level,
            'duration_days': self.duration_days,
            'trend_direction': self.trend_direction,
            'vix_level': self.vix_level,
            'include_gaps': self.include_gaps,
        }


@dataclass
class CounterfactualParams:
    """Parameters for counterfactual scenario generation."""
    deviation_point: datetime
    deviation_type: str  # 'vix_spike', 'sentiment_shift', 'regime_change', 'price_shock'
    magnitude: float = 1.0  # Multiplier for the effect size
    duration_days: int = 5  # How long the effect lasts
    decay_rate: float = 0.5  # How quickly the effect fades

    def to_dict(self) -> Dict[str, Any]:
        return {
            'deviation_point': self.deviation_point.isoformat(),
            'deviation_type': self.deviation_type,
            'magnitude': self.magnitude,
            'duration_days': self.duration_days,
            'decay_rate': self.decay_rate,
        }


class GenerativeMarketModel:
    """
    Generates synthetic market data for scenario analysis and counterfactual testing.

    This class provides multiple generation methods with graceful degradation:
    1. GARCH (if scipy available) - Most realistic volatility dynamics
    2. Bootstrap (numpy only) - Preserves historical return distribution
    3. Parametric (numpy only) - Simple normal/student-t simulation
    """

    # Volatility level mappings (annualized)
    VOLATILITY_LEVELS = {
        'low': 0.10,      # 10% annualized
        'normal': 0.20,   # 20% annualized
        'high': 0.35,     # 35% annualized
        'extreme': 0.60,  # 60% annualized
    }

    # Regime drift adjustments (annualized)
    REGIME_DRIFTS = {
        'bull': 0.15,     # +15% annualized trend
        'bear': -0.20,    # -20% annualized trend
        'neutral': 0.0,
        'choppy': 0.0,    # No trend but higher volatility clustering
    }

    def __init__(
        self,
        use_garch: bool = True,
        random_seed: Optional[int] = None,
        preferred_method: str = "garch",
    ):
        """
        Initialize the generative model.

        Args:
            use_garch: Whether to attempt GARCH modeling (requires scipy)
            random_seed: Random seed for reproducibility
            preferred_method: "garch", "bootstrap", or "parametric"
        """
        self.use_garch = use_garch and HAS_SCIPY
        self.preferred_method = preferred_method
        self._rng = np.random.default_rng(random_seed)

        if use_garch and not HAS_SCIPY:
            logger.warning("GARCH requested but scipy not available. Falling back to bootstrap.")
            self.preferred_method = "bootstrap"

        logger.info(f"GenerativeMarketModel initialized. Method: {self.preferred_method}")

    def generate_future_scenario(
        self,
        seed_data: np.ndarray,
        params: ScenarioParams,
    ) -> Dict[str, Any]:
        """
        Generates a synthetic future price series based on historical seed data.

        Args:
            seed_data: Historical price series (1D numpy array or pandas Series)
            params: ScenarioParams defining the scenario characteristics

        Returns:
            Dict containing:
                - 'prices': Generated price series
                - 'returns': Generated return series
                - 'volatility': Realized volatility series
                - 'metadata': Generation parameters and method used
        """
        # Convert to numpy if pandas
        if HAS_PANDAS and hasattr(seed_data, 'values'):
            seed_prices = seed_data.values.flatten()
        else:
            seed_prices = np.asarray(seed_data).flatten()

        # Calculate seed returns
        seed_returns = np.diff(np.log(seed_prices))

        # Get target volatility
        target_vol = self.VOLATILITY_LEVELS.get(params.volatility_level, 0.20)
        drift = self.REGIME_DRIFTS.get(params.regime_type, 0.0) + params.trend_direction

        # Daily drift and volatility
        daily_drift = drift / 252
        daily_vol = target_vol / np.sqrt(252)

        # Generate returns based on preferred method
        if self.preferred_method == "garch" and self.use_garch:
            generated_returns, volatilities = self._garch_volatility(
                seed_returns, params.duration_days, daily_vol
            )
        elif self.preferred_method == "bootstrap":
            generated_returns = self._bootstrap_returns(
                seed_returns, params.duration_days, block_size=5
            )
            # Scale to target volatility
            current_vol = np.std(generated_returns)
            if current_vol > 0:
                generated_returns = generated_returns * (daily_vol / current_vol)
            volatilities = np.full(params.duration_days, daily_vol)
        else:
            generated_returns = self._parametric_generation(
                daily_drift, daily_vol, params.duration_days
            )
            volatilities = np.full(params.duration_days, daily_vol)

        # Add drift
        generated_returns = generated_returns + daily_drift

        # Add gaps if requested
        if params.include_gaps:
            generated_returns = self._add_gaps(
                generated_returns,
                params.gap_frequency,
                params.gap_magnitude_pct
            )

        # Convert returns to prices
        last_price = seed_prices[-1]
        cumulative_returns = np.cumsum(generated_returns)
        generated_prices = last_price * np.exp(cumulative_returns)

        return {
            'prices': generated_prices,
            'returns': generated_returns,
            'volatility': volatilities,
            'seed_last_price': last_price,
            'metadata': {
                'method': self.preferred_method,
                'params': params.to_dict(),
                'daily_drift': daily_drift,
                'daily_vol': daily_vol,
            }
        }

    def generate_counterfactual_series(
        self,
        actual_series: np.ndarray,
        params: CounterfactualParams,
        timestamps: Optional[List[datetime]] = None,
    ) -> Dict[str, Any]:
        """
        Generates a counterfactual version of a price series.

        This answers "what if X had happened differently?" by creating an
        alternative trajectory starting from a deviation point.

        Args:
            actual_series: The actual historical price series
            params: CounterfactualParams specifying the deviation
            timestamps: Optional list of timestamps corresponding to prices

        Returns:
            Dict containing:
                - 'counterfactual_prices': The alternative price trajectory
                - 'deviation_index': Index where deviation starts
                - 'impact_zone': Boolean array marking affected periods
                - 'metadata': Generation details
        """
        if HAS_PANDAS and hasattr(actual_series, 'values'):
            prices = actual_series.values.flatten()
        else:
            prices = np.asarray(actual_series).flatten()

        # Find deviation point index
        if timestamps:
            try:
                deviation_idx = next(
                    i for i, t in enumerate(timestamps)
                    if t >= params.deviation_point
                )
            except StopIteration:
                deviation_idx = len(prices) - 1
        else:
            # Assume deviation_point is a fraction of the series
            deviation_idx = len(prices) // 2

        # Copy actual prices up to deviation point
        cf_prices = prices.copy()

        # Apply the deviation effect
        impact_zone = np.zeros(len(prices), dtype=bool)

        for i in range(deviation_idx, min(len(prices), deviation_idx + params.duration_days)):
            days_since = i - deviation_idx
            decay = np.exp(-params.decay_rate * days_since)
            impact = self._calculate_deviation_impact(
                params.deviation_type,
                params.magnitude * decay,
                prices[i] if i < len(prices) else prices[-1]
            )
            cf_prices[i] = cf_prices[i] * (1 + impact)
            impact_zone[i] = True

        # Propagate effect forward
        for i in range(deviation_idx + params.duration_days, len(prices)):
            # The counterfactual path continues from the perturbed trajectory
            original_return = (prices[i] - prices[i-1]) / prices[i-1] if i > 0 else 0
            cf_prices[i] = cf_prices[i-1] * (1 + original_return)

        return {
            'counterfactual_prices': cf_prices,
            'deviation_index': deviation_idx,
            'impact_zone': impact_zone,
            'original_prices': prices,
            'metadata': {
                'params': params.to_dict(),
                'affected_days': int(impact_zone.sum()),
            }
        }

    def _garch_volatility(
        self,
        seed_returns: np.ndarray,
        forecast_horizon: int,
        target_daily_vol: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate returns with GARCH(1,1) volatility dynamics.

        Uses simplified GARCH simulation without full parameter estimation.
        """
        # GARCH(1,1) parameters (typical equity market values)
        omega = target_daily_vol ** 2 * 0.05  # Long-run variance weight
        alpha = 0.10  # Shock impact
        beta = 0.85   # Persistence

        # Initialize
        volatilities = np.zeros(forecast_horizon)
        returns = np.zeros(forecast_horizon)

        # Start with recent realized volatility
        h_t = np.var(seed_returns[-20:]) if len(seed_returns) >= 20 else target_daily_vol ** 2

        for t in range(forecast_horizon):
            # Generate innovation
            z_t = self._rng.standard_normal()

            # Generate return
            vol_t = np.sqrt(h_t)
            returns[t] = z_t * vol_t
            volatilities[t] = vol_t

            # Update variance
            shock = returns[t] ** 2
            h_t = omega + alpha * shock + beta * h_t

        return returns, volatilities

    def _bootstrap_returns(
        self,
        seed_returns: np.ndarray,
        n_samples: int,
        block_size: int = 5,
    ) -> np.ndarray:
        """
        Generate returns using block bootstrap to preserve autocorrelation.
        """
        n_blocks = (n_samples // block_size) + 1
        blocks = []

        for _ in range(n_blocks):
            start_idx = self._rng.integers(0, max(1, len(seed_returns) - block_size))
            end_idx = min(start_idx + block_size, len(seed_returns))
            blocks.append(seed_returns[start_idx:end_idx])

        # Concatenate and trim
        bootstrapped = np.concatenate(blocks)[:n_samples]
        return bootstrapped

    def _parametric_generation(
        self,
        mu: float,
        sigma: float,
        n_samples: int,
        use_student_t: bool = True,
    ) -> np.ndarray:
        """
        Generate returns from a parametric distribution.
        """
        if use_student_t:
            # Student-t with 5 degrees of freedom (fat tails)
            df = 5
            returns = self._rng.standard_t(df, n_samples) * sigma * np.sqrt((df - 2) / df)
        else:
            returns = self._rng.normal(mu, sigma, n_samples)

        return returns

    def _add_gaps(
        self,
        returns: np.ndarray,
        gap_frequency: float,
        gap_magnitude: float,
    ) -> np.ndarray:
        """Add occasional gaps (jumps) to the return series."""
        gap_mask = self._rng.random(len(returns)) < gap_frequency
        gap_sizes = self._rng.normal(0, gap_magnitude, len(returns))

        modified_returns = returns.copy()
        modified_returns[gap_mask] += gap_sizes[gap_mask]

        return modified_returns

    def _calculate_deviation_impact(
        self,
        deviation_type: str,
        magnitude: float,
        current_price: float,
    ) -> float:
        """Calculate the price impact of a deviation event."""
        if deviation_type == "vix_spike":
            # VIX spike typically causes negative price impact
            return -0.02 * magnitude
        elif deviation_type == "sentiment_shift":
            # Sentiment shift can be positive or negative
            return 0.015 * magnitude * (1 if self._rng.random() > 0.5 else -1)
        elif deviation_type == "regime_change":
            # Regime change causes larger, more persistent impact
            return -0.03 * magnitude
        elif deviation_type == "price_shock":
            # Direct price shock
            return 0.05 * magnitude
        else:
            return 0.01 * magnitude

    def introspect(self) -> str:
        """Generate a human-readable description of the model's configuration."""
        lines = [
            "--- Generative Market Model ---",
            f"Preferred method: {self.preferred_method}",
            f"GARCH available: {self.use_garch}",
            f"Scipy available: {HAS_SCIPY}",
            f"Pandas available: {HAS_PANDAS}",
            "\nVolatility levels: " + ", ".join(f"{k}={v:.0%}" for k, v in self.VOLATILITY_LEVELS.items()),
            "Regime drifts: " + ", ".join(f"{k}={v:+.0%}" for k, v in self.REGIME_DRIFTS.items()),
        ]
        return "\n".join(lines)


# --- Singleton Implementation ---
_generative_model: Optional[GenerativeMarketModel] = None
_lock = threading.Lock()


def get_generative_model() -> GenerativeMarketModel:
    """Factory function to get the singleton GenerativeMarketModel instance."""
    global _generative_model
    if _generative_model is None:
        with _lock:
            if _generative_model is None:
                _generative_model = GenerativeMarketModel()
    return _generative_model
