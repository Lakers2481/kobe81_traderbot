"""
Factor Attribution for Signal Component Analysis.

Identifies which signal components (factors) drive trading profits.
Uses regression-based and SHAP-based decomposition.

Factors Analyzed:
- IBS (Internal Bar Strength)
- RSI values
- Sweep strength (for ICT strategies)
- ATR-based volatility
- Trend alignment (SMA)
- Volume characteristics
- Market regime
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """Types of signal factors."""
    IBS = auto()              # Internal Bar Strength
    RSI = auto()              # Relative Strength Index
    SWEEP_STRENGTH = auto()   # Turtle Soup sweep
    ATR_MULTIPLE = auto()     # ATR-based volatility
    TREND_ALIGNMENT = auto()  # Price vs SMA
    VOLUME_RATIO = auto()     # Volume vs average
    REGIME = auto()           # Market regime
    VIX = auto()              # VIX level
    SECTOR = auto()           # Sector classification
    GAP = auto()              # Overnight gap
    CUSTOM = auto()           # Custom factors


@dataclass
class FactorContribution:
    """Contribution of a single factor to PnL."""
    factor_type: FactorType
    factor_name: str
    contribution: float       # Absolute PnL contribution
    contribution_pct: float   # Percentage of total PnL
    coefficient: float        # Regression coefficient
    t_statistic: float        # Statistical significance
    p_value: float
    avg_factor_value: float   # Average value when trade taken
    correlation_with_pnl: float

    @property
    def is_significant(self) -> bool:
        """Check if factor is statistically significant (p < 0.05)."""
        return self.p_value < 0.05

    @property
    def is_positive_contributor(self) -> bool:
        """Check if factor contributes positively to PnL."""
        return self.contribution > 0


@dataclass
class AttributionResult:
    """Complete factor attribution result."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_pnl: float = 0.0
    total_trades: int = 0
    r_squared: float = 0.0  # Model fit
    adj_r_squared: float = 0.0

    factor_contributions: Dict[str, FactorContribution] = field(default_factory=dict)
    residual_pnl: float = 0.0  # PnL not explained by factors

    top_positive_factors: List[Tuple[str, float]] = field(default_factory=list)
    top_negative_factors: List[Tuple[str, float]] = field(default_factory=list)
    significant_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_pnl": round(self.total_pnl, 2),
            "total_trades": self.total_trades,
            "r_squared": round(self.r_squared, 4),
            "adj_r_squared": round(self.adj_r_squared, 4),
            "factor_contributions": {
                name: {
                    "contribution": round(fc.contribution, 2),
                    "contribution_pct": round(fc.contribution_pct, 2),
                    "coefficient": round(fc.coefficient, 6),
                    "p_value": round(fc.p_value, 4),
                    "is_significant": fc.is_significant,
                }
                for name, fc in self.factor_contributions.items()
            },
            "residual_pnl": round(self.residual_pnl, 2),
            "top_positive_factors": [(n, round(v, 2)) for n, v in self.top_positive_factors],
            "top_negative_factors": [(n, round(v, 2)) for n, v in self.top_negative_factors],
            "significant_factors": self.significant_factors,
        }


class FactorAttribution:
    """
    Analyze factor contributions to trading PnL.

    Uses OLS regression to decompose PnL into factor contributions.
    Optionally uses SHAP for more sophisticated attribution.
    """

    # Standard factor columns expected in trades DataFrame
    STANDARD_FACTORS = {
        "ibs": FactorType.IBS,
        "rsi": FactorType.RSI,
        "rsi_2": FactorType.RSI,
        "sweep_strength": FactorType.SWEEP_STRENGTH,
        "atr_multiple": FactorType.ATR_MULTIPLE,
        "trend_score": FactorType.TREND_ALIGNMENT,
        "volume_ratio": FactorType.VOLUME_RATIO,
        "regime_score": FactorType.REGIME,
        "vix": FactorType.VIX,
        "gap_pct": FactorType.GAP,
    }

    def __init__(
        self,
        min_trades: int = 30,
        standardize_factors: bool = True,
        include_interactions: bool = False,
    ):
        """
        Initialize factor attribution.

        Args:
            min_trades: Minimum trades for reliable attribution
            standardize_factors: Whether to z-score normalize factors
            include_interactions: Whether to include factor interactions
        """
        self.min_trades = min_trades
        self.standardize_factors = standardize_factors
        self.include_interactions = include_interactions

    def analyze(
        self,
        trades_df: pd.DataFrame,
        pnl_column: str = "pnl",
        factor_columns: Optional[List[str]] = None,
    ) -> AttributionResult:
        """
        Perform factor attribution analysis.

        Args:
            trades_df: DataFrame of trades with factor values
            pnl_column: Name of PnL column
            factor_columns: List of factor column names (auto-detect if None)

        Returns:
            AttributionResult with factor contributions
        """
        if len(trades_df) < self.min_trades:
            logger.warning(f"Insufficient trades ({len(trades_df)}) for attribution")
            return AttributionResult(
                total_trades=len(trades_df),
                total_pnl=trades_df[pnl_column].sum() if pnl_column in trades_df.columns else 0,
            )

        # Auto-detect factor columns if not provided
        if factor_columns is None:
            factor_columns = [
                col for col in trades_df.columns
                if col in self.STANDARD_FACTORS or col.startswith("factor_")
            ]

        if not factor_columns:
            logger.warning("No factor columns found for attribution")
            return AttributionResult(
                total_trades=len(trades_df),
                total_pnl=trades_df[pnl_column].sum(),
            )

        # Prepare data
        X, y, valid_factors = self._prepare_data(trades_df, pnl_column, factor_columns)

        if X is None or len(X) < self.min_trades:
            return AttributionResult(
                total_trades=len(trades_df),
                total_pnl=trades_df[pnl_column].sum(),
            )

        # Run regression
        result = self._run_regression(X, y, valid_factors)
        result.total_trades = len(X)
        result.total_pnl = y.sum()

        # Identify top factors
        result.top_positive_factors = sorted(
            [(name, fc.contribution) for name, fc in result.factor_contributions.items()
             if fc.contribution > 0],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        result.top_negative_factors = sorted(
            [(name, fc.contribution) for name, fc in result.factor_contributions.items()
             if fc.contribution < 0],
            key=lambda x: x[1],
        )[:5]

        result.significant_factors = [
            name for name, fc in result.factor_contributions.items()
            if fc.is_significant
        ]

        return result

    def _prepare_data(
        self,
        trades_df: pd.DataFrame,
        pnl_column: str,
        factor_columns: List[str],
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], List[str]]:
        """Prepare data for regression."""
        # Get valid columns
        valid_factors = [col for col in factor_columns if col in trades_df.columns]

        if not valid_factors:
            return None, None, []

        # Extract X and y
        data = trades_df[[pnl_column] + valid_factors].dropna()

        if len(data) < self.min_trades:
            return None, None, []

        y = data[pnl_column].values
        X = data[valid_factors].copy()

        # Standardize if requested
        if self.standardize_factors:
            X = (X - X.mean()) / (X.std() + 1e-10)

        return X, pd.Series(y), valid_factors

    def _run_regression(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        factor_names: List[str],
    ) -> AttributionResult:
        """Run OLS regression for factor attribution."""
        try:
            from scipy import stats

            # Add intercept
            X_with_const = X.copy()
            X_with_const["const"] = 1.0

            # OLS regression using numpy
            X_matrix = X_with_const.values
            y_vector = y.values

            # Beta = (X'X)^-1 X'y
            XtX = X_matrix.T @ X_matrix
            XtX_inv = np.linalg.pinv(XtX)
            beta = XtX_inv @ (X_matrix.T @ y_vector)

            # Predictions and residuals
            y_pred = X_matrix @ beta
            residuals = y_vector - y_pred

            # Calculate R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_vector - y_vector.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Adjusted R-squared
            n = len(y_vector)
            p = len(factor_names)
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else 0

            # Standard errors
            mse = ss_res / (n - p - 1) if n > p + 1 else 0
            se = np.sqrt(np.diag(XtX_inv) * mse)

            # t-statistics and p-values
            t_stats = beta / (se + 1e-10)
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))

            # Build factor contributions
            contributions = {}
            total_pnl = y_vector.sum()

            for i, factor_name in enumerate(factor_names):
                coef = beta[i]
                factor_values = X[factor_name].values

                # Contribution = coefficient * mean factor value * number of trades
                contribution = coef * factor_values.sum()
                contribution_pct = (contribution / total_pnl * 100) if total_pnl != 0 else 0

                # Correlation with PnL
                correlation = np.corrcoef(factor_values, y_vector)[0, 1]

                factor_type = self.STANDARD_FACTORS.get(factor_name, FactorType.CUSTOM)

                contributions[factor_name] = FactorContribution(
                    factor_type=factor_type,
                    factor_name=factor_name,
                    contribution=contribution,
                    contribution_pct=contribution_pct,
                    coefficient=coef,
                    t_statistic=t_stats[i],
                    p_value=p_values[i],
                    avg_factor_value=factor_values.mean(),
                    correlation_with_pnl=correlation if not np.isnan(correlation) else 0,
                )

            # Calculate residual PnL
            residual_pnl = total_pnl - sum(fc.contribution for fc in contributions.values())

            return AttributionResult(
                r_squared=r_squared,
                adj_r_squared=adj_r_squared,
                factor_contributions=contributions,
                residual_pnl=residual_pnl,
            )

        except Exception as e:
            logger.error(f"Regression failed: {e}")
            return AttributionResult()

    def analyze_with_shap(
        self,
        trades_df: pd.DataFrame,
        pnl_column: str = "pnl",
        factor_columns: Optional[List[str]] = None,
    ) -> AttributionResult:
        """
        Perform SHAP-based factor attribution (requires shap package).

        More sophisticated than OLS, captures non-linear relationships.
        """
        try:
            import shap
            from sklearn.ensemble import GradientBoostingRegressor
        except ImportError:
            logger.warning("SHAP or sklearn not available, falling back to OLS")
            return self.analyze(trades_df, pnl_column, factor_columns)

        if len(trades_df) < self.min_trades:
            return AttributionResult(
                total_trades=len(trades_df),
                total_pnl=trades_df[pnl_column].sum() if pnl_column in trades_df.columns else 0,
            )

        # Auto-detect factor columns
        if factor_columns is None:
            factor_columns = [
                col for col in trades_df.columns
                if col in self.STANDARD_FACTORS or col.startswith("factor_")
            ]

        if not factor_columns:
            return AttributionResult()

        # Prepare data
        data = trades_df[[pnl_column] + factor_columns].dropna()
        X = data[factor_columns].values
        y = data[pnl_column].values

        # Train gradient boosting model
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X, y)

        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Aggregate SHAP values per factor
        contributions = {}
        total_pnl = y.sum()

        for i, factor_name in enumerate(factor_columns):
            factor_shap = shap_values[:, i]
            contribution = factor_shap.sum()
            contribution_pct = (contribution / total_pnl * 100) if total_pnl != 0 else 0

            factor_type = self.STANDARD_FACTORS.get(factor_name, FactorType.CUSTOM)

            contributions[factor_name] = FactorContribution(
                factor_type=factor_type,
                factor_name=factor_name,
                contribution=contribution,
                contribution_pct=contribution_pct,
                coefficient=0,  # SHAP doesn't use coefficients
                t_statistic=0,
                p_value=0,
                avg_factor_value=X[:, i].mean(),
                correlation_with_pnl=np.corrcoef(X[:, i], y)[0, 1],
            )

        return AttributionResult(
            total_trades=len(data),
            total_pnl=total_pnl,
            r_squared=model.score(X, y),
            factor_contributions=contributions,
        )

    def generate_waterfall_data(self, result: AttributionResult) -> List[Dict[str, Any]]:
        """
        Generate data for waterfall chart visualization.

        Returns list of {name, value, running_total} for each factor.
        """
        waterfall = []
        running_total = 0

        # Sort by absolute contribution (largest first)
        sorted_factors = sorted(
            result.factor_contributions.items(),
            key=lambda x: abs(x[1].contribution),
            reverse=True,
        )

        for name, fc in sorted_factors:
            if abs(fc.contribution) < 0.01:  # Skip tiny contributions
                continue

            running_total += fc.contribution
            waterfall.append({
                "factor": name,
                "contribution": fc.contribution,
                "running_total": running_total,
                "is_positive": fc.contribution > 0,
            })

        # Add residual
        if abs(result.residual_pnl) > 0.01:
            running_total += result.residual_pnl
            waterfall.append({
                "factor": "Residual",
                "contribution": result.residual_pnl,
                "running_total": running_total,
                "is_positive": result.residual_pnl > 0,
            })

        return waterfall


# Default instance
DEFAULT_FACTOR_ATTRIBUTION = FactorAttribution()
