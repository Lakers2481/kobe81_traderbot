"""
Factor Validator - Alphalens Integration for IC/Quantile Analysis

Validates alpha factors using industry-standard metrics:
- Information Coefficient (IC)
- Quantile returns (Q1-Q5 spreads)
- Turnover analysis
- Statistical significance testing

USAGE:
    from research.factor_validator import FactorValidator

    validator = FactorValidator(prices_df)

    # Validate a single factor
    report = validator.validate_factor(alpha_series, name='momentum_20d')

    # Generate full tearsheet
    validator.generate_tearsheet(alpha_series, output_dir='reports/tearsheets/')

Created: 2026-01-07
Based on: Alphalens-reloaded (Quantopian fork)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import Alphalens
try:
    import alphalens
    from alphalens.utils import get_clean_factor_and_forward_returns
    from alphalens.performance import (
        factor_information_coefficient,
        mean_return_by_quantile,
        factor_returns,
        factor_alpha_beta,
    )
    from alphalens.plotting import (
        plot_quantile_returns_bar,
        plot_ic_ts,
        plot_ic_hist,
    )
    # plot_turnover_by_quantile may not exist in all versions
    try:
        from alphalens.plotting import plot_turnover_by_quantile
    except ImportError:
        plot_turnover_by_quantile = None
    HAS_ALPHALENS = True
except ImportError:
    HAS_ALPHALENS = False
    logger.warning("Alphalens not installed. Run: pip install alphalens-reloaded")


@dataclass
class FactorReport:
    """Comprehensive factor validation report."""
    name: str
    timestamp: str

    # Information Coefficient metrics
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_sharpe: float = 0.0  # IC / std(IC) * sqrt(252)
    ic_pvalue: float = 1.0

    # Quantile returns
    q1_return: float = 0.0  # Bottom quintile
    q5_return: float = 0.0  # Top quintile
    q5_q1_spread: float = 0.0  # Long-short spread
    monotonic: bool = False  # Is return monotonic across quintiles?

    # Forward return periods tested
    periods: List[int] = field(default_factory=lambda: [1, 5, 10])

    # Turnover metrics
    turnover_1d: float = 0.0
    turnover_5d: float = 0.0

    # Statistical significance
    t_stat: float = 0.0
    significant_5pct: bool = False
    significant_1pct: bool = False

    # Sample info
    num_observations: int = 0
    date_start: str = ""
    date_end: str = ""

    # Raw data for plotting
    ic_series: Optional[pd.Series] = None
    quantile_returns: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large data)."""
        return {
            'name': self.name,
            'timestamp': self.timestamp,
            'ic_mean': self.ic_mean,
            'ic_std': self.ic_std,
            'ic_sharpe': self.ic_sharpe,
            'ic_pvalue': self.ic_pvalue,
            'q1_return': self.q1_return,
            'q5_return': self.q5_return,
            'q5_q1_spread': self.q5_q1_spread,
            'monotonic': self.monotonic,
            'periods': self.periods,
            'turnover_1d': self.turnover_1d,
            'turnover_5d': self.turnover_5d,
            't_stat': self.t_stat,
            'significant_5pct': self.significant_5pct,
            'significant_1pct': self.significant_1pct,
            'num_observations': self.num_observations,
            'date_start': self.date_start,
            'date_end': self.date_end,
        }

    def is_valid_alpha(self, min_ic: float = 0.02, min_spread: float = 0.001) -> bool:
        """Check if factor meets minimum quality thresholds."""
        return (
            abs(self.ic_mean) >= min_ic and
            abs(self.q5_q1_spread) >= min_spread and
            self.significant_5pct
        )


class FactorValidator:
    """
    Validate alpha factors using Alphalens methodology.

    Generates comprehensive reports including:
    - Information Coefficient (IC) analysis
    - Quantile return spreads
    - Turnover analysis
    - Statistical significance testing
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        quantiles: int = 5,
        periods: Tuple[int, ...] = (1, 5, 10),
    ):
        """
        Initialize the factor validator.

        Args:
            prices: DataFrame with columns [timestamp, symbol, close]
                   OR multi-index DataFrame with symbols as columns
            quantiles: Number of quantiles for analysis (default 5 = quintiles)
            periods: Forward return periods to analyze (in days)
        """
        self.quantiles = quantiles
        self.periods = periods
        self.reports: List[FactorReport] = []

        # Convert to Alphalens-friendly format (multi-index with date, symbol)
        if 'symbol' in prices.columns:
            self.prices = prices.pivot(index='timestamp', columns='symbol', values='close')
        else:
            self.prices = prices

        # Ensure datetime index
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            self.prices.index = pd.to_datetime(self.prices.index)

        logger.info(f"FactorValidator initialized: {len(self.prices)} dates, {self.prices.shape[1]} symbols")

    def validate_factor(
        self,
        factor: pd.Series,
        name: str = "unnamed_factor",
        by_group: Optional[pd.Series] = None,
    ) -> FactorReport:
        """
        Validate a single alpha factor.

        Args:
            factor: Series with multi-index (date, symbol) containing factor values
            name: Name for the factor
            by_group: Optional group labels for sector analysis

        Returns:
            FactorReport with comprehensive metrics
        """
        report = FactorReport(
            name=name,
            timestamp=datetime.now().isoformat(),
            periods=list(self.periods),
        )

        if not HAS_ALPHALENS:
            logger.warning("Alphalens not available, using fallback validation")
            return self._fallback_validation(factor, name, report)

        try:
            # Prepare data for Alphalens
            factor_data = get_clean_factor_and_forward_returns(
                factor,
                self.prices,
                quantiles=self.quantiles,
                periods=self.periods,
                max_loss=0.35,  # Allow up to 35% data loss
            )

            # Calculate Information Coefficient
            ic = factor_information_coefficient(factor_data)

            report.ic_mean = float(ic.mean().iloc[0])
            report.ic_std = float(ic.std().iloc[0])
            report.ic_sharpe = report.ic_mean / (report.ic_std + 1e-8) * np.sqrt(252)

            # Statistical significance
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(ic.iloc[:, 0].dropna(), 0)
            report.t_stat = float(t_stat)
            report.ic_pvalue = float(p_value)
            report.significant_5pct = p_value < 0.05
            report.significant_1pct = p_value < 0.01

            # Quantile returns
            quantile_returns = mean_return_by_quantile(factor_data)[0]

            if len(quantile_returns) >= self.quantiles:
                # Get first period returns
                period_col = quantile_returns.columns[0]
                q_returns = quantile_returns[period_col]

                report.q1_return = float(q_returns.iloc[0])  # Bottom quintile
                report.q5_return = float(q_returns.iloc[-1])  # Top quintile
                report.q5_q1_spread = report.q5_return - report.q1_return

                # Check monotonicity
                report.monotonic = self._check_monotonic(q_returns)

                report.quantile_returns = quantile_returns

            # IC series for plotting
            report.ic_series = ic.iloc[:, 0]

            # Sample info
            report.num_observations = len(factor_data)
            report.date_start = str(factor_data.index.get_level_values(0).min().date())
            report.date_end = str(factor_data.index.get_level_values(0).max().date())

            # Turnover (if we have multiple periods)
            try:
                from alphalens.performance import factor_autocorrelation
                autocorr = factor_autocorrelation(factor_data, period=1)
                report.turnover_1d = float(1 - autocorr.mean())
            except:
                pass

        except Exception as e:
            logger.error(f"Factor validation failed: {e}")
            return self._fallback_validation(factor, name, report)

        self.reports.append(report)
        return report

    def validate_multiple(
        self,
        factors: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Validate multiple factors and return comparison table.

        Args:
            factors: Dictionary mapping factor names to factor Series

        Returns:
            DataFrame comparing all factors
        """
        results = []

        for name, factor in factors.items():
            logger.info(f"Validating factor: {name}")
            report = self.validate_factor(factor, name)
            results.append(report.to_dict())

        df = pd.DataFrame(results)

        # Sort by IC Sharpe
        if 'ic_sharpe' in df.columns:
            df = df.sort_values('ic_sharpe', ascending=False)

        return df

    def generate_tearsheet(
        self,
        factor: pd.Series,
        name: str = "factor",
        output_dir: str = "reports/factor_tearsheets",
        save_plots: bool = True,
    ) -> FactorReport:
        """
        Generate full tearsheet with plots.

        Args:
            factor: Factor Series to analyze
            name: Factor name
            output_dir: Directory for output files
            save_plots: Whether to save plots as images

        Returns:
            FactorReport with full analysis
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # First validate
        report = self.validate_factor(factor, name)

        # Create output directory
        output_path = Path(output_dir) / name
        output_path.mkdir(parents=True, exist_ok=True)

        if not HAS_ALPHALENS:
            # Save basic report
            with open(output_path / 'report.json', 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            return report

        try:
            # Prepare data for plotting
            factor_data = get_clean_factor_and_forward_returns(
                factor,
                self.prices,
                quantiles=self.quantiles,
                periods=self.periods,
                max_loss=0.35,
            )

            if save_plots:
                # IC time series
                fig, ax = plt.subplots(figsize=(12, 4))
                plot_ic_ts(factor_information_coefficient(factor_data), ax=ax)
                plt.tight_layout()
                plt.savefig(output_path / 'ic_timeseries.png', dpi=100)
                plt.close()

                # IC histogram
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_ic_hist(factor_information_coefficient(factor_data), ax=ax)
                plt.tight_layout()
                plt.savefig(output_path / 'ic_histogram.png', dpi=100)
                plt.close()

                # Quantile returns bar
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_quantile_returns_bar(mean_return_by_quantile(factor_data)[0], ax=ax)
                plt.tight_layout()
                plt.savefig(output_path / 'quantile_returns.png', dpi=100)
                plt.close()

                logger.info(f"Saved tearsheet plots to {output_path}")

        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")

        # Save JSON report
        with open(output_path / 'report.json', 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        # Save summary markdown
        self._save_markdown_summary(report, output_path / 'README.md')

        return report

    def get_best_factors(
        self,
        n: int = 10,
        metric: str = 'ic_sharpe',
        min_observations: int = 100,
    ) -> pd.DataFrame:
        """
        Get top performing factors from validation history.

        Args:
            n: Number of top factors to return
            metric: Metric to rank by
            min_observations: Minimum observations required

        Returns:
            DataFrame with top factors
        """
        if not self.reports:
            return pd.DataFrame()

        df = pd.DataFrame([r.to_dict() for r in self.reports])

        # Filter by minimum observations
        df = df[df['num_observations'] >= min_observations]

        # Sort by metric
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)

        return df.head(n)

    # ========== PRIVATE METHODS ==========

    def _check_monotonic(self, quantile_returns: pd.Series) -> bool:
        """Check if returns are monotonically increasing across quantiles."""
        values = quantile_returns.values
        increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
        decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
        return increasing or decreasing

    def _fallback_validation(
        self,
        factor: pd.Series,
        name: str,
        report: FactorReport,
    ) -> FactorReport:
        """Basic validation without Alphalens."""
        try:
            # Simple IC calculation
            if isinstance(factor.index, pd.MultiIndex):
                dates = factor.index.get_level_values(0).unique()
            else:
                dates = factor.index.unique()

            # Calculate forward returns
            forward_returns = self.prices.pct_change().shift(-1)

            # Simple correlation
            ic_values = []
            for date in dates[:100]:  # Sample for speed
                try:
                    if isinstance(factor.index, pd.MultiIndex):
                        f = factor.loc[date]
                    else:
                        f = factor.loc[date] if date in factor.index else None

                    if f is not None and date in forward_returns.index:
                        r = forward_returns.loc[date]
                        corr = f.corr(r) if hasattr(f, 'corr') else 0
                        if not np.isnan(corr):
                            ic_values.append(corr)
                except:
                    pass

            if ic_values:
                report.ic_mean = float(np.mean(ic_values))
                report.ic_std = float(np.std(ic_values))
                report.ic_sharpe = report.ic_mean / (report.ic_std + 1e-8) * np.sqrt(252)

            report.num_observations = len(factor)
            report.date_start = str(dates.min())
            report.date_end = str(dates.max())

        except Exception as e:
            logger.warning(f"Fallback validation failed: {e}")

        return report

    def _save_markdown_summary(self, report: FactorReport, filepath: Path):
        """Save markdown summary of factor report."""
        content = f"""# Factor Analysis: {report.name}

Generated: {report.timestamp}

## Summary Statistics

| Metric | Value |
|--------|-------|
| IC Mean | {report.ic_mean:.4f} |
| IC Std | {report.ic_std:.4f} |
| IC Sharpe | {report.ic_sharpe:.2f} |
| T-Stat | {report.t_stat:.2f} |
| P-Value | {report.ic_pvalue:.4f} |
| Significant (5%) | {'Yes' if report.significant_5pct else 'No'} |

## Quantile Analysis

| Quantile | Return |
|----------|--------|
| Q1 (Bottom) | {report.q1_return:.4f} |
| Q5 (Top) | {report.q5_return:.4f} |
| Q5-Q1 Spread | {report.q5_q1_spread:.4f} |
| Monotonic | {'Yes' if report.monotonic else 'No'} |

## Data Info

- Observations: {report.num_observations:,}
- Date Range: {report.date_start} to {report.date_end}
- Periods Tested: {report.periods}

## Quality Assessment

**Valid Alpha:** {'YES' if report.is_valid_alpha() else 'NO'}

Criteria:
- IC Mean >= 0.02: {'PASS' if abs(report.ic_mean) >= 0.02 else 'FAIL'}
- Q5-Q1 Spread >= 0.1%: {'PASS' if abs(report.q5_q1_spread) >= 0.001 else 'FAIL'}
- Statistically Significant: {'PASS' if report.significant_5pct else 'FAIL'}
"""

        with open(filepath, 'w') as f:
            f.write(content)


def quick_validate(
    factor: pd.Series,
    prices: pd.DataFrame,
    name: str = "factor",
) -> FactorReport:
    """
    Quick factor validation.

    Args:
        factor: Factor values (multi-index: date, symbol)
        prices: Price DataFrame
        name: Factor name

    Returns:
        FactorReport
    """
    validator = FactorValidator(prices)
    return validator.validate_factor(factor, name)


if __name__ == '__main__':
    # Test the validator
    print(f"Alphalens available: {HAS_ALPHALENS}")

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    np.random.seed(42)

    # Create prices
    prices_list = []
    for symbol in symbols:
        price_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'close': 100 + np.random.randn(500).cumsum(),
        })
        prices_list.append(price_data)

    prices = pd.concat(prices_list, ignore_index=True)

    # Create sample factor (momentum)
    pivot_prices = prices.pivot(index='timestamp', columns='symbol', values='close')
    momentum = pivot_prices.pct_change(20)
    factor = momentum.stack()
    factor.index.names = ['date', 'asset']

    print(f"\nFactor shape: {factor.shape}")
    print(f"Factor sample:\n{factor.head()}")

    # Validate
    validator = FactorValidator(prices)
    report = validator.validate_factor(factor, name='momentum_20d')

    print(f"\n=== Factor Report: {report.name} ===")
    print(f"IC Mean: {report.ic_mean:.4f}")
    print(f"IC Sharpe: {report.ic_sharpe:.2f}")
    print(f"Q5-Q1 Spread: {report.q5_q1_spread:.4f}")
    print(f"Valid Alpha: {report.is_valid_alpha()}")
