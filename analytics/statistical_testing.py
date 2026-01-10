"""
Statistical Testing Module for Pattern Optimization

Implements professional quant-grade statistical methods for validating trading patterns:
- Binomial tests with Bonferroni correction for multiple testing (FWER control)
- Benjamini-Hochberg FDR correction for multiple testing (FDR control)
- Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
- Wilson confidence intervals for win rates

Based on academic research:
- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality"
- Benjamini, Y., & Hochberg, Y. (1995). "Controlling the False Discovery Rate:
  A Practical and Powerful Approach to Multiple Testing"
- Brown, L. D., Cai, T. T., & DasGupta, A. (2001). "Interval Estimation for
  a Binomial Proportion"

Author: Kobe Trading System
Date: 2026-01-08
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from scipy.stats import norm

# Handle scipy version compatibility (binom_test deprecated in scipy 1.7+)
try:
    from scipy.stats import binomtest
    def binom_test(x, n, p, alternative='two-sided'):
        """Wrapper for scipy.stats.binomtest (scipy >= 1.7)."""
        return binomtest(x, n, p, alternative=alternative).pvalue
except ImportError:
    from scipy.stats import binom_test


@dataclass
class StatisticalTestResult:
    """Results from statistical significance testing."""

    p_value: float
    """P-value from statistical test (lower is better)."""

    alpha: float
    """Significance level used."""

    alpha_adjusted: Optional[float]
    """Bonferroni-corrected significance level (if n_trials > 1)."""

    is_significant: bool
    """True if p_value < alpha_adjusted."""

    test_statistic: Optional[float] = None
    """Test statistic value (if applicable)."""

    n_trials: int = 1
    """Number of patterns tested (for multiple testing correction)."""


@dataclass
class DeflatedSharpeResult:
    """Results from Deflated Sharpe Ratio calculation."""

    deflated_sharpe: float
    """Deflated Sharpe Ratio (adjusted for multiple testing)."""

    sharpe_ratio: float
    """Raw Sharpe Ratio (unadjusted)."""

    sharpe_threshold: float
    """Expected max Sharpe from n_trials (under null hypothesis)."""

    standard_error: float
    """Standard error of Sharpe Ratio."""

    n_trials: int
    """Number of trials (patterns tested)."""

    n_observations: int
    """Number of return observations."""


@dataclass
class WilsonConfidenceInterval:
    """Wilson confidence interval for binomial proportion."""

    lower_bound: float
    """Lower bound of confidence interval."""

    upper_bound: float
    """Upper bound of confidence interval."""

    point_estimate: float
    """Point estimate (win_rate)."""

    confidence_level: float
    """Confidence level (e.g., 0.95 for 95% CI)."""

    sample_size: int
    """Total number of trials."""


@dataclass
class FDRResult:
    """Results from Benjamini-Hochberg False Discovery Rate correction."""

    significant: np.ndarray
    """Boolean array indicating which tests are significant after FDR control."""

    alpha: float
    """Desired FDR level (e.g., 0.05)."""

    critical_values: np.ndarray
    """Critical p-value threshold for each test (rank-based)."""

    n_significant: int
    """Number of tests declared significant."""

    n_total: int
    """Total number of tests."""

    rejection_threshold: Optional[float] = None
    """Largest p-value that was rejected (if any)."""


def benjamini_hochberg_fdr(
    p_values: list,
    alpha: float = 0.05
) -> FDRResult:
    """
    Apply Benjamini-Hochberg False Discovery Rate (FDR) correction.

    Controls the expected proportion of false discoveries among all discoveries.
    Less conservative than Bonferroni (family-wise error rate control).

    Procedure (Benjamini & Hochberg, 1995):
    1. Sort p-values in ascending order: p(1) <= p(2) <= ... <= p(m)
    2. For each rank i, compute threshold: (i/m) * alpha
    3. Find largest i where p(i) <= (i/m) * alpha
    4. Reject H0 for all tests 1, 2, ..., i

    Args:
        p_values: List of p-values from multiple tests
        alpha: Desired FDR level (default 0.05 = 5% expected false discoveries)

    Returns:
        FDRResult with significance mask and critical values

    Example:
        >>> # Test 294 parameter combinations
        >>> p_vals = [0.001, 0.023, 0.045, 0.078, 0.120, ...]
        >>> result = benjamini_hochberg_fdr(p_vals, alpha=0.05)
        >>> print(f"Significant: {result.n_significant}/{result.n_total}")
        >>> sig_indices = np.where(result.significant)[0]

    Reference:
        Benjamini, Y., & Hochberg, Y. (1995). "Controlling the False Discovery Rate:
        A Practical and Powerful Approach to Multiple Testing."
        Journal of the Royal Statistical Society, Series B, 57(1), 289-300.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    if len(p_values) == 0:
        raise ValueError("p_values cannot be empty")

    # Convert to numpy array and handle edge cases
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    # Check for invalid p-values
    if np.any(np.isnan(p_values)):
        raise ValueError("p_values contains NaN values")

    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("All p-values must be in [0, 1]")

    # Create array of original indices to track reordering
    original_indices = np.arange(m)

    # Sort p-values and track original positions
    sort_idx = np.argsort(p_values)
    p_sorted = p_values[sort_idx]
    original_indices_sorted = original_indices[sort_idx]

    # Compute critical values: (i/m) * alpha for i = 1, 2, ..., m
    ranks = np.arange(1, m + 1)
    critical_values_sorted = (ranks / m) * alpha

    # Find largest i where p(i) <= (i/m) * alpha
    # Equivalently: p_sorted <= critical_values_sorted
    comparisons = p_sorted <= critical_values_sorted

    if np.any(comparisons):
        # Find the largest index where condition holds
        max_idx = np.where(comparisons)[0][-1]
        rejection_threshold = p_sorted[max_idx]

        # All tests up to max_idx are significant
        significant_sorted = np.zeros(m, dtype=bool)
        significant_sorted[:max_idx + 1] = True
    else:
        # No tests are significant
        significant_sorted = np.zeros(m, dtype=bool)
        rejection_threshold = None

    # Unsort to match original order
    significant = np.zeros(m, dtype=bool)
    critical_values = np.zeros(m, dtype=float)

    significant[original_indices_sorted] = significant_sorted
    critical_values[original_indices_sorted] = critical_values_sorted

    n_significant = int(np.sum(significant))

    return FDRResult(
        significant=significant,
        alpha=alpha,
        critical_values=critical_values,
        n_significant=n_significant,
        n_total=m,
        rejection_threshold=rejection_threshold
    )


def compute_binomial_pvalue(
    wins: int,
    total: int,
    null_prob: float = 0.5,
    alpha: float = 0.05,
    n_trials: int = 1,
    alternative: str = "greater"
) -> StatisticalTestResult:
    """
    Compute statistical significance of win rate using binomial test.

    Tests the null hypothesis that the true win rate equals null_prob (default 50%).
    Applies Bonferroni correction when testing multiple patterns.

    Args:
        wins: Number of successful outcomes (e.g., up days after pattern)
        total: Total number of pattern occurrences
        null_prob: Null hypothesis probability (default 0.5 for random)
        alpha: Significance level before correction (default 0.05)
        n_trials: Number of patterns tested (for Bonferroni correction)
        alternative: "greater" (win_rate > null_prob) or "two-sided"

    Returns:
        StatisticalTestResult with p-value and significance determination

    Example:
        >>> # Test if 23 wins out of 33 trials is significant
        >>> result = compute_binomial_pvalue(wins=23, total=33, n_trials=9)
        >>> print(f"p-value: {result.p_value:.4f}")
        >>> print(f"Significant: {result.is_significant}")
    """
    if total == 0:
        raise ValueError("total must be greater than 0")

    if wins > total:
        raise ValueError(f"wins ({wins}) cannot exceed total ({total})")

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    # Compute binomial test p-value
    p_value = binom_test(wins, total, p=null_prob, alternative=alternative)

    # Apply Bonferroni correction
    alpha_adjusted = alpha / n_trials if n_trials > 1 else alpha

    # Determine significance
    is_significant = p_value < alpha_adjusted

    return StatisticalTestResult(
        p_value=p_value,
        alpha=alpha,
        alpha_adjusted=alpha_adjusted,
        is_significant=is_significant,
        n_trials=n_trials
    )


def bonferroni_correction(alpha: float, n_trials: int) -> float:
    """
    Apply Bonferroni correction for multiple hypothesis testing.

    Adjusts the significance level to control family-wise error rate (FWER)
    when testing multiple patterns simultaneously.

    Args:
        alpha: Desired family-wise significance level (e.g., 0.05)
        n_trials: Number of independent tests (patterns tested)

    Returns:
        Adjusted alpha for individual tests

    Example:
        >>> # Testing 9 patterns, want overall 5% significance
        >>> alpha_adj = bonferroni_correction(0.05, 9)
        >>> print(f"Adjusted alpha: {alpha_adj:.4f}")  # 0.0056
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    return alpha / n_trials


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> DeflatedSharpeResult:
    """
    Calculate Deflated Sharpe Ratio adjusting for multiple testing bias.

    Based on Bailey & López de Prado (2014):
    "The Deflated Sharpe Ratio: Correcting for Selection Bias,
     Backtest Overfitting and Non-Normality"

    The DSR adjusts for the fact that when testing multiple strategies,
    the best-performing one is likely inflated due to selection bias.

    Formula:
        DSR = (SR - SR_threshold) / SE(SR)

        where:
        - SR = observed Sharpe Ratio
        - SR_threshold = expected max SR from n_trials under null
        - SE(SR) = standard error accounting for skewness/kurtosis

    Args:
        returns: Array of strategy returns (NOT annualized)
        n_trials: Number of strategies/patterns tested
        risk_free_rate: Annual risk-free rate (default 0.0)
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        DeflatedSharpeResult with DSR, raw SR, and threshold

    Example:
        >>> returns = np.random.randn(1000) * 0.01  # Daily returns
        >>> result = deflated_sharpe_ratio(returns, n_trials=9)
        >>> print(f"DSR: {result.deflated_sharpe:.2f}")
        >>> print(f"Raw SR: {result.sharpe_ratio:.2f}")
    """
    if len(returns) < 10:
        raise ValueError(f"Need at least 10 returns, got {len(returns)}")

    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    # Convert to numpy array
    returns = np.asarray(returns)

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) < 10:
        raise ValueError(f"After removing NaN, only {len(returns)} returns remain")

    # Calculate annualized Sharpe Ratio
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)  # Use sample std

    if std_return == 0:
        raise ValueError("Returns have zero variance")

    # Annualized Sharpe Ratio
    sharpe_ratio = (mean_return * periods_per_year - risk_free_rate) / (std_return * np.sqrt(periods_per_year))

    # Calculate skewness and kurtosis
    from scipy.stats import skew, kurtosis
    skewness = skew(returns)
    kurt = kurtosis(returns)  # Excess kurtosis

    # Number of observations
    n = len(returns)

    # Standard error of Sharpe Ratio (accounts for non-normality)
    # Formula from Bailey & López de Prado (2014), Equation 7
    se_sr = np.sqrt(
        (1 + 0.5 * sharpe_ratio**2 - skewness * sharpe_ratio + (kurt / 4) * sharpe_ratio**2) / n
    )

    # Sharpe Ratio threshold (expected max from n_trials under null)
    # Approximation using inverse normal CDF
    # Under null hypothesis (SR=0), max of n_trials ~ N(0,1)
    if n_trials == 1:
        sr_threshold = 0.0
    else:
        # Expected value of max of n_trials standard normals
        # Approximation: E[max(Z1,...,Zn)] ≈ Φ^(-1)(1 - 1/n)
        sr_threshold = norm.ppf(1 - 1 / n_trials)

    # Deflated Sharpe Ratio
    if se_sr == 0:
        deflated_sr = 0.0
    else:
        deflated_sr = (sharpe_ratio - sr_threshold) / se_sr

    return DeflatedSharpeResult(
        deflated_sharpe=deflated_sr,
        sharpe_ratio=sharpe_ratio,
        sharpe_threshold=sr_threshold,
        standard_error=se_sr,
        n_trials=n_trials,
        n_observations=n
    )


def wilson_confidence_interval(
    wins: int,
    total: int,
    confidence_level: float = 0.95
) -> WilsonConfidenceInterval:
    """
    Calculate Wilson confidence interval for binomial proportion.

    More accurate than normal approximation, especially for small samples
    or extreme proportions (near 0 or 1).

    Based on Wilson (1927) and recommended by Brown et al. (2001).

    Args:
        wins: Number of successes
        total: Total number of trials
        confidence_level: Confidence level (default 0.95 for 95% CI)

    Returns:
        WilsonConfidenceInterval with lower/upper bounds

    Example:
        >>> # 23 wins out of 33 trials
        >>> ci = wilson_confidence_interval(23, 33, 0.95)
        >>> print(f"Win rate: {ci.point_estimate:.1%}")
        >>> print(f"95% CI: [{ci.lower_bound:.1%}, {ci.upper_bound:.1%}]")
    """
    if total == 0:
        raise ValueError("total must be greater than 0")

    if wins > total:
        raise ValueError(f"wins ({wins}) cannot exceed total ({total})")

    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be between 0 and 1, got {confidence_level}")

    # Point estimate
    p_hat = wins / total

    # Z-score for confidence level
    alpha = 1 - confidence_level
    z = norm.ppf(1 - alpha / 2)

    # Wilson score interval
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))) / denominator

    lower_bound = max(0.0, center - margin)
    upper_bound = min(1.0, center + margin)

    return WilsonConfidenceInterval(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        point_estimate=p_hat,
        confidence_level=confidence_level,
        sample_size=total
    )


def interpret_deflated_sharpe(dsr: float) -> str:
    """
    Interpret Deflated Sharpe Ratio value.

    Args:
        dsr: Deflated Sharpe Ratio

    Returns:
        Human-readable interpretation

    Example:
        >>> interpretation = interpret_deflated_sharpe(1.5)
        >>> print(interpretation)  # "Significant - pattern survives multiple testing"
    """
    if dsr >= 2.0:
        return "Highly significant - strong evidence even after multiple testing corrections"
    elif dsr >= 1.5:
        return "Significant - pattern survives multiple testing with good confidence"
    elif dsr >= 1.0:
        return "Marginally significant - pattern may be real but needs more validation"
    elif dsr >= 0.0:
        return "Not significant - pattern likely due to chance/overfitting"
    else:
        return "Negative DSR - pattern performs worse than expected by chance"


def interpret_pvalue(p_value: float, alpha_adjusted: float) -> str:
    """
    Interpret p-value relative to adjusted significance level.

    Args:
        p_value: Observed p-value
        alpha_adjusted: Bonferroni-corrected significance level

    Returns:
        Human-readable interpretation

    Example:
        >>> interpretation = interpret_pvalue(0.001, 0.0056)
        >>> print(interpretation)  # "Highly significant - strong evidence"
    """
    if p_value < alpha_adjusted / 10:
        return "Highly significant - very strong evidence against null hypothesis"
    elif p_value < alpha_adjusted:
        return "Significant - pattern is statistically validated"
    elif p_value < alpha_adjusted * 2:
        return "Marginally significant - borderline, needs more data"
    else:
        return "Not significant - insufficient evidence, likely random chance"


def interpret_fdr_result(fdr_result: FDRResult) -> str:
    """
    Interpret Benjamini-Hochberg FDR result.

    Args:
        fdr_result: FDRResult from benjamini_hochberg_fdr()

    Returns:
        Human-readable interpretation with discovery statistics

    Example:
        >>> result = benjamini_hochberg_fdr(p_values, alpha=0.05)
        >>> interpretation = interpret_fdr_result(result)
        >>> print(interpretation)
    """
    discovery_rate = (fdr_result.n_significant / fdr_result.n_total) * 100

    if fdr_result.n_significant == 0:
        return (
            f"No discoveries at FDR = {fdr_result.alpha:.1%}. "
            f"All {fdr_result.n_total} tests failed to reject null hypothesis. "
            "Either no true effects exist or sample size is insufficient."
        )
    elif fdr_result.n_significant == fdr_result.n_total:
        return (
            f"All {fdr_result.n_total} tests significant at FDR = {fdr_result.alpha:.1%}. "
            f"Exceptionally strong evidence - all hypotheses rejected. "
            f"Threshold p-value: {fdr_result.rejection_threshold:.4f}."
        )
    elif discovery_rate >= 50:
        return (
            f"High discovery rate: {fdr_result.n_significant}/{fdr_result.n_total} "
            f"({discovery_rate:.1f}%) significant at FDR = {fdr_result.alpha:.1%}. "
            f"Strong evidence of systematic effects. "
            f"Threshold p-value: {fdr_result.rejection_threshold:.4f}."
        )
    elif discovery_rate >= 20:
        return (
            f"Moderate discovery rate: {fdr_result.n_significant}/{fdr_result.n_total} "
            f"({discovery_rate:.1f}%) significant at FDR = {fdr_result.alpha:.1%}. "
            f"Evidence of real effects among tested hypotheses. "
            f"Threshold p-value: {fdr_result.rejection_threshold:.4f}."
        )
    elif discovery_rate >= 5:
        return (
            f"Low discovery rate: {fdr_result.n_significant}/{fdr_result.n_total} "
            f"({discovery_rate:.1f}%) significant at FDR = {fdr_result.alpha:.1%}. "
            f"Sparse evidence - most hypotheses fail. "
            f"Threshold p-value: {fdr_result.rejection_threshold:.4f}."
        )
    else:
        return (
            f"Minimal discoveries: {fdr_result.n_significant}/{fdr_result.n_total} "
            f"({discovery_rate:.1f}%) significant at FDR = {fdr_result.alpha:.1%}. "
            f"Very limited evidence. Validate thoroughly before acting. "
            f"Threshold p-value: {fdr_result.rejection_threshold:.4f}."
        )
