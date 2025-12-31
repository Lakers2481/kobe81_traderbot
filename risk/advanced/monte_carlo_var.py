"""
Monte Carlo Value at Risk (VaR) Simulation Module.

Advanced VaR calculation using Monte Carlo simulation with:
- Correlated returns using Cholesky decomposition
- Multiple confidence levels (95%, 99%)
- Conditional VaR (CVaR/Expected Shortfall)
- Stress testing scenarios
- Portfolio risk metrics

MERGED FROM GAME_PLAN_2K28 - Production Ready
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky

logger = logging.getLogger(__name__)


class StressScenario(Enum):
    """Predefined stress test scenarios."""
    MARKET_CRASH = "market_crash"
    SECTOR_ROTATION = "sector_rotation"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    CUSTOM = "custom"


@dataclass
class VaRResult:
    """Result of VaR calculation."""
    var: float  # Value at Risk (dollar amount)
    cvar: float  # Conditional VaR (expected shortfall)
    var_pct: float  # VaR as percentage of portfolio
    cvar_pct: float  # CVaR as percentage
    simulations: int  # Number of simulations run
    confidence_level: float  # Confidence level used
    horizon_days: int  # Risk horizon in days
    portfolio_value: float  # Current portfolio value
    worst_case: float  # Worst simulated loss
    best_case: float  # Best simulated gain
    percentile_5: float  # 5th percentile outcome
    percentile_95: float  # 95th percentile outcome
    mean_outcome: float  # Mean simulated outcome


@dataclass
class StressTestResult:
    """Result of stress test."""
    scenario: str
    portfolio_loss: float  # Expected loss in scenario
    loss_pct: float  # Loss as percentage
    var_breach_probability: float  # Probability of breaching VaR
    description: str
    affected_positions: List[str]  # Most affected positions


# Sector mapping for correlation estimation
SECTOR_MAP = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'META': 'Technology', 'NVDA': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology',
    'CRM': 'Technology', 'ADBE': 'Technology', 'ORCL': 'Technology', 'CSCO': 'Technology',
    # Financial
    'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'GS': 'Financial',
    'MS': 'Financial', 'C': 'Financial', 'V': 'Financial', 'MA': 'Financial',
    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare',
    'ABBV': 'Healthcare', 'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
    # Consumer
    'AMZN': 'Consumer', 'TSLA': 'Consumer', 'HD': 'Consumer', 'NKE': 'Consumer',
    'MCD': 'Consumer', 'SBUX': 'Consumer', 'WMT': 'Staples', 'COST': 'Consumer',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
}


class MonteCarloVaR:
    """
    Monte Carlo Value at Risk Calculator.

    Simulates correlated portfolio returns using Cholesky decomposition
    to estimate potential losses at various confidence levels.

    Features:
    - Multi-asset portfolio simulation
    - Correlated returns modeling
    - Multiple confidence levels
    - Conditional VaR (CVaR/ES)
    - Stress testing
    - Scenario analysis
    """

    def __init__(
        self,
        num_simulations: int = 10000,
        confidence_levels: Optional[List[float]] = None,
        horizon_days: int = 5,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Monte Carlo VaR calculator.

        Args:
            num_simulations: Number of Monte Carlo simulations (default: 10,000)
            confidence_levels: List of confidence levels (default: [0.95, 0.99])
            horizon_days: Risk horizon in days (default: 5)
            random_seed: Random seed for reproducibility (optional)
        """
        self.num_simulations = num_simulations
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.horizon_days = horizon_days
        self.random_seed = random_seed

        # REPRODUCIBILITY: Use numpy Generator for deterministic simulations
        # Default seed is 42 for consistency with other modules
        seed = random_seed if random_seed is not None else 42
        self._rng = np.random.Generator(np.random.PCG64(seed))

        # Also set global seed for backward compatibility
        if random_seed is not None:
            np.random.seed(random_seed)

        logger.info(
            f"MonteCarloVaR initialized: {num_simulations:,} simulations, "
            f"{horizon_days}-day horizon, confidence levels: {self.confidence_levels}"
        )

    def calculate_var(
        self,
        positions: pd.DataFrame,
        correlation_matrix: Optional[np.ndarray] = None,
        confidence_level: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk for portfolio positions.

        Args:
            positions: DataFrame with columns:
                - symbol: Stock symbol
                - quantity: Number of shares
                - price: Current price per share
                - expected_return: Expected daily return (mean)
                - volatility: Daily return volatility (std dev)
            correlation_matrix: Optional pre-calculated correlation matrix
            confidence_level: Confidence level to use (default: first in list)

        Returns:
            Dict with VaR metrics
        """
        if positions.empty:
            logger.warning("Empty positions DataFrame provided")
            return self._empty_result()

        # Validate required columns
        required_cols = ['symbol', 'quantity', 'price', 'expected_return', 'volatility']
        missing_cols = [col for col in required_cols if col not in positions.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Use first confidence level if not specified
        conf_level = confidence_level or self.confidence_levels[0]

        # Calculate portfolio value
        positions = positions.copy()
        positions['value'] = positions['quantity'] * positions['price']
        portfolio_value = positions['value'].sum()

        if portfolio_value <= 0:
            logger.warning("Portfolio value is zero or negative")
            return self._empty_result()

        # Calculate position weights
        positions['weight'] = positions['value'] / portfolio_value

        logger.info(
            f"Calculating VaR for portfolio: ${portfolio_value:,.2f}, "
            f"{len(positions)} positions"
        )

        # Reset random seed if specified for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Get or estimate correlation matrix
        if correlation_matrix is None:
            correlation_matrix = self._get_correlation_matrix(positions)

        # Simulate correlated returns
        simulated_returns = self._simulate_correlated_returns(
            positions,
            correlation_matrix,
            self.horizon_days,
        )

        # Calculate portfolio returns for each simulation
        portfolio_returns = self._calculate_portfolio_returns(
            simulated_returns,
            positions['weight'].values,
        )

        # Calculate portfolio value changes
        portfolio_changes = portfolio_returns * portfolio_value

        # Calculate VaR and CVaR
        var_value, cvar_value = self._calculate_percentile(
            portfolio_changes,
            conf_level,
        )

        # Calculate additional statistics
        worst_case = np.min(portfolio_changes)
        best_case = np.max(portfolio_changes)
        percentile_5 = np.percentile(portfolio_changes, 5)
        percentile_95 = np.percentile(portfolio_changes, 95)
        mean_outcome = np.mean(portfolio_changes)

        result = {
            'var': abs(var_value),
            'cvar': abs(cvar_value),
            'var_pct': abs(var_value / portfolio_value) * 100,
            'cvar_pct': abs(cvar_value / portfolio_value) * 100,
            'simulations': self.num_simulations,
            'confidence_level': conf_level,
            'horizon_days': self.horizon_days,
            'portfolio_value': portfolio_value,
            'worst_case': abs(worst_case),
            'best_case': best_case,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'mean_outcome': mean_outcome,
            'positions_count': len(positions),
        }

        logger.info(
            f"VaR calculated: ${result['var']:,.2f} ({result['var_pct']:.2f}%), "
            f"CVaR: ${result['cvar']:,.2f} ({result['cvar_pct']:.2f}%) "
            f"at {conf_level*100:.0f}% confidence"
        )

        return result

    def stress_test(
        self,
        positions: pd.DataFrame,
        scenario: StressScenario = StressScenario.MARKET_CRASH,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> StressTestResult:
        """
        Run stress test scenario on portfolio.

        Args:
            positions: DataFrame with position data
            scenario: Predefined scenario to test
            custom_params: Custom parameters for scenario

        Returns:
            StressTestResult with scenario impact analysis
        """
        if positions.empty:
            logger.warning("Empty positions for stress test")
            return self._empty_stress_result(scenario.value)

        positions = positions.copy()
        positions['value'] = positions['quantity'] * positions['price']
        portfolio_value = positions['value'].sum()

        # Apply scenario parameters
        if scenario == StressScenario.MARKET_CRASH:
            return_shock = -0.20
            vol_multiplier = 2.0
            corr_adjustment = 0.3
            description = "Severe market crash: -20% returns, 2x volatility, high correlations"
        elif scenario == StressScenario.SECTOR_ROTATION:
            return_shock = -0.10
            vol_multiplier = 1.0
            corr_adjustment = -0.2
            description = "Sector rotation: -10% sector-specific selloff"
        elif scenario == StressScenario.VOLATILITY_SPIKE:
            return_shock = 0.0
            vol_multiplier = 3.0
            corr_adjustment = 0.2
            description = "Volatility spike: 3x normal volatility"
        elif scenario == StressScenario.CORRELATION_BREAKDOWN:
            return_shock = -0.15
            vol_multiplier = 1.5
            corr_adjustment = 0.5
            description = "Correlation breakdown: all assets move together"
        elif scenario == StressScenario.CUSTOM:
            if not custom_params:
                raise ValueError("Custom scenario requires custom_params")
            return_shock = custom_params.get('return_shock', 0.0)
            vol_multiplier = custom_params.get('volatility_multiplier', 1.0)
            corr_adjustment = custom_params.get('correlation_adjustment', 0.0)
            description = custom_params.get('description', 'Custom stress scenario')
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        logger.info(f"Running stress test: {scenario.value}")

        # Create stressed positions
        stressed_positions = positions.copy()
        stressed_positions['expected_return'] = (
            positions['expected_return'] + return_shock / self.horizon_days
        )
        stressed_positions['volatility'] = positions['volatility'] * vol_multiplier

        # Create stressed correlation matrix
        base_corr_matrix = self._get_correlation_matrix(positions)
        stressed_corr_matrix = self._adjust_correlations(base_corr_matrix, corr_adjustment)

        # Run Monte Carlo with stressed parameters
        simulated_returns = self._simulate_correlated_returns(
            stressed_positions,
            stressed_corr_matrix,
            self.horizon_days,
        )

        # Calculate portfolio returns
        weights = (positions['value'] / portfolio_value).values
        portfolio_returns = self._calculate_portfolio_returns(simulated_returns, weights)
        portfolio_changes = portfolio_returns * portfolio_value

        # Calculate expected loss
        expected_loss = np.mean(portfolio_changes)
        loss_pct = (expected_loss / portfolio_value) * 100

        # Calculate VaR breach probability
        current_var_result = self.calculate_var(positions)
        current_var = -current_var_result['var']
        var_breaches = np.sum(portfolio_changes < current_var)
        var_breach_prob = var_breaches / self.num_simulations

        # Identify most affected positions
        position_impacts = []
        for idx, row in stressed_positions.iterrows():
            impact = row['expected_return'] * row['value'] * self.horizon_days
            position_impacts.append((row['symbol'], impact))

        position_impacts.sort(key=lambda x: x[1])
        affected_positions = [sym for sym, _ in position_impacts[:5]]

        result = StressTestResult(
            scenario=scenario.value,
            portfolio_loss=abs(expected_loss),
            loss_pct=abs(loss_pct),
            var_breach_probability=var_breach_prob,
            description=description,
            affected_positions=affected_positions,
        )

        logger.info(
            f"Stress test complete: {scenario.value} | "
            f"Expected loss: ${result.portfolio_loss:,.2f} ({result.loss_pct:.2f}%) | "
            f"VaR breach probability: {result.var_breach_probability:.1%}"
        )

        return result

    def _simulate_correlated_returns(
        self,
        positions: pd.DataFrame,
        correlation_matrix: np.ndarray,
        horizon_days: int,
    ) -> np.ndarray:
        """Simulate correlated returns using Cholesky decomposition."""
        n_assets = len(positions)

        expected_returns = positions['expected_return'].values
        volatilities = positions['volatility'].values

        # Scale for time horizon
        horizon_expected_returns = expected_returns * horizon_days
        horizon_volatilities = volatilities * np.sqrt(horizon_days)

        # Ensure positive semi-definite
        correlation_matrix = self._ensure_positive_semidefinite(correlation_matrix)

        try:
            chol_matrix = cholesky(correlation_matrix, lower=True)
        except np.linalg.LinAlgError:
            logger.warning("Cholesky failed, using eigenvalue fallback")
            eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            chol_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))

        # Generate uncorrelated standard normal random variables
        # Use seeded Generator for reproducibility
        uncorrelated_random = self._rng.standard_normal((self.num_simulations, n_assets))

        # Apply Cholesky decomposition to create correlations
        correlated_random = uncorrelated_random @ chol_matrix.T

        # Scale by volatilities and add expected returns
        simulated_returns = horizon_expected_returns + correlated_random * horizon_volatilities

        return simulated_returns

    def _get_correlation_matrix(self, positions: pd.DataFrame) -> np.ndarray:
        """Get or estimate correlation matrix for positions."""
        n_assets = len(positions)

        # Check for historical returns
        if 'historical_returns' in positions.columns:
            try:
                returns_data = []
                for returns in positions['historical_returns']:
                    if returns is not None and len(returns) > 0:
                        returns_data.append(returns)
                    else:
                        returns_data.append([0.0])

                returns_df = pd.DataFrame(returns_data).T
                corr_matrix = returns_df.corr().values

                if np.all(np.isfinite(corr_matrix)):
                    logger.info("Using empirical correlation matrix")
                    return corr_matrix
            except Exception as e:
                logger.warning(f"Failed to calculate empirical correlations: {e}")

        # Fallback: sector-based correlation estimates
        logger.info("Using sector-based correlation estimates")
        corr_matrix = np.eye(n_assets)
        symbols = positions['symbol'].values

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                sym1, sym2 = symbols[i], symbols[j]
                sector1 = SECTOR_MAP.get(sym1, 'Unknown')
                sector2 = SECTOR_MAP.get(sym2, 'Unknown')

                if sector1 == sector2 and sector1 != 'Unknown':
                    corr = 0.6
                else:
                    corr = 0.3

                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        return corr_matrix

    def _calculate_portfolio_returns(
        self,
        asset_returns: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Calculate portfolio returns from asset returns and weights."""
        return asset_returns @ weights

    def _calculate_percentile(
        self,
        portfolio_values: np.ndarray,
        confidence: float,
    ) -> Tuple[float, float]:
        """Calculate VaR and CVaR from simulated portfolio values."""
        var_percentile = (1 - confidence) * 100
        var_value = np.percentile(portfolio_values, var_percentile)

        worse_than_var = portfolio_values[portfolio_values <= var_value]
        cvar_value = np.mean(worse_than_var) if len(worse_than_var) > 0 else var_value

        return var_value, cvar_value

    def _ensure_positive_semidefinite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure correlation matrix is positive semi-definite."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        adjusted_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        d = np.sqrt(np.diag(adjusted_matrix))
        adjusted_matrix = adjusted_matrix / np.outer(d, d)

        return adjusted_matrix

    def _adjust_correlations(
        self,
        correlation_matrix: np.ndarray,
        adjustment: float,
    ) -> np.ndarray:
        """Adjust correlation matrix for stress testing."""
        n = correlation_matrix.shape[0]
        adjusted = correlation_matrix.copy()

        for i in range(n):
            for j in range(i + 1, n):
                current_corr = adjusted[i, j]
                if adjustment > 0:
                    new_corr = current_corr + adjustment * (1 - current_corr)
                else:
                    new_corr = current_corr + adjustment * current_corr

                new_corr = np.clip(new_corr, -0.99, 0.99)
                adjusted[i, j] = new_corr
                adjusted[j, i] = new_corr

        return self._ensure_positive_semidefinite(adjusted)

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty VaR result."""
        return {
            'var': 0.0, 'cvar': 0.0, 'var_pct': 0.0, 'cvar_pct': 0.0,
            'simulations': 0, 'confidence_level': 0.0, 'horizon_days': 0,
            'portfolio_value': 0.0, 'worst_case': 0.0, 'best_case': 0.0,
            'percentile_5': 0.0, 'percentile_95': 0.0, 'mean_outcome': 0.0,
            'positions_count': 0,
        }

    def _empty_stress_result(self, scenario: str) -> StressTestResult:
        """Return empty stress test result."""
        return StressTestResult(
            scenario=scenario, portfolio_loss=0.0, loss_pct=0.0,
            var_breach_probability=0.0, description="Empty portfolio",
            affected_positions=[],
        )

    def format_telegram_alert(
        self,
        var_result: Dict[str, Any],
        include_details: bool = True,
    ) -> str:
        """Format VaR result for Telegram notification."""
        var_pct = var_result.get('var_pct', 0)

        if var_pct > 10:
            emoji, risk_level = "🔴", "HIGH"
        elif var_pct > 5:
            emoji, risk_level = "🟡", "MODERATE"
        else:
            emoji, risk_level = "🟢", "LOW"

        msg = f"{emoji} <b>MONTE CARLO VaR - {risk_level} RISK</b>\n\n"
        msg += f"<b>Confidence:</b> {var_result.get('confidence_level', 0)*100:.0f}%\n"
        msg += f"<b>Horizon:</b> {var_result.get('horizon_days', 0)} days\n"
        msg += f"<b>Simulations:</b> {var_result.get('simulations', 0):,}\n\n"
        msg += f"<b>VaR:</b> ${var_result.get('var', 0):,.2f} ({var_pct:.2f}%)\n"
        msg += f"<b>CVaR:</b> ${var_result.get('cvar', 0):,.2f} ({var_result.get('cvar_pct', 0):.2f}%)\n\n"

        if include_details:
            msg += f"<b>Portfolio:</b> ${var_result.get('portfolio_value', 0):,.2f}\n"
            msg += f"<b>Positions:</b> {var_result.get('positions_count', 0)}\n"
            msg += f"<b>Worst Case:</b> -${var_result.get('worst_case', 0):,.2f}\n"
            msg += f"<b>Best Case:</b> +${var_result.get('best_case', 0):,.2f}\n"

        return msg
