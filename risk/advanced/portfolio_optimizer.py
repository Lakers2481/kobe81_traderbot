"""
OR-Tools Portfolio Optimizer

Hard constraint enforcement for portfolio construction:
- Max position size (20% notional)
- Max sector exposure (30%)
- Max correlated positions (40%)
- Beta neutrality option
- Sector balance constraints

Uses Google OR-Tools for efficient constraint solving.

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from core.structured_log import get_logger

logger = get_logger(__name__)

# Lazy import OR-Tools
_ortools = None


def _get_ortools():
    """Lazy import OR-Tools."""
    global _ortools
    if _ortools is None:
        try:
            from ortools.linear_solver import pywraplp
            _ortools = pywraplp
        except ImportError:
            logger.warning("OR-Tools not installed. Install with: pip install ortools")
            raise ImportError("OR-Tools required. Install with: pip install ortools")
    return _ortools


@dataclass
class PortfolioConstraints:
    """Portfolio constraint parameters."""

    # Position constraints
    max_position_pct: float = 0.20  # Max 20% per position
    min_position_pct: float = 0.01  # Min 1% per position (if included)
    max_positions: int = 10  # Maximum number of positions

    # Sector constraints
    max_sector_pct: float = 0.30  # Max 30% per sector
    min_sectors: int = 3  # Minimum number of sectors

    # Correlation constraints
    max_correlation_exposure: float = 0.40  # Max 40% in highly correlated positions
    correlation_threshold: float = 0.70  # Positions above this are "highly correlated"

    # Beta constraints
    target_beta: Optional[float] = None  # Target portfolio beta (None = no constraint)
    beta_tolerance: float = 0.20  # +/- tolerance for beta

    # Diversification
    min_effective_positions: float = 4.0  # Minimum effective number of positions (ENP)


@dataclass
class Position:
    """Position in portfolio."""
    symbol: str
    weight: float
    sector: str
    beta: float
    expected_return: float = 0.0
    volatility: float = 0.0


class PortfolioOptimizer:
    """
    OR-Tools based portfolio optimizer.

    Optimizes portfolio weights subject to hard constraints.
    """

    def __init__(self, constraints: Optional[PortfolioConstraints] = None):
        """
        Initialize optimizer.

        Args:
            constraints: Portfolio constraints (uses defaults if None)
        """
        self.constraints = constraints or PortfolioConstraints()

    def optimize_weights(
        self,
        candidates: List[Dict[str, Any]],
        equity: float,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Optimize portfolio weights for given candidates.

        Args:
            candidates: List of candidate positions with:
                - symbol: Stock symbol
                - sector: Sector classification
                - beta: Stock beta
                - expected_return: Expected return (optional)
                - signal_score: Signal quality score (for prioritization)
            equity: Total portfolio equity
            correlation_matrix: Optional correlation matrix (symbols as index/columns)

        Returns:
            Dict with optimized weights and status
        """
        if not candidates:
            return {'status': 'NO_CANDIDATES', 'weights': {}, 'message': 'No candidates provided'}

        try:
            pywraplp = _get_ortools()
        except ImportError:
            return self._fallback_optimize(candidates, equity)

        # Create solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            logger.warning("OR-Tools SCIP solver not available, using fallback")
            return self._fallback_optimize(candidates, equity)

        n = len(candidates)
        c = self.constraints

        # Decision variables: weight for each candidate (0 to max_position_pct)
        weights = {}
        included = {}  # Binary: is position included?

        for i, cand in enumerate(candidates):
            symbol = cand['symbol']
            weights[symbol] = solver.NumVar(0, c.max_position_pct, f'w_{symbol}')
            included[symbol] = solver.IntVar(0, 1, f'inc_{symbol}')

        # Constraint 1: Weights sum to 1 (or less for partial deployment)
        solver.Add(sum(weights.values()) <= 1.0)
        solver.Add(sum(weights.values()) >= 0.5)  # At least 50% deployed

        # Constraint 2: Link weight to inclusion (if included, weight >= min)
        for cand in candidates:
            symbol = cand['symbol']
            solver.Add(weights[symbol] >= c.min_position_pct * included[symbol])
            solver.Add(weights[symbol] <= c.max_position_pct * included[symbol])

        # Constraint 3: Max number of positions
        solver.Add(sum(included.values()) <= c.max_positions)

        # Constraint 4: Sector exposure limits
        sectors = {}
        for cand in candidates:
            sector = cand.get('sector', 'Unknown')
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(cand['symbol'])

        for sector, symbols in sectors.items():
            sector_weight = sum(weights[s] for s in symbols)
            solver.Add(sector_weight <= c.max_sector_pct)

        # Constraint 5: Minimum sectors (if enough positions)
        if len(sectors) >= c.min_sectors:
            sector_indicators = {}
            for sector, symbols in sectors.items():
                sector_indicators[sector] = solver.IntVar(0, 1, f'sector_{sector}')
                # Sector is included if any position in it is included
                for symbol in symbols:
                    solver.Add(sector_indicators[sector] >= included[symbol])
            solver.Add(sum(sector_indicators.values()) >= min(c.min_sectors, len(sectors)))

        # Constraint 6: Beta constraint (if specified)
        if c.target_beta is not None:
            portfolio_beta = sum(
                weights[cand['symbol']] * cand.get('beta', 1.0)
                for cand in candidates
            )
            solver.Add(portfolio_beta >= c.target_beta - c.beta_tolerance)
            solver.Add(portfolio_beta <= c.target_beta + c.beta_tolerance)

        # Constraint 7: Correlation constraint (if matrix provided)
        if correlation_matrix is not None and len(correlation_matrix) > 0:
            # Limit exposure to highly correlated positions
            for i, cand1 in enumerate(candidates):
                for j, cand2 in enumerate(candidates):
                    if i >= j:
                        continue
                    s1, s2 = cand1['symbol'], cand2['symbol']
                    if s1 in correlation_matrix.index and s2 in correlation_matrix.columns:
                        corr = abs(correlation_matrix.loc[s1, s2])
                        if corr >= c.correlation_threshold:
                            # Combined weight of highly correlated pairs limited
                            solver.Add(weights[s1] + weights[s2] <= c.max_correlation_exposure)

        # Objective: Maximize expected returns weighted by signal score
        objective = sum(
            weights[cand['symbol']] * cand.get('signal_score', cand.get('expected_return', 1.0))
            for cand in candidates
        )
        solver.Maximize(objective)

        # Solve
        solver.SetTimeLimit(5000)  # 5 second timeout
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            result_weights = {}
            for cand in candidates:
                symbol = cand['symbol']
                w = weights[symbol].solution_value()
                if w >= c.min_position_pct:
                    result_weights[symbol] = round(w, 4)

            # Calculate portfolio metrics
            portfolio_beta = sum(
                result_weights.get(cand['symbol'], 0) * cand.get('beta', 1.0)
                for cand in candidates
            )

            sector_weights = {}
            for cand in candidates:
                sector = cand.get('sector', 'Unknown')
                sector_weights[sector] = sector_weights.get(sector, 0) + result_weights.get(cand['symbol'], 0)

            # Effective number of positions (ENP)
            weights_array = np.array(list(result_weights.values()))
            if len(weights_array) > 0 and weights_array.sum() > 0:
                normalized = weights_array / weights_array.sum()
                enp = 1.0 / (normalized ** 2).sum()
            else:
                enp = 0

            return {
                'status': 'OPTIMAL' if status == pywraplp.Solver.OPTIMAL else 'FEASIBLE',
                'weights': result_weights,
                'portfolio_beta': round(portfolio_beta, 3),
                'sector_weights': {k: round(v, 4) for k, v in sector_weights.items() if v > 0},
                'num_positions': len(result_weights),
                'total_weight': round(sum(result_weights.values()), 4),
                'effective_positions': round(enp, 2),
                'objective_value': solver.Objective().Value()
            }

        else:
            logger.warning(f"OR-Tools solver status: {status}")
            return self._fallback_optimize(candidates, equity)

    def _fallback_optimize(
        self,
        candidates: List[Dict[str, Any]],
        equity: float
    ) -> Dict[str, Any]:
        """
        Fallback optimization when OR-Tools unavailable.

        Uses simple equal-weight with constraints.
        """
        c = self.constraints

        # Sort by signal score
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get('signal_score', x.get('expected_return', 0)),
            reverse=True
        )

        # Take top N positions
        selected = sorted_candidates[:c.max_positions]

        # Apply sector limits
        sector_counts = {}
        final_selected = []
        for cand in selected:
            sector = cand.get('sector', 'Unknown')
            if sector_counts.get(sector, 0) < 3:  # Max 3 per sector
                final_selected.append(cand)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Equal weight (capped at max_position_pct)
        n = len(final_selected)
        if n == 0:
            return {'status': 'NO_CANDIDATES', 'weights': {}}

        weight = min(1.0 / n, c.max_position_pct)
        weights = {cand['symbol']: round(weight, 4) for cand in final_selected}

        return {
            'status': 'FALLBACK',
            'weights': weights,
            'num_positions': len(weights),
            'total_weight': round(sum(weights.values()), 4),
            'message': 'Used fallback equal-weight optimization'
        }

    def validate_portfolio(
        self,
        positions: List[Position],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Validate existing portfolio against constraints.

        Args:
            positions: List of current positions
            correlation_matrix: Optional correlation matrix

        Returns:
            Dict with validation results and violations
        """
        c = self.constraints
        violations = []
        warnings = []

        total_weight = sum(p.weight for p in positions)

        # Check position limits
        for p in positions:
            if p.weight > c.max_position_pct:
                violations.append(f"{p.symbol}: weight {p.weight:.1%} > max {c.max_position_pct:.1%}")
            if p.weight < c.min_position_pct and p.weight > 0:
                warnings.append(f"{p.symbol}: weight {p.weight:.1%} < min {c.min_position_pct:.1%}")

        # Check sector limits
        sector_weights = {}
        for p in positions:
            sector_weights[p.sector] = sector_weights.get(p.sector, 0) + p.weight

        for sector, weight in sector_weights.items():
            if weight > c.max_sector_pct:
                violations.append(f"Sector {sector}: {weight:.1%} > max {c.max_sector_pct:.1%}")

        # Check number of positions
        if len(positions) > c.max_positions:
            violations.append(f"Positions: {len(positions)} > max {c.max_positions}")

        # Check beta
        if c.target_beta is not None:
            portfolio_beta = sum(p.weight * p.beta for p in positions) / total_weight if total_weight > 0 else 0
            if abs(portfolio_beta - c.target_beta) > c.beta_tolerance:
                violations.append(f"Beta: {portfolio_beta:.2f} outside target {c.target_beta:.2f} +/- {c.beta_tolerance:.2f}")

        # Check correlation
        if correlation_matrix is not None:
            for i, p1 in enumerate(positions):
                for j, p2 in enumerate(positions):
                    if i >= j:
                        continue
                    if p1.symbol in correlation_matrix.index and p2.symbol in correlation_matrix.columns:
                        corr = abs(correlation_matrix.loc[p1.symbol, p2.symbol])
                        if corr >= c.correlation_threshold:
                            combined = p1.weight + p2.weight
                            if combined > c.max_correlation_exposure:
                                violations.append(
                                    f"{p1.symbol}-{p2.symbol} corr={corr:.2f}, combined={combined:.1%} > max {c.max_correlation_exposure:.1%}"
                                )

        # Check ENP
        if positions:
            weights_array = np.array([p.weight for p in positions])
            if weights_array.sum() > 0:
                normalized = weights_array / weights_array.sum()
                enp = 1.0 / (normalized ** 2).sum()
                if enp < c.min_effective_positions:
                    warnings.append(f"ENP: {enp:.1f} < min {c.min_effective_positions:.1f} (concentrated)")

        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'metrics': {
                'total_weight': total_weight,
                'num_positions': len(positions),
                'sector_weights': sector_weights,
            }
        }

    def suggest_rebalance(
        self,
        current_positions: List[Position],
        new_candidates: List[Dict[str, Any]],
        equity: float
    ) -> Dict[str, Any]:
        """
        Suggest rebalancing trades.

        Args:
            current_positions: List of current positions
            new_candidates: New candidate signals
            equity: Total equity

        Returns:
            Dict with suggested trades
        """
        # Combine current and new
        all_candidates = []

        # Current positions as candidates (with priority)
        for p in current_positions:
            all_candidates.append({
                'symbol': p.symbol,
                'sector': p.sector,
                'beta': p.beta,
                'expected_return': p.expected_return,
                'signal_score': 1.0,  # Existing positions get base score
                'current_weight': p.weight
            })

        # Add new candidates
        for c in new_candidates:
            if c['symbol'] not in [p.symbol for p in current_positions]:
                all_candidates.append({
                    **c,
                    'current_weight': 0
                })

        # Optimize
        result = self.optimize_weights(all_candidates, equity)

        if result['status'] in ['OPTIMAL', 'FEASIBLE']:
            # Calculate trades needed
            trades = []
            for cand in all_candidates:
                symbol = cand['symbol']
                current = cand.get('current_weight', 0)
                target = result['weights'].get(symbol, 0)
                diff = target - current

                if abs(diff) > 0.01:  # 1% threshold
                    trades.append({
                        'symbol': symbol,
                        'action': 'BUY' if diff > 0 else 'SELL',
                        'weight_change': round(diff, 4),
                        'dollar_amount': round(abs(diff) * equity, 2)
                    })

            result['suggested_trades'] = trades
            result['num_trades'] = len(trades)

        return result


# Singleton instance
_optimizer: Optional[PortfolioOptimizer] = None


def get_portfolio_optimizer() -> PortfolioOptimizer:
    """Get or create singleton optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PortfolioOptimizer()
    return _optimizer


def optimize_portfolio(
    candidates: List[Dict[str, Any]],
    equity: float
) -> Dict[str, Any]:
    """Convenience function to optimize portfolio."""
    return get_portfolio_optimizer().optimize_weights(candidates, equity)


def validate_portfolio(positions: List[Position]) -> Dict[str, Any]:
    """Convenience function to validate portfolio."""
    return get_portfolio_optimizer().validate_portfolio(positions)


if __name__ == "__main__":
    # Demo usage
    print("=== OR-Tools Portfolio Optimizer Demo ===\n")

    optimizer = PortfolioOptimizer()

    # Sample candidates
    candidates = [
        {'symbol': 'AAPL', 'sector': 'Technology', 'beta': 1.2, 'signal_score': 0.85},
        {'symbol': 'MSFT', 'sector': 'Technology', 'beta': 1.1, 'signal_score': 0.80},
        {'symbol': 'GOOGL', 'sector': 'Technology', 'beta': 1.3, 'signal_score': 0.75},
        {'symbol': 'JPM', 'sector': 'Financials', 'beta': 1.15, 'signal_score': 0.70},
        {'symbol': 'BAC', 'sector': 'Financials', 'beta': 1.25, 'signal_score': 0.65},
        {'symbol': 'XOM', 'sector': 'Energy', 'beta': 0.9, 'signal_score': 0.72},
        {'symbol': 'CVX', 'sector': 'Energy', 'beta': 0.95, 'signal_score': 0.68},
        {'symbol': 'JNJ', 'sector': 'Healthcare', 'beta': 0.7, 'signal_score': 0.78},
        {'symbol': 'PFE', 'sector': 'Healthcare', 'beta': 0.65, 'signal_score': 0.60},
        {'symbol': 'PG', 'sector': 'Consumer', 'beta': 0.5, 'signal_score': 0.55},
    ]

    print("Optimizing portfolio for 10 candidates...")
    result = optimizer.optimize_weights(candidates, equity=100000)

    print(f"\nStatus: {result['status']}")
    print(f"Total Weight: {result.get('total_weight', 0):.1%}")
    print(f"Positions: {result.get('num_positions', 0)}")
    print(f"Portfolio Beta: {result.get('portfolio_beta', 'N/A')}")
    print(f"Effective Positions: {result.get('effective_positions', 'N/A')}")

    print("\nOptimal Weights:")
    for symbol, weight in sorted(result.get('weights', {}).items(), key=lambda x: -x[1]):
        print(f"  {symbol}: {weight:.1%}")

    print("\nSector Allocation:")
    for sector, weight in sorted(result.get('sector_weights', {}).items(), key=lambda x: -x[1]):
        print(f"  {sector}: {weight:.1%}")

    # Validation demo
    print("\n\n=== Portfolio Validation Demo ===")
    positions = [
        Position('AAPL', 0.25, 'Technology', 1.2),  # Over limit
        Position('MSFT', 0.20, 'Technology', 1.1),
        Position('JPM', 0.15, 'Financials', 1.15),
        Position('XOM', 0.10, 'Energy', 0.9),
    ]

    validation = optimizer.validate_portfolio(positions)
    print(f"\nValid: {validation['valid']}")
    if validation['violations']:
        print("Violations:")
        for v in validation['violations']:
            print(f"  - {v}")
    if validation['warnings']:
        print("Warnings:")
        for w in validation['warnings']:
            print(f"  - {w}")
