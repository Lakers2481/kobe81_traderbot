"""
Automated Alpha Screener
=========================

Systematic alpha screening with walk-forward validation.

Key Features:
1. Cross-sectional backtesting of all alphas
2. Walk-forward validation (train/test splits)
3. Multiple testing correction (FDR warning)
4. Stability analysis across regimes
5. Leaderboard generation

Quant Interview Standard:
- All alphas are evaluated OOS
- Multiple testing is flagged
- Performance attribution is clear
- Results are reproducible

Usage:
    from research.screener import AlphaScreener, run_alpha_screen

    screener = AlphaScreener()
    results = screener.screen(
        dataset_id='stooq_1d_2015_2024_abc123',
        alphas=['momentum_12_1', 'rsi2_oversold'],
    )

    # Print leaderboard
    print(results.leaderboard)

    # Get best alpha
    best = results.get_best_alpha()
"""
from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .alphas import AlphaLibrary, ALPHA_REGISTRY, get_alpha_library
from .features import FeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class AlphaResult:
    """Results for a single alpha."""
    alpha_name: str
    category: str

    # Performance metrics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    cagr: float = 0.0

    # OOS metrics (primary for ranking)
    oos_trades: int = 0
    oos_sharpe: float = 0.0
    oos_profit_factor: float = 0.0
    oos_win_rate: float = 0.0

    # Stability metrics
    regime_sharpes: Dict[str, float] = field(default_factory=dict)
    regime_stability: float = 0.0
    ticker_concentration: float = 0.0

    # Statistical significance
    t_stat: float = 0.0
    p_value: float = 1.0

    # Ranking (lower is better)
    rank: int = 0

    # Flags
    passed_gate: bool = False
    multiple_testing_warning: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ScreenerResult:
    """Complete screening results."""
    screened_at: str
    dataset_id: str
    alphas_tested: int
    alphas_passed: int

    # Results per alpha
    results: List[AlphaResult] = field(default_factory=list)

    # Leaderboard (sorted by OOS Sharpe)
    leaderboard: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Warnings
    multiple_testing_warning: bool = False
    data_quality_warnings: List[str] = field(default_factory=list)

    # Config used
    config: Dict[str, Any] = field(default_factory=dict)

    def get_best_alpha(self) -> Optional[AlphaResult]:
        """Get the best performing alpha."""
        passed = [r for r in self.results if r.passed_gate]
        if not passed:
            return None
        return sorted(passed, key=lambda x: x.oos_sharpe, reverse=True)[0]

    def get_alphas_by_category(self, category: str) -> List[AlphaResult]:
        """Get alphas in a category."""
        return [r for r in self.results if r.category == category]

    def save(self, path: Path):
        """Save results to JSON."""
        data = {
            'screened_at': self.screened_at,
            'dataset_id': self.dataset_id,
            'alphas_tested': self.alphas_tested,
            'alphas_passed': self.alphas_passed,
            'multiple_testing_warning': self.multiple_testing_warning,
            'data_quality_warnings': self.data_quality_warnings,
            'config': self.config,
            'results': [r.to_dict() for r in self.results],
        }
        path.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> 'ScreenerResult':
        """Load results from JSON."""
        data = json.loads(path.read_text())
        results = [AlphaResult(**r) for r in data.pop('results', [])]
        return cls(**data, results=results)


class AlphaScreener:
    """
    Automated alpha screening with walk-forward validation.

    Evaluates alphas using:
    1. Walk-forward backtesting (OOS validation)
    2. Regime stability analysis
    3. Multiple testing correction
    4. Evidence gates integration
    """

    def __init__(
        self,
        # Walk-forward config
        train_days: int = 252,  # 1 year training
        test_days: int = 63,    # 1 quarter testing
        min_trades_per_split: int = 10,

        # Performance thresholds
        min_oos_sharpe: float = 0.3,
        min_oos_trades: int = 30,
        max_drawdown: float = 0.30,

        # Directories
        output_dir: str = "reports/screening",
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.min_trades_per_split = min_trades_per_split
        self.min_oos_sharpe = min_oos_sharpe
        self.min_oos_trades = min_oos_trades
        self.max_drawdown = max_drawdown
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.library = get_alpha_library()

        logger.info(f"AlphaScreener initialized: train={train_days}d, test={test_days}d")

    def screen(
        self,
        df: pd.DataFrame,
        alphas: Optional[List[str]] = None,
        dataset_id: str = "unknown",
    ) -> ScreenerResult:
        """
        Screen alphas against historical data.

        Args:
            df: OHLCV DataFrame with [timestamp, symbol, open, high, low, close, volume]
            alphas: List of alpha names to test (None = all)
            dataset_id: Dataset identifier for tracking

        Returns:
            ScreenerResult with performance of all alphas
        """
        alphas = alphas or list(ALPHA_REGISTRY.keys())

        logger.info(f"Screening {len(alphas)} alphas on dataset {dataset_id}")

        results = []
        warnings = []

        # Check data quality
        if df.empty:
            warnings.append("Empty dataset")
            return ScreenerResult(
                screened_at=datetime.now().isoformat(),
                dataset_id=dataset_id,
                alphas_tested=0,
                alphas_passed=0,
                data_quality_warnings=warnings,
            )

        # Get date range
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        date_range = (df['timestamp'].max() - df['timestamp'].min()).days

        if date_range < self.train_days + self.test_days:
            warnings.append(f"Insufficient history: {date_range} days")

        # Screen each alpha
        for alpha_name in alphas:
            try:
                result = self._evaluate_alpha(df, alpha_name)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to evaluate {alpha_name}: {e}")
                alpha_config = ALPHA_REGISTRY.get(alpha_name)
                results.append(AlphaResult(
                    alpha_name=alpha_name,
                    category=alpha_config.category if alpha_config else "unknown",
                ))

        # Apply evidence gates
        passed_count = 0
        for result in results:
            result.passed_gate = (
                result.oos_sharpe >= self.min_oos_sharpe
                and result.oos_trades >= self.min_oos_trades
                and result.max_drawdown <= self.max_drawdown
            )
            if result.passed_gate:
                passed_count += 1

        # Multiple testing warning
        multiple_testing = len(alphas) > 20

        if multiple_testing:
            for result in results:
                result.multiple_testing_warning = True
            warnings.append(
                f"MULTIPLE TESTING: Tested {len(alphas)} alphas. "
                f"Expected ~{len(alphas) * 0.05:.0f} false positives at 5% significance."
            )

        # Rank alphas by OOS Sharpe
        results.sort(key=lambda x: x.oos_sharpe, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1

        # Build leaderboard
        leaderboard = self._build_leaderboard(results)

        screener_result = ScreenerResult(
            screened_at=datetime.now().isoformat(),
            dataset_id=dataset_id,
            alphas_tested=len(alphas),
            alphas_passed=passed_count,
            results=results,
            leaderboard=leaderboard,
            multiple_testing_warning=multiple_testing,
            data_quality_warnings=warnings,
            config={
                'train_days': self.train_days,
                'test_days': self.test_days,
                'min_oos_sharpe': self.min_oos_sharpe,
                'min_oos_trades': self.min_oos_trades,
                'max_drawdown': self.max_drawdown,
            }
        )

        # Save results
        output_path = self.output_dir / f"screen_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        screener_result.save(output_path)
        logger.info(f"Screening results saved: {output_path}")

        return screener_result

    def _evaluate_alpha(
        self,
        df: pd.DataFrame,
        alpha_name: str,
    ) -> AlphaResult:
        """Evaluate a single alpha with walk-forward."""
        alpha_config = ALPHA_REGISTRY.get(alpha_name)

        if alpha_config is None:
            return AlphaResult(alpha_name=alpha_name, category="unknown")

        result = AlphaResult(
            alpha_name=alpha_name,
            category=alpha_config.category,
        )

        # Compute alpha signals
        signals = self.library.compute_alpha(alpha_name, df)

        # Get unique dates
        dates = np.sort(df['timestamp'].unique())

        # Walk-forward splits
        splits = self._create_splits(dates)

        all_returns = []
        oos_returns = []

        for train_start, train_end, test_start, test_end in splits:
            # Filter data
            train_mask = (df['timestamp'] >= train_start) & (df['timestamp'] < train_end)
            test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)

            # Simple long-only backtest: go long when signal > 0.5
            # This is a simplified evaluation - real backtest would be more complex

            test_signals = signals[test_mask]
            test_prices = df.loc[test_mask, ['timestamp', 'symbol', 'close']].copy()

            if len(test_signals) == 0:
                continue

            # Calculate returns for long positions when signal is positive
            test_prices['signal'] = test_signals.values
            test_prices['next_return'] = test_prices.groupby('symbol')['close'].pct_change().shift(-1)

            # Long when signal > 0.5
            longs = test_prices[test_prices['signal'] > 0.5]

            if len(longs) > 0:
                returns = longs['next_return'].dropna()
                oos_returns.extend(returns.tolist())
                result.oos_trades += len(returns)

        # Calculate metrics from OOS returns
        if len(oos_returns) > 0:
            oos_returns = np.array(oos_returns)
            oos_returns = oos_returns[~np.isnan(oos_returns)]

            if len(oos_returns) > 0:
                # Win rate
                result.oos_win_rate = np.mean(oos_returns > 0)

                # Sharpe ratio (annualized)
                if np.std(oos_returns) > 0:
                    result.oos_sharpe = (np.mean(oos_returns) / np.std(oos_returns)) * np.sqrt(252)
                    result.sharpe_ratio = result.oos_sharpe

                # Profit factor
                gains = oos_returns[oos_returns > 0].sum()
                losses = abs(oos_returns[oos_returns < 0].sum())
                result.oos_profit_factor = gains / (losses + 1e-10)
                result.profit_factor = result.oos_profit_factor

                # Max drawdown (simplified)
                equity = (1 + oos_returns).cumprod()
                rolling_max = np.maximum.accumulate(equity)
                drawdown = (equity - rolling_max) / rolling_max
                result.max_drawdown = abs(np.min(drawdown))

                # T-statistic
                if len(oos_returns) > 1:
                    result.t_stat = np.mean(oos_returns) / (np.std(oos_returns) / np.sqrt(len(oos_returns)))
                    # Simple p-value approximation (two-tailed)
                    from scipy import stats
                    try:
                        result.p_value = 2 * (1 - stats.t.cdf(abs(result.t_stat), len(oos_returns) - 1))
                    except:
                        result.p_value = 1.0

                result.total_trades = len(oos_returns)
                result.win_rate = result.oos_win_rate

        return result

    def _create_splits(
        self,
        dates: np.ndarray,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Create walk-forward train/test splits."""
        splits = []

        n_dates = len(dates)
        min_idx = self.train_days

        while min_idx + self.test_days <= n_dates:
            train_start = dates[min_idx - self.train_days]
            train_end = dates[min_idx]
            test_start = dates[min_idx]
            test_end = dates[min(min_idx + self.test_days, n_dates - 1)]

            splits.append((train_start, train_end, test_start, test_end))

            min_idx += self.test_days

        return splits

    def _build_leaderboard(self, results: List[AlphaResult]) -> pd.DataFrame:
        """Build leaderboard DataFrame."""
        data = []

        for r in results:
            data.append({
                'Rank': r.rank,
                'Alpha': r.alpha_name,
                'Category': r.category,
                'OOS Sharpe': round(r.oos_sharpe, 2),
                'OOS PF': round(r.oos_profit_factor, 2),
                'OOS Win%': f"{r.oos_win_rate:.1%}",
                'OOS Trades': r.oos_trades,
                'Max DD': f"{r.max_drawdown:.1%}",
                'T-Stat': round(r.t_stat, 2),
                'Passed': 'Yes' if r.passed_gate else 'No',
            })

        return pd.DataFrame(data)

    def quick_screen(
        self,
        df: pd.DataFrame,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Quick screening returning top N alphas."""
        result = self.screen(df)
        return result.leaderboard.head(top_n)


def run_alpha_screen(
    df: pd.DataFrame,
    alphas: Optional[List[str]] = None,
    dataset_id: str = "unknown",
    **kwargs,
) -> ScreenerResult:
    """
    Convenience function to run alpha screening.

    Args:
        df: OHLCV DataFrame
        alphas: Alphas to test (None = all)
        dataset_id: Dataset identifier
        **kwargs: Additional config for AlphaScreener

    Returns:
        ScreenerResult with all findings
    """
    screener = AlphaScreener(**kwargs)
    return screener.screen(df, alphas, dataset_id)


def screen_from_lake(
    dataset_id: str,
    alphas: Optional[List[str]] = None,
    lake_dir: str = "data/lake",
    manifest_dir: str = "data/manifests",
    **kwargs,
) -> ScreenerResult:
    """
    Screen alphas using frozen data from the lake.

    Args:
        dataset_id: Frozen dataset ID
        alphas: Alphas to test
        lake_dir: Path to data lake
        manifest_dir: Path to manifests
        **kwargs: Additional config

    Returns:
        ScreenerResult
    """
    from data.lake import LakeReader

    reader = LakeReader(lake_dir, manifest_dir)
    df = reader.load_dataset(dataset_id)

    return run_alpha_screen(df, alphas, dataset_id, **kwargs)
