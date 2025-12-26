"""
Bayesian Hyperparameter Optimization for Kobe Trading System.

Uses Optuna for intelligent hyperparameter search.
10-50x faster than grid search with better results.

Key Features:
- TPE (Tree-structured Parzen Estimator) sampler for efficient search
- Early stopping/pruning for bad trials
- Parameter importance analysis
- Multi-objective optimization (Sharpe + low drawdown)
- Full logging and visualization support
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Wrap imports for CI compatibility
try:
    import optuna
    from optuna.trial import Trial
    from optuna.importance import get_param_importances
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    duration_seconds: float
    all_trials: List[dict]
    importance: Dict[str, float]
    study_name: str = "optimization"
    optimization_direction: str = "maximize"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': self.n_trials,
            'duration_seconds': self.duration_seconds,
            'all_trials': self.all_trials,
            'importance': self.importance,
            'study_name': self.study_name,
            'optimization_direction': self.optimization_direction,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Optimization Results: {self.study_name}",
            "=" * 60,
            f"Best Value: {self.best_value:.4f} ({self.optimization_direction})",
            f"Trials Completed: {self.n_trials}",
            f"Duration: {self.duration_seconds:.2f}s",
            "",
            "Best Parameters:",
        ]
        for key, val in self.best_params.items():
            lines.append(f"  {key}: {val}")

        lines.append("")
        lines.append("Parameter Importance:")
        sorted_importance = sorted(self.importance.items(), key=lambda x: x[1], reverse=True)
        for key, val in sorted_importance[:5]:
            lines.append(f"  {key}: {val:.4f}")

        return "\n".join(lines)


@dataclass
class ParameterSpace:
    """Definition of parameter search space."""
    name: str
    param_type: str  # 'int', 'float', 'categorical'
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    step: Optional[float] = None
    log: bool = False  # Log-uniform distribution for exponential ranges

    def __post_init__(self):
        """Validate parameter space definition."""
        if self.param_type not in ['int', 'float', 'categorical']:
            raise ValueError(f"param_type must be 'int', 'float', or 'categorical', got {self.param_type}")

        if self.param_type in ['int', 'float']:
            if self.low is None or self.high is None:
                raise ValueError(f"low and high must be specified for {self.param_type} parameters")
            if self.low >= self.high:
                raise ValueError(f"low ({self.low}) must be less than high ({self.high})")

        if self.param_type == 'categorical':
            if not self.choices or len(self.choices) == 0:
                raise ValueError("choices must be specified for categorical parameters")


class StrategyOptimizer:
    """
    Bayesian optimization for trading strategy parameters.

    Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler for
    efficient hyperparameter search. Typically 10-50x faster than
    grid search with better final performance.

    Example:
        >>> from backtest.engine import BacktestEngine
        >>> from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
        >>> optimizer = StrategyOptimizer(
        ...     strategy_class=ConnorsRSI2Strategy,
        ...     backtest_engine=engine,
        ...     objective='sharpe',
        ...     n_trials=100
        ... )
        >>> optimizer.define_parameter_space(CONNORS_RSI2_PARAM_SPACE)
        >>> result = optimizer.optimize(train_data)
        >>> print(result.best_params)
    """

    def __init__(
        self,
        strategy_class: type,
        backtest_engine: Any,
        objective: str = 'sharpe',
        direction: str = 'maximize',
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        sampler: str = 'tpe',
        pruner: Optional[str] = 'median',
        study_name: Optional[str] = None,
        custom_objective_fn: Optional[Callable] = None,
    ):
        """
        Initialize strategy optimizer.

        Args:
            strategy_class: Strategy class to optimize (e.g., ConnorsRSI2Strategy)
            backtest_engine: BacktestEngine instance for running backtests
            objective: Metric to optimize ('sharpe', 'profit_factor', 'win_rate', etc.)
            direction: 'maximize' or 'minimize'
            n_trials: Number of trials to run
            timeout: Timeout in seconds (None = no limit)
            n_jobs: Number of parallel jobs (1 = sequential, -1 = all cores)
            sampler: Sampler algorithm ('tpe', 'cmaes', 'random')
            pruner: Pruning algorithm ('median', 'percentile', None)
            study_name: Name for the optimization study
            custom_objective_fn: Custom objective function (result) -> float
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna not installed. Run: pip install optuna"
            )

        self.strategy_class = strategy_class
        self.backtest_engine = backtest_engine
        self.objective = objective
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.sampler = sampler
        self.pruner = pruner
        self.study_name = study_name or f"kobe_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.custom_objective_fn = custom_objective_fn

        self.study: Optional[optuna.Study] = None
        self.param_space: List[ParameterSpace] = []
        self.best_result: Optional[Any] = None

        # Caching for backtest data
        self._train_data: Optional[pd.DataFrame] = None
        self._symbol: str = "OPT_TEST"

    def define_parameter_space(self, params: List[ParameterSpace]):
        """
        Define the hyperparameter search space.

        Args:
            params: List of ParameterSpace objects defining the search space
        """
        self.param_space = params
        logger.info(f"Defined parameter space with {len(params)} parameters")

    def _suggest_params(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggest parameters for a trial using Optuna.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested parameter values
        """
        params = {}
        for p in self.param_space:
            if p.param_type == 'int':
                params[p.name] = trial.suggest_int(
                    p.name,
                    int(p.low),
                    int(p.high),
                    step=int(p.step or 1)
                )
            elif p.param_type == 'float':
                if p.log:
                    params[p.name] = trial.suggest_float(
                        p.name,
                        p.low,
                        p.high,
                        log=True
                    )
                else:
                    params[p.name] = trial.suggest_float(
                        p.name,
                        p.low,
                        p.high,
                        step=p.step
                    )
            elif p.param_type == 'categorical':
                params[p.name] = trial.suggest_categorical(p.name, p.choices)

        return params

    def _objective(self, trial: Trial) -> float:
        """
        Objective function for optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Objective metric value
        """
        # Get suggested parameters
        params = self._suggest_params(trial)

        try:
            # Create strategy instance with suggested parameters
            strategy = self.strategy_class(params=params)

            # Run backtest
            result = self.backtest_engine.run(
                ohlcv_data=self._train_data,
                symbol=self._symbol,
                strategy=strategy
            )

            # Extract metrics
            metrics = result.get('metrics', {})

            # Handle case with no trades
            if not metrics or metrics.get('total_trades', 0) == 0:
                logger.debug(f"Trial {trial.number}: No trades generated")
                return -999.0

            # Custom objective function if provided
            if self.custom_objective_fn:
                return self.custom_objective_fn(result)

            # Return objective metric
            if self.objective == 'sharpe':
                return metrics.get('sharpe_ratio', -999.0)
            elif self.objective == 'profit_factor':
                return metrics.get('profit_factor', 0.0)
            elif self.objective == 'win_rate':
                return metrics.get('win_rate', 0.0)
            elif self.objective == 'sortino':
                return metrics.get('sortino_ratio', -999.0)
            elif self.objective == 'calmar':
                return metrics.get('calmar_ratio', -999.0)
            elif self.objective == 'total_return':
                return metrics.get('total_return_pct', -999.0)
            elif self.objective == 'expectancy':
                return metrics.get('expectancy', -999.0)
            else:
                raise ValueError(f"Unknown objective: {self.objective}")

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return -999.0

    def optimize(
        self,
        train_data: pd.DataFrame,
        symbol: str = "OPT_TEST",
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.

        Args:
            train_data: Historical OHLCV data for backtesting
            symbol: Symbol name for backtest
            verbose: Show progress bar

        Returns:
            OptimizationResult with best parameters and metrics
        """
        if not self.param_space:
            raise ValueError("Parameter space not defined. Call define_parameter_space() first.")

        # Store train data for objective function
        self._train_data = train_data
        self._symbol = symbol

        # Create sampler
        if self.sampler == 'tpe':
            sampler = optuna.samplers.TPESampler(seed=42)
        elif self.sampler == 'cmaes':
            sampler = optuna.samplers.CmaEsSampler(seed=42)
        else:
            sampler = optuna.samplers.RandomSampler(seed=42)

        # Create pruner
        pruner = None
        if self.pruner == 'median':
            pruner = optuna.pruners.MedianPruner()
        elif self.pruner == 'percentile':
            pruner = optuna.pruners.PercentilePruner(percentile=25.0)

        # Create study
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name
        )

        # Run optimization
        start_time = datetime.now()

        logger.info(f"Starting optimization: {self.n_trials} trials, objective={self.objective}")

        # Suppress Optuna logs for cleaner output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=verbose
        )

        duration = (datetime.now() - start_time).total_seconds()

        logger.info(f"Optimization complete in {duration:.2f}s")
        logger.info(f"Best value: {self.study.best_value:.4f}")
        logger.info(f"Best params: {self.study.best_params}")

        # Get parameter importance
        try:
            importance = get_param_importances(self.study)
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
            importance = {p.name: 0.0 for p in self.param_space}

        # Compile results
        result = OptimizationResult(
            best_params=self.study.best_params,
            best_value=self.study.best_value,
            n_trials=len(self.study.trials),
            duration_seconds=duration,
            all_trials=[
                {
                    'number': t.number,
                    'params': t.params,
                    'value': t.value,
                    'state': str(t.state)
                }
                for t in self.study.trials
            ],
            importance=importance,
            study_name=self.study_name,
            optimization_direction=self.direction
        )

        self.best_result = result
        return result

    def get_optimization_history(self) -> pd.DataFrame:
        """Get DataFrame of all trials."""
        if not self.study:
            raise ValueError("No optimization has been run yet")

        return self.study.trials_dataframe()

    def plot_optimization_history(self, output_path: Optional[str] = None):
        """Plot optimization history."""
        if not self.study:
            raise ValueError("No optimization has been run yet")

        try:
            from optuna.visualization import plot_optimization_history
            fig = plot_optimization_history(self.study)
            if output_path:
                fig.write_html(output_path)
                logger.info(f"Optimization history saved to {output_path}")
            else:
                fig.show()
        except ImportError:
            logger.warning("plotly not installed. Cannot create visualization.")

    def plot_param_importances(self, output_path: Optional[str] = None):
        """Plot parameter importance."""
        if not self.study:
            raise ValueError("No optimization has been run yet")

        try:
            from optuna.visualization import plot_param_importances
            fig = plot_param_importances(self.study)
            if output_path:
                fig.write_html(output_path)
                logger.info(f"Parameter importance plot saved to {output_path}")
            else:
                fig.show()
        except ImportError:
            logger.warning("plotly not installed. Cannot create visualization.")


class MultiObjectiveOptimizer(StrategyOptimizer):
    """
    Multi-objective optimization (e.g., Sharpe AND low drawdown).

    Uses Pareto optimization to find trade-off solutions between
    multiple competing objectives.

    Example:
        >>> optimizer = MultiObjectiveOptimizer(
        ...     strategy_class=ConnorsRSI2Strategy,
        ...     backtest_engine=engine,
        ...     objectives=['sharpe', 'max_drawdown'],
        ...     directions=['maximize', 'minimize']
        ... )
        >>> result = optimizer.optimize(train_data)
        >>> pareto_front = optimizer.get_pareto_front()
    """

    def __init__(
        self,
        strategy_class: type,
        backtest_engine: Any,
        objectives: List[str] = ['sharpe', 'max_drawdown'],
        directions: List[str] = ['maximize', 'minimize'],
        **kwargs
    ):
        """
        Initialize multi-objective optimizer.

        Args:
            strategy_class: Strategy class to optimize
            backtest_engine: BacktestEngine instance
            objectives: List of objective metrics
            directions: List of optimization directions
            **kwargs: Additional arguments passed to StrategyOptimizer
        """
        if len(objectives) != len(directions):
            raise ValueError("objectives and directions must have same length")

        kwargs.pop('objective', None)
        kwargs.pop('direction', None)

        super().__init__(
            strategy_class=strategy_class,
            backtest_engine=backtest_engine,
            objective='multi',
            direction='maximize',
            **kwargs
        )

        self.objectives = objectives
        self.directions = directions

    def _objective(self, trial: Trial) -> Tuple[float, ...]:
        """Multi-objective function."""
        params = self._suggest_params(trial)

        try:
            strategy = self.strategy_class(params=params)
            result = self.backtest_engine.run(
                ohlcv_data=self._train_data,
                symbol=self._symbol,
                strategy=strategy
            )

            metrics = result.get('metrics', {})

            if not metrics or metrics.get('total_trades', 0) == 0:
                return tuple([-999.0] * len(self.objectives))

            values = []
            for obj in self.objectives:
                if obj == 'sharpe':
                    values.append(metrics.get('sharpe_ratio', -999.0))
                elif obj == 'max_drawdown':
                    values.append(abs(metrics.get('max_drawdown', 0.0)))
                elif obj == 'profit_factor':
                    values.append(metrics.get('profit_factor', 0.0))
                elif obj == 'win_rate':
                    values.append(metrics.get('win_rate', 0.0))
                elif obj == 'sortino':
                    values.append(metrics.get('sortino_ratio', -999.0))
                elif obj == 'total_return':
                    values.append(metrics.get('total_return_pct', -999.0))
                else:
                    raise ValueError(f"Unknown objective: {obj}")

            return tuple(values)

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return tuple([-999.0] * len(self.objectives))

    def optimize(
        self,
        train_data: pd.DataFrame,
        symbol: str = "OPT_TEST",
        verbose: bool = True
    ) -> OptimizationResult:
        """Run multi-objective optimization."""
        if not self.param_space:
            raise ValueError("Parameter space not defined")

        self._train_data = train_data
        self._symbol = symbol

        if self.sampler == 'tpe':
            sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
        else:
            sampler = optuna.samplers.RandomSampler(seed=42)

        self.study = optuna.create_study(
            directions=self.directions,
            sampler=sampler,
            study_name=self.study_name
        )

        start_time = datetime.now()

        logger.info(f"Starting multi-objective optimization: {self.n_trials} trials")
        logger.info(f"Objectives: {self.objectives}")
        logger.info(f"Directions: {self.directions}")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=verbose
        )

        duration = (datetime.now() - start_time).total_seconds()

        logger.info(f"Multi-objective optimization complete in {duration:.2f}s")

        best_trials = self.study.best_trials
        if best_trials:
            best_trial = best_trials[0]
            best_params = best_trial.params
            best_value = best_trial.values
        else:
            best_params = {}
            best_value = tuple([0.0] * len(self.objectives))

        logger.info(f"Best Pareto solution: {best_value}")
        logger.info(f"Best params: {best_params}")

        try:
            importance = get_param_importances(self.study, target=lambda t: t.values[0])
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
            importance = {p.name: 0.0 for p in self.param_space}

        result = OptimizationResult(
            best_params=best_params,
            best_value=best_value[0] if isinstance(best_value, tuple) else best_value,
            n_trials=len(self.study.trials),
            duration_seconds=duration,
            all_trials=[
                {
                    'number': t.number,
                    'params': t.params,
                    'values': t.values if hasattr(t, 'values') else [t.value],
                    'state': str(t.state)
                }
                for t in self.study.trials
            ],
            importance=importance,
            study_name=self.study_name,
            optimization_direction=str(self.directions)
        )

        self.best_result = result
        return result

    def get_pareto_front(self) -> List[dict]:
        """Get Pareto-optimal trials."""
        if not self.study:
            raise ValueError("No optimization has been run yet")

        pareto_trials = []
        for trial in self.study.best_trials:
            pareto_trials.append({
                'params': trial.params,
                'values': trial.values,
                'number': trial.number
            })

        logger.info(f"Found {len(pareto_trials)} Pareto-optimal solutions")
        return pareto_trials

    def plot_pareto_front(self, output_path: Optional[str] = None):
        """Plot Pareto front (requires 2 objectives)."""
        if not self.study:
            raise ValueError("No optimization has been run yet")

        if len(self.objectives) != 2:
            raise ValueError("Pareto front plot requires exactly 2 objectives")

        try:
            from optuna.visualization import plot_pareto_front
            fig = plot_pareto_front(self.study, target_names=self.objectives)
            if output_path:
                fig.write_html(output_path)
                logger.info(f"Pareto front saved to {output_path}")
            else:
                fig.show()
        except ImportError:
            logger.warning("plotly not installed. Cannot create visualization.")


# =============================================================================
# PRE-DEFINED PARAMETER SPACES FOR KOBE STRATEGIES
# =============================================================================

# Connors RSI-2 Strategy parameter space
CONNORS_RSI2_PARAM_SPACE = [
    ParameterSpace('rsi_period', 'int', low=2, high=5, step=1),
    ParameterSpace('rsi_threshold', 'float', low=5.0, high=15.0, step=1.0),
    ParameterSpace('sma_period', 'categorical', choices=[150, 200, 250]),
    ParameterSpace('exit_rsi_threshold', 'int', low=55, high=75, step=5),
    ParameterSpace('atr_period', 'int', low=10, high=20, step=2),
    ParameterSpace('atr_multiplier', 'float', low=1.5, high=3.0, step=0.5),
    ParameterSpace('max_hold_bars', 'int', low=3, high=7, step=1),
]

# IBS (Internal Bar Strength) Strategy parameter space
IBS_PARAM_SPACE = [
    ParameterSpace('ibs_threshold', 'float', low=0.1, high=0.3, step=0.05),
    ParameterSpace('sma_period', 'categorical', choices=[150, 200, 250]),
    ParameterSpace('exit_ibs_threshold', 'float', low=0.7, high=0.9, step=0.05),
    ParameterSpace('atr_period', 'int', low=10, high=20, step=2),
    ParameterSpace('atr_multiplier', 'float', low=1.5, high=3.0, step=0.5),
    ParameterSpace('max_hold_bars', 'int', low=3, high=7, step=1),
]

# Quick parameter space for fast testing
QUICK_PARAM_SPACE = [
    ParameterSpace('rsi_threshold', 'float', low=8.0, high=12.0, step=1.0),
    ParameterSpace('exit_rsi_threshold', 'int', low=60, high=70, step=5),
    ParameterSpace('atr_multiplier', 'float', low=2.0, high=2.5, step=0.25),
]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_optimize(
    strategy_class: type,
    backtest_engine: Any,
    train_data: pd.DataFrame,
    n_trials: int = 50,
    objective: str = 'sharpe',
    param_space: Optional[List[ParameterSpace]] = None
) -> OptimizationResult:
    """
    Quick optimization with default Connors RSI-2 parameter space.

    Args:
        strategy_class: Strategy class to optimize
        backtest_engine: BacktestEngine instance
        train_data: Historical OHLCV data
        n_trials: Number of trials (default: 50)
        objective: Metric to optimize (default: 'sharpe')
        param_space: Custom parameter space (default: CONNORS_RSI2_PARAM_SPACE)

    Returns:
        OptimizationResult with best parameters

    Example:
        >>> from backtest.engine import BacktestEngine
        >>> from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
        >>> result = quick_optimize(
        ...     ConnorsRSI2Strategy,
        ...     backtest_engine,
        ...     train_data,
        ...     n_trials=100
        ... )
        >>> print(result.best_params)
    """
    optimizer = StrategyOptimizer(
        strategy_class=strategy_class,
        backtest_engine=backtest_engine,
        objective=objective,
        n_trials=n_trials
    )

    space = param_space or CONNORS_RSI2_PARAM_SPACE
    optimizer.define_parameter_space(space)

    return optimizer.optimize(train_data, verbose=True)


def multi_objective_optimize(
    strategy_class: type,
    backtest_engine: Any,
    train_data: pd.DataFrame,
    n_trials: int = 100,
    param_space: Optional[List[ParameterSpace]] = None
) -> Tuple[OptimizationResult, List[dict]]:
    """
    Multi-objective optimization (Sharpe + low drawdown).

    Args:
        strategy_class: Strategy class to optimize
        backtest_engine: BacktestEngine instance
        train_data: Historical OHLCV data
        n_trials: Number of trials
        param_space: Custom parameter space

    Returns:
        Tuple of (OptimizationResult, Pareto front solutions)

    Example:
        >>> result, pareto = multi_objective_optimize(
        ...     ConnorsRSI2Strategy,
        ...     backtest_engine,
        ...     train_data
        ... )
        >>> print(f"Found {len(pareto)} Pareto-optimal solutions")
    """
    optimizer = MultiObjectiveOptimizer(
        strategy_class=strategy_class,
        backtest_engine=backtest_engine,
        objectives=['sharpe', 'max_drawdown'],
        directions=['maximize', 'minimize'],
        n_trials=n_trials
    )

    space = param_space or CONNORS_RSI2_PARAM_SPACE
    optimizer.define_parameter_space(space)

    result = optimizer.optimize(train_data, verbose=True)
    pareto_front = optimizer.get_pareto_front()

    return result, pareto_front
