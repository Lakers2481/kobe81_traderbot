"""
Qlib-Inspired Alpha Factory for Kobe Trading System.

Provides a unified workflow for alpha research:
1. YAML-based workflow configuration
2. Automated feature engineering
3. Model training and evaluation
4. Backtesting integration
5. Report generation

Created: 2026-01-07
Based on: Microsoft Qlib patterns
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import hashlib

import numpy as np
import pandas as pd

# Try importing YAML
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None


logger = logging.getLogger(__name__)


# =============================================================================
# WORKFLOW CONFIGURATION
# =============================================================================

@dataclass
class DataConfig:
    """Data preparation configuration."""
    universe_path: str = "data/universe/optionable_liquid_800.csv"
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    min_history_days: int = 252
    provider: str = "polygon"  # polygon, stooq, yfinance


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    alpha_categories: List[str] = field(default_factory=lambda: [
        "momentum", "mean_reversion", "volatility", "volume", "technical"
    ])
    lookback_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    normalize: bool = True
    lag_features: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5])


@dataclass
class ModelConfig:
    """Model training configuration."""
    model_type: str = "lightgbm"  # lightgbm, xgboost, linear, ensemble
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    target_horizon: int = 5  # days ahead
    target_type: str = "returns"  # returns, direction, quantile
    hyperparams: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000.0
    max_positions: int = 10
    position_size: float = 0.10  # 10% per position
    commission: float = 0.001
    slippage: float = 0.001
    rebalance_freq: str = "daily"  # daily, weekly


@dataclass
class WorkflowConfig:
    """Complete workflow configuration."""
    name: str = "default_workflow"
    description: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    output_dir: str = "reports/alpha_factory"
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "WorkflowConfig":
        """Load configuration from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML required: pip install pyyaml")

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkflowConfig":
        """Create config from dictionary."""
        return cls(
            name=d.get("name", "default_workflow"),
            description=d.get("description", ""),
            data=DataConfig(**d.get("data", {})),
            features=FeatureConfig(**d.get("features", {})),
            model=ModelConfig(**d.get("model", {})),
            backtest=BacktestConfig(**d.get("backtest", {})),
            output_dir=d.get("output_dir", "reports/alpha_factory"),
            random_seed=d.get("random_seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "data": self.data.__dict__,
            "features": self.features.__dict__,
            "model": self.model.__dict__,
            "backtest": self.backtest.__dict__,
            "output_dir": self.output_dir,
            "random_seed": self.random_seed,
        }

    def get_hash(self) -> str:
        """Get deterministic hash for reproducibility."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]


# =============================================================================
# WORKFLOW RESULTS
# =============================================================================

@dataclass
class WorkflowResult:
    """Results from a complete workflow run."""
    workflow_id: str
    config_hash: str
    timestamp: str

    # Data stats
    symbols_loaded: int = 0
    date_range: tuple = ("", "")

    # Feature stats
    features_generated: int = 0
    alphas_used: List[str] = field(default_factory=list)

    # Model stats
    model_type: str = ""
    train_score: float = 0.0
    valid_score: float = 0.0
    test_score: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)

    # Backtest stats
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0

    # Paths
    model_path: str = ""
    report_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "config_hash": self.config_hash,
            "timestamp": self.timestamp,
            "data": {
                "symbols_loaded": self.symbols_loaded,
                "date_range": self.date_range,
            },
            "features": {
                "features_generated": self.features_generated,
                "alphas_used": self.alphas_used,
            },
            "model": {
                "model_type": self.model_type,
                "train_score": self.train_score,
                "valid_score": self.valid_score,
                "test_score": self.test_score,
                "feature_importance": self.feature_importance,
            },
            "backtest": {
                "total_return": self.total_return,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "num_trades": self.num_trades,
            },
            "paths": {
                "model_path": self.model_path,
                "report_path": self.report_path,
            },
        }


# =============================================================================
# ALPHA FACTORY
# =============================================================================

class AlphaFactory:
    """
    Qlib-inspired alpha research factory.

    Orchestrates the complete alpha research workflow:
    1. Data preparation
    2. Feature engineering (using AlphaLibrary)
    3. Model training
    4. Backtesting
    5. Report generation

    Usage:
        factory = AlphaFactory()
        result = factory.run_workflow("config/alpha_workflows/momentum.yaml")

        # Or with dict config
        result = factory.run_workflow(config_dict)
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or "reports/alpha_factory")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._alpha_library = None
        self._vectorbt_miner = None
        self._factor_validator = None

        logger.info("AlphaFactory initialized")

    # -------------------------------------------------------------------------
    # Component Access
    # -------------------------------------------------------------------------

    @property
    def alpha_library(self):
        """Lazy load AlphaLibrary."""
        if self._alpha_library is None:
            try:
                from research.alpha_library import get_alpha_library
                self._alpha_library = get_alpha_library()
            except ImportError:
                logger.warning("AlphaLibrary not available")
        return self._alpha_library

    @property
    def vectorbt_miner(self):
        """Lazy load VectorBT miner."""
        if self._vectorbt_miner is None:
            try:
                from research.vectorbt_miner import VectorBTMiner
                self._vectorbt_miner = VectorBTMiner()
            except ImportError:
                logger.warning("VectorBTMiner not available")
        return self._vectorbt_miner

    def get_factor_validator(self, prices: pd.DataFrame):
        """Get FactorValidator with prices."""
        try:
            from research.factor_validator import FactorValidator, HAS_ALPHALENS
            if HAS_ALPHALENS:
                return FactorValidator(prices)
        except ImportError:
            pass
        return None

    # -------------------------------------------------------------------------
    # Workflow Execution
    # -------------------------------------------------------------------------

    def run_workflow(
        self,
        config: WorkflowConfig | Dict[str, Any] | str,
        verbose: bool = True,
    ) -> WorkflowResult:
        """
        Run complete alpha research workflow.

        Args:
            config: WorkflowConfig, dict, or path to YAML file
            verbose: Print progress

        Returns:
            WorkflowResult with all metrics and paths
        """
        # Parse config
        if isinstance(config, str):
            config = WorkflowConfig.from_yaml(config)
        elif isinstance(config, dict):
            config = WorkflowConfig.from_dict(config)

        # Create workflow ID
        workflow_id = f"{config.name}_{config.get_hash()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if verbose:
            print(f"\n{'='*60}")
            print(f"ALPHA FACTORY WORKFLOW: {config.name}")
            print(f"{'='*60}")

        result = WorkflowResult(
            workflow_id=workflow_id,
            config_hash=config.get_hash(),
            timestamp=datetime.now().isoformat(),
        )

        # Step 1: Data Preparation
        if verbose:
            print("\n[1/5] Data Preparation...")
        data = self._prepare_data(config.data)
        result.symbols_loaded = len(data) if isinstance(data, dict) else 1
        result.date_range = (config.data.start_date, config.data.end_date)

        # Step 2: Feature Engineering
        if verbose:
            print("\n[2/5] Feature Engineering...")
        features, alphas_used = self._extract_features(data, config.features)
        result.features_generated = len(features.columns) if features is not None else 0
        result.alphas_used = alphas_used

        # Step 3: Model Training
        if verbose:
            print("\n[3/5] Model Training...")
        model, train_score, valid_score, test_score, importance = self._train_model(
            features, data, config.model
        )
        result.model_type = config.model.model_type
        result.train_score = train_score
        result.valid_score = valid_score
        result.test_score = test_score
        result.feature_importance = importance

        # Step 4: Backtesting
        if verbose:
            print("\n[4/5] Backtesting...")
        backtest_results = self._run_backtest(model, data, features, config.backtest)
        result.total_return = backtest_results.get("total_return", 0.0)
        result.sharpe_ratio = backtest_results.get("sharpe_ratio", 0.0)
        result.max_drawdown = backtest_results.get("max_drawdown", 0.0)
        result.win_rate = backtest_results.get("win_rate", 0.0)
        result.profit_factor = backtest_results.get("profit_factor", 0.0)
        result.num_trades = backtest_results.get("num_trades", 0)

        # Step 5: Report Generation
        if verbose:
            print("\n[5/5] Generating Report...")
        report_path = self._generate_report(result, config)
        result.report_path = str(report_path)

        if verbose:
            print(f"\n{'='*60}")
            print("WORKFLOW COMPLETE")
            print(f"{'='*60}")
            print(f"  Symbols: {result.symbols_loaded}")
            print(f"  Features: {result.features_generated}")
            print(f"  Model: {result.model_type}")
            print(f"  Test Score: {result.test_score:.4f}")
            print(f"  Sharpe: {result.sharpe_ratio:.2f}")
            print(f"  Win Rate: {result.win_rate:.1%}")
            print(f"  Report: {result.report_path}")

        return result

    # -------------------------------------------------------------------------
    # Workflow Steps
    # -------------------------------------------------------------------------

    def _prepare_data(self, config: DataConfig) -> Dict[str, pd.DataFrame]:
        """Load and prepare data from universe."""
        data = {}

        # Load universe
        universe_path = Path(config.universe_path)
        if universe_path.exists():
            universe_df = pd.read_csv(universe_path)
            symbols = universe_df["symbol"].tolist()[:100]  # Limit for speed
        else:
            symbols = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"]

        # Load data for each symbol
        try:
            from data.providers.polygon_eod import PolygonEODProvider
            provider = PolygonEODProvider()

            for symbol in symbols:
                try:
                    df = provider.get_bars(
                        symbol,
                        config.start_date,
                        config.end_date,
                    )
                    if df is not None and len(df) >= config.min_history_days:
                        data[symbol] = df
                except Exception as e:
                    logger.debug(f"Failed to load {symbol}: {e}")

        except ImportError:
            logger.warning("Polygon provider not available, using synthetic data")
            # Generate synthetic data for testing
            for symbol in symbols[:10]:
                dates = pd.date_range(config.start_date, config.end_date, freq='D')
                np.random.seed(hash(symbol) % 2**32)
                returns = np.random.randn(len(dates)) * 0.02
                prices = 100 * np.cumprod(1 + returns)
                data[symbol] = pd.DataFrame({
                    'open': prices * 0.995,
                    'high': prices * 1.01,
                    'low': prices * 0.99,
                    'close': prices,
                    'volume': np.random.randint(1000000, 10000000, len(dates)),
                }, index=dates)

        logger.info(f"Loaded data for {len(data)} symbols")
        return data

    def _extract_features(
        self,
        data: Dict[str, pd.DataFrame],
        config: FeatureConfig,
    ) -> tuple[Optional[pd.DataFrame], List[str]]:
        """Extract alpha features from data."""
        if not data:
            return None, []

        all_features = []
        alphas_used = []

        for symbol, df in data.items():
            if self.alpha_library is not None:
                # Use AlphaLibrary for feature extraction
                symbol_features = pd.DataFrame(index=df.index)

                for category in config.alpha_categories:
                    try:
                        category_alphas = self.alpha_library.get_alphas_by_category(category)
                        for alpha_name, alpha_func in category_alphas.items():
                            try:
                                alpha_values = alpha_func(df)
                                symbol_features[f"{alpha_name}"] = alpha_values
                                if alpha_name not in alphas_used:
                                    alphas_used.append(alpha_name)
                            except Exception as e:
                                logger.debug(f"Failed to compute {alpha_name}: {e}")
                    except Exception as e:
                        logger.debug(f"Failed to get category {category}: {e}")

                symbol_features["symbol"] = symbol
                all_features.append(symbol_features)
            else:
                # Basic features without AlphaLibrary
                symbol_features = pd.DataFrame(index=df.index)
                symbol_features["returns_5d"] = df["close"].pct_change(5)
                symbol_features["returns_20d"] = df["close"].pct_change(20)
                symbol_features["volatility_20d"] = df["close"].pct_change().rolling(20).std()
                symbol_features["symbol"] = symbol
                all_features.append(symbol_features)
                alphas_used = ["returns_5d", "returns_20d", "volatility_20d"]

        if all_features:
            features = pd.concat(all_features, axis=0)

            # Normalize if requested
            if config.normalize:
                numeric_cols = features.select_dtypes(include=[np.number]).columns
                features[numeric_cols] = (
                    features[numeric_cols] - features[numeric_cols].mean()
                ) / features[numeric_cols].std()

            # Add lag features if requested
            if config.lag_features:
                numeric_cols = features.select_dtypes(include=[np.number]).columns
                for lag in config.lag_periods:
                    for col in numeric_cols[:10]:  # Limit lags
                        features[f"{col}_lag{lag}"] = features.groupby("symbol")[col].shift(lag)

            logger.info(f"Generated {len(features.columns)} features using {len(alphas_used)} alphas")
            return features, alphas_used

        return None, []

    def _train_model(
        self,
        features: Optional[pd.DataFrame],
        data: Dict[str, pd.DataFrame],
        config: ModelConfig,
    ) -> tuple[Any, float, float, float, Dict[str, float]]:
        """Train prediction model."""
        if features is None or features.empty:
            return None, 0.0, 0.0, 0.0, {}

        # Create target
        targets = []
        for symbol, df in data.items():
            target = df["close"].pct_change(config.target_horizon).shift(-config.target_horizon)
            target_df = pd.DataFrame({"target": target, "symbol": symbol}, index=df.index)
            targets.append(target_df)

        if targets:
            target_df = pd.concat(targets, axis=0)
            features = features.join(target_df["target"])

        # Drop NaN
        features = features.dropna()

        if len(features) < 100:
            logger.warning("Not enough data for training")
            return None, 0.0, 0.0, 0.0, {}

        # Split data
        n = len(features)
        train_end = int(n * config.train_ratio)
        valid_end = int(n * (config.train_ratio + config.valid_ratio))

        train = features.iloc[:train_end]
        valid = features.iloc[train_end:valid_end]
        test = features.iloc[valid_end:]

        # Get feature columns (exclude symbol and target)
        feature_cols = [c for c in features.columns if c not in ["symbol", "target"]]

        X_train = train[feature_cols].values
        y_train = train["target"].values
        X_valid = valid[feature_cols].values
        y_valid = valid["target"].values
        X_test = test[feature_cols].values
        y_test = test["target"].values

        # Train model based on type
        model = None
        importance = {}

        if config.model_type == "lightgbm":
            try:
                import lightgbm as lgb
                params = {
                    "objective": "regression",
                    "metric": "rmse",
                    "verbosity": -1,
                    "n_estimators": 100,
                    **config.hyperparams,
                }
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
                importance = dict(zip(feature_cols, model.feature_importances_))
            except ImportError:
                logger.warning("LightGBM not available")

        elif config.model_type == "xgboost":
            try:
                import xgboost as xgb
                params = {
                    "objective": "reg:squarederror",
                    "n_estimators": 100,
                    "verbosity": 0,
                    **config.hyperparams,
                }
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
                importance = dict(zip(feature_cols, model.feature_importances_))
            except ImportError:
                logger.warning("XGBoost not available")

        else:
            # Default: Linear regression
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            importance = dict(zip(feature_cols, np.abs(model.coef_)))

        # Calculate scores (R-squared)
        if model is not None:
            train_score = model.score(X_train, y_train)
            valid_score = model.score(X_valid, y_valid)
            test_score = model.score(X_test, y_test)
        else:
            train_score = valid_score = test_score = 0.0

        # Sort importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])

        logger.info(f"Model trained: train={train_score:.4f}, valid={valid_score:.4f}, test={test_score:.4f}")
        return model, train_score, valid_score, test_score, importance

    def _run_backtest(
        self,
        model: Any,
        data: Dict[str, pd.DataFrame],
        features: Optional[pd.DataFrame],
        config: BacktestConfig,
    ) -> Dict[str, float]:
        """Run backtest with trained model."""
        if model is None or features is None:
            return {}

        # Simple backtest logic
        feature_cols = [c for c in features.columns if c not in ["symbol", "target"]]

        # Get predictions
        X = features[feature_cols].dropna().values
        if len(X) == 0:
            return {}

        predictions = model.predict(X)

        # Simple P&L calculation
        returns = features["target"].dropna().values[:len(predictions)]

        # Long when prediction > 0
        positions = np.where(predictions > 0, 1, 0)
        strategy_returns = positions * returns

        # Remove NaN
        strategy_returns = strategy_returns[~np.isnan(strategy_returns)]

        if len(strategy_returns) == 0:
            return {}

        # Calculate metrics
        total_return = np.sum(strategy_returns)
        sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)

        # Max drawdown
        cumulative = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown)

        # Win rate
        trades = strategy_returns[positions[:-1] == 1] if len(positions) > 1 else strategy_returns
        wins = np.sum(trades > 0)
        total_trades = len(trades)
        win_rate = wins / total_trades if total_trades > 0 else 0

        # Profit factor
        gross_profit = np.sum(trades[trades > 0])
        gross_loss = np.abs(np.sum(trades[trades < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "num_trades": total_trades,
        }

    def _generate_report(
        self,
        result: WorkflowResult,
        config: WorkflowConfig,
    ) -> Path:
        """Generate workflow report."""
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / f"{result.workflow_id}_report.json"

        report = {
            "workflow": result.to_dict(),
            "config": config.to_dict(),
            "generated_at": datetime.now().isoformat(),
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to {report_path}")
        return report_path

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def quick_alpha_test(
        self,
        alpha_name: str,
        universe: Optional[str] = None,
        start: str = "2020-01-01",
        end: str = "2024-12-31",
    ) -> Dict[str, Any]:
        """Quick test of a single alpha factor."""
        config = WorkflowConfig(
            name=f"quick_test_{alpha_name}",
            data=DataConfig(
                universe_path=universe or "data/universe/optionable_liquid_800.csv",
                start_date=start,
                end_date=end,
            ),
            features=FeatureConfig(
                alpha_categories=["all"],
            ),
            model=ModelConfig(
                model_type="linear",
            ),
        )

        result = self.run_workflow(config, verbose=False)
        return result.to_dict()

    def compare_alphas(
        self,
        alpha_names: List[str],
        **kwargs,
    ) -> pd.DataFrame:
        """Compare multiple alpha factors."""
        results = []

        for alpha in alpha_names:
            result = self.quick_alpha_test(alpha, **kwargs)
            results.append({
                "alpha": alpha,
                "test_score": result["model"]["test_score"],
                "sharpe": result["backtest"]["sharpe_ratio"],
                "win_rate": result["backtest"]["win_rate"],
            })

        return pd.DataFrame(results).sort_values("sharpe", ascending=False)


# =============================================================================
# SINGLETON
# =============================================================================

_alpha_factory: Optional[AlphaFactory] = None


def get_alpha_factory() -> AlphaFactory:
    """Get singleton AlphaFactory instance."""
    global _alpha_factory
    if _alpha_factory is None:
        _alpha_factory = AlphaFactory()
    return _alpha_factory


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    # Simple CLI
    factory = get_alpha_factory()

    if len(sys.argv) > 1:
        # Run workflow from YAML
        yaml_path = sys.argv[1]
        result = factory.run_workflow(yaml_path)
        print(f"\nReport saved to: {result.report_path}")
    else:
        # Demo with default config
        print("Running demo workflow...")
        config = WorkflowConfig(
            name="demo_workflow",
            data=DataConfig(
                start_date="2023-01-01",
                end_date="2024-12-31",
            ),
        )
        result = factory.run_workflow(config)
        print(f"\nReport saved to: {result.report_path}")
