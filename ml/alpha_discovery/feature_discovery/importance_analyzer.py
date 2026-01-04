"""
Feature importance analysis using permutation importance and SHAP.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@dataclass
class FeatureImportanceReport:
    """Complete report on feature predictive power."""
    report_id: str
    generated_at: datetime
    strategy: str
    total_trades: int
    total_features: int

    # Rankings
    permutation_importance: Dict[str, float] = field(default_factory=dict)
    shap_values: Dict[str, float] = field(default_factory=dict)
    correlation_with_win: Dict[str, float] = field(default_factory=dict)

    # Top features
    top_10_features: List[str] = field(default_factory=list)
    bottom_10_features: List[str] = field(default_factory=list)

    # Feature groups
    best_momentum_features: List[str] = field(default_factory=list)
    best_volatility_features: List[str] = field(default_factory=list)
    best_volume_features: List[str] = field(default_factory=list)

    # Recommendations
    features_to_add: List[str] = field(default_factory=list)
    features_to_remove: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'report_id': self.report_id,
            'generated_at': self.generated_at.isoformat(),
            'strategy': self.strategy,
            'total_trades': self.total_trades,
            'total_features': self.total_features,
            'permutation_importance': self.permutation_importance,
            'shap_values': self.shap_values,
            'correlation_with_win': self.correlation_with_win,
            'top_10_features': self.top_10_features,
            'bottom_10_features': self.bottom_10_features,
            'best_momentum_features': self.best_momentum_features,
            'best_volatility_features': self.best_volatility_features,
            'best_volume_features': self.best_volume_features,
            'features_to_add': self.features_to_add,
            'features_to_remove': self.features_to_remove,
        }

    def save(self, path: str) -> None:
        """Save report to JSON file."""
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance using multiple methods.

    Methods:
    - Permutation importance (model-agnostic)
    - SHAP values (if available)
    - Correlation analysis
    """

    def __init__(
        self,
        use_shap: bool = True,
        use_permutation: bool = True,
        n_permutations: int = 10,
        random_state: int = 42,
    ):
        """
        Initialize analyzer.

        Args:
            use_shap: Calculate SHAP values if available
            use_permutation: Calculate permutation importance
            n_permutations: Number of permutation iterations
            random_state: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

        self.use_shap = use_shap and SHAP_AVAILABLE
        self.use_permutation = use_permutation
        self.n_permutations = n_permutations
        self.random_state = random_state
        self._latest_report: Optional[FeatureImportanceReport] = None

    def analyze(
        self,
        trades_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_col: str = 'won',
        strategy: str = 'all',
    ) -> FeatureImportanceReport:
        """
        Run full feature importance analysis.

        Args:
            trades_df: DataFrame with trade data and features
            feature_cols: List of feature columns (auto-detect if None)
            target_col: Target column (binary win/loss)
            strategy: Strategy name for the report

        Returns:
            FeatureImportanceReport with rankings and recommendations
        """
        if trades_df.empty:
            logger.warning("Empty trades DataFrame")
            return self._empty_report(strategy)

        # Auto-detect feature columns
        if feature_cols is None:
            exclude_cols = {'timestamp', 'symbol', 'side', 'strategy', 'split', target_col, 'pnl'}
            feature_cols = [c for c in trades_df.columns
                           if c not in exclude_cols
                           and trades_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

        if not feature_cols:
            logger.warning("No numeric feature columns found")
            return self._empty_report(strategy)

        # Prepare data
        X = trades_df[feature_cols].fillna(0)
        y = trades_df[target_col] if target_col in trades_df.columns else (trades_df['pnl'] > 0).astype(int)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Train a simple model
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)

        # Calculate importance metrics
        perm_importance = {}
        shap_importance = {}
        correlations = {}

        # Permutation importance
        if self.use_permutation:
            perm_result = permutation_importance(
                model, X_test, y_test,
                n_repeats=self.n_permutations,
                random_state=self.random_state,
            )
            perm_importance = dict(zip(feature_cols, perm_result.importances_mean.tolist()))

        # SHAP values
        if self.use_shap and SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                mean_shap = np.abs(shap_values).mean(axis=0)
                shap_importance = dict(zip(feature_cols, mean_shap.tolist()))
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}")

        # Correlations
        for col in feature_cols:
            try:
                corr = X[col].corr(y.astype(float))
                correlations[col] = float(corr) if not np.isnan(corr) else 0.0
            except (ValueError, TypeError):
                correlations[col] = 0.0

        # Rank features
        combined_importance = {}
        for col in feature_cols:
            score = 0.0
            count = 0
            if col in perm_importance:
                score += perm_importance[col]
                count += 1
            if col in shap_importance:
                score += shap_importance[col]
                count += 1
            if col in correlations:
                score += abs(correlations[col])
                count += 1
            combined_importance[col] = score / max(count, 1)

        sorted_features = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
        top_10 = [f[0] for f in sorted_features[:10]]
        bottom_10 = [f[0] for f in sorted_features[-10:]]

        # Categorize features
        momentum_keywords = ['return', 'rsi', 'macd', 'momentum', 'mom']
        volatility_keywords = ['atr', 'vol', 'std', 'range', 'bb']
        volume_keywords = ['volume', 'obv', 'vwap', 'adv']

        best_momentum = [f for f in top_10 if any(k in f.lower() for k in momentum_keywords)]
        best_volatility = [f for f in top_10 if any(k in f.lower() for k in volatility_keywords)]
        best_volume = [f for f in top_10 if any(k in f.lower() for k in volume_keywords)]

        # Recommendations
        features_to_remove = [f for f in bottom_10 if combined_importance[f] < 0.01]

        report = FeatureImportanceReport(
            report_id=f"importance_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.utcnow(),
            strategy=strategy,
            total_trades=len(trades_df),
            total_features=len(feature_cols),
            permutation_importance=perm_importance,
            shap_values=shap_importance,
            correlation_with_win=correlations,
            top_10_features=top_10,
            bottom_10_features=bottom_10,
            best_momentum_features=best_momentum,
            best_volatility_features=best_volatility,
            best_volume_features=best_volume,
            features_to_add=[],  # Would require domain knowledge
            features_to_remove=features_to_remove,
        )

        self._latest_report = report
        logger.info(f"Analyzed {len(feature_cols)} features, top: {top_10[:3]}")
        return report

    def _empty_report(self, strategy: str) -> FeatureImportanceReport:
        """Create empty report for error cases."""
        return FeatureImportanceReport(
            report_id=f"empty_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.utcnow(),
            strategy=strategy,
            total_trades=0,
            total_features=0,
        )

    def get_latest_report(self) -> Optional[FeatureImportanceReport]:
        """Get the most recent report."""
        return self._latest_report
