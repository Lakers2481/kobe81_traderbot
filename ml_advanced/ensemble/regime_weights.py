"""
Regime-Conditional Ensemble Weights
====================================

Dynamically adjusts ensemble model weights based on detected market regime.
Different models perform better in different market conditions:
- Bull regime: Momentum models may outperform
- Bear regime: Mean-reversion models may outperform
- Neutral/Choppy regime: Balanced weights

Usage:
    from ml_advanced.ensemble.regime_weights import RegimeWeightAdjuster

    adjuster = RegimeWeightAdjuster()

    # Get regime-adjusted weights
    weights = adjuster.get_weights(regime="bull", base_weights={"lstm": 0.4, "xgb": 0.3, "lgb": 0.3})

    # Update with performance feedback
    adjuster.update_performance(regime="bull", model="lstm", accuracy=0.65)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RegimePerformance:
    """Track model performance within a regime."""
    model_name: str
    regime: str
    n_predictions: int = 0
    n_correct: int = 0
    total_return: float = 0.0
    sum_squared_return: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.n_predictions == 0:
            return 0.5  # Prior
        return self.n_correct / self.n_predictions

    @property
    def mean_return(self) -> float:
        if self.n_predictions == 0:
            return 0.0
        return self.total_return / self.n_predictions

    @property
    def sharpe(self) -> float:
        if self.n_predictions < 2:
            return 0.0
        mean_ret = self.mean_return
        variance = (self.sum_squared_return / self.n_predictions) - (mean_ret ** 2)
        std_ret = np.sqrt(max(0, variance))
        if std_ret == 0:
            return 0.0
        return mean_ret / std_ret * np.sqrt(252)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'regime': self.regime,
            'n_predictions': self.n_predictions,
            'accuracy': round(self.accuracy, 4),
            'mean_return': round(self.mean_return, 6),
            'sharpe': round(self.sharpe, 4),
        }


@dataclass
class RegimeWeights:
    """Weights for a specific regime."""
    regime: str
    weights: Dict[str, float]
    confidence: float = 1.0  # How confident we are in these weights
    last_updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Regime Weight Adjuster
# =============================================================================

class RegimeWeightAdjuster:
    """
    Adjusts ensemble weights based on market regime.

    Tracks model performance per regime and learns optimal weights
    over time. Uses Bayesian-like updating with prior weights.

    Args:
        models: List of model names in the ensemble
        regimes: List of regime names
        default_weights: Default equal weights if not specified
        learning_rate: How quickly to adjust weights based on performance
        min_samples: Minimum samples before adjusting weights
        state_file: Path to persist state
    """

    REGIMES = ["bull", "bear", "neutral", "high_volatility", "low_volatility"]

    def __init__(
        self,
        models: Optional[List[str]] = None,
        regimes: Optional[List[str]] = None,
        default_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.1,
        min_samples: int = 20,
        state_file: str = "state/regime_weights.json",
    ):
        self.models = models or ["lstm", "xgboost", "lightgbm"]
        self.regimes = regimes or self.REGIMES
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        self.state_file = Path(state_file)

        # Default equal weights
        if default_weights is None:
            n_models = len(self.models)
            default_weights = {m: 1.0 / n_models for m in self.models}
        self.default_weights = default_weights

        # Initialize regime weights with defaults
        self._regime_weights: Dict[str, RegimeWeights] = {}
        for regime in self.regimes:
            self._regime_weights[regime] = RegimeWeights(
                regime=regime,
                weights=default_weights.copy(),
                confidence=0.5,  # Low initial confidence
                last_updated=datetime.utcnow().isoformat(),
            )

        # Track performance per regime per model
        self._performance: Dict[str, Dict[str, RegimePerformance]] = {}
        for regime in self.regimes:
            self._performance[regime] = {}
            for model in self.models:
                self._performance[regime][model] = RegimePerformance(
                    model_name=model, regime=regime
                )

        # Load existing state
        self._load_state()

        logger.info(
            f"RegimeWeightAdjuster initialized: models={self.models}, "
            f"regimes={self.regimes}, lr={learning_rate}"
        )

    def get_weights(
        self,
        regime: str,
        base_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Get regime-adjusted weights.

        Args:
            regime: Current market regime
            base_weights: Optional base weights to blend with regime weights

        Returns:
            Dictionary of model weights (sum to 1.0)
        """
        if regime not in self._regime_weights:
            logger.warning(f"Unknown regime: {regime}, using neutral")
            regime = "neutral"

        regime_w = self._regime_weights[regime]

        if base_weights is None:
            weights = regime_w.weights.copy()
        else:
            # Blend base weights with regime weights based on confidence
            weights = {}
            for model in self.models:
                base = base_weights.get(model, self.default_weights.get(model, 0))
                regime_val = regime_w.weights.get(model, self.default_weights.get(model, 0))
                # Higher confidence = more regime weight influence
                weights[model] = (1 - regime_w.confidence) * base + regime_w.confidence * regime_val

        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        logger.debug(f"Regime {regime} weights: {weights}")
        return weights

    def update_performance(
        self,
        regime: str,
        model: str,
        correct: bool = True,
        return_pct: float = 0.0,
    ) -> None:
        """
        Update model performance for a regime.

        Args:
            regime: Current market regime
            model: Model name
            correct: Whether prediction was correct
            return_pct: Realized return from the trade
        """
        if regime not in self._performance:
            logger.warning(f"Unknown regime: {regime}")
            return

        if model not in self._performance[regime]:
            self._performance[regime][model] = RegimePerformance(
                model_name=model, regime=regime
            )

        perf = self._performance[regime][model]
        perf.n_predictions += 1
        if correct:
            perf.n_correct += 1
        perf.total_return += return_pct
        perf.sum_squared_return += return_pct ** 2

        # Update weights if we have enough samples
        if perf.n_predictions >= self.min_samples:
            self._update_regime_weights(regime)

        # Persist state
        self._save_state()

    def _update_regime_weights(self, regime: str) -> None:
        """Recompute weights for a regime based on performance."""
        perf_dict = self._performance.get(regime, {})

        # Calculate performance score for each model
        scores = {}
        for model, perf in perf_dict.items():
            if perf.n_predictions >= self.min_samples:
                # Combine accuracy and Sharpe for score
                score = 0.6 * perf.accuracy + 0.4 * (perf.sharpe / 2 + 0.5)  # Normalize Sharpe
                scores[model] = max(0.01, score)  # Minimum score to prevent zero weights
            else:
                scores[model] = self.default_weights.get(model, 1.0 / len(self.models))

        # Normalize scores to weights
        total_score = sum(scores.values())
        if total_score > 0:
            new_weights = {m: s / total_score for m, s in scores.items()}
        else:
            new_weights = self.default_weights.copy()

        # Blend with existing weights using learning rate
        old_weights = self._regime_weights[regime].weights
        blended_weights = {}
        for model in self.models:
            old_w = old_weights.get(model, self.default_weights.get(model, 0))
            new_w = new_weights.get(model, self.default_weights.get(model, 0))
            blended_weights[model] = (1 - self.learning_rate) * old_w + self.learning_rate * new_w

        # Normalize blended weights
        total = sum(blended_weights.values())
        if total > 0:
            blended_weights = {k: v / total for k, v in blended_weights.items()}

        # Update confidence based on number of samples
        total_samples = sum(p.n_predictions for p in perf_dict.values())
        confidence = min(1.0, total_samples / (self.min_samples * len(self.models) * 5))

        self._regime_weights[regime] = RegimeWeights(
            regime=regime,
            weights=blended_weights,
            confidence=confidence,
            last_updated=datetime.utcnow().isoformat(),
        )

        logger.info(
            f"Updated regime {regime} weights: {blended_weights}, confidence={confidence:.2f}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get regime weight statistics."""
        stats = {
            'regimes': {},
            'performance': {},
        }

        for regime in self.regimes:
            stats['regimes'][regime] = self._regime_weights[regime].to_dict()
            stats['performance'][regime] = {
                model: perf.to_dict()
                for model, perf in self._performance.get(regime, {}).items()
            }

        return stats

    def get_best_model(self, regime: str) -> Tuple[str, float]:
        """Get the best performing model for a regime."""
        weights = self.get_weights(regime)
        best_model = max(weights, key=weights.get)
        return best_model, weights[best_model]

    def _save_state(self) -> None:
        """Persist state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'regime_weights': {
                regime: rw.to_dict() for regime, rw in self._regime_weights.items()
            },
            'performance': {
                regime: {
                    model: asdict(perf)
                    for model, perf in model_perfs.items()
                }
                for regime, model_perfs in self._performance.items()
            },
            'last_updated': datetime.utcnow().isoformat(),
        }

        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save regime weights state: {e}")

    def _load_state(self) -> None:
        """Load state from file."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Restore regime weights
            for regime, rw_dict in state.get('regime_weights', {}).items():
                if regime in self._regime_weights:
                    self._regime_weights[regime] = RegimeWeights(**rw_dict)

            # Restore performance
            for regime, model_perfs in state.get('performance', {}).items():
                if regime not in self._performance:
                    self._performance[regime] = {}
                for model, perf_dict in model_perfs.items():
                    self._performance[regime][model] = RegimePerformance(**perf_dict)

            logger.info(f"Loaded regime weights state from {self.state_file}")

        except Exception as e:
            logger.warning(f"Failed to load regime weights state: {e}")

    def reset(self) -> None:
        """Reset all weights to defaults."""
        for regime in self.regimes:
            self._regime_weights[regime] = RegimeWeights(
                regime=regime,
                weights=self.default_weights.copy(),
                confidence=0.5,
                last_updated=datetime.utcnow().isoformat(),
            )
            self._performance[regime] = {
                model: RegimePerformance(model_name=model, regime=regime)
                for model in self.models
            }
        self._save_state()
        logger.info("Regime weights reset to defaults")


# =============================================================================
# Global Instance
# =============================================================================

_global_adjuster: Optional[RegimeWeightAdjuster] = None


def get_regime_weight_adjuster(
    models: Optional[List[str]] = None,
    **kwargs,
) -> RegimeWeightAdjuster:
    """Get or create global regime weight adjuster."""
    global _global_adjuster

    if _global_adjuster is None:
        _global_adjuster = RegimeWeightAdjuster(models=models, **kwargs)

    return _global_adjuster


def set_regime_weight_adjuster(adjuster: RegimeWeightAdjuster) -> None:
    """Set global regime weight adjuster."""
    global _global_adjuster
    _global_adjuster = adjuster


def get_regime_adjusted_weights(
    regime: str,
    base_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Convenience function to get regime-adjusted weights."""
    adjuster = get_regime_weight_adjuster()
    return adjuster.get_weights(regime, base_weights)
