"""
Ensemble Model Loader
=====================

Loads trained XGBoost and LightGBM models into the EnsemblePredictor.

Usage:
    from ml_advanced.ensemble.loader import load_ensemble_models

    predictor = load_ensemble_models()
    if predictor:
        result = predictor.predict_with_confidence(features)
"""

import json
import logging
from pathlib import Path
from typing import Optional

from .ensemble_predictor import (
    EnsemblePredictor,
    XGBoostWrapper,
    LightGBMWrapper,
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
)

logger = logging.getLogger(__name__)

# Default model directory
ENSEMBLE_MODEL_DIR = Path(__file__).resolve().parents[2] / "state" / "models" / "ensemble"


def load_ensemble_models(
    model_dir: Optional[Path] = None,
) -> Optional[EnsemblePredictor]:
    """
    Load trained XGBoost and LightGBM models into an EnsemblePredictor.

    Args:
        model_dir: Directory containing models and config. Defaults to
                   state/models/ensemble/

    Returns:
        Configured EnsemblePredictor with models loaded, or None if no models found.
    """
    if model_dir is None:
        model_dir = ENSEMBLE_MODEL_DIR

    if not model_dir.exists():
        logger.warning(f"Ensemble model directory not found: {model_dir}")
        return None

    # Load config if available
    config_path = model_dir / "ensemble_config.json"
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            logger.info(f"Loaded ensemble config: {list(config.get('models', {}).keys())}")
        except Exception as e:
            logger.warning(f"Failed to load ensemble config: {e}")

    # Create predictor
    predictor = EnsemblePredictor()
    models_loaded = 0

    # Load XGBoost model
    xgb_path = model_dir / "xgboost_model.json"
    if xgb_path.exists() and XGBOOST_AVAILABLE:
        try:
            xgb_wrapper = XGBoostWrapper(name="xgboost", task_type="classification")
            xgb_wrapper.load(xgb_path)

            # Get weight from config or use default
            weight = config.get("models", {}).get("xgboost", {}).get("weight", 0.5)
            predictor.add_model("xgboost", xgb_wrapper, weight=weight)
            models_loaded += 1

            metrics = config.get("models", {}).get("xgboost", {}).get("metrics", {})
            logger.info(
                f"XGBoost loaded: acc={metrics.get('test_accuracy', 'N/A')}, "
                f"auc={metrics.get('test_auc', 'N/A')}"
            )
        except Exception as e:
            logger.warning(f"Failed to load XGBoost model: {e}")
    elif not XGBOOST_AVAILABLE:
        logger.info("XGBoost not installed, skipping")

    # Load LightGBM model
    lgb_path = model_dir / "lightgbm_model.txt"
    if lgb_path.exists() and LIGHTGBM_AVAILABLE:
        try:
            lgb_wrapper = LightGBMWrapper(name="lightgbm", task_type="classification")
            lgb_wrapper.load(lgb_path)

            # Get weight from config or use default
            weight = config.get("models", {}).get("lightgbm", {}).get("weight", 0.5)
            predictor.add_model("lightgbm", lgb_wrapper, weight=weight)
            models_loaded += 1

            metrics = config.get("models", {}).get("lightgbm", {}).get("metrics", {})
            logger.info(
                f"LightGBM loaded: acc={metrics.get('test_accuracy', 'N/A')}, "
                f"auc={metrics.get('test_auc', 'N/A')}"
            )
        except Exception as e:
            logger.warning(f"Failed to load LightGBM model: {e}")
    elif not LIGHTGBM_AVAILABLE:
        logger.info("LightGBM not installed, skipping")

    if models_loaded == 0:
        logger.warning("No ensemble models loaded")
        return None

    logger.info(f"Ensemble predictor loaded with {models_loaded} model(s)")
    return predictor


# Singleton instance
_ensemble_predictor: Optional[EnsemblePredictor] = None


def get_ensemble_predictor() -> Optional[EnsemblePredictor]:
    """Get or create singleton EnsemblePredictor with loaded models."""
    global _ensemble_predictor
    if _ensemble_predictor is None:
        _ensemble_predictor = load_ensemble_models()
    return _ensemble_predictor


def reset_ensemble_predictor() -> None:
    """Reset the singleton (for testing)."""
    global _ensemble_predictor
    _ensemble_predictor = None
