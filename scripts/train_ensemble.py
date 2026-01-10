#!/usr/bin/env python3
"""
Train Ensemble Models (XGBoost + LightGBM)
==========================================

Trains XGBoost and LightGBM models for trade success prediction.
These tree-based models complement the LSTM model with:
- Feature importance insights
- Fast inference
- Robustness to feature scaling

Usage:
    python scripts/train_ensemble.py \
        --data data/training \
        --output models/ensemble_v1 \
        --n-estimators 500

Requirements:
    - xgboost
    - lightgbm
    - Training data from generate_training_data.py
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ensemble models")
    parser.add_argument(
        "--data",
        type=str,
        default="data/training",
        help="Training data directory (default: data/training)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/ensemble_v1",
        help="Output directory (default: models/ensemble_v1)"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Number of trees (default: 500)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Max tree depth (default: 6)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate (default: 0.05)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion (default: 0.2)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()


def load_training_data(data_dir: Path) -> tuple:
    """
    Load training data for tree models.

    Uses flat features (2D) rather than LSTM sequences.
    """
    features = pd.read_parquet(data_dir / "features.parquet")
    labels = pd.read_parquet(data_dir / "labels.parquet")

    # Get feature columns (exclude metadata)
    feature_cols = [c for c in features.columns
                   if c not in ['symbol', 'entry_timestamp', 'timestamp', 'date']]

    X = features[feature_cols].values
    y = labels['won'].values

    logger.info(f"Loaded training data: X={X.shape}, y={len(y)}")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Win rate: {y.mean()*100:.1f}%")

    return X, y, feature_cols


def preprocess_features(X: np.ndarray, feature_names: list) -> tuple:
    """
    Preprocess features for tree models.

    - Handle NaN values
    - Clip extreme outliers
    """
    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip extreme values
    X = np.clip(X, -100, 100)

    logger.info(f"Preprocessed: min={X.min():.2f}, max={X.max():.2f}")

    return X, feature_names


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    """Split data into train and test sets."""
    np.random.seed(42)
    n = len(X)
    indices = np.random.permutation(n)
    split_idx = int(n * (1 - test_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return (X[train_idx], X[test_idx],
            y[train_idx], y[test_idx])


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05
) -> tuple:
    """Train XGBoost classifier."""
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost not available. pip install xgboost")
        return None, {}

    logger.info("Training XGBoost...")

    # Compute scale_pos_weight for imbalanced data
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # Create DMatrix with feature names
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist',
        'random_state': 42,
    }

    evallist = [(dtrain, 'train'), (dtest, 'eval')]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=evallist,
        early_stopping_rounds=50,
        verbose_eval=50
    )

    # Evaluate
    y_pred = model.predict(dtest)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = (y_pred_binary == y_test).mean()
    auc = compute_auc(y_test, y_pred)

    metrics = {
        'accuracy': float(accuracy),
        'auc': float(auc),
        'best_iteration': model.best_iteration,
        'scale_pos_weight': scale_pos_weight,
    }

    logger.info(f"XGBoost results: accuracy={accuracy:.4f}, AUC={auc:.4f}")

    return model, metrics


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05
) -> tuple:
    """Train LightGBM classifier."""
    if not LIGHTGBM_AVAILABLE:
        logger.warning("LightGBM not available. pip install lightgbm")
        return None, {}

    logger.info("Training LightGBM...")

    # Compute class weights
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    test_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_names, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'scale_pos_weight': scale_pos_weight,
        'num_leaves': 2 ** max_depth - 1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'random_state': 42,
        'verbose': -1,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'eval'],
        callbacks=callbacks
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = (y_pred_binary == y_test).mean()
    auc = compute_auc(y_test, y_pred)

    metrics = {
        'accuracy': float(accuracy),
        'auc': float(auc),
        'best_iteration': model.best_iteration,
        'scale_pos_weight': scale_pos_weight,
    }

    logger.info(f"LightGBM results: accuracy={accuracy:.4f}, AUC={auc:.4f}")

    return model, metrics


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUC-ROC score."""
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_pred)
    except ImportError:
        # Simple AUC approximation
        return 0.5


def get_feature_importance(model, feature_names: list, model_type: str) -> pd.DataFrame:
    """Extract feature importance from trained model."""
    if model_type == 'xgboost' and model is not None:
        importance = model.get_score(importance_type='gain')
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ])
    elif model_type == 'lightgbm' and model is not None:
        importance = model.feature_importance(importance_type='gain')
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
    else:
        return pd.DataFrame()

    return df.sort_values('importance', ascending=False).reset_index(drop=True)


def save_models(
    output_dir: Path,
    xgb_model,
    lgb_model,
    feature_names: list,
    xgb_metrics: dict,
    lgb_metrics: dict
):
    """Save trained models and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save XGBoost
    if xgb_model is not None:
        xgb_path = output_dir / "xgb.json"
        xgb_model.save_model(str(xgb_path))
        logger.info(f"Saved XGBoost to {xgb_path}")

        # Also save feature importance
        xgb_importance = get_feature_importance(xgb_model, feature_names, 'xgboost')
        xgb_importance.to_csv(output_dir / "xgb_feature_importance.csv", index=False)

    # Save LightGBM
    if lgb_model is not None:
        lgb_path = output_dir / "lgb.txt"
        lgb_model.save_model(str(lgb_path))
        logger.info(f"Saved LightGBM to {lgb_path}")

        # Also save feature importance
        lgb_importance = get_feature_importance(lgb_model, feature_names, 'lightgbm')
        lgb_importance.to_csv(output_dir / "lgb_feature_importance.csv", index=False)

    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "xgboost": xgb_metrics,
        "lightgbm": lgb_metrics,
    }

    with open(output_dir / "ensemble_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {output_dir / 'ensemble_metadata.json'}")


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("=" * 60)
    logger.info("TRAIN ENSEMBLE MODELS (XGBoost + LightGBM)")
    logger.info("=" * 60)

    data_dir = Path(args.data)
    output_dir = Path(args.output)

    # Check for training data
    if not (data_dir / "features.parquet").exists():
        logger.error(f"Training data not found in {data_dir}")
        logger.error("Run: python scripts/generate_training_data.py first")
        sys.exit(1)

    # Check for libraries
    if not XGBOOST_AVAILABLE and not LIGHTGBM_AVAILABLE:
        logger.error("Neither XGBoost nor LightGBM available!")
        logger.error("Install: pip install xgboost lightgbm")
        sys.exit(1)

    # Load data
    logger.info("\n[1/4] Loading training data...")
    X, y, feature_names = load_training_data(data_dir)

    # Preprocess
    logger.info("\n[2/4] Preprocessing features...")
    X, feature_names = preprocess_features(X, feature_names)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, args.test_size)
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train models
    logger.info("\n[3/4] Training models...")

    xgb_model, xgb_metrics = train_xgboost(
        X_train, y_train, X_test, y_test, feature_names,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )

    lgb_model, lgb_metrics = train_lightgbm(
        X_train, y_train, X_test, y_test, feature_names,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )

    # Save
    logger.info("\n[4/4] Saving models...")
    save_models(output_dir, xgb_model, lgb_model, feature_names, xgb_metrics, lgb_metrics)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ENSEMBLE TRAINING COMPLETE")
    logger.info("=" * 60)

    if xgb_metrics:
        logger.info(f"XGBoost: accuracy={xgb_metrics['accuracy']:.4f}, AUC={xgb_metrics['auc']:.4f}")
    if lgb_metrics:
        logger.info(f"LightGBM: accuracy={lgb_metrics['accuracy']:.4f}, AUC={lgb_metrics['auc']:.4f}")

    logger.info(f"\nModels saved to: {output_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. python scripts/train_hmm_regime.py")
    logger.info("  2. Update ml_advanced/ensemble/loader.py to load these models")
    logger.info("  3. python scripts/scan.py --cap 50 --ml --ensemble")


if __name__ == "__main__":
    main()
