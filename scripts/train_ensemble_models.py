"""
Train XGBoost and LightGBM models for the Ensemble Predictor.

This script trains boosting models on historical signal data and saves them
in the format required by EnsemblePredictor (XGBoost Booster, LightGBM Booster).

Usage:
    python scripts/train_ensemble_models.py

Output:
    state/models/ensemble/xgboost_model.json
    state/models/ensemble/lightgbm_model.txt
    state/models/ensemble/ensemble_config.json
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

import xgboost as xgb
import lightgbm as lgb

# Feature columns expected by the models
FEATURE_COLS = [
    'atr14',
    'sma20_over_200',
    'rv20',
    'don20_width',
    'pos_in_don20',
    'ret5',
    'log_vol',
]


def load_training_data() -> pd.DataFrame:
    """Load the signal dataset for training."""
    dataset_path = ROOT / "data" / "ml" / "signal_dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training data not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df)} samples from {dataset_path}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and labels for training."""
    # Ensure all feature columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLS].fillna(0).values
    y = df['label'].values

    print(f"Features shape: {X.shape}")
    print(f"Labels: {np.sum(y == 1)} wins, {np.sum(y == 0)} losses ({100*np.mean(y):.1f}% win rate)")

    return X, y


def train_xgboost(X_train, y_train, X_val, y_val) -> tuple:
    """Train XGBoost model and return booster + metrics."""
    print("\n" + "="*60)
    print("Training XGBoost...")
    print("="*60)

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLS)

    # Parameters optimized for binary classification
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'max_depth': 4,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 50,
        'seed': 42,
        'verbosity': 1,
    }

    # Train with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    # Evaluate
    y_pred_proba = booster.predict(dval)
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        'accuracy': float(accuracy_score(y_val, y_pred)),
        'auc': float(roc_auc_score(y_val, y_pred_proba)),
        'brier': float(brier_score_loss(y_val, y_pred_proba)),
        'best_iteration': booster.best_iteration,
    }

    print("\nXGBoost Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Brier: {metrics['brier']:.4f}")
    print(f"  Best Iteration: {metrics['best_iteration']}")

    return booster, metrics


def train_lightgbm(X_train, y_train, X_val, y_val) -> tuple:
    """Train LightGBM model and return booster + metrics."""
    print("\n" + "="*60)
    print("Training LightGBM...")
    print("="*60)

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=train_data)

    # Parameters optimized for binary classification
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'seed': 42,
        'verbose': -1,
    }

    # Train with early stopping
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    # Evaluate
    y_pred_proba = booster.predict(X_val)
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        'accuracy': float(accuracy_score(y_val, y_pred)),
        'auc': float(roc_auc_score(y_val, y_pred_proba)),
        'brier': float(brier_score_loss(y_val, y_pred_proba)),
        'best_iteration': booster.best_iteration,
    }

    print("\nLightGBM Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Brier: {metrics['brier']:.4f}")
    print(f"  Best Iteration: {metrics['best_iteration']}")

    return booster, metrics


def save_models(xgb_booster, xgb_metrics, lgb_booster, lgb_metrics):
    """Save trained models and config."""
    output_dir = ROOT / "state" / "models" / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save XGBoost model
    xgb_path = output_dir / "xgboost_model.json"
    xgb_booster.save_model(str(xgb_path))
    print(f"\nSaved XGBoost model to {xgb_path}")

    # Save LightGBM model
    lgb_path = output_dir / "lightgbm_model.txt"
    lgb_booster.save_model(str(lgb_path))
    print(f"Saved LightGBM model to {lgb_path}")

    # Save ensemble config
    config = {
        'created_at': datetime.now().isoformat(),
        'feature_columns': FEATURE_COLS,
        'models': {
            'xgboost': {
                'path': 'xgboost_model.json',
                'weight': 0.5,
                'metrics': xgb_metrics,
            },
            'lightgbm': {
                'path': 'lightgbm_model.txt',
                'weight': 0.5,
                'metrics': lgb_metrics,
            },
        },
        'default_weights': {
            'xgboost': 0.5,
            'lightgbm': 0.5,
        },
    }

    config_path = output_dir / "ensemble_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Saved ensemble config to {config_path}")

    return output_dir


def main():
    print("="*60)
    print("ENSEMBLE MODEL TRAINING")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_training_data()

    # Prepare features
    X, y = prepare_features(df)

    # Time-based split (60% train, 20% val, 20% test)
    # First split: 80% train+val, 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # No shuffle for time series
    )

    # Second split: 75% train, 25% val (of the 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, shuffle=False
    )

    print("\nData splits:")
    print(f"  Train: {len(X_train)} samples ({100*np.mean(y_train):.1f}% win rate)")
    print(f"  Val:   {len(X_val)} samples ({100*np.mean(y_val):.1f}% win rate)")
    print(f"  Test:  {len(X_test)} samples ({100*np.mean(y_test):.1f}% win rate)")

    # Train models
    xgb_booster, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)
    lgb_booster, lgb_metrics = train_lightgbm(X_train, y_train, X_val, y_val)

    # Final test evaluation
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)

    # XGBoost on test
    dtest = xgb.DMatrix(X_test, feature_names=FEATURE_COLS)
    xgb_test_pred = xgb_booster.predict(dtest)
    xgb_test_acc = accuracy_score(y_test, (xgb_test_pred > 0.5).astype(int))
    xgb_test_auc = roc_auc_score(y_test, xgb_test_pred)
    print(f"XGBoost Test: Accuracy={xgb_test_acc:.4f}, AUC={xgb_test_auc:.4f}")
    xgb_metrics['test_accuracy'] = float(xgb_test_acc)
    xgb_metrics['test_auc'] = float(xgb_test_auc)

    # LightGBM on test
    lgb_test_pred = lgb_booster.predict(X_test)
    lgb_test_acc = accuracy_score(y_test, (lgb_test_pred > 0.5).astype(int))
    lgb_test_auc = roc_auc_score(y_test, lgb_test_pred)
    print(f"LightGBM Test: Accuracy={lgb_test_acc:.4f}, AUC={lgb_test_auc:.4f}")
    lgb_metrics['test_accuracy'] = float(lgb_test_acc)
    lgb_metrics['test_auc'] = float(lgb_test_auc)

    # Ensemble on test
    ensemble_pred = (xgb_test_pred + lgb_test_pred) / 2
    ensemble_acc = accuracy_score(y_test, (ensemble_pred > 0.5).astype(int))
    ensemble_auc = roc_auc_score(y_test, ensemble_pred)
    print(f"Ensemble Test: Accuracy={ensemble_acc:.4f}, AUC={ensemble_auc:.4f}")

    # Save models
    output_dir = save_models(xgb_booster, xgb_metrics, lgb_booster, lgb_metrics)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Models saved to: {output_dir}")
    print(f"Ensemble Test Accuracy: {ensemble_acc:.4f}")
    print(f"Ensemble Test AUC: {ensemble_auc:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
