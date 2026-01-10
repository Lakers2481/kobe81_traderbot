#!/usr/bin/env python3
"""
Train LSTM Confidence Model
============================

Trains the multi-output LSTM model for trade confidence scoring.

Outputs three predictions:
1. Direction: Probability of positive return
2. Magnitude: Expected return percentage
3. Success: Probability of winning trade

Usage:
    python scripts/train_lstm_confidence.py \
        --data data/training \
        --output models/lstm_confidence_v1.h5 \
        --epochs 100 \
        --batch-size 32

Requirements:
    - TensorFlow 2.x
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LSTM confidence model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/training",
        help="Training data directory (default: data/training)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/lstm_confidence_v1.h5",
        help="Output model path (default: models/lstm_confidence_v1.h5)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation split (default: 0.2)"
    )
    parser.add_argument(
        "--class-weight",
        action="store_true",
        help="Apply class weighting for imbalanced data"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()


def load_training_data(data_dir: Path) -> tuple:
    """
    Load training data from disk.

    Returns:
        X: 3D feature array (n_samples, lookback, n_features)
        y_direction: Binary direction labels
        y_magnitude: Return magnitude labels
        y_success: Binary success labels
    """
    X = np.load(data_dir / "X_lstm.npy")
    labels = pd.read_parquet(data_dir / "labels.parquet")

    y_direction = labels['direction'].values
    y_magnitude = labels['magnitude'].values
    y_success = labels['won'].values

    logger.info(f"Loaded training data: X={X.shape}, y_direction={len(y_direction)}")
    logger.info(f"Win rate in training data: {y_success.mean()*100:.1f}%")

    return X, y_direction, y_magnitude, y_success


def preprocess_features(X: np.ndarray) -> np.ndarray:
    """
    Preprocess features for LSTM training.

    - Handle NaN values
    - Clip extreme values
    - Normalize to reasonable range
    """
    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip extreme values
    X = np.clip(X, -10, 10)

    # Per-feature standardization (across all samples and timesteps)
    n_samples, lookback, n_features = X.shape
    X_flat = X.reshape(-1, n_features)

    # Compute mean and std per feature
    mean = np.nanmean(X_flat, axis=0)
    std = np.nanstd(X_flat, axis=0)
    std = np.where(std == 0, 1, std)  # Avoid division by zero

    X_normalized = (X_flat - mean) / std
    X_normalized = X_normalized.reshape(n_samples, lookback, n_features)

    logger.info(f"Preprocessed features: min={X_normalized.min():.2f}, max={X_normalized.max():.2f}")

    return X_normalized


def compute_class_weights(y: np.ndarray) -> dict:
    """Compute class weights for imbalanced binary classification."""
    n_positive = y.sum()
    n_negative = len(y) - n_positive
    total = len(y)

    weight_positive = total / (2 * n_positive) if n_positive > 0 else 1.0
    weight_negative = total / (2 * n_negative) if n_negative > 0 else 1.0

    weights = {0: weight_negative, 1: weight_positive}
    logger.info(f"Class weights: {weights}")
    return weights


def train_model(
    X_train: np.ndarray,
    y_direction: np.ndarray,
    y_magnitude: np.ndarray,
    y_success: np.ndarray,
    output_path: Path,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    class_weight: dict = None
):
    """Train the LSTM confidence model."""
    try:
        from ml_advanced.lstm_confidence.model import LSTMConfidenceModel
        from ml_advanced.lstm_confidence.config import LSTMConfig
    except ImportError as e:
        logger.error(f"Failed to import LSTM model: {e}")
        logger.error("Make sure TensorFlow is installed: pip install tensorflow")
        sys.exit(1)

    # Create config based on data dimensions
    n_samples, lookback, n_features = X_train.shape

    config = LSTMConfig(
        lookback_bars=lookback,
        n_features=n_features,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        model_path=str(output_path),
        # Architecture
        lstm_units_1=128,
        lstm_units_2=64,
        dropout_rate=0.3,
        # Learning
        learning_rate=0.001,
        early_stopping_patience=15,
    )

    logger.info("Creating LSTM model with config:")
    logger.info(f"  lookback_bars: {lookback}")
    logger.info(f"  n_features: {n_features}")
    logger.info(f"  lstm_units: [{config.lstm_units_1}, {config.lstm_units_2}]")

    # Create and build model
    model = LSTMConfidenceModel(config)
    model.build()

    # Train
    logger.info(f"Training on {len(X_train)} samples...")

    history = model.train(
        X_train=X_train,
        y_direction=y_direction,
        y_magnitude=y_magnitude,
        y_success=y_success,
        class_weight=class_weight,
    )

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    # Log training results
    final_loss = history.get('loss', [0])[-1]
    final_val_loss = history.get('val_loss', [0])[-1] if 'val_loss' in history else 0

    logger.info("Training complete!")
    logger.info(f"  Final loss: {final_loss:.4f}")
    logger.info(f"  Final val_loss: {final_val_loss:.4f}")
    logger.info(f"  Model saved to: {output_path}")

    return model, history


def evaluate_model(model, X_test, y_direction, y_magnitude, y_success):
    """Evaluate model on test data."""
    results = model.evaluate(X_test, y_direction, y_magnitude, y_success)

    logger.info("Evaluation results:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value:.4f}")

    return results


def save_training_metadata(
    output_dir: Path,
    history: dict,
    data_shape: tuple,
    epochs_trained: int,
    final_metrics: dict
):
    """Save training metadata for reproducibility."""
    metadata = {
        "created_at": datetime.now().isoformat(),
        "model_type": "lstm_confidence",
        "data_shape": list(data_shape),
        "epochs_trained": epochs_trained,
        "final_loss": float(history.get('loss', [0])[-1]),
        "final_val_loss": float(history.get('val_loss', [0])[-1]) if 'val_loss' in history else 0,
        "metrics": {k: float(v) for k, v in final_metrics.items()} if final_metrics else {},
        "config": {
            "lstm_units": [128, 64],
            "dropout": 0.3,
            "learning_rate": 0.001,
        }
    }

    metadata_path = output_dir / "lstm_training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved training metadata to {metadata_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("=" * 60)
    logger.info("TRAIN LSTM CONFIDENCE MODEL")
    logger.info("=" * 60)

    data_dir = Path(args.data)
    output_path = Path(args.output)

    # Check for training data
    if not (data_dir / "X_lstm.npy").exists():
        logger.error(f"Training data not found in {data_dir}")
        logger.error("Run: python scripts/generate_training_data.py first")
        sys.exit(1)

    # Load data
    logger.info("\n[1/4] Loading training data...")
    X, y_direction, y_magnitude, y_success = load_training_data(data_dir)

    # Preprocess
    logger.info("\n[2/4] Preprocessing features...")
    X = preprocess_features(X)

    # Compute class weights if requested
    class_weight = None
    if args.class_weight:
        class_weight = compute_class_weights(y_success)

    # Train
    logger.info("\n[3/4] Training model...")
    try:
        model, history = train_model(
            X_train=X,
            y_direction=y_direction,
            y_magnitude=y_magnitude,
            y_success=y_success,
            output_path=output_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            class_weight=class_weight,
        )
    except ImportError as e:
        logger.error(f"TensorFlow not available: {e}")
        logger.error("Install with: pip install tensorflow")
        sys.exit(1)

    # Quick evaluation
    logger.info("\n[4/4] Evaluating model...")
    eval_size = min(500, len(X))
    results = evaluate_model(
        model,
        X[:eval_size],
        y_direction[:eval_size],
        y_magnitude[:eval_size],
        y_success[:eval_size]
    )

    # Save metadata
    save_training_metadata(
        output_path.parent,
        history,
        X.shape,
        len(history.get('loss', [])),
        results
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("LSTM TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {output_path}")
    logger.info(f"Training samples: {len(X)}")
    logger.info(f"Epochs trained: {len(history.get('loss', []))}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. python scripts/train_ensemble.py --data data/training")
    logger.info("  2. Update config/base.yaml: quality_gate.min_score: 70")
    logger.info("  3. Test: python scripts/scan.py --cap 50 --ml")


if __name__ == "__main__":
    main()
