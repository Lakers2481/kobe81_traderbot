#!/usr/bin/env python3
"""
Train HMM Regime Detector
=========================

Trains a Hidden Markov Model for market regime detection using SPY and VIX data.

The model identifies 3 hidden states:
1. Bullish: Trending up, low volatility
2. Neutral: Sideways, moderate volatility
3. Bearish: Trending down, high volatility

Usage:
    python scripts/train_hmm_regime.py \
        --spy-file data/polygon_cache/SPY.csv \
        --vix-file data/polygon_cache/VIX.csv \
        --output models/hmm_regime_v1.pkl \
        --lookback-days 504

Requirements:
    - hmmlearn
    - SPY and VIX historical data
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.structured_log import jlog

logger = logging.getLogger(__name__)

# Check for hmmlearn
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train HMM regime detector")
    parser.add_argument(
        "--spy-file",
        type=str,
        default="data/polygon_cache/SPY.csv",
        help="SPY OHLCV file (default: data/polygon_cache/SPY.csv)"
    )
    parser.add_argument(
        "--vix-file",
        type=str,
        default="data/polygon_cache/VIX.csv",
        help="VIX OHLCV file (default: data/polygon_cache/VIX.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/hmm_regime_v1.pkl",
        help="Output model path (default: models/hmm_regime_v1.pkl)"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=504,
        help="Training lookback in trading days (default: 504 = ~2 years)"
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=3,
        help="Number of hidden states (default: 3)"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=100,
        help="Max EM iterations (default: 100)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()


def load_ohlcv(filepath: Path) -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)

    # Normalize column names
    df.columns = df.columns.str.lower()

    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

    # Ensure we have Close column with capital C for consistency
    if 'close' in df.columns:
        df['Close'] = df['close']

    return df


def prepare_features(spy_data: pd.DataFrame, vix_data: pd.DataFrame, lookback_vol: int = 21) -> pd.DataFrame:
    """
    Prepare observable features for HMM training.

    Features:
    - returns: 10-day SPY momentum
    - volatility: 21-day realized volatility (annualized)
    - vix: VIX level
    - breadth: Price relative to 50-day SMA (proxy for market breadth)
    """
    # Align dates
    spy_close = spy_data['Close']
    vix_close = vix_data['Close']

    common_dates = spy_close.index.intersection(vix_close.index)
    spy_close = spy_close.loc[common_dates]
    vix_close = vix_close.loc[common_dates]

    features = pd.DataFrame(index=common_dates)

    # 10-day momentum (returns)
    features['returns'] = spy_close.pct_change(10) * 100

    # Realized volatility (annualized)
    daily_returns = spy_close.pct_change()
    features['volatility'] = daily_returns.rolling(lookback_vol).std() * np.sqrt(252) * 100

    # VIX level
    features['vix'] = vix_close

    # Market breadth proxy: price relative to 50-day SMA
    sma_50 = spy_close.rolling(50).mean()
    features['breadth'] = (spy_close / sma_50 - 1) * 100

    # Drop NaN
    features = features.dropna()

    logger.info(f"Prepared {len(features)} feature samples")
    logger.info(f"Date range: {features.index[0]} to {features.index[-1]}")

    return features


def train_hmm(
    features: pd.DataFrame,
    n_states: int = 3,
    n_iter: int = 100,
    random_state: int = 42
) -> tuple:
    """
    Train Gaussian HMM on market features.

    Returns:
        model: Trained GaussianHMM
        state_labels: Mapping of state indices to regime names
        training_stats: Dict with training statistics
    """
    if not HMM_AVAILABLE:
        raise ImportError("hmmlearn not installed. pip install hmmlearn")

    logger.info(f"Training {n_states}-state Gaussian HMM...")

    # Standardize features
    feature_means = features.mean()
    feature_stds = features.std()
    X_standardized = (features - feature_means) / feature_stds

    # Create and train model
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type='full',
        n_iter=n_iter,
        random_state=random_state,
        verbose=True
    )

    model.fit(X_standardized.values)

    # Get regime sequence
    regime_sequence = model.predict(X_standardized.values)

    # Label states based on characteristics
    state_labels = label_states(features, regime_sequence, n_states)

    # Calculate log likelihood
    log_likelihood = model.score(X_standardized.values)

    training_stats = {
        'n_samples': len(features),
        'n_states': n_states,
        'log_likelihood': float(log_likelihood),
        'n_iterations': n_iter,
        'feature_means': feature_means.to_dict(),
        'feature_stds': feature_stds.to_dict(),
        'transition_matrix': model.transmat_.tolist(),
        'state_means': model.means_.tolist(),
    }

    logger.info(f"Training complete. Log-likelihood: {log_likelihood:.2f}")

    return model, state_labels, training_stats


def label_states(features: pd.DataFrame, regime_sequence: np.ndarray, n_states: int) -> dict:
    """
    Label HMM states based on feature characteristics.

    States are labeled by their mean return:
    - Highest mean return -> BULLISH
    - Lowest mean return -> BEARISH
    - Middle -> NEUTRAL
    """
    state_characteristics = {}

    for state_id in range(n_states):
        mask = regime_sequence == state_id
        state_features = features[mask]

        if len(state_features) == 0:
            state_characteristics[state_id] = {
                'mean_return': 0.0,
                'mean_vol': 0.0,
                'mean_vix': 0.0,
                'n_samples': 0
            }
        else:
            state_characteristics[state_id] = {
                'mean_return': state_features['returns'].mean(),
                'mean_vol': state_features['volatility'].mean(),
                'mean_vix': state_features['vix'].mean(),
                'n_samples': len(state_features)
            }

    # Sort by mean return
    sorted_states = sorted(
        state_characteristics.items(),
        key=lambda x: x[1]['mean_return']
    )

    state_labels = {}
    if len(sorted_states) >= 3:
        state_labels[sorted_states[0][0]] = 'BEARISH'
        state_labels[sorted_states[1][0]] = 'NEUTRAL'
        state_labels[sorted_states[2][0]] = 'BULLISH'
    else:
        for i, (state_id, _) in enumerate(sorted_states):
            if i == 0:
                state_labels[state_id] = 'BEARISH'
            elif i == len(sorted_states) - 1:
                state_labels[state_id] = 'BULLISH'
            else:
                state_labels[state_id] = 'NEUTRAL'

    logger.info("State labels assigned:")
    for state_id, label in state_labels.items():
        stats = state_characteristics[state_id]
        logger.info(f"  State {state_id} -> {label}: "
                   f"mean_return={stats['mean_return']:.2f}%, "
                   f"mean_vix={stats['mean_vix']:.1f}, "
                   f"n_samples={stats['n_samples']}")

    return state_labels


def analyze_regimes(features: pd.DataFrame, model, state_labels: dict) -> pd.DataFrame:
    """
    Analyze regime distribution and transitions.
    """
    # Standardize features
    feature_means = features.mean()
    feature_stds = features.std()
    X_standardized = (features - feature_means) / feature_stds

    # Get regime sequence
    regime_sequence = model.predict(X_standardized.values)

    # Count regimes
    regime_df = pd.DataFrame({
        'date': features.index,
        'state_id': regime_sequence,
        'regime': [state_labels[s] for s in regime_sequence]
    })

    regime_counts = regime_df['regime'].value_counts()
    logger.info("\nRegime distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(regime_df) * 100
        logger.info(f"  {regime}: {count} days ({pct:.1f}%)")

    # Analyze transitions
    transition_matrix = model.transmat_
    logger.info("\nTransition probabilities:")
    for i, from_label in enumerate(['BEARISH', 'NEUTRAL', 'BULLISH']):
        probs = []
        for j, to_label in enumerate(['BEARISH', 'NEUTRAL', 'BULLISH']):
            # Find state ids for labels
            from_id = [k for k, v in state_labels.items() if v == from_label][0] if from_label in state_labels.values() else i
            to_id = [k for k, v in state_labels.items() if v == to_label][0] if to_label in state_labels.values() else j
            prob = transition_matrix[from_id, to_id]
            probs.append(f"{to_label}={prob:.2f}")
        logger.info(f"  {from_label} -> {', '.join(probs)}")

    return regime_df


def save_model(
    output_path: Path,
    model,
    state_labels: dict,
    feature_means: dict,
    feature_stds: dict,
    training_stats: dict
):
    """Save trained model to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'model': model,
        'state_labels': state_labels,
        'feature_names': ['returns', 'volatility', 'vix', 'breadth'],
        'feature_means': feature_means,
        'feature_stds': feature_stds,
        'n_states': model.n_components,
        'is_fitted': True,
        'last_train_timestamp': datetime.now(),
        'training_window_days': training_stats['n_samples'],
        'staleness_threshold_days': 30,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"Saved model to {output_path}")

    # Also save metadata as JSON (for inspection)
    metadata_path = output_path.parent / "hmm_regime_metadata.json"
    metadata = {
        "created_at": datetime.now().isoformat(),
        "n_states": model.n_components,
        "state_labels": {str(k): v for k, v in state_labels.items()},
        "training_stats": {
            k: v for k, v in training_stats.items()
            if k not in ['state_means', 'transition_matrix']
        },
        "feature_names": ['returns', 'volatility', 'vix', 'breadth'],
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("=" * 60)
    logger.info("TRAIN HMM REGIME DETECTOR")
    logger.info("=" * 60)

    if not HMM_AVAILABLE:
        logger.error("hmmlearn not installed!")
        logger.error("Install with: pip install hmmlearn")
        sys.exit(1)

    spy_path = Path(args.spy_file)
    vix_path = Path(args.vix_file)
    output_path = Path(args.output)

    # Load data
    logger.info("\n[1/4] Loading market data...")

    try:
        spy_data = load_ohlcv(spy_path)
        logger.info(f"Loaded SPY: {len(spy_data)} bars")
    except FileNotFoundError:
        logger.error(f"SPY data not found: {spy_path}")
        logger.error("Fetch with: python scripts/prefetch_polygon_universe.py")
        sys.exit(1)

    try:
        vix_data = load_ohlcv(vix_path)
        logger.info(f"Loaded VIX: {len(vix_data)} bars")
    except FileNotFoundError:
        logger.warning(f"VIX data not found: {vix_path}")
        logger.warning("Will estimate VIX from SPY realized volatility")
        # Create synthetic VIX from realized vol
        returns = spy_data['Close'].pct_change().dropna()
        realized_vol = returns.rolling(21).std() * np.sqrt(252) * 100
        vix_data = pd.DataFrame({'Close': realized_vol * 0.9 + 2})
        vix_data.index = realized_vol.index

    # Limit to lookback window
    if args.lookback_days > 0:
        spy_data = spy_data.tail(args.lookback_days)
        vix_data = vix_data.tail(args.lookback_days)
        logger.info(f"Limited to last {args.lookback_days} days")

    # Prepare features
    logger.info("\n[2/4] Preparing features...")
    features = prepare_features(spy_data, vix_data)

    if len(features) < 100:
        logger.error(f"Insufficient data: {len(features)} samples (need 100+)")
        sys.exit(1)

    # Train model
    logger.info("\n[3/4] Training HMM...")
    model, state_labels, training_stats = train_hmm(
        features,
        n_states=args.n_states,
        n_iter=args.n_iter
    )

    # Analyze regimes
    logger.info("\n[4/4] Analyzing regimes...")
    analyze_regimes(features, model, state_labels)

    # Save model
    save_model(
        output_path,
        model,
        state_labels,
        features.mean().to_dict(),
        features.std().to_dict(),
        training_stats
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("HMM TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {output_path}")
    logger.info(f"Training samples: {len(features)}")
    logger.info(f"Log-likelihood: {training_stats['log_likelihood']:.2f}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Update config/base.yaml: quality_gate.min_score: 70")
    logger.info("  2. Test regime detection: python -c \"from ml_advanced.hmm_regime_detector import HMMRegimeDetector; d = HMMRegimeDetector(); d.load_model('models/hmm_regime_v1.pkl')\"")
    logger.info("  3. Run scanner with ML: python scripts/scan.py --cap 50 --ml")


if __name__ == "__main__":
    main()
