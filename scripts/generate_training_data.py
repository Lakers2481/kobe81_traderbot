#!/usr/bin/env python3
"""
Generate Training Data for ML Models
=====================================

Extracts features and labels from walk-forward backtest results
to create training datasets for LSTM, XGBoost, LightGBM, and HMM models.

Inputs:
    - wf_outputs/*/split_*/trade_list.csv: Trade history from walk-forward
    - data/polygon_cache/*.csv: Historical OHLCV data for feature extraction

Outputs:
    - data/training/features.parquet: Feature matrix
    - data/training/labels.parquet: Trade outcome labels (win/loss, return)
    - data/training/metadata.json: Dataset statistics

Usage:
    python scripts/generate_training_data.py \
        --wf-dir wf_outputs \
        --cache-dir data/polygon_cache \
        --output data/training \
        --min-trades 100
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_features.feature_pipeline import FeaturePipeline, FeatureConfig

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate ML training data from backtest results")
    parser.add_argument(
        "--wf-dir",
        type=str,
        default="wf_outputs",
        help="Walk-forward output directory (default: wf_outputs)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/polygon_cache",
        help="OHLCV data cache directory (default: data/polygon_cache)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training",
        help="Output directory for training data (default: data/training)"
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=50,
        help="Minimum trades required for valid dataset (default: 50)"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=20,
        help="Lookback bars for LSTM sequences (default: 20)"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="ibs,rsi2,and,ibs_rsi,turtle_soup",
        help="Comma-separated strategies to include (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()


def load_trade_lists(wf_dir: Path, strategies: List[str]) -> pd.DataFrame:
    """
    Load all trade lists from walk-forward outputs.

    Returns DataFrame with columns:
        timestamp, symbol, side, qty, price, strategy, split
    """
    all_trades = []

    for strategy in strategies:
        strategy_dir = wf_dir / strategy
        if not strategy_dir.exists():
            logger.debug(f"Strategy directory not found: {strategy_dir}")
            continue

        for split_dir in sorted(strategy_dir.glob("split_*")):
            trade_file = split_dir / "trade_list.csv"
            if not trade_file.exists():
                continue

            try:
                df = pd.read_csv(trade_file)
                if df.empty or len(df) == 0:
                    continue

                df['strategy'] = strategy
                df['split'] = split_dir.name
                all_trades.append(df)

            except Exception as e:
                logger.warning(f"Failed to load {trade_file}: {e}")

    if not all_trades:
        return pd.DataFrame()

    trades = pd.concat(all_trades, ignore_index=True)
    trades['timestamp'] = pd.to_datetime(trades['timestamp'])

    logger.info(f"Loaded {len(trades)} trade rows from {len(all_trades)} files")
    return trades


def pair_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Pair BUY and SELL trades to compute P&L.

    Returns DataFrame with:
        entry_timestamp, exit_timestamp, symbol, strategy, split,
        entry_price, exit_price, qty, pnl, won
    """
    paired = []

    # Group by symbol, strategy, split for correct pairing
    for (symbol, strategy, split), group in trades.groupby(['symbol', 'strategy', 'split']):
        group = group.sort_values('timestamp')

        # Simple pairing: BUY followed by SELL
        buys = group[group['side'] == 'BUY'].reset_index(drop=True)
        sells = group[group['side'] == 'SELL'].reset_index(drop=True)

        # Match by order (first buy with first sell, etc.)
        n_pairs = min(len(buys), len(sells))

        for i in range(n_pairs):
            buy = buys.iloc[i]
            sell = sells.iloc[i]

            # Skip if sell before buy (shouldn't happen in valid data)
            if sell['timestamp'] <= buy['timestamp']:
                continue

            entry_price = buy['price']
            exit_price = sell['price']
            qty = buy['qty']

            pnl = (exit_price - entry_price) * qty
            pnl_pct = (exit_price - entry_price) / entry_price
            won = pnl > 0

            paired.append({
                'entry_timestamp': buy['timestamp'],
                'exit_timestamp': sell['timestamp'],
                'symbol': symbol,
                'strategy': strategy,
                'split': split,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'qty': qty,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'won': won,
                'holding_days': (sell['timestamp'] - buy['timestamp']).days
            })

    if not paired:
        return pd.DataFrame()

    result = pd.DataFrame(paired)
    logger.info(f"Paired {len(result)} trades: {result['won'].sum()} wins, {(~result['won']).sum()} losses")
    return result


def load_ohlcv_data(cache_dir: Path, symbol: str) -> Optional[pd.DataFrame]:
    """Load OHLCV data for a symbol from cache."""
    # Try multiple filename patterns
    patterns = [
        f"{symbol.upper()}.csv",
        f"{symbol.lower()}.csv",
        f"{symbol}.csv",
    ]

    for pattern in patterns:
        filepath = cache_dir / pattern
        if filepath.exists():
            try:
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

                return df
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

    return None


def extract_features_at_entry(
    trades: pd.DataFrame,
    cache_dir: Path,
    lookback: int = 20,
    feature_pipeline: Optional[FeaturePipeline] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extract features at trade entry timestamps.

    For LSTM: Returns 3D array (n_samples, lookback, n_features)
    For tree models: Returns 2D array (n_samples, n_features)

    Returns:
        X_lstm: 3D feature array for LSTM
        X_flat: 2D feature DataFrame for tree models
    """
    if feature_pipeline is None:
        feature_pipeline = FeaturePipeline(FeatureConfig(
            shift_features=False,  # Don't shift - we're extracting at specific timestamps
        ))

    X_sequences = []
    X_flat_rows = []
    valid_indices = []

    # Track expected feature count for consistency
    expected_n_features = None
    expected_feature_cols = None

    symbols_loaded = {}  # Cache loaded symbols

    for idx, trade in trades.iterrows():
        symbol = trade['symbol']
        entry_ts = trade['entry_timestamp']

        # Load symbol data if not cached
        if symbol not in symbols_loaded:
            ohlcv = load_ohlcv_data(cache_dir, symbol)
            if ohlcv is None:
                logger.debug(f"No data for {symbol}")
                continue
            symbols_loaded[symbol] = ohlcv

        ohlcv = symbols_loaded[symbol]

        # Find entry date in data
        try:
            # Find the bar at or before entry timestamp
            mask = ohlcv.index <= entry_ts
            if not mask.any():
                continue

            entry_idx = ohlcv[mask].index[-1]
            loc = ohlcv.index.get_loc(entry_idx)

            # Need at least lookback bars before entry
            if loc < lookback:
                continue

            # Extract sequence for LSTM
            sequence_data = ohlcv.iloc[loc - lookback:loc + 1].copy()

            # Extract features
            features_df = feature_pipeline.extract(sequence_data, symbol)

            if len(features_df) < lookback:
                continue

            # Get feature columns (exclude OHLCV)
            feature_cols = [c for c in features_df.columns
                          if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'date']]

            if not feature_cols:
                continue

            # Ensure consistent feature count across all samples
            if expected_n_features is None:
                expected_n_features = len(feature_cols)
                expected_feature_cols = feature_cols
                logger.info(f"First sample: {expected_n_features} features")
            elif len(feature_cols) != expected_n_features:
                # Use intersection of features
                feature_cols = [c for c in feature_cols if c in expected_feature_cols]
                if len(feature_cols) != expected_n_features:
                    # Skip samples with mismatched features
                    continue

            # LSTM sequence: last lookback rows
            sequence = features_df[expected_feature_cols].iloc[-lookback:].values

            # Verify shape
            if sequence.shape != (lookback, expected_n_features):
                continue

            # Flat features: last row only (for tree models)
            flat_row = features_df[expected_feature_cols].iloc[-1].to_dict()
            flat_row['symbol'] = symbol
            flat_row['entry_timestamp'] = entry_ts

            X_sequences.append(sequence)
            X_flat_rows.append(flat_row)
            valid_indices.append(idx)

        except Exception as e:
            logger.debug(f"Feature extraction failed for {symbol} at {entry_ts}: {e}")
            continue

    if not X_sequences:
        return np.array([]), pd.DataFrame(), []

    X_lstm = np.stack(X_sequences, axis=0)
    X_flat = pd.DataFrame(X_flat_rows)

    logger.info(f"Extracted features for {len(valid_indices)} trades")
    logger.info(f"LSTM shape: {X_lstm.shape}")
    logger.info(f"Flat features: {len(X_flat.columns)} columns")

    return X_lstm, X_flat, valid_indices


def create_labels(trades: pd.DataFrame, valid_indices: List[int]) -> pd.DataFrame:
    """
    Create label DataFrame for valid trades.

    Columns:
        won: Binary (1=win, 0=loss)
        pnl_pct: Return percentage
        direction: 1 for positive, 0 for negative
        magnitude: Absolute return
    """
    valid_trades = trades.loc[valid_indices].copy()

    labels = pd.DataFrame({
        'won': valid_trades['won'].astype(int),
        'direction': (valid_trades['pnl_pct'] > 0).astype(int),
        'magnitude': valid_trades['pnl_pct'].abs(),
        'pnl_pct': valid_trades['pnl_pct'],
        'holding_days': valid_trades['holding_days'],
        'strategy': valid_trades['strategy'],
        'symbol': valid_trades['symbol'],
    })

    logger.info(f"Created labels: {labels['won'].sum()} wins ({labels['won'].mean()*100:.1f}%)")
    return labels


def save_training_data(
    X_lstm: np.ndarray,
    X_flat: pd.DataFrame,
    labels: pd.DataFrame,
    output_dir: Path,
    feature_names: List[str]
):
    """Save training data to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save LSTM features as numpy
    np.save(output_dir / "X_lstm.npy", X_lstm)
    logger.info(f"Saved LSTM features: {X_lstm.shape}")

    # Save flat features as parquet
    X_flat.to_parquet(output_dir / "features.parquet")
    logger.info(f"Saved flat features: {X_flat.shape}")

    # Save labels
    labels.to_parquet(output_dir / "labels.parquet")
    logger.info(f"Saved labels: {labels.shape}")

    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "n_samples": len(labels),
        "n_features_flat": len(X_flat.columns),
        "n_features_lstm": X_lstm.shape[2] if len(X_lstm.shape) == 3 else 0,
        "lookback": X_lstm.shape[1] if len(X_lstm.shape) == 3 else 0,
        "feature_names": feature_names,
        "win_rate": float(labels['won'].mean()),
        "strategies": labels['strategy'].unique().tolist(),
        "symbols": labels['symbol'].unique().tolist()[:50],  # Top 50
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {output_dir / 'metadata.json'}")


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("=" * 60)
    logger.info("GENERATE TRAINING DATA")
    logger.info("=" * 60)

    wf_dir = Path(args.wf_dir)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output)
    strategies = [s.strip() for s in args.strategies.split(",")]

    logger.info(f"Walk-forward dir: {wf_dir}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Strategies: {strategies}")

    # Step 1: Load trade lists
    logger.info("\n[1/4] Loading trade lists...")
    trades = load_trade_lists(wf_dir, strategies)

    if trades.empty:
        logger.error("No trades found in walk-forward outputs")
        logger.error("Run walk-forward backtest first: python scripts/run_wf_polygon.py")
        sys.exit(1)

    # Step 2: Pair trades to compute P&L
    logger.info("\n[2/4] Pairing trades...")
    paired_trades = pair_trades(trades)

    if len(paired_trades) < args.min_trades:
        logger.error(f"Only {len(paired_trades)} paired trades (need {args.min_trades})")
        logger.error("Run more walk-forward splits or relax --min-trades")
        sys.exit(1)

    # Step 3: Extract features at entry timestamps
    logger.info("\n[3/4] Extracting features...")
    feature_pipeline = FeaturePipeline(FeatureConfig(
        shift_features=False,
    ))

    X_lstm, X_flat, valid_indices = extract_features_at_entry(
        paired_trades,
        cache_dir,
        lookback=args.lookback,
        feature_pipeline=feature_pipeline
    )

    if len(valid_indices) < args.min_trades:
        logger.error(f"Only {len(valid_indices)} valid samples (need {args.min_trades})")
        logger.error("Check that cache_dir contains OHLCV data for traded symbols")
        sys.exit(1)

    # Step 4: Create labels
    logger.info("\n[4/4] Creating labels and saving...")
    labels = create_labels(paired_trades, valid_indices)

    # Get feature names
    feature_cols = [c for c in X_flat.columns
                   if c not in ['symbol', 'entry_timestamp']]

    # Save everything
    save_training_data(X_lstm, X_flat, labels, output_dir, feature_cols)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING DATA GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(labels)}")
    logger.info(f"Win rate: {labels['won'].mean()*100:.1f}%")
    logger.info(f"LSTM features shape: {X_lstm.shape}")
    logger.info(f"Flat features: {len(feature_cols)} columns")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. python scripts/train_lstm_confidence.py --data data/training")
    logger.info("  2. python scripts/train_ensemble.py --data data/training")
    logger.info("  3. python scripts/train_hmm_regime.py")


if __name__ == "__main__":
    main()
