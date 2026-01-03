"""
Temporal Fusion Transformer (TFT) for Financial Forecasting.

Implements TFT architecture from the paper:
"Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
(Lim et al., 2019)

TFT is state-of-the-art for time series forecasting with:
- Multi-horizon forecasting
- Interpretable attention mechanisms
- Handling of static and time-varying features
- Variable selection network
- Quantile predictions for uncertainty

Key advantages for trading:
- Interpretable: shows which features drive predictions
- Multi-horizon: can predict returns at multiple timeframes
- Uncertainty: provides confidence intervals via quantiles
- Handles mixed inputs: static (sector) + time-varying (price, volume)

Usage:
    from ml_advanced.tft.temporal_fusion import TFTForecaster

    forecaster = TFTForecaster()
    forecaster.fit(train_df)
    predictions = forecaster.predict(test_df)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.structured_log import jlog

# Check for PyTorch Forecasting availability
try:
    import torch
    from pytorch_forecasting import (
        TemporalFusionTransformer,
        TimeSeriesDataSet,
        GroupNormalizer,
        QuantileLoss
    )
    from pytorch_forecasting.data import NaNLabelEncoder
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False
    jlog("tft_not_available", level="INFO",
         message="Install: pip install pytorch-forecasting pytorch-lightning")


@dataclass
class TFTConfig:
    """Configuration for Temporal Fusion Transformer."""

    # Model architecture
    hidden_size: int = 64  # Size of hidden layers
    attention_head_size: int = 4  # Attention heads
    dropout: float = 0.1  # Dropout rate
    hidden_continuous_size: int = 32  # Size for continuous variables
    lstm_layers: int = 2  # Number of LSTM layers

    # Data configuration
    max_encoder_length: int = 60  # Lookback window (days)
    max_prediction_length: int = 5  # Forecast horizon (days)
    min_encoder_length: int = 30  # Minimum lookback

    # Training
    batch_size: int = 64
    max_epochs: int = 50
    learning_rate: float = 0.001
    early_stop_patience: int = 5
    gradient_clip_val: float = 0.1

    # Features
    time_varying_known_reals: List[str] = field(
        default_factory=lambda: ['day_of_week', 'month', 'is_month_end']
    )
    time_varying_unknown_reals: List[str] = field(
        default_factory=lambda: [
            'close', 'open', 'high', 'low', 'volume',
            'return_1d', 'return_5d', 'volatility_20d',
            'rsi_14', 'atr_14'
        ]
    )
    static_categoricals: List[str] = field(
        default_factory=lambda: ['symbol']
    )
    static_reals: List[str] = field(
        default_factory=lambda: []
    )

    # Targets
    target: str = 'return_5d_forward'  # Target variable
    quantiles: List[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9]
    )

    # GPU
    use_gpu: bool = True


class TFTForecaster:
    """
    Temporal Fusion Transformer forecaster for trading.

    Provides multi-horizon return predictions with:
    - Quantile forecasts (uncertainty)
    - Feature importance
    - Attention weights (interpretability)
    """

    def __init__(self, config: Optional[TFTConfig] = None):
        self.config = config or TFTConfig()
        self.model = None
        self.training_dataset = None
        self.trainer = None
        self._is_fitted = False

    def _prepare_data(
        self,
        df: pd.DataFrame,
        training: bool = True
    ) -> TimeSeriesDataSet:
        """
        Prepare data for TFT.

        Args:
            df: DataFrame with OHLCV and features
            training: Whether this is training data

        Returns:
            TimeSeriesDataSet for TFT
        """
        if not TFT_AVAILABLE:
            raise ImportError("pytorch-forecasting not installed")

        df = df.copy()
        df.columns = df.columns.str.lower()

        # Ensure required columns
        if 'time_idx' not in df.columns:
            df['time_idx'] = range(len(df))

        if 'symbol' not in df.columns:
            df['symbol'] = 'STOCK'

        # Add target if not present
        if self.config.target not in df.columns:
            if 'close' in df.columns:
                # Create forward return
                forward_days = int(self.config.target.split('_')[1].replace('d', ''))
                df[self.config.target] = df['close'].pct_change(forward_days).shift(-forward_days)

        # Fill missing feature columns with defaults
        for col in self.config.time_varying_unknown_reals:
            if col not in df.columns:
                if col == 'return_1d':
                    df[col] = df['close'].pct_change()
                elif col == 'return_5d':
                    df[col] = df['close'].pct_change(5)
                elif col == 'volatility_20d':
                    df[col] = df['close'].pct_change().rolling(20).std()
                elif col == 'rsi_14':
                    df[col] = 50.0  # Placeholder
                elif col == 'atr_14':
                    df[col] = (df['high'] - df['low']).rolling(14).mean()
                else:
                    df[col] = 0.0

        for col in self.config.time_varying_known_reals:
            if col not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    if col == 'day_of_week':
                        df[col] = df.index.dayofweek
                    elif col == 'month':
                        df[col] = df.index.month
                    elif col == 'is_month_end':
                        df[col] = df.index.is_month_end.astype(float)
                else:
                    df[col] = 0

        # Drop rows with NaN target
        df = df.dropna(subset=[self.config.target])

        # Ensure numeric types
        for col in df.columns:
            if df[col].dtype == 'object':
                if col not in self.config.static_categoricals:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Create dataset
        if training:
            dataset = TimeSeriesDataSet(
                df,
                time_idx='time_idx',
                target=self.config.target,
                group_ids=['symbol'],
                max_encoder_length=self.config.max_encoder_length,
                max_prediction_length=self.config.max_prediction_length,
                min_encoder_length=self.config.min_encoder_length,
                time_varying_known_reals=self.config.time_varying_known_reals,
                time_varying_unknown_reals=[
                    c for c in self.config.time_varying_unknown_reals if c in df.columns
                ],
                static_categoricals=self.config.static_categoricals,
                static_reals=[c for c in self.config.static_reals if c in df.columns],
                target_normalizer=GroupNormalizer(groups=['symbol']),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True
            )
            self.training_dataset = dataset
        else:
            if self.training_dataset is None:
                raise ValueError("Must fit before predicting")
            dataset = TimeSeriesDataSet.from_dataset(
                self.training_dataset,
                df,
                predict=True,
                stop_randomization=True
            )

        return dataset

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        val_split: float = 0.2
    ) -> 'TFTForecaster':
        """
        Fit the TFT model.

        Args:
            train_df: Training DataFrame
            val_df: Optional validation DataFrame
            val_split: Fraction for validation if val_df not provided

        Returns:
            Self
        """
        if not TFT_AVAILABLE:
            jlog("tft_fit_skipped", level="WARNING",
                 message="pytorch-forecasting not installed")
            return self

        jlog("tft_fitting", level="INFO",
             samples=len(train_df),
             epochs=self.config.max_epochs)

        # Split validation if not provided
        if val_df is None:
            split_idx = int(len(train_df) * (1 - val_split))
            val_df = train_df.iloc[split_idx:]
            train_df = train_df.iloc[:split_idx]

        # Prepare datasets
        training_dataset = self._prepare_data(train_df, training=True)
        validation_dataset = self._prepare_data(val_df, training=False)

        # Create dataloaders
        train_dataloader = training_dataset.to_dataloader(
            train=True,
            batch_size=self.config.batch_size,
            num_workers=0
        )
        val_dataloader = validation_dataset.to_dataloader(
            train=False,
            batch_size=self.config.batch_size,
            num_workers=0
        )

        # Initialize model
        self.model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=self.config.learning_rate,
            hidden_size=self.config.hidden_size,
            attention_head_size=self.config.attention_head_size,
            dropout=self.config.dropout,
            hidden_continuous_size=self.config.hidden_continuous_size,
            lstm_layers=self.config.lstm_layers,
            output_size=len(self.config.quantiles),
            loss=QuantileLoss(quantiles=self.config.quantiles),
            reduce_on_plateau_patience=3
        )

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stop_patience,
                mode='min'
            ),
            LearningRateMonitor()
        ]

        # Trainer
        accelerator = 'gpu' if self.config.use_gpu and torch.cuda.is_available() else 'cpu'

        self.trainer = Trainer(
            max_epochs=self.config.max_epochs,
            accelerator=accelerator,
            gradient_clip_val=self.config.gradient_clip_val,
            callbacks=callbacks,
            enable_progress_bar=True,
            logger=False
        )

        # Train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.trainer.fit(
                self.model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )

        self._is_fitted = True

        jlog("tft_fitted", level="INFO",
             best_val_loss=self.trainer.callback_metrics.get('val_loss', 'N/A'))

        return self

    def predict(
        self,
        df: pd.DataFrame,
        return_attention: bool = False
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Generate predictions.

        Args:
            df: DataFrame to predict on
            return_attention: Whether to return attention weights

        Returns:
            Dict with predictions and optionally attention weights
        """
        if not TFT_AVAILABLE or not self._is_fitted:
            jlog("tft_predict_skipped", level="WARNING")
            return {'predictions': np.array([]), 'quantiles': {}}

        # Prepare data
        dataset = self._prepare_data(df, training=False)
        dataloader = dataset.to_dataloader(
            train=False,
            batch_size=self.config.batch_size,
            num_workers=0
        )

        # Predict
        predictions = self.model.predict(
            dataloader,
            return_x=True,
            return_index=True
        )

        result = {
            'predictions': predictions.output.numpy(),
            'quantiles': {
                q: predictions.output[:, :, i].numpy()
                for i, q in enumerate(self.config.quantiles)
            }
        }

        # Get attention if requested
        if return_attention:
            interpretation = self.model.interpret_output(
                predictions.output,
                reduction="sum"
            )
            result['attention'] = interpretation

        return result

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the model.

        Returns:
            DataFrame with feature importances
        """
        if not self._is_fitted:
            return pd.DataFrame()

        try:
            interpretation = self.model.interpret_output(
                self.model.predict(
                    self.training_dataset.to_dataloader(
                        train=False, batch_size=self.config.batch_size
                    )
                ),
                reduction="sum"
            )

            # Variable selection weights
            encoder_weights = interpretation.get('encoder_variables', {})
            decoder_weights = interpretation.get('decoder_variables', {})

            importance = {}
            for var, weight in encoder_weights.items():
                importance[f"encoder_{var}"] = float(weight.mean())
            for var, weight in decoder_weights.items():
                importance[f"decoder_{var}"] = float(weight.mean())

            return pd.DataFrame(
                list(importance.items()),
                columns=['feature', 'importance']
            ).sort_values('importance', ascending=False)

        except Exception as e:
            jlog("tft_importance_error", level="WARNING", error=str(e))
            return pd.DataFrame()

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        if self.model is not None:
            torch.save({
                'model_state': self.model.state_dict(),
                'config': self.config
            }, path)
            jlog("tft_saved", level="INFO", path=str(path))

    def load(self, path: Union[str, Path]) -> 'TFTForecaster':
        """Load model from disk."""
        if TFT_AVAILABLE:
            checkpoint = torch.load(path)
            self.config = checkpoint['config']
            # Model would need to be recreated with training dataset
            jlog("tft_loaded", level="INFO", path=str(path))
        return self

    def visualize_attention(
        self,
        df: pd.DataFrame,
        output_dir: Union[str, Path] = "reports/tft_attention",
        symbol: Optional[str] = None,
        save_format: str = "png"
    ) -> Optional[Path]:
        """
        Visualize TFT attention weights as heatmap.

        Shows which time steps and features the model focuses on when making
        predictions. Essential for interpretability and understanding model behavior.

        Args:
            df: DataFrame with features (will use last sample)
            output_dir: Directory to save attention plots
            symbol: Optional symbol name for filename
            save_format: Image format ('png' or 'html')

        Returns:
            Path to saved visualization or None if failed
        """
        if not TFT_AVAILABLE or not self._is_fitted:
            jlog("tft_viz_skipped", level="WARNING",
                 message="TFT not fitted or pytorch-forecasting not available")
            return None

        try:
            import matplotlib.pyplot as plt
            import matplotlib

            # Use non-interactive backend for server/headless environments
            matplotlib.use('Agg')

            # Get predictions with attention
            predictions = self.predict(df, return_attention=True)

            if 'attention' not in predictions:
                jlog("tft_viz_no_attention", level="WARNING")
                return None

            attention = predictions['attention']

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol_str = f"_{symbol}" if symbol else ""
            filename = f"tft_attention{symbol_str}_{timestamp}.{save_format}"
            filepath = output_path / filename

            # Create attention heatmap
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'TFT Attention Analysis{" - " + symbol if symbol else ""}',
                        fontsize=14, fontweight='bold')

            # Plot 1: Encoder variable importance
            if 'encoder_variables' in attention:
                ax = axes[0, 0]
                enc_vars = attention['encoder_variables']
                if hasattr(enc_vars, 'mean'):
                    enc_importance = {k: float(v.mean()) for k, v in enc_vars.items()}
                    sorted_vars = sorted(enc_importance.items(), key=lambda x: x[1], reverse=True)
                    names = [x[0] for x in sorted_vars[:10]]
                    values = [x[1] for x in sorted_vars[:10]]
                    ax.barh(names, values, color='steelblue')
                    ax.set_xlabel('Importance')
                    ax.set_title('Encoder Variable Importance (Top 10)')
                    ax.invert_yaxis()

            # Plot 2: Decoder variable importance
            if 'decoder_variables' in attention:
                ax = axes[0, 1]
                dec_vars = attention['decoder_variables']
                if hasattr(dec_vars, 'mean'):
                    dec_importance = {k: float(v.mean()) for k, v in dec_vars.items()}
                    sorted_vars = sorted(dec_importance.items(), key=lambda x: x[1], reverse=True)
                    names = [x[0] for x in sorted_vars[:10]]
                    values = [x[1] for x in sorted_vars[:10]]
                    ax.barh(names, values, color='darkorange')
                    ax.set_xlabel('Importance')
                    ax.set_title('Decoder Variable Importance (Top 10)')
                    ax.invert_yaxis()

            # Plot 3: Static variable importance
            if 'static_variables' in attention:
                ax = axes[1, 0]
                static_vars = attention['static_variables']
                if hasattr(static_vars, 'mean'):
                    static_importance = {k: float(v.mean()) for k, v in static_vars.items()}
                    sorted_vars = sorted(static_importance.items(), key=lambda x: x[1], reverse=True)
                    names = [x[0] for x in sorted_vars[:5]]
                    values = [x[1] for x in sorted_vars[:5]]
                    ax.barh(names, values, color='forestgreen')
                    ax.set_xlabel('Importance')
                    ax.set_title('Static Variable Importance')
                    ax.invert_yaxis()
            else:
                axes[1, 0].text(0.5, 0.5, 'No static variables',
                               ha='center', va='center', fontsize=12)
                axes[1, 0].set_title('Static Variable Importance')

            # Plot 4: Attention over time
            if 'attention' in attention:
                ax = axes[1, 1]
                attn_weights = attention['attention']
                if hasattr(attn_weights, 'numpy'):
                    attn_matrix = attn_weights.mean(dim=0).numpy() if len(attn_weights.shape) > 2 else attn_weights.numpy()
                    if len(attn_matrix.shape) == 2:
                        im = ax.imshow(attn_matrix, aspect='auto', cmap='viridis')
                        ax.set_xlabel('Query Position')
                        ax.set_ylabel('Key Position')
                        ax.set_title('Temporal Attention Pattern')
                        plt.colorbar(im, ax=ax, label='Attention Weight')
            else:
                axes[1, 1].text(0.5, 0.5, 'Temporal attention not available',
                               ha='center', va='center', fontsize=12)
                axes[1, 1].set_title('Temporal Attention Pattern')

            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)

            jlog("tft_attention_saved", level="INFO", path=str(filepath))
            return filepath

        except Exception as e:
            jlog("tft_viz_error", level="WARNING", error=str(e))
            return None


class TFTSignalGenerator:
    """
    Generate trading signals from TFT predictions.

    Uses quantile predictions to:
    - Generate directional signals
    - Assess prediction confidence
    - Size positions based on uncertainty
    """

    def __init__(
        self,
        forecaster: TFTForecaster,
        long_threshold: float = 0.01,  # Expected return for long
        short_threshold: float = -0.01,  # Expected return for short
        confidence_threshold: float = 0.6  # Minimum confidence
    ):
        self.forecaster = forecaster
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.confidence_threshold = confidence_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from TFT predictions.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with signal columns
        """
        predictions = self.forecaster.predict(df)

        if len(predictions['predictions']) == 0:
            return pd.DataFrame()

        # Get median prediction (0.5 quantile)
        median_pred = predictions['quantiles'].get(0.5, predictions['predictions'][:, :, 2])

        # Get uncertainty (IQR)
        q25 = predictions['quantiles'].get(0.25, predictions['predictions'][:, :, 1])
        q75 = predictions['quantiles'].get(0.75, predictions['predictions'][:, :, 3])
        uncertainty = q75 - q25

        # Generate signals
        signals = pd.DataFrame(index=range(len(median_pred)))
        signals['tft_prediction'] = median_pred[:, 0] if median_pred.ndim > 1 else median_pred
        signals['tft_uncertainty'] = uncertainty[:, 0] if uncertainty.ndim > 1 else uncertainty

        # Confidence: inverse of uncertainty, normalized
        max_uncertainty = signals['tft_uncertainty'].max()
        if max_uncertainty > 0:
            signals['tft_confidence'] = 1 - (signals['tft_uncertainty'] / max_uncertainty)
        else:
            signals['tft_confidence'] = 0.5

        # Directional signals
        signals['tft_signal'] = 0
        signals.loc[
            (signals['tft_prediction'] > self.long_threshold) &
            (signals['tft_confidence'] > self.confidence_threshold),
            'tft_signal'
        ] = 1
        signals.loc[
            (signals['tft_prediction'] < self.short_threshold) &
            (signals['tft_confidence'] > self.confidence_threshold),
            'tft_signal'
        ] = -1

        return signals


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_tft_forecaster(
    hidden_size: int = 64,
    max_epochs: int = 50,
    use_gpu: bool = True
) -> TFTForecaster:
    """
    Create TFT forecaster with common settings.

    Args:
        hidden_size: Model hidden size
        max_epochs: Maximum training epochs
        use_gpu: Whether to use GPU

    Returns:
        TFTForecaster instance
    """
    config = TFTConfig(
        hidden_size=hidden_size,
        max_epochs=max_epochs,
        use_gpu=use_gpu
    )
    return TFTForecaster(config)


def train_and_predict_tft(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = 'return_5d_forward'
) -> Dict[str, Any]:
    """
    Train TFT and generate predictions.

    Convenience function for quick experimentation.

    Args:
        train_df: Training data
        test_df: Test data
        target: Target variable name

    Returns:
        Dict with model, predictions, and feature importance
    """
    config = TFTConfig(target=target)
    forecaster = TFTForecaster(config)

    forecaster.fit(train_df)
    predictions = forecaster.predict(test_df)
    importance = forecaster.get_feature_importance()

    return {
        'forecaster': forecaster,
        'predictions': predictions,
        'feature_importance': importance
    }
