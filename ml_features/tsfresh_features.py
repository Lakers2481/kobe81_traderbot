"""
tsfresh Automated Feature Extraction.

Wraps the tsfresh library for automated extraction of 787+ time series features.
tsfresh systematically extracts features based on statistical hypothesis tests.

Features include:
- Statistical moments (mean, std, skew, kurtosis)
- Entropy measures (sample entropy, permutation entropy)
- Autocorrelation functions
- Fourier transform coefficients
- Wavelet coefficients
- Change point detection features
- Non-linear dynamics (Lyapunov exponent approximations)
- And many more...

Source: https://tsfresh.readthedocs.io/
Paper: "tsfresh - A Python package for automatic extraction of relevant features from time series"

Usage:
    from ml_features.tsfresh_features import TSFreshExtractor

    extractor = TSFreshExtractor()
    features = extractor.extract(df)  # 787+ features per symbol
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Dict, Union, Set
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.structured_log import jlog

# Check for tsfresh availability
try:
    from tsfresh import extract_features, select_features
    from tsfresh.feature_extraction import (
        ComprehensiveFCParameters,
        MinimalFCParameters,
        EfficientFCParameters
    )
    from tsfresh.utilities.dataframe_functions import impute
    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False
    jlog("tsfresh_not_available", level="WARNING",
         message="Install tsfresh: pip install tsfresh")


@dataclass
class TSFreshConfig:
    """Configuration for tsfresh feature extraction."""

    # Feature extraction mode
    mode: str = "efficient"  # "minimal", "efficient", "comprehensive"

    # Columns to extract features from
    feature_columns: List[str] = field(default_factory=lambda: ['close', 'volume', 'high', 'low'])

    # Rolling window for feature extraction
    window_size: int = 20  # Extract features over rolling window

    # Performance
    n_jobs: int = 4  # Parallel jobs for extraction
    disable_progressbar: bool = True
    show_warnings: bool = False

    # Feature selection
    select_relevant: bool = False  # Use tsfresh's feature selection
    fdr_level: float = 0.05  # False discovery rate for selection

    # Caching
    cache_features: bool = True
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "cache" / "tsfresh")


class TSFreshExtractor:
    """
    Automated time series feature extraction using tsfresh.

    Extracts hundreds of features from time series data:
    - Statistical features (mean, std, quantiles, etc.)
    - Autocorrelation and partial autocorrelation
    - Entropy measures
    - FFT coefficients
    - Linear trend features
    - Non-linear features
    - Change quantile features
    - Many more...
    """

    def __init__(self, config: Optional[TSFreshConfig] = None):
        self.config = config or TSFreshConfig()
        self._feature_cache: Dict[str, pd.DataFrame] = {}

        # Create cache directory
        if self.config.cache_features:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_extraction_settings(self):
        """Get tsfresh extraction settings based on mode."""
        if not TSFRESH_AVAILABLE:
            return None

        if self.config.mode == "minimal":
            return MinimalFCParameters()
        elif self.config.mode == "efficient":
            return EfficientFCParameters()
        elif self.config.mode == "comprehensive":
            return ComprehensiveFCParameters()
        else:
            return EfficientFCParameters()

    def _prepare_data_for_tsfresh(
        self,
        df: pd.DataFrame,
        symbol: str = "SYMBOL"
    ) -> pd.DataFrame:
        """
        Prepare DataFrame for tsfresh format.

        tsfresh expects data in long format:
        - id: unique identifier for each time series
        - time: time index
        - value columns: values to extract features from
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Reset index to get time column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'time'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'time'})
        elif 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'time'})
        else:
            df['time'] = range(len(df))

        # Add id column
        df['id'] = symbol

        # Convert time to numeric for tsfresh
        if pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = (df['time'] - df['time'].min()).dt.total_seconds()

        return df

    def extract(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract tsfresh features from DataFrame.

        Args:
            df: DataFrame with OHLCV data
            symbol: Optional symbol for identification
            columns: Optional list of columns to extract from

        Returns:
            DataFrame with extracted features
        """
        if not TSFRESH_AVAILABLE:
            jlog("tsfresh_extraction_skipped", level="WARNING",
                 message="tsfresh not installed")
            return df

        symbol = symbol or "SYMBOL"
        columns = columns or self.config.feature_columns

        # Check cache
        cache_key = f"{symbol}_{len(df)}_{self.config.mode}"
        if self.config.cache_features and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        df_orig = df.copy()
        df_orig.columns = df_orig.columns.str.lower()

        # Filter to available columns
        available_cols = [c for c in columns if c in df_orig.columns]
        if not available_cols:
            jlog("tsfresh_no_columns", level="WARNING",
                 requested=columns, available=list(df_orig.columns))
            return df_orig

        # Prepare data
        df_tsfresh = self._prepare_data_for_tsfresh(df_orig[available_cols + ['close']], symbol)

        # Suppress warnings if configured
        if not self.config.show_warnings:
            warnings.filterwarnings("ignore")

        try:
            # Extract features
            settings = self._get_extraction_settings()

            # For rolling window extraction, we need to create multiple "ids"
            if self.config.window_size > 0:
                features_list = []
                window = self.config.window_size

                for i in range(window, len(df_tsfresh)):
                    window_df = df_tsfresh.iloc[i-window:i].copy()
                    window_df['id'] = f"{symbol}_{i}"
                    features_list.append(window_df)

                if features_list:
                    stacked_df = pd.concat(features_list, ignore_index=True)

                    extracted = extract_features(
                        stacked_df,
                        column_id='id',
                        column_sort='time',
                        default_fc_parameters=settings,
                        n_jobs=self.config.n_jobs,
                        disable_progressbar=self.config.disable_progressbar
                    )

                    # Impute missing values
                    extracted = impute(extracted)

                    # Get last row for each original index
                    # Map back to original indices
                    result_indices = [int(idx.split('_')[-1]) for idx in extracted.index]
                    extracted.index = result_indices
                    extracted = extracted.sort_index()

                    # Align with original DataFrame
                    features = pd.DataFrame(index=df_orig.index)
                    for col in extracted.columns:
                        features[f'tsf_{col}'] = np.nan
                        for idx in extracted.index:
                            if idx < len(features):
                                features.iloc[idx][f'tsf_{col}'] = extracted.loc[idx, col]

                else:
                    features = pd.DataFrame(index=df_orig.index)
            else:
                # Single extraction for entire series
                extracted = extract_features(
                    df_tsfresh,
                    column_id='id',
                    column_sort='time',
                    default_fc_parameters=settings,
                    n_jobs=self.config.n_jobs,
                    disable_progressbar=self.config.disable_progressbar
                )
                extracted = impute(extracted)

                # Broadcast to all rows (same features for entire series)
                features = pd.DataFrame(
                    np.tile(extracted.values, (len(df_orig), 1)),
                    columns=[f'tsf_{c}' for c in extracted.columns],
                    index=df_orig.index
                )

            # Merge with original
            result = pd.concat([df_orig, features], axis=1)

            # Cache result
            if self.config.cache_features:
                self._feature_cache[cache_key] = result

            jlog("tsfresh_extracted", level="DEBUG",
                 symbol=symbol,
                 n_features=len(features.columns),
                 mode=self.config.mode)

            return result

        except Exception as e:
            jlog("tsfresh_extraction_error", level="ERROR", error=str(e))
            return df_orig

        finally:
            if not self.config.show_warnings:
                warnings.filterwarnings("default")

    def extract_minimal(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Extract minimal feature set (fastest, ~10 features per column).
        """
        original_mode = self.config.mode
        self.config.mode = "minimal"
        result = self.extract(df, symbol)
        self.config.mode = original_mode
        return result

    def extract_efficient(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Extract efficient feature set (balanced, ~100 features per column).
        """
        original_mode = self.config.mode
        self.config.mode = "efficient"
        result = self.extract(df, symbol)
        self.config.mode = original_mode
        return result

    def extract_comprehensive(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Extract comprehensive feature set (slowest, ~800 features per column).
        """
        original_mode = self.config.mode
        self.config.mode = "comprehensive"
        result = self.extract(df, symbol)
        self.config.mode = original_mode
        return result

    def select_relevant_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fdr_level: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Select statistically relevant features using tsfresh's built-in selection.

        Uses hypothesis testing to identify features with significant
        relationship to the target variable.

        Args:
            X: Feature DataFrame
            y: Target variable
            fdr_level: False discovery rate level (default from config)

        Returns:
            DataFrame with only relevant features
        """
        if not TSFRESH_AVAILABLE:
            return X

        fdr_level = fdr_level or self.config.fdr_level

        try:
            # Get only tsfresh columns
            tsf_cols = [c for c in X.columns if c.startswith('tsf_')]
            if not tsf_cols:
                return X

            X_tsf = X[tsf_cols].copy()

            # Select relevant features
            X_selected = select_features(X_tsf, y, fdr_level=fdr_level)

            # Combine with non-tsfresh columns
            non_tsf_cols = [c for c in X.columns if not c.startswith('tsf_')]
            result = pd.concat([X[non_tsf_cols], X_selected], axis=1)

            jlog("tsfresh_feature_selection", level="DEBUG",
                 original_features=len(tsf_cols),
                 selected_features=len(X_selected.columns),
                 fdr_level=fdr_level)

            return result

        except Exception as e:
            jlog("tsfresh_selection_error", level="WARNING", error=str(e))
            return X

    def clear_cache(self) -> None:
        """Clear feature cache."""
        self._feature_cache.clear()
        jlog("tsfresh_cache_cleared", level="DEBUG")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_global_extractor: Optional[TSFreshExtractor] = None

def extract_tsfresh_features(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    mode: str = "efficient"
) -> pd.DataFrame:
    """
    Extract tsfresh features with default settings.

    Args:
        df: DataFrame with OHLCV data
        symbol: Optional symbol identifier
        mode: Extraction mode ("minimal", "efficient", "comprehensive")

    Returns:
        DataFrame with tsfresh features added
    """
    global _global_extractor

    if _global_extractor is None:
        _global_extractor = TSFreshExtractor(TSFreshConfig(mode=mode))
    else:
        _global_extractor.config.mode = mode

    return _global_extractor.extract(df, symbol)


def get_tsfresh_feature_names(mode: str = "efficient") -> List[str]:
    """
    Get list of feature names that would be extracted.

    Args:
        mode: Extraction mode

    Returns:
        List of feature names
    """
    if not TSFRESH_AVAILABLE:
        return []

    if mode == "minimal":
        settings = MinimalFCParameters()
    elif mode == "efficient":
        settings = EfficientFCParameters()
    else:
        settings = ComprehensiveFCParameters()

    # Get feature names from settings
    feature_names = []
    for func_name, params_list in settings.items():
        if params_list is None:
            feature_names.append(func_name)
        else:
            for params in params_list:
                param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
                feature_names.append(f"{func_name}__{param_str}")

    return feature_names
