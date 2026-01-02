"""
Purged K-Fold Cross-Validation for Time Series.

Implements Lopez de Prado's purged cross-validation from
"Advances in Financial Machine Learning" (2018).

Standard K-Fold CV is INVALID for time series because:
1. Data is not IID (observations are dependent)
2. Labels may overlap (trade holding periods)
3. Leakage from future to past folds

Purged K-Fold CV solves these issues by:
1. Respecting temporal order
2. Purging train samples that overlap with test labels
3. Adding an embargo period between train and test

Usage:
    from backtest.purged_cv import PurgedKFold, CombinatorialPurgedKFold

    # Basic usage
    cv = PurgedKFold(n_splits=5, purge_gap=5)
    for train_idx, test_idx in cv.split(X, y, times):
        model.fit(X[train_idx], y[train_idx])
        score = model.score(X[test_idx], y[test_idx])

    # Combinatorial (all path combinations)
    cv = CombinatorialPurgedKFold(n_splits=5, n_test_splits=2)
    for train_idx, test_idx in cv.split(X, y, times):
        ...
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Iterator, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.structured_log import jlog


@dataclass
class PurgedKFoldConfig:
    """Configuration for Purged K-Fold CV."""

    n_splits: int = 5  # Number of folds
    purge_gap: int = 5  # Gap between train and test (in samples)
    embargo_pct: float = 0.01  # Percentage of data to embargo after test
    allow_empty_train: bool = False  # Allow folds with no train data


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation.

    Key differences from standard K-Fold:
    1. Temporal awareness: train always before test
    2. Purging: removes train samples that overlap with test labels
    3. Embargo: adds gap after test to prevent leakage

    This is essential for time series and financial data where
    observations are not independent.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 5,
        embargo_pct: float = 0.01
    ):
        """
        Initialize Purged K-Fold.

        Args:
            n_splits: Number of folds
            purge_gap: Minimum gap between train end and test start
            embargo_pct: Percentage of data to embargo after each test fold
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        times: Optional[pd.Series] = None,
        pred_times: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.

        Args:
            X: Features array/DataFrame
            y: Labels (optional)
            times: Observation timestamps (for purging)
            pred_times: Prediction end times (for label overlap detection)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        if n_samples < self.n_splits:
            raise ValueError(f"Cannot have {self.n_splits} splits with {n_samples} samples")

        # Calculate fold boundaries
        fold_size = n_samples // self.n_splits
        embargo_size = max(1, int(n_samples * self.embargo_pct))

        for i in range(self.n_splits):
            # Test fold boundaries
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            # Train folds: all data before test (with gap) and after test (with embargo)
            train_before_end = max(0, test_start - self.purge_gap)
            train_after_start = min(n_samples, test_end + embargo_size)

            train_indices = np.concatenate([
                np.arange(0, train_before_end),
                np.arange(train_after_start, n_samples)
            ])
            test_indices = np.arange(test_start, test_end)

            # Additional purging based on label overlap
            if times is not None and pred_times is not None:
                train_indices = self._purge_overlapping(
                    train_indices, test_indices, times, pred_times
                )

            if len(train_indices) == 0:
                jlog("purged_cv_empty_fold", level="WARNING",
                     fold=i, test_size=len(test_indices))
                continue

            yield train_indices, test_indices

    def _purge_overlapping(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        times: pd.Series,
        pred_times: pd.Series
    ) -> np.ndarray:
        """
        Remove train samples whose labels overlap with test period.

        When a trade label spans multiple time periods (e.g., holding period),
        we must purge any train samples whose label period overlaps with
        the test period to prevent leakage.

        Args:
            train_indices: Current train indices
            test_indices: Test indices
            times: Observation start times
            pred_times: Observation end times (label spans from time to pred_time)

        Returns:
            Purged train indices
        """
        test_start = times.iloc[test_indices.min()]
        test_end = pred_times.iloc[test_indices.max()]

        # Purge train samples whose prediction window overlaps test period
        purged_train = []
        for idx in train_indices:
            train_start = times.iloc[idx]
            train_end = pred_times.iloc[idx]

            # No overlap if train ends before test starts OR train starts after test ends
            no_overlap = (train_end < test_start) or (train_start > test_end)

            if no_overlap:
                purged_train.append(idx)

        return np.array(purged_train)

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


class TimeSeriesSplit:
    """
    Time Series Split with expanding window.

    Always trains on past data and tests on future data.
    Similar to walk-forward validation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0
    ):
        """
        Initialize Time Series Split.

        Args:
            n_splits: Number of splits
            max_train_size: Maximum training set size (optional)
            test_size: Size of each test set (optional, defaults to n_samples/(n_splits+1))
            gap: Number of samples to skip between train and test
        """
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.

        Args:
            X: Features array
            y: Labels (optional)
            groups: Group labels (optional, for grouped splits)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        # Calculate test size if not specified
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        # Starting point for first split
        start = n_samples - test_size * self.n_splits - self.gap * (self.n_splits - 1)

        for i in range(self.n_splits):
            # Test boundaries
            test_start = start + (test_size + self.gap) * i
            test_end = test_start + test_size

            # Train boundaries
            train_start = 0
            train_end = test_start - self.gap

            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, min(test_end, n_samples))

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged K-Fold Cross-Validation.

    Instead of testing on single folds, tests on combinations of folds.
    This generates more test paths for better strategy evaluation.

    Example with 5 splits, 2 test splits:
    - (Test: 0,1 | Train: 2,3,4)
    - (Test: 0,2 | Train: 1,3,4)
    - (Test: 0,3 | Train: 1,2,4)
    - ... (C(5,2) = 10 combinations)

    Benefits:
    - More robust performance estimation
    - Tests strategy on more out-of-sample paths
    - Better for strategy validation
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_gap: int = 5,
        embargo_pct: float = 0.01
    ):
        """
        Initialize Combinatorial Purged K-Fold.

        Args:
            n_splits: Total number of folds
            n_test_splits: Number of folds to use as test in each iteration
            purge_gap: Gap between train and test
            embargo_pct: Embargo percentage
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        times: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate combinatorial train/test splits.

        Args:
            X: Features array
            y: Labels (optional)
            times: Timestamps (optional)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        from itertools import combinations

        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo_size = max(1, int(n_samples * self.embargo_pct))

        # Calculate fold boundaries
        folds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            folds.append(np.arange(start, end))

        # Generate all combinations of test folds
        for test_fold_ids in combinations(range(self.n_splits), self.n_test_splits):
            # Test indices from selected folds
            test_indices = np.concatenate([folds[i] for i in test_fold_ids])

            # Train indices from remaining folds with purging
            train_folds = [i for i in range(self.n_splits) if i not in test_fold_ids]
            train_indices = []

            for fold_id in train_folds:
                fold_indices = folds[fold_id]

                # Check for adjacency to test folds and apply purge/embargo
                is_before_test = any(fold_id == t - 1 for t in test_fold_ids)
                is_after_test = any(fold_id == t + 1 for t in test_fold_ids)

                if is_before_test:
                    # Purge: remove samples too close to test
                    purge_start = max(fold_indices) - self.purge_gap
                    fold_indices = fold_indices[fold_indices < purge_start]
                elif is_after_test:
                    # Embargo: remove samples too close to test
                    embargo_end = min(fold_indices) + embargo_size
                    fold_indices = fold_indices[fold_indices >= embargo_end]

                if len(fold_indices) > 0:
                    train_indices.extend(fold_indices)

            train_indices = np.array(sorted(train_indices))

            if len(train_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Return total number of combinations."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)


class WalkForwardCV:
    """
    Walk-Forward Cross-Validation.

    Rolling window approach that simulates real trading:
    1. Train on fixed window
    2. Test on next period
    3. Roll forward and repeat

    This is the most realistic validation for trading strategies.
    """

    def __init__(
        self,
        train_size: int = 252,  # 1 year of daily data
        test_size: int = 63,    # 1 quarter
        step_size: Optional[int] = None,  # Step between windows (default = test_size)
        anchored: bool = False  # If True, train always starts at beginning
    ):
        """
        Initialize Walk-Forward CV.

        Args:
            train_size: Size of training window
            test_size: Size of test window
            step_size: Step between consecutive windows
            anchored: If True, use expanding window (anchor at start)
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.anchored = anchored

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward train/test splits.

        Args:
            X: Features array
            y: Labels (optional)
            groups: Group labels (optional)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        # Starting position
        start = 0 if self.anchored else 0

        while True:
            # Calculate current window boundaries
            if self.anchored:
                train_start = 0
                train_end = start + self.train_size
            else:
                train_start = start
                train_end = start + self.train_size

            test_start = train_end
            test_end = test_start + self.test_size

            # Check if we have enough data
            if test_end > n_samples:
                break

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

            # Move forward
            start += self.step_size

    def get_n_splits(self, X: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> int:
        """Return approximate number of splits."""
        if X is None:
            return -1  # Unknown without data size

        n_samples = len(X)
        total_size = self.train_size + self.test_size

        if self.anchored:
            # Expanding window: number of test periods that fit
            return max(0, (n_samples - self.train_size) // self.step_size)
        else:
            # Rolling window
            return max(0, (n_samples - total_size) // self.step_size + 1)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def purged_cross_val_score(
    model,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    cv: Optional[Union[PurgedKFold, TimeSeriesSplit]] = None,
    scoring: str = 'accuracy',
    times: Optional[pd.Series] = None
) -> np.ndarray:
    """
    Cross-validation with purging.

    Similar to sklearn's cross_val_score but with purged CV.

    Args:
        model: Sklearn-compatible model with fit/predict
        X: Features
        y: Labels
        cv: Cross-validator (default: PurgedKFold)
        scoring: Scoring method ('accuracy', 'f1', 'roc_auc', 'neg_mse')
        times: Timestamps for purging

    Returns:
        Array of scores for each fold
    """
    if cv is None:
        cv = PurgedKFold(n_splits=5)

    scores = []

    for train_idx, test_idx in cv.split(X, y, times):
        # Get train/test data
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]

        if isinstance(y, pd.Series):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            y_train, y_test = y[train_idx], y[test_idx]

        # Fit and predict
        model.fit(X_train, y_train)

        # Score
        if scoring == 'accuracy':
            y_pred = model.predict(X_test)
            score = (y_pred == y_test).mean()
        elif scoring == 'f1':
            from sklearn.metrics import f1_score
            y_pred = model.predict(X_test)
            score = f1_score(y_test, y_pred, average='weighted')
        elif scoring == 'roc_auc':
            from sklearn.metrics import roc_auc_score
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.predict(X_test)
            score = roc_auc_score(y_test, y_proba)
        elif scoring == 'neg_mse':
            from sklearn.metrics import mean_squared_error
            y_pred = model.predict(X_test)
            score = -mean_squared_error(y_test, y_pred)
        else:
            raise ValueError(f"Unknown scoring: {scoring}")

        scores.append(score)

    return np.array(scores)


def get_walk_forward_splits(
    n_samples: int,
    train_days: int = 252,
    test_days: int = 63
) -> List[Tuple[range, range]]:
    """
    Get walk-forward split indices.

    Convenience function for simple walk-forward analysis.

    Args:
        n_samples: Total number of samples
        train_days: Training period length
        test_days: Test period length

    Returns:
        List of (train_range, test_range) tuples
    """
    cv = WalkForwardCV(train_size=train_days, test_size=test_days)
    X_dummy = np.zeros((n_samples, 1))

    splits = []
    for train_idx, test_idx in cv.split(X_dummy):
        splits.append((
            range(train_idx.min(), train_idx.max() + 1),
            range(test_idx.min(), test_idx.max() + 1)
        ))

    return splits
