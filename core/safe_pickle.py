"""
Safe Pickle/Joblib Loading Utilities.

SECURITY FIX (2026-01-04): Pickle deserialization can execute arbitrary code.
This module provides safe loading functions that validate paths before loading.

Usage:
    from core.safe_pickle import safe_load, safe_pickle_load, safe_joblib_load

    # Load with path validation
    model = safe_joblib_load(Path("models/deployed/my_model.pkl"))
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional, Set

logger = logging.getLogger(__name__)

# Allowed directories for model/data loading
# Models should only be loaded from these trusted directories
ALLOWED_MODEL_DIRS: Set[Path] = {
    Path("models").resolve(),
    Path("ml_meta").resolve(),
    Path("ml_features").resolve(),
    Path("ml_advanced").resolve(),
    Path("state").resolve(),
    Path("backtest").resolve(),
}


class UnsafePathError(Exception):
    """Raised when attempting to load from an unsafe path."""
    pass


def is_safe_path(path: Path, allowed_dirs: Optional[Set[Path]] = None) -> bool:
    """
    Check if a path is within allowed directories.

    Args:
        path: The path to validate
        allowed_dirs: Set of allowed directories (defaults to ALLOWED_MODEL_DIRS)

    Returns:
        True if path is within an allowed directory
    """
    if allowed_dirs is None:
        allowed_dirs = ALLOWED_MODEL_DIRS

    resolved = path.resolve()

    # Check if path is within any allowed directory
    for allowed in allowed_dirs:
        try:
            # Python 3.9+ has is_relative_to
            if hasattr(resolved, 'is_relative_to'):
                if resolved.is_relative_to(allowed):
                    return True
            else:
                # Fallback for older Python
                try:
                    resolved.relative_to(allowed)
                    return True
                except ValueError:
                    pass
        except Exception:
            pass

    return False


def safe_pickle_load(path: Path, allowed_dirs: Optional[Set[Path]] = None) -> Any:
    """
    Safely load a pickle file after path validation.

    Args:
        path: Path to pickle file
        allowed_dirs: Set of allowed directories

    Returns:
        Loaded object

    Raises:
        UnsafePathError: If path is not in allowed directories
        FileNotFoundError: If file doesn't exist
    """
    if not is_safe_path(path, allowed_dirs):
        logger.critical(f"SECURITY: Blocked unsafe pickle load from {path}")
        raise UnsafePathError(f"Cannot load pickle from unsafe path: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    with open(path, 'rb') as f:
        return pickle.load(f)


def safe_joblib_load(path: Path, allowed_dirs: Optional[Set[Path]] = None) -> Any:
    """
    Safely load a joblib file after path validation.

    Args:
        path: Path to joblib file
        allowed_dirs: Set of allowed directories

    Returns:
        Loaded object

    Raises:
        UnsafePathError: If path is not in allowed directories
        FileNotFoundError: If file doesn't exist
    """
    if not is_safe_path(path, allowed_dirs):
        logger.critical(f"SECURITY: Blocked unsafe joblib load from {path}")
        raise UnsafePathError(f"Cannot load joblib from unsafe path: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Joblib file not found: {path}")

    import joblib
    return joblib.load(path)


def safe_load(path: Path, allowed_dirs: Optional[Set[Path]] = None) -> Any:
    """
    Auto-detect and safely load pickle or joblib file.

    Args:
        path: Path to file
        allowed_dirs: Set of allowed directories

    Returns:
        Loaded object
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in ('.joblib', '.pkl', '.pickle'):
        # Try joblib first (handles sklearn models better)
        try:
            return safe_joblib_load(path, allowed_dirs)
        except ImportError:
            return safe_pickle_load(path, allowed_dirs)

    return safe_pickle_load(path, allowed_dirs)
