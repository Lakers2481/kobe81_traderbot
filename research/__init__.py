"""
Kobe Trading System - Research Module
======================================

Quant-interview-grade alpha research infrastructure.

Components:
- features.py: Feature extraction pipeline
- alphas.py: Alpha library (classic + custom)
- screener.py: Automated alpha screening with walk-forward

This module enables systematic hypothesis generation, testing,
and rejection - the scientific method applied to trading.
"""

from .features import (
    FeatureExtractor,
    extract_features,
    FEATURE_REGISTRY,
)

from .alphas import (
    Alpha,
    AlphaLibrary,
    get_alpha_library,
)

from .screener import (
    AlphaScreener,
    ScreenerResult,
    run_alpha_screen,
)

__all__ = [
    # Features
    'FeatureExtractor',
    'extract_features',
    'FEATURE_REGISTRY',
    # Alphas
    'Alpha',
    'AlphaLibrary',
    'get_alpha_library',
    # Screener
    'AlphaScreener',
    'ScreenerResult',
    'run_alpha_screen',
]
