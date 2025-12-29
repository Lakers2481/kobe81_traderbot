"""
Feature Discovery Engine - Find predictive features using SHAP and permutation importance.
"""

from .importance_analyzer import (
    FeatureImportanceReport,
    FeatureImportanceAnalyzer,
)

__all__ = [
    'FeatureImportanceReport',
    'FeatureImportanceAnalyzer',
]
