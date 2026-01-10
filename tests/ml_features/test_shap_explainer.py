"""
Comprehensive Unit Tests for SHAP Explainer Module
===================================================

Tests the SHAP integration for explaining ML model predictions
in the trading system.

Run: python -m pytest tests/ml_features/test_shap_explainer.py -v
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch


class TestFeatureContribution:
    """Tests for FeatureContribution dataclass."""

    def test_feature_contribution_creation(self):
        """Test basic FeatureContribution instantiation."""
        from ml_features.shap_explainer import FeatureContribution

        contrib = FeatureContribution(
            feature_name="rsi_14",
            feature_value=25.5,
            shap_value=0.15,
            impact="positive",
            rank=1,
        )

        assert contrib.feature_name == "rsi_14"
        assert contrib.feature_value == 25.5
        assert contrib.shap_value == 0.15
        assert contrib.impact == "positive"
        assert contrib.rank == 1

    def test_feature_contribution_serialization(self):
        """Test contribution to_dict serialization."""
        from ml_features.shap_explainer import FeatureContribution

        contrib = FeatureContribution(
            feature_name="macd_signal",
            feature_value=-0.0234,
            shap_value=-0.08,
            impact="negative",
            rank=2,
        )

        data = contrib.to_dict()

        assert data['feature'] == "macd_signal"
        assert 'value' in data
        assert data['shap_value'] == -0.08
        assert data['impact'] == "negative"


class TestSHAPExplanation:
    """Tests for SHAPExplanation dataclass."""

    def test_explanation_creation(self):
        """Test SHAPExplanation instantiation."""
        from ml_features.shap_explainer import SHAPExplanation, ModelType

        explanation = SHAPExplanation(
            shap_values=np.array([0.1, -0.05, 0.2, -0.1]),
            base_value=0.5,
            feature_names=["rsi", "macd", "volume", "price"],
            feature_values=np.array([30, -0.02, 1500000, 150.5]),
            prediction=0.65,
            model_type=ModelType.TREE,
        )

        assert len(explanation.shap_values) == 4
        assert explanation.base_value == 0.5
        assert explanation.prediction == 0.65
        assert explanation.model_type == ModelType.TREE

    def test_get_top_features(self):
        """Test extraction of top-k features by importance."""
        from ml_features.shap_explainer import SHAPExplanation, ModelType

        explanation = SHAPExplanation(
            shap_values=np.array([0.1, -0.3, 0.05, 0.2, -0.15]),
            base_value=0.5,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            feature_values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            prediction=0.4,
        )

        top_3 = explanation.get_top_features(k=3)

        assert len(top_3) == 3
        # f2 has highest absolute SHAP value (-0.3)
        assert top_3[0].feature_name == "f2"
        assert top_3[0].rank == 1

    def test_get_positive_contributors(self):
        """Test extraction of positive contributors."""
        from ml_features.shap_explainer import SHAPExplanation

        explanation = SHAPExplanation(
            shap_values=np.array([0.2, -0.1, 0.15, -0.05]),
            base_value=0.5,
            feature_names=["f1", "f2", "f3", "f4"],
            feature_values=np.array([1, 2, 3, 4]),
            prediction=0.7,
        )

        positive = explanation.get_positive_contributors()

        # Should return f1 (0.2) and f3 (0.15)
        assert len(positive) == 2
        assert all(c.shap_value > 0 for c in positive)
        assert all(c.impact == "positive" for c in positive)

    def test_get_negative_contributors(self):
        """Test extraction of negative contributors."""
        from ml_features.shap_explainer import SHAPExplanation

        explanation = SHAPExplanation(
            shap_values=np.array([0.1, -0.2, 0.05, -0.15]),
            base_value=0.5,
            feature_names=["f1", "f2", "f3", "f4"],
            feature_values=np.array([1, 2, 3, 4]),
            prediction=0.3,
        )

        negative = explanation.get_negative_contributors()

        # Should return f2 (-0.2) and f4 (-0.15)
        assert len(negative) == 2
        assert all(c.shap_value < 0 for c in negative)
        assert all(c.impact == "negative" for c in negative)

    def test_to_narrative(self):
        """Test human-readable narrative generation."""
        from ml_features.shap_explainer import SHAPExplanation

        explanation = SHAPExplanation(
            shap_values=np.array([0.15, -0.1, 0.2]),
            base_value=0.5,
            feature_names=["rsi", "volume", "momentum"],
            feature_values=np.array([28.5, 1.5, 0.8]),
            prediction=0.75,
        )

        narrative = explanation.to_narrative()

        assert "Prediction:" in narrative
        assert "0.75" in narrative or "0.7500" in narrative
        assert "base" in narrative.lower()
        assert "momentum" in narrative  # Top feature

    def test_to_dict_serialization(self):
        """Test explanation serialization."""
        from ml_features.shap_explainer import SHAPExplanation

        explanation = SHAPExplanation(
            shap_values=np.array([0.1, -0.1]),
            base_value=0.5,
            feature_names=["f1", "f2"],
            feature_values=np.array([1.0, 2.0]),
            prediction=0.5,
        )

        data = explanation.to_dict()

        assert 'prediction' in data
        assert 'base_value' in data
        assert 'top_features' in data
        assert 'model_type' in data


class TestModelType:
    """Tests for ModelType enum."""

    def test_model_types(self):
        """Test available model types."""
        from ml_features.shap_explainer import ModelType

        assert ModelType.TREE.value == "tree"
        assert ModelType.DEEP.value == "deep"
        assert ModelType.KERNEL.value == "kernel"
        assert ModelType.LINEAR.value == "linear"


class TestSHAPExplainer:
    """Tests for the main SHAPExplainer class."""

    def test_initialization(self):
        """Test explainer initialization."""
        from ml_features.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer(cache_explainers=True)

        assert explainer._cache_explainers is True
        assert explainer._explainer_cache == {}
        assert explainer._last_explanation is None

    def test_initialization_no_cache(self):
        """Test explainer with caching disabled."""
        from ml_features.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer(cache_explainers=False)

        assert explainer._cache_explainers is False

    def test_explain_prediction_mock_fallback(self):
        """Test mock explanation when SHAP unavailable."""
        from ml_features.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer()

        # Create a mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.7]))

        X = pd.DataFrame({
            'rsi': [30.0],
            'macd': [-0.02],
            'volume': [1500000],
        })

        # Even without SHAP, should return mock explanation
        explanation = explainer.explain_prediction(mock_model, X, "tree")

        assert explanation is not None
        assert explanation.prediction == 0.7
        assert len(explanation.feature_names) == 3

    def test_explain_prediction_with_numpy(self):
        """Test explanation with numpy array input."""
        from ml_features.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer()

        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.6]))

        X = np.array([[30.0, -0.02, 1500000]])

        explanation = explainer.explain_prediction(mock_model, X, "tree")

        assert explanation is not None
        # Feature names should be auto-generated
        assert "feature_0" in explanation.feature_names

    def test_get_latest_explanation(self):
        """Test retrieval of last explanation."""
        from ml_features.shap_explainer import SHAPExplainer, SHAPExplanation, FeatureContribution

        explainer = SHAPExplainer()

        # Initially None
        assert explainer.get_latest_explanation() is None

        # Manually set _last_explanation to test the getter
        # (We can't rely on mock models working with SHAP internals)
        from ml_features.shap_explainer import ModelType

        test_explanation = SHAPExplanation(
            shap_values=np.array([0.2]),
            base_value=0.5,
            feature_names=["f1"],
            feature_values=np.array([1.0]),
            prediction=0.7,
            model_type=ModelType.TREE
        )
        explainer._last_explanation = test_explanation

        assert explainer.get_latest_explanation() is test_explanation

    def test_clear_cache(self):
        """Test cache clearing."""
        from ml_features.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer(cache_explainers=True)

        # Manually add something to cache
        explainer._explainer_cache['test_key'] = Mock()

        assert len(explainer._explainer_cache) == 1

        explainer.clear_cache()

        assert len(explainer._explainer_cache) == 0

    def test_infer_model_type_tree(self):
        """Test model type inference for tree models."""
        from ml_features.shap_explainer import SHAPExplainer, ModelType

        explainer = SHAPExplainer()

        # Mock XGBoost model
        mock_xgb = Mock()
        mock_xgb.__class__.__name__ = "XGBClassifier"

        assert explainer._infer_model_type(mock_xgb) == ModelType.TREE

    def test_infer_model_type_deep(self):
        """Test model type inference for deep learning."""
        from ml_features.shap_explainer import SHAPExplainer, ModelType

        explainer = SHAPExplainer()

        mock_lstm = Mock()
        mock_lstm.__class__.__name__ = "LSTMModel"

        assert explainer._infer_model_type(mock_lstm) == ModelType.DEEP

    def test_infer_model_type_linear(self):
        """Test model type inference for linear models."""
        from ml_features.shap_explainer import SHAPExplainer, ModelType

        explainer = SHAPExplainer()

        mock_linear = Mock()
        mock_linear.__class__.__name__ = "LogisticRegression"

        assert explainer._infer_model_type(mock_linear) == ModelType.LINEAR


class TestExplainEnsemble:
    """Tests for ensemble explanation."""

    def test_explain_ensemble_basic(self):
        """Test ensemble explanation with multiple models."""
        from ml_features.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer()

        # Create mock models
        mock_model1 = Mock()
        mock_model1.__class__.__name__ = "XGBClassifier"
        mock_model1.predict = Mock(return_value=np.array([0.7]))

        mock_model2 = Mock()
        mock_model2.__class__.__name__ = "LGBMClassifier"
        mock_model2.predict = Mock(return_value=np.array([0.6]))

        models = {
            "xgboost": mock_model1,
            "lightgbm": mock_model2,
        }

        X = pd.DataFrame({'f1': [1.0], 'f2': [2.0]})

        aggregated, individual = explainer.explain_ensemble(models, X)

        assert aggregated is not None
        assert len(individual) == 2
        assert "xgboost" in individual
        assert "lightgbm" in individual

    def test_explain_ensemble_with_weights(self):
        """Test ensemble explanation with custom weights."""
        from ml_features.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer()

        mock_model1 = Mock()
        mock_model1.__class__.__name__ = "XGBClassifier"
        mock_model1.predict = Mock(return_value=np.array([0.8]))

        mock_model2 = Mock()
        mock_model2.__class__.__name__ = "RandomForest"
        mock_model2.predict = Mock(return_value=np.array([0.4]))

        models = {
            "model1": mock_model1,
            "model2": mock_model2,
        }
        weights = {"model1": 0.7, "model2": 0.3}

        X = pd.DataFrame({'f1': [1.0]})

        aggregated, _ = explainer.explain_ensemble(models, X, model_weights=weights)

        # Weighted prediction should be closer to model1
        # 0.7 * 0.8 + 0.3 * 0.4 = 0.68
        assert aggregated.prediction == pytest.approx(0.68, rel=0.1)


class TestSingletonPattern:
    """Tests for singleton accessor."""

    def test_get_shap_explainer_returns_instance(self):
        """Test singleton accessor."""
        from ml_features.shap_explainer import get_shap_explainer

        exp1 = get_shap_explainer()
        exp2 = get_shap_explainer()

        assert exp1 is exp2

    def test_explain_prediction_convenience_function(self):
        """Test the module-level convenience function."""
        from ml_features.shap_explainer import explain_prediction

        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.55]))

        X = pd.DataFrame({'feature': [1.0]})

        explanation = explain_prediction(mock_model, X)

        assert explanation is not None
        assert explanation.prediction == 0.55


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_feature(self):
        """Test explanation with single feature."""
        from ml_features.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer()

        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.5]))

        X = pd.DataFrame({'only_feature': [42.0]})

        explanation = explainer.explain_prediction(mock_model, X)

        assert len(explanation.feature_names) == 1
        assert explanation.feature_names[0] == 'only_feature'

    def test_many_features(self):
        """Test explanation with many features."""
        from ml_features.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer()

        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.6]))

        # 50 features
        X = pd.DataFrame({f'f{i}': [float(i)] for i in range(50)})

        explanation = explainer.explain_prediction(mock_model, X)

        assert len(explanation.feature_names) == 50
        # Top 10 should be extractable
        top_10 = explanation.get_top_features(k=10)
        assert len(top_10) == 10

    def test_batch_input(self):
        """Test explanation with batch input (multiple rows)."""
        from ml_features.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer()

        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.5, 0.6, 0.7]))

        X = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0],
            'f2': [4.0, 5.0, 6.0],
        })

        # Should use mean for batch
        explanation = explainer.explain_prediction(mock_model, X)

        assert explanation is not None
