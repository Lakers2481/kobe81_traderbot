"""
Tests for Data Exploration Module.

Tests feature importance, data registry, and feature discovery.
"""
import pytest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from data_exploration import (
    # Feature Importance
    FeatureImportance,
    ImportanceResult,
    ImportanceMethod,
    analyze_importance,
    get_top_features,
    # Data Registry
    DataRegistry,
    DataSource,
    FeatureDefinition,
    register_source,
    get_available_features,
    # Feature Discovery
    FeatureDiscovery,
    DiscoveredFeature,
    discover_features,
    suggest_features,
)

from data_exploration.data_registry import FeatureCategory, DataFrequency
from data_exploration.feature_discovery import FeatureTransform


class TestFeatureImportance:
    """Tests for FeatureImportance."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 200

        X = pd.DataFrame({
            'feature_a': np.random.randn(n),
            'feature_b': np.random.randn(n),
            'feature_c': np.random.randn(n),
        })

        # Make y correlated with feature_a
        y = X['feature_a'] * 0.5 + np.random.randn(n) * 0.1

        return X, pd.Series(y, name='target')

    def test_initialization(self):
        """Should initialize with defaults."""
        analyzer = FeatureImportance()
        assert analyzer.default_method == ImportanceMethod.CORRELATION
        assert analyzer.n_permutations == 100

    def test_correlation_importance(self, sample_data):
        """Should calculate correlation importance."""
        X, y = sample_data
        analyzer = FeatureImportance()

        results = analyzer.analyze(X, y, ImportanceMethod.CORRELATION)

        assert len(results) == 3
        # feature_a should be most important
        assert results[0].feature_name == 'feature_a'
        assert results[0].rank == 1
        assert results[0].importance > 0.5

    def test_mutual_info_importance(self, sample_data):
        """Should calculate mutual information importance."""
        X, y = sample_data
        analyzer = FeatureImportance()

        results = analyzer.analyze(X, y, ImportanceMethod.MUTUAL_INFO)

        assert len(results) == 3
        # All should have non-negative importance
        assert all(r.importance >= 0 for r in results)

    def test_permutation_importance(self, sample_data):
        """Should calculate permutation importance."""
        X, y = sample_data
        analyzer = FeatureImportance(n_permutations=5)  # Reduce for speed

        results = analyzer.analyze(X, y, ImportanceMethod.PERMUTATION)

        assert len(results) == 3
        # Should have std_error
        assert results[0].std_error is not None

    def test_get_top_n(self, sample_data):
        """Should get top N features."""
        X, y = sample_data
        analyzer = FeatureImportance()

        results = analyzer.analyze(X, y)
        top = analyzer.get_top_n(results, n=2)

        assert len(top) == 2
        assert top[0].rank == 1

    def test_to_dataframe(self, sample_data):
        """Should convert to DataFrame."""
        X, y = sample_data
        analyzer = FeatureImportance()

        results = analyzer.analyze(X, y)
        df = analyzer.to_dataframe(results)

        assert isinstance(df, pd.DataFrame)
        assert 'feature_name' in df.columns
        assert 'importance' in df.columns
        assert len(df) == 3

    def test_compare_methods(self, sample_data):
        """Should compare multiple methods."""
        X, y = sample_data
        analyzer = FeatureImportance()

        comparison = analyzer.compare_methods(X, y)

        assert 'correlation' in comparison
        assert 'mutual_info' in comparison

    def test_result_to_dict(self):
        """Should convert result to dictionary."""
        result = ImportanceResult(
            feature_name='test_feature',
            importance=0.75,
            rank=1,
            method=ImportanceMethod.CORRELATION,
        )

        d = result.to_dict()

        assert d['feature_name'] == 'test_feature'
        assert d['importance'] == 0.75
        assert d['method'] == 'correlation'


class TestDataRegistry:
    """Tests for DataRegistry."""

    @pytest.fixture
    def registry(self):
        """Create registry with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = DataRegistry(
                registry_path=Path(tmpdir) / 'registry.json',
                auto_save=False,
            )
            yield registry

    def test_initialization(self, registry):
        """Should initialize empty."""
        assert len(registry._sources) == 0
        assert len(registry._features) == 0

    def test_register_source(self, registry):
        """Should register data source."""
        source = DataSource(
            name='polygon',
            provider='Polygon.io',
            features=['open', 'high', 'low', 'close', 'volume'],
        )

        registry.register_source(source)

        assert 'polygon' in registry._sources
        assert registry.get_source('polygon') is not None

    def test_register_feature(self, registry):
        """Should register feature."""
        feature = FeatureDefinition(
            name='rsi_14',
            category=FeatureCategory.MOMENTUM,
            description='14-period RSI',
            lookback=14,
        )

        registry.register_feature(feature)

        assert 'rsi_14' in registry._features
        assert registry.get_feature('rsi_14') is not None

    def test_list_sources(self, registry):
        """Should list all sources."""
        registry.register_source(DataSource(name='source1', provider='p1'))
        registry.register_source(DataSource(name='source2', provider='p2'))

        sources = registry.list_sources()

        assert len(sources) == 2

    def test_list_features_with_filter(self, registry):
        """Should filter features by category."""
        registry.register_feature(FeatureDefinition(
            name='rsi', category=FeatureCategory.MOMENTUM
        ))
        registry.register_feature(FeatureDefinition(
            name='atr', category=FeatureCategory.VOLATILITY
        ))
        registry.register_feature(FeatureDefinition(
            name='macd', category=FeatureCategory.MOMENTUM
        ))

        momentum = registry.list_features(category=FeatureCategory.MOMENTUM)

        assert len(momentum) == 2
        assert all(f.category == FeatureCategory.MOMENTUM for f in momentum)

    def test_search_features(self, registry):
        """Should search features."""
        registry.register_feature(FeatureDefinition(
            name='rsi_14', category=FeatureCategory.MOMENTUM,
            description='Relative Strength Index'
        ))
        registry.register_feature(FeatureDefinition(
            name='macd', category=FeatureCategory.MOMENTUM,
            description='Moving Average Convergence'
        ))

        results = registry.search_features('strength')

        assert len(results) == 1
        assert results[0].name == 'rsi_14'

    def test_get_stats(self, registry):
        """Should return statistics."""
        registry.register_source(DataSource(name='s1', provider='p1'))
        registry.register_feature(FeatureDefinition(
            name='f1', category=FeatureCategory.MOMENTUM
        ))

        stats = registry.get_stats()

        assert stats['total_sources'] == 1
        assert stats['total_features'] == 1

    def test_feature_to_dict(self):
        """Should convert feature to dictionary."""
        feature = FeatureDefinition(
            name='test',
            category=FeatureCategory.TREND,
            description='Test feature',
        )

        d = feature.to_dict()

        assert d['name'] == 'test'
        assert d['category'] == 'trend'

    def test_source_to_dict(self):
        """Should convert source to dictionary."""
        source = DataSource(
            name='test_source',
            provider='TestProvider',
            features=['a', 'b', 'c'],
        )

        d = source.to_dict()

        assert d['name'] == 'test_source'
        assert d['provider'] == 'TestProvider'


class TestFeatureDiscovery:
    """Tests for FeatureDiscovery."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        np.random.seed(42)
        n = 300

        df = pd.DataFrame({
            'close': np.cumsum(np.random.randn(n)) + 100,
            'volume': np.abs(np.random.randn(n) * 1000000),
            'high': np.cumsum(np.random.randn(n)) + 101,
            'low': np.cumsum(np.random.randn(n)) + 99,
        })

        # Create target (forward return)
        df['forward_return'] = df['close'].pct_change().shift(-1)

        return df

    def test_initialization(self):
        """Should initialize with defaults."""
        discovery = FeatureDiscovery()
        assert discovery.min_correlation == 0.05
        assert discovery.min_stability == 0.3

    def test_discover_from_columns(self, sample_data):
        """Should discover features."""
        discovery = FeatureDiscovery(min_correlation=0.01, min_stability=0.0)

        discovered = discovery.discover_from_columns(
            sample_data,
            target_column='forward_return',
            base_columns=['close', 'volume'],
        )

        assert len(discovered) > 0
        assert all(isinstance(f, DiscoveredFeature) for f in discovered)

    def test_discovered_feature_apply(self, sample_data):
        """Should apply discovered feature transformation."""
        feature = DiscoveredFeature(
            name='close_returns_1',
            transform=FeatureTransform.RETURNS,
            base_column='close',
            parameters={'period': 1},
        )

        result = feature.apply(sample_data)

        assert len(result) == len(sample_data)
        assert result.notna().sum() > 0

    def test_feature_transforms(self, sample_data):
        """Should apply different transforms."""
        transforms = [
            (FeatureTransform.RETURNS, {'period': 5}),
            (FeatureTransform.VOLATILITY, {'window': 20}),
            (FeatureTransform.ZSCORE, {'window': 20}),
            (FeatureTransform.MOMENTUM, {'period': 10}),
        ]

        for transform, params in transforms:
            feature = DiscoveredFeature(
                name=f'test_{transform.value}',
                transform=transform,
                base_column='close',
                parameters=params,
            )

            result = feature.apply(sample_data)
            assert len(result) == len(sample_data), f"Failed for {transform}"

    def test_suggest_combinations(self, sample_data):
        """Should suggest feature combinations."""
        discovery = FeatureDiscovery(min_correlation=0.01, min_stability=0.0)

        discovered = discovery.discover_from_columns(
            sample_data,
            target_column='forward_return',
        )

        suggestions = discovery.suggest_combinations(discovered, max_combinations=5)

        assert len(suggestions) <= 5
        assert all(isinstance(s, DiscoveredFeature) for s in suggestions)

    def test_feature_quality_metrics(self, sample_data):
        """Should calculate quality metrics."""
        discovery = FeatureDiscovery(min_correlation=0.0, min_stability=0.0)

        discovered = discovery.discover_from_columns(
            sample_data,
            target_column='forward_return',
        )

        if discovered:
            feature = discovered[0]
            assert hasattr(feature, 'correlation_with_target')
            assert hasattr(feature, 'stability_score')
            assert hasattr(feature, 'information_ratio')

    def test_discovered_to_dict(self):
        """Should convert to dictionary."""
        feature = DiscoveredFeature(
            name='test_feature',
            transform=FeatureTransform.RETURNS,
            base_column='close',
            parameters={'period': 1},
            correlation_with_target=0.15,
        )

        d = feature.to_dict()

        assert d['name'] == 'test_feature'
        assert d['transform'] == 'returns'
        assert d['correlation_with_target'] == 0.15


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n = 100

        X = pd.DataFrame({
            'a': np.random.randn(n),
            'b': np.random.randn(n),
        })
        y = X['a'] * 0.5 + np.random.randn(n) * 0.1

        return X, pd.Series(y)

    def test_analyze_importance(self, sample_data):
        """Should analyze with convenience function."""
        X, y = sample_data

        results = analyze_importance(X, y)

        assert isinstance(results, list)
        assert len(results) == 2

    def test_get_top_features(self, sample_data):
        """Should get top feature names."""
        X, y = sample_data

        top = get_top_features(X, y, n=1)

        assert len(top) == 1
        assert top[0] == 'a'  # Most correlated

    def test_discover_features_function(self):
        """Should discover with convenience function."""
        np.random.seed(42)
        df = pd.DataFrame({
            'close': np.cumsum(np.random.randn(200)) + 100,
            'forward_return': np.random.randn(200) * 0.01,
        })

        discovered = discover_features(df, 'forward_return')

        assert isinstance(discovered, list)

    def test_suggest_features_function(self):
        """Should suggest feature names."""
        np.random.seed(42)
        df = pd.DataFrame({
            'close': np.cumsum(np.random.randn(200)) + 100,
            'forward_return': np.random.randn(200) * 0.01,
        })

        suggestions = suggest_features(df, 'forward_return', n=5)

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5


# Run with: pytest tests/test_data_exploration.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
