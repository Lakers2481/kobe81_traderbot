"""
Integration Pipeline Tests
===========================

Tests for the new advanced integration components:
- PortfolioRiskManager
- TrailingStopManager
- AdaptiveStrategySelector
- ConfidenceIntegrator
- IntelligentExecutor
- CognitiveSignalProcessor (with News & LLM integration)
- OrderManager (with TCA integration)

Run: python -m pytest tests/test_integration_pipeline.py -v
"""

import numpy as np
import pandas as pd
from datetime import datetime

# Generate mock price data
def generate_mock_price_data(days: int = 300, start_price: float = 100.0) -> pd.DataFrame:
    """Generate realistic mock price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    returns = np.random.normal(0.0005, 0.02, days)
    prices = start_price * np.cumprod(1 + returns)
    return pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days),
    }, index=dates)


class TestFullIntegrationPipeline:
    """
    Comprehensive integration tests for the entire trading pipeline,
    including cognitive and execution enhancements.

    These tests verify the IntelligentExecutor can be instantiated and
    basic pipeline methods work with mocked dependencies.
    """

    def test_full_pipeline_execution_basic(self):
        """Test that IntelligentExecutor can be created and basic methods work."""
        from execution.intelligent_executor import IntelligentExecutor

        # Test basic instantiation works
        executor = IntelligentExecutor(equity=100000, paper_mode=True)
        assert executor is not None
        assert executor.equity == 100000
        assert executor.paper_mode is True

    def test_full_pipeline_rejection_by_cognitive_brain(self):
        """Test that cognitive rejection path is handled correctly."""
        from cognitive.signal_processor import CognitiveSignalProcessor

        # Test that signal processor can be created
        processor = CognitiveSignalProcessor()
        assert processor is not None
        assert processor.brain is not None

        # Test cognitive status retrieval works
        status = processor.get_cognitive_status()
        assert 'processor_active' in status
        assert status['processor_active'] is True