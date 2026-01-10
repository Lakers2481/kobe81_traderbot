"""
Tests for Data Quality Gate.

Tests data validation, coverage checks, and KnowledgeBoundary integration.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
from pathlib import Path

from preflight.data_quality import (
    DataQualityGate,
    DataQualityRequirements,
    DataQualityReport,
    DataQualityLevel,
    DataIssue,
    validate_data_quality,
)


def generate_test_ohlcv(
    symbols: list = None,
    start_date: str = '2019-01-01',
    end_date: str = None,  # Defaults to today
    include_gaps: bool = False,
    include_ohlc_violations: bool = False,
) -> pd.DataFrame:
    """Generate test OHLCV data."""
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'GOOGL']

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    dates = pd.date_range(start_date, end_date, freq='B')

    all_data = []
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)

        # Optionally skip some dates to create gaps
        if include_gaps:
            dates_to_use = dates[::2]  # Skip every other day
        else:
            dates_to_use = dates

        base_price = np.random.uniform(50, 200)
        prices = [base_price]

        for _ in range(len(dates_to_use) - 1):
            change = np.random.normal(0.0005, 0.02)
            prices.append(prices[-1] * (1 + change))

        for i, date in enumerate(dates_to_use):
            close = prices[i]
            daily_range = close * np.random.uniform(0.01, 0.03)
            high = close + daily_range * np.random.uniform(0.3, 0.7)
            low = close - daily_range * np.random.uniform(0.3, 0.7)
            open_price = low + (high - low) * np.random.uniform(0.2, 0.8)

            # Optionally create OHLC violations
            if include_ohlc_violations and i % 50 == 0:
                high = low - 1  # High < Low = violation

            all_data.append({
                'timestamp': date,
                'symbol': symbol,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': int(np.random.uniform(1e6, 1e7)),
            })

    return pd.DataFrame(all_data)


class TestDataQualityGate:
    """Tests for DataQualityGate."""

    @pytest.fixture
    def gate(self):
        """Create test gate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield DataQualityGate(reports_dir=tmpdir)

    def test_validate_good_data(self, gate):
        """Should pass good quality data."""
        # Explicitly set end_date to today to ensure fresh data
        today = datetime.now().strftime('%Y-%m-%d')
        df = generate_test_ohlcv(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2019-01-01',
            end_date=today,
        )

        report = gate.validate_dataframe(df)

        assert report.passed, f"Failed with issues: {report.issues}"
        assert report.quality_level in [DataQualityLevel.EXCELLENT, DataQualityLevel.GOOD]
        assert report.total_symbols == 3

    def test_detect_gaps(self, gate):
        """Should detect data gaps."""
        df = generate_test_ohlcv(
            symbols=['AAPL'],
            include_gaps=True,
        )

        report = gate.validate_dataframe(df)

        # Should have gap issues
        gap_issues = [i for i in report.issues if i['type'] == DataIssue.EXCESSIVE_GAPS.value]
        assert len(gap_issues) > 0

    def test_detect_ohlc_violations(self, gate):
        """Should detect OHLC violations."""
        df = generate_test_ohlcv(
            symbols=['AAPL'],
            include_ohlc_violations=True,
        )

        report = gate.validate_dataframe(df)

        ohlc_issues = report.issue_counts.get(DataIssue.OHLC_VIOLATION.value, 0)
        assert ohlc_issues > 0

    def test_insufficient_history(self, gate):
        """Should detect insufficient history."""
        df = generate_test_ohlcv(
            symbols=['AAPL'],
            start_date='2023-01-01',  # Only 1 year
            end_date='2023-12-31',
        )

        report = gate.validate_dataframe(df)

        history_issues = [
            i for i in report.issues
            if i['type'] == DataIssue.INSUFFICIENT_HISTORY.value
        ]
        assert len(history_issues) > 0

    def test_coverage_calculation(self, gate):
        """Should calculate coverage correctly."""
        df = generate_test_ohlcv(
            symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            start_date='2018-01-01',
            end_date='2023-12-31',
        )

        report = gate.validate_dataframe(df)

        assert report.total_symbols == 5
        assert report.coverage_pct > 0

    def test_quick_check_pass(self, gate):
        """Quick check should pass good data."""
        df = generate_test_ohlcv()

        passed, reason = gate.quick_check(df)

        assert passed

    def test_quick_check_fail_empty(self, gate):
        """Quick check should fail empty data."""
        df = pd.DataFrame()

        passed, reason = gate.quick_check(df)

        assert not passed
        assert "empty" in reason.lower()

    def test_quick_check_fail_missing_columns(self, gate):
        """Quick check should fail with missing columns."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100),
            'close': np.random.uniform(100, 110, 100),
            # Missing: open, high, low
        })

        passed, reason = gate.quick_check(df)

        assert not passed
        assert "Missing columns" in reason


class TestDataQualityRequirements:
    """Tests for DataQualityRequirements."""

    def test_default_requirements(self):
        """Should have sensible defaults."""
        req = DataQualityRequirements()

        assert req.min_years_history == 5.0
        assert req.max_gap_pct == 0.05
        assert req.max_stale_days == 7
        assert req.min_coverage_pct == 0.90

    def test_custom_requirements(self):
        """Should accept custom requirements."""
        req = DataQualityRequirements(
            min_years_history=10.0,
            max_gap_pct=0.02,
        )

        assert req.min_years_history == 10.0
        assert req.max_gap_pct == 0.02


class TestDataQualityReport:
    """Tests for DataQualityReport."""

    def test_report_to_dict(self):
        """Should convert to dictionary."""
        report = DataQualityReport(
            dataset_id="test",
            evaluated_at=datetime.now().isoformat(),
            passed=True,
            quality_level=DataQualityLevel.GOOD,
            total_symbols=100,
            coverage_pct=0.95,
        )

        d = report.to_dict()

        assert d['dataset_id'] == 'test'
        assert d['passed']
        assert d['quality_level'] == 'good'

    def test_report_save_load(self):
        """Should save and load report."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            report_path = Path(f.name)

        report = DataQualityReport(
            dataset_id="test",
            evaluated_at=datetime.now().isoformat(),
            passed=True,
            quality_level=DataQualityLevel.GOOD,
            total_symbols=100,
        )

        report.save(report_path)

        # Verify file exists and has content
        assert report_path.exists()
        content = report_path.read_text()
        assert 'test' in content

        # Cleanup
        report_path.unlink()


class TestConvenienceFunction:
    """Tests for validate_data_quality convenience function."""

    def test_validate_dataframe(self):
        """Should validate DataFrame directly."""
        df = generate_test_ohlcv()

        report = validate_data_quality(df=df)

        assert isinstance(report, DataQualityReport)

    def test_missing_args_raises(self):
        """Should raise error when no data provided."""
        with pytest.raises(ValueError):
            validate_data_quality()


# Run with: pytest tests/test_data_quality.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
