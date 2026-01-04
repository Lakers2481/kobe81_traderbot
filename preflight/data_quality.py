"""
Data Quality Gates for Frozen Datasets.

Validates data quality before backtesting or live trading:
- Coverage: Minimum years of history per symbol
- Gaps: Maximum allowable missing days
- Staleness: Data freshness requirements
- Integrity: Hash verification for frozen datasets
- Consistency: OHLC relationship checks

Integrates with KnowledgeBoundary for uncertainty/stand-down decisions.

Usage:
    from preflight.data_quality import DataQualityGate

    gate = DataQualityGate()

    # Validate a frozen dataset
    report = gate.validate_dataset(dataset_id='stooq_1d_2015_2025_abc123')

    if report.passed:
        print("Data quality OK, proceed with backtest")
    else:
        print(f"Data quality issues: {report.issues}")

    # Check with KnowledgeBoundary integration
    kb_report = gate.assess_with_knowledge_boundary(dataset_id, context)
    if kb_report.should_stand_down:
        print("STAND DOWN - data uncertainty too high")
"""
from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"   # Production-grade
    GOOD = "good"             # Acceptable for backtesting
    MARGINAL = "marginal"     # Caution advised
    POOR = "poor"             # Not recommended
    FAILED = "failed"         # Do not use


class DataIssue(Enum):
    """Types of data quality issues."""
    INSUFFICIENT_HISTORY = "insufficient_history"
    EXCESSIVE_GAPS = "excessive_gaps"
    STALE_DATA = "stale_data"
    INTEGRITY_FAILURE = "integrity_failure"
    OHLC_VIOLATION = "ohlc_violation"
    MISSING_SYMBOLS = "missing_symbols"
    LOW_COVERAGE = "low_coverage"
    SUSPICIOUS_PRICES = "suspicious_prices"
    VOLUME_ANOMALY = "volume_anomaly"
    DUPLICATES = "duplicates"


@dataclass
class SymbolCoverage:
    """Coverage statistics for a single symbol."""
    symbol: str
    first_date: str
    last_date: str
    total_days: int
    trading_days: int
    gap_days: int
    gap_pct: float
    years_of_history: float
    ohlc_violations: int
    price_anomalies: int

    @property
    def is_sufficient(self) -> bool:
        """Check if coverage is sufficient."""
        return self.years_of_history >= 5 and self.gap_pct < 0.05


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    dataset_id: str
    evaluated_at: str
    passed: bool
    quality_level: DataQualityLevel

    # Coverage metrics
    total_symbols: int = 0
    symbols_with_sufficient_coverage: int = 0
    coverage_pct: float = 0.0
    min_years_of_history: float = 0.0
    max_gap_pct: float = 0.0
    avg_gap_pct: float = 0.0

    # Data integrity
    integrity_verified: bool = False
    hash_match: bool = False

    # Freshness
    most_recent_date: str = ""
    days_since_update: int = 0
    is_stale: bool = False

    # Issues found
    issues: List[Dict[str, Any]] = field(default_factory=list)
    issue_counts: Dict[str, int] = field(default_factory=dict)

    # Per-symbol details
    symbol_coverage: List[SymbolCoverage] = field(default_factory=list)

    # KnowledgeBoundary integration
    uncertainty_level: str = "unknown"
    should_stand_down: bool = False
    confidence_adjustment: float = 0.0

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "evaluated_at": self.evaluated_at,
            "passed": self.passed,
            "quality_level": self.quality_level.value,
            "total_symbols": self.total_symbols,
            "symbols_with_sufficient_coverage": self.symbols_with_sufficient_coverage,
            "coverage_pct": round(self.coverage_pct, 4),
            "min_years_of_history": round(self.min_years_of_history, 2),
            "max_gap_pct": round(self.max_gap_pct, 4),
            "avg_gap_pct": round(self.avg_gap_pct, 4),
            "integrity_verified": self.integrity_verified,
            "most_recent_date": self.most_recent_date,
            "days_since_update": self.days_since_update,
            "is_stale": self.is_stale,
            "issue_count": len(self.issues),
            "issue_counts": self.issue_counts,
            "uncertainty_level": self.uncertainty_level,
            "should_stand_down": self.should_stand_down,
            "recommendations": self.recommendations,
        }

    def save(self, path: Path):
        """Save report to JSON."""
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))


@dataclass
class DataQualityRequirements:
    """Requirements for data quality validation."""
    min_years_history: float = 5.0
    max_gap_pct: float = 0.05  # 5%
    max_stale_days: int = 7
    min_coverage_pct: float = 0.90  # 90% of symbols must meet requirements
    max_ohlc_violations_pct: float = 0.01  # 1%
    max_price_anomaly_pct: float = 0.01
    price_spike_threshold: float = 0.50  # 50% daily move = suspicious
    min_volume_threshold: int = 0  # Minimum daily volume


class DataQualityGate:
    """
    Data quality validation gate for frozen datasets.

    Validates:
    - Coverage (years of history)
    - Gaps (missing trading days)
    - Integrity (hash verification)
    - Freshness (staleness)
    - Consistency (OHLC relationships)

    Integrates with KnowledgeBoundary for uncertainty assessment.
    """

    def __init__(
        self,
        requirements: Optional[DataQualityRequirements] = None,
        lake_dir: str = "data/lake",
        manifest_dir: str = "data/manifests",
        reports_dir: str = "reports/data_quality",
    ):
        self.requirements = requirements or DataQualityRequirements()
        self.lake_dir = Path(lake_dir)
        self.manifest_dir = Path(manifest_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Lazy load dependencies
        self._knowledge_boundary = None

        logger.info("DataQualityGate initialized")

    @property
    def knowledge_boundary(self):
        """Lazy load KnowledgeBoundary."""
        if self._knowledge_boundary is None:
            try:
                from cognitive.knowledge_boundary import KnowledgeBoundary
                self._knowledge_boundary = KnowledgeBoundary()
            except ImportError:
                logger.warning("KnowledgeBoundary not available")
        return self._knowledge_boundary

    def validate_dataset(
        self,
        dataset_id: str,
        verify_integrity: bool = True,
    ) -> DataQualityReport:
        """
        Validate a frozen dataset.

        Args:
            dataset_id: Dataset identifier
            verify_integrity: Whether to verify file hashes

        Returns:
            DataQualityReport with validation results
        """
        report = DataQualityReport(
            dataset_id=dataset_id,
            evaluated_at=datetime.now().isoformat(),
            passed=False,
            quality_level=DataQualityLevel.FAILED,
        )

        try:
            # Load data from lake
            from data.lake import LakeReader
            reader = LakeReader(
                lake_dir=self.lake_dir,
                manifest_dir=self.manifest_dir,
            )

            df = reader.load_dataset(
                dataset_id,
                verify_integrity=verify_integrity,
            )

            if df.empty:
                report.issues.append({
                    "type": DataIssue.INTEGRITY_FAILURE.value,
                    "message": "Dataset is empty or could not be loaded",
                })
                return report

            report.integrity_verified = verify_integrity
            report.hash_match = True  # If we got here, integrity passed

        except FileNotFoundError:
            report.issues.append({
                "type": DataIssue.INTEGRITY_FAILURE.value,
                "message": f"Dataset not found: {dataset_id}",
            })
            return report
        except Exception as e:
            report.issues.append({
                "type": DataIssue.INTEGRITY_FAILURE.value,
                "message": f"Failed to load dataset: {str(e)}",
            })
            return report

        # Validate the data
        return self._validate_dataframe(df, report)

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        dataset_id: str = "inline",
    ) -> DataQualityReport:
        """
        Validate a DataFrame directly (for testing or inline data).

        Args:
            df: DataFrame with OHLCV data
            dataset_id: Identifier for the report

        Returns:
            DataQualityReport with validation results
        """
        report = DataQualityReport(
            dataset_id=dataset_id,
            evaluated_at=datetime.now().isoformat(),
            passed=False,
            quality_level=DataQualityLevel.FAILED,
        )

        return self._validate_dataframe(df, report)

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        report: DataQualityReport,
    ) -> DataQualityReport:
        """Internal validation logic."""

        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Get symbols
        if 'symbol' in df.columns:
            symbols = df['symbol'].unique()
        else:
            symbols = ['UNKNOWN']
            df['symbol'] = 'UNKNOWN'

        report.total_symbols = len(symbols)

        # === Check 1: Per-symbol coverage ===
        symbol_coverages = []
        issue_counts = {issue.value: 0 for issue in DataIssue}

        for symbol in symbols:
            sym_df = df[df['symbol'] == symbol].copy()

            if sym_df.empty:
                continue

            sym_df = sym_df.sort_values('timestamp')

            first_date = sym_df['timestamp'].min()
            last_date = sym_df['timestamp'].max()
            total_days = (last_date - first_date).days

            # Estimate trading days (approximate)
            trading_days = len(sym_df)

            # Calculate gaps
            if len(sym_df) > 1:
                date_diffs = sym_df['timestamp'].diff().dropna()
                # Gaps are days where diff > 3 (accounting for weekends)
                gap_days = sum(1 for d in date_diffs if d.days > 3)
                expected_trading_days = total_days * 252 / 365  # Approximate
                gap_pct = gap_days / max(expected_trading_days, 1)
            else:
                gap_days = 0
                gap_pct = 0.0

            years_of_history = total_days / 365.0

            # Check OHLC violations
            ohlc_violations = 0
            if all(col in sym_df.columns for col in ['open', 'high', 'low', 'close']):
                # High should be >= open, close, low
                high_violations = (
                    (sym_df['high'] < sym_df['open']) |
                    (sym_df['high'] < sym_df['close']) |
                    (sym_df['high'] < sym_df['low'])
                ).sum()

                # Low should be <= open, close, high
                low_violations = (
                    (sym_df['low'] > sym_df['open']) |
                    (sym_df['low'] > sym_df['close']) |
                    (sym_df['low'] > sym_df['high'])
                ).sum()

                ohlc_violations = high_violations + low_violations

            # Check price anomalies
            price_anomalies = 0
            if 'close' in sym_df.columns and len(sym_df) > 1:
                returns = sym_df['close'].pct_change().abs()
                price_anomalies = (returns > self.requirements.price_spike_threshold).sum()

            coverage = SymbolCoverage(
                symbol=symbol,
                first_date=first_date.strftime('%Y-%m-%d'),
                last_date=last_date.strftime('%Y-%m-%d'),
                total_days=total_days,
                trading_days=trading_days,
                gap_days=gap_days,
                gap_pct=gap_pct,
                years_of_history=years_of_history,
                ohlc_violations=ohlc_violations,
                price_anomalies=price_anomalies,
            )

            symbol_coverages.append(coverage)

            # Track issues
            if years_of_history < self.requirements.min_years_history:
                issue_counts[DataIssue.INSUFFICIENT_HISTORY.value] += 1
                report.issues.append({
                    "type": DataIssue.INSUFFICIENT_HISTORY.value,
                    "symbol": symbol,
                    "message": f"{symbol}: Only {years_of_history:.1f} years (need {self.requirements.min_years_history})",
                })

            if gap_pct > self.requirements.max_gap_pct:
                issue_counts[DataIssue.EXCESSIVE_GAPS.value] += 1
                report.issues.append({
                    "type": DataIssue.EXCESSIVE_GAPS.value,
                    "symbol": symbol,
                    "message": f"{symbol}: {gap_pct:.1%} gaps (max {self.requirements.max_gap_pct:.1%})",
                })

            if ohlc_violations > 0:
                issue_counts[DataIssue.OHLC_VIOLATION.value] += ohlc_violations
                report.issues.append({
                    "type": DataIssue.OHLC_VIOLATION.value,
                    "symbol": symbol,
                    "message": f"{symbol}: {ohlc_violations} OHLC violations",
                })

            if price_anomalies > 0:
                issue_counts[DataIssue.SUSPICIOUS_PRICES.value] += price_anomalies

        report.symbol_coverage = symbol_coverages
        report.issue_counts = issue_counts

        # === Aggregate metrics ===
        if symbol_coverages:
            sufficient = [c for c in symbol_coverages if c.is_sufficient]
            report.symbols_with_sufficient_coverage = len(sufficient)
            report.coverage_pct = len(sufficient) / len(symbol_coverages)
            report.min_years_of_history = min(c.years_of_history for c in symbol_coverages)
            report.max_gap_pct = max(c.gap_pct for c in symbol_coverages)
            report.avg_gap_pct = sum(c.gap_pct for c in symbol_coverages) / len(symbol_coverages)

        # === Check 2: Staleness ===
        if 'timestamp' in df.columns:
            most_recent = df['timestamp'].max()
            report.most_recent_date = most_recent.strftime('%Y-%m-%d')
            report.days_since_update = (datetime.now() - most_recent).days
            report.is_stale = report.days_since_update > self.requirements.max_stale_days

            if report.is_stale:
                issue_counts[DataIssue.STALE_DATA.value] = 1
                report.issues.append({
                    "type": DataIssue.STALE_DATA.value,
                    "message": f"Data is {report.days_since_update} days old (max {self.requirements.max_stale_days})",
                })

        # === Check 3: Duplicates ===
        if 'symbol' in df.columns and 'timestamp' in df.columns:
            duplicates = df.duplicated(subset=['symbol', 'timestamp']).sum()
            if duplicates > 0:
                issue_counts[DataIssue.DUPLICATES.value] = duplicates
                report.issues.append({
                    "type": DataIssue.DUPLICATES.value,
                    "message": f"{duplicates} duplicate rows found",
                })

        # === Determine quality level ===
        critical_issues = (
            issue_counts.get(DataIssue.INTEGRITY_FAILURE.value, 0) +
            issue_counts.get(DataIssue.INSUFFICIENT_HISTORY.value, 0) * 0.1
        )

        if critical_issues > 0 or report.coverage_pct < 0.5:
            report.quality_level = DataQualityLevel.FAILED
        elif report.coverage_pct < 0.7 or report.is_stale:
            report.quality_level = DataQualityLevel.POOR
        elif report.coverage_pct < 0.85:
            report.quality_level = DataQualityLevel.MARGINAL
        elif report.coverage_pct < 0.95:
            report.quality_level = DataQualityLevel.GOOD
        else:
            report.quality_level = DataQualityLevel.EXCELLENT

        # === Pass/Fail determination ===
        report.passed = (
            report.quality_level in [DataQualityLevel.EXCELLENT, DataQualityLevel.GOOD]
            and report.coverage_pct >= self.requirements.min_coverage_pct
            and not report.is_stale
        )

        # === Generate recommendations ===
        report.recommendations = self._generate_recommendations(report)

        # Save report
        report_path = self.reports_dir / f"{report.dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report.save(report_path)

        return report

    def _generate_recommendations(self, report: DataQualityReport) -> List[str]:
        """Generate recommendations based on issues found."""
        recs = []

        if report.issue_counts.get(DataIssue.INSUFFICIENT_HISTORY.value, 0) > 0:
            recs.append(
                f"Exclude symbols with <{self.requirements.min_years_history} years history"
            )

        if report.issue_counts.get(DataIssue.EXCESSIVE_GAPS.value, 0) > 0:
            recs.append(
                f"Review symbols with >{self.requirements.max_gap_pct:.0%} gaps - may need re-fetching"
            )

        if report.is_stale:
            recs.append(
                f"Update dataset - data is {report.days_since_update} days old"
            )

        if report.issue_counts.get(DataIssue.OHLC_VIOLATION.value, 0) > 0:
            recs.append(
                "Investigate OHLC violations - possible data corruption"
            )

        if report.coverage_pct < self.requirements.min_coverage_pct:
            recs.append(
                f"Coverage {report.coverage_pct:.0%} below {self.requirements.min_coverage_pct:.0%} threshold"
            )

        if not recs:
            recs.append("Data quality acceptable for backtesting")

        return recs

    def assess_with_knowledge_boundary(
        self,
        dataset_id: str,
        context: Optional[Dict[str, Any]] = None,
        verify_integrity: bool = True,
    ) -> DataQualityReport:
        """
        Validate dataset with KnowledgeBoundary integration.

        Combines data quality checks with uncertainty assessment
        for stand-down recommendations.

        Args:
            dataset_id: Dataset identifier
            context: Additional context for KnowledgeBoundary
            verify_integrity: Whether to verify file hashes

        Returns:
            DataQualityReport with KB assessment
        """
        # First, get basic validation
        report = self.validate_dataset(dataset_id, verify_integrity)

        # Then integrate with KnowledgeBoundary
        if self.knowledge_boundary and context is not None:
            try:
                # Build signal for KB assessment
                signal = {
                    'strategy': context.get('strategy', 'unknown'),
                    'entry_price': context.get('entry_price', 100),
                }

                # Enhance context with data quality info
                enhanced_context = context.copy()
                enhanced_context.update({
                    'data_quality': report.quality_level.value,
                    'data_coverage_pct': report.coverage_pct,
                    'data_gaps_pct': report.avg_gap_pct,
                    'data_stale': report.is_stale,
                    'data_issues_count': len(report.issues),
                })

                # Get KB assessment
                kb_assessment = self.knowledge_boundary.assess_knowledge_state(
                    signal, enhanced_context
                )

                report.uncertainty_level = kb_assessment.uncertainty_level.value
                report.should_stand_down = kb_assessment.should_stand_down
                report.confidence_adjustment = kb_assessment.confidence_adjustment

                # Add KB recommendations
                if kb_assessment.should_stand_down:
                    report.recommendations.insert(0, "STAND DOWN - high uncertainty")
                    report.passed = False

                for rec in kb_assessment.recommendations:
                    if rec not in report.recommendations:
                        report.recommendations.append(rec)

            except Exception as e:
                logger.warning(f"KnowledgeBoundary integration failed: {e}")

        return report

    def quick_check(
        self,
        df: pd.DataFrame,
    ) -> Tuple[bool, str]:
        """
        Quick pass/fail check without full report.

        Args:
            df: DataFrame to check

        Returns:
            Tuple of (passed, reason)
        """
        if df.empty:
            return False, "DataFrame is empty"

        # Check required columns
        required = ['timestamp', 'open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return False, f"Missing columns: {missing}"

        # Check minimum rows
        if len(df) < 252:  # Less than 1 year
            return False, f"Insufficient data: {len(df)} rows"

        # Check for nulls
        null_pct = df[required].isnull().sum().sum() / (len(df) * len(required))
        if null_pct > 0.05:
            return False, f"Too many nulls: {null_pct:.1%}"

        return True, "Quick check passed"


# Convenience function
def validate_data_quality(
    dataset_id: str = None,
    df: pd.DataFrame = None,
    requirements: Optional[DataQualityRequirements] = None,
) -> DataQualityReport:
    """
    Convenience function to validate data quality.

    Args:
        dataset_id: Dataset ID (for frozen datasets)
        df: DataFrame (for inline validation)
        requirements: Custom requirements

    Returns:
        DataQualityReport
    """
    gate = DataQualityGate(requirements=requirements)

    if dataset_id:
        return gate.validate_dataset(dataset_id)
    elif df is not None:
        return gate.validate_dataframe(df)
    else:
        raise ValueError("Must provide dataset_id or df")
