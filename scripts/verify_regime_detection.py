"""
Verify Regime Detection Accuracy (Jim Simons Standard)
=======================================================

Measures how accurately the HMM regime detector classifies market regimes.

Ground Truth Definition:
    - BULL: SPY 60-day forward return >= +10%
    - BEAR: SPY 60-day forward return <= -10%
    - NEUTRAL: SPY 60-day forward return between -10% and +10%

Metrics:
    - Accuracy: Overall % correct
    - Precision: Of predicted BULL, how many are actually BULL?
    - Recall: Of actual BULL, how many did we predict?
    - F1 Score: Harmonic mean of precision and recall
    - Transition Lag: How many days until regime change detected?

Jim Simons Standard:
    - Accuracy > 70% across all regimes = GOOD
    - Precision/Recall > 65% per regime = ACCEPTABLE
    - Transition lag < 10 days = RESPONSIVE
    - If lag > 20 days = TOO SLOW (missed opportunity)

Usage:
    python scripts/verify_regime_detection.py
    python scripts/verify_regime_detection.py --forward-days 90
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from ml_advanced.hmm_regime_detector import HMMRegimeDetector, MarketRegime
from data.providers.polygon_eod import fetch_daily_bars_polygon


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RegimeAccuracyMetrics:
    """Accuracy metrics for a single regime."""
    regime: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    support: int  # Number of actual instances
    confusion_matrix: Dict[str, int]


@dataclass
class RegimeDetectionReport:
    """Full regime detection verification report."""
    overall_accuracy: float
    regime_metrics: List[RegimeAccuracyMetrics]
    avg_transition_lag_days: Optional[float]
    median_transition_lag_days: Optional[float]
    confusion_matrix: pd.DataFrame
    passed: bool
    failure_reasons: List[str]

    # Thresholds
    accuracy_threshold: float = 0.70
    precision_threshold: float = 0.65
    recall_threshold: float = 0.65
    lag_threshold_days: float = 10.0


# ============================================================================
# Ground Truth Definition
# ============================================================================

def define_ground_truth(
    spy_data: pd.DataFrame,
    forward_days: int = 60,
    bull_threshold: float = 0.10,
    bear_threshold: float = -0.10,
) -> pd.DataFrame:
    """
    Define ground truth regimes based on forward returns.

    Args:
        spy_data: SPY OHLCV data
        forward_days: Days forward to look (default: 60 trading days ≈ 3 months)
        bull_threshold: Return threshold for BULL (default: +10%)
        bear_threshold: Return threshold for BEAR (default: -10%)

    Returns:
        DataFrame with 'ground_truth_regime' column
    """
    spy_data = spy_data.copy()

    # Calculate forward returns
    spy_data['close_future'] = spy_data['close'].shift(-forward_days)
    spy_data['forward_return'] = (spy_data['close_future'] - spy_data['close']) / spy_data['close']

    # Classify regimes
    def classify_regime(fwd_ret):
        if pd.isna(fwd_ret):
            return None
        elif fwd_ret >= bull_threshold:
            return 'BULLISH'
        elif fwd_ret <= bear_threshold:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    spy_data['ground_truth_regime'] = spy_data['forward_return'].apply(classify_regime)

    return spy_data


# ============================================================================
# HMM Predictions
# ============================================================================

def load_hmm_model(model_path: Path) -> HMMRegimeDetector:
    """Load trained HMM model."""
    if not model_path.exists():
        raise FileNotFoundError(f"HMM model not found at {model_path}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print(f"[OK] Loaded HMM model from {model_path}")

    # Check if it's a dict (old format) or HMMRegimeDetector object
    if isinstance(model_data, dict):
        # Reconstruct HMMRegimeDetector from saved state
        detector = HMMRegimeDetector()
        detector.model = model_data['model']
        detector.state_labels = {int(k): MarketRegime(v) for k, v in model_data['state_labels'].items()}
        detector.feature_names = model_data['feature_names']
        detector.feature_means = model_data['feature_means']
        detector.feature_stds = model_data['feature_stds']
        detector.n_states = model_data['n_states']
        detector.is_fitted = model_data['is_fitted']
    else:
        detector = model_data

    print(f"  States: {detector.state_labels}")
    print(f"  Is Fitted: {detector.is_fitted}")

    return detector


def get_hmm_predictions(
    detector: HMMRegimeDetector,
    spy_data: pd.DataFrame,
    vix_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Get HMM predictions for SPY data."""
    spy_data = spy_data.copy()

    # Prepare features
    try:
        features = detector.prepare_features(spy_data, vix_data)
    except Exception as e:
        print(f"[ERROR] Failed to prepare features: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Failed to prepare features for HMM: {e}")

    if features is None or len(features) == 0:
        raise ValueError("Failed to prepare features for HMM - features is None or empty")

    print(f"[DEBUG] Features shape: {features.shape}")

    # Get predictions
    if not detector.is_fitted:
        raise ValueError("HMM model is not fitted")

    predicted_states = detector.model.predict(features)

    # Map states to regime names
    regime_names = [detector.state_labels.get(state, 'UNKNOWN').name for state in predicted_states]

    # Create result DataFrame
    result = spy_data.copy()
    result['hmm_predicted_regime'] = None
    result.iloc[:len(regime_names), result.columns.get_loc('hmm_predicted_regime')] = regime_names

    return result


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: List[str],
) -> pd.DataFrame:
    """Calculate confusion matrix."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def calculate_metrics_per_regime(
    y_true: pd.Series,
    y_pred: pd.Series,
    regime: str,
) -> RegimeAccuracyMetrics:
    """Calculate precision, recall, F1 for a single regime."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    # Binary classification: regime vs rest
    y_true_binary = (y_true == regime).astype(int)
    y_pred_binary = (y_pred == regime).astype(int)

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0.0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0.0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0.0)

    # Overall accuracy for this regime
    correct = (y_true_binary == y_pred_binary).sum()
    total = len(y_true_binary)
    accuracy = correct / total if total > 0 else 0.0

    # Support (actual count)
    support = y_true_binary.sum()

    # Confusion matrix for this regime
    true_positive = ((y_true == regime) & (y_pred == regime)).sum()
    false_positive = ((y_true != regime) & (y_pred == regime)).sum()
    false_negative = ((y_true == regime) & (y_pred != regime)).sum()
    true_negative = ((y_true != regime) & (y_pred != regime)).sum()

    return RegimeAccuracyMetrics(
        regime=regime,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
        support=int(support),
        confusion_matrix={
            'TP': int(true_positive),
            'FP': int(false_positive),
            'FN': int(false_negative),
            'TN': int(true_negative),
        }
    )


def calculate_transition_lag(
    df: pd.DataFrame,
    ground_truth_col: str = 'ground_truth_regime',
    predicted_col: str = 'hmm_predicted_regime',
) -> Tuple[Optional[float], Optional[float], List[int]]:
    """
    Calculate how long it takes HMM to detect regime changes.

    Returns:
        (avg_lag, median_lag, all_lags)
    """
    df = df.dropna(subset=[ground_truth_col, predicted_col])

    # Detect ground truth regime changes
    df['gt_regime_change'] = (df[ground_truth_col] != df[ground_truth_col].shift(1))

    lags = []
    for idx in df[df['gt_regime_change']].index:
        # Get new regime
        new_regime = df.loc[idx, ground_truth_col]

        # Find when HMM detected it
        future_df = df.loc[idx:]
        detected_mask = future_df[predicted_col] == new_regime
        if detected_mask.any():
            detected_idx = detected_mask.idxmax()
            lag = detected_idx - idx
            lags.append(lag)

    if not lags:
        return None, None, []

    avg_lag = float(np.mean(lags))
    median_lag = float(np.median(lags))

    return avg_lag, median_lag, lags


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(
    df: pd.DataFrame,
    ground_truth_col: str = 'ground_truth_regime',
    predicted_col: str = 'hmm_predicted_regime',
) -> RegimeDetectionReport:
    """Generate comprehensive regime detection report."""
    # Drop rows with missing values
    df_clean = df.dropna(subset=[ground_truth_col, predicted_col])

    if len(df_clean) == 0:
        return RegimeDetectionReport(
            overall_accuracy=0.0,
            regime_metrics=[],
            avg_transition_lag_days=None,
            median_transition_lag_days=None,
            confusion_matrix=pd.DataFrame(),
            passed=False,
            failure_reasons=["No data available for comparison"],
        )

    y_true = df_clean[ground_truth_col]
    y_pred = df_clean[predicted_col]

    # Overall accuracy
    overall_accuracy = (y_true == y_pred).mean()

    # Per-regime metrics
    regimes = ['BULLISH', 'NEUTRAL', 'BEARISH']
    regime_metrics = [
        calculate_metrics_per_regime(y_true, y_pred, regime)
        for regime in regimes
    ]

    # Confusion matrix
    cm = calculate_confusion_matrix(y_true, y_pred, regimes)

    # Transition lag
    avg_lag, median_lag, _ = calculate_transition_lag(df_clean, ground_truth_col, predicted_col)

    # Check if passed
    failure_reasons = []

    if overall_accuracy < 0.70:
        failure_reasons.append(f"Overall accuracy {overall_accuracy:.1%} < 70% threshold")

    for metric in regime_metrics:
        if metric.precision < 0.65:
            failure_reasons.append(
                f"{metric.regime} precision {metric.precision:.1%} < 65% threshold"
            )
        if metric.recall < 0.65:
            failure_reasons.append(
                f"{metric.regime} recall {metric.recall:.1%} < 65% threshold"
            )

    if avg_lag is not None and avg_lag > 10.0:
        failure_reasons.append(f"Average transition lag {avg_lag:.1f} days > 10 day threshold")

    passed = len(failure_reasons) == 0

    return RegimeDetectionReport(
        overall_accuracy=float(overall_accuracy),
        regime_metrics=regime_metrics,
        avg_transition_lag_days=avg_lag,
        median_transition_lag_days=median_lag,
        confusion_matrix=cm,
        passed=passed,
        failure_reasons=failure_reasons,
    )


def print_report(report: RegimeDetectionReport):
    """Pretty print regime detection report."""
    print("\n" + "=" * 80)
    print("REGIME DETECTION ACCURACY VERIFICATION")
    print("Jim Simons / Renaissance Technologies Standard")
    print("=" * 80)

    print(f"\nOverall Accuracy: {report.overall_accuracy:.1%}")

    print(f"\nPer-Regime Metrics:")
    print(f"  {'Regime':<10} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1':<10} {'Support'}")
    print(f"  {'-'*70}")
    for metric in report.regime_metrics:
        print(f"  {metric.regime:<10} {metric.accuracy:<10.1%} {metric.precision:<11.1%} "
              f"{metric.recall:<10.1%} {metric.f1_score:<10.2f} {metric.support}")

    print(f"\nConfusion Matrix:")
    print(report.confusion_matrix.to_string())

    print(f"\nTransition Lag:")
    if report.avg_transition_lag_days is not None:
        print(f"  Average: {report.avg_transition_lag_days:.1f} days")
        print(f"  Median: {report.median_transition_lag_days:.1f} days")
    else:
        print(f"  [WARN] No regime transitions detected")

    # Verdict
    print("\n" + "=" * 80)
    if report.passed:
        print("[OK] PASSED - Regime detection meets Jim Simons standard")
    else:
        print("[FAIL] NEEDS IMPROVEMENT:")
        for reason in report.failure_reasons:
            print(f"  - {reason}")
    print("=" * 80)


def save_report(report: RegimeDetectionReport):
    """Save report to file."""
    output_file = Path("reports/REGIME_DETECTION_VERIFICATION.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("# Regime Detection Accuracy Verification\n")
        f.write("**Jim Simons / Renaissance Technologies Standard**\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Status:** {'PASSED' if report.passed else 'NEEDS IMPROVEMENT'}\n\n")
        f.write("---\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Overall Accuracy:** {report.overall_accuracy:.1%}\n")
        if report.avg_transition_lag_days:
            f.write(f"- **Average Transition Lag:** {report.avg_transition_lag_days:.1f} days\n")
        f.write("\n---\n\n")

        f.write("## Per-Regime Metrics\n\n")
        f.write("| Regime | Accuracy | Precision | Recall | F1 | Support |\n")
        f.write("|--------|----------|-----------|--------|----|----------|\n")
        for metric in report.regime_metrics:
            f.write(f"| {metric.regime} | {metric.accuracy:.1%} | {metric.precision:.1%} | "
                   f"{metric.recall:.1%} | {metric.f1_score:.2f} | {metric.support} |\n")
        f.write("\n---\n\n")

        f.write("## Confusion Matrix\n\n")
        f.write("```\n")
        f.write(report.confusion_matrix.to_string())
        f.write("\n```\n\n---\n\n")

        if not report.passed:
            f.write("## Issues Detected\n\n")
            for reason in report.failure_reasons:
                f.write(f"- {reason}\n")
            f.write("\n---\n\n")

        f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Verification Standard:** Jim Simons / Renaissance Technologies\n")

    print(f"\n[OK] Report saved to {output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify HMM regime detection accuracy (Jim Simons standard)"
    )
    parser.add_argument(
        "--forward-days",
        type=int,
        default=60,
        help="Days forward for ground truth (default: 60)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/hmm_regime_v1.pkl",
        help="Path to HMM model",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date for analysis",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date for analysis",
    )

    args = parser.parse_args()

    # Load HMM model
    print(f"\nLoading HMM model from {args.model_path}...")
    detector = load_hmm_model(Path(args.model_path))

    # Fetch SPY data
    print(f"\nFetching SPY data ({args.start_date} to {args.end_date})...")
    spy_data = fetch_daily_bars_polygon(
        symbol='SPY',
        start=args.start_date,
        end=args.end_date,
        cache_dir=Path('data/cache/eod'),
    )

    if spy_data is None or len(spy_data) == 0:
        print("[ERROR] Failed to fetch SPY data")
        sys.exit(1)

    print(f"[OK] Loaded {len(spy_data)} days of SPY data")

    # Fetch VIX data (or use empty DataFrame as fallback)
    print(f"\nFetching VIX data...")
    try:
        vix_data = fetch_daily_bars_polygon(
            symbol='VIX',
            start=args.start_date,
            end=args.end_date,
            cache_dir=Path('data/cache/eod'),
        )
        print(f"[OK] Loaded {len(vix_data)} days of VIX data")
    except Exception as e:
        print(f"[WARN] Failed to fetch VIX data: {e}")
        print(f"[INFO] Using empty DataFrame (HMM will estimate from SPY volatility)")
        vix_data = pd.DataFrame()  # Empty DataFrame instead of None

    # Define ground truth
    print(f"\nDefining ground truth (forward_days={args.forward_days})...")
    spy_with_truth = define_ground_truth(spy_data, forward_days=args.forward_days)
    print(f"[OK] Ground truth defined")

    # Get HMM predictions
    print(f"\nGetting HMM predictions...")
    spy_with_predictions = get_hmm_predictions(detector, spy_with_truth, vix_data=vix_data)
    print(f"[OK] Predictions generated")

    # Generate report
    print(f"\nGenerating accuracy report...")
    report = generate_report(spy_with_predictions)

    # Print results
    print_report(report)

    # Save to file
    save_report(report)

    # Exit code
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
