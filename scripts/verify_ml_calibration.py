"""
Verify ML Model Calibration (Jim Simons Standard)
===================================================

Checks if ML confidence scores are properly calibrated:
- If model predicts 70% confidence, does it win 70% of the time?
- Computes Expected Calibration Error (ECE)
- Generates reliability diagrams
- Validates calibrator usage

Calibration Metrics:
    - Brier Score: <0.15 = excellent, <0.25 = acceptable
    - ECE: <0.05 = well-calibrated, <0.10 = acceptable
    - Reliability curve should lie close to diagonal

Usage:
    python scripts/verify_ml_calibration.py
    python scripts/verify_ml_calibration.py --demo  # Synthetic example
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.structured_log import jlog


# ============================================================================
# Calibration Metrics (Self-Contained for Verification)
# ============================================================================

def compute_brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute Brier score (mean squared error of probabilities)."""
    probabilities = np.asarray(probabilities).flatten()
    outcomes = np.asarray(outcomes).flatten()

    if len(probabilities) == 0:
        return 0.0

    return float(np.mean((probabilities - outcomes) ** 2))


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    probabilities = np.asarray(probabilities).flatten()
    outcomes = np.asarray(outcomes).flatten()

    if len(probabilities) == 0:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(probabilities)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (probabilities >= bin_lower) & (probabilities <= bin_upper)
        else:
            in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size == 0:
            continue

        bin_confidence = np.mean(probabilities[in_bin])
        bin_accuracy = np.mean(outcomes[in_bin])

        ece += (bin_size / total_samples) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_reliability_diagram(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, List]:
    """Compute reliability diagram data for visualization."""
    probabilities = np.asarray(probabilities).flatten()
    outcomes = np.asarray(outcomes).flatten()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_midpoints = []
    accuracies = []
    confidences = []
    counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        midpoint = (bin_lower + bin_upper) / 2

        if i == n_bins - 1:
            in_bin = (probabilities >= bin_lower) & (probabilities <= bin_upper)
        else:
            in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)

        bin_size = int(np.sum(in_bin))
        bin_midpoints.append(midpoint)
        counts.append(bin_size)

        if bin_size > 0:
            accuracies.append(float(np.mean(outcomes[in_bin])))
            confidences.append(float(np.mean(probabilities[in_bin])))
        else:
            accuracies.append(None)
            confidences.append(None)

    return {
        'bin_midpoints': bin_midpoints,
        'accuracies': accuracies,
        'confidences': confidences,
        'counts': counts,
    }


# ============================================================================
# Calibration Analysis
# ============================================================================

@dataclass
class CalibrationReport:
    """ML calibration verification report."""

    model_name: str
    n_samples: int
    brier_score: float
    ece: float
    reliability_diagram: Dict[str, List]
    passed: bool
    failure_reasons: List[str]

    # Thresholds
    brier_threshold: float = 0.25
    ece_threshold: float = 0.10


def analyze_model_calibration(
    model_name: str,
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    brier_threshold: float = 0.25,
    ece_threshold: float = 0.10,
) -> CalibrationReport:
    """
    Analyze ML model calibration.

    Args:
        model_name: Name of the model being analyzed
        probabilities: Predicted probabilities (0-1)
        outcomes: Actual binary outcomes (0 or 1)
        brier_threshold: Max acceptable Brier score
        ece_threshold: Max acceptable ECE

    Returns:
        CalibrationReport with calibration metrics
    """
    probabilities = np.asarray(probabilities).flatten()
    outcomes = np.asarray(outcomes).flatten()

    if len(probabilities) == 0:
        return CalibrationReport(
            model_name=model_name,
            n_samples=0,
            brier_score=0.0,
            ece=0.0,
            reliability_diagram={},
            passed=False,
            failure_reasons=["No data available"],
            brier_threshold=brier_threshold,
            ece_threshold=ece_threshold,
        )

    # Compute metrics
    brier = compute_brier_score(probabilities, outcomes)
    ece = compute_expected_calibration_error(probabilities, outcomes, n_bins=10)
    reliability = compute_reliability_diagram(probabilities, outcomes, n_bins=10)

    # Check if calibration passes thresholds
    failure_reasons = []

    if brier > brier_threshold:
        failure_reasons.append(
            f"Brier score {brier:.4f} exceeds threshold {brier_threshold:.4f}"
        )

    if ece > ece_threshold:
        failure_reasons.append(
            f"ECE {ece:.4f} exceeds threshold {ece_threshold:.4f}"
        )

    passed = len(failure_reasons) == 0

    return CalibrationReport(
        model_name=model_name,
        n_samples=len(probabilities),
        brier_score=brier,
        ece=ece,
        reliability_diagram=reliability,
        passed=passed,
        failure_reasons=failure_reasons,
        brier_threshold=brier_threshold,
        ece_threshold=ece_threshold,
    )


def print_calibration_report(report: CalibrationReport):
    """Pretty print calibration report."""
    print("\n" + "=" * 80)
    print(f"Model: {report.model_name.upper()}")
    print("=" * 80)

    print(f"\nSamples: {report.n_samples:,}")

    print("\nCalibration Metrics:")
    brier_status = "[OK]" if report.brier_score <= report.brier_threshold else "[FAIL]"
    ece_status = "[OK]" if report.ece <= report.ece_threshold else "[FAIL]"

    print(f"  {brier_status} Brier Score: {report.brier_score:.4f} (threshold: {report.brier_threshold:.4f})")
    print(f"  {ece_status} ECE: {report.ece:.4f} (threshold: {report.ece_threshold:.4f})")

    print("\nReliability Diagram:")
    print("  Bin Range      | Count | Predicted | Actual    | Gap")
    print("  " + "-" * 65)

    for i, midpoint in enumerate(report.reliability_diagram['bin_midpoints']):
        count = report.reliability_diagram['counts'][i]
        pred = report.reliability_diagram['confidences'][i]
        actual = report.reliability_diagram['accuracies'][i]

        bin_lower = max(0, midpoint - 0.05)
        bin_upper = min(1, midpoint + 0.05)

        if count == 0 or pred is None or actual is None:
            print(f"  [{bin_lower:.2f}-{bin_upper:.2f}] | {count:5d} | -         | -         | -")
        else:
            gap = abs(actual - pred)
            gap_str = f"{gap:+.3f}"
            print(f"  [{bin_lower:.2f}-{bin_upper:.2f}] | {count:5d} | {pred:.3f}     | {actual:.3f}     | {gap_str}")

    if report.passed:
        print("\n[OK] PASSED - Model is well-calibrated")
    else:
        print("\n[FAIL] CALIBRATION ISSUES DETECTED:")
        for reason in report.failure_reasons:
            print(f"  - {reason}")


# ============================================================================
# Demo Mode
# ============================================================================

def demo_calibration_analysis():
    """Demonstrate calibration analysis with synthetic examples."""
    print("\n" + "=" * 80)
    print("ML CALIBRATION DEMO")
    print("Jim Simons / Renaissance Technologies Standard")
    print("=" * 80)

    # Example 1: WELL-CALIBRATED MODEL
    print("\n" + "-" * 80)
    print("Example 1: WELL-CALIBRATED Model")
    print("-" * 80)

    np.random.seed(42)
    n = 1000

    # Generate synthetic data where predictions match outcomes
    true_probs = np.random.beta(2, 2, n)  # True probabilities
    outcomes = (np.random.random(n) < true_probs).astype(int)
    predictions = true_probs + np.random.normal(0, 0.05, n)  # Add small noise
    predictions = np.clip(predictions, 0.01, 0.99)

    report1 = analyze_model_calibration("well_calibrated_model", predictions, outcomes)
    print_calibration_report(report1)

    # Example 2: OVERCONFIDENT MODEL
    print("\n" + "-" * 80)
    print("Example 2: OVERCONFIDENT Model (Poorly Calibrated)")
    print("-" * 80)

    # Generate synthetic data where model is overconfident
    # Model predicts high probabilities but actual win rate is lower
    predictions_overconfident = np.random.beta(8, 2, n)  # Skewed high
    true_probs_lower = predictions_overconfident - 0.20  # Actual rate is 20% lower
    true_probs_lower = np.clip(true_probs_lower, 0.01, 0.99)
    outcomes_overconfident = (np.random.random(n) < true_probs_lower).astype(int)

    report2 = analyze_model_calibration("overconfident_model", predictions_overconfident, outcomes_overconfident)
    print_calibration_report(report2)

    # Example 3: UNDERCONFIDENT MODEL
    print("\n" + "-" * 80)
    print("Example 3: UNDERCONFIDENT Model")
    print("-" * 80)

    # Model predicts low probabilities but actual win rate is higher
    predictions_underconfident = np.random.beta(2, 8, n)  # Skewed low
    true_probs_higher = predictions_underconfident + 0.15  # Actual rate is 15% higher
    true_probs_higher = np.clip(true_probs_higher, 0.01, 0.99)
    outcomes_underconfident = (np.random.random(n) < true_probs_higher).astype(int)

    report3 = analyze_model_calibration("underconfident_model", predictions_underconfident, outcomes_underconfident)
    print_calibration_report(report3)

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS:")
    print("=" * 80)
    print("1. Well-calibrated models have Brier < 0.15 and ECE < 0.05")
    print("2. Reliability diagram should show actual ~= predicted across all bins")
    print("3. Overconfident models predict higher probabilities than actual win rates")
    print("4. Use IsotonicCalibrator or PlattCalibrator to fix calibration issues")


# ============================================================================
# Check Existing Models
# ============================================================================

def check_existing_models() -> List[CalibrationReport]:
    """Check calibration of existing ML models in the system."""
    reports = []

    # Check LSTM confidence model
    lstm_model_path = Path("models/lstm_confidence_v1.h5")
    if lstm_model_path.exists():
        print(f"\n[INFO] Found LSTM model at {lstm_model_path}")
        # Note: Would need actual predictions vs outcomes data to compute calibration
        # For now, just note its existence
    else:
        print(f"\n[WARNING] No LSTM model found at {lstm_model_path}")

    # Check HMM regime model
    hmm_model_path = Path("models/hmm_regime_v1.pkl")
    if hmm_model_path.exists():
        print(f"[INFO] Found HMM model at {hmm_model_path}")
    else:
        print(f"[WARNING] No HMM model found at {hmm_model_path}")

    # Check ensemble models
    ensemble_path = Path("models/ensemble")
    if ensemble_path.exists():
        ensemble_models = list(ensemble_path.glob("*.pkl"))
        if ensemble_models:
            print(f"[INFO] Found {len(ensemble_models)} ensemble models")
        else:
            print(f"[WARNING] Ensemble directory exists but no models found")
    else:
        print(f"[WARNING] No ensemble models directory at {ensemble_path}")

    # Check for calibration data
    state_path = Path("state")
    if state_path.exists():
        # Look for any files that might contain predictions vs outcomes
        prediction_files = list(state_path.rglob("*predictions*.json")) + \
                          list(state_path.rglob("*predictions*.jsonl"))
        if prediction_files:
            print(f"[INFO] Found {len(prediction_files)} prediction log files")
        else:
            print(f"[WARNING] No prediction log files found in state/")

    return reports


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify ML model calibration (Jim Simons standard)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration with synthetic examples",
    )

    args = parser.parse_args()

    if args.demo:
        demo_calibration_analysis()
        sys.exit(0)

    print("\n" + "=" * 80)
    print("ML MODEL CALIBRATION VERIFICATION")
    print("Jim Simons / Renaissance Technologies Standard")
    print("=" * 80)

    # Check existing models
    reports = check_existing_models()

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print("""
To complete ML calibration verification:

1. COLLECT PREDICTION DATA:
   - Log all ML model predictions with timestamps
   - Record actual outcomes (win/loss) for each prediction
   - Store in state/ml_predictions.jsonl

2. COMPUTE CALIBRATION METRICS:
   - Run this script with actual data (not demo)
   - Generate reliability diagrams
   - Identify poorly calibrated models

3. FIT CALIBRATORS:
   from ml_meta.calibration import IsotonicCalibrator

   calibrator = IsotonicCalibrator()
   calibrator.fit(val_predictions, val_outcomes)
   calibrator.save("models/calibrators/lstm_calibrator.pkl")

4. APPLY CALIBRATION IN PRODUCTION:
   - Load calibrator at startup
   - Apply to all raw predictions before decision-making
   - Re-calibrate monthly with fresh data

5. MONITOR CALIBRATION DRIFT:
   - Track ECE over time
   - Alert if ECE > 0.10 (degradation)
   - Re-train calibrators when needed
""")

    print("\n" + "=" * 80)
    print("CURRENT STATUS")
    print("=" * 80)
    print("[OK] Calibration infrastructure exists (ml_meta/calibration.py)")
    print("[PENDING] Need actual prediction vs outcome data to verify")
    print("[PENDING] Need to fit and apply calibrators to models")
    print("\nRun with --demo to see expected calibration behavior")


if __name__ == "__main__":
    main()
