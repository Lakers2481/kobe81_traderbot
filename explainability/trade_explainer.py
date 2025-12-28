"""
Trade Explainer - Generating Human-Readable Trade Rationales
============================================================

This module provides a helper function to build a human-readable explanation
for a given trading signal. It constructs a narrative by combining:
- A pre-defined, strategy-specific description.
- The signal's own "reason" field, if present.
- A calculated risk/reward ratio.
- (Optionally) An analysis of the top contributing features from a machine
  learning model's coefficients.

It is designed to fail gracefully, providing a useful explanation even when
complex model-specific details like SHAP values or coefficients are unavailable.

Usage:
    from explainability.trade_explainer import explain_trade

    signal = {'strategy': 'ibs_rsi', 'symbol': 'AAPL', ...}
    explanation = explain_trade(signal)
    print(explanation['narrative'])
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import pandas as pd


@dataclass
class ExplainConfig:
    """Configuration for the trade explainer."""
    top_k: int = 5 # The number of top features to show in the explanation.


def _strategy_from_signal(sig: Dict[str, Any]) -> str:
    """Helper function to normalize the strategy name from a signal dictionary."""
    s = str(sig.get("strategy", "")).lower()
    if "ibs" in s or "rsi" in s:
        return "ibs_rsi"
    if "turtle" in s or "ict" in s:
        return "turtle_soup"
    return s or "unknown"


def _coef_explain(model: Any, xrow: pd.Series, top_k: int) -> List[Dict[str, Any]]:
    """
    Attempts to explain a prediction by multiplying model coefficients by feature values.
    This is a simple proxy for feature importance, suitable for linear models.

    Args:
        model: The trained model object.
        xrow: A pandas Series representing the feature row for the prediction.
        top_k: The number of top contributing features to return.

    Returns:
        A list of dictionaries, each containing a feature and its calculated contribution.
        Returns an empty list if coefficients cannot be extracted.
    """
    try:
        from sklearn.pipeline import Pipeline
        
        # --- Attempt to extract the core estimator from various model wrappers ---
        est = None
        # Handle CalibratedClassifierCV
        if hasattr(model, 'calibrated_classifiers_') and model.calibrated_classifiers_:
            cc = model.calibrated_classifiers_[0]
            est = getattr(cc, 'estimator', None) or getattr(cc, 'base_estimator', None)
        # Handle other wrappers like VotingClassifier
        elif hasattr(model, 'base_estimator'):
            est = model.base_estimator
        
        # If the estimator is a scikit-learn Pipeline, find the classifier step.
        if isinstance(est, Pipeline) and 'clf' in est.named_steps:
            clf = est.named_steps['clf']
            # Check if the classifier has coefficients (like LogisticRegression or an SVM).
            if hasattr(clf, 'coef_'):
                coef = clf.coef_.flatten()
                # Calculate contribution: feature_value * coefficient_weight
                contrib = [(f, float(c) * float(xrow.get(f, 0.0))) for f, c in zip(xrow.index, coef)]
                contrib.sort(key=lambda t: abs(t[1]), reverse=True)
                return [{"feature": f, "contribution": round(v, 6)} for f, v in contrib[:top_k]]
    except Exception:
        # Fail gracefully if any step of the coefficient extraction fails.
        pass
    return []


def explain_trade(
    signal: Dict[str, Any],
    features_row: Optional[pd.Series] = None,
    model: Optional[Any] = None,
    cfg: Optional[ExplainConfig] = None
) -> Dict[str, Any]:
    """
    Constructs a comprehensive, human-readable explanation for a trading signal.

    Args:
        signal: The signal dictionary to explain.
        features_row: (Optional) The row of features corresponding to the signal.
        model: (Optional) The ML model used to generate the signal's confidence.
        cfg: (Optional) Configuration for the explanation.

    Returns:
        A dictionary containing the 'summary' and a detailed 'narrative'.
    """
    cfg = cfg or ExplainConfig()
    narrative_parts = []
    
    # --- Extract key signal details ---
    strategy_name = _strategy_from_signal(signal)
    symbol = signal.get("symbol", "?")
    entry = signal.get("entry_price")
    stop = signal.get("stop_loss")
    take_profit = signal.get("take_profit")
    side = signal.get("side", "?").upper()
    reason = signal.get("reason")

    # --- Build the narrative step-by-step ---

    # 1. Add a canned description based on the strategy name.
    if strategy_name == "ibs_rsi":
        narrative_parts.append("This is an IBS+RSI mean-reversion signal, which identifies a deep close within the bar combined with an oversold RSI(2) reading during a general uptrend.")
    elif strategy_name == "turtle_soup":
        narrative_parts.append("This is an ICT Turtle Soup signal, which looks for a liquidity sweep and reversal pattern near a recent high or low.")
    else:
        narrative_parts.append("This signal was generated by a custom ruleset.")

    # 2. Append any specific reason provided in the signal itself.
    if reason:
        narrative_parts.append(f"The generating rule specifically noted: '{reason}'.")

    # 3. Calculate and add the Risk/Reward ratio.
    if entry and stop and take_profit:
        try:
            risk_per_share = abs(float(entry) - float(stop))
            reward_per_share = abs(float(take_profit) - float(entry))
            rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0
            narrative_parts.append(f"The trade has a Risk/Reward ratio of approximately {rr_ratio:.2f}R.")
        except (ValueError, TypeError):
            pass # Ignore if prices are not valid numbers.

    # 4. If a model and features are provided, attempt to explain feature contributions.
    feature_contributions: List[Dict[str, Any]] = []
    if features_row is not None and model is not None:
        feature_contributions = _coef_explain(model, features_row, cfg.top_k)
        if feature_contributions:
            narrative_parts.append("The model's decision was influenced by the following features (estimated contribution = feature value × coefficient):")
            for c in feature_contributions:
                narrative_parts.append(f"  - {c['feature']}: {c['contribution']:+.4f}")

    # 5. Assemble the final output dictionary.
    summary = f"Signal to go {side} on {symbol} at ${entry}" if entry else f"Signal for {symbol} {side}"
    
    return {
        "summary": summary,
        "narrative": " ".join(narrative_parts),
        "contributions": feature_contributions,
        "strategy": strategy_name,
    }
