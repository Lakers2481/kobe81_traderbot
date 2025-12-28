from __future__ import annotations

"""
Feature importance utilities (optional).

Supports permutation importance for any sklearn-compatible model; attempts SHAP
if available but degrades gracefully.
"""

from typing import Optional
import pandas as pd


def permutation_importance(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5, random_state: int = 42) -> pd.DataFrame:
    try:
        from sklearn.inspection import permutation_importance as _pi
        res = _pi(model, X, y, n_repeats=n_repeats, random_state=random_state)
        imp = pd.DataFrame({
            "feature": list(X.columns),
            "importances_mean": res.importances_mean,
            "importances_std": res.importances_std,
        }).sort_values("importances_mean", ascending=False).reset_index(drop=True)
        return imp
    except Exception:
        return pd.DataFrame(columns=["feature","importances_mean","importances_std"])


def shap_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    try:
        import shap
        explainer = shap.Explainer(model.predict_proba, X)
        sv = explainer(X)
        # Mean abs shap for class 1 if available, else aggregate
        import numpy as np
        vals = sv.values
        if vals.ndim == 3 and vals.shape[2] >= 2:
            arr = np.abs(vals[..., 1]).mean(axis=0)
        else:
            arr = np.abs(vals).mean(axis=0)
        df = pd.DataFrame({"feature": list(X.columns), "mean_abs_shap": arr}).sort_values("mean_abs_shap", ascending=False)
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["feature","mean_abs_shap"])

