from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


DEPLOYED_DIR = Path("state/models/deployed")
CANDIDATE_DIR = Path("state/models/candidates")


FEATURE_COLS = [
    'atr14', 'sma20_over_200', 'rv20', 'don20_width', 'pos_in_don20', 'ret5', 'log_vol'
]


def default_model() -> Pipeline:
    """
    Use GradientBoosting for better non-linear pattern detection.
    Parameters tuned to avoid overfitting on weak correlations:
    - n_estimators=100: moderate ensemble size
    - max_depth=3: shallow trees to prevent overfitting
    - min_samples_leaf=50: require substantial samples per leaf
    - subsample=0.8: add randomness for robustness
    """
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=50,
            subsample=0.8,
            learning_rate=0.1,
            random_state=42,
        )),
    ])


def model_paths(strategy: str, kind: str = 'candidate') -> Tuple[Path, Path]:
    base = DEPLOYED_DIR if kind == 'deployed' else CANDIDATE_DIR
    pkl = base / f"meta_{strategy.lower()}.pkl"
    meta = base / f"meta_{strategy.lower()}.json"
    return pkl, meta


def save_model(strategy: str, model: Pipeline, metadata: Dict, kind: str = 'candidate') -> None:
    base = DEPLOYED_DIR if kind == 'deployed' else CANDIDATE_DIR
    base.mkdir(parents=True, exist_ok=True)
    pkl, meta = model_paths(strategy, kind=kind)
    joblib.dump(model, pkl)
    meta.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def load_model(strategy: str) -> Optional[Pipeline]:
    # Prefer deployed models; fallback to candidates if not deployed yet
    pkl, _ = model_paths(strategy, kind='deployed')
    if not pkl.exists():
        pkl, _ = model_paths(strategy, kind='candidate')
        if not pkl.exists():
            return None
    return joblib.load(pkl)


def predict_proba(model: Pipeline, df_features: pd.DataFrame) -> np.ndarray:
    x = df_features.reindex(columns=FEATURE_COLS, fill_value=0.0)
    try:
        proba = model.predict_proba(x)[:, 1]
    except Exception:
        # If model lacks predict_proba (unlikely for LR), fallback
        preds = model.predict(x)
        proba = np.clip(preds.astype(float), 0.0, 1.0)
    return proba
