from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except Exception:  # pragma: no cover
    LIME_AVAILABLE = False


def permutation_importance_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 7,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )
    imp = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return imp


def lime_explanation_for_packet(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    idx: int = 0,
) -> Dict[str, object]:
    if not LIME_AVAILABLE:
        return {
            "status": "lime_not_available",
            "message": "Install 'lime' to generate local packet explanations.",
            "features": [],
        }

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["normal", "attack"],
        discretize_continuous=True,
        mode="classification",
    )

    explanation = explainer.explain_instance(
        X_test[idx],
        model.predict_proba,
        num_features=min(12, len(feature_names)),
    )
    items = [{"rule": r, "weight": float(w)} for r, w in explanation.as_list()]
    return {"status": "ok", "features": items}
