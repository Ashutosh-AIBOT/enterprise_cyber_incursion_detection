from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from path_utils import ensure_dirs
from src.anomaly_models import run_anomaly_suite
from src.data_loader import load_nsl_kdd, make_binary_target
from src.explainability import lime_explanation_for_packet, permutation_importance_report
from src.preprocessing import split_and_prepare
from src.sampling_manager import SamplingManager
from src.supervised_models import train_supervised_suite


def run_training(sampling_strategy: str = "adasyn_tomek") -> dict:
    paths = ensure_dirs()
    raw_dir = paths["RAW"]
    models_dir = paths["MODELS"]
    charts_dir = paths["CHARTS"]

    df = load_nsl_kdd(raw_dir)
    X, y = make_binary_target(df)
    bundle = split_and_prepare(X, y)

    X_train_t = bundle.preprocessor.fit_transform(bundle.X_train)
    X_test_t = bundle.preprocessor.transform(bundle.X_test)

    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()
        X_test_t = X_test_t.toarray()

    sampler = SamplingManager(random_state=42)
    sampled = sampler.apply(X_train_t, bundle.y_train.values, strategy=sampling_strategy)

    anomaly_metrics = run_anomaly_suite(sampled.X_resampled, X_test_t, bundle.y_test.values)
    models, sup_metrics = train_supervised_suite(
        sampled.X_resampled,
        sampled.y_resampled,
        X_test_t,
        bundle.y_test.values,
    )

    for name, model in models.items():
        joblib.dump(model, models_dir / f"{name}.pkl")

    joblib.dump(bundle.preprocessor, models_dir / "preprocessor.pkl")

    best_supervised = max(sup_metrics.items(), key=lambda item: item[1]["roc_auc"])
    best_model_name, best_scores = best_supervised

    feature_names = [f"feature_{i}" for i in range(X_train_t.shape[1])]
    imp_df = permutation_importance_report(
        models[best_model_name],
        X_test_t,
        bundle.y_test.values,
        feature_names=feature_names,
    )
    imp_df.to_csv(charts_dir / "permutation_importance.csv", index=False)

    lime_payload = lime_explanation_for_packet(
        model=models[best_model_name],
        X_train=sampled.X_resampled,
        X_test=X_test_t,
        feature_names=feature_names,
        idx=0,
    )

    leaderboard_rows = []
    for model_name, m in sup_metrics.items():
        leaderboard_rows.append({"model": model_name, **m})

    for model_name, auc in anomaly_metrics.items():
        leaderboard_rows.append(
            {
                "model": model_name,
                "roc_auc": auc,
                "f1": None,
                "precision": None,
                "recall": None,
            }
        )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("roc_auc", ascending=False)
    leaderboard.to_csv(models_dir / "leaderboard.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.barh(leaderboard["model"], leaderboard["roc_auc"])
    plt.gca().invert_yaxis()
    plt.title("Project 11 - Model ROC-AUC Leaderboard")
    plt.xlabel("ROC-AUC")
    plt.tight_layout()
    plt.savefig(charts_dir / "model_roc_auc_leaderboard.png", dpi=180)
    plt.close()

    payload = {
        "sampling_strategy": sampling_strategy,
        "sampling_ratio": sampler.class_ratio(sampled.y_resampled)[2],
        "best_model": best_model_name,
        "best_model_scores": best_scores,
        "supervised_models": sup_metrics,
        "anomaly_models": anomaly_metrics,
        "lime": lime_payload,
    }

    with open(models_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


if __name__ == "__main__":
    result = run_training(sampling_strategy="adasyn_tomek")
    print(json.dumps(result, indent=2))
