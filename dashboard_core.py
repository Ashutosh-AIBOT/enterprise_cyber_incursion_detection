import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


SUPPORT_ARTIFACT_HINTS = {
    "scaler",
    "transformer",
    "encoder",
    "feature_columns",
    "class_names",
    "latest_data",
    "metrics",
    "residuals",
}

TARGET_CANDIDATES = {
    "target",
    "label",
    "class",
    "default",
    "readmitted",
    "attrition",
    "is_fraud",
    "churn",
    "failure",
    "y",
}


@dataclass
class ProjectConfig:
    key: str
    title: str
    subtitle: str
    icon: str
    domain: str
    objective: str
    business_value: str
    prediction_label: str
    highlights: List[str]


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #0d0d0d;
          --panel: #141414;
          --panel-2: #1f1f1f;
          --text: #f5f5f5;
          --muted: #bfbfbf;
          --line: #2f2f2f;
          --accent: #d9d9d9;
        }
        .stApp {
          background: radial-gradient(circle at top right, #1b1b1b 0%, #0d0d0d 45%);
          color: var(--text);
        }
        h1, h2, h3, h4, h5 { color: #ffffff !important; letter-spacing: 0.2px; }
        p, li, label, .stCaption, .stMarkdown, .stText { color: var(--muted) !important; }
        .block-container { padding-top: 1.2rem !important; padding-bottom: 2rem !important; }
        [data-testid="stMetric"] {
          background: linear-gradient(160deg, var(--panel) 0%, var(--panel-2) 100%);
          border: 1px solid var(--line);
          border-radius: 12px;
          padding: 8px;
        }
        .app-card {
          border: 1px solid var(--line);
          border-radius: 14px;
          padding: 16px;
          background: linear-gradient(140deg, var(--panel) 0%, #181818 100%);
          margin-bottom: 12px;
        }
        .small-note {
          font-size: 0.9rem;
          color: #b2b2b2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_pickle(path: str) -> Any:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _safe_listdir(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    try:
        return sorted(os.listdir(path))
    except Exception:
        return []


def _find_table_file(models_dir: str) -> Optional[str]:
    for name in ["performance_metrics.csv", "final_model_comparison.csv", "regression_metrics.csv"]:
        p = os.path.join(models_dir, name)
        if os.path.exists(p):
            return p
    return None


def _load_results_table(models_dir: str) -> pd.DataFrame:
    results_json = os.path.join(models_dir, "results.json")
    table = pd.DataFrame()

    if os.path.exists(results_json):
        payload = _read_json(results_json)
        if isinstance(payload, dict):
            rows = []
            for model_name, vals in payload.items():
                row = {"Model": model_name}
                if isinstance(vals, dict):
                    row.update(vals)
                rows.append(row)
            table = pd.DataFrame(rows)

    table_file = _find_table_file(models_dir)
    if table_file:
        try:
            extra = pd.read_csv(table_file)
            if table.empty:
                table = extra
            elif "Model" in extra.columns and "Model" in table.columns:
                table = pd.concat([table, extra], ignore_index=True).drop_duplicates(subset=["Model"], keep="first")
        except Exception:
            pass

    if not table.empty:
        metric_cols = [c for c in table.columns if c.lower() not in {"model", "name"}]
        table = table.rename(columns={"name": "Model"})
        if "Model" not in table.columns:
            table.insert(0, "Model", [f"Model {i+1}" for i in range(len(table))])
        for col in metric_cols:
            if table[col].dtype == object:
                table[col] = pd.to_numeric(table[col], errors="ignore")

    return table


def _extract_model(model_blob: Any) -> Any:
    if isinstance(model_blob, dict):
        for k in ["model", "estimator", "pipeline", "clf", "regressor"]:
            if k in model_blob:
                return model_blob[k]
    return model_blob


def _display_name_from_file(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    return base.replace("_", " ").replace("-", " ").title()


def _load_model_registry(models_dir: str) -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    for file_name in _safe_listdir(models_dir):
        if not file_name.endswith(".pkl"):
            continue
        path = os.path.join(models_dir, file_name)
        obj = _load_pickle(path)
        display = _display_name_from_file(file_name)
        if obj is None:
            registry[display] = {
                "file_name": file_name,
                "model": None,
                "predictable": False,
                "status": "Load failed",
            }
            continue

        extracted = _extract_model(obj)
        predictable = hasattr(extracted, "predict")
        lower_name = file_name.lower()
        support_like = any(h in lower_name for h in SUPPORT_ARTIFACT_HINTS)
        status = "Predict-ready"
        if not predictable:
            status = "Artifact only"
        elif support_like:
            status = "Predict-ready (support artifact name)"

        registry[display] = {
            "file_name": file_name,
            "model": extracted,
            "predictable": predictable,
            "status": status,
        }
    return registry


def _load_models(models_dir: str) -> Dict[str, Any]:
    registry = _load_model_registry(models_dir)
    return {name: info["model"] for name, info in registry.items() if info["predictable"]}


def _load_feature_columns(project_dir: str, models_dir: str, model: Any) -> List[str]:
    expected_count = None
    if hasattr(model, "n_features_in_"):
        try:
            expected_count = int(model.n_features_in_)
        except Exception:
            expected_count = None

    feature_file = os.path.join(models_dir, "feature_columns.pkl")
    if os.path.exists(feature_file):
        feats = _load_pickle(feature_file)
        if isinstance(feats, (list, tuple)):
            feat_list = [str(x) for x in feats]
            if expected_count and len(feat_list) >= expected_count:
                return feat_list[:expected_count]
            return feat_list

    if hasattr(model, "feature_names_in_"):
        try:
            return [str(x) for x in list(model.feature_names_in_)]
        except Exception:
            pass
    if hasattr(model, "feature_names_"):
        try:
            names = list(model.feature_names_)
            if names:
                return [str(x) for x in names]
        except Exception:
            pass

    data_dir = os.path.join(project_dir, "data")
    csv_candidates: List[str] = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_candidates.append(os.path.join(root, f))
    csv_candidates.sort()

    for csv_path in csv_candidates[:5]:
        try:
            df = pd.read_csv(csv_path, nrows=200)
        except Exception:
            continue
        if df.empty:
            continue
        cols = [
            c
            for c in df.columns
            if str(c).strip().lower() not in TARGET_CANDIDATES and not str(c).lower().startswith("unnamed")
        ]
        if cols:
            feat_list = [str(c) for c in cols]
            if expected_count:
                if len(feat_list) >= expected_count:
                    return feat_list[:expected_count]
                for i in range(len(feat_list), expected_count):
                    feat_list.append(f"feature_{i}")
                return feat_list
            return feat_list[:25]

    return []


def _top_metric(results_df: pd.DataFrame) -> Tuple[str, str]:
    if results_df.empty:
        return "No benchmark yet", "-"

    numeric_cols = []
    for col in results_df.columns:
        if col == "Model":
            continue
        if pd.api.types.is_numeric_dtype(results_df[col]):
            numeric_cols.append(col)

    if not numeric_cols:
        return "Benchmarks loaded", str(len(results_df))

    ranked_metric = None
    for pref in ["auroc", "accuracy", "f1", "r2", "mae", "rmse", "mse"]:
        for col in numeric_cols:
            if pref in col.lower():
                ranked_metric = col
                break
        if ranked_metric:
            break

    if ranked_metric is None:
        ranked_metric = numeric_cols[0]

    ascending = any(k in ranked_metric.lower() for k in ["mae", "rmse", "mse", "loss", "error"])
    ranked = results_df.sort_values(ranked_metric, ascending=ascending)
    best = ranked.iloc[0]
    return str(best.get("Model", "Best model")), f"{ranked_metric}: {best[ranked_metric]:.4f}"


def _choose_rank_metric(results_df: pd.DataFrame) -> Optional[str]:
    if results_df.empty:
        return None
    numeric_cols = [c for c in results_df.columns if c != "Model" and pd.api.types.is_numeric_dtype(results_df[c])]
    if not numeric_cols:
        return None
    for pref in ["auroc", "roc_auc", "accuracy", "f1", "recall", "precision", "r2", "mae", "rmse", "mse"]:
        for col in numeric_cols:
            if pref in col.lower():
                return col
    return numeric_cols[0]


def _rank_models(results_df: pd.DataFrame) -> pd.DataFrame:
    ranked = results_df.copy()
    metric = _choose_rank_metric(ranked)
    if metric is None:
        ranked.insert(0, "Rank", range(1, len(ranked) + 1))
        return ranked
    ascending = any(k in metric.lower() for k in ["mae", "rmse", "mse", "loss", "error"])
    ranked = ranked.sort_values(metric, ascending=ascending).reset_index(drop=True)
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    return ranked


def _render_model_registry(registry: Dict[str, Dict[str, Any]], results_df: pd.DataFrame) -> None:
    st.subheader("All Trained Model Artifacts")
    if not registry:
        st.info("No model artifacts found in `models/`.")
        return

    result_names = set()
    if not results_df.empty and "Model" in results_df.columns:
        result_names = {str(x).strip().lower() for x in results_df["Model"].astype(str).tolist()}

    rows = []
    for name, info in registry.items():
        rows.append(
            {
                "Model": name,
                "File": info["file_name"],
                "Predictable": "Yes" if info["predictable"] else "No",
                "Status": info["status"],
                "In Ranking Table": "Yes" if name.strip().lower() in result_names else "No",
            }
        )

    registry_df = pd.DataFrame(rows).sort_values(["Predictable", "Model"], ascending=[False, True])
    st.dataframe(registry_df, use_container_width=True)


def _render_overview(
    cfg: ProjectConfig,
    model_registry: Dict[str, Dict[str, Any]],
    results_df: pd.DataFrame,
    chart_count: int,
    notebook_count: int,
) -> None:
    best_model, best_metric = _top_metric(results_df)
    predictable_models = sum(1 for info in model_registry.values() if info["predictable"])
    total_artifacts = len(model_registry)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Artifacts", total_artifacts)
    c2.metric("Charts Available", chart_count)
    c3.metric("Notebooks", notebook_count)
    c4.metric("Top Performer", best_model)
    c5, c6 = st.columns(2)
    c5.metric("Predict-Ready Models", predictable_models)
    c6.metric("Support Artifacts", max(total_artifacts - predictable_models, 0))

    st.markdown(
        f"""
        <div class='app-card'>
          <h3>{cfg.title}</h3>
          <p><b>Domain:</b> {cfg.domain}</p>
          <p><b>Objective:</b> {cfg.objective}</p>
          <p><b>Business Value:</b> {cfg.business_value}</p>
          <p class='small-note'><b>Current Top Metric:</b> {best_metric}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_results(results_df: pd.DataFrame) -> None:
    st.subheader("Model Ranking and Benchmark")
    if results_df.empty:
        st.info("No benchmark file found yet. Add `results.json` or metrics CSV in the `models/` folder.")
        return

    ranked_df = _rank_models(results_df)
    st.dataframe(ranked_df, use_container_width=True)

    numeric_cols = [c for c in results_df.columns if c != "Model" and pd.api.types.is_numeric_dtype(results_df[c])]
    if numeric_cols:
        default_metric = _choose_rank_metric(results_df)
        default_ix = numeric_cols.index(default_metric) if default_metric in numeric_cols else 0
        metric = st.selectbox("Metric for leaderboard chart", numeric_cols, index=default_ix)

        chart_df = ranked_df[["Model", metric]].dropna().set_index("Model")
        if not chart_df.empty:
            st.bar_chart(chart_df)

        if len(numeric_cols) > 1:
            st.caption("Metric trend by rank")
            trend_df = ranked_df.set_index("Rank")[numeric_cols].dropna(how="all")
            if not trend_df.empty:
                st.line_chart(trend_df)


def _make_single_row_form(feature_cols: List[str]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for col in feature_cols:
        col_l = col.lower()
        if any(k in col_l for k in ["id", "code"]):
            values[col] = st.text_input(col, value="0")
        elif any(k in col_l for k in ["date", "month", "day", "year"]):
            values[col] = st.number_input(col, value=1, step=1)
        elif any(k in col_l for k in ["is_", "flag", "bool"]):
            values[col] = int(st.selectbox(col, [0, 1], key=f"{col}_flag"))
        else:
            values[col] = st.number_input(col, value=0.0, step=0.1)
    return values


def _predict_with_model(model: Any, input_df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    preds = model.predict(input_df)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(input_df)
        except Exception:
            proba = None
    return np.asarray(preds), None if proba is None else np.asarray(proba)


def _render_predict(
    project_dir: str,
    models_dir: str,
    model_registry: Dict[str, Dict[str, Any]],
    cfg: ProjectConfig,
) -> None:
    st.subheader(f"{cfg.prediction_label} Center")

    if not model_registry:
        st.warning("No model artifacts found in `models/`.")
        return

    ordered_names = sorted(model_registry.keys())
    model_name = st.selectbox("Choose any discovered model artifact", ordered_names)
    selected = model_registry[model_name]
    model = selected["model"]

    st.caption(f"Artifact: `{selected['file_name']}` | Status: **{selected['status']}**")
    if not selected["predictable"] or model is None:
        st.warning("This artifact is not prediction-capable. Select another model to run prediction.")
        return

    feature_cols = _load_feature_columns(project_dir, models_dir, model)

    if not feature_cols:
        st.warning("Feature columns are unavailable. Add `models/feature_columns.pkl` to unlock manual input UI.")

    mode = st.radio("Prediction mode", ["Single Record", "Batch CSV"], horizontal=True)

    if mode == "Single Record":
        if not feature_cols:
            st.info("Switch to Batch CSV mode, or provide feature columns metadata.")
            return

        with st.form("single_prediction_form"):
            values = _make_single_row_form(feature_cols)
            run = st.form_submit_button("Run Prediction", type="primary")

        if run:
            X = pd.DataFrame([values])
            try:
                preds, proba = _predict_with_model(model, X)
                st.success(f"Prediction: **{preds[0]}**")
                if proba is not None and proba.shape[0] > 0:
                    probs = proba[0]
                    st.write("Class probabilities")
                    st.dataframe(pd.DataFrame({"class_index": range(len(probs)), "probability": probs}), use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    if mode == "Batch CSV":
        st.caption("Upload CSV containing model features. Extra columns are ignored; missing columns cause validation error.")
        file = st.file_uploader("Upload batch input CSV", type=["csv"])
        if file is None:
            return

        try:
            batch = pd.read_csv(file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return

        required = feature_cols if feature_cols else list(batch.columns)
        missing = [c for c in required if c not in batch.columns]
        if missing:
            st.error(f"Missing required columns: {missing[:10]}")
            return

        X = batch[required].copy()

        if st.button("Run Batch Prediction", type="primary"):
            try:
                preds, proba = _predict_with_model(model, X)
                out = batch.copy()
                out["prediction"] = preds
                if proba is not None and proba.ndim == 2:
                    for i in range(proba.shape[1]):
                        out[f"prob_class_{i}"] = proba[:, i]
                st.success(f"Generated {len(out)} predictions")
                st.dataframe(out.head(100), use_container_width=True)
                st.download_button("Download Predictions", out.to_csv(index=False), file_name=f"{cfg.key}_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")


def _list_images(charts_dir: str) -> List[str]:
    if not os.path.exists(charts_dir):
        return []
    imgs: List[str] = []
    for root, _, files in os.walk(charts_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                imgs.append(os.path.join(root, f))
    return sorted(imgs)


def _render_charts(charts_dir: str) -> None:
    st.subheader("Visual Analytics Gallery")
    images = _list_images(charts_dir)
    if not images:
        st.info("No chart images found in `charts/`.")
        return

    cols = st.columns(2)
    for i, img_path in enumerate(images):
        cols[i % 2].image(img_path, caption=os.path.basename(img_path), use_container_width=True)


def _render_notebooks(notebooks_dir: str) -> None:
    st.subheader("Notebook Traceability")
    if not os.path.exists(notebooks_dir):
        st.info("No `notebooks/` directory found.")
        return

    notebook_files = [f for f in _safe_listdir(notebooks_dir) if f.endswith(".ipynb")]
    if not notebook_files:
        st.info("No notebooks found.")
        return

    rows = []
    for nb in notebook_files:
        nb_path = os.path.join(notebooks_dir, nb)
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            cells = payload.get("cells", [])
            code_cells = sum(1 for c in cells if c.get("cell_type") == "code")
            executed = sum(1 for c in cells if c.get("cell_type") == "code" and c.get("execution_count") is not None)
            rows.append({
                "Notebook": nb,
                "Code Cells": code_cells,
                "Executed Cells": executed,
                "Execution Coverage": f"{(executed / code_cells * 100):.1f}%" if code_cells else "0%",
            })
        except Exception:
            rows.append({"Notebook": nb, "Code Cells": "-", "Executed Cells": "-", "Execution Coverage": "-"})

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_highlights(cfg: ProjectConfig) -> None:
    st.subheader("Recruiter-Impact Highlights")
    for i, point in enumerate(cfg.highlights, start=1):
        st.markdown(
            f"""
            <div class='app-card'>
              <b>{i}. {point}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )


def run_project_app(cfg: ProjectConfig, project_dir: str) -> None:
    st.set_page_config(page_title=cfg.title, page_icon=cfg.icon, layout="wide")
    _inject_theme()

    models_dir = os.path.join(project_dir, "models")
    charts_dir = os.path.join(project_dir, "charts")
    notebooks_dir = os.path.join(project_dir, "notebooks")

    model_registry = _load_model_registry(models_dir)
    models = {name: info["model"] for name, info in model_registry.items() if info["predictable"]}
    results_df = _load_results_table(models_dir)
    chart_count = len(_list_images(charts_dir))
    notebook_count = len([f for f in _safe_listdir(notebooks_dir) if f.endswith(".ipynb")])

    with st.sidebar:
        st.markdown(f"## {cfg.icon} {cfg.key}")
        st.caption(cfg.subtitle)
        page = st.radio(
            "Navigate",
            [
                "Executive Overview",
                "Prediction Center",
                "Model Benchmarks",
                "Model Registry",
                "Charts",
                "Notebooks",
                "Recruiter Highlights",
            ],
        )
        st.write("---")
        st.write(f"Predict-ready: **{len(models)}**")
        st.write(f"All artifacts: **{len(model_registry)}**")
        st.write(f"Charts: **{chart_count}**")
        st.write(f"Notebooks: **{notebook_count}**")

    st.title(f"{cfg.icon} {cfg.title}")
    st.caption(cfg.subtitle)

    if page == "Executive Overview":
        _render_overview(cfg, model_registry, results_df, chart_count, notebook_count)

    if page == "Prediction Center":
        _render_predict(project_dir, models_dir, model_registry, cfg)

    if page == "Model Benchmarks":
        _render_results(results_df)

    if page == "Model Registry":
        _render_model_registry(model_registry, results_df)

    if page == "Charts":
        _render_charts(charts_dir)

    if page == "Notebooks":
        _render_notebooks(notebooks_dir)

    if page == "Recruiter Highlights":
        _render_highlights(cfg)

    st.write("---")
    st.caption(f"{cfg.key} | Industry-Style Portfolio Dashboard")
