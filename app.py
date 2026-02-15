from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ROOT_DIR / "model" / "saved_models"
DATA_DIR = ROOT_DIR / "data"

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Naive Bayes (Gaussian)",
    "random_forest": "Random Forest (Ensemble)",
    "xgboost": "XGBoost (Ensemble)",
}


@st.cache_data
def load_metadata():
    metrics = pd.read_csv(ARTIFACTS_DIR / "metrics.csv")
    with (ARTIFACTS_DIR / "reports.json").open("r", encoding="utf-8") as f:
        reports = json.load(f)
    with (ARTIFACTS_DIR / "schema.json").open("r", encoding="utf-8") as f:
        schema = json.load(f)
    return metrics, reports, schema


@st.cache_data
def load_holdout_data():
    holdout_path = DATA_DIR / "holdout_test_with_labels.csv"
    if holdout_path.exists():
        return pd.read_csv(holdout_path)
    return pd.DataFrame()


@st.cache_resource
def load_model(model_name: str):
    return joblib.load(MODELS_DIR / f"{model_name}.joblib")


def score_for_auc(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def display_metric_cards(metric_values: dict):
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col1.metric("Accuracy", f"{metric_values['accuracy']:.4f}")
    col2.metric("AUC", f"{metric_values['auc']:.4f}")
    col3.metric("Precision", f"{metric_values['precision']:.4f}")
    col4.metric("Recall", f"{metric_values['recall']:.4f}")
    col5.metric("F1", f"{metric_values['f1']:.4f}")
    col6.metric("MCC", f"{metric_values['mcc']:.4f}")


def render_dataset_visuals(data: pd.DataFrame, target_column: str, feature_columns: list[str]) -> None:
    st.markdown("### Dataset Description and Feature Insights")

    total_rows, total_cols = data.shape
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in data.columns if col not in numeric_cols]
    total_missing = int(data.isna().sum().sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", total_rows)
    col2.metric("Columns", total_cols)
    col3.metric("Numeric Features", len([c for c in numeric_cols if c in feature_columns]))
    col4.metric("Categorical Features", len([c for c in categorical_cols if c in feature_columns]))
    st.caption(f"Total missing values in selected dataset: {total_missing}")

    st.markdown("#### Feature Types")
    dtype_df = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "dtype": [str(data[col].dtype) if col in data.columns else "missing" for col in feature_columns],
            }
        )
        .sort_values(by="dtype")
        .reset_index(drop=True)
    )
    st.dataframe(dtype_df, width="stretch")

    st.markdown("#### Missing Values by Feature")
    missing_by_col = data[feature_columns].isna().sum().sort_values(ascending=False)
    st.bar_chart(missing_by_col.head(15))

    if target_column in data.columns:
        st.markdown("#### Target Class Distribution")
        class_dist = data[target_column].value_counts(dropna=False).sort_index()
        st.bar_chart(class_dist)
    else:
        st.info(f"`{target_column}` is not present in this dataset, so class distribution is skipped.")

    st.markdown("#### Feature Distribution Explorer")
    selected_feature = st.selectbox("Choose a feature", options=feature_columns)
    if selected_feature in numeric_cols:
        values = data[selected_feature].dropna().astype(float)
        if values.empty:
            st.info("No numeric values available for this feature.")
        else:
            bins = st.slider(
                "Number of histogram bins",
                min_value=5,
                max_value=40,
                value=12,
                step=1,
                key=f"bins_{selected_feature}",
            )
            counts, edges = np.histogram(values, bins=bins)
            labels = []
            for idx in range(len(edges) - 1):
                left = int(round(edges[idx]))
                right = int(round(edges[idx + 1]))
                labels.append(f"{left}-{right}")

            hist_df = pd.DataFrame({"range": labels, "count": counts})
            st.bar_chart(hist_df.set_index("range")["count"])
            st.caption(
                f"{selected_feature} summary -> min: {values.min():.0f}, "
                f"median: {values.median():.0f}, max: {values.max():.0f}, "
                f"mean: {values.mean():.1f}"
            )
            if selected_feature == "age":
                st.info(
                    "Age chart meaning: each bar is an age range, and the bar height is "
                    "how many records fall in that range."
                )
    else:
        top_cat = data[selected_feature].astype(str).value_counts().head(20)
        st.bar_chart(top_cat)

    if target_column in data.columns:
        st.markdown("#### Feature vs Target Snapshot")
        categorical_feature_options = [col for col in feature_columns if col not in numeric_cols]
        if categorical_feature_options:
            feature_for_target_view = st.selectbox(
                "Choose a categorical feature to compare with target",
                options=categorical_feature_options,
            )
            relation = pd.crosstab(
                data[feature_for_target_view].astype(str),
                data[target_column],
                normalize="index",
            )
            st.dataframe(relation.head(25), width="stretch")
        else:
            st.info("No categorical features available for target comparison.")


def main():
    st.set_page_config(page_title="ML Assignment 2 - Classifier Dashboard", layout="wide")
    st.markdown(
        """
        <style>
        div[data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.15rem;
            font-weight: 700;
        }
        .student-card {
            background-color: #111827;
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 14px 16px;
            margin: 4px 0 12px 0;
        }
        .student-card-title {
            font-size: 0.9rem;
            color: #d1d5db;
            margin: 0;
        }
        .student-card-value {
            font-size: 1.15rem;
            color: #ffffff;
            font-weight: 700;
            margin: 2px 0 0 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("ML Assignment 2: Adult Income Classification")
    st.caption("Upload test CSV data, choose a model, and inspect evaluation results.")
    st.markdown(
        """
        <div class="student-card">
            <p class="student-card-title">Student Details</p>
            <p class="student-card-value">R Angatha Ram Kisan | BITS ID: 2025aa05033</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not ARTIFACTS_DIR.exists():
        st.error("Artifacts not found. Run `python model/train_models.py` first.")
        st.stop()

    metrics_df, reports, schema = load_metadata()
    holdout_data = load_holdout_data()
    target_column = schema["target_column"]
    feature_columns = schema["feature_columns"]
    uploaded_file = st.file_uploader("Upload test CSV (optional)", type=["csv"])

    uploaded_data = None
    if uploaded_file is not None:
        uploaded_data = pd.read_csv(uploaded_file)
        missing_cols = [col for col in feature_columns if col not in uploaded_data.columns]
        if missing_cols:
            st.error(
                "Uploaded data is missing required feature columns: "
                + ", ".join(missing_cols[:10])
            )
            st.stop()

    analysis_data = uploaded_data if uploaded_data is not None else holdout_data
    if analysis_data.empty:
        st.warning("No holdout dataset found for dataset visuals. Run training once to generate it.")
    else:
        source_name = "uploaded CSV" if uploaded_data is not None else "saved holdout test set"
        st.caption(f"Dataset visualization source: {source_name}")

    tab_dataset, tab_models, tab_eval = st.tabs(
        ["Dataset & Features", "Model Comparison", "Interactive Evaluation"]
    )

    with tab_dataset:
        if analysis_data.empty:
            st.info("Upload a CSV with required features to view dataset insights.")
        else:
            render_dataset_visuals(analysis_data, target_column, feature_columns)

    with tab_models:
        st.subheader("Model Comparison (Holdout Test Set)")
        comparison = metrics_df.copy()
        comparison["model_name"] = comparison["model_name"].map(MODEL_LABELS)
        st.dataframe(comparison, width="stretch")

    with tab_eval:
        st.subheader("Interactive Model Evaluation")
        model_name = st.selectbox(
            "Select a model",
            options=list(MODEL_LABELS.keys()),
            format_func=lambda name: MODEL_LABELS[name],
        )

        selected_model = load_model(model_name)
        holdout_row = metrics_df.loc[metrics_df["model_name"] == model_name].iloc[0].to_dict()
        display_metric_cards(holdout_row)

        st.markdown("#### Holdout Confusion Matrix")
        holdout_cm = reports[model_name]["confusion_matrix"]
        st.dataframe(
            pd.DataFrame(holdout_cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]),
            width="content",
        )

        st.markdown("#### Holdout Classification Report")
        holdout_report_df = pd.DataFrame(reports[model_name]["classification_report"]).transpose()
        st.dataframe(holdout_report_df, width="stretch")

        if uploaded_data is None:
            st.info(
                "No CSV uploaded. Metrics shown above are from the saved holdout test set. "
                "You can upload `data/sample_upload_test_data.csv` to test upload flow."
            )
            return

        data = uploaded_data.copy()

        X_upload = data[feature_columns].copy()
        y_pred = selected_model.predict(X_upload)
        data["predicted_income"] = y_pred

        st.markdown("#### Uploaded Data Predictions")
        st.dataframe(data.head(20), width="stretch")

        if target_column in data.columns:
            y_true = data[target_column].astype(int)
            y_score = score_for_auc(selected_model, X_upload)
            metrics_uploaded = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "auc": float(roc_auc_score(y_true, y_score)) if y_score is not None else 0.0,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
            }

            st.markdown("### Uploaded Dataset Evaluation")
            display_metric_cards(metrics_uploaded)

            upload_cm = confusion_matrix(y_true, y_pred)
            st.markdown("#### Uploaded Confusion Matrix")
            st.dataframe(
                pd.DataFrame(upload_cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]),
                width="content",
            )

            st.markdown("#### Uploaded Classification Report")
            report_df = pd.DataFrame(
                classification_report(y_true, y_pred, output_dict=True)
            ).transpose()
            st.dataframe(report_df, width="stretch")
        else:
            st.warning(
                f"Uploaded CSV does not include `{target_column}`. "
                "Predictions are shown, but uploaded-set metrics cannot be computed."
            )


if __name__ == "__main__":
    main()
