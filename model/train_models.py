"""Train six required classifiers on the UCI Adult Income dataset.

Artifacts written by this script are consumed by the Streamlit app.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET_COLUMN = "class"
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ROOT_DIR / "model" / "saved_models"


def _one_hot_encoder() -> OneHotEncoder:
    """Create a compatible OneHotEncoder across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_adult_income() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and lightly clean the Adult Income dataset."""
    dataset = fetch_openml(name="adult", version=2, as_frame=True)
    frame = dataset.frame.copy()
    frame = frame.replace("?", np.nan)

    X = frame.drop(columns=[TARGET_COLUMN])
    y_raw = frame[TARGET_COLUMN].astype(str).str.strip().str.rstrip(".")
    y = y_raw.map({"<=50K": 0, ">50K": 1})
    if y.isna().any():
        raise ValueError("Unexpected target labels found in Adult dataset.")

    return X, y.astype(int)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for numeric and categorical features."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_cols),
            ("numeric", numeric_transformer, numeric_cols),
        ],
        remainder="drop",
    )


def build_model_catalog() -> Dict[str, object]:
    """Return all required assignment models."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=7),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=16,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=140,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=4,
            eval_metric="logloss",
        ),
    }


def probability_score(model: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
    """Extract the positive-class score for AUC."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    raise ValueError("Model does not expose probability-like scores for AUC.")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_adult_income()
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    metrics_rows = []
    reports = {}

    for model_name, estimator in build_model_catalog().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_score = probability_score(pipeline, X_test)

        row = {
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "auc": float(roc_auc_score(y_test, y_score)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
        }
        metrics_rows.append(row)

        reports[model_name] = {
            "classification_report": classification_report(
                y_test, y_pred, target_names=["<=50K", ">50K"], output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        joblib.dump(pipeline, MODELS_DIR / f"{model_name}.joblib", compress=3)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="f1", ascending=False)
    metrics_df.to_csv(ARTIFACTS_DIR / "metrics.csv", index=False)

    with (ARTIFACTS_DIR / "reports.json").open("w", encoding="utf-8") as report_file:
        json.dump(reports, report_file, indent=2)

    holdout = X_test.copy()
    holdout[TARGET_COLUMN] = y_test.values
    holdout.to_csv(DATA_DIR / "holdout_test_with_labels.csv", index=False)
    holdout.sample(n=min(300, len(holdout)), random_state=RANDOM_STATE).to_csv(
        DATA_DIR / "sample_upload_test_data.csv", index=False
    )

    schema = {"feature_columns": X.columns.tolist(), "target_column": TARGET_COLUMN}
    with (ARTIFACTS_DIR / "schema.json").open("w", encoding="utf-8") as schema_file:
        json.dump(schema, schema_file, indent=2)

    print("Training complete. Artifacts saved in:", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
