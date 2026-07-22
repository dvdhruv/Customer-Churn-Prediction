"""
Trains a churn prediction model with:
- A stratified train/test split (preserves the real ~73/27 class ratio)
- Class-imbalance handling (scale_pos_weight for XGBoost)
- A Logistic Regression baseline to compare against XGBoost
- Hyperparameter tuning via RandomizedSearchCV (5-fold CV, scored on ROC-AUC)
- Saved evaluation artifacts: metrics report, confusion matrix, ROC curve,
  and a SHAP feature-importance summary plot

Run with:  python src/train_model.py
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay, classification_report,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data_preprocessing import prepare_dataset, build_preprocessor

REPORTS_DIR = "reports"
MODELS_DIR = "models"


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k:>10}: {v}")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

    return metrics, y_pred, y_proba


def save_confusion_matrix(y_test, y_pred, path):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Stayed", "Churned"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Stayed", "Churned"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - XGBoost (tuned)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_roc_curve(model, X_test, y_test, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve - XGBoost (tuned)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_shap_summary(pipeline, X_test, path):
    """Explains which features drive churn predictions the most - the
    single most useful artifact for the business framing of this project."""

    import shap

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_transformed)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_test_transformed, feature_names=feature_names,
        show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = prepare_dataset("data/raw/churn.csv")

    # Class imbalance ratio, fed to XGBoost's scale_pos_weight
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"Class balance in training set -> stayed: {neg}, churned: {pos} "
          f"(scale_pos_weight={scale_pos_weight:.2f})")

    # ------------------------------------------------------------------
    # Baseline: Logistic Regression
    # ------------------------------------------------------------------
    baseline_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )),
    ])
    baseline_pipeline.fit(X_train, y_train)
    baseline_metrics, _, _ = evaluate(
        "Baseline: Logistic Regression", baseline_pipeline, X_test, y_test
    )

    # ------------------------------------------------------------------
    # Main model: XGBoost, hyperparameter-tuned with RandomizedSearchCV
    # ------------------------------------------------------------------
    xgb_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        )),
    ])

    param_distributions = {
        "classifier__n_estimators": [100, 200, 300, 400],
        "classifier__max_depth": [3, 4, 5, 6, 8],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="roc_auc",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\nBest CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    best_pipeline = search.best_estimator_
    xgb_metrics, y_pred, y_proba = evaluate(
        "Tuned XGBoost", best_pipeline, X_test, y_test
    )

    # ------------------------------------------------------------------
    # Save evaluation artifacts
    # ------------------------------------------------------------------
    save_confusion_matrix(y_test, y_pred, f"{REPORTS_DIR}/confusion_matrix.png")
    save_roc_curve(best_pipeline, X_test, y_test, f"{REPORTS_DIR}/roc_curve.png")

    try:
        save_shap_summary(best_pipeline, X_test, f"{REPORTS_DIR}/shap_summary.png")
        shap_saved = True
    except ImportError:
        print("\n[warning] `shap` is not installed - skipping SHAP summary plot. "
              "Install with `pip install shap` to enable it.")
        shap_saved = False

    report = {
        "baseline_logistic_regression": baseline_metrics,
        "tuned_xgboost": xgb_metrics,
        "best_cv_roc_auc": round(search.best_score_, 4),
        "best_params": search.best_params_,
        "class_balance_train": {"stayed": int(neg), "churned": int(pos)},
        "shap_summary_saved": shap_saved,
    }

    with open(f"{REPORTS_DIR}/metrics_report.json", "w") as f:
        json.dump(report, f, indent=2)

    joblib.dump(best_pipeline, f"{MODELS_DIR}/churn_model.pkl")

    print(f"\nSaved model -> {MODELS_DIR}/churn_model.pkl")
    print(f"Saved reports -> {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
