# Customer Churn Prediction

Predicts whether a telecom customer is likely to churn, using the IBM
Telco Customer Churn dataset (7,043 customers, 21 raw attributes).

## What this project does

- **EDA** (`notebooks/EDA.ipynb`) — data quality checks, class imbalance
  analysis, and the specific patterns (contract type, tenure, add-on
  services, spend behavior) that motivated the feature engineering below.
- **Feature engineering** (`src/data_preprocessing.py`) — three engineered
  features derived directly from EDA findings: `tenure_group` (risk bucket),
  `avg_monthly_spend` (lifetime spend rate), `num_addon_services`
  (engagement/loyalty signal).
- **Modeling** (`src/train_model.py`):
  - Stratified train/test split (preserves the real ~73/27 class ratio)
  - A Logistic Regression **baseline** to justify the choice of XGBoost
  - **Class imbalance handling** via `scale_pos_weight` (XGBoost) /
    `class_weight="balanced"` (baseline) — accuracy alone is misleading
    on an imbalanced target, so precision/recall/F1/ROC-AUC are all reported
  - **Hyperparameter tuning** via `RandomizedSearchCV` (5-fold CV, ROC-AUC scoring)
  - Saved evaluation artifacts: confusion matrix, ROC curve, SHAP feature
    importance summary, and a JSON metrics report (`reports/`)
- **Prediction** (`src/predict.py`) — loads a single saved pipeline
  (preprocessing + model bundled together) so inference always encodes
  inputs exactly the way training did.
- **App** (`app/app.py`) — Streamlit form covering the customer's full
  profile (not just 2 fields), showing churn probability and the key
  drivers behind it.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# 1. Train the model (also generates reports/ artifacts)
cd src
python train_model.py

# 2. Launch the app
cd ..
streamlit run app/app.py
```

## Project structure

```
Customer-Churn-Prediction/
├── app/app.py                  # Streamlit UI
├── data/raw/churn.csv          # Raw dataset
├── models/churn_model.pkl      # Saved pipeline (preprocessing + model)
├── notebooks/EDA.ipynb         # Exploratory data analysis
├── reports/                    # Confusion matrix, ROC curve, SHAP plot, metrics.json
├── src/
│   ├── data_preprocessing.py   # Feature engineering + preprocessing pipeline
│   ├── train_model.py          # Training, tuning, evaluation
│   └── predict.py              # Inference on a single customer record
└── requirements.txt
```

## Key results

See `reports/metrics_report.json` after training for exact numbers on your
run (they'll vary slightly by random seed/tuning iteration draw), including
a side-by-side comparison of the Logistic Regression baseline vs. the
tuned XGBoost model.

## Notable design decisions

- **Single pipeline artifact.** Preprocessing and the model are saved
  together via `sklearn.pipeline.Pipeline`, so the app can pass in raw,
  human-readable fields and get consistent predictions — no manual
  one-hot-encoding or column alignment at inference time.
- **Imbalance-aware evaluation.** With a ~73/27 class split, a model that
  always predicts "no churn" scores ~73% accuracy while being useless.
  Precision, recall, F1, and ROC-AUC are all reported so the tradeoff
  between catching churners (recall) and false alarms (precision) is visible.
