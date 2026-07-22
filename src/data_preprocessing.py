"""
Data loading, feature engineering, and preprocessing pipeline for the
Telco Customer Churn dataset (IBM's public sample dataset - a standard,
well-recognized benchmark for churn prediction, so it's a safe, familiar
dataset for an interviewer to reason about).

Design choice: instead of manually one-hot-encoding with pd.get_dummies()
and then patching missing columns at inference time (which is fragile and
was the source of a real bug in an earlier version of this project - the
deployed app only asked for 2 fields and silently zero-filled ~28 others),
we build a proper sklearn ColumnTransformer + Pipeline. The ENTIRE pipeline
(preprocessing + model) is saved as a single artifact, so at inference time
we just pass in the raw, human-readable customer fields and the pipeline
handles encoding identically to how it did during training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Churn"
ID_COL = "customerID"

# Base numeric columns present in the raw dataset
BASE_NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

# Engineered numeric features (created in engineer_features)
ENGINEERED_NUMERIC_FEATURES = ["avg_monthly_spend", "num_addon_services"]

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES

# Base categorical columns
BASE_CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

# Engineered categorical feature (tenure bucket)
ENGINEERED_CATEGORICAL_FEATURES = ["tenure_group"]

CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES

ADDON_SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def load_data(path):
    return pd.read_csv(path)


def engineer_features(df):
    """Add hand-crafted features. Used identically during training and at
    inference (single-row prediction), so behavior never drifts between
    the two."""

    df = df.copy()

    # Clean TotalCharges (a handful of rows have blank strings in the raw data)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    # --- Feature 1: tenure bucket ---
    # Captures non-linear "new customer vs loyal customer" risk that raw
    # tenure alone doesn't express well to a linear-ish model.
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12", "13-24", "25-48", "49-60", "61-72"],
    ).astype(str)

    # --- Feature 2: average monthly spend across the customer's lifetime ---
    # Differs from MonthlyCharges (current rate) - flags customers whose
    # spend has drifted a lot from their historical average.
    safe_tenure = df["tenure"].replace(0, 1)
    df["avg_monthly_spend"] = df["TotalCharges"] / safe_tenure

    # --- Feature 3: number of add-on services subscribed ---
    # A simple loyalty/engagement signal - customers with more add-ons
    # tend to be more invested in the service.
    def count_addons(row):
        return sum(1 for col in ADDON_SERVICE_COLUMNS if row[col] == "Yes")

    df["num_addon_services"] = df.apply(count_addons, axis=1)

    return df


def clean_raw_data(df):
    """Drop ID column and encode the target. Leaves feature columns as raw
    strings/numbers - encoding happens inside the sklearn pipeline."""

    df = df.copy()

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"No": 0, "Yes": 1})

    return df


def build_preprocessor():
    """ColumnTransformer that one-hot-encodes categoricals (ignoring unseen
    categories at inference instead of crashing) and scales numerics."""

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def prepare_dataset(path):
    """Full pipeline from raw CSV to train/test split, ready for modeling."""

    df = load_data(path)
    df = clean_raw_data(df)
    df = df.dropna(subset=[TARGET_COL])
    df = engineer_features(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL].astype(int)

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # preserve the ~73/27 class ratio in both splits
    )
