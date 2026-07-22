"""
Loads the saved pipeline (preprocessing + model bundled together) and
predicts churn from a single raw customer record.

Because the pipeline was saved as one artifact (see train_model.py), this
file no longer needs to manually recreate one-hot-encoded columns or guess
which fields were left out - that was the source of a real bug in an
earlier version of this project (the deployed app silently zero-filled
~28 of ~30 model features). Now the raw fields go in, the pipeline handles
encoding exactly as it did during training, and predictions are consistent.
"""

import joblib
import pandas as pd

from data_preprocessing import engineer_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES

_pipeline = None


def get_pipeline(model_path="models/churn_model.pkl"):
    global _pipeline

    if _pipeline is None:
        _pipeline = joblib.load(model_path)

    return _pipeline


def predict_churn(raw_customer: dict, model_path="models/churn_model.pkl"):
    """raw_customer: dict of the RAW fields (e.g. 'Contract': 'Month-to-month',
    'tenure': 5, 'MonthlyCharges': 70.35, ...) - the same fields collected
    directly from the Streamlit form, no manual encoding needed."""

    pipeline = get_pipeline(model_path)

    df = pd.DataFrame([raw_customer])
    df = engineer_features(df)
    df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]

    return {
        "prediction": "Customer will churn" if prediction == 1 else "Customer will stay",
        "will_churn": bool(prediction == 1),
        "churn_probability": round(float(probability), 4),
    }
