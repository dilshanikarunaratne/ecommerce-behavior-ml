import os
import joblib
from preprocessing import preprocess_data
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "purchase_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")


def assign_segment(probability):
    if probability >= 0.80:
        return "High Intent"
    elif probability >= 0.50:
        return "Likely Buyer"
    elif probability >= 0.20:
        return "Browsing"
    else:
        return "Low Intent"


def make_predictions(new_data):
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    processed_data = preprocess_data(new_data.copy())

    if "Revenue" in processed_data.columns:
        processed_data = processed_data.drop("Revenue", axis=1)

    processed_data = processed_data.reindex(columns=feature_columns, fill_value=0)

    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)[:, 1]

    results = new_data.copy()
    results["purchase_prediction"] = predictions
    results["purchase_probability"] = probabilities
    results["behavior_segment"] = results["purchase_probability"].apply(assign_segment)

    return results