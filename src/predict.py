import os
import joblib
from preprocessing import preprocess_data
import pandas as pd
from segmentation import assign_behavior_segment

# get root project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# path to the saved trained model
MODEL_PATH = os.path.join(BASE_DIR, "models", "purchase_model.pkl")
# path to saved feature column names
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

# ------------------------
# main prediction function 
# ------------------------
def make_predictions(new_data):
    model = joblib.load(MODEL_PATH) # load saved trained model
    feature_columns = joblib.load(FEATURES_PATH) # load saved feature column names

    processed_data = preprocess_data(new_data.copy()) # runs same preprocessing pipeline. copy() is to prevent modifying original data

    if "Revenue" in processed_data.columns:
        processed_data = processed_data.drop("Revenue", axis=1)

    processed_data = processed_data.reindex(columns=feature_columns, fill_value=0) # to ensure exact same columns, exact same order

    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)[:, 1]

    results = new_data.copy() # predictions are saved here
    results["purchase_prediction"] = predictions
    results["purchase_probability"] = probabilities
    results["behavior_segment"] = results.apply(assign_behavior_segment, axis=1)

    return results