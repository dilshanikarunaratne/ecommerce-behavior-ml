import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from segmentation import assign_behavior_segment
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from preprocessing import preprocess_data


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "online_shoppers_intention.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

MODEL_PATH = os.path.join(MODELS_DIR, "purchase_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
OUTPUT_DATA_PATH = os.path.join(PROCESSED_DIR, "data_with_predictions.csv")


def assign_segment(probability):
    if probability >= 0.80:
        return "High Intent"
    elif probability >= 0.50:
        return "Likely Buyer"
    elif probability >= 0.20:
        return "Browsing"
    else:
        return "Low Intent"


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy)
    print("ROC-AUC:", roc_auc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return roc_auc


def train_model():
    df = pd.read_csv(DATA_PATH)
    df = preprocess_data(df)

    X = df.drop("Revenue", axis=1)
    y = df["Revenue"]

    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {}

    # 1. Logistic Regression baseline
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    lr_pipeline.fit(X_train, y_train)
    models["Logistic Regression"] = lr_pipeline

    # 2. Random Forest baseline
    rf_pipeline = Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    rf_pipeline.fit(X_train, y_train)
    models["Random Forest"] = rf_pipeline

    # 3. Tuned Random Forest using GridSearchCV
    print("\nRunning GridSearchCV for Random Forest...")

    rf_grid_pipeline = Pipeline([
        ("model", RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ))
    ])

    rf_params = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [5, 10, 15, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    }

    rf_grid = GridSearchCV(
        estimator=rf_grid_pipeline,
        param_grid=rf_params,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)

    print("\nBest Random Forest Parameters:")
    print(rf_grid.best_params_)

    print("\nBest Cross-Validation ROC-AUC:")
    print(rf_grid.best_score_)

    models["Tuned Random Forest"] = rf_grid.best_estimator_

    # Compare all models
    print("\n=== Final Model Comparison ===")

    best_model = None
    best_name = None
    best_score = 0

    for name, model in models.items():
        roc_auc = evaluate_model(name, model, X_test, y_test)

        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model
            best_name = name

    print(f"\nBest Model: {best_name}")
    print(f"Best Test ROC-AUC: {best_score:.4f}")

    # Train best model on full dataset
    best_model.fit(X, y)

    df["purchase_prediction"] = best_model.predict(X)
    df["purchase_probability"] = best_model.predict_proba(X)[:, 1]

    df["behavior_segment"] = df.apply(assign_behavior_segment, axis=1)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURES_PATH)

    df.to_csv(OUTPUT_DATA_PATH, index=False)

    print("\nModel and feature columns saved successfully.")
    print("Model:", MODEL_PATH)
    print("Features:", FEATURES_PATH)
    print("Predicted dataset:", OUTPUT_DATA_PATH)


if __name__ == "__main__":
    train_model()