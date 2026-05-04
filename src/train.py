import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from preprocessing import preprocess_data

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "online_shoppers_intention.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

MODEL_PATH = os.path.join(MODELS_DIR, "purchase_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
OUTPUT_DATA_PATH = os.path.join(PROCESSED_DIR, "data_with_predictions.csv")


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

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),

        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                class_weight="balanced"
            ))
        ])
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("model", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                eval_metric="logloss"
            ))
        ])

    best_model = None
    best_name = None
    best_score = 0

    print("\n=== Model Comparison ===")

    for name, pipeline in models.items():
        print(f"\n--- {name} ---")

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print("Accuracy:", accuracy)
        print("ROC-AUC:", roc_auc)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        if roc_auc > best_score:
            best_score = roc_auc
            best_model = pipeline
            best_name = name

    print(f"\nBest model: {best_name}")
    print(f"Best ROC-AUC: {best_score:.4f}")

    best_model.fit(X, y)

    df["purchase_probability"] = best_model.predict_proba(X)[:, 1]
    df["purchase_prediction"] = best_model.predict(X)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURES_PATH)

    df.to_csv(OUTPUT_DATA_PATH, index=False)

    print("\nPipeline model and feature columns saved successfully.")
    print("Model:", MODEL_PATH)
    print("Features:", FEATURES_PATH)
    print("Predicted data:", OUTPUT_DATA_PATH)


if __name__ == "__main__":
    train_model()