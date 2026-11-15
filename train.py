from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/heart.csv")
MODEL_PATH = Path("model/heart_model.joblib")
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Download the Kaggle Heart Disease Dataset and "
            "place its CSV at data/heart.csv."
        )

    df = pd.read_csv(DATA_PATH)
    # The Kaggle dataset normally uses these exact feature/target column names.
    required = FEATURES + ["target"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df[required].dropna()
    X, y = df[FEATURES], df["target"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
