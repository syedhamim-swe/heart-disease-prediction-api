from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.schemas import HeartFeatures

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "heart_model.joblib"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Binary heart disease prediction using a scikit-learn classifier.",
    version="1.0.0",
)

model = None

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run `python train.py` after "
            "placing the Kaggle dataset CSV at data/heart.csv."
        )
    model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/info")
def info():
    return {
        "model_type": "scikit-learn Pipeline (StandardScaler + LogisticRegression)",
        "features": FEATURES,
        "target": "target (1 = disease present, 0 = absence)",
    }

@app.post("/predict")
def predict(payload: HeartFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    values = payload.model_dump()
    row = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    return {
        "heart_disease": bool(prediction),
        "prediction": prediction,
        "probability": round(probability, 4),
    }
