# main.py
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Heart Disease API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev; restrict if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Canonical UCI-style feature order (numeric-encoded)
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

# Optional trained artifacts (if you have them)
MODEL_FILE   = Path("model_heart.pkl")
SCALER_FILE  = Path("scaler_heart.pkl")
FEATURES_FILE= Path("features_heart.json")

_model = None
_scaler = None
_model_feats: List[str] = []

def _try_load():
    global _model, _scaler, _model_feats
    try:
        if MODEL_FILE.exists():
            _model = pickle.load(open(MODEL_FILE, "rb"))
        if SCALER_FILE.exists():
            _scaler = pickle.load(open(SCALER_FILE, "rb"))
        if FEATURES_FILE.exists():
            _model_feats = json.load(open(FEATURES_FILE))
    except Exception as e:
        print(f"⚠️ Could not load artifacts: {e}")

_try_load()

class PredictBody(BaseModel):
    # Named fields (expected from Streamlit)
    age: float
    sex: int = Field(description="0=female, 1=male")
    cp: int = Field(description="Chest pain type (0–3)")
    trestbps: float
    chol: float
    fbs: int = Field(description="Fasting blood sugar > 120 mg/dl (0/1)")
    restecg: int = Field(description="Resting ECG (0–2)")
    thalach: float
    exang: int = Field(description="Exercise induced angina (0/1)")
    oldpeak: float
    slope: int = Field(description="Slope of ST segment (0–2)")
    ca: int = Field(description="Number of major vessels (0–3)")
    thal: int = Field(description="Thalassemia (0=normal,1=fixed,2=reversible)")

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": _model is not None and _scaler is not None and bool(_model_feats),
        "expected_features": FEATURES,
        "model_features": _model_feats or None,
    }

def _fallback_proba(row: pd.Series) -> float:
    """
    Lightweight heuristic to mimic a classifier probability.
    Returns probability of heart disease (0..1).
    """
    score = 0.0
    score += 0.02 * (row["age"] - 50)
    score += 0.015 * (row["trestbps"] - 130)
    score += 0.01 * (row["chol"] - 200)
    score -= 0.02 * (row["thalach"] - 150)
    score += 0.5  * row["exang"]
    score += 0.25 * row["oldpeak"]
    score += 0.2  * (row["cp"] in [0, 1])  # typical/atypical angina risk-ish
    score += 0.15 * row["fbs"]
    score += 0.1  * (row["restecg"] == 2)
    score += 0.2  * row["sex"]            # (dataset is historically male-heavy)
    score += 0.25 * (row["slope"] == 0)
    score += 0.2  * (row["ca"] >= 1)
    score += 0.2  * (row["thal"] in [1, 2])

    # squish to (0..1) via logistic
    proba = 1 / (1 + np.exp(-score/3.0))
    return float(max(0.0, min(1.0, proba)))

@app.post("/predict")
def predict(body: PredictBody) -> Dict[str, Any]:
    data = {k: getattr(body, k) for k in FEATURES}
    df = pd.DataFrame([data])

    # Use trained model if available
    if _model is not None and _scaler is not None and _model_feats:
        try:
            X = df[_model_feats].values
            Xs = _scaler.transform(X)
            if hasattr(_model, "predict_proba"):
                proba = float(_model.predict_proba(Xs)[0, 1])
            else:
                # fallback if regression model
                proba = float(np.clip(_model.predict(Xs)[0], 0, 1))
            label = int(proba >= 0.5)
            return {"probability": proba, "label": label, "used": "trained-model"}
        except Exception as e:
            print(f"⚠️ Model failed, using heuristic: {e}")

    # Heuristic
    p = _fallback_proba(df.iloc[0])
    return {"probability": p, "label": int(p >= 0.5), "used": "heuristic-fallback"}
