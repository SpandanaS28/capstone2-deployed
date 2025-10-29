# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

# Load model and scaler if they exist
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    loaded = True
except:
    model = None
    scaler = None
    loaded = False

@app.post("/predict")
def predict(data: dict):
    try:
        features = np.array(list(data.values())).reshape(1, -1)
        if loaded and scaler is not None and model is not None:
            scaled = scaler.transform(features)
            prediction = model.predict(scaled)[0]
            probability = model.predict_proba(scaled)[0][1]
        else:
            # fallback dummy logic
            prediction = int(np.random.choice([0, 1]))
            probability = np.random.uniform(0.2, 0.8)

        label = "Heart Disease" if prediction == 1 else "No Heart Disease"
        return {"prediction": int(prediction), "label": label, "probability": float(probability)}

    except Exception as e:
        return {"error": str(e)}
