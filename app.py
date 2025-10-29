# app.py  — Cloud-friendly: uses API if reachable, else local model, else heuristic
import streamlit as st
import numpy as np
import requests
import pickle
from pathlib import Path

st.set_page_config(page_title="Heart Disease Prediction App", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to check the possibility of heart disease.")

# ---------- API URL (optional) ----------
api_url = st.text_input(
    "FastAPI URL (optional, use a PUBLIC URL when deployed on Streamlit Cloud)",
    value="",
    placeholder="e.g., https://your-fastapi.onrender.com",
    help="Leave blank to run locally in Streamlit (no backend needed)."
)

# ---------- Try to load local model/scaler if present ----------
MODEL_FILE = Path("model.pkl")
SCALER_FILE = Path("scaler.pkl")

local_model = None
local_scaler = None
try:
    if MODEL_FILE.exists():
        local_model = pickle.load(open(MODEL_FILE, "rb"))
    if SCALER_FILE.exists():
        local_scaler = pickle.load(open(SCALER_FILE, "rb"))
except Exception as e:
    st.warning(f"Could not load local model files: {e}")

# ---------- Inputs ----------
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 120, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 120)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 220)
with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", 50, 250, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)

slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0=normal, 1=fixed, 2=reversible)", [0, 1, 2])

def label_and_badge(pred: int, prob: float):
    lbl = "Heart Disease" if pred == 1 else "No Heart Disease"
    st.subheader(f"Prediction: {lbl}")
    st.metric("Probability", f"{prob*100:.2f}%")
    if pred == 1:
        st.error("🚨 Risk Level: High — Consult a doctor immediately.")
    else:
        st.success("💚 Risk Level: Low — Continue healthy habits.")

def predict_via_api(payload: dict):
    if not api_url.strip():
        return None
    try:
        # health check (optional)
        requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        r = requests.post(f"{api_url.rstrip('/')}/predict", json=payload, timeout=10)
        if r.status_code == 200:
            j = r.json()
            pred = int(j.get("prediction", 0))
            prob = float(j.get("probability", 0.5))
            return pred, prob, "remote-api"
        else:
            st.warning(f"API error: {r.text}")
            return None
    except Exception:
        return None

def predict_locally(payload: dict):
    x = np.array(list(payload.values()), dtype=float).reshape(1, -1)
    # try real local model
    if local_model is not None and local_scaler is not None:
        xs = local_scaler.transform(x)
        try:
            prob = float(local_model.predict_proba(xs)[0][1])
        except Exception:
            # some models don’t have predict_proba
            prob = float(np.clip(local_model.predict(xs)[0], 0, 1))
        pred = int(prob >= 0.5)
        return pred, prob, "local-model"
    # fallback heuristic
    sex_bin = 1 if payload["sex"] == 1 else 0
    risk = (
        0.02*(payload["age"]-45)
        + 0.04*(payload["cp"])
        + 0.015*((payload["trestbps"]-120)/10.0)
        + 0.012*((payload["chol"]-220)/50.0)
        + 0.03*(payload["fbs"])
        + 0.02*(payload["restecg"])
        - 0.02*((payload["thalach"]-150)/10.0)
        + 0.03*(payload["exang"])
        + 0.04*(payload["oldpeak"])
        + 0.02*(payload["slope"])
        + 0.03*(payload["ca"])
        + 0.03*(payload["thal"])
        + 0.02*sex_bin
    )
    prob = float(np.clip(0.5 + risk, 0.01, 0.99))
    pred = int(prob >= 0.5)
    return pred, prob, "heuristic"

if st.button("Predict"):
    payload = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    result = predict_via_api(payload)
    if result is None:
        # API missing/unreachable -> local or heuristic
        pred, prob, src = predict_locally(payload)
        st.info(f"Using: {src.replace('-', ' ')}")
        label_and_badge(pred, prob)
    else:
        pred, prob, src = result
        st.info(f"Using: {src.replace('-', ' ')}")
        label_and_badge(pred, prob)
