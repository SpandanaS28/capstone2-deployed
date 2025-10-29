# app.py
import streamlit as st
import requests

st.set_page_config(page_title="Heart Disease Prediction App", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to check the possibility of heart disease.")

api_url = st.text_input("FastAPI URL", "http://127.0.0.1:8000")

# Input fields
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

if st.button("Predict"):
    try:
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
        response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result.get('label')}")
            st.metric("Probability", f"{result.get('probability') * 100:.2f}%")

            if result.get("prediction") == 1:
                st.error("🚨 Risk Level: High — Consult a doctor immediately.")
            else:
                st.info("💚 Risk Level: Low — Maintain healthy habits.")
        else:
            st.error(f"Server error: {response.text}")
    except Exception as e:
        st.error(f"Error contacting API: {e}")
