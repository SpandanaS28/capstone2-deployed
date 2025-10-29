# app.py
import requests
import streamlit as st

st.set_page_config(page_title="❤️ Heart Disease Prediction App", page_icon="❤️", layout="centered")

st.markdown("## ❤️ Heart Disease Prediction App")
st.caption("Enter patient details to check the possibility of heart disease.")

API = st.text_input("FastAPI URL", "http://127.0.0.1:8000", help="Keep your FastAPI running in another terminal.")
predict_url = f"{API.rstrip('/')}/predict"
health_url  = f"{API.rstrip('/')}/health"

with st.expander("Backend status", expanded=True):
    try:
        r = requests.get(health_url, timeout=5)
        h = r.json()
        if h.get("ok"):
            st.success(f"API ready • model_loaded={h.get('model_loaded')}")
        else:
            st.warning("API responded but not OK.")
    except Exception as e:
        st.error(f"Cannot reach API: {e}")

with st.form("heart_form"):
    # Left column
    c1, c2 = st.columns(2)

    with c1:
        age        = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)
        sex_lbl    = st.selectbox("Sex", ["Male", "Female"])
        sex        = 1 if sex_lbl == "Male" else 0
        cp         = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        trestbps   = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 120, step=1)
        chol       = st.number_input("Cholesterol (mg/dl)", 100, 600, 220, step=1)
        fbs        = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        restecg    = st.selectbox("Resting ECG (0–2)", [0, 1, 2])

    with c2:
        thalach    = st.number_input("Max Heart Rate Achieved", 60, 250, 150, step=1)
        exang      = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak    = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1, format="%.2f")
        slope      = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
        ca         = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
        thal       = st.selectbox("Thalassemia (0=normal, 1=fixed defect, 2=reversible defect)", [0, 1, 2])

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = dict(
        age=age, sex=sex, cp=int(cp), trestbps=trestbps, chol=chol,
        fbs=int(fbs), restecg=int(restecg), thalach=thalach, exang=int(exang),
        oldpeak=float(oldpeak), slope=int(slope), ca=int(ca), thal=int(thal)
    )
    try:
        r = requests.post(predict_url, json=payload, timeout=10)
        data = r.json()

        if "probability" not in data:
            st.error(data)
        else:
            prob = float(data["probability"])           # probability of disease
            label = int(data["label"])                  # 1 = disease detected
            used  = data.get("used", "unknown")

            st.subheader("Results")
            st.write("**Prediction Value:**", f"{int(label)}")

            if label == 1:
                st.error("❌ **Heart Disease Detected.**")
            else:
                st.success("✅ **No Heart Disease Detected.**")

            st.write("**Probability:**", f"{prob*100:.2f}%")

            # Risk banner (to mimic your screenshot’s colored box)
            if prob >= 0.75:
                st.error("⚠️ **Risk Level: High Risk – Consult a doctor immediately.**")
            elif prob >= 0.4:
                st.warning("⚠️ **Risk Level: Moderate – Needs attention.**")
            else:
                st.info("🟢 **Risk Level: Low Risk – Continue healthy habits.**")

            with st.expander("Details"):
                st.json({"request": payload, "response": data, "endpoint": predict_url})
            st.caption(f"Model used: {used}")
    except Exception as e:
        st.error(f"Request failed: {e}")
