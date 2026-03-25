import streamlit as st
import joblib
import numpy as np

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Mortar Strength Predictor",
    page_icon="🏗️",
    layout="wide"
)

# ==============================
# LOAD MODEL
# ==============================

model_data = joblib.load("my_KNN_model.pkl")

best_knn = model_data["model"]
x_scaler = model_data["x_scaler"]

# ==============================
# CUSTOM STYLE
# ==============================

st.markdown("""
<style>

.big-title {
    font-size:40px;
    font-weight:bold;
    color:#1f4e79;
}

.section-title {
    font-size:22px;
    font-weight:bold;
    color:#2c7fb8;
}

.result-box {
    background-color:#e6f2ff;
    padding:20px;
    border-radius:12px;
    font-size:28px;
    font-weight:bold;
    text-align:center;
    color:#003366;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================

st.markdown(
    '<p class="big-title">🏗️ Ternary Blended Mortar Strength Predictor</p>',
    unsafe_allow_html=True
)

st.markdown("""
This application predicts **28-day compressive strength (MPa)**  
using a machine-learning model trained on **1,095 laboratory samples**.
""")

# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("📊 Model Information")

st.sidebar.markdown("""
**Model:** K-Nearest Neighbors (KNN)

**Samples:** 1,095

**Output Range:**  
2.95 – 85.04 MPa

**Scaling Method:**  
MinMaxScaler
""")

# ==============================
# INPUT LAYOUT
# ==============================

col1, col2, col3 = st.columns(3)

# ------------------------------
# PRIMARY MATERIALS
# ------------------------------

with col1:

    st.markdown(
        '<p class="section-title">🧱 Primary Materials</p>',
        unsafe_allow_html=True
    )

    sand = st.slider(
        "Sand (460.58 – 1367.04 kg/m³)",
        460.58, 1367.04, 975.0
    )

    cement = st.slider(
        "Cement (566.24 – 1381.73 kg/m³)",
        566.24, 1381.73, 800.0
    )

    wc = st.slider(
        "Water/Cement Ratio (0.28 – 0.37)",
        0.28, 0.37, 0.33
    )

# ------------------------------
# SCMs (Correct Section)
# ------------------------------

with col2:

    st.markdown(
        '<p class="section-title">⚗️ Supplementary Cementitious Materials (SCMs)</p>',
        unsafe_allow_html=True
    )

    fly_ash = st.slider(
        "Fly Ash (0 – 242.67 kg/m³)",
        0.0, 242.67, 100.0
    )

    silica_fume = st.slider(
        "Silica Fume (0 – 242.67 kg/m³)",
        0.0, 242.67, 80.0
    )

# ------------------------------
# CHEMICAL ADMIXTURES (Fixed)
# ------------------------------

with col3:

    st.markdown(
        '<p class="section-title">🧪 Chemical Admixtures</p>',
        unsafe_allow_html=True
    )

    sp = st.slider(
        "Superplasticizer (0 – 8.09 kg/m³)",
        0.0, 8.09, 2.0
    )

    retarder = st.slider(
        "Retarder (0 – 4.04 kg/m³)",
        0.0, 4.04, 1.0
    )

    accelerator = st.slider(
        "Accelerator (0 – 4.04 kg/m³)",
        0.0, 4.04, 1.0
    )

# ==============================
# PREDICTION BUTTON
# ==============================

st.markdown("")

if st.button("🚀 Predict Compressive Strength"):

    features = np.array([[
        sand,
        cement,
        fly_ash,
        silica_fume,
        sp,
        retarder,
        accelerator,
        wc
    ]])

    X_scaled = x_scaler.transform(features)

    prediction = best_knn.predict(X_scaled)[0]

    st.markdown(
        f'<div class="result-box">Predicted Strength: {prediction:.2f} MPa</div>',
        unsafe_allow_html=True
    )

# ==============================
# DATASET SUMMARY TABLE
# ==============================

st.markdown("---")

st.markdown("### 📊 Dataset Statistical Summary")

stats_data = {
    "Parameter": [
        "Sand",
        "Cement",
        "Fly Ash",
        "Silica Fume",
        "Superplasticizer",
        "Retarder",
        "Accelerator",
        "W/C"
    ],

    "Range": [
        "460.58–1367.04",
        "566.24–1381.73",
        "0–242.67",
        "0–242.67",
        "0–8.09",
        "0–4.04",
        "0–4.04",
        "0.28–0.37"
    ],

    "Mean": [
        1078.98,
        783.92,
        72.77,
        55.85,
        1.80,
        0.12,
        0.61,
        0.33
    ]
}

st.dataframe(stats_data)
