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
# INDUSTRY STYLE (Times New Roman)
# ==============================

st.markdown("""
<style>

/* Apply Times New Roman everywhere */
html, body, [class*="css"]  {
    font-family: 'Times New Roman', serif;
}

/* Main Title */
.main-title {
    font-size:42px;
    font-weight:bold;
    color:#1f4e79;
    text-align:center;
}

/* Section Boxes */
.input-box {
    background-color:#f7f9fc;
    padding:20px;
    border-radius:12px;
    border:2px solid #d0d7e2;
}

/* Section Title */
.section-title {
    font-size:22px;
    font-weight:bold;
    color:#003366;
}

/* Result Styling */
.result-box {
    background-color:#e6f2ff;
    padding:25px;
    border-radius:12px;
    border:2px solid #2c7fb8;
    font-size:30px;
    font-weight:bold;
    text-align:center;
    color:#003366;
}

/* Subtitle */
.subtitle {
    text-align:center;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================

st.markdown(
'<p class="main-title">🏗️ Ternary Blended Mortar Strength Predictor</p>',
unsafe_allow_html=True
)

st.markdown(
'<p class="subtitle">Prediction of 28-Day Compressive Strength using Machine Learning Model trained on 1,095 laboratory samples</p>',
unsafe_allow_html=True
)

st.markdown("---")

# ==============================
# INPUT COLUMNS
# ==============================

col1, col2, col3 = st.columns(3)

# ==============================
# PRIMARY MATERIALS
# ==============================

with col1:

    st.markdown('<div class="input-box">', unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# SCM MATERIALS
# ==============================

with col2:

    st.markdown('<div class="input-box">', unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# CHEMICAL ADMIXTURES
# ==============================

with col3:

    st.markdown('<div class="input-box">', unsafe_allow_html=True)

    st.markdown(
    '<p class="section-title">🧪 Chemical Admixtures</p>',
    unsafe_allow_html=True
    )

    # MOVED HERE (correct placement)
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

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# PREDICTION
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

st.markdown("---")

# ==============================
# FOOTER
# ==============================

st.markdown("""
**Model:** K-Nearest Neighbors (KNN)  
**Dataset Size:** 1,095 Samples  
**Prediction Range:** 2.95 – 85.04 MPa  
""")
